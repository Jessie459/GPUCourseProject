#pragma once
#include "bvh.h"

BVH::BVH(const std::vector<float>& h_vertices,
         const std::vector<unsigned int>& h_indices)
{
    // set vertices
    d_vertices = h_vertices;

    // set triangles
    unsigned int numTriangles = h_indices.size() / 3;
    d_triangles.resize(numTriangles);
    for (unsigned int i = 0; i < numTriangles; i++)
    {
        triangle tri;
        tri.idx0 = h_indices[i * 3];
        tri.idx1 = h_indices[i * 3 + 1];
        tri.idx2 = h_indices[i * 3 + 2];
        d_triangles[i] = tri;
    }

    // set flags that indicate whether triangles collide
    d_flags.resize(numTriangles, 0);
}

void BVH::constructBVH()
{
    if (d_vertices.size() == 0 || d_triangles.size() == 0)
        return;

    unsigned int numTriangles = d_triangles.size();
    unsigned int numInternalNodes = numTriangles - 1;
    unsigned int numNodes = numTriangles * 2 - 1;

    // construct bounding box for each leaf node
    // -----------------------------------------
    constexpr float INF = std::numeric_limits<float>::infinity();
    aabb defaultBBox;
    defaultBBox.upper.x = -INF; defaultBBox.lower.x = INF;
    defaultBBox.upper.y = -INF; defaultBBox.lower.y = INF;
    defaultBBox.upper.z = -INF; defaultBBox.lower.z = INF;
    d_aabbs.resize(numNodes, defaultBBox);

    float* vertices = d_vertices.data().get();
    thrust::transform(
        d_triangles.begin(), d_triangles.end(),
        d_aabbs.begin() + numInternalNodes,
        [vertices] __device__ (const triangle& tri)
        {
            aabb bbox;
            getAABB(tri, vertices, bbox);
            return bbox;
        }
    );

    aabb topBBox = thrust::reduce(
        d_aabbs.begin() + numInternalNodes, d_aabbs.end(),
        defaultBBox,
        [] __device__ (const aabb& lhs, const aabb& rhs)
        {
            return merge(lhs, rhs);
        }
    );

    // calculate morton code for each leaf node
    // ----------------------------------------
    thrust::device_vector<unsigned int> d_codes(numTriangles);
    thrust::transform(
        d_aabbs.begin() + numInternalNodes, d_aabbs.end(),
        d_codes.begin(),
        [topBBox] __device__ (const aabb& box)
        {
            float3 point = centroid(box);
            point.x -= topBBox.lower.x;
            point.y -= topBBox.lower.y;
            point.z -= topBBox.lower.z;
            point.x /= (topBBox.upper.x - topBBox.lower.x);
            point.y /= (topBBox.upper.y - topBBox.lower.y);
            point.z /= (topBBox.upper.z - topBBox.lower.z);
            return mortonCode(point);
        }
    );

    // sort bounding boxes and counts of trianlges by morton codes
    // -----------------------------------------------------------
    thrust::device_vector<unsigned int> d_counts(numTriangles);
    thrust::copy(thrust::make_counting_iterator<unsigned int>(0),
                 thrust::make_counting_iterator<unsigned int>(numTriangles),
                 d_counts.begin());

    auto tuple = thrust::make_tuple(d_aabbs.begin() + numInternalNodes, d_counts.begin());
    thrust::stable_sort_by_key(d_codes.begin(), d_codes.end(), thrust::make_zip_iterator(tuple));

    // check morton codes are unique
    // -----------------------------
    /*thrust::device_vector<unsigned long long int> d_codes64(numTriangles);
    auto iter = thrust::unique_copy(
        d_codes.begin(),
        d_codes.end(),
        d_codes64.begin());
    bool isUnique = (d_codes64.end() == iter);
    if (!isUnique)
    {
        printf("warnning: morton codes are not unique\n");
        thrust::transform(
            d_codes.begin(),
            d_codes.end(),
            d_counts.begin(),
            d_codes64.begin(),
            [] __device__(unsigned int code, unsigned int index)
            {
                unsigned long long int code64 = code;
                code64 <<= 32;
                code64 |= index;
                return code64;
            });
    }*/

    // construct leaf nodes
    // --------------------
    node defaultNode;
    defaultNode.parentIdx = UINT_MAX;
    defaultNode.lchildIdx = UINT_MAX;
    defaultNode.rchildIdx = UINT_MAX;
    defaultNode.objectIdx = UINT_MAX;
    d_nodes.resize(numNodes, defaultNode);

    thrust::transform(
        d_counts.begin(), d_counts.end(),
        d_nodes.begin() + numInternalNodes,
        [] __device__ (unsigned int index)
        {
            node n;
            n.parentIdx = UINT_MAX;
            n.lchildIdx = UINT_MAX;
            n.rchildIdx = UINT_MAX;
            n.objectIdx = index;
            return n;
        }
    );

    // construct internal nodes
    // ------------------------
    node* nodes = this->d_nodes.data().get();
    aabb* aabbs = this->d_aabbs.data().get();
    unsigned int* codes = d_codes.data().get();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numInternalNodes),
        [nodes, codes, numTriangles] __device__ (unsigned int index)
        {
            nodes[index].objectIdx = UINT_MAX;

            uint2 range = determineRange(codes, numTriangles, index);
            int split = findSplit(codes, range.x, range.y);

            nodes[index].lchildIdx = split;
            nodes[index].rchildIdx = split + 1;
            if (thrust::min(range.x, range.y) == split)
                nodes[index].lchildIdx += numTriangles - 1;
            if (thrust::max(range.x, range.y) == split + 1)
                nodes[index].rchildIdx += numTriangles - 1;

            nodes[nodes[index].lchildIdx].parentIdx = index;
            nodes[nodes[index].rchildIdx].parentIdx = index;
        }
    );

    // construct bounding box for each internal node
    // ---------------------------------------------
    thrust::device_vector<int> d_nodeFlags(numInternalNodes, 0);
    int* nodeFlags = d_nodeFlags.data().get();
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(numInternalNodes),
        thrust::make_counting_iterator(numNodes),
        [nodes, aabbs, nodeFlags] __device__(unsigned int index)
        {
            unsigned int parentIdx = nodes[index].parentIdx;
            while (parentIdx != UINT_MAX)
            {
                int oldFlag = atomicCAS(nodeFlags + parentIdx, 0, 1);
                if (oldFlag == 0)   // the first thread arrived
                    return;         // wait for the other thread
                assert(oldFlag == 1);

                unsigned int lchildIdx = nodes[parentIdx].lchildIdx;
                unsigned int rchildIdx = nodes[parentIdx].rchildIdx;
                aabb lbbox = aabbs[lchildIdx];
                aabb rbbox = aabbs[rchildIdx];
                aabbs[parentIdx] = merge(lbbox, rbbox);

                parentIdx = nodes[parentIdx].parentIdx;
            }
        }
    );
}

void BVH::traverseBVH()
{
    unsigned int numTriangles = d_triangles.size();

    const aabb* aabbs = d_aabbs.data().get();
    const node* nodes = d_nodes.data().get();

    const float* vertices = d_vertices.data().get();
    const triangle* triangles = d_triangles.data().get();
    unsigned int* flags = d_flags.data().get();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numTriangles),
        [aabbs, nodes, vertices, triangles, numTriangles, flags]
        __device__ (unsigned int index)
        {
            traverse(aabbs, nodes, vertices, triangles, numTriangles, index, flags);
        }
    );
}

__device__
void traverse(const aabb* aabbs, const node* nodes, const float* vertices, const triangle* triangles,
              const unsigned int numTriangles, const unsigned int index, unsigned int* flags)
{
    aabb bbox;
    getAABB(triangles[index], vertices, bbox);

    constexpr unsigned int STACK_SIZE = 64;
    unsigned int stack[STACK_SIZE];
    unsigned int stackIdx = 0;
    stack[stackIdx] = 0;
    ++stackIdx;

    // traverse nodes starting from the root
    do
    {
        --stackIdx;
        unsigned int parentIdx = stack[stackIdx];
        unsigned int lchildIdx = nodes[parentIdx].lchildIdx;
        unsigned int rchildIdx = nodes[parentIdx].rchildIdx;

        if (intersect(bbox, aabbs[lchildIdx]))
        {
            unsigned int objectIdx = nodes[lchildIdx].objectIdx;
            if (objectIdx != UINT_MAX) // leaf node
            {
                if (index >= objectIdx)
                    continue;

                // remove adjacent triangles
                triangle tri0 = triangles[index];
                triangle tri1 = triangles[objectIdx];

                if (tri0.idx0 == tri1.idx0 || tri0.idx0 == tri1.idx1 || tri0.idx0 == tri1.idx2 ||
                    tri0.idx1 == tri1.idx0 || tri0.idx1 == tri1.idx1 || tri0.idx1 == tri1.idx2 ||
                    tri0.idx2 == tri1.idx0 || tri0.idx2 == tri1.idx1 || tri0.idx2 == tri1.idx2)
                    continue;

                // collision detection between triangles
                vec3f p0(vertices[tri0.idx0 * 3], vertices[tri0.idx0 * 3 + 1], vertices[tri0.idx0 * 3 + 2]);
                vec3f p1(vertices[tri0.idx1 * 3], vertices[tri0.idx1 * 3 + 1], vertices[tri0.idx1 * 3 + 2]);
                vec3f p2(vertices[tri0.idx2 * 3], vertices[tri0.idx2 * 3 + 1], vertices[tri0.idx2 * 3 + 2]);

                vec3f q0(vertices[tri1.idx0 * 3], vertices[tri1.idx0 * 3 + 1], vertices[tri1.idx0 * 3 + 2]);
                vec3f q1(vertices[tri1.idx1 * 3], vertices[tri1.idx1 * 3 + 1], vertices[tri1.idx1 * 3 + 2]);
                vec3f q2(vertices[tri1.idx2 * 3], vertices[tri1.idx2 * 3 + 1], vertices[tri1.idx2 * 3 + 2]);

                if (collide(p0, p1, p2, q0, q1, q2))
                {
                    atomicAdd(&flags[index], 1);
                    atomicAdd(&flags[objectIdx], 1);
                    // flags[index] = 1;
                    // flags[objectIdx] = 1;
                }
            }
            else // internal node
            {
                if (stackIdx < STACK_SIZE)
                {
                    stack[stackIdx] = lchildIdx;
                    ++stackIdx;
                }
                else
                    printf("WARNING: the traverse stack is full!\n");
            }
        }
        if (intersect(bbox, aabbs[rchildIdx]))
        {
            unsigned int objectIdx = nodes[rchildIdx].objectIdx;
            if (objectIdx != UINT_MAX) // leaf node
            {
                if (index >= objectIdx)
                    continue;

                // remove adjacent triangles
                triangle tri0 = triangles[index];
                triangle tri1 = triangles[objectIdx];

                if (tri0.idx0 == tri1.idx0 || tri0.idx0 == tri1.idx1 || tri0.idx0 == tri1.idx2 ||
                    tri0.idx1 == tri1.idx0 || tri0.idx1 == tri1.idx1 || tri0.idx1 == tri1.idx2 ||
                    tri0.idx2 == tri1.idx0 || tri0.idx2 == tri1.idx1 || tri0.idx2 == tri1.idx2)
                    continue;

                // collision detection between triangles
                vec3f p0(vertices[tri0.idx0 * 3], vertices[tri0.idx0 * 3 + 1], vertices[tri0.idx0 * 3 + 2]);
                vec3f p1(vertices[tri0.idx1 * 3], vertices[tri0.idx1 * 3 + 1], vertices[tri0.idx1 * 3 + 2]);
                vec3f p2(vertices[tri0.idx2 * 3], vertices[tri0.idx2 * 3 + 1], vertices[tri0.idx2 * 3 + 2]);

                vec3f q0(vertices[tri1.idx0 * 3], vertices[tri1.idx0 * 3 + 1], vertices[tri1.idx0 * 3 + 2]);
                vec3f q1(vertices[tri1.idx1 * 3], vertices[tri1.idx1 * 3 + 1], vertices[tri1.idx1 * 3 + 2]);
                vec3f q2(vertices[tri1.idx2 * 3], vertices[tri1.idx2 * 3 + 1], vertices[tri1.idx2 * 3 + 2]);

                if (collide(p0, p1, p2, q0, q1, q2))
                {
                    atomicAdd(&flags[index], 1);
                    atomicAdd(&flags[objectIdx], 1);
                    // flags[index] = 1;
                    // flags[objectIdx] = 1;
                }
            }
            else // internal node
            {
                if (stackIdx < STACK_SIZE)
                {
                    stack[stackIdx] = rchildIdx;
                    ++stackIdx;
                }
                else
                    printf("WARNING: the traverse stack is full!\n");
            }
        }
    } while (stackIdx > 0);
}


std::vector<unsigned int> BVH::getFlags()
{
    std::vector<unsigned int> flags(d_flags.size());
    cudaMemcpy(&flags[0], d_flags.data().get(), sizeof(unsigned int) * d_flags.size(), cudaMemcpyDeviceToHost);
    return flags;
}

__device__
int findSplit(unsigned int* mortonCodes, int first, int last)
{
    /**
        Identical Morton codes => split the range in the middle.
    **/
    unsigned int firstCode = mortonCodes[first];
    unsigned int lastCode = mortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = clz(firstCode, lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest triangle that
    // shares more than commonPrefix bits with the first one.
    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = mortonCodes[newSplit];
            int splitPrefix = clz(firstCode, splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

__device__
uint2 determineRange(unsigned int* mortonCodes, int numTriangles, int index)
{
    /**
        determine the range of keys covered by each internal node (as well as its children)
        direction is found by looking at the neighboring keys ki-1 , ki , ki+1
        the index is either the beginning of the range or the end of the range
    **/
    if (index == 0)
    {
        return make_uint2(0, numTriangles - 1);
    }

    // determine direction of the range
    unsigned int mortonCode = mortonCodes[index];
    int lCommonPrefix = clz(mortonCode, mortonCodes[index - 1]);
    int rCommonPrefix = clz(mortonCode, mortonCodes[index + 1]);
    int minCommonPrefix = thrust::min(lCommonPrefix, rCommonPrefix);
    int direction = (rCommonPrefix > lCommonPrefix) ? 1 : -1;

    // compute upper bound for the length of the range
    int step = 0;
    int s = 2;

    int i = index + direction * s;
    int commonPrefix = -1;
    if (i >= 0 && i < numTriangles)
        commonPrefix = clz(mortonCode, mortonCodes[i]);

    while (commonPrefix > minCommonPrefix)
    {
        s <<= 1;
        i = index + direction * s;
        commonPrefix = -1;
        if (i >= 0 && i < numTriangles)
            commonPrefix = clz(mortonCode, mortonCodes[i]);
    }
    s >>= 1;
    while (s > 0)
    {
        i = index + direction * (step + s);
        commonPrefix = -1;
        if (i >= 0 && i < numTriangles)
            commonPrefix = clz(mortonCode, mortonCodes[i]);
        if (commonPrefix > minCommonPrefix)
            step += s;
        s >>= 1;
    }

    unsigned int other = index + direction * step;
    if (direction < 0)
        thrust::swap(index, other);
    return make_uint2(index, other);
}