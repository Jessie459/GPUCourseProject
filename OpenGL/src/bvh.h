#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <climits>

#include "morton.h"
#include "triangle.h"

struct node
{
    unsigned int parentIdx;
    unsigned int lchildIdx;
    unsigned int rchildIdx;
    unsigned int objectIdx;
};

class BVH
{
public:
    BVH(const std::vector<float>& h_vertices,
        const std::vector<unsigned int>& h_indices);

    void constructBVH();
    void traverseBVH();
    std::vector<unsigned int> getFlags();

private:
    thrust::device_vector<float> d_vertices;
    thrust::device_vector<triangle> d_triangles;

    thrust::device_vector<aabb> d_aabbs;
    thrust::device_vector<node> d_nodes;

    thrust::device_vector<unsigned int> d_flags;
};

__device__
int findSplit(unsigned int* mortonCodes, int first, int last);

__device__
uint2 determineRange(unsigned int* mortonCodes, int numTriangles, int index);

__device__
void traverse(const aabb* aabbs, const node* nodes, const float* vertices, const triangle* triangles,
              const unsigned int numTriangles, const unsigned int index, unsigned int* flags);