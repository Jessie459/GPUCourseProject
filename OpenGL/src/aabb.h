#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

struct aabb
{
    float3 upper;
    float3 lower;
};

__device__ __host__
inline bool intersect(const aabb& lhs, const aabb& rhs)
{
    if (lhs.upper.x <= rhs.lower.x || rhs.upper.x <= lhs.lower.x) { return false; }
    if (lhs.upper.y <= rhs.lower.y || rhs.upper.y <= lhs.lower.y) { return false; }
    if (lhs.upper.z <= rhs.lower.z || rhs.upper.z <= lhs.lower.z) { return false; }
    return true;
}

__device__ __host__
inline aabb merge(const aabb& lhs, const aabb& rhs)
{
    aabb merged;
    merged.upper.x = fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = fmax(lhs.upper.y, rhs.upper.y);
    merged.upper.z = fmax(lhs.upper.z, rhs.upper.z);
    merged.lower.x = fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = fmin(lhs.lower.y, rhs.lower.y);
    merged.lower.z = fmin(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__
inline float3 centroid(const aabb& box)
{
    float3 c;
    c.x = (box.upper.x + box.lower.x) * 0.5f;
    c.y = (box.upper.y + box.lower.y) * 0.5f;
    c.z = (box.upper.z + box.lower.z) * 0.5f;
    return c;
}