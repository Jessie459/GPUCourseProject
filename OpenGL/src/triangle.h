#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "aabb.h"
#include "vec3f.h"

struct triangle
{
    unsigned int idx0;
    unsigned int idx1;
    unsigned int idx2;
};

__device__ __host__
inline float fmax3(float a, float b, float c)
{
    return fmax(a, fmax(b, c));
}

__device__ __host__
inline float fmin3(float a, float b, float c)
{
    return fmin(a, fmin(b, c));
}

__device__ __host__
inline void getAABB(const triangle& tri, const float* vertices, aabb& bbox)
{
    float v0_x = vertices[tri.idx0 * 3];
    float v0_y = vertices[tri.idx0 * 3 + 1];
    float v0_z = vertices[tri.idx0 * 3 + 2];

    float v1_x = vertices[tri.idx1 * 3];
    float v1_y = vertices[tri.idx1 * 3 + 1];
    float v1_z = vertices[tri.idx1 * 3 + 2];

    float v2_x = vertices[tri.idx2 * 3];
    float v2_y = vertices[tri.idx2 * 3 + 1];
    float v2_z = vertices[tri.idx2 * 3 + 2];

    float minX = fmin3(v0_x, v1_x, v2_x);
    float minY = fmin3(v0_y, v1_y, v2_y);
    float minZ = fmin3(v0_z, v1_z, v2_z);

    float maxX = fmax3(v0_x, v1_x, v2_x);
    float maxY = fmax3(v0_y, v1_y, v2_y);
    float maxZ = fmax3(v0_z, v1_z, v2_z);

    bbox.lower.x = minX; bbox.lower.y = minY; bbox.lower.z = minZ;
    bbox.upper.x = maxX; bbox.upper.y = maxY; bbox.upper.z = maxZ;
}

__device__ __host__
inline int project3(vec3f& ax, vec3f& p1, vec3f& p2, vec3f& p3)
{
    float P1 = ax.dot(p1);
    float P2 = ax.dot(p2);
    float P3 = ax.dot(p3);

    float mx1 = fmax3(P1, P2, P3);
    float mn1 = fmin3(P1, P2, P3);

    if (mn1 > 0) return 0;
    if (0 > mx1) return 0;
    return 1;
}

__device__ __host__
inline int project6(vec3f& ax,
                    vec3f& p1, vec3f& p2, vec3f& p3,
                    vec3f& q1, vec3f& q2, vec3f& q3)
{
    float P1 = ax.dot(p1);
    float P2 = ax.dot(p2);
    float P3 = ax.dot(p3);
    float Q1 = ax.dot(q1);
    float Q2 = ax.dot(q2);
    float Q3 = ax.dot(q3);

    float mx1 = fmax3(P1, P2, P3);
    float mn1 = fmin3(P1, P2, P3);
    float mx2 = fmax3(Q1, Q2, Q3);
    float mn2 = fmin3(Q1, Q2, Q3);

    if (mn1 > mx2) return 0;
    if (mn2 > mx1) return 0;
    return 1;
}

__device__ __host__
inline bool collide(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
    vec3f p1;
    vec3f p2 = P2 - P1;
    vec3f p3 = P3 - P1;
    vec3f q1 = Q1 - P1;
    vec3f q2 = Q2 - P1;
    vec3f q3 = Q3 - P1;

    vec3f e1 = p2 - p1;
    vec3f e2 = p3 - p2;
    vec3f e3 = p1 - p3;

    vec3f f1 = q2 - q1;
    vec3f f2 = q3 - q2;
    vec3f f3 = q1 - q3;

    vec3f n1 = e1.cross(e2);
    vec3f m1 = f1.cross(f2);

    vec3f g1 = e1.cross(n1);
    vec3f g2 = e2.cross(n1);
    vec3f g3 = e3.cross(n1);

    vec3f h1 = f1.cross(m1);
    vec3f h2 = f2.cross(m1);
    vec3f h3 = f3.cross(m1);

    vec3f ef11 = e1.cross(f1);
    vec3f ef12 = e1.cross(f2);
    vec3f ef13 = e1.cross(f3);
    vec3f ef21 = e2.cross(f1);
    vec3f ef22 = e2.cross(f2);
    vec3f ef23 = e2.cross(f3);
    vec3f ef31 = e3.cross(f1);
    vec3f ef32 = e3.cross(f2);
    vec3f ef33 = e3.cross(f3);

    vec3f p1_q1 = p1 - q1;
    vec3f p2_q1 = p2 - q1;
    vec3f p3_q1 = p3 - q1;
    if (!project3(n1, q1, q2, q3)) return false;
    if (!project3(m1, p1_q1, p2_q1, p3_q1)) return false;

    if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

    return true;
}