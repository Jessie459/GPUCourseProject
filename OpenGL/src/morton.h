#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


// Calculates a 30-bit Morton code for the given 3D point
// located within the unit cube [0,1].
__device__
inline unsigned int mortonCode(float x, float y, float z)
{
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__device__
inline unsigned int mortonCode(float3 point)
{
    float x = fmin(fmax(point.x * 1024.0f, 0.0f), 1023.0f);
    float y = fmin(fmax(point.y * 1024.0f, 0.0f), 1023.0f);
    float z = fmin(fmax(point.z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__device__
inline int clz(const unsigned int lhs, const unsigned int rhs)
{
    return __clz(lhs ^ rhs);
}