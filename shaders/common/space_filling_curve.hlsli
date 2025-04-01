#ifndef _SRENDERER_SPACCE_FILLING_CURVE_INCLUDED_
#define _SRENDERER_SPACCE_FILLING_CURVE_INCLUDED_

#include "cpp_compatible.hlsli"

/***************************************************************************
# Implementation of various space-filling curves.
# 1. Z-order/Morton curve 2D
@ 2. Z-order/Morton curve 3D
 **************************************************************************/

// The following code about 2D Z-Order is adapted from:
// @ref: NVIDIAGameWorks/ RTXDI
// @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/
//       1018431b07fae1ea02ac2002eb7bdee882538d30/rtxdi-sdk/include/rtxdi/RtxdiMath.hlsli#L55

// "Explodes" an integer, i.e. inserts a 0 between each bit.
// Takes inputs up to 16 bit wide.
// For example, 0b11111111 -> 0b1010101010101010
uint IntegerExplode1Bit(uint x) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

// Reverse of IntegerExplode1Bit, i.e. takes every other bit in the integer and compresses
// those bits into a dense bit firld. Takes 32-bit inputs, produces 16-bit outputs.
// For example, 0b'abcdefgh' -> 0b'0000bdfh'
uint IntegerCompact1Bit(uint x) {
    x = (x & 0x11111111) | ((x & 0x44444444) >> 1);
    x = (x & 0x03030303) | ((x & 0x30303030) >> 2);
    x = (x & 0x000F000F) | ((x & 0x0F000F00) >> 4);
    x = (x & 0x000000FF) | ((x & 0x00FF0000) >> 8);
    return x;
}

// Converts a 2D position to a linear index following a Z-curve pattern.
uint ZCurve2DToMortonCode(uint2 xy) {
    return IntegerExplode1Bit(xy[0]) | (IntegerExplode1Bit(xy[1]) << 1);
}

// Converts a linear to a 2D position following a Z-curve pattern.
uint2 MortonCodeToZCurve2D(uint index) {
    return uint2(IntegerCompact1Bit(index), IntegerCompact1Bit(index >> 1));
}

// The following code about 3D Z-Order is adapted from:
// @ref: Thinking Parallel, Part III: Tree Construction on the GPU
// @url: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

// "Explodes" an integer, i.e. inserts two 0-s between each bit.
// Takes inputs up to 10 bit wide.
// For example, 0b11111111 -> 0b001001001001001001001001
uint IntegerExplode2Bit(uint x) {
    x = (x * 0x00010001) & 0xFF0000FF;
    x = (x * 0x00000101) & 0x0F00F00F;
    x = (x * 0x00000011) & 0xC30C30C3;
    x = (x * 0x00000005) & 0x49249249;
    return x;
}

// Converts a 3D position to a linear index following a Z-curve pattern.
uint ZCurve3DToMortonCode(uint3 xyz) {
    const uint xx = IntegerExplode2Bit(xyz.x);
    const uint yy = IntegerExplode2Bit(xyz.y);
    const uint zz = IntegerExplode2Bit(xyz.z);
    return xx * 4 + yy * 2 + zz;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint ZCurve3DToMortonCode(float3 xyz) {
    const float x = min(max(xyz.x * 1024.0f, 0.0f), 1023.0f);
    const float y = min(max(xyz.y * 1024.0f, 0.0f), 1023.0f);
    const float z = min(max(xyz.z * 1024.0f, 0.0f), 1023.0f);
    return ZCurve3DToMortonCode(uint3(uint(x), uint(y), uint(z)));
}

void updateSmallestIndexAndDistance(
    inout_ref(float) smallestDistance,
    inout_ref(int) smallestIndex,
    float currentDistance,
    int currentIndex)
{
    if (currentDistance < smallestDistance) {
        smallestIndex = currentIndex;
        smallestDistance = currentDistance;
    }
}

void shift_update(inout_ref(int) integer, int value, int value_size_in_bit = 1) {
    integer <<= value_size_in_bit;
    integer |= value;
}

uint direction_code_bits(int sections) {
    return (3 + 2 * sections + 1) + 1;
}

int direction_code(in_ref(float3) vec, const int sections) {
    // this function takes a direction vector (normalized) i.e. a point on the unit sphere centered on 0,0,0.
    // it splits the unit sphere into the 8 octants and then treats each as a triangle in space that gets tesselated
    // the points on the sphere are categorized with their squared distance
    if (dot(vec, vec) < 0.0001) {
        // we found a source "omnidirectional" light (indicated by its normal vector being equal to 0). needs its own index
        return (1 << (3 + 2 * sections + 1)); // 3 shifts for the octant + 2 * sections as each subdivision has 4 divison.
    }
    
    // get base vector::defines the quadrant the vector is in
    int final_int = 0;
    float3 x_base_vec = { 0, 0, 0 };
    float3 y_base_vec = { 0, 0, 0 };
    float3 z_base_vec = { 0, 0, 0 };

    float3 plane_normal = { 0, 0, 0 };
    float normal_constant = 1.f / sqrt(3.f); // all normals do have the same constant value their only difference is their sign
    if (vec.x >= 0) {
        x_base_vec.x = 1;
        shift_update(final_int, 1);
        plane_normal.x = normal_constant;
    }
    else {
        x_base_vec.x = -1;
        shift_update(final_int, 0);
        plane_normal.x = -normal_constant;
    }
    if (vec.y >= 0) {
        y_base_vec.y = 1;
        shift_update(final_int, 1);
        plane_normal.y = normal_constant;
    }
    else {
        y_base_vec.y = -1;
        shift_update(final_int, 0);
        plane_normal.y = -normal_constant;
    }
    if (vec.z >= 0) {
        z_base_vec.z = 1;
        shift_update(final_int, 1);
        plane_normal.z = normal_constant;
    }
    else {
        z_base_vec.z = -1;
        shift_update(final_int, 0);
        plane_normal.z = -normal_constant;
    }

    float scaling_factor = normal_constant / dot(plane_normal, vec);

    float3 projected_vec = vec * scaling_factor;
    // float3 projected_vec = vec ;

    for (int i = 0; i < sections; i++) {
        float3 xy_mix_vector = normalize(x_base_vec + y_base_vec);
        float3 xz_mix_vector = normalize(x_base_vec + z_base_vec);
        float3 yz_mix_vector = normalize(y_base_vec + z_base_vec);

        float distSq_x = dot(projected_vec, x_base_vec);
        float distSq_y = dot(projected_vec, y_base_vec);
        float distSq_z = dot(projected_vec, z_base_vec);

        float distSq_xy = dot(projected_vec, xy_mix_vector);
        float distSq_xz = dot(projected_vec, xz_mix_vector);
        float distSq_yz = dot(projected_vec, yz_mix_vector);

        float subsection0 = distSq_y + distSq_xy + distSq_yz;            // corner with y (top part)
        float subsection1 = distSq_z + distSq_xz + distSq_yz;            // corner with z (left part)
        float subsection2 = distSq_x + distSq_xy + distSq_xz;            // corner with x (right part)
        float subsection3 = (distSq_xz + distSq_xy + distSq_yz) * 0.85f; // middle part. multiplied with 0.85 as this somewhat compensates the projected curvature of this method and balances the deviations over all subsections

        float smallest_distance = 100; // distances cant be bigger than 4. we pick 100 to just to be sure
        int smallest_index = -1;

        updateSmallestIndexAndDistance(smallest_distance, smallest_index, subsection0, 0);
        updateSmallestIndexAndDistance(smallest_distance, smallest_index, subsection1, 1);
        updateSmallestIndexAndDistance(smallest_distance, smallest_index, subsection2, 2);
        updateSmallestIndexAndDistance(smallest_distance, smallest_index, subsection3, 3);
        
        // updating the new basevectors for next iteration
        if (smallest_index == 0) {
            x_base_vec = xy_mix_vector;
            z_base_vec = yz_mix_vector;
        }
        if (smallest_index == 1) {
            x_base_vec = xz_mix_vector;
            y_base_vec = yz_mix_vector;
        }
        if (smallest_index == 2) {
            y_base_vec = xy_mix_vector;
            z_base_vec = xz_mix_vector;
        }
        if (smallest_index == 3) {
            x_base_vec = xy_mix_vector;
            y_base_vec = yz_mix_vector;
            z_base_vec = xz_mix_vector;
        }
        
        shift_update(final_int, smallest_index, 2);
    }
    return final_int;
}

uint32_t inverse_morton3(uint32_t x) {
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xC30C30C3;
    x = (x | (x >> 4)) & 0x0F00F00F;
    x = (x | (x >> 8)) & 0xFF0000FF;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}

#endif