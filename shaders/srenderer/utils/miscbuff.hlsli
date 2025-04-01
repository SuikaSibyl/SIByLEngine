#ifndef _SRENDERER_MISC_BUFF_HLSLI_
#define _SRENDERER_MISC_BUFF_HLSLI_

#include "common/math.hlsli"
#include "common/geometry.hlsli"
#include "srenderer/scene-binding.hlsli"

struct SampledGrid {
    int nx; int ny; int nz;
    int offset;
    bool valid;
    
    __init() { valid = false; }

    float look_up(float3 p) {
        // Compute voxel coordinates and offsets for p
        float3 pSamples = float3(
            p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
        int3 pi = (int3)floor(pSamples);
        float3 d = pSamples - (float3)pi;
        // Return trilinearly interpolated voxel values
        float d00 = lerp(look_up(pi),
                         look_up(pi + int3(1, 0, 0)), d.x);
        float d10 = lerp(look_up(pi + int3(0, 1, 0)),
                         look_up(pi + int3(1, 1, 0)), d.x);
        float d01 = lerp(look_up(pi + int3(0, 0, 1)),
                         look_up(pi + int3(1, 0, 1)), d.x);
        float d11 = lerp(look_up(pi + int3(0, 1, 1)),
                         look_up(pi + int3(1, 1, 1)), d.x);
        return lerp(lerp(d00, d10, d.y), lerp(d01, d11, d.y), d.z);
    }
    
    float look_up(int3 p) {
        bounds3i sampleBounds = { int3(0, 0, 0), int3(nx, ny, nz) };
        p = clamp(p, sampleBounds.pMin, sampleBounds.pMax - int3(1, 1, 1));
        // if (!bounds3i::inside_exclusive(p, sampleBounds)) return 0;
        return GPUScene_grid_storage[offset + (p.z * ny + p.y) * nx + p.x];
    }

    int x_size() { return nx; }
    int y_size() { return ny; }
    int z_size() { return nz; }
};

struct RGBSampledGrid {
    int nx; int ny; int nz;
    int offset;
    bool valid;

    __init() { valid = false; }

    float3 look_up(float3 p) {
        // Compute voxel coordinates and offsets for p
        float3 pSamples = float3(
            p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
        int3 pi = (int3)floor(pSamples);
        float3 d = pSamples - (float3)pi;
        // Return trilinearly interpolated voxel values
        float3 d00 = lerp(look_up(pi),
                         look_up(pi + int3(1, 0, 0)), d.x);
        float3 d10 = lerp(look_up(pi + int3(0, 1, 0)),
                         look_up(pi + int3(1, 1, 0)), d.x);
        float3 d01 = lerp(look_up(pi + int3(0, 0, 1)),
                         look_up(pi + int3(1, 0, 1)), d.x);
        float3 d11 = lerp(look_up(pi + int3(0, 1, 1)),
                         look_up(pi + int3(1, 1, 1)), d.x);
        return lerp(lerp(d00, d10, d.y), lerp(d01, d11, d.y), d.z);
    }
    
    float3 look_up(int3 p) {
        bounds3i sampleBounds = { int3(0, 0, 0), int3(nx, ny, nz) };
        p = clamp(p, sampleBounds.pMin, sampleBounds.pMax - int3(1, 1, 1));
        // if (!bounds3i::inside_exclusive(p, sampleBounds)) return 0;
        return float3(
            GPUScene_grid_storage[offset + ((p.z * ny + p.y) * nx + p.x) * 3 + 0],
            GPUScene_grid_storage[offset + ((p.z * ny + p.y) * nx + p.x) * 3 + 1],
            GPUScene_grid_storage[offset + ((p.z * ny + p.y) * nx + p.x) * 3 + 2]
        );
    }

    int x_size() { return nx; }
    int y_size() { return ny; }
    int z_size() { return nz; }
};

#endif // _SRENDERER_MISC_BUFF_HLSLI_