#ifndef _SRENDERER_MEDIUM_GRID_SLANG_
#define _SRENDERER_MEDIUM_GRID_SLANG_

#include "common/math.slang"
#include "common/geometry.slang"
#include "srenderer/spt-bindings.slang"

struct SampledGrid {
    int nx; int ny; int nz;
    int offset;
    bool valid;
    
    __init() { 
        valid = false; 
        nx = ny = nz = 0;
        offset = 0;
    }

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
        return se_medium_grid_buffer[offset + (p.z * ny + p.y) * nx + p.x];
    }

    int x_size() { return nx; }
    int y_size() { return ny; }
    int z_size() { return nz; }
};

struct RGBSampledGrid {
    int nx; int ny; int nz;
    int offset;
    bool valid;

    __init() { 
        valid = false; 
        nx = ny = nz = 0;
        offset = 0;
    }

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
            se_medium_grid_buffer[offset + ((p.z * ny + p.y) * nx + p.x) * 3 + 0],
            se_medium_grid_buffer[offset + ((p.z * ny + p.y) * nx + p.x) * 3 + 1],
            se_medium_grid_buffer[offset + ((p.z * ny + p.y) * nx + p.x) * 3 + 2]
        );
    }

    int x_size() { return nx; }
    int y_size() { return ny; }
    int z_size() { return nz; }
};

#endif // _SRENDERER_MEDIUM_GRID_SLANG_