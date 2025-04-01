#ifndef _SRENDERER_MEDIUM_MEDIUM_HLSLI_
#define _SRENDERER_MEDIUM_MEDIUM_HLSLI_

#include "common/geometry.hlsli"
#include "common/raycast.hlsli"
#include "phase.hlsli"
#include "srenderer/spt.hlsli"
#include "srenderer/utils/miscbuff.hlsli"

namespace imedium {
struct RayMajorantSegment {
    float3 sigma_maj;
    float tMin;
    float tMax;
    bool is_valid;
};

struct HomogeneousMajorantIterator {
    RayMajorantSegment seg;
    bool called;

    __init() { called = true; }
    __init(float t_min, float tMax, float3 sigma_maj) {
        seg = { sigma_maj, t_min, tMax, true };
        called = false;
    }
    
    [mutating]
    RayMajorantSegment next() {
        if (called) {
            RayMajorantSegment seg = { float3(0), 0, 0, false };
            return seg;
        }
        called = true;
        return seg;
    }
};

struct MajorantGrid {
    bounds3 bounds;
    int3 resolution;
    int offset;

    float look_up(int3 idx) {
        return GPUScene_grid_storage[offset +
                                      (idx.z * resolution.y + idx.y) * resolution.x + idx.x];
    }

    float look_up(int x, int y, int z) { return look_up(int3(x, y, z)); }

    void set(int3 idx, float val) {
        GPUScene_grid_storage[offset +
                               (idx.z * resolution.y + idx.y) * resolution.x + idx.x] = val;
    }

    // returns the bounding box
    // corresponding to the specified voxel in the grid
    bounds3 voxel_bounds(int3 idx) {
        float3 p0 = float3(
            float(idx.x) / resolution.x,
            float(idx.y) / resolution.y,
            float(idx.z) / resolution.z);
        float3 p1 = float3(
            float(idx.x + 1) / resolution.x,
            float(idx.y + 1) / resolution.y,
            float(idx.z + 1) / resolution.z);
        return bounds3(p0, p1);
    }
};

struct DDAMajorantIterator {
    float3 sigma_t;
    float tMin = k_inf;
    float tMax = -k_inf;
    MajorantGrid grid;

    float nextCrossingT[3];
    float deltaT[3];
    int step[3];
    int voxelLimit[3];
    int voxel[3];

    __init() {}
    __init(Ray ray, float tMin, float tMax, MajorantGrid grid, float3 sigma_t) {
        this.tMin = tMin;
        this.tMax = tMax;
        this.grid = grid;
        this.sigma_t = sigma_t;
        // Set up 3D DDA for ray through the majorant grid
        float3 diag = grid.bounds.diagonal();
        float3 direction = float3(
            ray.direction.x / diag.x,
            ray.direction.y / diag.y,
            ray.direction.z / diag.z);
        Ray ray_grid = { grid.bounds.offset(ray.origin), 0, direction, k_inf };
        float3 gridIntersect = ray_grid.at(tMin);
        for (int axis = 0; axis < 3; ++axis) {
            // Initialize ray stepping parameters for axis
            // Compute current voxel for axis and handle negative zero direction
            voxel[axis] = int(clamp(
                gridIntersect[axis] * grid.resolution[axis],
                0, grid.resolution[axis] - 1));
            deltaT[axis] = 1 / (abs(ray_grid.direction[axis]) * grid.resolution[axis]);
            if (ray_grid.direction[axis] == -0.f)
                ray_grid.direction[axis] = 0.f;

            if (ray_grid.direction[axis] >= 0) {
                // Handle ray with positive direction for voxel stepping
                float nextVoxelPos = float(voxel[axis] + 1) / grid.resolution[axis];
                nextCrossingT[axis] = tMin +
                    (nextVoxelPos - gridIntersect[axis]) /
                        ray_grid.direction[axis];
                step[axis] = 1;
                voxelLimit[axis] = grid.resolution[axis];
            }
            else {
                // Handle ray with negative direction for voxel stepping
                float nextVoxelPos = float(voxel[axis]) / grid.resolution[axis];
                nextCrossingT[axis] = tMin + (nextVoxelPos - gridIntersect[axis]) /
                                                 ray_grid.direction[axis];
                step[axis] = -1;
                voxelLimit[axis] = -1;
            }
        }
    }

    // Generate the majorant segment for the current voxel
    // and take a step to the next using the DDA/
    // Traversal terminates when the remaining parametric range [tmin, tmax] is degenerate
    [mutating]
    RayMajorantSegment next() {
        if (tMin >= tMax) {
            RayMajorantSegment seg;
            seg.is_valid = false;
            return seg;
        }

        // Find stepAxis for stepping to next voxel and exit point tVoxelExit
        int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
                   ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
                   ((nextCrossingT[1] < nextCrossingT[2]));
        const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
        int stepAxis = cmpToAxis[bits];
        float tVoxelExit = min(tMax, nextCrossingT[stepAxis]);

        // Get maxDensity for current voxel and initialize RayMajorantSegment, seg
        // Computing the majorant for the current voxel is a matter of
        // multiplying sigma_t with the maximum density value over the voxel’s volume.
        float3 sigma_maj = sigma_t * grid.look_up(voxel[0], voxel[1], voxel[2]);
        RayMajorantSegment seg = { sigma_maj, tMin, tVoxelExit, true };

        // Advance to next voxel in maximum density grid
        // updating the DDAMajorantIterator’s state to reflect
        // stepping to the next voxel in the ray’s path
        tMin = tVoxelExit;
        if (nextCrossingT[stepAxis] > tMax) tMin = tMax;
        voxel[stepAxis] += step[stepAxis];
        if (voxel[stepAxis] == voxelLimit[stepAxis]) tMin = tMax;
        nextCrossingT[stepAxis] += deltaT[stepAxis];

        return seg;
    }
};

struct RayMajorantIterator {
    enum IteratorType { homogeneous, dda, };
    HomogeneousMajorantIterator homogeneous;
    DDAMajorantIterator dda;
    IteratorType type;
    
    [mutating]
    RayMajorantSegment next() {
        if (type == IteratorType::homogeneous) {
            return homogeneous.next();
        } else if (type == IteratorType::dda) {
            return dda.next();
        }
    }
};

// wraps up the values that describe 
// scattering and emission at a point inside a medium
struct MediumProperties {
    float3 sigma_a;
    float3 sigma_s;
    float3 Le;
    PhasePacket phase;

    // when initialized to their default values, 
    // its member variables together
    // indicate no scattering or emission
    __init() {
        sigma_a = float3(0);
        sigma_s = float3(0);
        Le = float3(0);
        phase = { float3(0) };
    }
};
}; // namespace imedium

// Interface for medium parameters
interface IMediumParameter : IDifferentiable {};

// Interface for medium
interface IMedium {
    // Associated a parameter type for each medium
    associatedtype TParam : IMediumParameter;

    // indicates whether it includes any volumetric emission
    static bool is_emissive(TParam param);

    // returns information about the scattering and emission properties
    // of the medium at a specified rendering-space point 
    // in the form of a MediumProperties object.
    static imedium::MediumProperties sample_point(float3 p, TParam param);

    // provides information about the medium’s majorant sigma_maj along the ray's extent
    static imedium::RayMajorantIterator sample_ray(Ray ray, float tMax, TParam param);
};

// Homogeneous Medium
// ----------------------------------------------------
struct HomogeneousMediumParameter: IMediumParameter {
    float3 sigma_a;
    float3 sigma_s;
    float3 Le;
    PhasePacket phase;
    
    __init() {}
    __init(MediumPacket packet) {
        sigma_a = packet.get_sigma_a();
        sigma_s = packet.get_sigma_s();
        Le = float3(0);
        phase = { packet.get_g() };
    }
};

struct HomogeneousMedium : IMedium {
    // Associated a parameter type for each medium
    typedef HomogeneousMediumParameter TParam;

    // indicates whether it includes any volumetric emission
    static bool is_emissive(TParam param) {
        return maxComponent(param.Le) > 0;
    }

    // returns information about the scattering and emission properties
    // of the medium at a specified rendering-space point
    // in the form of a MediumProperties object.
    static imedium::MediumProperties sample_point(float3 p, TParam param) {
        imedium::MediumProperties props;
        props.sigma_a = param.sigma_a;
        props.sigma_s = param.sigma_s;
        props.Le = param.Le;
        props.phase = param.phase;
        return props;
    }

    // provides information about the medium’s majorant sigma_maj along the ray's extent
    static imedium::RayMajorantIterator sample_ray(Ray ray, float tMax, TParam param) {
        imedium::RayMajorantIterator iter;
        iter.type = imedium::RayMajorantIterator::IteratorType::homogeneous;
        iter.homogeneous = imedium::HomogeneousMajorantIterator(
            0, tMax, param.sigma_a + param.sigma_s);
        return iter;
    }
};

// Uniform Grid Medium
// ----------------------------------------------------
struct GridMediumParameter : IMediumParameter {
    bounds3 bounds;
    float3 sigma_a_spec;
    float3 sigma_s_spec;
    float temperatureScale;
    float3 Le_spec;
    float LeScale;
    float4x4 w2o;
    float4x4 o2w;
    bool isEmissive;
    PhasePacket phase;
    bounds3 temperatureBounds;
    imedium::MajorantGrid majorantGrid;
    SampledGrid densityGrid;
    SampledGrid LeGrid;
    SampledGrid temperatureGrid;
    
    __init() {}
    __init(MediumPacket packet) {
        sigma_a_spec = packet.get_sigma_a();
        sigma_s_spec = packet.get_sigma_s();
        Le_spec = float3(0);
        phase = { packet.get_g() };
        LeScale = packet.LeScale;
        temperatureScale = packet.temperatureScale;
        isEmissive = false;

        o2w = packet.medium_to_world();
        w2o = packet.world_to_medium();
        bounds = bounds3(packet.bound_min, packet.bound_max);
        temperatureBounds = bounds3(packet.temperature_bound_min, packet.temperature_bound_max);

        densityGrid = {
            packet.density_xyz.x, packet.density_xyz.y,
            packet.density_xyz.z, packet.density_offset,
            packet.density_offset != -1
        };

        temperatureGrid = {
            packet.temperature_xyz.x, packet.temperature_xyz.y,
            packet.temperature_xyz.z, packet.temperature_offset,
            packet.temperature_offset != -1
        };

        if(temperatureGrid.valid) {
            isEmissive = true;
        }
        
        majorantGrid = {
            bounds, packet.majorant_xyz, packet.majorant_offset
        };
    }
};

struct GridMedium : IMedium {
    // Associated a parameter type for each medium
    typedef GridMediumParameter TParam;

    // indicates whether it includes any volumetric emission
    static bool is_emissive(TParam param) {
        return param.isEmissive;
    }

    // returns information about the scattering and emission properties
    // of the medium at a specified rendering-space point
    // in the form of a MediumProperties object.
    static imedium::MediumProperties sample_point(float3 p, TParam param) {
        // Sample spectra for grid medium sigma_a and sigma_s
        float3 sigma_a = param.sigma_a_spec;
        float3 sigma_s = param.sigma_s_spec;
        // Scale scattering coefficients by medium density at p
        p = mul(float4(p, 1), param.o2w).xyz;
        float3 p_density = param.bounds.offset(p);

        float d = param.densityGrid.look_up(p_density);
        sigma_a *= d;
        sigma_s *= d;
        // Compute grid emission Le at p
        float3 Le = float3(0);
        if (param.isEmissive) {
            // float scale = param.LeScale.look_up(p);
            float scale = param.LeScale;
            if (scale > 0) {
                // Compute emitted radiance using temperatureGrid or Le_spec
                if (param.temperatureGrid.valid) {
                    float3 p_temperature = param.temperatureBounds.offset(p);
                    float temp = param.temperatureGrid.look_up(p_temperature);
                    Le = param.LeScale * 100 * black_body_spectrum(temp * param.temperatureScale);
                } else
                    Le = scale * param.Le_spec;
            }
        }

        imedium::MediumProperties props;
        props.sigma_a = sigma_a;
        props.sigma_s = sigma_s;
        props.Le = Le;
        props.phase = param.phase;
        return props;
    }
    
    // provides information about the medium’s majorant sigma_maj along the ray's extent
    static imedium::RayMajorantIterator sample_ray(Ray ray, float raytMax, TParam param) {
        // Transform ray to medium’s space and compute bounds overlap
        // apply inverse transformation to ray
        ray.origin = mul(float4(ray.origin, 1), param.w2o).xyz;
        ray.direction = mul(float4(ray.direction, 0), param.w2o).xyz;
        float tMin; float tMax;
        if (!param.bounds.intersect_p(ray.origin, ray.direction, raytMax, tMin, tMax))
            return {};
        // Sample spectra for grid medium sigma_a and sigma_s
        float3 sigma_a = param.sigma_a_spec;
        float3 sigma_s = param.sigma_s_spec;
        float3 sigma_t = sigma_a + sigma_s;
        // Create DDA majorant iterator for grid medium
        imedium::RayMajorantIterator iter;
        iter.type = imedium::RayMajorantIterator::IteratorType::dda;
        iter.dda = imedium::DDAMajorantIterator(
            ray, tMin, tMax, param.majorantGrid, sigma_t);
        return iter;
        
        // imedium::RayMajorantIterator iter;
        // iter.type = imedium::RayMajorantIterator::IteratorType::homogeneous;
        // iter.homogeneous = imedium::HomogeneousMajorantIterator(
        //     0, raytMax, param.sigma_s_spec + param.sigma_a_spec);
        // return iter;
    }
};

// RGB Grid Medium
// ----------------------------------------------------
struct RGBGridMediumParameter : IMediumParameter {
    bounds3 bounds;
    float scale;
    bool isEmissive;
    PhasePacket phase;
    float4x4 w2o;
    float4x4 o2w;
    RGBSampledGrid sigma_a_Grid;
    RGBSampledGrid sigma_s_Grid;
    imedium::MajorantGrid majorantGrid;

    __init() {}
    __init(MediumPacket packet) {
        o2w = packet.medium_to_world();
        
        // w2o = packet.world_to_medium();
        // o2w = float4x4(1);
        // w2o = float4x4(1);
        bounds = bounds3(
            mul(float4(packet.bound_min, 1), o2w).xyz,
            mul(float4(packet.bound_max, 1), o2w).xyz, );
        scale = packet.scale;
        phase = { packet.get_g() };
        isEmissive = false;

        sigma_a_Grid = {
            packet.density_xyz.x, packet.density_xyz.y,
            packet.density_xyz.z, packet.density_offset,
            packet.density_offset != -1
        };

        sigma_s_Grid = {
            packet.temperature_xyz.x, packet.temperature_xyz.y,
            packet.temperature_xyz.z, packet.temperature_offset,
            packet.temperature_offset != -1
        };

        majorantGrid = {
            bounds, packet.majorant_xyz, packet.majorant_offset
        };
    }
};

struct RGBGridMedium : IMedium {
    // Associated a parameter type for each medium
    typedef RGBGridMediumParameter TParam;
    
    // indicates whether it includes any volumetric emission
    static bool is_emissive(TParam param) {
        return param.isEmissive;
    }

    // returns information about the scattering and emission properties
    // of the medium at a specified rendering-space point
    // in the form of a MediumProperties object.
    static imedium::MediumProperties sample_point(float3 p, TParam param) {
        // Scale scattering coefficients by medium density at p
        // p = mul(float4(p, 1), param.w2o).xyz;
        float3 p_density = param.bounds.offset(p);

        // Sample spectra for grid medium sigma_a and sigma_s
        float3 sigma_a = param.sigma_a_Grid.look_up(p_density) * param.scale;
        float3 sigma_s = param.sigma_s_Grid.look_up(p_density) * param.scale;
        
        // Compute grid emission Le at p
        float3 Le = float3(0);

        imedium::MediumProperties props;
        props.sigma_a = sigma_a;
        props.sigma_s = sigma_s;
        props.Le = Le;
        props.phase = param.phase;
        return props;
    }
    
    // provides information about the medium’s majorant sigma_maj along the ray's extent
    static imedium::RayMajorantIterator sample_ray(Ray ray, float raytMax, TParam param) {
        // Transform ray to medium’s space and compute bounds overlap
        // apply inverse transformation to ray
        // ray.origin = mul(float4(ray.origin, 1), param.w2o).xyz;
        // ray.direction = mul(float4(ray.direction, 0), param.w2o).xyz;
        float tMin; float tMax;
        if (!param.bounds.intersect_p(ray.origin, ray.direction, raytMax, tMin, tMax))
            return {};
        // Sample spectra for grid medium sigma_a and sigma_s
        float3 sigma_t = { 1.f, 1.f, 1.f };
        // Create DDA majorant iterator for grid medium
        imedium::RayMajorantIterator iter;
        iter.type = imedium::RayMajorantIterator::IteratorType::dda;
        iter.dda = imedium::DDAMajorantIterator(
            ray, tMin, tMax, param.majorantGrid, sigma_t);
        return iter;
    }
};

#endif // _SRENDERER_MEDIUM_MEDIUM_HLSLI_