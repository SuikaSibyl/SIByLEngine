#include "srenderer/spt.slang"
#include "srenderer/materials.slang"
#include "common/random.slang"
#include "common/hashing.slang"

struct ChoosePoint {
    MinimalHit hit;
};

RWTexture2D<float4> rw_output;
RWStructuredBuffer<ChoosePoint> rw_chose_point;

enum DisplyaModeEnum {
    PATH_TRACING = 0,
    ALBEDO = 1,
    GEOMETRY_NORMAL = 2,
    SHADING_NORMAL = 3,
};

[[vk::push_constant]]
cbuffer PushConstants {
    DisplyaModeEnum display_mode;
    int random_seed;
    int mouse_state;
    int highlight_geometry;
    int2 mouse_pixel;
};

float3 path_tracing(Ray ray, inout random::RandomSampler sampler, int scene_index) {
    Ray primary_ray = ray;
    IntersectionPayload primary_payload = intersection_ray_query(ray, scene_index);
    // If the primary ray hit nothing, return the background color (white):
    // if (!primary_payload.hit.has_hit()) return float3(1, 1, 1);

    // return float3(1, 0, 1);
    GeometryData geometry = scene_read_geometry(primary_payload.hit.geometryID);
    Optional<MaterialData> material = scene_read_material(geometry);
    if (!material.hasValue) return float3(1, 0, 1);

    float3 radiance = float3(0.f, 0.f, 0.f);
    
    // If the primary ray hit something, sample the BSDF:
    if (let bs = primary_payload.bsdf_sample(primary_ray, sampler.next_float3())) {
        Ray ray = primary_payload.hit.spawn_ray(bs.wo);
        IntersectionPayload payload = intersection_ray_query(ray, scene_index);
        if (payload.hit.has_hit()) {
            // If the ray hit something, accumulate the radiance:
            GeometryData next_geometry = scene_read_geometry(payload.hit.geometryID);
            Optional<MaterialData> material = scene_read_material(next_geometry.materialID);
            radiance += bs.bsdf * material.value.emission() / bs.pdf;
        }
        else {
            radiance += bs.bsdf / bs.pdf;
        }
    }

    return radiance;
}

[shader("compute")]
[numthreads(32, 4, 1)]
[require(spirv, rayquery)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    int2 resolution; rw_output.GetDimensions(resolution.x, resolution.y);
    const int2 pixel = int2(dtid.xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    random::RandomSampler sampler = random::init_random_sampler(pixel, random_seed);

    uint scene_index = pixel.x / (resolution.x / 2)
                        + 2 * (pixel.y / (resolution.y / 2));
    scene_index = 0; // TEMPORARY FIXED TO SCENE 0
    const int2 resolution_per_scene = resolution / 2;
    const int2 pixel_per_scene = pixel % resolution_per_scene;

    CameraData camera = scene_read_camera(scene_index);
    Ray ray = camera.spawn(pixel_per_scene, resolution_per_scene, false);

    float3 output = float3(0, 0, 0);

    if (mouse_state == 1 || mouse_state == 2) {
        if (all(mouse_pixel == pixel))
            rw_chose_point[0].hit = intersection_ray_query_minimal(ray, scene_index);
    }

    bool is_edge = false;
    if (highlight_geometry > 0) {
        MinimalHit central_hit = intersection_ray_query_minimal(ray, scene_index);
        if (central_hit.geometryID == highlight_geometry) {
            for (int i = 0; i < 8; ++i) {
                Ray jitter_ray = ray;
                jitter_ray.direction = ray.direction + random_point_inside_sphere(sampler.next_float2()) * 0.004f;
                MinimalHit jitter_hit = intersection_ray_query_minimal(jitter_ray, scene_index);
                if (jitter_hit.geometryID != central_hit.geometryID) {
                    is_edge = true;
                    break;
                }
            }

            if (is_edge == false) {
                int primitiveID = central_hit.primitiveID;
                random::RandomSampler tmp; tmp.state = primitiveID;
                rw_output[pixel] = float4(tmp.next_float3(), 1);
                return;
            }
        }
    }

    if (display_mode == DisplyaModeEnum::PATH_TRACING) {
        output = path_tracing(ray, sampler, scene_index);
    }
    else {
        IntersectionPayload payload = intersection_ray_query(ray, scene_index);
        if (payload.hit.has_hit()) {
            GeometryData geometry = scene_read_geometry(payload.hit.geometryID);
            Optional<MaterialData> material = scene_read_material(geometry.materialID);
            if (material.hasValue) {
                switch (display_mode) {
                case DisplyaModeEnum::ALBEDO: output = material.value.albedo() + material.value.emission(); break;
                case DisplyaModeEnum::GEOMETRY_NORMAL: output = payload.hit.geometryNormal * 0.5 + 0.5; break;
                case DisplyaModeEnum::SHADING_NORMAL: output = payload.hit.shadingNormal * 0.5 + 0.5; break;
                default: output = float3(1, 0, 1); break;
                }
            }
        }
    }

    if (is_edge) output = float3(1, 0.8, 0.5);

    rw_output[pixel] = float4(output, 1);
}