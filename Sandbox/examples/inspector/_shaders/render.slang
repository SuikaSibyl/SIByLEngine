#include "common/gaussian.hlsli"
#include "common/math.hlsli"
#include "srenderer/lights.hlsli"
#include "srenderer/lights/envmap.hlsli"
#include "srenderer/materials/lambertian.hlsli"
#include "srenderer/rayquery.hlsli"
#include "srenderer/materials.hlsli"

[[vk::push_constant]] cbuffer PushConstants {
    int random_seed;
    InspectorMode inspector_mode;
    float wireframe_thickness;
    float wireframe_smoothing;
    float3 wireframe_color;
};

enum InspectorMode {
    ALBEDO,
    EMISSION,
    FACEFORWARD,
    BARYCENTRIC,
    NORMAL,
    POSITION,
    ADDITIONAL_TEXTURE,
};

RWTexture2D<float4> output;

groupshared float3 smem[6][34];

float3 fwidth_compute(float3 v, int2 index) {
    smem[index.y + 1][index.x + 1] = v;
    // handle boundary
    if (index.x == 0) { smem[index.y + 1][index.x + 0] = v; }
    if (index.x == 31) { smem[index.y + 1][index.x + 2] = v; }
    if (index.y == 0) { smem[index.y + 0][index.x + 1] = v; }
    if (index.y == 3) { smem[index.y + 2][index.x + 1] = v; }
    GroupMemoryBarrierWithGroupSync();
    float3 delta_x = abs(smem[index.y + 1][index.x + 2] - smem[index.y + 1][index.x + 0]);
    float3 delta_y = abs(smem[index.y + 2][index.x + 1] - smem[index.y + 0][index.x + 1]);
    return delta_x + delta_y;
}

float3 add_wireframe(float3 input_color, float3 barys, int2 groupID) {
    const float3 deltas = fwidth_compute(barys, groupID);
    const float3 smoothing = deltas * wireframe_smoothing;
    const float3 thickness = deltas * wireframe_thickness;
    barys = smoothstep(thickness, thickness + smoothing, barys);
    float minBary = min(barys.x, min(barys.y, barys.z));
    return lerp(wireframe_color, input_color, minBary);
}

float3 inspector_pt(
    int2 pixel, int2 resolution,
    inout RandomSamplerState RNG,
    int2 groupID
) {
    // cast primary ray
    // const float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    const float2 jitter = 0;
    Ray ray = generateRay(pixel + jitter, resolution, false, fetchActiveCamera());
    
    PrimaryPayload payload = PrimaryRayQuery(ray);
    bool hit = HasHit(payload.hit);
    
    // Show
    float3 output = float3(1, 0, 1);
    if (hit) {
        float2 texcoord = payload.hit.texcoord;
        Optional<MaterialData> mat = fetchMaterialData(payload.hit);
        switch (inspector_mode) {
        case InspectorMode::ALBEDO: {
            if (mat == none) return float3(1, 0, 1);
            output = materials::albedo(mat.value, texcoord);
            float3 visual_emission = materials::emission(mat.value);
            if (any(visual_emission > 0)) {
                visual_emission /= maxComponent(visual_emission);
                output += visual_emission;
            }
            break;
        }
        case InspectorMode::EMISSION: {
            if (mat == none) return float3(1, 0, 1);
            output = materials::emission(mat.value); break;
        }
        case InspectorMode::ADDITIONAL_TEXTURE: {
            if (mat == none) return float3(1, 0, 1);
            if (mat.value.ext1_tex == -1) return float3(1, 0, 0);
            output = SampleTexture2D(mat.value.ext1_tex, texcoord, true).rgb; break;
        }
        case InspectorMode::FACEFORWARD: output = IsFaceForward(payload.hit) ? float3(0, 1, 0) : float3(1, 0, 0); break;
        case InspectorMode::BARYCENTRIC: output = float3(payload.hit.texcoord, 0); break;
        case InspectorMode::NORMAL: output = payload.hit.shadingNormal * 0.5 + 0.5; break;
        case InspectorMode::POSITION: output = payload.hit.position; break;
        }
    }
    return add_wireframe(output, float3(payload.hit.barycentric, 
        1 - payload.hit.barycentric.x - payload.hit.barycentric.y), groupID);
}

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(1024, 1024);
    const int2 pixel = int2(dtid.xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;

    RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);
    output[pixel] = float4(inspector_pt(pixel, resolution, RNG, gtid.xy), 1);
}