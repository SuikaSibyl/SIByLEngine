#include "../srenderer/scene-binding.hlsli"
#include "structures.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    uint camera_idx;
};

StructuredBuffer<Line3DData> u_lines;

struct AssembledVertex {
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceID;
};

struct CoarseVertex { float3 color; };
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    float3 quad[30] = {
        // triangle 0
        float3(0, -0.5, 0),
        float3(0, -0.5, 1),
        float3(0, 0.5, 1),
        // triangle 1
        float3(0, -0.5, 0),
        float3(0, 0.5, 1),
        float3(0, 0.5, 0),
        // triangle 2
        float3(0, 0, 0),
        float3(3.061616997868383e-17, 0.5, 0),
        float3(-0.35355339059327373, 0.3535533905932738, 0),
        // triangle 3
        float3(0, 0, 0),
        float3(-0.35355339059327373, 0.3535533905932738, 0),
        float3(-0.5, 6.123233995736766e-17, 0),
        // triangle 4
        float3(0, 0, 0),
        float3(-0.5, 6.123233995736766e-17, 0),
        float3(-0.35355339059327384, -0.35355339059327373, 0),
        // triangle 5
        float3(0, 0, 0),
        float3(-0.35355339059327384, -0.35355339059327373, 0),
        float3(-9.184850993605148e-17, -0.5, 0),
        // triangle 6
        float3(0, 0, 1),
        float3(-9.184850993605148e-17, -0.5, 1),
        float3(0.3535533905932737, -0.35355339059327384, 1),
        // triangle 7
        float3(0, 0, 1),
        float3(0.3535533905932737, -0.35355339059327384, 1),
        float3(0.5, -1.2246467991473532e-16, 1),
        // triangle 8
        float3(0, 0, 1),
        float3(0.5, -1.2246467991473532e-16, 1),
        float3(0.35355339059327384, 0.3535533905932737, 1),
        // triangle 9
        float3(0, 0, 1),
        float3(0.35355339059327384, 0.3535533905932737, 1),
        float3(1.5308084989341916e-16, 0.5, 1)
    };

    const int vertexID = assembledVertex.vertexId;
    const int instanceID = assembledVertex.instanceId;

    const float2 res = float2(resolution);
    const float3 position = quad[vertexID];
    const Line3DData line = u_lines[instanceID];
    const CameraData camera = GPUScene_camera[camera_idx];

    // get vertex positions
    const float4 clip0 = mul(float4(line.position_1, 1.0f), camera.viewProjMat);
    const float4 clip1 = mul(float4(line.position_2, 1.0f), camera.viewProjMat);

    const float2 screen0 = resolution * (0.5 * clip0.xy / clip0.w + 0.5);
    const float2 screen1 = resolution * (0.5 * clip1.xy / clip1.w + 0.5);

    float2 xBasis = normalize(screen1 - screen0);
    float2 yBasis = float2(-xBasis.y, xBasis.x);
    float2 pt0 = screen0 + line.width * (position.x * xBasis + position.y * yBasis);
    float2 pt1 = screen1 + line.width * (position.x * xBasis + position.y * yBasis);
    float2 pt = lerp(pt0, pt1, position.z);
    float4 clip = lerp(clip0, clip1, position.z);

    VertexStageOutput output;
    output.coarseVertex.color = line.color;
    output.sv_position = float4(clip.w * (2.0 * pt / resolution - 1.0), clip.z, clip.w);
    return output;
}

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    out float4 o_color: SV_Target0,
    out uint o_index: SV_Target1) : SV_Target
{
    o_color = float4(coarseVertex.color, 1.0f);
    o_index = 0;
    return;
}