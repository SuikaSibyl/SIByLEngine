#include "../srenderer/scene-binding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint camera_idx;
};

struct BillboardData {
    float3 position;
    int iconIDX;
    float2 size;
    int recordID;
    int padding;
};

struct AssembledVertex {
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceID;
};

struct CoarseVertex {
    float2 uv;
    uint geoID;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

StructuredBuffer<BillboardData> u_billboards;

Texture2D<float4> IconsTextures;
SamplerState IconsSampler;

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    static const float2 vertex_buffer[4] = {
        float2(-0.5f, -0.5f),
        float2(0.5f, -0.5f),
        float2(-0.5f, 0.5f),
        float2(0.5f, 0.5f)
    };

    static const float2 uv_buffer[4] = {
        float2(0.0f, 0.0f),
        float2(1.0f, 0.0f),
        float2(0.0f, 1.0f),
        float2(1.0f, 1.0f)
    };
    
    static const int indices[6] = {
        0, 1, 2,
        2, 1, 3
    };

    const int vertexID = indices[assembledVertex.vertexId];
    const int instanceID = assembledVertex.instanceId;
    const BillboardData billboard = u_billboards[instanceID];
    const CameraData camera = GPUScene_camera[camera_idx];

    const float3 positionWS = billboard.position
        + normalize(camera.cameraU) * vertex_buffer[vertexID].x * billboard.size.x
        + normalize(camera.up) * vertex_buffer[vertexID].y * billboard.size.y;
    
    const float4 positionCS = mul(float4(positionWS, 1.0f), GPUScene_camera[camera_idx].viewProjMat);

    VertexStageOutput output;
    output.coarseVertex.geoID = instanceID;
    output.coarseVertex.uv = uv_buffer[vertexID];
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    out float4 o_color: SV_Target0,
    out uint o_index: SV_Target1) : SV_Target
{
    const int instanceID = coarseVertex.geoID;
    const BillboardData billboard = u_billboards[instanceID];

    float2 uv = coarseVertex.uv;
    float2 target_uv = (1 - uv) / 10;
    float4 color = IconsTextures.Sample(IconsSampler, target_uv);
    if(color.a < 0.05f)
        discard;

    o_color = color;
    o_index = billboard.recordID;
    return;
}