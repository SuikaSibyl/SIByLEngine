#include "reverse-wos.hlsli"
#include "core/distributions.hlsli"

StructuredBuffer<GreenSphere2D> sb_spheres;

// [[vk::push_constant]]
// cbuffer PushConstants {
//     int2 resolution;
// };

struct AssembledVertex { 
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceId; };
struct CoarseVertex { float2 coord; uint geoID; };
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(AssembledVertex v_in) {
    const int vertexID = v_in.vertexId;
    const int instanceID = v_in.instanceId;
    static const float2 vertex_buffer[10] = {
        float2(+1.067f, +0.000f),
        float2(+0.863f, -0.627f),
        float2(+0.330f, -1.015f),
        float2(-0.330f, -1.015f),
        float2(-0.863f, -0.627f),
        float2(-1.067f, +0.000f),
        float2(-0.863f, +0.627f),
        float2(-0.330f, +1.015f),
        float2(+0.330f, +1.015f),
        float2(+0.863f, +0.627f),
    };

    float2 position = { 0, 0 };
    const int subvertexID = vertexID % 3;
    const int triangleID = vertexID / 3;
    if (subvertexID == 2) {
        position = { 0, 0 };
    } else {
        int listID = triangleID + subvertexID;
        if (listID >= 10) listID -= 10;
        position = vertex_buffer[listID];
    }

    const GreenSphere2D sphere = sb_spheres[instanceID];
    // coord & screen in the range [-1, 1]
    const float2 coord = position * sphere.radius + sphere.center;
    const float2 screen = coord;
    const float4 positionCS = float4(screen, 1, 1);
    
    VertexStageOutput output;
    output.coarseVertex.geoID = instanceID;
    output.coarseVertex.coord = coord;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
void fragmentMain(
    in CoarseVertex vert: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    float4 svPos: SV_POSITION,
    in float3 bary: SV_Barycentrics,
    out float4 o_color: SV_Target0) : SV_Target
{
    const float2 p = vert.coord;
    const int instance = vert.geoID;
    const GreenSphere2D sphere = sb_spheres[instance];
    const float l = distance(p, sphere.center);
    
    if (l < sphere.radius) {
        HarmonicGreensFnBall2 green;
        green.update_ball(sphere.center, sphere.radius);
        const float g = green.evaluate(l);
        if(isinf(g)) {
            discard;
        }
        o_color = float4(g * sphere.source * sphere.weight, sphere.weight);
    }
    else {
        discard;
    }
}
