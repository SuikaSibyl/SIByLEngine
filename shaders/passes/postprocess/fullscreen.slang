struct AssembledVertex { int vertexId : SV_VertexId; };
struct VertexStageOutput { float4 sv_position : SV_Position; };

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const uint id = assembledVertex.vertexId;
    const float2 uv = float2((id << 1) & 2, id & 2);
    
    VertexStageOutput output;
    output.sv_position = float4(uv * float2(2) + float2(-1), 1, 1);
    return output;
}