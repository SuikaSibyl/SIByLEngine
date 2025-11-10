#ifndef _SRENDERER_SCENE_BINDINGS_
#define _SRENDERER_SCENE_BINDINGS_
#define SRENDERER_SCENE_TLAS_COUNT 1

#include "common/ray.slang"
#include "common/math.slang"
#include "srenderer/spt-definition.slang"
#include "srenderer/lights/lightbvh.slang"

struct IndexBuffer { ConstBufferPointer<uint> ref; };
struct PositionBuffer { ConstBufferPointer<float> ref; };
struct VertexBuffer { ConstBufferPointer<float> ref; };
struct LightBVHNodeBuffer { ConstBufferPointer<LightBVHNode> ref; };
struct LightBVHTrailBuffer { ConstBufferPointer<uint32_t> ref; };

RWStructuredBuffer<IndexBuffer>         se_index_buffers;
RWStructuredBuffer<PositionBuffer>      se_position_buffers;
RWStructuredBuffer<VertexBuffer>        se_vertex_buffers;
RWStructuredBuffer<CameraData>          se_camera_buffers;
RWStructuredBuffer<GeometryData>        se_geometry_buffers;
RWStructuredBuffer<MaterialData>        se_material_buffers;
RWStructuredBuffer<LightData>           se_light_buffer;
RWStructuredBuffer<SceneData>           se_scene_buffer;
RWStructuredBuffer<LightBVHNodeBuffer>  se_lightbvh_nodes;
RWStructuredBuffer<LightBVHTrailBuffer> se_lightbvh_trails;
RWStructuredBuffer<MediumData>          se_medium_buffer;
RWStructuredBuffer<float>               se_medium_grid_buffer;
Sampler2D                               se_textures[];
RaytracingAccelerationStructure         se_scene_tlas[SRENDERER_SCENE_TLAS_COUNT];

float3 scene_read_position(int meshID, int positionID) { 
    return float3(
        se_position_buffers[meshID].ref[positionID * 3 + 0],
        se_position_buffers[meshID].ref[positionID * 3 + 1],
        se_position_buffers[meshID].ref[positionID * 3 + 2]
    );
}
// Read a vertex index from the index buffer of a mesh.
int scene_read_index(int meshID, int indexID) { return se_index_buffers[meshID].ref[indexID]; }
// Read a camera info from the camera buffer.
CameraData scene_read_camera(int cameraID) { return se_camera_buffers[cameraID]; }
// Read a geometry info from the geometry buffer.
GeometryData scene_read_geometry(int geometryID) { return se_geometry_buffers[geometryID]; }
// Read a material info from the material buffer.
MaterialData scene_read_material(int materialID) { return se_material_buffers[materialID]; }
// Read a material info from the material buffer.
Optional<MaterialData> scene_read_material(GeometryData geometry) {
    if (geometry.materialID < 0) return none;
    return se_material_buffers[geometry.materialID];
}
// Read a camera info from the camera buffer.
MediumData scene_read_medium(int mediumID) { return se_medium_buffer[mediumID]; }
// Read a LightBVHNode from the light BVH node buffer.
LightBVHNode scene_read_lbvh_node(int lbvhID, int nodeID) { return se_lightbvh_nodes[lbvhID].ref[nodeID]; }
// Read a LightBVHTrail from the light BVH trail buffer.
uint32_t scene_read_lbvh_trail(int lbvhID, int trailID) { return se_lightbvh_trails[lbvhID].ref[trailID]; }

// Read a vertex indices from the index buffer of a mesh.
int3 scene_read_triangle_indices(const GeometryData data, int triangleIndex) {
    return int3(
        scene_read_index(data.meshID, data.indexOffset + triangleIndex * 3 + 0),
        scene_read_index(data.meshID, data.indexOffset + triangleIndex * 3 + 1),
        scene_read_index(data.meshID, data.indexOffset + triangleIndex * 3 + 2)
    );
}

float3 scene_read_vertex_normal(int meshID, int vertexIndex) {
    return float3(
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 0],
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 1],
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 2]
    );
}

float3 scene_read_vertex_tangent(int meshID, int vertexIndex) {
    return float3(
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 3],
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 4],
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 5]
    );
}

float2 scene_read_vertex_texcoord(int meshID, int vertexIndex) {
    return float2(
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 6],
        se_vertex_buffers[meshID].ref[vertexIndex * 8 + 7]
    );
}

/**
 * Fetches the geometry hit info for a given geometry ID and primitive ID.
 * @param geometryID The geometry ID.
 * @param bary The barycentric coordinates.
 * @param primitiveID The primitive ID.
 * @return The geometry hit info.
 */
GeometryHit fetch_trimesh_geometry_hit(
    int geometryID,
    float3 bary,
    int primitiveID
) {
    const GeometryData geometry = scene_read_geometry(geometryID);

    GeometryHit hit;
    hit.barycentric = bary.yz;
    hit.primitiveID = primitiveID;
    hit.geometryID = geometryID;

    const int3 index = scene_read_triangle_indices(geometry, primitiveID);

    float3 vertexPositions[3];
    vertexPositions[0] = scene_read_position(geometry.meshID, index[0] + geometry.vertexOffset);
    vertexPositions[1] = scene_read_position(geometry.meshID, index[1] + geometry.vertexOffset);
    vertexPositions[2] = scene_read_position(geometry.meshID, index[2] + geometry.vertexOffset);

    const float4x4 o2w = geometry.object_to_world();
    const float3 positionOS = interpolate(vertexPositions, bary);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    hit.position = positionWS;

    float2 vertexUVs[3];
    vertexUVs[0] = scene_read_vertex_texcoord(geometry.meshID, index[0] + geometry.vertexOffset);
    vertexUVs[1] = scene_read_vertex_texcoord(geometry.meshID, index[1] + geometry.vertexOffset);
    vertexUVs[2] = scene_read_vertex_texcoord(geometry.meshID, index[2] + geometry.vertexOffset);
    float2 uv = interpolate(vertexUVs, bary);
    hit.texcoord = uv;
    if (any(isnan(hit.texcoord)))
        hit.texcoord = float2(.5);

    float3 objectSpaceFlatNormal = normalize(cross(
        vertexPositions[1] - vertexPositions[0],
        vertexPositions[2] - vertexPositions[0]));

    float4x4 o2wn = geometry.object_to_world_normal();
    const float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
    hit.geometryNormal = flatNormal;

    float3 normals[3];
    normals[0] = scene_read_vertex_normal(geometry.meshID, index[0] + geometry.vertexOffset);
    normals[1] = scene_read_vertex_normal(geometry.meshID, index[1] + geometry.vertexOffset);
    normals[2] = scene_read_vertex_normal(geometry.meshID, index[2] + geometry.vertexOffset);
    float3 vertexNormalOS = interpolate(normals, bary);
    float3 gvertexNormalWS = normalize(mul(float4(vertexNormalOS, 0.0), o2wn).xyz);
    hit.shadingNormal = gvertexNormalWS;
    if (all(hit.shadingNormal == 0) || any(isnan(hit.shadingNormal)))
        hit.shadingNormal = hit.geometryNormal;

    float3 tangents[3];
    tangents[0] = scene_read_vertex_tangent(geometry.meshID, index[0] + geometry.vertexOffset);
    tangents[1] = scene_read_vertex_tangent(geometry.meshID, index[1] + geometry.vertexOffset);
    tangents[2] = scene_read_vertex_tangent(geometry.meshID, index[2] + geometry.vertexOffset);
    float3 tangentOS = interpolate(tangents, bary);
    float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);
    hit.tangent = tangentWS;

    // compute lambda for ray cone based lod sampling
    const float2 uv_10 = vertexUVs[1] - vertexUVs[0];
    const float2 uv_20 = vertexUVs[2] - vertexUVs[0];
    const float t_a = abs(uv_10.x * uv_20.y - uv_20.x * uv_10.y);
    const float3 edge_10 = mul(float4(vertexPositions[1] - vertexPositions[0], 0.0), o2w).xyz;
    const float3 edge_20 = mul(float4(vertexPositions[2] - vertexPositions[0], 0.0), o2w).xyz;
    const float p_a = length(cross(edge_10, edge_20));
    hit.lambda = t_a / p_a;
    hit.set_hit(true);
    return hit;
}

GeometryHit fetch_trimesh_geometry_hit(
    int geometryID, float3 bary,
    int primitiveID, Ray ray
) {
    GeometryHit hit = fetch_trimesh_geometry_hit(geometryID, bary, primitiveID);
    const float frontFace = (dot(hit.geometryNormal, ray.direction) < 0) ? 1 : -1;
    hit.set_face_forward(frontFace == 1);
    hit.shadingNormal *= frontFace;
    hit.geometryNormal *= frontFace;
    return hit;
}

LightData scene_read_light(int lightID) { return se_light_buffer[lightID]; }

SceneData scene_read_scene_info() { return se_scene_buffer[0]; }

Sampler2D scene_read_texture(int textureID) { return se_textures[textureID]; }

#endif // _SRENDERER_SCENE_BINDINGS_