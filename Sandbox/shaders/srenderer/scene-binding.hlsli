#ifndef _SRENDERER_SCENE_BINDING_
#define _SRENDERER_SCENE_BINDING_

#include "common/camera.hlsli"
#include "common/raycast.hlsli"
#include "lights/lightbvh.hlsli"
#include "spt.hlsli"
#include "spt-diff.hlsli"

/**********************************************************************
****                   Common Scene Data Structures                ****
**********************************************************************/

ByteAddressBuffer GPUScene_position;
ByteAddressBuffer GPUScene_vertex;
RWStructuredBuffer<uint32_t> GPUScene_index;
RWStructuredBuffer<GeometryData> GPUScene_geometry;
RWStructuredBuffer<CameraData> GPUScene_camera;
RWStructuredBuffer<MaterialData> GPUScene_material;
RWStructuredBuffer<LightPacket> GPUScene_light;
RaytracingAccelerationStructure GPUScene_tlas;
RWStructuredBuffer<LightBVHNode> GPUScene_light_bvh;
RWStructuredBuffer<uint32_t> GPUScene_light_trail;
RWStructuredBuffer<SceneDescription> GPUScene_description;
RWStructuredBuffer<MediumPacket> GPUScene_medium;
RWStructuredBuffer<float> GPUScene_grid_storage;
Sampler2D GPUScene_textures[];
Sampler2D LUT_blackbody_spectrum;

float3 black_body_spectrum(float temperature) {
    float alpha = clamp((temperature - 10) / (20000 - 10), 0, 1);
    return LUT_blackbody_spectrum.Sample(float2(alpha, 0.5)).xyz;
}

float3 fetchVertexPosition(int vertexIndex) { return GPUScene_position.Load<float3>(vertexIndex * 12); }
float3 fetchVertexNormal(int vertexIndex) { return GPUScene_vertex.Load<float3>(vertexIndex * 32 + 0); }
float3 fetchVertexTangent(int vertexIndex) { return GPUScene_vertex.Load<float3>(vertexIndex * 32 + 12); }
float2 fetchVertexTexCoord(int vertexIndex) { return GPUScene_vertex.Load<float2>(vertexIndex * 32 + 24); }
uint3 fetchTriangleIndices(in const GeometryData geoInfo, int triangleIndex) { return uint3(
    GPUScene_index[geoInfo.indexOffset + triangleIndex * 3 + 0],
    GPUScene_index[geoInfo.indexOffset + triangleIndex * 3 + 1],
    GPUScene_index[geoInfo.indexOffset + triangleIndex * 3 + 2]);
}

int fetchActiveCameraID() { return GPUScene_description[0].active_camera_index; }
CameraData fetchActiveCamera() { return GPUScene_camera[fetchActiveCameraID()]; }

MediumPacket fetchMedium(int mediumID) {
    if(mediumID < 0) return MediumPacket();
    return GPUScene_medium[mediumID];
}

MediumPacket fetchCameraMedium(CameraData camera) {
    return fetchMedium(camera.mediumID);
}

int fetchCameraMediumIdx(CameraData camera) { return camera.mediumID; }

MediumPacket updateMedium(GeometryData geometry, GeometryHit hit, MediumPacket medium) {
    if (geometry.mediumIDInterior != geometry.mediumIDExterior) {
        // at medium transition, update medium
        if (IsFaceForward(hit)) return fetchMedium(geometry.mediumIDInterior);
        else return fetchMedium(geometry.mediumIDExterior);
    }
    return medium;
}

int updateMedium(GeometryData geometry, GeometryHit hit, int mediumID) {
    if (geometry.mediumIDInterior != geometry.mediumIDExterior) {
        // at medium transition, update medium
        if (IsFaceForward(hit)) return geometry.mediumIDInterior;
        else return geometry.mediumIDExterior;
    }
    return mediumID;
}

bounds3 fetchAllLightBounds() {
    bounds3 allb;
    allb.pMin = GPUScene_description[0].light_bound_min;
    allb.pMax = GPUScene_description[0].light_bound_max;
    return allb;
}

float4 sampleTexture(int textureID, float2 uv) {
    if (textureID >= 0)
        return GPUScene_textures[textureID].Sample(uv);
    else return float4(1, 1, 1, 1);
}

[BackwardDifferentiable]
float4 SampleTexture2D(int textureID, no_diff float2 uv, bool is_differentiable = false) {
    if (!is_differentiable) {
        if (textureID >= 0)
            return no_diff GPUScene_textures[textureID].Sample(uv);
        else return float4(1, 1, 1, 1);
    } else {
        if (textureID >= 0) {
            DifferentiableParameter param = GPUScene_param_packets[textureID];
            float4 texel = float4(0, 0, 0, 1);
            if (param.dim_2 > 0) texel.x = software_bilinear_interpolation_load(uv, 0, param);
            if (param.dim_2 > 1) texel.y = software_bilinear_interpolation_load(uv, 1, param);
            if (param.dim_2 > 2) texel.z = software_bilinear_interpolation_load(uv, 2, param);
            return texel;
        }
        else return float4(1, 1, 1, 1);
    }
}

[BackwardDifferentiable]
float4 SampleTexture2D_w_aux(int textureID, no_diff float2 uv, no_diff float4 aux, bool is_differentiable = false) {
    if (!is_differentiable) {
        if (textureID >= 0)
            return no_diff GPUScene_textures[textureID].Sample(uv);
        else return float4(1, 1, 1, 1);
    } else {
        if (textureID >= 0) {
            DifferentiableParameter param = GPUScene_param_packets[textureID];
            float4 texel = float4(0, 0, 0, 1);
            if (param.dim_2 > 0) texel.x = software_bilinear_interpolation_load_w_aux_float2(uv, 0, param, aux.x);
            if (param.dim_2 > 1) texel.y = software_bilinear_interpolation_load_w_aux_float2(uv, 1, param, aux.y);
            if (param.dim_2 > 2) texel.z = software_bilinear_interpolation_load_w_aux_float2(uv, 2, param, aux.z);
            if (param.dim_2 > 3) texel.w = software_bilinear_interpolation_load_w_aux_float2(uv, 3, param, aux.w);
            return texel;
        }
        else return float4(1, 1, 1, 1);
    }
}

float4 SampleTexture2DGradient(int textureID, no_diff float2 uv, bool is_differentiable = false) {
    if (!is_differentiable) {
        return float4(0, 0, 0, 0);
    } else {
        if (textureID >= 0) {
            DifferentiableParameter param = GPUScene_param_packets[textureID];
            float4 texel = float4(0, 0, 0, 0);
            if (param.dim_2 > 0) texel.x = software_nearest_load_grad(uv, 0, param);
            if (param.dim_2 > 1) texel.y = software_nearest_load_grad(uv, 1, param);
            if (param.dim_2 > 2) texel.z = software_nearest_load_grad(uv, 2, param);
            return texel;
        }
        else return float4(0, 0, 0, 0);
    }
}

float4 SampleTexture2DAux(int textureID, no_diff float2 uv, int aux_idx, bool is_differentiable = false) {
    if (!is_differentiable) {
        return float4(0, 0, 0, 0);
    } else {
        if (textureID >= 0) {
            DifferentiableParameter param = GPUScene_param_packets[textureID];
            float4 texel = float4(0, 0, 0, 0);
            if (param.dim_2 > 0) texel.x = software_nearest_load_aux(uv, 0, aux_idx, param);
            if (param.dim_2 > 1) texel.y = software_nearest_load_aux(uv, 1, aux_idx, param);
            if (param.dim_2 > 2) texel.z = software_nearest_load_aux(uv, 2, aux_idx, param);
            if (param.dim_2 > 3) texel.w = software_nearest_load_aux(uv, 3, aux_idx, param);
            return texel;
        }
        else return float4(0, 0, 0, 0);
    }
}

/**
 * Fetches the geometry hit info for a given geometry ID and primitive ID.
 * @param geometryID The geometry ID.
 * @param bary The barycentric coordinates.
 * @param primitiveID The primitive ID.
 * @return The geometry hit info.
 */
GeometryHit fetchTrimeshGeometryHit(
    int geometryID,
    in_ref(float3) bary,
    int primitiveID
) {
    const GeometryData geometry = GPUScene_geometry[geometryID];
    
    GeometryHit hit;
    hit.barycentric = bary.yz;
    hit.primitiveID = primitiveID;
    hit.geometryID = geometryID;
    
    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

    float3 vertexPositions[3];
    vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));
    
    const float4x4 o2w = ObjectToWorld(geometry);
    const float3 positionOS = interpolate(vertexPositions, bary);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    hit.position = positionWS;

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    float2 uv = interpolate(vertexUVs, bary);
    hit.texcoord = uv;
    if (any(isnan(hit.texcoord)))
        hit.texcoord = float2(.5);
    
    float3 objectSpaceFlatNormal = normalize(cross(
        vertexPositions[1] - vertexPositions[0],
        vertexPositions[2] - vertexPositions[0]));

    float4x4 o2wn = ObjectToWorldNormal(geometry);
    const float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
    hit.geometryNormal = flatNormal;

    float3 normals[3];
    normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    float3 vertexNormalOS = interpolate(normals, bary);
    float3 gvertexNormalWS = normalize(mul(float4(vertexNormalOS, 0.0), o2wn).xyz);
    hit.shadingNormal = gvertexNormalWS;
    if (all(hit.shadingNormal == 0) || any(isnan(hit.shadingNormal)))
        hit.shadingNormal = hit.geometryNormal;

    float3 tangents[3];
    tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
    tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
    tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
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

    SetHit(hit, true);

    return hit;
}

GeometryHit fetchTrimeshGeometryHit(
    in const int geometryID,
    in const float3 bary,
    in const int primitiveID,
    in const Ray ray
) {
    GeometryHit hit = fetchTrimeshGeometryHit(geometryID, bary, primitiveID);
    const float frontFace = (dot(hit.geometryNormal, ray.direction) < 0) ? 1 : -1;
    SetFaceForward(hit, frontFace == 1);
    hit.shadingNormal *= frontFace;
    hit.geometryNormal *= frontFace;
    return hit;
}

Optional<MaterialData> fetchMaterialData(GeometryHit hit) {
    const int matID = GPUScene_geometry[hit.geometryID].materialID;
    if (matID == -1) return none;
    return GPUScene_material[matID];
}

#endif // _SRENDERER_SCENE_BINDING_