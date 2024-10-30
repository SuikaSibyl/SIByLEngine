#ifndef _SRENDERER_SPT_HEADER_
#define _SRENDERER_SPT_HEADER_

#include "common/geometry.hlsli"
#include "common/random.hlsli"
#include "common/raycast.hlsli"

/**********************************************************************
****                    Common Payloads Structures                 ****
**********************************************************************/
/** Geometry hit infomation */
struct GeometryHit {
    float3 position;
    uint geometryID;
    float3 shadingNormal;
    uint primitiveID;
    float3 geometryNormal;
    uint flags;
    float2 barycentric;
    float2 texcoord;
    float4 tangent;
    float3 padding;
    float lambda;
};

/** Primary Payload Struct */
struct PrimaryPayload {
    GeometryHit hit;
    RandomSamplerState RND;
};

struct RawPayload {
    int primitiveID;
    int geometryID;
    float2 barycentric;
    bool hasHit;
};

// geometry info
struct GeometryData {
    uint vertexOffset;
    uint indexOffset;
    int materialID;
    uint indexSize;
    int16_t mediumIDExterior = -1;
    int16_t mediumIDInterior = -1;
    int lightID;
    uint primitiveType;
    float oddNegativeScaling;
    float4 transform[3];
    float4 transformInverse[3];
};

struct MaterialData {
    int bxdf_type;
    int bitfield;
    int16_t albedo_tex;
    int16_t normal_tex;
    int16_t ext1_tex;
    int16_t ext2_tex;
    float4 floatvec_0;
    float4 floatvec_1;
    float4 floatvec_2;
};

enum LightType {
    DIRECTIONAL,
    POINT,
    SPOT,
    SPHERE,
    RECTANGLE,
    MESH_PRIMITIVE,
    ENVIRONMENT,
    VPL,
    MAX_ENUM,
};

struct LightPacket {
    LightType light_type;
    int bitfield;
    uint uintscalar_0;
    uint uintscalar_1;
    float4 floatvec_0;
    float4 floatvec_1;
    float4 floatvec_2;
};

struct MediumPacket {
    enum MediumType {
        HOMOGENEOUS,
        GRID_MEDIUM,
        RGBGRID_MEDIUM,
        MAX_ENUM,
    };

    float3 sigma_a = float3(0, 0, 0);
    uint32_t bitfield = 0x80000000;
    float3 sigma_s = float3(0, 0, 0);
    float scale = 1.0f;
    float3 aniso = float3(0, 0, 0);
    float temperatureScale = 0.0f;
    float3 bound_min;
    float LeScale;
    float3 bound_max;
    MediumType medium_type = MediumType::HOMOGENEOUS;
    int3 density_xyz;
    int density_offset;
    int3 temperature_xyz;
    int temperature_offset;
    int3 le_xyz;
    int le_offset;
    int3 majorant_xyz;
    int majorant_offset;
    float3 temperature_bound_min;
    float temperature_start;
    float3 temperature_bound_max;
    float pdadding;
    float4 transform[3];
    float4 transformInverse[3];
    
    float3 get_g() { return clamp(aniso, -1 + 1e-3, 1 - 1e-3); }
    float3 get_sigma_a() { return sigma_a * scale; }
    float3 get_sigma_s() { return sigma_s * scale; }
    float3 get_sigma_t() { return sigma_a * scale + sigma_s * scale; }
    MediumType get_medium_type() { return medium_type; }
    
    float4x4 medium_to_world() { return transpose(float4x4(transformInverse[0], transformInverse[1], transformInverse[2], float4(0, 0, 0, 1))); }
    float4x4 world_to_medium() { return transpose(float4x4(transform[0], transform[1], transform[2], float4(0, 0, 0, 1))); }
};

bool IsValidMedium(MediumPacket medium) {
    return medium.bitfield != 0x80000000;
}

struct SceneDescription {
    float3 light_bound_min;
    int max_light_count;
    float3 light_bound_max;
    int active_camera_index;
};

float4x4 ObjectToWorld(in const GeometryData geometry) {
    return transpose(float4x4(geometry.transform[0], 
        geometry.transform[1], geometry.transform[2], float4(0, 0, 0, 1)));
}

float4x4 WorldToObject(in const GeometryData geometry) {
    return transpose(float4x4(geometry.transformInverse[0], 
        geometry.transformInverse[1], geometry.transformInverse[2], float4(0, 0, 0, 1)));
}

float4x4 ObjectToWorldNormal(in const GeometryData geometry) {
    return float4x4(geometry.transformInverse[0], 
        geometry.transformInverse[1], geometry.transformInverse[2], float4(0, 0, 0, 1));
}

static const uint GeometryHitFlag_HitShift = 0x00;
static const uint GeometryHitFlag_HitMask = 0x01;

static const uint GeometryFlag_FaceForwardShift = 0x01;
static const uint GeometryFlag_FaceForwardMask = 0x01;

static const uint GeometryTypeFlag_HitShift = 0x02;
static const uint GeometryTypeFlag_HitMask = 0xff;

bool HasHit(in const GeometryHit hit) { return ((hit.flags >> GeometryHitFlag_HitShift) & GeometryHitFlag_HitMask) != 0; }
void SetHit(inout GeometryHit hit, bool value) {
    // if not hit, reset the geometryID and primitiveID
    if (!value) { hit.geometryID = uint(-1); hit.primitiveID = uint(-1); }
    hit.flags = (hit.flags & ~(GeometryHitFlag_HitMask << GeometryHitFlag_HitShift)) 
            | (value ? 1 : 0) << GeometryHitFlag_HitShift;
}

bool IsFaceForward(in const GeometryHit hit) { return ((hit.flags >> GeometryFlag_FaceForwardShift) & GeometryFlag_FaceForwardShift) != 0; }
void SetFaceForward(inout GeometryHit hit, bool value) {
    hit.flags = (hit.flags & ~(GeometryFlag_FaceForwardMask << GeometryFlag_FaceForwardShift)) 
            | (value ? 1 : 0) << GeometryFlag_FaceForwardShift;
}

float3 offsetPositionAlongNormal(float3 worldPosition, float3 normal) {
    // Convert the normal to an integer offset.
    const float int_scale = 256.0f;
    const int3 of_i = int3(int_scale * normal);
    // Offset each component of worldPosition using its binary representation.
    // Handle the sign bits correctly.
    const float3 p_i = float3( //
        asfloat(asint(worldPosition.x) + ((worldPosition.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(worldPosition.y) + ((worldPosition.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(worldPosition.z) + ((worldPosition.z < 0) ? -of_i.z : of_i.z)));
    // Use a floating-point offset instead for points near (0,0,0), the origin.
    const float origin = 1.0f / 32.0f;
    const float floatScale = 1.0f / 65536.0f;
    return float3( //
        abs(worldPosition.x) < origin ? worldPosition.x + floatScale * normal.x : p_i.x,
        abs(worldPosition.y) < origin ? worldPosition.y + floatScale * normal.y : p_i.y,
        abs(worldPosition.z) < origin ? worldPosition.z + floatScale * normal.z : p_i.z);
}

Ray SpawnRay(
    in_ref(GeometryHit) isect,
    in_ref(float3) dir) {
    const float3 offsetDir = faceforward(isect.geometryNormal, 
        -dir, isect.geometryNormal);
    float3 offsetedPosition = offsetPositionAlongNormal(isect.position, offsetDir);
    if (any(abs(offsetedPosition) >= 8)) { offsetedPosition += 1e-4 * offsetDir; } 
    Ray ray;
    ray.origin = offsetedPosition;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = k_inf;
    return ray;
}

Ray SpawnRay(float3 pos, float3 dir) {
    Ray ray;
    ray.origin = pos;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = 1e6;
    return ray;
} 

Ray SpawnVisibilityRay(
    in_ref(GeometryHit) isect,
    in_ref(float3) position) {
    float3 dir = position - isect.position;
    float distance = length(dir);
    Ray visiblityRay = SpawnRay(isect, dir / distance);
    visiblityRay.tMax = distance - min(0.001, distance * 0.002);
    return visiblityRay;
}

Ray SpawnVisibilityRay(
    in_ref(float3) origin,
    in_ref(float3) target) {
    float3 dir = target - origin;
    float distance = length(dir);
    Ray visiblityRay = SpawnRay(origin, dir / distance);
    visiblityRay.tMax = distance - min(0.01, distance * 0.02);
    return visiblityRay;
}

#endif // _SRENDERER_SPT_HEADER_