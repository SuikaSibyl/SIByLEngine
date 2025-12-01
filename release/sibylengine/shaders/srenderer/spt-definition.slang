#ifndef _SRENDERER_SPT_DEFINITION_HLSLI_
#define _SRENDERER_SPT_DEFINITION_HLSLI_

#include "common/random.slang"
#include "common/ray.slang"
#include "common/geometry.slang"
#include "common/math.slang"

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

    float3 albedo() { return float3(floatvec_0.x, floatvec_0.y, floatvec_0.z); }
    float3 emission() { return float3(floatvec_1.x, floatvec_1.y, floatvec_1.z); }
};

struct GeometryData {
    uint vertexOffset;
    uint indexOffset;
    int materialID;
    uint indexSize;
    int16_t mediumIDExterior = -1;
    int16_t mediumIDInterior = -1;
    int16_t primitiveType;
    int16_t meshID;
    int lightID;
    float oddNegativeScaling;
    float4 transform[3];
    float4 transformInverse[3];

    float4x4 object_to_world() { return transpose(float4x4(transform[0], transform[1], transform[2], float4(0, 0, 0, 1))); }

    float4x4 object_to_world_normal() { return float4x4(transformInverse[0], transformInverse[1], transformInverse[2], float4(0, 0, 0, 1)); }

    float4x4 world_to_object() { return transpose(float4x4(transformInverse[0], transformInverse[1], transformInverse[2], float4(0, 0, 0, 1))); }
};

struct CameraData {
    float4x4 viewMat;             ///< Camera view matrix.
    float4x4 invViewMat;          ///< Camera inverse view matrix.
    float4x4 projMat;             ///< Camera projection matrix.
    float4x4 invProjMat;          ///< Camera inverse projection matrix.
    float4x4 viewProjMat;         ///< Camera view-projection matrix.
    float4x4 invViewProj;         ///< Camera inverse view-projection matrix.
    float4x4 viewProjMatNoJitter; ///< Camera view-projection matrix. No jittering is applied!
    float4x4 projMatNoJitter;     ///< Camera projection matrix. No jittering is applied!

    float3 posW;       ///< Camera world-space position.
    float focalLength; ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
    float3 prevPosW;   ///< Camera world-space position associated to previous frame.
    float rectArea;    ///< Recrtangles area A at z=1 plane.
    float3 up;         ///< Camera world-space up vector.
    float aspectRatio; ///< Camera film frame aspect ratio, equal to frameWidth / frameHeight
    float3 target;     ///< Camera target point in world-space.
    float nearZ;       ///< Camera near plane.
    float3 cameraU;    ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.
    float farZ;        ///< Camera far plane.
    float3 cameraV;    ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float jitterX;     ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float3 cameraW;    ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float jitterY;     ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    float frameHeight;    ///< Camera film frame height in mm. 24 is the height of a 35mm film
    float frameWidth;     ///< Camera film frame width in mm.  42 2/3 is the width assuming 24mm height and a 16:9 aspect ratio
    float focalDistance;  ///< Camera focal distance in scene units.
    float apertureRadius; ///< Camera aperture radius in scene units.
    float shutterSpeed;   ///< Camera shutter speed in seconds.
    float ISOSpeed;       ///< Camera film speed based on ISO standards.
    int mediumID;
    float _padding2;

    float2 clipToWindowScale;
    float2 clipToWindowBias;

    int2 get_viewport_size() { return int2(clipToWindowScale * float2(2, -2)); }
    float2 get_inv_viewport_size() { return float2(1.f) / float2(clipToWindowScale * float2(2, -2)); }
    float4x4 get_view_proj_mat() { return viewProjMat; }
    float2 get_screen_pixel_pos(float3 posW) {
        const float4 clip = mul(float4(posW, 1.0f), viewProjMat);
        return get_viewport_size() * (0.5 * clip.xy / clip.w + 0.5);
    }

    Ray spawn(float2 ndc) {
        Ray ray;
        ray.origin = posW;
        // Compute the normalized ray direction assuming a pinhole camera.
        ray.direction = normalize(ndc.x * cameraU + ndc.y * cameraV  + cameraW);
        float invCos = 1.f / dot(normalize(cameraW), ray.direction);
        ray.tMin = nearZ * invCos * 2;
        ray.tMax = farZ * invCos * 2;
        return ray;
    }
    
    /** Compute the ray corresponding to a given pixel */
    Ray spawn(float2 pixel, uint2 frameDim, bool applyJitter) {
        // Compute the normalized ray direction assuming a pinhole camera.
        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        float2 p = (float2(pixel) + float2(0.5f, 0.5f)) / float2(frameDim);
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        if (applyJitter)
            p += float2(-jitterX, jitterY);
        // Compute ndc of the corresponding pixel.
        float2 ndc = float2(2.0f, -2.0f) * p + float2(-1.0f, 1.0f);
        return spawn(ndc);
    }
};

static const uint GeometryHitFlag_HitShift = 0x00;
static const uint GeometryHitFlag_HitMask = 0x01;
static const uint GeometryFlag_FaceForwardShift = 0x01;
static const uint GeometryFlag_FaceForwardMask = 0x01;
static const uint GeometryTypeFlag_HitShift = 0x02;
static const uint GeometryTypeFlag_HitMask = 0xff;

float3 offset_position_along_normal(float3 worldPosition, float3 normal) {
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

struct GeometryHit {
    float3  position;
    uint    geometryID;
    float3  shadingNormal;
    uint    primitiveID;
    float3  geometryNormal;
    uint    flags;
    float2  barycentric;
    float2  texcoord;
    float4  tangent;
    float3  padding;
    float   lambda;

    bool has_hit() { return ((flags >> GeometryHitFlag_HitShift) & GeometryHitFlag_HitMask) != 0; }
    [mutating]
    void set_hit(bool value) {
        // if not hit, reset the geometryID and primitiveID
        if (!value) { geometryID = uint(-1); primitiveID = uint(-1); }
        flags = (flags & ~(GeometryHitFlag_HitMask << GeometryHitFlag_HitShift)) 
            | (value ? 1 : 0) << GeometryHitFlag_HitShift;
    }

    bool is_face_forward() { return ((flags >> GeometryFlag_FaceForwardShift) & GeometryFlag_FaceForwardShift) != 0; }
    [mutating]
    void set_face_forward(bool value) {
        flags = (flags & ~(GeometryFlag_FaceForwardMask << GeometryFlag_FaceForwardShift)) 
            | (value ? 1 : 0) << GeometryFlag_FaceForwardShift;
    }

    Frame get_shading_frame() {
        float faceforward = is_face_forward() ? 1.f : -1.f;
        return Frame.from_normal(shadingNormal * faceforward); 
    }
    
    Ray spawn_ray(float3 dir) {
        const float3 offsetDir = faceforward(geometryNormal, -dir, geometryNormal);
        float3 offsetedPosition = offset_position_along_normal(position, offsetDir);
        if (any(abs(offsetedPosition) >= 8)) { offsetedPosition += 1e-4 * offsetDir; }
        Ray ray;
        ray.origin = offsetedPosition;
        ray.direction = dir;
        ray.tMin = 0.000;
        ray.tMax = M_INF;
        return ray;
    }

    Ray spawn_visibility_ray(float3 target) {
        float3 dir = target - position;
        float distance = length(dir);
        Ray visiblityRay = spawn_ray(dir / distance);
        visiblityRay.tMax = distance - max(1e-4, distance * 0.002);
        return visiblityRay;
    }
};

struct IntersectionPayload {
    GeometryHit hit;
    random::RandomSampler sampler;
};

enum LightType {
    DIRECTIONAL,
    POINT,
    SPOT,
    MESH_PRIMITIVE,
    SPHERE,
    RECTANGLE,
    ENVIRONMENT,
    VPL,
    MAX_ENUM,
};

struct LightData {
    LightType light_type;
    int bitfield;
    uint uintscalar_0;
    uint uintscalar_1;
    float4 floatvec_0;
    float4 floatvec_1;
    float4 floatvec_2;
};

struct SceneData {
    float3 lightBoundsMin;
    int nonDistantLightCount;
    float3 lightBoundsMax;
    int distantLightCount;
    int environmentLightID;
    int geometryOffset;
    int lightOffset;
    int padding1;

    bounds3 light_bounds() { return bounds3(lightBoundsMin, lightBoundsMax); }
};

#endif // _SRENDERER_SPT_DEFINITION_HLSLI_