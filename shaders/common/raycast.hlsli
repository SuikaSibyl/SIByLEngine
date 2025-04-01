#ifndef _SRENDERER_COMMON_RAYCAST_HEADER_
#define _SRENDERER_COMMON_RAYCAST_HEADER_

#include "math.hlsli"
#include "camera.hlsli"
#include "geometry.hlsli"

RayDesc ToRayDesc(in_ref(Ray) ray) {
    RayDesc raydesc = {};
    raydesc.Origin = ray.origin;
    raydesc.Direction = ray.direction;
    raydesc.TMin = ray.tMin;
    raydesc.TMax = ray.tMax;
    return raydesc;
}

/** Initialize a ray */
void initialize(
    inout Ray ray,
    in const float3 origin,
    in const float3 direction
) {
    ray.origin = origin;
    ray.direction = direction;
    ray.tMin = 0;
    ray.tMax = k_inf;
}

/** Initialize a ray */
void initialize(
    inout Ray ray,
    in const float3 origin,
    in const float3 direction,
    in const float tmin,
    in const float tmax
) {
    ray.origin = origin;
    ray.direction = direction;
    ray.tMin = tmin;
    ray.tMax = tmax;
}

struct CameraSample {
    float2 pFilm;
    float2 pLens;
};

/** Randomly generate a camera ray. */
Ray generateRay(in const float2 ndc, in_ref(CameraData) cameraData) {
    Ray ray;
    ray.origin = cameraData.posW;
    // Compute the normalized ray direction assuming a pinhole camera.
    ray.direction = normalize(ndc.x * cameraData.cameraU 
        + ndc.y * cameraData.cameraV 
        + cameraData.cameraW);

    float invCos = 1.f / dot(normalize(cameraData.cameraW), ray.direction);
    ray.tMin = cameraData.nearZ * invCos * 2;
    ray.tMax = cameraData.farZ * invCos * 2;

    return ray;
}

/** Compute the ray corresponding to a given sample */
Ray generateRay(in const CameraSample cameraSample, in_ref(CameraData) cameraData) {
    float2 ndc = cameraSample.pFilm * 2 - float2(1);
    return generateRay(ndc, cameraData);
}

/** Compute the ray corresponding to a given pixel */
Ray generateRay(
    float2 pixel,
    uint2 frameDim,
    bool applyJitter,
    in_ref(CameraData) cameraData
) {
    // Compute the normalized ray direction assuming a pinhole camera.
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    float2 p = (float2(pixel) + float2(0.5f, 0.5f)) / float2(frameDim);
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    if (applyJitter)
        p += float2(-cameraData.jitterX, cameraData.jitterY);
    // Compute ndc of the corresponding pixel.
    float2 ndc = float2(2.0f, -2.0f) * p + float2(-1.0f, 1.0f);
    return generateRay(ndc, cameraData);
}

bool intersect(
    in_ref(Ray) ray,
    in_ref(AABB) aabb
) {
    // Calculate the t-values for each axis
    const float3 tMin = (aabb.min - ray.origin) / ray.direction;
    const float3 tMax = (aabb.max - ray.origin) / ray.direction;
    // Find the intersection interval
    const float3 tEnter = min(tMin, tMax);
    const float3 tExit = max(tMin, tMax);
    const float tNear = maxComponent(tEnter);
    const float tFar = minComponent(tExit);
    // Check for intersection validity
    if (tNear > tFar || tFar < 0.0f) {
        // No intersection or behind the ray
        return false;
    }
    return true;
}

float intersectTMin(
    in_ref(Ray) ray,
    in_ref(AABB) aabb
) {
    // Calculate the t-values for each axis
    const float3 tMin = (aabb.min - ray.origin) / ray.direction;
    const float3 tMax = (aabb.max - ray.origin) / ray.direction;
    // Find the intersection interval
    const float3 tEnter = min(tMin, tMax);
    const float3 tExit = max(tMin, tMax);
    const float tNear = maxComponent(tEnter);
    const float tFar = minComponent(tExit);
    // Check for intersection validity
    if (tNear > tFar || tFar < 0.0f) {
        // No intersection or behind the ray
        return -1.f;
    }
    return tNear;
}

bool validRefPoint(in const float3 ref_point) {
    return !(isnan(ref_point.x) || isnan(ref_point.y) || isnan(ref_point.z));
}

#endif // !_SRENDERER_COMMON_RAYCAST_HEADER_