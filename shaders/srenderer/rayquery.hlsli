#ifndef _SRENDERER_RAYQUERY_HEADER_
#define _SRENDERER_RAYQUERY_HEADER_

#include "scene-binding.hlsli"
#include "srenderer/shapes/sphere.hlsli"
#include "srenderer/shapes/cube.hlsli"
#include "srenderer/shapes/rectangle.hlsli"

PrimaryPayload PrimaryRayQuery(in const Ray ray) {
    PrimaryPayload payload;
    SetHit(payload.hit, false);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GPUScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if(true) {
                q.CommitNonOpaqueTriangleHit();
            }
        }
        case CANDIDATE_PROCEDURAL_PRIMITIVE: {
            const uint primitiveType = q.CandidateInstanceID();
            const uint primitiveID = q.CandidatePrimitiveIndex();
            const uint geometryID = q.CandidateInstanceIndex() + q.CandidateGeometryIndex();
            GeometryData geometry = GPUScene_geometry[geometryID];
            // sphere primitive
            if (primitiveType == 1) {
                float4x4 o2w = ObjectToWorld(geometry);
                const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
                const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
                SphereParameter sphere = { sphere_center, sphere_radius };
                // Sphere intersection
                const float tHit = Sphere::hit(ray, sphere);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
            // rectangle primitive
            else if (primitiveType == 2) {
                RectangleParameter rect = { ObjectToWorld(geometry), WorldToObject(geometry) };
                const float tHit = Rectangle::hit(ray, rect);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
            // cube primitive
            else if (primitiveType == 3) {
                CubeParameter cube = { ObjectToWorld(geometry), WorldToObject(geometry) };
                const float tHit = Cube::hit(ray, cube);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
        }
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        // Do hit shading
        const uint primitiveID = q.CommittedPrimitiveIndex();
        const uint geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        const float2 bary = q.CommittedTriangleBarycentrics();
        const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
        payload.hit = fetchTrimeshGeometryHit(geometryID, barycentrics, primitiveID, ray);
        SetHit(payload.hit, true);
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        const uint primitiveType = q.CommittedInstanceID();
        const uint primitiveID = q.CommittedPrimitiveIndex();
        const uint geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        if (primitiveType == 1) {
            GeometryData geometry = GPUScene_geometry[geometryID];
            payload.hit = fetchSphereGeometryHit(geometry, ray, q.CommittedRayT());
            payload.hit.geometryID = geometryID;
            payload.hit.primitiveID = primitiveID;
            SetHit(payload.hit, true);
            break;
        }
        else if (primitiveType == 2) {
            GeometryData geometry = GPUScene_geometry[geometryID];
            payload.hit = fetchRectangleGeometryHit(geometry, ray, q.CommittedRayT());
            payload.hit.geometryID = geometryID;
            payload.hit.primitiveID = primitiveID;
            SetHit(payload.hit, true);
            break;
        }
        else if (primitiveType == 3) {
            GeometryData geometry = GPUScene_geometry[geometryID];
            payload.hit = fetchCubeGeometryHit(geometry, ray, q.CommittedRayT());
            payload.hit.geometryID = geometryID;
            payload.hit.primitiveID = primitiveID;
            SetHit(payload.hit, true);
            break;
        }
        break;
    }
    case COMMITTED_NOTHING: {
        break;
    }
    }

    return payload;
}

bool VisibilityRayQuery(in const Ray ray) {
    PrimaryPayload payload;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GPUScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            q.CommitNonOpaqueTriangleHit();
            // break;
        }
        case CANDIDATE_PROCEDURAL_PRIMITIVE: {
            const uint primitiveType = q.CandidateInstanceID();
            const uint primitiveID = q.CandidatePrimitiveIndex();
            const uint geometryID = q.CandidateInstanceIndex() + q.CandidateGeometryIndex();
            GeometryData geometry = GPUScene_geometry[geometryID];
            // sphere primitive
            if (primitiveType == 1) {
                float4x4 o2w = ObjectToWorld(geometry);
                const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
                const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
                SphereParameter sphere = { sphere_center, sphere_radius };
                // Sphere intersection
                const float tHit = Sphere::hit(ray, sphere);
                if (tHit > 0 && tHit < ray.tMax) q.CommitProceduralPrimitiveHit(tHit);
            }
            // rectangle primitive
            else if (primitiveType == 2) {
                RectangleParameter rect = { ObjectToWorld(geometry), WorldToObject(geometry) };
                const float tHit = Rectangle::hit(ray, rect);
                if (tHit > 0 && tHit < ray.tMax) q.CommitProceduralPrimitiveHit(tHit);
            }
            // cube primitive
            else if (primitiveType == 3) {
                CubeParameter cube = { ObjectToWorld(geometry), WorldToObject(geometry) };
                const float tHit = Cube::hit(ray, cube);
                if (tHit > 0 && tHit < ray.tMax) q.CommitProceduralPrimitiveHit(tHit);
            }
        }
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: return false;
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: return false;
    case COMMITTED_NOTHING: break;
    }

    return true;
}

RawPayload RawRayQuery(in const Ray ray) {
    RawPayload payload; payload.hasHit = false;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GPUScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if (true) {
                q.CommitNonOpaqueTriangleHit();
            }
            if (false) {
                q.Abort();
            }
            break;
        }
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        payload.primitiveID = q.CommittedPrimitiveIndex();
        payload.geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        payload.barycentric = q.CommittedTriangleBarycentrics();
        payload.hasHit = true;
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        break;
    }
    case COMMITTED_NOTHING: {
        break;
    }
    }

    return payload;
}

#endif // _SRENDERER_RAYQUERY_HEADER_