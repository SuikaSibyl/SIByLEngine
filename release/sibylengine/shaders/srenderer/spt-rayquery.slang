#ifndef _SRENDERER_SPT_RAYQUERY_HLSLI_
#define _SRENDERER_SPT_RAYQUERY_HLSLI_

#include "srenderer/shapes/cube.slang"
#include "srenderer/shapes/rectangle.slang"
#include "srenderer/shapes/sphere.slang"
#include "srenderer/spt-bindings.slang"
#include "srenderer/spt-definition.slang"

Ray spawn_ray(float3 pos, float3 dir) {
    Ray ray;
    ray.origin = pos;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = 1e6;
    return ray;
}

Ray spawn_visibility_ray(float3 origin, float3 target) {
    float3 dir = target - origin;
    float distance = length(dir);
    Ray visiblityRay = spawn_ray(origin, dir / distance);
    visiblityRay.tMax = distance - min(0.01, distance * 0.02);
    return visiblityRay;
}

struct MinimalHit {
    int     primitiveType;
    uint    primitiveID;
    uint    geometryID;
    uint    padding;
    float2  barycentrics;
};

MinimalHit intersection_ray_query_minimal(Ray ray, uint bvhID = 0) {
    MinimalHit payload;
    payload.primitiveType = -1;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(se_scene_tlas[bvhID], 0, 0xff, ray.to_desc());
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if (true) {
                q.CommitNonOpaqueTriangleHit();
            }
        } break;
        case CANDIDATE_PROCEDURAL_PRIMITIVE: {
            const uint primitiveType = q.CandidateInstanceID();
            const uint primitiveID = q.CandidatePrimitiveIndex();
            const uint geometryID = q.CandidateInstanceIndex() + q.CandidateGeometryIndex();
            GeometryData geometry = scene_read_geometry(geometryID);
            // sphere primitive
            if (primitiveType == 1) {
                float4x4 o2w = geometry.object_to_world();
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
                RectangleParameter rect = { geometry.object_to_world(), geometry.world_to_object() };
                const float tHit = Rectangle::hit(ray, rect);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
            // cube primitive
            else if (primitiveType == 3) {
                CubeParameter cube = { geometry.object_to_world(), geometry.world_to_object() };
                const float tHit = Cube::hit(ray, cube);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
        } break;
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        // Do hit shading
        const uint primitiveID = q.CommittedPrimitiveIndex();
        const uint geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        const float2 bary = q.CommittedTriangleBarycentrics();
        
        payload.primitiveType = 0; // Triangle
        payload.primitiveID = primitiveID;
        payload.geometryID = geometryID;
        payload.barycentrics = bary;
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        const uint primitiveType = q.CommittedInstanceID();
        const uint primitiveID = q.CommittedPrimitiveIndex();
        const uint geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        payload.primitiveType = primitiveType; // Triangle
        payload.primitiveID = primitiveID;
        payload.geometryID = geometryID;
        break;
    }
    case COMMITTED_NOTHING: {
        break;
    }
    }

    return payload;
}

IntersectionPayload intersection_ray_query(Ray ray, uint bvhID = 0) {
    IntersectionPayload payload;
    payload.hit.set_hit(false);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(se_scene_tlas[bvhID], 0, 0xff, ray.to_desc());
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if (true) {
                q.CommitNonOpaqueTriangleHit();
            }
        } break;
        case CANDIDATE_PROCEDURAL_PRIMITIVE: {
            const uint primitiveType = q.CandidateInstanceID();
            const uint primitiveID = q.CandidatePrimitiveIndex();
            const uint geometryID = q.CandidateInstanceIndex() + q.CandidateGeometryIndex();
            GeometryData geometry = scene_read_geometry(geometryID);
            // sphere primitive
            if (primitiveType == 1) {
                float4x4 o2w = geometry.object_to_world();
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
                RectangleParameter rect = { geometry.object_to_world(), geometry.world_to_object() };
                const float tHit = Rectangle::hit(ray, rect);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
            // cube primitive
            else if (primitiveType == 3) {
                CubeParameter cube = { geometry.object_to_world(), geometry.world_to_object() };
                const float tHit = Cube::hit(ray, cube);
                if (tHit > q.RayTMin() && tHit < q.CommittedRayT())
                    q.CommitProceduralPrimitiveHit(tHit);
            }
        } break;
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        // Do hit shading
        const uint primitiveID = q.CommittedPrimitiveIndex();
        const uint geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        const float2 bary = q.CommittedTriangleBarycentrics();
        const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
        payload.hit = fetch_trimesh_geometry_hit(geometryID, barycentrics, primitiveID, ray);
        payload.hit.set_hit(true);
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        const uint primitiveType = q.CommittedInstanceID();
        const uint primitiveID = q.CommittedPrimitiveIndex();
        const uint geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        GeometryData geometry = scene_read_geometry(geometryID);
        if (primitiveType == 1) {
            payload.hit = fetch_sphere_geometry_hit(geometry, ray, q.CommittedRayT());
            payload.hit.geometryID = geometryID;
            payload.hit.primitiveID = primitiveID;
            payload.hit.set_hit(true);
            break;
        }
        else if (primitiveType == 2) {
            payload.hit = fetch_rectangle_geometry_hit(geometry, ray, q.CommittedRayT());
            payload.hit.geometryID = geometryID;
            payload.hit.primitiveID = primitiveID;
            payload.hit.set_hit(true);
            break;
        }
        else if (primitiveType == 3) {
            payload.hit = fetch_cube_geometry_hit(geometry, ray, q.CommittedRayT());
            payload.hit.geometryID = geometryID;
            payload.hit.primitiveID = primitiveID;
            payload.hit.set_hit(true);
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

bool visibility_ray_query(in const Ray ray, uint bvhID = 0) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(se_scene_tlas[bvhID], 0, 0xff, ray.to_desc());
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
            GeometryData geometry = scene_read_geometry(geometryID);
            // sphere primitive
            if (primitiveType == 1) {
                float4x4 o2w = geometry.object_to_world();
                const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
                const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
                SphereParameter sphere = { sphere_center, sphere_radius };
                // Sphere intersection
                const float tHit = Sphere::hit(ray, sphere);
                if (tHit > 0 && tHit < ray.tMax) q.CommitProceduralPrimitiveHit(tHit);
            }
            // rectangle primitive
            else if (primitiveType == 2) {
                RectangleParameter rect = { geometry.object_to_world(), geometry.world_to_object() };
                const float tHit = Rectangle::hit(ray, rect);
                if (tHit > 0 && tHit < ray.tMax) q.CommitProceduralPrimitiveHit(tHit);
            }
            // cube primitive
            else if (primitiveType == 3) {
                CubeParameter cube = { geometry.object_to_world(), geometry.world_to_object() };
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


#endif // _SRENDERER_SPT_RAYQUERY_HLSLI_