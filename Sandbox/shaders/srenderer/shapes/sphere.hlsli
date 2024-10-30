#ifndef _SRENDERER_SHAPES_SPHERE_HLSLI_
#define _SRENDERER_SHAPES_SPHERE_HLSLI_

#include "common/raycast.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/shapes/shape.hlsli"

struct SphereParameter {
    float3  center;
    float   radius;
};

struct Sphere {
    typedef SphereParameter TParam;

    static float hit(Ray ray, SphereParameter sphere) {
        const float3 center = sphere.center;
        const float radius = sphere.radius;
        const float3 origin = ray.origin;
        const float3 direction = ray.direction;
        const float3 oc = origin - center;
        const float a = dot(direction, direction);
        const float b = 2.0 * dot(oc, direction);
        const float c = dot(oc, oc) - radius * radius;
        // Degenerated case
        if (a == 0) {
            if (b == 0)
                return -1.0; // if no solve
            return -c / b;
        }
        float3 term = oc - (dot(oc, direction) * direction);
        float discriminant = 4 * a * (radius * radius - dot(term, term));
        if (discriminant < 0)
            return -1.0; // if no solve
        const float root_discriminant = sqrt(discriminant);
        float t0, t1;
        if (b >= 0) {
            t0 = (-b - root_discriminant) / (2 * a);
            t1 = 2 * c / (-b - root_discriminant);
        } else {
            t0 = 2 * c / (-b + root_discriminant);
            t1 = (-b + root_discriminant) / (2 * a);
        }
        float t = -1;
        if (t0 >= 0 && t0 < k_inf)
            t = t0;
        if (t1 >= 0 && t1 < k_inf && t < 0)
            t = t1;
        return t;
    }

    static ishape::sample sample(ishape::sample_in i, SphereParameter param) {
        const float3 ref_point = i.position;
        const float3 center = param.center;
        const float radius = param.radius;
        bool validRef = validRefPoint(ref_point);
        
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        if (!validRef || distance_squared(ref_point, center) < radius * radius) {
            float z = 1 - 2 * i.uv.x;
            float r_ = sqrt(max(0., 1 - z * z));
            float phi = 2 * k_pi * i.uv.y;
            float3 offset = normalize(float3(r_ * cos(phi), r_ * sin(phi), z));
            ishape::sample sample;
            sample.position = center + radius * offset;
            sample.normal = offset;
            sample.pdf = 1. / surface_area(param);
            return sample;
        }
        // Otherwise sample a ray inside a cone towards the sphere center.
        // Build a coordinate system with n pointing towards the sphere
        float3 dir_to_center = normalize(center - ref_point);
        float3x3 frame = createONB(dir_to_center);
        float sin_theta_max_sq = radius * radius / distance_squared(ref_point, center);
        float cos_theta_max = sqrt(max(0.f, 1 - sin_theta_max_sq));
        float3 n_on_sphere;
        ishape::sample sample;
        if (cos_theta_max > 0.999999) {
            n_on_sphere = -normalize(mul(cosineSampleHemisphere(i.uv), frame));
            sample.position = radius * n_on_sphere + center;
            sample.normal = n_on_sphere;
            const float3 mdir_to_center = normalize(ref_point - center);
            const float3 dir = normalize(sample.position - ref_point);
            sample.pdf = cosineHemispherePdf(abs(dot(mdir_to_center, normalize(sample.position - center)))) 
            * distance_squared(ref_point, sample.position) / abs(dot(n_on_sphere, dir));
            return sample;
        }
        else {
            // Uniformly interpolate between 1 (angle 0) and max
            float cos_theta = (1 - i.uv[0]) + i.uv[0] * cos_theta_max;
            float sin_theta = sqrt(max(0.f, 1 - cos_theta * cos_theta));
            float phi = i.uv[1] * 2 * k_pi;
            // Compute angle from center of sphere to sampled point on surface
            float dc = distance(ref_point, center);
            float ds = dc * cos_theta - sqrt(max(0.f, radius * radius - dc * dc * sin_theta * sin_theta));
            float cos_alpha = (dc * dc + radius * radius - ds * ds) / (2 * dc * radius);
            float sin_alpha = sqrt(max(0.f, 1 - cos_alpha * cos_alpha));
            // get the normal of the sample
            n_on_sphere = -normalize(mul(float3(sin_alpha * cos(phi), sin_alpha * sin(phi), cos_alpha), frame));
            sample.position = radius * n_on_sphere + center;
            sample.normal = n_on_sphere;
            // Uniform sampling PDF of a cone.
            sample.pdf = 1 / (2 * k_pi * (1 - cos_theta_max));
            return sample;
        }
    }

    static float surface_area(SphereParameter param) {
        return 4 * k_pi * param.radius * param.radius;
    }
    
    static float sample_pdf(ishape::pdf_in i, SphereParameter param) {
        const float3 ref_point = i.ref_point;
        const float3 sample_pos = i.sample_point;
        const float3 center = param.center;
        const float r = param.radius;
        
        bool validRef = validRefPoint(ref_point);
        const float3 p_on_sphere = sample_pos;
        const float3 n_on_sphere = normalize(sample_pos - center);
        const float3 dir = normalize(p_on_sphere - ref_point);

        if (!validRef || distance_squared(ref_point, center) < r * r) {
            // If the reference point is inside the sphere, just sample the whole sphere uniformly
            return 1. / surface_area(param)
               * distance_squared(ref_point, p_on_sphere) / abs(dot(n_on_sphere, dir));
        }
        float sin_theta_max_sq = r * r / distance_squared(ref_point, center);
        float cos_theta_max = sqrt(max(0.f, 1 - sin_theta_max_sq));
        if (cos_theta_max > 0.999999) {
            float3 mdir_to_center = normalize(ref_point - center);
            return cosineHemispherePdf(abs(dot(mdir_to_center, normalize(sample_pos - center)))) 
            * distance_squared(ref_point, p_on_sphere) / abs(dot(n_on_sphere, dir));
        }
        else {
            // Uniform sampling PDF of a cone.
            float pdf_solid_angle = 1 / (2 * k_pi * (1 - cos_theta_max));
            // Convert it back to area measure
            // Output the result
            return pdf_solid_angle;
        }
    }
};

GeometryHit fetchSphereGeometryHit(GeometryData geometry, Ray ray, float t) {
    const float4x4 o2w = ObjectToWorld(geometry);
    const float4x4 o2wn = ObjectToWorldNormal(geometry);
    const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
    // Record the intersection
    // *: we should re-normalize it, using tHit is super unstable
    const float3 hitPoint = sphere_center + sphere_radius * normalize(ray.origin + ray.direction * t - sphere_center);
    const float3 geometric_normal = normalize(hitPoint - sphere_center);
    const float3 cartesian = normalize(mul(o2wn, float4(geometric_normal, 0)).xyz);
    // We use the spherical coordinates as uv
    // We use the convention that y is up axis.
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    const float elevation = acos(clamp(cartesian.y, -1., 1.));
    const float azimuth = atan2(cartesian.z, cartesian.x);
    float3x3 TBN = createFrame(geometric_normal);

    GeometryHit hit;
    hit.position = hitPoint;
    hit.shadingNormal = geometric_normal;
    hit.geometryNormal = geometric_normal;
    hit.barycentric = float2(0, 0);
    hit.texcoord = float2(-azimuth * k_inv_2_pi, elevation * k_inv_pi);
    hit.tangent = float4(TBN[0], 0);
    hit.barycentric = float2(0.333);
    if (distance(ray.origin, sphere_center) >= sphere_radius) 
        SetFaceForward(hit, true);
    else {
        SetFaceForward(hit, false);
        hit.shadingNormal = -hit.shadingNormal;
        hit.geometryNormal = -hit.geometryNormal;
    }
    return hit;
}

#endif // _SRENDERER_SHAPES_SPHERE_HLSLI_