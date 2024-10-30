#ifndef _SRENDERER_LIGHT_BVH_HEADER_
#define _SRENDERER_LIGHT_BVH_HEADER_

#include "common/geometry.hlsli"
#include "common/octahedral.hlsli"
#include "common/ratiocv.hlsli"

struct LightBounds {
    float3 bounds_min;  // bounding box min
    float  phi;         // emitted power
    float3 bounds_max;  // bounding box max
    float  cosTheta_o;  // the maximum deviation of the emitter’s surface normal from w
    float3 w;           // the principal normal direction w
    float cosTheta_e;   // specifies the angle beyond theta_o up to which there may be emission
    float3 rgb;         // rgb power
    bool   twoSided;    // whether direction w should be 
                        // negated to specify a second cone that uses 
                        // the same pair of angles

    float3 centroid() { return (bounds_min + bounds_max) / 2; }

    static float cosSubClamped(
        float sinTheta_a, float cosTheta_a,
        float sinTheta_b, float cosTheta_b) {
        if (cosTheta_a > cosTheta_b) return 1;
        return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
    };
    
    static float sinSubClamped(
        float sinTheta_a, float cosTheta_a,
        float sinTheta_b, float cosTheta_b) {
        if (cosTheta_a > cosTheta_b) return 0;
        return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
    };

    float importance(float3 p, float3 n) {
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        float3 pc = (bounds_min + bounds_max) / 2;
        float d2 = distance_squared(p, pc);
        d2 = max(d2, length(bounds_max - bounds_min) / 2);
        // Define cosine and sine clamped subtraction lambdas

        // Compute sine and cosine of angle to vector w, theta_w
        float3 wi = normalize(p - pc);
        float cosTheta_w = dot(float3(w), wi);
        if (twoSided) cosTheta_w = abs(cosTheta_w);
        float sinTheta_w = safe_sqrt(1 - sqr(cosTheta_w));
        // Compute cos theta_b for reference point
        bounds3 bounds = bounds3(bounds_min, bounds_max);
        float cosTheta_b = DirectionCone::bound_subtended_directions(bounds, p).cosTheta;
        float sinTheta_b = safe_sqrt(1 - sqr(cosTheta_b));
        // Compute cos theta' and test against cos theta_e
        float sinTheta_o = safe_sqrt(1 - sqr(cosTheta_o));
        float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        float cosThetap = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
        if (cosThetap <= cosTheta_e) return 0;
        // Return final importance at reference point
        float importance = phi * cosThetap / d2;
        // Account for cos theta_i in importance at surfaces
        if (any(n != float3(0, 0, 0))) {
            float cosTheta_i = abs(dot(wi, n));
            float sinTheta_i = safe_sqrt(1 - sqr(cosTheta_i));
            float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
            importance *= cosThetap_i;
        }
        return importance;
    }
};

struct DirectionCone {
    float3 w;
    float cosTheta;
    __init() { w = float3(0, 0, 0); cosTheta = k_inf; }
    __init(float3 w, float cosTheta) {
        this.w = w;
        this.cosTheta = cosTheta;
    }
    bool is_empty() { return cosTheta == k_inf; }
    static DirectionCone entire_sphere() { 
        return DirectionCone(float3(0, 0, 1), -1);
    }

    static DirectionCone bound_subtended_directions(bounds3 b, float3 p) {
        // Compute bounding sphere for b and check if p is inside
        float radius; float3 pCenter;
        b.bounding_sphere(pCenter, radius);
        if (distance_squared(p, pCenter) < sqr(radius))
            return DirectionCone::entire_sphere();
        // Compute and return DirectionCone for bounding sphere
        float3 w = normalize(pCenter - p);
        float sin2ThetaMax = sqr(radius) / distance_squared(pCenter, p);
        float cosThetaMax = safe_sqrt(1 - sin2ThetaMax);
        return DirectionCone(w, cosThetaMax);
    }
};

enum ImportanceFacotr {
    Use_Power       = 1 << 1,
    Use_Distance    = 1 << 2,
    Use_Cone        = 1 << 3,
};

// 32 bytes
struct LightBVHNode {
    float phi;
    uint32_t w;
    uint32_t bitfield;
    uint32_t qb_0;
    uint32_t qb_1;
    uint32_t qb_2;
    uint16_t color_u;
    uint16_t color_v;
    uint32_t nodeinfo;

    float power_r() { return rgb().x; }
    float power_g() { return rgb().y; }
    float power_b() { return rgb().z; }
    float power_luminance() { return phi; }
    float3 rgb() { float3 _yuv = yuv(); return yuv2rgb(_yuv); }
    float3 yuv() { return float3(phi, unpack_cpu_half(color_u), unpack_cpu_half(color_v)); }

    bounds3 bounds(bounds3 allb) {
        bounds3 b;
        b.pMin = float3(
            lerp(allb.pMin.x, allb.pMax.x, (qb_0 >> 16) / 65535.f),
            lerp(allb.pMin.y, allb.pMax.y, (qb_1 >> 16) / 65535.f),
            lerp(allb.pMin.z, allb.pMax.z, (qb_2 >> 16) / 65535.f));
        b.pMax = float3(
            lerp(allb.pMin.x, allb.pMax.x, (qb_0 & 0xffff) / 65535.f),
            lerp(allb.pMin.y, allb.pMax.y, (qb_1 & 0xffff) / 65535.f),
            lerp(allb.pMin.z, allb.pMax.z, (qb_2 & 0xffff) / 65535.f));
        return b;
    }
    
    float importance(float3 p, float3 n, bounds3 allb, uint factors = 7) {
        bounds3 b = bounds(allb);
        float cosTheta_o = cos_theta_o();
        float cosTheta_e = cos_theta_e();
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        float3 pc = (b.pMin + b.pMax) / 2;
        float d2 = distance_squared(p, pc);
        d2 = max(d2, length(b.diagonal()) / 2);
        // Compute sine and cosine of angle to vector w, theta_w
        float3 wi = normalize(p - pc);
        float cosTheta_w = dot(Unorm32OctahedronToUnitVector(w), wi);
        if (two_sided()) cosTheta_w = abs(cosTheta_w);
        float sinTheta_w = safe_sqrt(1 - sqr(cosTheta_w));
        // Compute cos theta_b for reference point
        float cosTheta_b = DirectionCone::bound_subtended_directions(b, p).cosTheta;
        float sinTheta_b = safe_sqrt(1 - sqr(cosTheta_b));
        // Compute cos theta' and test against cos theta_e
        float sinTheta_o = safe_sqrt(1 - sqr(cosTheta_o));
        float cosTheta_x = LightBounds::cosSubClamped(
            sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        float sinTheta_x = LightBounds::sinSubClamped(
            sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        float cosThetap = LightBounds::cosSubClamped(
            sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
        // if (bool(factors & (uint)ImportanceFacotr::Use_Cone)) {
        //     if (cosThetap <= cosTheta_e) return 0;
        // }
        // Return final importance at reference point
        float importance = 1.f;
        if (bool(factors & (uint)ImportanceFacotr::Use_Power)) importance *= phi;
        if (bool(factors & (uint)ImportanceFacotr::Use_Distance)) importance /= d2;
        if (bool(factors & (uint)ImportanceFacotr::Use_Cone)) {
            // importance *= cosThetap;
            // Account for cos theta_i in importance at surfaces
            if (any(n != float3(0, 0, 0))) {
                float cosTheta_i = abs(dot(wi, n));
                float sinTheta_i = safe_sqrt(1 - sqr(cosTheta_i));
                float cosThetap_i = LightBounds::cosSubClamped(
                    sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
                importance *= cosThetap_i;
            }
        }
        // importance = rgb().b * importance / phi;
        
        return importance;
    }

    primal_auxiliary<3> importance_with_aux(float3 p, float3 n, bounds3 allb, uint factors = 7) {
        primal_auxiliary<3> o;
        o.primal = importance(p, n, allb, factors);
        o.auxiliaries = rgb() * o.primal / phi;
        return o;
    }

    bool is_leaf() { return (nodeinfo >> 31) != 0; }
    int child_or_light_index() { return int(nodeinfo & 0x7FFFFFFF); }
    bool two_sided() { return (bitfield & 1) != 0; }
    float cos_theta_o() { return 2 * ((bitfield >> 17) / 32767.f) - 1; }
    float cos_theta_e() { return 2 * (((bitfield >> 2) & 0x7FFF) / 32767.f) - 1; }

    // maps the provided value (which is expected to be the cosine 
    // of an angle and thus between -1 and 1) to a 15-bit unsigned integer
    static uint quantize_cos(float c) {
        return uint(floor(32767.f * ((c + 1) / 2)));
    }

    // remaps a coordinate value c between min and max to the range
    // [0, 2^16 - 1] range of values that an unsigned 16-bit integer can store.
    static float quantize_bounds(float c, float min, float max) {
        if (min == max) return 0;
        return 65535.f * clamp((c - min) / (max - min), 0, 1);
    }
    
    static LightBVHNode from_lightbounds(LightBounds lb, bounds3 allb) {
        uint qCosTheta_o = quantize_cos(lb.cosTheta_o);
        uint qCosTheta_e = quantize_cos(lb.cosTheta_e);
        uint twoSided = lb.twoSided ? 1 : 0;
        LightBVHNode node;
        node.bitfield = (qCosTheta_o << 17) | (qCosTheta_e << 2) | twoSided;
        node.phi = lb.phi;
        node.w = UnitVectorToUnorm32Octahedron(normalize(lb.w));
        // Quantize bounding box into qb
        uint qb[3];
        for (int c = 0; c < 3; ++c) {
            uint qb_lc = uint(floor(quantize_bounds(
                lb.bounds_min[c], allb.pMin[c], allb.pMax[c])));
            uint qb_rc = uint(ceil(quantize_bounds(
                lb.bounds_max[c], allb.pMin[c], allb.pMax[c])));
            qb[c] = (qb_lc << 16) | qb_rc;
        }
        node.qb_0 = qb[0];
        node.qb_1 = qb[1];
        node.qb_2 = qb[2];
    }

    static LightBVHNode make_leaf(uint lightID, LightBounds bounds) {
    }
};

DirectionCone union (DirectionCone a, DirectionCone b) {
    // Handle the cases where one or both cones are empty
    if (a.is_empty()) return b;
    if (b.is_empty()) return a;
    // Handle the cases where one cone is inside the other
    float theta_a = safe_acos(a.cosTheta);
    float theta_b = safe_acos(b.cosTheta);
    float theta_d = angle_between(a.w, b.w);
    if (min(theta_d + theta_b, k_pi) <= theta_a) return a;
    if (min(theta_d + theta_a, k_pi) <= theta_b) return b;
    // Compute the spread angle of the merged cone, theta_o
    float theta_o = (theta_a + theta_d + theta_b) / 2;
    if (theta_o >= k_pi)
        return DirectionCone::entire_sphere();
    // Find the merged cone’s axis and return cone union
    float theta_r = theta_o - theta_a;
    float3 wr = cross(a.w, b.w);
    if (length_squared(wr) == 0)
        return DirectionCone::entire_sphere();
    float3 w = mul(a.w, rotate(degrees(theta_r), wr));
    return DirectionCone(w, cos(theta_o));
}

LightBounds union(LightBounds a, LightBounds b) {
    // If one LightBounds has zero power, return the other
    if (a.phi == 0) return b;
    if (b.phi == 0) return a;
    // Find average direction and updated angles for LightBounds
    DirectionCone cone = union(DirectionCone(a.w, a.cosTheta_o),
                               DirectionCone(b.w, b.cosTheta_o));
    float cosTheta_o = cone.cosTheta;
    float cosTheta_e = min(a.cosTheta_e, b.cosTheta_e);
    // Return final LightBounds union
    LightBounds lb;
    lb.bounds_min = min(a.bounds_min, b.bounds_min);
    lb.bounds_max = max(a.bounds_max, b.bounds_max);
    lb.w = cone.w;
    lb.phi = a.phi + b.phi;
    lb.rgb = a.rgb + b.rgb;
    lb.cosTheta_o = cosTheta_o;
    lb.cosTheta_e = cosTheta_e;
    lb.twoSided = a.twoSided || b.twoSided;
    return lb;
}

#endif // _SRENDERER_LIGHT_BVH_HEADER_