#ifndef _SRENDERER_GEOMETRY_HEADER_
#define _SRENDERER_GEOMETRY_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"

float3 to_world(in_ref(float3x3) frame, in_ref(float3) v) {
    return v[0] * frame[0] + v[1] * frame[1] + v[2] * frame[2];
}

float3 to_local(in_ref(float3x3) frame, in_ref(float3) v) {
    return float3(dot(v, frame[0]), dot(v, frame[1]), dot(v, frame[2]));
}

/** Compute bitangent according to tangent and normal */
float3 computeBitangent(in float3 tangent, in float3 normal) {
    return cross(normal, tangent);
}

float3x3 buildTangentToWorld(in float4 tangentWS, in float3 normalWS) {
    // Gram-Schmidt process
    float3 T = normalize(tangentWS.xyz);
    float3 N = normalize(normalWS);
    // re-orthogonalize T with respect to N
    T = normalize(T - dot(T, N) * N);
    // then retrieve perpendicular vector B with the cross product of T and N
    float3 B = cross(N, T) * tangentWS.w;
    return float3x3(T, B, N);
}

/**
 * Create a frame from normal.
 * This is probably the most stable one I have ever seen.
 * And the neumerical stability turns out to be very important here.
 * The implementation is based on Frisvad's paper
 * "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization"
 * @url: https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
 * Learned it from lajolla renderer's code.
 * @param n the input normal
 * @return the created frame
 */
float3x3 createFrame(in float3 n) {
    if (n[2] < float(-1 + 1e-6)) {
        return float3x3(float3(0, -1, 0), float3(-1, 0, 0), n);
    } else {
        const float a = 1 / (1 + n[2]);
        const float b = -n[0] * n[1] * a;
        return float3x3(float3(1 - n[0] * n[0] * a, b, -n[0]),
                    float3(b, 1 - n[1] * n[1] * a, -n[1]), n);
    }
}

/**
 * Create ONB (orthonormal bases bases) from normal.
 * Just a alias of createFrame.
 * @param n the input normal
 * @return the ONB matrix
 */
float3x3 createONB(in float3 n) {
    return createFrame(n);
}

/************************************************************************
*                           Bounding Box Utils                          *
************************************************************************/
struct AABB {
    float3 min;
    float3 max;
};

AABB aabbUnion(in AABB a, in AABB b) {
    AABB result;
    result.min = min(a.min, b.min);
    result.max = max(a.max, b.max);
    return result;
}

bool PointInAABB(in float3 p, in AABB aabb) {
    return all(aabb.min < p) && all(p < aabb.max);
}

/************************************************************************
*                            Rotation Utils                             *
************************************************************************/

/** Return a matrix that rotate vector around axis by angle. */
float4x4 rotationMatrix(
    in const float3 axis,     // should be normalized
    in const float angle
) {
    const float s = sin(angle);
    const float c = cos(angle);
    const float oc = 1.0 - c;
    return float4x4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

/** Rotate vector around axis by angle */
float3 rotate(
    in const float3 v,
    in const float3 axis,     // should be normalized
    in const float angle
) {
    return mul(rotationMatrix(axis, angle), float4(v, 0.0)).xyz;
}

float3x3 getRotationMatrixFromAToB(in const float3 A, in const float3 B) {
    const float3 v = cross(A, B);
    const float c = dot(A, B);
    const float f = 1.f / (1.f + c);
    const float xx = v.x * v.x;
    const float yy = v.y * v.y;
    const float zz = v.z * v.z;
    const float xy = v.x * v.y;
    const float yz = v.y * v.z;
    const float xz = v.x * v.z;

    // Special case: A = (-B)
    if (1.f + c == 0.f)
        return {
            -1.f, 0.f, 0.f, // row 1
            0.f, -1.f, 0.f, // row 2
            0.f, 0.f, -1.f, // row 3
        };

    return {
        1.f - (yy + zz) * f, xy * f - v.z, v.y + xz * f, // Row 1
        v.z + xy * f, 1.f - (xx + zz) * f, yz * f - v.x, // Row 2
        xz * f - v.y, yz * f + v.x, 1.f - (xx + yy) * f, // Row 3
    };
}

/************************************************************************
*                             Normal Utils                              *
************************************************************************/

// Normal Cone
// -----------------------------------------------------------------------

struct NormalCone {
    float3 direction;
    float cosAngle;
};

struct NormalConeEx {
    float3 direction;
    float theta_e;
    float theta_o;
    bool  valid;

    __init(float3 _direction,
           float _theta_e,
           float _theta_o,
           bool _valid)
    {
        direction = _direction;
        theta_e = _theta_e;
        theta_o = _theta_o;
        valid = _valid;
    }
};

NormalConeEx coneUnion(
    in NormalConeEx a,
    in NormalConeEx b
) {
    if(!a.valid && !b.valid) return NormalConeEx(
        float3(0.0f, 0.0f, 0.0f),
        0.0f,
        0.0f,
        false
    );
    if(!a.valid) return b;
    if(!b.valid) return a;

    NormalConeEx result;
    result.valid = true;
    if (b.theta_o > a.theta_o) {
        // Swap(a, b);
        NormalConeEx tmp = a; a = b; b = tmp;
    }
    const float theta_d = acos(dot(a.direction, b.direction));
    const float theta_e = max(a.theta_e, b.theta_e);
    if (min(theta_d + b.theta_o, k_pi) <= a.theta_o) {
        result.direction = a.direction;
        result.theta_o = a.theta_o;
        result.theta_e = theta_e;
        return result;
    }
    else {
        const float theta_o = (theta_d + a.theta_o + b.theta_o) * 0.5f;
        if(k_pi <= theta_o) {
            return NormalConeEx(a.direction, k_pi, theta_e, true);
        }
        const float theta_r = theta_o - a.theta_o;
        const float3 direction = rotate(a.direction, normalize(cross(a.direction, b.direction)), -theta_r);
        if(isnan(direction.x)) {
            return NormalConeEx(a.direction, theta_o, theta_e, true);
        }
        return NormalConeEx(normalize(direction), theta_o, theta_e, true);
    }
}

float3 packNormal(in const float3 normal) {
    return 0.5 * float3(normal + 1.0);
}

float3 unpackNormal(in const float3 normal) {
    return normalize(normal * 2.0 - 1.0);
}

float3 getBentNormal(float3 geometryNormal, float3 shadingNormal, float3 viewDirection) {
    // Flip the normal in case we're looking at the geometry from its back side
    if (dot(geometryNormal, viewDirection) > 0) {
        geometryNormal = -geometryNormal;
        shadingNormal = -shadingNormal;
    }
    // Specular reflection in shading normal
    float3 R = reflect(viewDirection, shadingNormal);
    float a = dot(geometryNormal, R);
    if (a < 0) { // Perturb normal
        float b = max(0.001, dot(shadingNormal, geometryNormal));
        return normalize(-viewDirection + normalize(R - shadingNormal * a / b));
    }
    return shadingNormal;
}

/************************************************************************
*                             Intersection                              *
************************************************************************/

/** Light weight ray triangle intersection
* @ref: https://github.com/SebLague/Gamedev-Maths/blob/master/PointInTriangle.cs */
bool ray_triangle_intersection_lightweight(
    in_ref(float3) ray, in_ref(float3) rayDir,
    in_ref(float3) A, in_ref(float3) B, in_ref(float3) C,
    out_ref(float2) barycentricCoord
) {
    float3 normal = normalize(cross(B - A, C - A));
    float t = dot(-ray, normal) / dot(normal, rayDir);
    float3 P = ray + rayDir * t;  // hit point

    float s1 = C.y - A.y;
    float s2 = C.x - A.x;
    float s3 = B.y - A.y;
    float s4 = P.y - A.y;

    float w1 = (A.x * s1 + s4 * s2 - P.x * s1) / (s3 * s2 - (B.x - A.x) * s1);
    float w2 = (s4 - w1 * s3) / s1;

    barycentricCoord = float2(w1, w2);
    return w1 >= 0.0 && w2 >= 0.0 && (w1 + w2) <= 1.0 ? true : false;
}

static const float MachineEpsilon = 1.0e-6f;

float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

bool ray_triangle_intersection_pbrt(
    in_ref(float3) ro, in_ref(float3) rd, in_ref(float) tmin, in_ref(float) tmax,
    in_ref(float3) p0, in_ref(float3) p1, in_ref(float3) p2,
    out_ref(float3) barycentricCoord
) {
    // barycentricCoord initialize
    barycentricCoord = float3(0.0f, 0.0f, 0.0f);
    // Perform ray–triangle intersection test
    // ** Translate vertices based on ray origin
    float3 p0t = p0 - ro;
    float3 p1t = p1 - ro;
    float3 p2t = p2 - ro;
    // ** Permute components of triangle vertices and ray direction
    int kz = maxDimension(abs(rd));
    int kx = kz + 1; if (kx == 3) kx = 0;
    int ky = kx + 1; if (ky == 3) ky = 0;
    float3 d = permute(rd, kx, ky, kz);
    p0t = permute(p0t, kx, ky, kz);
    p1t = permute(p1t, kx, ky, kz);
    p2t = permute(p2t, kx, ky, kz);
    // ** Apply shear transformation to translated vertex positions
    float Sx = -d.x / d.z;
    float Sy = -d.y / d.z;
    float Sz = 1.f / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;
    // Compute edge function coefficients e0, e1, and e2>> 
    float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    float e2 = p0t.x * p1t.y - p0t.y * p1t.x;
    // Fall back to double-precision test at triangle edges
    if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) {
        double p2txp1ty = double(p2t.x) * double(p1t.y);
        double p2typ1tx = double(p2t.y) * double(p1t.x);
        // e0 = float(p2typ1tx - p2txp1ty);
        // double p0txp2ty = (double)p0t.x * (double)p2t.y;
        // double p0typ2tx = (double)p0t.y * (double)p2t.x;
        // e1 = float(p0typ2tx - p0txp2ty);
        // double p1txp0ty = (double)p1t.x * (double)p0t.y;
        // double p1typ0tx = (double)p1t.y * (double)p0t.x;
        // e2 = float(p1typ0tx - p1txp0ty);
    }
    // Perform triangle edge and determinant tests 
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    float det = e0 + e1 + e2;
    if (det == 0)
        return false;
    // Compute scaled hit distance to triangle and test against ray  range>> 
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < tmax * det))
        return false;
    else if (det > 0 && (tScaled <= 0 || tScaled > tmax * det))
        return false;

    // Compute barycentric coordinates and  value for triangle intersection
    float invDet = 1 / det;
    float b0 = e0 * invDet;
    float b1 = e1 * invDet;
    float b2 = e2 * invDet;
    float t = tScaled * invDet;

    // // Ensure that computed triangle is conservatively greater than zero
    // // ** Compute  term for triangle error bounds
    // float maxZt = maxComponent(abs(float3(p0t.z, p1t.z, p2t.z)));
    // float deltaZ = gamma(3) * maxZt;
    // // ** Compute  and  terms for triangle  error bounds
    // float maxXt = maxComponent(abs(float3(p0t.x, p1t.x, p2t.x)));
    // float maxYt = maxComponent(abs(float3(p0t.y, p1t.y, p2t.y)));
    // float deltaX = gamma(5) * (maxXt + maxZt);
    // float deltaY = gamma(5) * (maxYt + maxZt);
    // // ** Compute  term for triangle  error bounds
    // float deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt +
    //                         deltaX * maxYt);
    // // ** Compute  term for triangle  error bounds and check t
    // float maxE = maxComponent(abs(float3(e0, e1, e2)));
    // float deltaT = 3 * (gamma(3) * maxE * maxZt + deltaE * maxZt +
    //                         deltaZ * maxE) * abs(invDet);
    // if (t <= deltaT)
    //     return false;

    // Interpolate  parametric coordinates and hit point
    float3 pHit = b0 * p0 + b1 * p1 + b2 * p2;
    barycentricCoord = float3(b0, b1, b2);
    // float3 uvHit = b0 * uv[0] + b1 * uv[1] + b2 * uv[2];
    return true;
}

float rayBoxIntersect(float3 rpos, float3 rdir, float3 vmin, float3 vmax) {
    float t[10];
    t[1] = (vmin.x - rpos.x) / rdir.x;
    t[2] = (vmax.x - rpos.x) / rdir.x;
    t[3] = (vmin.y - rpos.y) / rdir.y;
    t[4] = (vmax.y - rpos.y) / rdir.y;
    t[5] = (vmin.z - rpos.z) / rdir.z;
    t[6] = (vmax.z - rpos.z) / rdir.z;
    t[7] = max(max(min(t[1], t[2]), min(t[3], t[4])), min(t[5], t[6]));
    t[8] = min(min(max(t[1], t[2]), max(t[3], t[4])), max(t[5], t[6]));
    t[9] = (t[8] < 0 || t[7] > t[8]) ? 0 : t[7];
    return t[9];
}

// Converting PDF from Area to Solid angle
float PdfAtoW(float aPdfA, float aDist2, float aCosThere) {
    const float absCosTheta = abs(aCosThere);
    if (absCosTheta < 1e-6f)
        return 0.0;
    return aPdfA * aDist2 / absCosTheta;
}

#endif