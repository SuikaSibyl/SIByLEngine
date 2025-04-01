#ifndef _SRENDERER_LTC_HEADER_
#define _SRENDERER_LTC_HEADER_

Sampler2D<float4> lut_ltc1;
Sampler2D<float4> lut_ltc2;

namespace ltc {

/**
 * Vector form of edge integration.
 * Without project to the plane (dot with the normal),
 * and can be used for proxy sphere clipping.
 * The length of the vector is the form factor of the light polygon,
 * in the direction of this vector.
 */
float3 integrate_edge_vec(float3 v1, float3 v2) {
    // Using built-in acos() function will result flaws
    // Using fitting result for calculating acos()
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;
    float theta_sintheta = (x > 0.0) ? v 
        : (0.5 / sqrt(max(1.0 - x * x, 1e-7)) - v);
    return cross(v1, v2) * theta_sintheta;
}

float3x3 inv_mat_for_ggx(
    float3 N, // normal vector
    float3 V, // view vector
    float roughness,
    out float shadowing,
    out float fresnel
) {
    const float dotNV = clamp(dot(N, V), 0.0f, 1.0f);
    // use roughness and sqrt(1-cos_theta) to sample M_texture
    float2 uv = float2(roughness, sqrt(1.0f - dotNV));
    const float LUT_SIZE = 64.0; // ltc_texture size
    const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
    const float LUT_BIAS = 0.5 / LUT_SIZE;
    uv = uv * LUT_SCALE + LUT_BIAS;
    // get 4 parameters for inverse_M
    float4 t1 = lut_ltc1.Sample(uv);
    // Get 2 parameters for Fresnel calculation
    float4 t2 = lut_ltc2.Sample(uv);
    shadowing = t2.x;
    fresnel = t2.y;
    return float3x3(
        float3(t1.x, 0, t1.y),
        float3(0, 1, 0),
        float3(t1.z, 0, t1.w)
    );
}

/**
 * Evaluate the LTC polygon lighting
 */
float3 evaluate(
    float3 N, // normal
    float3 V, // view
    float3 P, // position
    float3x3 Minv, // inv M
    float3 points[4], // points of the polygon
    bool twoSided //
) {
    // construct orthonormal basis around N
    float3 T1, T2;
    T1 = normalize(V - N * dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    Minv = mul(transpose(float3x3(T1, T2, N)), Minv);
    // Minv = Minv * transpose(mat3(N, T2, T1));

    // polygon (allocate 4 vertices for clipping)
    float3 L[4];
    // transform polygon from LTC back to origin Do (cosine weighted)
    L[0] = mul((points[0] - P), Minv);
    L[1] = mul((points[1] - P), Minv);
    L[2] = mul((points[2] - P), Minv);
    L[3] = mul((points[3] - P), Minv);

    // use tabulated horizon-clipped sphere
    // check if the shading point is behind the light
    float3 dir = points[0] - P; // LTC space
    float3 lightNormal = cross(points[1] - points[0], points[3] - points[0]);
    bool behind = (dot(dir, lightNormal) < 0.0);
    
    // cos weighted space
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);

    // integrate
    float3 vsum = float3(0.0);
    vsum += integrate_edge_vec(L[0], L[1]);
    vsum += integrate_edge_vec(L[1], L[2]);
    vsum += integrate_edge_vec(L[2], L[3]);
    vsum += integrate_edge_vec(L[3], L[0]);
    
    // form factor of the polygon in direction vsum
    float len = length(vsum);

    float z = vsum.z / len;
    if (behind)
        z = -z;

    float2 uv = float2(z * 0.5f + 0.5f, len); // range [0, 1]
    const float LUT_SIZE = 64.0;              // ltc_texture size
    const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
    const float LUT_BIAS = 0.5 / LUT_SIZE;
    uv = uv * LUT_SCALE + LUT_BIAS;
    
    // Fetch the form factor for horizon clipping
    float scale = lut_ltc2.Sample(uv).b;
    // float scale = 1.f;

    float sum = len * scale;
    if (!behind && !twoSided)
        sum = 0.0;

    // Outgoing radiance (solid angle) for the entire polygon
    float3 Lo_i = float3(sum, sum, sum);
    return Lo_i;
}
}

#endif // !_SRENDERER_LTC_HEADER_