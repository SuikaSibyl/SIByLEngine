#ifndef _SRENDERER_COMMMON_SAMPLING_HEADER_
#define _SRENDERER_COMMMON_SAMPLING_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"
#include "error.hlsli"
#include "random.hlsli"

/** Uniform sample on 2D disk boundary */
float2 uniform_on_disk(float u) {
    float phi = 2.0f * M_PI * u;
    float sin_phi; float cos_phi;
    sincos(phi, sin_phi, cos_phi);
    return float2(cos_phi, sin_phi);
}

/** Uiform sample on 3D sphere boundary */
float3 UniformOnSphere(in_ref(float2) u) {
    const float z = 1.0f - 2.0f * u.x;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = k_2pi * u[1];
    return float3(r * cos(phi), r * sin(phi), z);
}

float PdfUniformOnSphere() {
    return 1.f / (4.f * k_pi);
}

float3 RandomPointOnHemiphere(in_ref(float2) u) {
    const float z = u.x;
    float r = sqrt(max(0.0f, 1.0f - u.x * u.x));
    float phi = k_2pi * u[1];
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 SampleUniformCone(
    in_ref(float2) rnd,
    float cosThetaMax
) {
    const float cosTheta = (1.f - rnd[0]) + rnd[0] * cosThetaMax;
    const float sinTheta = sqrt(1.f - cosTheta * cosTheta);
    const float phi = rnd[1] * k_2pi;
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta); 
}

float PdfUniformCone(float cosThetaMax) { 
    return 1.f / (k_2pi * (1 - cosThetaMax)); 
}

/**
 * Uniform Random in 3D geometics
 */

float3 randomPointInSphere(inout_ref(RandomSamplerState) rngState) {
    const float theta = 2 * k_pi * SampleUniformFloat(rngState); // Random in [0, 2pi]
    const float u = 2.0 * SampleUniformFloat(rngState) - 1.0;    // Random in [-1, 1]
    const float r = sqrt(1.0 - u * u);
    return float3(r * cos(theta), r * sin(theta), u);
}

[Differentiable]
float3 randomPointInSphere(in_ref(float2) rvec) {
    const float theta = 2 * k_pi * rvec.x; // Random in [0, 2pi]
    const float u = 2.0 * rvec.y - 1.0;    // Random in [-1, 1]
    const float r = sqrt(1.0 - u * u);
    return float3(r * cos(theta), r * sin(theta), u);
}

/**
 * Computes a low discrepancy spherically distributed direction on the unit sphere,
 * for the given index in a set of samples. Each direction is unique in
 * the set, but the set of directions is always the same.
 */
float3 sphericalFibonacci(float sampleIndex, float numSamples) {
    const float b = (sqrt(5.f) * 0.5f + 0.5f) - 1.f;
    const float phi = k_2pi * frac(sampleIndex * b);
    const float cosTheta = 1.f - (2.f * sampleIndex + 1.f) * (1.f / numSamples);
    const float sinTheta = sqrt(saturate(1.f - (cosTheta * cosTheta)));
    return float3((cos(phi) * sinTheta), (sin(phi) * sinTheta), cosTheta);
}

float2 uniformSampleDisk(in_ref(float2) u) {
    float r = sqrt(u.x);
    float theta = 2 * k_pi * u.y;
    return float2(r * cos(theta), r * sin(theta));
}

float2 concentricSampleDisk(in_ref(float2) u) {
    // Map uniform random numbers to[−1, 1]^2
    float2 uOffset = 2.f * u - float2(1, 1);
    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0)
        return float2(0, 0);
    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = k_pi_over_4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = k_pi_over_2 - k_pi_over_4 * (uOffset.x / uOffset.y);
    }
    return r * float2(cos(theta), sin(theta));
}

float3 cosineSampleHemisphere(in_ref(float2) u) {
    const float2 d = concentricSampleDisk(u);
    const float z = sqrt(max(0., 1 - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

float3 cosineSampleHemisphere(in_ref(float2) u, out_ref(float) inv_pdf) {
    const float3 smp = cosineSampleHemisphere(u);
    inv_pdf = k_pi / abs(smp.z);
    return smp;
}

float cosineHemispherePdf(float cosTheta) {
    return cosTheta * k_inv_pi;
}

/**
* Give a random direction in the hemisphere with cosine distribution.
* @param N Normal of the shading surface.
* @param rnd Random number pair in [0, 1].
*/
float3 CosineWeightedHemisphereSample(in_ref(float3) N, in_ref(float2) rnd) {
    const float3 randOffset = randomPointInSphere(rnd);
    return normalize(N + randOffset);
}

[Differentiable]
float3 sample_cos_hemisphere(float2 u) {
    return normalize(float3(0, 0, 1) + randomPointInSphere(u));
}

[Differentiable]
float pdf_cos_hemisphere(float3 d) {
    return max(d[2], 0.f) * k_inv_pi;
}

int sample_discrete<let N : int>(Array<float, N> weights, inout float u) {
    // Compute sum of weights
    float sum_weights = 0;
    for (int i = 0; i < N; ++i)
        sum_weights += weights[i];
    // Compute rescaled u' sample
    float up = u * sum_weights;
    if (up == sum_weights)
        up = next_float_down(up);
    // Find offset in weights corresponding to u'
    int offset = 0;
    float sum = 0;
    while (sum + weights[offset] <= up) {
        sum += weights[offset++];
    }
    u = min((up - sum) / weights[offset], k_one_minus_epsilon);
    return offset;
}

int sample_discrete<let N : int>(Array<float, N> weights, inout float u, inout float pmf) {
    // Compute sum of weights
    float sum_weights = 0;
    for (int i = 0; i < N; ++i)
        sum_weights += weights[i];
    // Compute rescaled u' sample
    float up = u * sum_weights;
    if (up == sum_weights)
        up = next_float_down(up);
    // Find offset in weights corresponding to u'
    int offset = 0;
    float sum = 0;
    while (sum + weights[offset] <= up) {
        sum += weights[offset++];
    }
    // Compute PMF and remapped u value, if necessary
    pmf = weights[offset] / sum_weights;
    u = min((up - sum) / weights[offset], k_one_minus_epsilon);
    return offset;
}

#endif // !_SRENDERER_COMMMON_SAMPLING_HEADER_