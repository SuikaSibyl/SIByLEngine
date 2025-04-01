#ifndef _SRENDERER_VMF_HEADER_
#define _SRENDERER_VMF_HEADER_

#include "cpp_compatible.hlsli"
#include "geometry.hlsli"
#include "sampling.hlsli"

/**
 * Statistics for estimate the parameters of vMF distribution.
 */
struct vMFMLEStatistics {
    float3 sumeightedDirections;
    float sumWeights;
    // constructor for initialize the statistics
    __init(in_ref(float4) statistics) {
        sumeightedDirections = statistics.xyz;
        sumWeights = statistics.w;
    }
    // pack the statistics into a float4
    float4 Pack() {
        return float4(sumeightedDirections, sumWeights);
    }
};

/**
 * One lobe von Mises–Fisher distribution.
 * Implementation follows the note from Wenzel Jakob:
 * "Numerically stable sampling of the von Mises Fisher distribution on S2 (and other tricks)"
 * @url: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
 */
struct vMFDistribution {
    // @ u, mean direction
    float3 u;
    // @ kappa, concentration parameter
    // The greater the value of k, the higher the concentration
    // of the distribution around the mean direction u.
    // k = 0, uniform distribution on sphere.
    float k;
    // Draw a sample from the distribution
    float3 DrawSample(in_ref(float2) rnd) {
        if (k == 0) return UniformOnSphere(rnd);
        const float W = 1 + log(rnd.y + (1 - rnd.y) * exp(-2 * k)) / k;
        const float2 V = float2(cos(2 * k_pi * rnd.x), sin(2 * k_pi * rnd.x));
        const float tmp = sqrt(1 - W * W);
        const float3 dir = float3(tmp * V, W);
        return to_world(createONB(u), dir);
    }
    /**
     * Evaluate the probability density function at a given direction
     * @param direction, the direction to evaluate.
     * @return the probability density at the given direction.
     */
    [Differentiable]
    float Pdf(in_ref(float3) direction) {
        if (k == 0) return 1 / (4 * k_pi);
        const float dotted = dot(direction, u);
        const float e1 = 2 * k_pi * (1 - exp(-2 * k));
        const float e2 = exp(k * (dotted - 1));
        return k / e1 * e2;
    }
    /**
     * Initialize the distribution with a sufficient static vector
     * and the mean cosine value.
     * @param rk: sufficient_static_vector
     * @param rk_: mean_cosine_value
     */
    __init(in_ref(vMFMLEStatistics) statistics) {
        const float len = length(statistics.sumeightedDirections);
        u = statistics.sumeightedDirections / len;
        const float mean_cos = len / statistics.sumWeights;
        const float mean_cos2 = mean_cos * mean_cos;
        k = (3 * mean_cos - mean_cos2 * mean_cos) / (1 - mean_cos2);
        k = clamp(k, 0.f, 10000.f); // avoid numerical instability
    }
    /** Initialize the distribution with a u and k. */
    __init(in_ref(float3) u, in_ref(float) v) {
        this.u = u;
        this.k = v;
    }
};

#endif // !_SRENDERER_VMF_HEADER_