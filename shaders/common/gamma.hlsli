#ifndef _SRENDERER_GAMMA_DISTRIBUTION_HEADER_
#define _SRENDERER_GAMMA_DISTRIBUTION_HEADER_

#include "cpp_compatible.hlsli"
#include "geometry.hlsli"
#include "sampling.hlsli"

struct GammaDistribution<let t : int> {
    // techniques for Normal and Gamma Sampling
    // *url: https://wiki.math.uwaterloo.ca/statwiki/index.php?title=techniques_for_Normal_and_Gamma_Sampling
    static float sample(Array<float, t> u) {
        float sum = 0.f;
        for (int i = 0; i < t; i++)
            sum += -log(u[i]);
        return sum;
    }
};

struct BetaDistribution<let alpha : int, let beta : int> {
    static float sample(Array<float, alpha> u_alpha,
                        Array<float, beta> u_beta) {
        float x = GammaDistribution<alpha>.sample(u_alpha);
        float y = GammaDistribution<beta>.sample(u_beta);
        return x / (x + y);
    }

    static float pdf(float x) {
        return pow(x, alpha - 1) * pow(1 - x, beta - 1) * 630;
    }
};

#endif // _SRENDERER_GAMMA_DISTRIBUTION_HEADER_