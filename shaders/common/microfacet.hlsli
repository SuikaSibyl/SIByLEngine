#ifndef _SRENDERER_COMMON_MICROFACET_HEADER_
#define _SRENDERER_COMMON_MICROFACET_HEADER_

#include "cpp_compatible.hlsli"
#include "geometry.hlsli"
#include "math.hlsli"
#include "sampling.hlsli"
#include "mapping.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// A Header for utilities and functions related to microfacet models.
// ----------------------------------------------------------------------------------------
// An approach to model the surface reflection and transmission is
// assuming rough surfaces can be modeled as a collection of small microfacets.
// The surface model are often described by:
///////////////////////////////////////////////////////////////////////////////////////////
// Microfacet Distribution Functions (D-term / N-Term / NDF)
// ----------------------------------------------------------------------------------------
// The Normal Distribution Function (NDF) describes the
// differential area of microfacets with the surface normal w_h.
// The grater the variation of microfacet normals, the rougher the surface.
///////////////////////////////////////////////////////////////////////////////////////////
// Masking and Shadowing Functions (G-term)
// ----------------------------------------------------------------------------------------
// The Masking and Shadowing functions describe the probability of
// a microfacet being visible from the view and light directions.
// Invisiblility is caused by either masking or shadowing.
// However, it is also possible that a microfacet is indirectly visible by interreflections.
///////////////////////////////////////////////////////////////////////////////////////////

interface IMicrofacetParameter : IDifferentiable {};

interface IMicrofacetDistribution {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : IMicrofacetParameter;
    
    /**
     * D() describes the differential area of microfacets
     * oriented with the given normal vector wh
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float D(no_diff const float3 wh, TParam param);

    /**
     * G(w) gives the fraction of microfacets in a differential area
     * that are visible from the direction w.
     * @param w: the direction vector given
     */
    [Differentiable]
    static float G1(no_diff const float3 w, TParam param);

    /**
     * G(wo, wi) gives the fraction of microfacets in a differential area
     * that are visible from both directions wo and wi.
     * @param wo: the outgoing direction vector
     * @param wi: the incoming direction vector
     */
    [Differentiable]
    static float G(no_diff const float3 wo, no_diff const float3 wi, TParam param);

    /**
     * Lambda(w) gives the invisible masked microfacet area
     * per visible microfacet area
     * @param w: the direction vector given
     */
    [Differentiable]
    static float Lambda(no_diff const float3 w, TParam param);

    /**
     * Sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_wh_vnormal(no_diff const float3 wo, no_diff const float2 u, TParam param);

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_vnormal(no_diff const float3 wo, no_diff const float3 wh, TParam param);

    /**
     * Sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_wh_normal(no_diff const float3 wo, no_diff const float2 u, TParam param);

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_normal(no_diff const float3 wo, no_diff const float3 wh, TParam param);
};

interface IMicrofacetDerivative {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : IMicrofacetParameter;

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D'(wh),
     * where hereby only considering the positive half.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_pos_wh(no_diff const float3 wo, no_diff const float2 u, TParam param);

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D'(wh),
     * where hereby only considering the negative half.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_neg_wh(no_diff const float3 wo, no_diff const float2 u, TParam param);

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution derivative D'(wh),
     * where hereby only considering the positive half.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_pos(no_diff const float3 wo, no_diff const float3 wh, TParam param);

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution derivative D'(wh),
     * where hereby only considering the positive half.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_neg(no_diff const float3 wo, no_diff const float3 wh, TParam param);
}

// struct IsotropicBeckmannDistribution : IMicrofacetDistribution {
//     // Alpha parameter, related to roughness
//     // see "RoughnessToAlpha" function
//     float alpha;

//     __init(float _alpha) { alpha = _alpha; }

//     float D(float3 wh) {
//         const float alpha2 = alpha * alpha;
//         const float tan2Theta = theta_phi_coord::Tan2Theta(wh);
//         if (isinf(tan2Theta)) return 0.;
//         const float cos4Theta = theta_phi_coord::Cos2Theta(wh)
//             * theta_phi_coord::Cos2Theta(wh);
//         return exp(-tan2Theta / (alpha2)) /
//                (k_pi * alpha2 * cos4Theta);
//     }

//     /** G(w) gives the fraction of microfacets in a differential area
//         * that are visible from the direction w.
//         * @param w: the direction vector given
//         */
//     float G1(float3 w) { return 1. / (1. + Lambda(w)); }

//     /**
//         * G(wo, wi) gives the fraction of microfacets in a differential area
//         * that are visible from both directions wo and wi.
//         * @param wo: the outgoing direction vector
//         * @param wi: the incoming direction vector
//         */
//     float G(float3 wo, float3 wi) {
//         return 1. / (1. + Lambda(wo) + Lambda(wi));
//     }
    
//     float Lambda(float3 w) {
//         float absTanTheta = abs(theta_phi_coord::TanTheta(w));
//         if (isinf(absTanTheta)) return 0.;
//         // Compute alpha for direction w
//         float a = 1 / (alpha * absTanTheta);
//         if (a >= 1.6f) return 0;
//         return (1 - 1.259f * a + 0.396f * a * a) /
//                (3.535f * a + 2.181f * a * a);
//     }

//     /**
//      * Sampling a microfacet orientation by sampling from
//      * the normal distribution D(wh) directly.
//      * @param wo: the outgoing direction vector
//      * @param u: a pair of random number
//      */
//     float3 sample_wh(float3 wo, float2 u) {
//         // Sample full distribution of normals for Beckmann distribution
//         // Compute tan^2theta and phi for Beckmann distribution sample
//         float tan2Theta;
//         float phi;

//         float logSample = log(1 - u[0]);
//         if (isinf(logSample)) logSample = 0;
//         tan2Theta = -alpha * alpha * logSample;
//         phi = u[1] * 2 * k_pi;

//         // Map sampled Beckmann angles to normal direction wh
//         float cosTheta = 1 / sqrt(1 + tan2Theta);
//         float sinTheta = sqrt(max(0., 1 - cosTheta * cosTheta));
//         float3 wh = theta_phi_coord::SphericalDirection(sinTheta, cosTheta, phi);
//         if (!(wo.z * wh.z > 0)) wh = -wh;
//         return wh;
//     }

//     float pdf(float3 wo, float3 wh) {
//         return D(wh) * theta_phi_coord::AbsCosTheta(wh);
//     }
// };

// struct AnisotropicBeckmannDistribution : IMicrofacetDistribution {
//     // Alpha parameter, related to roughness
//     // see "RoughnessToAlpha" function
//     float alpha_x; // alpha in x direction
//     float alpha_y; // alpha in y direction

//     __init(float _alpha_x, float _alpha_y) {
//         alpha_x = _alpha_x;
//         alpha_y = _alpha_y;
//     }

//     float D(float3 wh) {
//         const float tan2Theta = theta_phi_coord::Tan2Theta(wh);
//         if (isinf(tan2Theta)) return 0.;
//         const float cos4Theta = theta_phi_coord::Cos2Theta(wh)
//             * theta_phi_coord::Cos2Theta(wh);
//         return exp(-tan2Theta *
//                    (theta_phi_coord::Cos2Phi(wh) / (alpha_x * alpha_x) +
//                     theta_phi_coord::Sin2Phi(wh) / (alpha_y * alpha_y))) /
//                (k_pi * alpha_x * alpha_y * cos4Theta);
//     }

//     /** G(w) gives the fraction of microfacets in a differential area
//      * that are visible from the direction w.
//      * @param w: the direction vector given
//      */
//     float G1(float3 w) { return 1. / (1. + Lambda(w)); }

//     /**
//      * G(wo, wi) gives the fraction of microfacets in a differential area
//      * that are visible from both directions wo and wi.
//      * @param wo: the outgoing direction vector
//      * @param wi: the incoming direction vector
//      */
//     float G(float3 wo, float3 wi) {
//         return 1. / (1. + Lambda(wo) + Lambda(wi));
//     }

//     float Lambda(float3 w) {
//         float absTanTheta = abs(theta_phi_coord::TanTheta(w));
//         if (isinf(absTanTheta)) return 0.;
//         // Compute alpha for direction w
//         float alpha = sqrt(theta_phi_coord::Cos2Phi(w) * alpha_x * alpha_x 
//             + theta_phi_coord::Sin2Phi(w) * alpha_y * alpha_y);
//         float a = 1 / (alpha * absTanTheta);
//         if (a >= 1.6f) return 0;
//         return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
//     }

//     /**
//      * Sampling a microfacet orientation by sampling from
//      * the normal distribution D(wh) directly.
//      * @param wo: the outgoing direction vector
//      * @param u: a pair of random number
//      */
//     float3 sample_wh(float3 wo, float2 u) {
//         // Sample full distribution of normals for Beckmann distribution
//         // Compute tan^2theta and phi for Beckmann distribution sample

//         // Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
//         float tan2Theta;
//         float phi;
//         if (alpha_x == alpha_y) {
//             float logSample = log(1 - u[0]);
//             tan2Theta = -alpha_x * alpha_x * logSample;
//             phi = u[1] * 2 * k_pi;
//         } else {
//             // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
//             // distribution
//             float logSample = log(1 - u[0]);
//             phi = atan(alpha_y / alpha_x *
//                        tan(2 * k_pi * u[1] + 0.5f * k_pi));
//             if (u[1] > 0.5f) phi += k_pi;
//             float sinPhi = sin(phi);
//             float cosPhi = cos(phi);
//             float alphax2 = alpha_x * alpha_x;
//             float alphay2 = alpha_y * alpha_y;
//             tan2Theta = -logSample /
//                         (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
//         }

//         // Map sampled Beckmann angles to normal direction _wh_
//         float cosTheta = 1 / sqrt(1 + tan2Theta);
//         float sinTheta = sqrt(max(0.f, 1 - cosTheta * cosTheta));
//         float3 wh = theta_phi_coord::SphericalDirection(sinTheta, cosTheta, phi);
//         if (!(wo.z * wh.z > 0)) wh = -wh;
//         return wh;
//     }

//     float pdf(float3 wo, float3 wh) {
//         return D(wh) * theta_phi_coord::AbsCosTheta(wh);
//     }
// };

struct IsotropicTrowbridgeReitzParameter : IMicrofacetParameter {
    float alpha; // closely related to roughness
    __init(float _alpha) { alpha = _alpha; }
};

struct IsotropicTrowbridgeReitzDistribution : IMicrofacetDistribution {
    // Specify that the associated `TParam` type is `IsotropicTrowbridgeReitzParameter`.
    typedef IsotropicTrowbridgeReitzParameter TParam;

    /**
     * D() describes the differential area of microfacets
     * oriented with the given normal vector wh
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float D(no_diff const float3 wh, IsotropicTrowbridgeReitzParameter param) {
        float tan2Theta = theta_phi_coord::Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.;
        float cos4Theta = theta_phi_coord::Cos2Theta(wh) 
                        * theta_phi_coord::Cos2Theta(wh);
        float e = tan2Theta / (param.alpha * param.alpha);
        return 1. / (k_pi * param.alpha * param.alpha * cos4Theta * (1 + e) * (1 + e));
    }
    
    /** G(w) gives the fraction of microfacets in a differential area
     * that are visible from the direction w.
     * @param w: the direction vector given
     */
    [Differentiable]
    static float G1(no_diff const float3 w, IsotropicTrowbridgeReitzParameter param) {
        return 1. / (1. + Lambda(w, param)); 
    }

    /**
     * G(wo, wi) gives the fraction of microfacets in a differential area
     * that are visible from both directions wo and wi.
     * @param wo: the outgoing direction vector
     * @param wi: the incoming direction vector
     */
    [Differentiable]
    static float G(no_diff const float3 wo, no_diff const float3 wi,
            IsotropicTrowbridgeReitzParameter param) {
        return 1. / (1. + Lambda(wo, param) + Lambda(wi, param));
    }

    /**
     * Lambda(w) for Anisotropic Trowbridge-Reitz microfacet.
     * Lambda(w) gives the invisible masked microfacet area
     * per visible microfacet area
     * @param w: the direction vector given
     */
    [Differentiable]
    static float Lambda(no_diff const float3 w, IsotropicTrowbridgeReitzParameter param) {
        float absTanTheta = abs(theta_phi_coord::TanTheta(w));
        if (isinf(absTanTheta)) return 0.;
        // Compute _alpha_ for direction _w_
        float alpha2Tan2Theta = (param.alpha * absTanTheta) * (param.alpha * absTanTheta);
        return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
    }

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_wh_vnormal(no_diff const float3 wo, no_diff const float2 u,
                                    IsotropicTrowbridgeReitzParameter param) {
        // Transform w to hemispherical configuration
        float3 wh = normalize(float3(param.alpha * wo.x, param.alpha * wo.y, wo.z));
        if (wh.z < 0) wh = -wh;
        // Find orthonormal basis for visible normal sampling
        float3 T1 = (wh.z < 0.99999f) ? normalize(cross(float3(0, 0, 1), wh))
                                      : float3(1, 0, 0);
        float3 T2 = cross(wh, T1);
        // Generate uniformly distributed points on the unit disk
        float2 p = mapping::square_to_disk_polar(u);
        // Warp hemispherical projection for visible normal sampling
        float h = sqrt(1 - sqr(p.x));
        p.y = lerp(h, p.y, (1 + wh.z) / 2);
        // Reproject to hemisphere and transform normal to ellipsoid configuration
        float pz = sqrt(max(0, 1 - length_squared(float2(p))));
        float3 nh = p.x * T1 + p.y * T2 + pz * wh;
        return normalize(float3(param.alpha * nh.x, param.alpha * nh.y, max(1e-6f, nh.z)));
    }

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_vnormal(no_diff const float3 wo, no_diff const float3 wh,
                     IsotropicTrowbridgeReitzParameter param) {
        return D(wh, param) * G1(wo, param) * abs(dot(wo, wh)) /
               theta_phi_coord::AbsCosTheta(wo);
    }

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_wh_normal(no_diff const float3 wo, no_diff const float2 u,
                            IsotropicTrowbridgeReitzParameter param) {
        float phi = (2 * k_pi) * u[1];
        float tanTheta2 = param.alpha * param.alpha * u[0] / (1.0f - u[0]);
        float cosTheta = 1 / sqrt(1 + tanTheta2);
        float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        float3 wh = theta_phi_coord::SphericalDirection(sinTheta, cosTheta, phi);
        if (!theta_phi_coord::SameHemisphere(wo, wh)) wh = -wh;
        return wh;
    }

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_normal(no_diff const float3 wo, no_diff const float3 wh,
                     IsotropicTrowbridgeReitzParameter param) {
        return D(wh, param) * theta_phi_coord::AbsCosTheta(wh);
    }
    
    /**
     * Sample the Trowbridge-Reitz microfacet normal distribution function.
     * @url: https://github.com/mmp/pbrt-v3/blob/master/src/core/microfacet.cpp#L284
     */
    [Differentiable]
    static void TrowbridgeReitzSample11(
        float cosTheta, float U1, float U2,
        inout float slope_x, inout float slope_y) {
        // special case (normal incidence)
        if (cosTheta > .9999) {
            float r = sqrt(U1 / (1 - U1));
            float phi = k_2pi * U2;
            slope_x = r * cos(phi);
            slope_y = r * sin(phi);
            return;
        }
        float sinTheta = sqrt(max(0., 1. - cosTheta * cosTheta));
        float tanTheta = sinTheta / cosTheta;
        float a = 1 / tanTheta;
        float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));
        // sample slope_x
        float A = 2 * U1 / G1 - 1;
        float tmp = 1.f / (A * A - 1.f);
        if (tmp > 1e10) tmp = 1e10;
        float B = tanTheta;
        float D = sqrt(max(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.));
        float slope_x_1 = B * tmp - D;
        float slope_x_2 = B * tmp + D;
        slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;
        // sample slope_y
        float S;
        if (U2 > 0.5f) {
            S = 1.f;
            U2 = 2.f * (U2 - .5f);
        } else {
            S = -1.f;
            U2 = 2.f * (.5f - U2);
        }
        float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
                  (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        slope_y = S * z * sqrt(1.f + slope_x * slope_x);
    }
    
    /**
     * Sample the Trowbridge-Reitz microfacet normal distribution function.
     * @url: github.com/mmp/pbrt-v3/blob/master/src/core/microfacet.cpp#L284
     */
    [Differentiable]
    static float3 TrowbridgeReitzSample(
        float3 wi, float alpha, float U1, float U2) {
        // 1. stretch wi
        float3 wiStretched = normalize(float3(alpha * wi.x, alpha * wi.y, wi.z));
        // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
        float slope_x; float slope_y;
        TrowbridgeReitzSample11(theta_phi_coord::CosTheta(wiStretched), U1, U2, slope_x, slope_y);
        // 3. rotate
        float tmp = theta_phi_coord::CosPhi(wiStretched) * slope_x 
                  - theta_phi_coord::SinPhi(wiStretched) * slope_y;
        slope_y = theta_phi_coord::SinPhi(wiStretched) * slope_x 
                + theta_phi_coord::CosPhi(wiStretched) * slope_y;
        slope_x = tmp;
        // 4. unstretch
        slope_x = alpha * slope_x;
        slope_y = alpha * slope_y;
        // 5. compute normal
        return normalize(float3(-slope_x, -slope_y, 1.));
    }

    static bool effectively_smooth(float alpha) {
        return alpha < 1e-3f;
    }
};

struct IsotropicTrowbridgeReitzDerivative : IMicrofacetDerivative {
    // Associated a parameter type for each microfacet distribution
    typedef IsotropicTrowbridgeReitzParameter TParam;

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D'(wh),
     * where hereby only considering the positive half.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_pos_wh(no_diff const float3 wo, no_diff const float2 u, TParam param) {
        const float C = param.alpha * sqrt(-2. / (sqrt(u.x) - 1.) - 1.);
        const float theta = atan(C);
        const float phi = u.y * k_2pi;
        float st; float ct;
        float sp; float cp;
        sincos(theta, st, ct);
        sincos(phi, sp, cp);
        return float3(st * cp, st * sp, ct);
    }

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D'(wh),
     * where hereby only considering the negative half.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_neg_wh(no_diff const float3 wo, no_diff const float2 u, TParam param) {
        const float x = u.x;
        const float sqrtomx = sqrt(1 - x);
        const float alpha2 = param.alpha * param.alpha;
        const float alpha4 = alpha2 * alpha2;
        const float tmp = x * (2 - 2 * sqrtomx + 2 * alpha4 * (1 + sqrtomx) - x * (alpha2 - 1) * (alpha2 - 1));
        const float numerator = 2 * param.alpha * sqrt(tmp);
        const float denominator = x + 4 * alpha2 * sqrtomx - x * alpha4;
        const float theta = 0.5 * atan(numerator / denominator);
        const float phi = u.y * k_2pi;
        return theta_phi_coord::SphericalDirection(sin(theta), cos(theta), phi);
    }
    
    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution derivative D'(wh),
     * where hereby only considering the positive half.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_pos(no_diff const float3 wo, no_diff const float3 wh, IsotropicTrowbridgeReitzParameter param) {
        return max(pdf_unpostivized(wh, param), 0.);
    }
    
    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution derivative D'(wh),
     * where hereby only considering the positive half.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_neg(no_diff const float3 wo, no_diff const float3 wh, IsotropicTrowbridgeReitzParameter param) {
        return -min(pdf_unpostivized(wh, param), 0.);
    }

    [Differentiable]
    static float pdf_unpostivized(no_diff const float3 wh, IsotropicTrowbridgeReitzParameter param) {
        const float ct = theta_phi_coord::CosTheta(wh);
        const float ct_2 = ct * ct;
        const float ct_3 = ct * ct_2;
        const float tt_2 = (wh.x * wh.x + wh.y * wh.y) / ct_2;
        const float alpha_2 = param.alpha * param.alpha;

        const float numerator = 4.f * alpha_2 * (tt_2 - alpha_2);
        const float temp = alpha_2 + tt_2;
        const float denominator = k_pi * ct_3 * temp * temp * temp;
        return numerator / denominator;
    }
}

struct AnisotropicTrowbridgeReitzParameter : IMicrofacetParameter {
    float alpha_x; // closely related to roughness, x direction
    float alpha_y; // closely related to roughness, y direction
    __init(float _alpha_x, float _alpha_y) { 
        alpha_x = _alpha_x;
        alpha_y = _alpha_y;
    }
};

struct AnisotropicTrowbridgeReitzDistribution : IMicrofacetDistribution {
    // Specify that the associated `TParam` type is `AnisotropicTrowbridgeReitzParameter`.
    typedef AnisotropicTrowbridgeReitzParameter TParam;

    /**
     * D() describes the differential area of microfacets
     * oriented with the given normal vector wh
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float D(no_diff const float3 wh, AnisotropicTrowbridgeReitzParameter param) {
        float tan2Theta = theta_phi_coord::Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0;
        float cos4Theta = sqr(theta_phi_coord::Cos2Theta(wh));
        float e = tan2Theta * (sqr(theta_phi_coord::CosPhi(wh) / param.alpha_x) +
                               sqr(theta_phi_coord::SinPhi(wh) / param.alpha_y));
        return 1 / (k_pi * param.alpha_x * param.alpha_y * cos4Theta * sqr(1 + e));
    }

    /** G(w) gives the fraction of microfacets in a differential area
     * that are visible from the direction w.
     * @param w: the direction vector given
     */
    [Differentiable]
    static float G1(no_diff const float3 w, AnisotropicTrowbridgeReitzParameter param) {
        return 1. / (1. + Lambda(w, param));
    }

    /**
     * G(wo, wi) gives the fraction of microfacets in a differential area
     * that are visible from both directions wo and wi.
     * @param wo: the outgoing direction vector
     * @param wi: the incoming direction vector
     */
    [Differentiable]
    static float G(no_diff const float3 wo, no_diff const float3 wi,
                   AnisotropicTrowbridgeReitzParameter param) {
        return 1. / (1. + Lambda(wo, param) + Lambda(wi, param));
    }

    /**
     * Lambda(w) for Anisotropic Trowbridge-Reitz microfacet.
     * Lambda(w) gives the invisible masked microfacet area
     * per visible microfacet area
     * @param w: the direction vector given
     */
    [Differentiable]
    static float Lambda(no_diff const float3 w, AnisotropicTrowbridgeReitzParameter param) {
        float tan2Theta = theta_phi_coord::Tan2Theta(w);
        if (isinf(tan2Theta)) return 0;
        float alpha2 = sqr(theta_phi_coord::CosPhi(w) * param.alpha_x) 
                     + sqr(theta_phi_coord::SinPhi(w) * param.alpha_y);
        return (sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_wh_vnormal(no_diff const float3 wo, no_diff const float2 u,
                                    AnisotropicTrowbridgeReitzParameter param) {
        // Transform w to hemispherical configuration
        float3 wh = normalize(float3(param.alpha_x * wo.x, param.alpha_y * wo.y, wo.z));
        if (wh.z < 0) wh = -wh;
        // Find orthonormal basis for visible normal sampling
        float3 T1 = (wh.z < 0.99999f) ? normalize(cross(float3(0, 0, 1), wh))
                                      : float3(1, 0, 0);
        float3 T2 = cross(wh, T1);
        // Generate uniformly distributed points on the unit disk
        float2 p = mapping::square_to_disk_polar(u);
        // Warp hemispherical projection for visible normal sampling
        float h = sqrt(1 - sqr(p.x));
        p.y = lerp(h, p.y, (1 + wh.z) / 2);
        // Reproject to hemisphere and transform normal to ellipsoid configuration
        float pz = sqrt(max(0, 1 - length_squared(float2(p))));
        float3 nh = p.x * T1 + p.y * T2 + pz * wh;
        return normalize(float3(param.alpha_x * nh.x, param.alpha_y * nh.y, max(1e-6f, nh.z)));
    }

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_vnormal(no_diff const float3 wo, no_diff const float3 wh,
                             AnisotropicTrowbridgeReitzParameter param) {
        return D(wh, param) * G1(wo, param) * abs(dot(wo, wh)) /
               theta_phi_coord::AbsCosTheta(wo);
    }

    /**
     * Sampling a microfacet orientation by sampling from
     * the normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param u: a pair of random number
     */
    [Differentiable]
    static float3 sample_wh_normal(no_diff const float3 wo, no_diff const float2 u,
                                   AnisotropicTrowbridgeReitzParameter param) {
        float phi = (2 * k_pi) * u[1];
        float tanTheta2 = param.alpha_x * param.alpha_y * u[0] / (1.0f - u[0]);
        float cosTheta = 1 / sqrt(1 + tanTheta2);
        float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        float3 wh = theta_phi_coord::SphericalDirection(sinTheta, cosTheta, phi);
        if (!theta_phi_coord::SameHemisphere(wo, wh)) wh = -wh;
        return wh;
    }

    /**
     * PDF of sampling a microfacet orientation by sampling from
     * the (visible) normal distribution D(wh) directly.
     * @param wo: the outgoing direction vector
     * @param wh: the half vector of the microfacet
     */
    [Differentiable]
    static float pdf_normal(no_diff const float3 wo, no_diff const float3 wh,
                            AnisotropicTrowbridgeReitzParameter param) {
        return D(wh, param) * theta_phi_coord::AbsCosTheta(wh);
    }
    
    static bool effectively_smooth(AnisotropicTrowbridgeReitzParameter params) {
        return max(params.alpha_x, params.alpha_y) < 1e-3f;
    }
};


/**
 * Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @url: https://www.disneyanimation.com/publications/physically-based-shading-at-disney
 * @param NdotH The cosine of the angle between the half vector and normal vector.
 * @param roughness The roughness of the surface.
 */
[Differentiable]
float GTR2_NDF(float n_dot_h, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (k_pi * t * t);
}

/**
 * Anisotropic Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @url: https://www.disneyanimation.com/publications/physically-based-shading-at-disney
 * @param NdotH The cosine of the angle between the half vector and normal vector.
 * @param HDotX The cosine of the angle between the half vector and tangent vector.
 * @param HDotY The cosine of the angle between the half vector and bitangent vector.
 * @param ax The roughness of the surface in x direction.
 * @param ay The roughness of the surface in y direction.
 */
float AnisotropicGTR2_NDF(float NDotH, float HDotX, float HDotY, float ax, float ay) {
    const float a = HDotX / ax;
    const float b = HDotY / ay;
    const float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (k_pi * ax * ay * c * c);
}

/**
 * GGX Normal Distribution Function.
 * Which is equivalent to the Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @param NdotH The cosine of the angle between the half vector and normal vector.
 * @param roughness The roughness of the surface.
 */
float GGX_NDF(float n_dot_h, float roughness) {
    return GTR2_NDF(n_dot_h, roughness);
}

/**
 * Anisotropic GGX Normal Distribution Function.
 * Which is equivalent to the Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @param H The half vector in world space.
 * @param frame The shading frame.
 * @param ax The roughness of the surface in x direction.
 * @param ay The roughness of the surface in y direction.
 */
float AnisotropicGGX_NDF(in_ref(float3) H, in_ref(float3x3) frame, float ax, float ay) {
    const float3 hl = to_local(frame, H);
    return AnisotropicGTR2_NDF(hl.z, hl.x, hl.y, ax, ay);
}

/**
 * Cast a [0, 1] roughness value to alpha value
 * used in microfacet models.
 */
float RoughnessToAlpha(float roughness) {
    roughness = max(roughness, 1e-3);
    float x = log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x +
           0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}


/**
 * Isotropic GGX Masking Term.
 * @url: https://jcgt.org/published/0003/02/03/paper.pdf
 * @param v_local The vector in local shading frame.
 * @param roughness The roughness of the surface.
 */
[Differentiable]
float IsotropicGGX_Masking(no_diff float3 v_local, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float3 v2 = v_local * v_local;
    const float Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

/**
 * Anisotropic GGX Masking Term.
 * @url: https://jcgt.org/published/0003/02/03/paper.pdf
 * @param v_local The vector in local shading frame.
 * @param ax The roughness of the surface in x direction.
 * @param ay The roughness of the surface in y direction.
 */
float AnisotropicGGX_Masking(in_ref(float3) v_local, float ax, float ay) {
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;
    const float3 v2 = v_local * v_local;
    const float Lambda = (-1 + sqrt(1 + (v2.x * ax2 + v2.y * ay2) / v2.z)) / 2;
    return float(1) / (1 + Lambda);
}

/**
 * Sample the GGX distribution of visible normals.
 * See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
 * @url: https://jcgt.org/published/0007/04/01/
 */
float3 SampleVisibleNormals(
    in_ref(float3) local_dir_in,
    float alpha,
    in_ref(float2) rnd_param
) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    float negative = 1;
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
        negative = -1;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(float3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * k_pi * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = float3(t1, t2, sqrt(max(0, 1 - t1 * t1 - t2 * t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    float3x3 hemi_frame = createFrame(hemi_dir_in);
    float3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return negative * normalize(float3(alpha * hemi_N.x, alpha * hemi_N.y, max(0, hemi_N.z)));
}

///////////////////////////////////////////////////////////////////////////////////////////
// Fresnel reflectance, and Fresnel term. (F-term)
// ----------------------------------------------------------------------------------------
// The Fresnel reflectance describes the ratio of reflection to transmission.
// We start with the Fresnel reflectance of a perfectly smooth surfaces,
// which is abstracted as the IFresnel interface, and have different variants
// among several important classes of materials:
// - DielectricFresnel: for dielectric materials
// - ConductorFresnel: for conductive materials
//
// The code below references the following resources:
// @url: https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
// @url: https://www.pbr-book.org/3ed-2018/Reflection_Models/Fresnel_Incidence_Effects
///////////////////////////////////////////////////////////////////////////////////////////

[Differentiable]
float FresnelDielectric(no_diff float cosTheta_i, float eta) {
    cosTheta_i = clamp(cosTheta_i, -1, 1);
    // Potentially flip interface orientation for Fresnel equations
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }
    // Compute cosTheta_t for Fresnel equations using Snell’s law
    float sin2Theta_i = 1 - sqr(cosTheta_i);
    float sin2Theta_t = sin2Theta_i / sqr(eta);
    if (sin2Theta_t >= 1) return 1.f;
    float cosTheta_t = safe_sqrt(1 - sin2Theta_t);
    float r_parl = (eta * cosTheta_i - cosTheta_t) /
                   (eta * cosTheta_i + cosTheta_t);
    float r_perp = (cosTheta_i - eta * cosTheta_t) /
                   (cosTheta_i + eta * cosTheta_t);
    return (sqr(r_parl) + sqr(r_perp)) / 2;
}

[Differentiable]
float FresnelComplex(float cosTheta_i, complex eta) {
    cosTheta_i = clamp(cosTheta_i, 0, 1);
    // Compute complex cosθt for Fresnel equations using Snell's law
    float sin2Theta_i = 1 - sqr(cosTheta_i);
    complex sin2Theta_t = sin2Theta_i / sqr(eta);
    complex cosTheta_t = sqrt(1 - sin2Theta_t);
    complex r_parl = (eta * cosTheta_i - cosTheta_t) /
                     (eta * cosTheta_i + cosTheta_t);
    complex r_perp = (cosTheta_i - eta * cosTheta_t) /
                     (cosTheta_i + eta * cosTheta_t);
    return (norm(r_parl) + norm(r_perp)) / 2;
}

[Differentiable]
float3 FresnelComplex(float cosTheta_i, complex3 eta) {
    cosTheta_i = clamp(cosTheta_i, 0, 1);
    // Compute complex cosθt for Fresnel equations using Snell's law
    float sin2Theta_i = 1 - sqr(cosTheta_i);
    complex3 sin2Theta_t = complex3(sin2Theta_i) / sqr(eta);
    complex3 cosTheta_t = sqrt(complex3(1) - sin2Theta_t);
    complex3 r_parl = (eta * complex3(cosTheta_i) - cosTheta_t) /
                      (eta * complex3(cosTheta_i) + cosTheta_t);
    complex3 r_perp = (complex3(cosTheta_i) - eta * cosTheta_t) /
                      (complex3(cosTheta_i) + eta * cosTheta_t);
    return (norm(r_parl) + norm(r_perp)) / 2;
}

/**
 * @param n: The surface normal.
 * @param eta: The relative index of refraction;
 *  it specifies the IOR ratio of the object interior relative to the outside,
 *  as indicated by the surface normal n.
 */
float3 safe_refract(float3 wi, float3 n, inout float eta) {
    float cosTheta_i = dot(n, wi);
    // Potentially flip interface orientation for Snell's law
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }
    // Compute cosθ_t using Snell's law
    float sin2Theta_i = max(0, 1 - sqr(cosTheta_i));
    float sin2Theta_t = sin2Theta_i / sqr(eta);
    // Handle total internal reflection case
    if (sin2Theta_t >= 1) return float3(0);
    float cosTheta_t = safe_sqrt(1 - sin2Theta_t);
    // return refracted direction
    return -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
}

interface IFresnel {
    float3 eval(float cosThetaI);
}

struct DielectricFresnel : IFresnel {
    float etaI;
    float etaT;

    __init(float _etaI, float _etaT) {
        etaI = _etaI;
        etaT = _etaT;
    }

    static float3 eval_explicit(
        float cosThetaI, float etaI, float etaT) {
        cosThetaI = clamp(cosThetaI, -1, 1);
        // Potentially swap indices of refraction
        bool entering = cosThetaI > 0.f;
        if (!entering) {
            swap(etaI, etaT);
            cosThetaI = abs(cosThetaI);
        }
        // Compute cosThetaT using Snell's law
        float sinThetaI = sqrt(max(0.0, 1 - cosThetaI * cosThetaI));
        float sinThetaT = etaI / etaT * sinThetaI;
        // Handle total internal reflection
        if (sinThetaT >= 1)
            return 1;
        float cosThetaT = sqrt(max(0.0, 1 - sinThetaT * sinThetaT));
        float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                      ((etaT * cosThetaI) + (etaI * cosThetaT));
        float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                      ((etaI * cosThetaI) + (etaT * cosThetaT));
        return (Rparl * Rparl + Rperp * Rperp) / 2;
    }

    float3 eval(float cosThetaI) {
        return eval_explicit(cosThetaI, etaI, etaT);
    }
};

struct FresnelFresnel : IFresnel {
    float3 etaI;
    float3 etaT;
    float3 k;

    __init(float3 _etaI, float3 _etaT, float3 _k) {
        etaI = _etaI;
        etaT = _etaT;
        k = _k;
    }

    // @url: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    static float3 eval_explicit(
        float cosThetaI, float3 etai, float3 etat, float3 k) {
        float cosThetaI = clamp(cosThetaI, -1, 1);
        float3 eta = etat / etai;
        float3 etak = k / etai;
        
        float cosThetaI2 = cosThetaI * cosThetaI;
        float sinThetaI2 = 1. - cosThetaI2;
        float3 eta2 = eta * eta;
        float3 etak2 = etak * etak;
        
        float3 t0 = eta2 - etak2 - sinThetaI2;
        float3 a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
        float3 t1 = a2plusb2 + cosThetaI2;
        float3 a = sqrt(0.5f * (a2plusb2 + t0));
        float3 t2 = 2. * cosThetaI * a;
        float3 Rs = (t1 - t2) / (t1 + t2);
        
        float3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
        float3 t4 = t2 * sinThetaI2;
        float3 Rp = Rs * (t3 - t4) / (t3 + t4);
        
        return 0.5 * (Rp + Rs);    
    }

    float3 eval(float cosThetaI) {
        return eval_explicit(cosThetaI, etaI, etaT, k);
    }
};

/**
 * The Schlick Fresnel approximation is:
 *   R = R(0) + (1 - R(0)) (1 - cos theta)^5
 * where R(0) is the reflectance at normal indicence.
 * @param VdotH The cosine of the angle between the half vector and view vector.
 * @return The schlick weight, a.k.a., (1 - (h*v))^5
 */
float SchlickWeight(float VdotH) {
    const float m = clamp(1.0 - VdotH, 0.0, 1.0);
    const float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

/**
 * The Schlick Fresnel approximation is:
 *   R = R(0) + (1 - R(0)) (1 - cos theta)^5
 * where R(0) is the reflectance at normal indicence.
 * @param VdotH The cosine of the angle between the half vector and view vector.
 * @return The Schlick Fresnel approximation result.
 */
float SchlickFresnel(float F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }
float2 SchlickFresnel(in_ref(float2) F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }
float3 SchlickFresnel(in_ref(float3) F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }
float4 SchlickFresnel(in_ref(float4) F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }

// This function is meant to be differentiated only w.r.t x for Newton-Rhapson
[Differentiable]
float base_dx_phi_cdf(float x, 
    no_diff float u, no_diff float a, no_diff float b, 
    no_diff float k1, no_diff float k2, no_diff float k3) {
    float t1 = -u;
    float t2 = k1 * atan(k2 * tan(x)) / k3;
    float t3 = k1 * sin(2.0 * x) / ((a + 1.0) * (a + b + 2.0 + (a - b) * cos(2.0 * x)));
    return t1 + t2 + t3;
}

// This function is meant to be differentiated only w.r.t x for Newton-Rhapson
[Differentiable]
float base_dy_phi_cdf(float x, no_diff float u, no_diff float a, no_diff float b) {
    float c = 4.0 * safe_sqrt(a + 1.0) * pow(b + 1.0, 1.5) * k_inv_pi;
    float t1 = -u;
    float t2 = c * 0.5 * atan(safe_sqrt((b + 1.0) / (a + 1.0)) * tan(x)) / (safe_sqrt(a + 1) * pow(b + 1.0, 1.5));
    float t3 = -c * 0.5 * sin(2.0 * x) / ((b + 1.0) * (a + b + 2.0 + (a - b) * cos(2.0 * x)));
    return t1 + t2 + t3;
}

// This uses Cem Yuksels hybrid Newton + Bisection method outlined here (same for theta) sampling
// http://www.cemyuksel.com/research/polynomials/polynomial_roots_hpg2022.pdf
inline float base_dxy_phi_bisection_search(float x_s, float x_e, float u, float a, float b, bool deriv_nu) {
    const int BISECTION_SEARCH_MAX_DEPTH = 64;
    const float CDF_INV_TOL = 1e-6;
    float x_r = (x_s.x + x_e.x) / 2.0;
    float f_x_s = 0, f_x_e = 0;
    const float k1 = 2.0 * safe_sqrt(b + 1.0) * pow(a + 1.0, 1.5) * k_inv_pi;
    const float k2 = safe_sqrt((b + 1.0) / (a + 1.0));
    const float k3 = (safe_sqrt(b + 1) * pow(a + 1.0, 1.5));
    f_x_s = deriv_nu ? base_dx_phi_cdf(x_s, u, a, b, k1, k2, k3) : base_dy_phi_cdf(x_s, u, a, b);
    f_x_e = deriv_nu ? base_dx_phi_cdf(x_e, u, a, b, k1, k2, k3) : base_dy_phi_cdf(x_e, u, a, b);
    for (int i = 0; i < BISECTION_SEARCH_MAX_DEPTH; i++) {
        DifferentialPair<float> x_r_pair = diffPair(x_r);
        DifferentialPair<float >f_x_r = deriv_nu 
            ? fwd_diff(base_dx_phi_cdf)(x_r_pair, u, a, b, k1, k2, k3)
            : fwd_diff(base_dy_phi_cdf)(x_r_pair, u, a, b);
        if (abs(f_x_r.p) < CDF_INV_TOL) return x_r.x;
        if (f_x_s.x * f_x_r.p > 0) {
            x_s.x = x_r.x;
            f_x_s = f_x_r.p;
        } else {
            x_e.x = x_r.x;
            f_x_e = f_x_r.p;
        }
        const float x_n = x_r.x - f_x_r.p / f_x_r.d;
        x_r.x = x_s.x < x_n && x_n < x_e.x ? x_n : (x_s.x + x_e.x) / 2.0;
    }
    return x_r.x;
}

// returns: phi ~ p(phi) \in [0, pi/2]
float base_dxy_sample_phi(float u, float a, float b, bool deriv_nu) {
    float EPS_CLIP = 1e-4;
    u = max(min(u, 1.0 - EPS_CLIP), EPS_CLIP); // restrict absolute edge cases
    return base_dxy_phi_bisection_search(0.0, k_pi_over_2, u, a, b, deriv_nu);
}

float a_phi(float cos2phi, float a, float b) {
    float sin2phi = float(1) - cos2phi;
    return a * cos2phi + b * sin2phi;
}

// Note: This is only defined on [0, pi/2]
// p(phi) for the derivative w.r.t nu
float base_dx_pdf_phi(float cos2phi, float a, float b, bool is_ggx_or_beckmann) {
    if (is_ggx_or_beckmann) { a = a - 1.0; b = b - 1.0;}
    const float sin2phi = float(1) - cos2phi;
    const float C = 4.0 * safe_sqrt(b + float(1)) * pow(a + float(1), float(1.5)) * k_inv_pi;
    return C * cos2phi / pow(float(1) + a_phi(cos2phi, a, b), float(2));
}

// Note: This is only defined on [0, pi/2]
// p(phi) for the derivative w.r.t nv
float base_dy_pdf_phi(float cos2phi, float a, float b, bool is_ggx_or_beckmann) {
    if (is_ggx_or_beckmann) { a = a - 1.0; b = b - 1.0;}
    float sin2phi = float(1) - cos2phi;
    float C = 4.0 * safe_sqrt(a + float(1)) * pow(b + float(1), float(1.5)) * k_inv_pi;
    return C * sin2phi / pow(float(1) + a_phi(cos2phi, a, b), float(2));
}

float sample_all_dxy_base_phi(
    float a, float b, float u,
    bool deriv_nu,
    bool is_ggx_or_beckmann) {
    if (is_ggx_or_beckmann) { a = a - 1.0; b = b - 1.0;}
    // First sample phi
    float phi = 0.0; float u_phi = u;
    // phi is in [0, pi/2], convert to [0, 2pi]
    if (u_phi < 0.25) {
        u_phi = 1.0 - 4.0 * (0.25 - u_phi);
        phi = base_dxy_sample_phi(u_phi, a, b, deriv_nu);
        phi = phi;
    } else if (u_phi < 0.5) {
        u_phi = 1.0 - 4.0 * (0.5 - u_phi);
        phi = base_dxy_sample_phi(u_phi, a, b, deriv_nu);
        // Mirror Flip about pi/2
        phi = k_pi_over_2 + (k_pi_over_2 - phi);
    } else if (u_phi < 0.75) {
        u_phi = 1.0 - 4.0 * (0.75 - u_phi);
        phi = base_dxy_sample_phi(u_phi, a, b, deriv_nu);
        // periodic with period pi
        phi = phi + k_pi;
    } else {
        u_phi = 1.0 - 4.0 * (1.0 - u_phi);
        phi = base_dxy_sample_phi(u_phi, a, b, deriv_nu);
        // periodic with period pi
        phi = k_pi + k_pi_over_2 + (k_pi_over_2 - phi);
    }
    return phi;
}

float pdf_all_dxy_base_phi(
    float a, float b,
    float cos_phi,
    bool xy_deriv,
    bool is_ggx_or_beckmann
) {
    // floatd cos_theta_h = CosTheta(H);
    // Note: Our p(phi) is only defined on [0,pi/2]. p(phi) is symmetric about pi
    // p(phi) is mirror symmetric about pi/2.
    // But cos2phi and sin2phi is also symmetric about pi and mirror symmetric about pi/2,
    // so we do not need to explicitly convert phi from [0,pi*2] to [0,2*pi]
    // floatd cos_phi_h = CosPhi(H);
    const float cos2phi_h = cos_phi * cos_phi;
    const float aphi = a_phi(cos2phi_h, a, b);
    float p_phi = xy_deriv 
        ? base_dx_pdf_phi(cos2phi_h, a, b, is_ggx_or_beckmann) 
        : base_dy_pdf_phi(cos2phi_h, a, b, is_ggx_or_beckmann);
    // There is a change of variables from phi \in [0,pi/2] to [0,2pi] because of which we need to account for the jacobian
    p_phi = p_phi / 4.0;
    return p_phi;
}

#endif // !_SRENDERER_COMMON_MICROFACET_HEADER_