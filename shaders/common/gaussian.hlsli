/* **************************************************
 * Copyright (c) 2023, Haolin Lu <hal128@ucsd.edu>
 * All rights reserved.
 ***************************************************/

#ifndef _SRENDERER_GAUSSIAN_HEADER_
#define _SRENDERER_GAUSSIAN_HEADER_

#include "math.hlsli"
#include "cpp_compatible.hlsli"
#include "linear_algebra.hlsli"

/**
 * Box-Muller transform takes two uniform samples u0 and u1,
 * and transforms them into two Gaussian distributed samples n0 and n1.
 * @param uv 2 uniform samples in range [0, 1]
 * @return 2 standard Gaussian distributed samples.
 */
float2 BoxMuller(in_ref(float2) uv) {
    const float u = max(uv.x, k_numeric_limits_float_min); // clamp u to avoid log(0)
    const float radius = sqrt(-2.0 * log(u)); // radius of the sample
    const float theta = 2 * k_pi * uv.y; // angle of the sample
    float sin_theta; float cos_theta;
    sincos(theta, sin_theta, cos_theta);
    return float2(radius * cos_theta, radius * sin_theta);
}

/**
 * Box-Muller transform takes two uniform samples u0 and u1,
 * and transforms them into two Gaussian distributed samples n0 and n1.
 * @param uv 2 uniform samples in range [0, 1]
 * @param mean mean of the Gaussian distribution
 * @param std standard deviation of the Gaussian distribution
 * @return 2 Gaussian distributed samples.
 */
float2 BoxMuller(in_ref(float2) uv, in_ref(float2) mean, in_ref(float2) std) {
    return BoxMuller(uv) * std + mean;
}

float definite_integral_0_a(float a, float std) {
    return 0.5f * erf(a / (std * k_sqrt2));
}

float definiet_integral_a_inf(float a, float std) {
    return 0.5f * (1.0f - erf(a / (std * k_sqrt2)));
}

struct MultivariateGaussian2D {
    float2x2 precision;  // precision matrix (inverse of the covariance matrix)
    float2 mean;         // mean of the distribution
    float normalization; // normalization constant
    float inv_det_sqrt;   // determinant of the covariance matrix
    
    __init(float2 _mean, float2x2 _covariance) {
        mean = _mean;
        precision = Inverse2x2(_covariance);
        const float det_precision = determinant(precision);
        inv_det_sqrt = 1.0f / sqrt(det_precision);
        normalization = sqrt(abs(det_precision)) / k_2pi;
    }
    // Draw a sample from the distribution
    float2 DrawSample(in_ref(float2) uv) {
        // First sample x from two normal distributions using Box-Muller method
        const float2 n = BoxMuller(uv);
        // Once I have vector x from N(0,1) I can transform it to v = A * x + mean,
        // where A is matrix from equation Covariance = A * A^TScalar
        const float c2 = precision._12 * precision._21;
        float2 dir;
        if (precision._11 > precision._22) {
            float a22 = sqrt(precision._11);
            float a12 = -precision._12 / a22;
            float a11 = sqrt(precision._22 - c2 / precision._11);
            dir.x = inv_det_sqrt * (a11 * n.x + a12 * n.y) + mean.x;
            dir.y = inv_det_sqrt * a22 * n.y + mean.y;
        } else {
            float a11 = sqrt(precision._22);
            float a21 = -precision._12 / a11;
            float a22 = sqrt(precision._11 - c2 / precision._22);
            dir.x = inv_det_sqrt * a11 * n.x + mean.x;
            dir.y = inv_det_sqrt * (a21 * n.x + a22 * n.y) + mean.y;
        }
        return dir;
    }
    // evaluate the probability density function at the given sample
    float Pdf(in_ref(float2) sample) {
        const float2 x = sample - mean;        
        return normalization * exp(-0.5 * QuadraticForm(precision, x));
    }
};

struct Gaussian2DSufficientStats {
    
};

#endif // !_SRENDERER_GAUSSIAN_HEADER_