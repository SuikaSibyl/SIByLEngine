#ifndef _SRENDERER_VM_HEADER_
#define _SRENDERER_VM_HEADER_

#include "math.hlsli"
#include "random.hlsli"
#include "hashing.hlsli"

namespace bessel_impl {
float _eval_poly<let Size : int>(
    float y, Array<float, Size> coef) {
    float result = coef[Size - 1];
    for (int i = Size - 2; i >= 0; i--)
    result = coef[i] + y * result;
    return result;
}

// order is either 0 or 1
float _log_modified_bessel_fn_order0(float x) {
    float _I0_COEF_SMALL[7] = {
        1.0,
        3.5156229,
        3.0899424,
        1.2067492,
        0.2659732,
        0.360768e-1,
        0.45813e-2,
    };

    float _I0_COEF_LARGE[9] = {
        0.39894228,
        0.1328592e-1,
        0.225319e-2,
        -0.157565e-2,
        0.916281e-2,
        -0.2057706e-1,
        0.2635537e-1,
        -0.1647633e-1,
        0.392377e-2,
    };

    float y = x / 3.75;
    y = y * y;
    float small = _eval_poly<7>(y, _I0_COEF_SMALL);
    small = log(small);
    // compute large solution
    y = 3.75 / x;
    float large = x - 0.5 * log(x) 
        + log(_eval_poly<9>(y, _I0_COEF_LARGE));
    return (x < 3.75) ? small : large;
}
}

struct vonMises {
    float mu;
    float kappa;

    __init() {}
    __init(float _mu, float _kappa) {
        mu = _mu;
        kappa = _kappa;
    }
    
    float sample(float u) {
        // As kappa -> 0 one obtains the
        // uniform distribution on the circle
        // [0, 1]->[0, 2 * pi]
        if (kappa < k_float_epsilon) {
            return 2.0 * k_pi * u;
        }
        // Otherwise, we use the rejection sampling
        // Initialize the random number generator
        // rejection sampling
        RandomSamplerState RNG;
        RNG.state = uint(double(u) * double(0xffffffffu));
        // Used same notation as Ref.~[2], p49
        const float a = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa);
        const float b = (a - sqrt(2.0 * a)) / (2.0 * kappa);
        const float r = (1.0 + b * b) / (2.0 * b);
        while (true) {
            // Pseudo-random numbers sampled 
            // from a uniform distribution [0,1]
            const float U1 = GetNextRandom(RNG);
            const float U2 = GetNextRandom(RNG);
            const float U3 = GetNextRandom(RNG);

            const float z = cos(k_pi * U1);
            const float f = (1.0 + r * z) / (r + z);
            const float c = kappa * (r - f);
            
            if (((c * (2.0 - c) - U2) > 0.0) 
                || ((log(c / U2) + 1.0 - c) > 0.0)) {
                return fmod(sign(U3 - 0.5) * acos(f) + mu, 2 * k_pi);
            }
        }
        return 0.0 / 0.0;
    }

    float pdf(float theta) {
        return exp(kappa * cos(theta - mu)) / (2.0 * k_pi *
        exp(bessel_impl::_log_modified_bessel_fn_order0(kappa)));
    }

    float log_pdf(float theta) {
        if (kappa == 0) {
            return -log(2.0 * k_pi);
        }
        return kappa * cos(theta - mu) - log(2.0 * k_pi)
        - bessel_impl::_log_modified_bessel_fn_order0(kappa);
    }
};

#endif // !_SRENDERER_VM_HEADER_