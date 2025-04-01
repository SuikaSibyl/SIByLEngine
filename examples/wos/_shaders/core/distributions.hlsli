#ifndef _WOS_DISTRIBUTIONS_HLSLI_
#define _WOS_DISTRIBUTIONS_HLSLI_

#include "common/math.hlsli"
#include "common/random.hlsli"
#include "common/sampling.hlsli"
#include "common/bessel.hlsli"

/**
 * Interface for Green's function of a ball
 */
interface IGreensFnBall<let DIM : int> {
    float norm();
    float evaluate(float r_);
    vector<float, DIM> get_sample_inside_ball();
    [mutating] float poisson_kernel();
    [mutating] void update_ball(vector<float, DIM> c_, float R_);
    [mutating] vector<float, DIM> sample_volume(out float pdf, inout RandomSamplerState rng);
    [mutating] vector<float, DIM> sample_surface(out float pdf, inout RandomSamplerState rng);
    [mutating] float pdf_sample_volume(vector<float, DIM>);
}

/**
 * Base for Green's function of a ball
 */
struct BGreensFnBall<let DIM : int> {
    typealias vector_t = vector<float, DIM>;

    vector_t c;     // ball center
    vector_t yVol;  // sampled point inside the ball
    vector_t ySurf; // sampled point on the surface of the ball
    float R;        // ball radius
    float r;        // distance of the sampled point inside the ball to the ball center

    __init() {
        c = vector_t(0);
        yVol = vector_t(0);
        ySurf = vector_t(0);
        R = 0.0f;
        r = 0.0f;
    }
    
    // updates the ball center and radius
    [mutating]
    void update_ball_base(vector_t c_, float R_) {
        c = c_;
        yVol = vector_t(0);
        ySurf = vector_t(0);
        R = R_;
        r = 0.0f;
    }
}

struct HarmonicGreensFnBall2 : IGreensFnBall<2> {
    typealias vector_t = vector<float, 2>;
    BGreensFnBall<2> base;

    // constructor
    __init() { base = BGreensFnBall<2>(); }
    // evaluates the norm of the Green's function
    float norm() { return 1.0f / (2.0f * M_PI); }

    // evaluates the Green's function
    float evaluate() { return log(base.R / base.r) / (2.0f * k_pi); }
    
    // evaluates the Green's function
    float evaluate(float r_) {
        if (r_ > base.R) return 0.0f;
        return log(base.R / r_) / (2.0f * k_pi);
    }

    vector_t get_sample_inside_ball() { return base.yVol; }

    // evaluates the Poisson Kernel
    // (normal derivative of the Green's function)
    float poisson_kernel() {
        return 1.0f / (2.0f * M_PI);
    }

    // evaluates the gradient of the Poisson Kernel
    vector_t poisson_kernel_gradient() {
        vector_t d = base.ySurf - base.c;
        return 2.0f * d / (2.0f * M_PI * base.R * base.R);
    }

    [mutating] void update_ball(vector_t c_, float R_) {
        base.update_ball_base(c_, R_);
    }

    // samples a point inside the ball
    [mutating] vector_t sample_volume(out float pdf, inout RandomSamplerState rng) {
        // TODO: can probably do better
        // rejection sample radius r from pdf 4.0 * r * ln(R / r) / R^2
        const float bound = 1.5f / base.R;
        // return GreensFnBall<2>::rejectionSampleGreensFn(bound, pdf);
        return reject_sample_greens_fn(bound, pdf, rng);
    }

    [mutating] vector_t sample_surface(out float pdf, inout RandomSamplerState rng) {
        base.ySurf = base.c + base.R * uniform_on_disk(GetNextRandom(rng));
        pdf = 1.0f / (2.0f * M_PI);
        return base.ySurf;
    }

    [mutating] float pdf_sample_volume(vector_t p) {
        return evaluate(distance(p, base.c)) / norm();
    }

    [mutating]
    vector_t reject_sample_greens_fn(float bound, out float pdf, inout RandomSamplerState rng) {
        // rejection sample
        int iter = 0;
        do {
            float u = GetNextRandom(rng);
            base.r = GetNextRandom(rng) * base.R;
            pdf = evaluate() / norm();
            float pdfRadius = (2.0f * M_PI * base.r) * pdf;
            iter++;
            if (u < pdfRadius / bound) { break; }
        } while (iter < 1000);

        base.r = max(5e-7f, base.r);
        base.yVol = base.c + base.r * uniform_on_disk(GetNextRandom(rng));
        return base.yVol;
    }
};

struct YukawaGreensFnBall2 : IGreensFnBall<2> {
    typealias vector_t = vector<float, 2>;

    float lambda, sqrtLambda;
    float muR, K0muR, I0muR, K1muR, I1muR;
    BGreensFnBall<2> base;

    __init(float lambda_) {
        base = BGreensFnBall<2>();
        lambda = lambda_;
        sqrtLambda = sqrt(lambda_);
        muR = K0muR = I0muR = K1muR = I1muR = 0.f;
    }

    float2 get_sample_inside_ball() { return base.yVol; }

    // updates the ball center and radius
    [mutating]
    void update_ball(float2 c_, float R_) {
        base.update_ball_base(c_, R_);
        muR = base.R * sqrtLambda;
        K0muR = (float)bessel::bessk0(double(muR));
        I0muR = (float)bessel::bessi0(double(muR));
        K1muR = (float)bessel::bessk1(double(muR));
        I1muR = (float)bessel::bessi1(double(muR));
    }

    // samples a point inside the ball
    [mutating]
    float2 sample_volume(out float pdf, inout RandomSamplerState rng) {
        // TODO: can probably do better
        // rejection sample radius r from pdf r * λ * (K_0(r√λ) * I_0(R√λ) - I_0(r√λ) * K_0(R√λ)) / (I_0(R√λ) - 1)
        float bound = (base.R <= lambda) 
            ? max(max(2.2f / base.R, 2.2f / lambda), max(0.6f * sqrt(base.R), 0.6f * sqrtLambda)) 
            : max(min(2.2f / base.R, 2.2f / lambda), min(0.6f * sqrt(base.R), 0.6f * sqrtLambda));
        return reject_sample_greens_fn(bound, pdf, rng);
    }

    [mutating] float pdf_sample_volume(float2 p) {
        return evaluate(distance(p, base.c)) / norm();
    }

    // evaluates the Green's function
    float evaluate() {
        double mur = base.r * sqrtLambda;
        double K0mur = bessel::bessk0(mur);
        double I0mur = bessel::bessi0(mur);
        return float(K0mur - (I0mur / double(I0muR)) * double(K0muR)) / (2.0f * M_PI);
    }

    // evaluates the Green's function
    float evaluate(float r_) {
        if (r_ > base.R) return 0.0f;
        double mur = r_ * sqrtLambda;
        double K0mur = bessel::bessk0(mur);
        double I0mur = bessel::bessi0(mur);
        return float(K0mur - (I0mur / double(I0muR)) * double(K0muR)) / (2.0f * M_PI);
    }

    // evaluates the gradient of the Green's function
    float2 gradient() {
        float2 d = base.yVol - base.c;
        float mur = base.r * sqrtLambda;
        float K1mur = (float)bessel::bessk1(double(mur));
        float I1mur = (float)bessel::bessi1(double(mur));
        float Qr = sqrtLambda * (K1mur - I1mur * K1muR / I1muR);
        return d * Qr / (2.0f * M_PI * base.r);
    }

    // evaluates the norm of the Green's function
    float norm() {
        return (1.0f - 2.0f * M_PI * poisson_kernel()) / lambda;
    }

    // samples a point on the surface of the ball
    [mutating]
    float2 sample_surface(out float pdf, inout RandomSamplerState rng) {
        base.ySurf = base.c + base.R * uniform_on_disk(GetNextRandom(rng));
        pdf = 1.0f / (2.0f * M_PI);
        return base.ySurf;
    }

    // evaluates the Poisson Kernel
    // (normal derivative of the Green's function)
    float poisson_kernel() {
        return 1.0f / (2.0f * M_PI * I0muR);
    }

    // evaluates the gradient of the Poisson Kernel
    float2 poisson_kernel_gradient() {
        float2 d = base.ySurf - base.c;
        float QR = sqrtLambda / (base.R * I1muR);
        return d * QR / (2.0f * M_PI);
    }

    [mutating]
    vector_t reject_sample_greens_fn(float bound, out float pdf, inout RandomSamplerState rng) {
        // rejection sample
        int iter = 0;
        do {
            float u = GetNextRandom(rng);
            base.r = GetNextRandom(rng) * base.R;
            pdf = evaluate() / norm();
            float pdfRadius = (2.0f * M_PI * base.r) * pdf;
            iter++;
            if (u < pdfRadius / bound) { break; }
        } while (iter < 1000);

        base.r = max(5e-7f, base.r);
        base.yVol = base.c + base.r * uniform_on_disk(GetNextRandom(rng));
        return base.yVol;
    }
};

// struct YukawaGreensFnBall3 : BGreensFnBall<3>, IGreensFnBall<3> {
//     float lambda, sqrtLambda; // potential
//     float muR, expmuR, sinhmuR, K32muR, I32muR;

//     __init(float lambda_) {
//         lambda = lambda_;
//         sqrtLambda = sqrt(lambda_);
//         muR = expmuR = sinhmuR = K32muR = I32muR = 0.f;
//     }

//     float3 get_sample_inside_ball() { return yVol; }

//     // updates the ball center and radius
//     [mutating]
//     void update_ball(float3 c_, float R_) {
//         update_ball_base(c_, R_);
//         muR = R * sqrtLambda;
//         expmuR = exp(-muR);
//         float exp2muR = expmuR * expmuR;
//         float coshmuR = (1.0f + exp2muR) / (2.0f * expmuR);
//         sinhmuR = (1.0f - exp2muR) / (2.0f * expmuR);
//         K32muR = expmuR * (1.0f + 1.0f / muR);
//         I32muR = coshmuR - sinhmuR / muR;
//     }

//     // samples a point inside the ball
//     [mutating]
//     float3 sample_volume(out float pdf, inout RandomSamplerState rng) {
//         // TODO: can probably do better
//         // rejection sample radius r from pdf r * λ * sinh((R - r)√λ) / (sinh(R√λ) - R√λ)
//         float bound = (R <= lambda) 
//             ? max(max(2.0f / R, 2.0f / lambda), max(0.5f * sqrt(R), 0.5f * sqrtLambda)) 
//             : max(min(2.0f / R, 2.0f / lambda), min(0.5f * sqrt(R), 0.5f * sqrtLambda));
//         return reject_sample_greens_fn(bound, pdf, rng);
//     }

//     [mutating] float pdf_sample_volume(float3 p) {
//         return evaluate(distance(p, c)) / norm();
//     }

//     // evaluates the Green's function
//     float evaluate() {
//         float mur = r * sqrtLambda;
//         float expmur = exp(-mur);
//         float sinhmur = (1.0f - expmur * expmur) / (2.0f * expmur);
//         return (expmur - expmuR * sinhmur / sinhmuR) / (4.0f * M_PI * r);
//     }

//     // evaluates the Green's function
//     float evaluate(float r_) {
//         if (r_ > R) return 0.0f;
//         float mur = r_ * sqrtLambda;
//         float expmur = exp(-mur);
//         float sinhmur = (1.0f - expmur * expmur) / (2.0f * expmur);
//         return (expmur - expmuR * sinhmur / sinhmuR) / (4.0f * M_PI * r);
//     }

//     // evaluates the gradient of the Green's function
//     float3 gradient() {
//         float3 d = yVol - c;
//         float mur = r * sqrtLambda;
//         float expmur = exp(-mur);
//         float exp2mur = expmur * expmur;
//         float coshmur = (1.0f + exp2mur) / (2.0f * expmur);
//         float sinhmur = (1.0f - exp2mur) / (2.0f * expmur);
//         float K32mur = expmur * (1.0f + 1.0f / mur);
//         float I32mur = coshmur - sinhmur / mur;
//         float Qr = sqrtLambda * (K32mur - I32mur * K32muR / I32muR);
//         return d * Qr / (4.0f * M_PI * r * r);
//     }

//     // evaluates the norm of the Green's function
//     float norm() {
//         return (1.0f - 4.0f * M_PI * poisson_kernel()) / lambda;
//     }

//     // samples a point on the surface of the ball
//     [mutating]
//     float3 sample_surface(out float pdf, inout RandomSamplerState rng) {
//         ySurf = c + R * UniformOnSphere(GetNextRandomFloat2(rng));
//         pdf = 1.0f / (4.0f * M_PI);
//         return ySurf;
//     }

//     // evaluates the Poisson Kernel 
//     // (normal derivative of the Green's function)
//     float poisson_kernel() {
//         return muR / (4.0f * M_PI * sinhmuR);
//     }

//     // evaluates the gradient of the Poisson Kernel
//     float3 poisson_kernel_gradient() {
//         float3 d = ySurf - c;
//         float QR = lambda / I32muR;
//         return d * QR / (4.0f * M_PI);
//     }

//     [mutating]
//     vector_t reject_sample_greens_fn(float bound, out float pdf, inout RandomSamplerState rng) {
//         // rejection sample
//         int iter = 0;
//         do {
//             float u = GetNextRandom(rng);
//             r = GetNextRandom(rng) * R;
//             pdf = evaluate() / norm();
//             float pdfRadius = (4.0f * M_PI * r * r) * pdf;
//             iter++;
//             if (u < pdfRadius / bound) { break; }
//         } while (iter < 1000);

//         r = max(5e-7f, r);
//         yVol = c + r * UniformOnSphere(GetNextRandomFloat2(rng));
//         return yVol;
//     }
// };

#endif // _WOS_DISTRIBUTIONS_HLSLI_