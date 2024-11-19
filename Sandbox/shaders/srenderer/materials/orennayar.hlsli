#ifndef _SRENDERER_ORENNAYAR_MATERIAL_
#define _SRENDERER_ORENNAYAR_MATERIAL_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/scene-binding.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Oren-Nayar BRDF
// ----------------------------------------------------------------------------------------
// Rough surfaces generally appear brighter when the illumination direction
// approaches the viewing direction. Oren-Nayar follows this observation by
// modeling the rough surfaces by V-shape microfacets described by a spherical
// Gaussian distribution with a single parameter sigma.
///////////////////////////////////////////////////////////////////////////////////////////

struct OrenNayarMaterial : IBxDFParameter {
    float3 R;       // Reflectance
    float sigma;    // Gaussian deviation [0, 1]

    __init() { R = float3(1.f); sigma = 0.f; }
    __init(MaterialData mat, float2 uv) {
        R = mat.floatvec_0.xyz * SampleTexture2D(mat.albedo_tex, uv,
            mat.is_albedo_tex_differentiable()) .rgb;

        const float ext_tex_r = SampleTexture2D(
            mat.ext1_tex, uv,
            mat.is_ext1_tex_differentiable()) .r;
        sigma = clamp(mat.floatvec_0.w * ext_tex_r, 0.f, 1.f);
    }

    [BackwardDifferentiable]
    static OrenNayarMaterial load(no_diff MaterialData data, no_diff float2 uv) {
        const float ext_tex_r = SampleTexture2D(
            data.ext1_tex, uv,
            data.is_ext1_tex_differentiable()).r;
        
        OrenNayarMaterial material = no_diff OrenNayarMaterial();
        material.R = data.floatvec_0.xyz * SampleTexture2D(data.albedo_tex, uv,
            data.is_albedo_tex_differentiable()) .rgb;
        material.sigma = clamp(data.floatvec_0.w * ext_tex_r, 0.f, 1.f);
        return material;
    }

    [BackwardDifferentiable]
    static OrenNayarMaterial load_w_aux(
        no_diff MaterialData data, no_diff float2 uv, 
        no_diff float sigma_aux
    ) {
        const float ext_tex_r = SampleTexture2D_w_aux(
            data.ext1_tex, uv,
            float4(sigma_aux, 0, 0, 0),
            data.is_ext1_tex_differentiable()) .r;
        
        OrenNayarMaterial material = no_diff OrenNayarMaterial();
        material.R = data.floatvec_0.xyz * SampleTexture2D(data.albedo_tex, uv,
                                                           data.is_albedo_tex_differentiable()) .rgb;
        material.sigma = clamp(data.floatvec_0.w * ext_tex_r, 0.f, 1.f);
        return material;
    }
};

struct OrenNayarBRDF : IBxDF {
    typedef OrenNayarMaterial TParam;

    // evaluate the BSDF
    [Differentiable]
    static float3 eval(no_diff ibsdf::eval_in i, OrenNayarMaterial material) {
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        float3 wo = no_diff i.shading_frame.to_local(i.wo);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);
        
        const float sinThetaI = theta_phi_coord::SinTheta(wi);
        const float sinThetaO = theta_phi_coord::SinTheta(wo);
        // Compute cosine term of Oren–Nayar model
        float maxCos = 0;
        if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
            float sinPhiI = theta_phi_coord::SinPhi(wi);
            float cosPhiI = theta_phi_coord::CosPhi(wi);
            float sinPhiO = theta_phi_coord::SinPhi(wo);
            float cosPhiO = theta_phi_coord::CosPhi(wo);
            float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = max(0.0, dCos);
        }
        // Compute sine and tangent terms of Oren–Nayar model
        float sinAlpha; float tanBeta;
        if (theta_phi_coord::AbsCosTheta(wi) > 
            theta_phi_coord::AbsCosTheta(wo)) {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / theta_phi_coord::AbsCosTheta(wi);
        } else {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / theta_phi_coord::AbsCosTheta(wo);
        }
        // Compute A and B terms of Oren–Nayar model
        const float sigma = material.sigma * k_2pi;
        const float sigma2 = sigma * sigma;
        const float A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        const float B = 0.45 * sigma2 / (sigma2 + 0.09);
        return material.R * k_inv_pi * (A + B * maxCos * sinAlpha * tanBeta) * abs(wo.z);
    }

    // importance sample the BSDF
    // here we simply use cosine-weighted hemisphere sampling
    static ibsdf::sample_out sample(ibsdf::sample_in i, OrenNayarMaterial material) {
        ibsdf::sample_out o;
        // Flip the shading frame if it is
        // inconsistent with the geometry normal.
        Frame frame = i.shading_frame;
        o.wo = frame.to_world(sample_cos_hemisphere(i.u.xy));
        o.pdf = abs(dot(frame.n, o.wo)) * k_inv_pi;

        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
    }
    
    // evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, OrenNayarMaterial material) {
        Frame frame = i.shading_frame;
        return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
    }

    static void backward_grad(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        OrenNayarMaterial material = OrenNayarMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval)(i.eval, material_pair, dL);
        bwd_diff(OrenNayarMaterial::load)(mat, texcoord, material_pair.d);
    }
    
    static void backward_grad_ratio(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) { 
        OrenNayarMaterial material = OrenNayarMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval)(i.eval, material_pair, dL);
        float aux = 1.f / (i.pdf * k_2pi);
        bwd_diff(OrenNayarMaterial::load_w_aux)(mat, texcoord, aux, material_pair.d);
    }

    static void backward_sigma_derivative_diffuse(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        OrenNayarMaterial material = OrenNayarMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval_term1)(i.eval, material_pair, dL);
        bwd_diff(OrenNayarMaterial::load)(mat, texcoord, material_pair.d);
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_sigma_derivative_diffuse(
        const ibsdf::sample_in i, OrenNayarMaterial material) {
        // Sample diffuse BRDF
        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(sample_cos_hemisphere(i.u.xy));
        o.pdf = abs(dot(i.shading_frame.n, o.wo)) * k_inv_pi;
        return o;
    }

    static void backward_sigma_derivative_spec(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        OrenNayarMaterial material = OrenNayarMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval_term2)(i.eval, material_pair, dL);
        bwd_diff(OrenNayarMaterial::load)(mat, texcoord, material_pair.d);
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_sigma_derivative_spec(
        const ibsdf::sample_in i, OrenNayarMaterial material) {
        // Sample the derivative: cos-weighted hemisphere
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);
        // Sample the phi and theta respectively
        // 1. first sample the phi
        // Sample t between [0, pi/2] using cosine hemispherical sampling
        const float phi_o = sample_dsigma_t2_phi(theta_phi_coord::Phi(wi), i.u.x);
        const float theta_o = sample_dsigma_t2_theta(theta_phi_coord::Theta(wi), i.u.y);
        const float3 wo = theta_phi_coord::SphericalDirection(theta_o, phi_o);

        // evaluate the pdf
        ibsdf::pdf_in pi;
        pi.wi = i.wi;
        pi.wo = i.shading_frame.to_world(wo);
        pi.geometric_normal = i.geometric_normal;
        pi.shading_frame = i.shading_frame;
        float pdf = pdf_dsigma_term2(pi);
        
        // Sample diffuse BRDF
        ibsdf::sample_out o;
        o.wo = pi.wo;
        o.pdf = pdf;
        return o;
    }

    // evaluate the first term of the BSDF (diffuse term)
    [Differentiable]
    static float3 eval_term1(no_diff ibsdf::eval_in i, OrenNayarMaterial material) {
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        float3 wo = no_diff i.shading_frame.to_local(i.wo);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);

        // Compute A and B terms of Oren–Nayar model
        const float sigma = material.sigma * k_2pi;
        const float sigma2 = sigma * sigma;
        const float A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        return material.R * k_inv_pi * A * abs(wo.z);
    }
    
    // evaluate the PDF of the first term of the BSDF
    // here we simply use cosine-weighted hemisphere sampling
    float pdf_term1(ibsdf::pdf_in i) {
        Frame frame = i.shading_frame;
        return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
    }

    // evaluate the BSDF Derivative of the 2st term
    // which is the more specular term
    [Differentiable]
    static float3 eval_term2(no_diff ibsdf::eval_in i, OrenNayarMaterial material) {
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        float3 wo = no_diff i.shading_frame.to_local(i.wo);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);

        const float sinThetaI = theta_phi_coord::SinTheta(wi);
        const float sinThetaO = theta_phi_coord::SinTheta(wo);
        // Compute cosine term of Oren–Nayar model
        float maxCos = 0;
        if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
            float sinPhiI = theta_phi_coord::SinPhi(wi);
            float cosPhiI = theta_phi_coord::CosPhi(wi);
            float sinPhiO = theta_phi_coord::SinPhi(wo);
            float cosPhiO = theta_phi_coord::CosPhi(wo);
            float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = max(0.0, dCos);
        }
        // Compute sine and tangent terms of Oren–Nayar model
        float sinAlpha; float tanBeta;
        if (theta_phi_coord::AbsCosTheta(wi) >
            theta_phi_coord::AbsCosTheta(wo)) {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / theta_phi_coord::AbsCosTheta(wi);
        } else {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / theta_phi_coord::AbsCosTheta(wo);
        }
        // Compute A and B terms of Oren–Nayar model
        const float sigma = material.sigma * k_2pi;
        const float sigma2 = sigma * sigma;
        const float B = 0.45 * sigma2 / (sigma2 + 0.09);
        return material.R * k_inv_pi * B * maxCos * sinAlpha * tanBeta * abs(wo.z);
    }

    // The CDF of the second term of the BSDF derivative
    // one less that theta_o and the other one more than it
    [Differentiable]
    static float dsigma_t2_theta1_cdf(float x, no_diff float u, no_diff float theta_o) {
        return (0.5 * (x - sin(x) * cos(x))) / (0.5 * (theta_o - sin(theta_o) * cos(theta_o))) - u;
    }

    // The CDF of the second term of the BSDF derivative
    // one less that theta_o and the other one more than it
    [Differentiable]
    static float dsigma_t2_theta2_cdf(float x, no_diff float u, no_diff float theta_o) {
        return (pow(sin(x), 3.0) - pow(sin(theta_o), 3.0)) / (1.0 - pow(sin(theta_o), 3.0)) - u;
    }

    // Sample the theta_i using the CDF with bisection search, with Newton's method
    // There are two PDFs, one less that theta_o and the other one more than it
    static float dsigma_t2_theta_cdf_bisection_search(
        float x_s, float x_e, 
        float u, bool less_than_theta_o, 
        float theta_o) {
        float x_r = (x_s + x_e) / 2.0;
        float x_n;
        float f_x_s = 0; float f_x_e = 0;
        f_x_s = less_than_theta_o ? dsigma_t2_theta1_cdf(x_s, u, theta_o) : dsigma_t2_theta2_cdf(x_s, u, theta_o);
        f_x_e = less_than_theta_o ? dsigma_t2_theta1_cdf(x_e, u, theta_o) : dsigma_t2_theta2_cdf(x_e, u, theta_o);
        for (int i = 0; i < 64; i++) {
            // evaluate the function at x_r
            var f_x_r = less_than_theta_o
                ? fwd_diff(dsigma_t2_theta1_cdf)(diffPair(x_r), u, theta_o) 
                : fwd_diff(dsigma_t2_theta2_cdf)(diffPair(x_r), u, theta_o);
            // check if the function is close to zero
            if (abs(f_x_r.p) < 1e-6) return x_r.x;
            // check if the function has the same sign
            // if it does, then update the x_s
            if (f_x_s.x * f_x_r.p > 0) {
                x_s.x = x_r.x;
                f_x_s = f_x_r.p;
            } 
            // else update the x_e
            else {
                x_e.x = x_r.x;
                f_x_e = f_x_r.p;
            }
            // Finally, update the x_r
            x_n = x_r.x - f_x_r.p / f_x_r.d;
            x_r.x = (x_s.x < x_n && x_n < x_e.x) ? x_n : (x_s.x + x_e.x) / 2.0;
        }
        return x_r.x;
    }

    // sample the phi_i with respect to the second term of the BSDF derivative
    // p(phi) = 0.5 max(0, cos(phi_i - phi_o))
    static float sample_dsigma_t2_phi(float phi_o, float u) {
        // Sample t between [0, pi/2] using cosine hemispherical sampling
        float phi_i;
        // Now, phi_i can either belong to [phi_o, phi_o + pi/2] 
        // or [phi_o - pi/2, phi_o] with equal prob
        // so we need to choose between the two lobes.
        if (u < 0.5) {
            u = 1.0 - 2.0 * (0.5 - u);
            float t = asin(u);
            phi_i = phi_o - t;
        } else {
            u = 1.0 - 2.0 * (1.0 - u);
            float t = asin(u);
            phi_i = phi_o + t;
        }
        return phi_i;
    }

    // sample the theta_i with respect to the second term of the BSDF derivative
    static float sample_dsigma_t2_theta(float theta_o, float u) {
        float A1 = 0.5 * sin(theta_o) * (theta_o - sin(theta_o) * cos(theta_o));
        float A2 = 1.0 / 3.0 * tan(theta_o) * (1.0 - pow(sin(theta_o), 3.0));
        float T = (A1 + A2);
        A1 = A1 / T; A2 = A2 / T;
        float theta_i = 0.0;
        // choose between theta_i < theta_o based on the conditions above
        if (u < A1) {
            u = u / A1;
            theta_i = dsigma_t2_theta_cdf_bisection_search(0, theta_o, u, true, theta_o);
        } else {
            u = (u - A1) / (1.0 - A1);
            theta_i = dsigma_t2_theta_cdf_bisection_search(theta_o, k_pi_over_2, u, false, theta_o);
        }
        return theta_i;
    }

    // sampling the derivative of the second term of the BSDF
    static ibsdf::dsample_out<OrenNayarMaterial> sample_dsigma_t2(
        ibsdf::sample_in i,
        OrenNayarMaterial material,
        float3 d_output) {
        // Sample the derivative: cos-weighted hemisphere
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);
        // Sample the phi and theta respectively
        // 1. first sample the phi
        // Sample t between [0, pi/2] using cosine hemispherical sampling
        const float phi_o = sample_dsigma_t2_phi(theta_phi_coord::Phi(wi), i.u.x);
        const float theta_o = sample_dsigma_t2_theta(theta_phi_coord::Theta(wi), i.u.y);
        const float3 wo = theta_phi_coord::SphericalDirection(theta_o, phi_o);
        
        // evaluate the pdf
        ibsdf::pdf_in pi;
        pi.wi = i.wi;
        pi.wo = i.shading_frame.to_world(wo);
        pi.geometric_normal = i.geometric_normal;
        pi.shading_frame = i.shading_frame;
        float pdf = pdf_dsigma_term2(pi);

        // Assemble the evaluation input
        ibsdf::eval_in i_eval;
        i_eval.wi = i.wi;
        i_eval.wo = pi.wo;
        i_eval.geometric_normal = i.geometric_normal;
        i_eval.shading_frame = i.shading_frame;
        // Evaluate the 1st term of the BSDF Derivative
        var material_pair = diffPair(material);
        bwd_diff(OrenNayarBRDF::eval_term2)(i_eval, material_pair, d_output / pdf);
        
        // Assemble the output
        ibsdf::dsample_out<OrenNayarMaterial> o_diff;
        o_diff.wo = pi.wo;
        o_diff.pdf = pdf;
        o_diff.dparam = material_pair.d;
        
        return o_diff;
    }

    // theta pdf1, 0 < theta_i < theta_o
    static float pdf_dsigma_t2_theta_1(float theta_o, float theta_i) {
        return max(sin(theta_i) / (0.5 * (theta_o - sin(theta_o) * cos(theta_o))), 0.0);
    }

    // theta pdf1, theta_o < theta_i < pi/2
    static float pdf_dsigma_t2_theta_2(float theta_o, float theta_i) {
        return max(sin(theta_i) * cos(theta_i) * 3.0 / (1.0 - pow(sin(theta_o), 3.0)), 0.0);
    }

    // pdf of the second term of the BSDF derivative sampling theta
    static float pdf_dsigma_t2_theta(float theta_o, float theta_i) {
        float A1 = 0.5 * sin(theta_o) * (theta_o - sin(theta_o) * cos(theta_o));
        float A2 = 1.0 / 3.0 * tan(theta_o) * (1.0 - pow(sin(theta_o), 3.0));
        float T = (A1 + A2);
        A1 = A1 / T; A2 = A2 / T;
        float pdf_theta_solid = 0.0;
        if (theta_i < theta_o) pdf_theta_solid = A1 * pdf_dsigma_t2_theta_1(theta_o, theta_i);
        else pdf_theta_solid = A2 * pdf_dsigma_t2_theta_2(theta_o, theta_i);
        return pdf_theta_solid;
    }

    // pdf of the second term of the BSDF derivative sampling phi
    static float pdf_dsigma_t2_phi(float phi_o, float phi_i) {
        return 0.5 * max(0.0, cos(phi_o - phi_i));
    }

    // evaluate the PDF of the second term of the BSDF derivative sampling
    static float pdf_dsigma_term2(ibsdf::pdf_in i) {
        float3 wi = i.shading_frame.to_local(i.wi);
        float3 wo = i.shading_frame.to_local(i.wo);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);
        float theta_i = theta_phi_coord::Theta(wi);
        float theta_o = theta_phi_coord::Theta(wo);
        float phi_i = theta_phi_coord::Phi(wi);
        float phi_o = theta_phi_coord::Phi(wo);
        
        float pdf_theta_solid = pdf_dsigma_t2_theta(theta_i, theta_o);
        float pdf_phi = pdf_dsigma_t2_phi(phi_i, phi_o);
        float pdf = pdf_theta_solid * pdf_phi;
        if (isnan(pdf)) pdf = 0.0;

        return pdf;
    }
};

#endif // _SRENDERER_ORENNAYAR_MATERIAL_