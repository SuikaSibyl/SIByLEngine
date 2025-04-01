#ifndef _SRENDERER_GGX_ANSIO_MATERIAL_
#define _SRENDERER_GGX_ANSIO_MATERIAL_

#include "bxdf.hlsli"
#include "common/math.hlsli"
#include "common/sampling.hlsli"
#include "common/microfacet.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Anisotropic GGX BRDF
///////////////////////////////////////////////////////////////////////////////////////////

struct AnisoGGXMaterial : IBxDFParameter {
    float alpha_x;
    float alpha_y;
};

struct AnisoGGXBRDF : IBxDF {
    typedef AnisoGGXMaterial TParam;

    [Differentiable]
    static float3 eval(no_diff ibsdf::eval_in i, AnisoGGXMaterial material) {
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        float3 wo = no_diff i.shading_frame.to_local(i.wo);
        if (wi.z < 0) wi = float3(wi.x, wi.y, -wi.z);
        if (wo.z < 0) return float3(0);

        float cos_theta_i = theta_phi_coord::CosTheta(wi);
        float cos_theta_o = theta_phi_coord::CosTheta(wo);
        float3 H = normalize(wi + wo);
        
        // if (cos_theta_i <= 0.f || cos_theta_o <= 0.f || dot(w_i, H) <= 0.f || dot(w_o, H) <= 0.f)
        //     return 0.f;

        // Fresnel Term
        // TODO: see if the min is required
        // floatd H_dot_w_o = dot(w_o, H);
        // vec3fd F = shlick_fresnel(H_dot_w_o, bsdf.fres_F0, lobe_type, derivative_type);
        float D = ggx_D(H, material.alpha_x, material.alpha_y);
        // floatd D = 1.0;
        // floatd G = 1.0;
        // cos_theta_o = 1.0;
        // float G = ggx_G1(wi, material.alpha_x, material.alpha_y) * ggx_G1(wo, material.alpha_x, material.alpha_y);
        float G = ggx_G1(wi, material.alpha_x, material.alpha_y);
        
        return D * abs(wo.z) / (4.0f) ;
        // vec3fd ret = F * G * D / (4.0f * cos_theta_o);
        // if (ret.v.x < 0) {
        //     std::cout << "negative  \n";
        // }

        // return F * G * D / (4.0f * cos_theta_o);
    }

    static ibsdf::sample_out sample(ibsdf::sample_in i, AnisoGGXMaterial material) {
        // Flip the incident direction if it is on the wrong side of the normal
        Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi = float3(wi.x, wi.y, -wi.z); i.wi = frame.to_world(wi); }

        // sample the microfacet normal with visible normal distribution
        AnisotropicTrowbridgeReitzParameter ggx_params;
        ggx_params.alpha_x = material.alpha_x;
        ggx_params.alpha_y = material.alpha_y;
        ibsdf::sample_out o = microfacet_reflection::sample_vnormal<
            AnisotropicTrowbridgeReitzDistribution>(i, ggx_params);
        
        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
    }

    float pdf(ibsdf::pdf_in i) {
        Frame frame = i.shading_frame;
        return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
    }

    // C is the parameter dependent constant (over the hemispherical domain)
    // var is the part that varies over the domain
    [Differentiable]
    static float ggx_D(no_diff float3 m, float alpha_x, float alpha_y) {
        const float temp = sqr(m.x / alpha_x) + sqr(m.y / alpha_y) + sqr(m.z);
        const float C = 1.0 / (k_pi * alpha_x * alpha_y);
        const float var = 1.0 / sqr(temp);
        const float res = C * var;
        return res;
        // if (derivative_type == DerivativeType::ggx_alphax || derivative_type == DerivativeType::ggx_alphay || derivative_type == DerivativeType::ggx_alpha) {
        //     if (lobe_type == BRDFLobeType::all)
        //         return res;
        //     else if (lobe_type == BRDFLobeType::d_ndf)
        //         return floatd(res.x, C.x * var.dx);
        //     else if (lobe_type == BRDFLobeType::d_norm)
        //         return floatd(res.x, C.dx * var.x);
        //     else {
        //         return 0.0 / 0.0;
        //         // assert(false);
        //         return 0.0;
        //     }
        // } else {
        //     return res;
        // }
    }

    static float ggx_D_diff_alpha_x(float3 m, float alpha_x, float alpha_y) {
        var alpha_x_pair = diffPair(alpha_x);
        var alpha_y_pair = diffPair(alpha_y);
        bwd_diff(AnisoGGXBRDF::ggx_D)(m, alpha_x_pair, alpha_y_pair, 1);
        return alpha_x_pair.d;
        // float x2 = sqr(m.x); float y2 = sqr(m.y); float z2 = sqr(m.z);
        // float a = alpha_x; float a2 = sqr(a);
        // float b = alpha_y; float b2 = sqr(b); float b3 = b2 * b;
        // float nominator = a2 * b3 * (b2 * (a2 * z2 - 3 * x2) + a2 * y2);
        // float tmp = b2 * (a2 * z2 + x2) + a2 * y2;
        // float denominator = k_pi * tmp * tmp * tmp;
        // return - nominator / denominator;
    }

    [Differentiable]
    static float ggx_G1(no_diff float3 v, float alpha_x, no_diff float alpha_y) {
        float xy_alpha_2 = sqr(alpha_x * 1) + sqr(0.2 * 1);
        float tan_theta_alpha_2 = xy_alpha_2 / sqr(1);
        float result = float(2.f) / (float(1.f) + sqrt(float(1.f) + tan_theta_alpha_2));
        return result;
    }
    
    // evaluate the derivative of the BSDF
    static AnisoGGXMaterial.Differential diff_eval(ibsdf::eval_in i, AnisoGGXMaterial material, float3 d_output) {
        var material_pair = diffPair(material);
        bwd_diff(AnisoGGXBRDF::eval)(i, material_pair, d_output);
        return material_pair.d;
    }
};

#endif // _SRENDERER_GGX_ANSIO_MATERIAL_