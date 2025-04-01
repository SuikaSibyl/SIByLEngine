#ifndef _SRENDERER_CONDUCTOR_BRDF_
#define _SRENDERER_CONDUCTOR_BRDF_

#include "bxdf.hlsli"
#include "srenderer/scene-binding.hlsli"
#include "srenderer/spt.hlsli"


struct ConductorMaterial : IBxDFParameter {
    float3 eta;  // real component of IoR
    float3 k;    // imaginary component of IoR
    float alpha;
    
    __init() {}
    __init(MaterialData data, float2 uv) {
        const float ext_tex_r = SampleTexture2D(
            data.ext1_tex, uv,
            data.is_ext1_tex_differentiable()).r;
        
        k = data.floatvec_0.xyz;
        alpha = data.floatvec_0.w * ext_tex_r;
        eta = data.floatvec_2.xyz;
    }
    
    [BackwardDifferentiable]
    static ConductorMaterial load(no_diff MaterialData data, no_diff float2 uv) {
        const float ext_tex_r = SampleTexture2D(
            data.ext1_tex, uv,
            data.is_ext1_tex_differentiable()).r;
        
        ConductorMaterial material = no_diff ConductorMaterial();
        material.k = data.floatvec_0.xyz;
        material.alpha = data.floatvec_0.w * ext_tex_r;
        material.eta = data.floatvec_2.xyz;
        return material;
    }
};

struct ConductorBRDF : IBxDF {
    typedef ConductorMaterial TParam;

    // Evaluate the BSDF
    [Differentiable]
    static float3 eval(no_diff ibsdf::eval_in i, ConductorMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = i.shading_frame.to_local(i.wh);
        if(wo.z < 0 || wh.z < 0) return float3(0, 0, 0);
        
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        float3 f = eval_isotropic_ggx_conductor(wi, wo, wh,
            material.eta, material.k, params);
        return f;
    }
    
    static void backward_grad(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        ConductorMaterial material = ConductorMaterial(mat, texcoord);
        // var material_pair = diffPair(material);
        // bwd_diff(eval)(i.eval, material_pair, dL);
        
        float2 dd_dg_dalpha = manual_backward_dalpha(i.eval, material, float3(1, 1, 1));
        float dalpha = dd_dg_dalpha.x; // + dd_dg_dalpha.y;
        ConductorMaterial.Differential d_material;
        d_material.alpha = dalpha * average(dL);
        d_material.eta = float3(0, 0, 0);
        d_material.k = float3(0, 0, 0);

        bwd_diff(ConductorMaterial::load)(mat, texcoord, d_material);
    }
    
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, ConductorMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        ibsdf::sample_out o;
        if (IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            float3 f = FresnelComplex(theta_phi_coord::AbsCosTheta(wi),
                                      complex3(material.eta, material.k));
            o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
            o.pdf = 1.f;
            o.bsdf = float3(f) / o.pdf;
            return o;
        }
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        o = microfacet_reflection::sample_vnormal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        eval_in.wh = o.wh;
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, ConductorMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        const float3 wh = normalize(wi + wo);
        if (IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            return 1.f;
        }
        
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        const float pdf = IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wi, wh, params);
        const float VdotH = abs(dot(wi, wh));
        return pdf / (4 * abs(VdotH));
    }

    static float2 manual_backward_dalpha(
        ibsdf::eval_in i, ConductorMaterial material, float3 dL) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = i.shading_frame.to_local(i.wh);
        if (wo.z < 0 || wh.z < 0) return {};
        
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        float d = IsotropicTrowbridgeReitzDistribution::D(wh, params);
        float g = IsotropicTrowbridgeReitzDistribution::G(wo, wi, params);
        if (wo.z <= 0 || wh.z <= 0 || wi.z <= 0) return {};
        // Evaluate Fresnel factor F for conductor BRDF
        float3 F = FresnelComplex(abs(dot(wi, wh)), complex3(material.eta, material.k));
        float3 weight = F / (4 * theta_phi_coord::AbsCosTheta(wi));
        
        var pair_params_d = diffPair(params);
        bwd_diff(IsotropicTrowbridgeReitzDistribution::D)(wh, pair_params_d, sum(dL * weight * g));
        
        var pair_params_g = diffPair(params);
        bwd_diff(IsotropicTrowbridgeReitzDistribution::G)(wo, wi, pair_params_g, sum(dL * weight * d));

        return { pair_params_d.d.alpha, pair_params_g.d.alpha };
    }

    [Differentiable]
    static float3 eval_isotropic_ggx_conductor(
        no_diff float3 wi,
        no_diff float3 wo,
        no_diff float3 wh,
        float3 eta,
        float3 k,
        IsotropicTrowbridgeReitzParameter params
    ) {
        if (wo.z <= 0 || wh.z <= 0 || wi.z <= 0) return float3(0, 0, 0);
        // Evaluate Fresnel factor F for conductor BRDF
        float3 F = FresnelComplex(abs(dot(wi, wh)), complex3(eta, k));
        float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                    * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        return f;
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_alpha_derivative_pos(
        const ibsdf::sample_in i, ConductorMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;

        float4 sample_pdf = microfacet_reflection::sample_pos<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);

        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wh = sample_pdf.xyz;
        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(reflect(-wi, wh));
        o.pdf = sample_pdf.w / (4 * abs(dot(wi, wh)));
        o.wh = i.shading_frame.to_world(wh);
        o.pdf_wh = sample_pdf.w;
        return o;
    }

    /** sample but not compute brdfd with postivized derivative sampling, negative */
    static ibsdf::sample_out sample_alpha_derivative_neg(
        const ibsdf::sample_in i, ConductorMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;

        float4 sample_pdf = microfacet_reflection::sample_neg<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
        
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wh = sample_pdf.xyz;
        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(reflect(-wi, wh));
        o.pdf = sample_pdf.w / (4 * abs(dot(wi, wh)));
        o.wh = i.shading_frame.to_world(wh);
        o.pdf_wh = sample_pdf.w;
        return o;
    }
    
    static void backward_alpha_derivative_pos(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        ConductorMaterial material = ConductorMaterial(mat, texcoord);
        // var material_pair = diffPair(material);
        // bwd_diff(eval)(i.eval, material_pair, float3(1, 1, 1));

        float2 dd_dg_dalpha = manual_backward_dalpha(i.eval, material, float3(1, 1, 1));
        float dalpha = max(dd_dg_dalpha.x, 0) + dd_dg_dalpha.y;
        // float dalpha = dd_dg_dalpha.y;

        // postprocess the derivatives
        ConductorMaterial.Differential d_material;
        d_material.alpha = dalpha * average(dL);
        d_material.eta = float3(0, 0, 0);
        d_material.k = float3(0, 0, 0);
        bwd_diff(ConductorMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_alpha_derivative_neg(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        ConductorMaterial material = ConductorMaterial(mat, texcoord);
        // var material_pair = diffPair(material);
        // bwd_diff(eval)(i.eval, material_pair, float3(1, 1, 1));
        // float dalpha = material_pair.d.alpha;

        float2 dd_dg_dalpha = manual_backward_dalpha(i.eval, material, float3(1, 1, 1));
        float dalpha = min(dd_dg_dalpha.x, 0) + dd_dg_dalpha.y;

        // postprocess the derivatives
        ConductorMaterial.Differential d_material;
        d_material.alpha = dalpha * average(dL);
        d_material.eta = float3(0, 0, 0);
        d_material.k = float3(0, 0, 0);
        bwd_diff(ConductorMaterial::load)(mat, texcoord, d_material);
    }
    
    /** The pdf of postivized derivative sampling, positive part */
    static float pdf_alpha_derivative_pos(const ibsdf::pdf_in i, ConductorMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        return microfacet_reflection::pdf_pos<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
    }

    /** The pdf of postivized derivative sampling, negative part */
    static float pdf_alpha_derivative_neg(const ibsdf::pdf_in i, ConductorMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        return microfacet_reflection::pdf_neg<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
    }
}

// struct AnisoConductorMaterial : IBxDFParameter {
//     float eta; // real component of IoR
//     float k;   // imaginary component of IoR
//     float alpha_x;
//     float alpha_y;
// };

// struct AnisoConductorBRDF : IBxDF {
//     typedef AnisoConductorMaterial TParam;

//     // Evaluate the BSDF
//     static float3 eval(ibsdf::eval_in i, AnisoConductorMaterial material) {
//         if (dot(i.geometric_normal, i.wi) < 0 ||
//             dot(i.geometric_normal, i.wo) < 0) {
//             // No light below the surface
//             return float3(0);
//         }
//         Frame frame = i.shading_frame;
//         // Lambertian BRDF
//         return 0.f;
//     }
//     // importance sample the BSDF
//     static ibsdf::sample_out sample(ibsdf::sample_in i, AnisoConductorMaterial material) {
//         ibsdf::sample_out o;
//         // For Lambertian, we importance sample the cosine hemisphere domain.
//         if (dot(i.geometric_normal, i.wi) < 0) {
//             // Incoming direction is below the surface.
//             o.bsdf = float3(0);
//             o.wo = float3(0);
//             o.pdf = 0;
//             return o;
//         }

//         AnisotropicTrowbridgeReitzParameter params;
//         params.alpha_x = material.alpha_x;
//         params.alpha_y = material.alpha_y;

//         const Frame frame = i.shading_frame;
//         const float3 wi = i.shading_frame.to_local(i.wi);
//         if (AnisotropicTrowbridgeReitzDistribution::effectively_smooth(params)) {
//             float f = FresnelComplex(theta_phi_coord::AbsCosTheta(wi),
//                                      complex(material.eta, material.k));
//             o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
//             o.pdf = 1.f;
//             o.bsdf = float3(f) / o.pdf;
//             return o;
//         }
//         // Sample rough conductor BRDF
//         // Sample microfacet normal wm and reflected direction wi
//         o = microfacet_reflection::sample_vnormal<
//             AnisotropicTrowbridgeReitzDistribution>(i, params);
//         // Evaluate Fresnel factor F for conductor BRDF
//         float F = FresnelComplex(abs(dot(i.wi, o.wh)), complex(material.eta, material.k));
//         float3 wh = i.shading_frame.to_local(o.wh);
//         float3 wo = i.shading_frame.to_local(o.wo);
//         float3 f = AnisotropicTrowbridgeReitzDistribution::D(wh, params)
//                     * AnisotropicTrowbridgeReitzDistribution::G(wo, wi, params)
//                     * F / (4 * theta_phi_coord::AbsCosTheta(wi));
//         o.bsdf = f / o.pdf;
//         return o;
//     }
//     // Evaluate the PDF of the BSDF sampling
//     float pdf(ibsdf::pdf_in i, AnisoConductorMaterial material) {
//         if (dot(i.geometric_normal, i.wi) < 0 ||
//             dot(i.geometric_normal, i.wo) < 0) {
//             // No light below the surface
//             return float(0);
//         }
//         // Flip the shading frame if it is
//         // inconsistent with the geometry normal.
//         Frame frame = i.shading_frame;

//         return 0.f;
//     }
// }

#endif // !_SRENDERER_CONDUCTOR_BRDF_