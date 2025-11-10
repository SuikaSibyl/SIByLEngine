#ifndef _SRENDERER_CHROMATIC_FICTION_BRDF_
#define _SRENDERER_CHROMATIC_FICTION_BRDF_

#include "common/sampling.slang"
#include "srenderer/materials/bxdf-microfacet.slang"
#include "srenderer/spt-definition.slang"

struct ChromaGGXMaterial : IBxDFParameter {
    float3 eta;     // real component of IoR
    float3 k;       // imaginary component of IoR
    float3 alpha;   // roughness across channels

    __init() {
        eta = float3(0.200f, 0.924f, 1.102f);
        k = float3(3.912f, 2.452f, 2.142f);
        alpha = float3(0.1f, 0.1f, 0.1f);
    }
    __init(MaterialData data, float2 uv) {
        k = data.floatvec_0.xyz;
        eta = data.floatvec_2.xyz;
        alpha = { data.floatvec_0.w, data.floatvec_1.w, data.floatvec_2.w };
    }
};

struct ChromaGGXBRDF : IBxDF {
    typedef ChromaGGXMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, ChromaGGXMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = i.wh.hasValue 
            ? i.shading_frame.to_local(i.wh.value) 
            : normalize(wi + wo);
        if (wo.z < 0 || wh.z < 0) return float3(0, 0, 0);

        float3 f = float3(0, 0, 0);
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        
        for (int i = 0; i < 3; ++i) {
            params.alpha = material.alpha[i];
            f[i] = eval_isotropic_ggx_conductor(wi, wo, 
                wh, material.eta, material.k, params)[i];            
        }
        return f;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, ChromaGGXMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        ibsdf::sample_out o;

        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha[clamp(int(i.u.z * 3), 0, 2)];
        o = microfacet_reflection::sample_vnormal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
        // evaluate the PDF
        ibsdf::pdf_in pdf_in;
        pdf_in.wi = i.wi;
        pdf_in.wo = o.wo;
        pdf_in.geometric_normal = i.geometric_normal;
        pdf_in.shading_frame = i.shading_frame;
        pdf_in.wh = o.wh;
        o.pdf = pdf(pdf_in, material);
        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        eval_in.wh = o.wh;
        o.bsdf = eval(eval_in, material);
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, ChromaGGXMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = i.shading_frame.to_local(i.wh);
        
        IsotropicTrowbridgeReitzParameter params;
        float pdf = 0;
        for (int i = 0; i < 3; ++i) {
            params.alpha = material.alpha[i];
            pdf += discard_nan_inf(IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wi, wh, params) / 3);
        }
        const float VdotH = abs(dot(wi, wh));
        return pdf / (4 * abs(VdotH));
    }
    
    // Evaluate the PDF of the BSDF sampling
    static float3 perchannel_pdf(ibsdf::pdf_in i, ChromaGGXMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = i.shading_frame.to_local(i.wh);
        
        IsotropicTrowbridgeReitzParameter params;
        float3 pdf = 0;
        for (int i = 0; i < 3; ++i) {
            params.alpha = material.alpha[i];
            pdf[i] = (IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wi, wh, params));
        }
        float3 ratio = pdf / average(pdf) * 0.99 + 0.01;
        return ratio;
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
                    * F / (4 * theta_phi_coord::abs_cos_theta(wi));
        return f;
    }
}

#endif // _SRENDERER_CHROMATIC_FICTION_BRDF_