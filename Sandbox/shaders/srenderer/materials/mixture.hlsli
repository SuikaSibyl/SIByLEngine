#ifndef _SRENDERER_GGX_DIFFUSE_MIXTURE_MATERIAL_
#define _SRENDERER_GGX_DIFFUSE_MIXTURE_MATERIAL_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/scene-binding.hlsli"
#include "srenderer/spt.hlsli"

struct MixtureMaterial : IBxDFParameter {
    float weight;   // weight of the diffuse material
    float alpha;    // roughness
    float eta;      // eta of the dielectric layer

    __init() {}
    __init(MaterialData mat, float2 uv) {
        const float ext_tex_r = SampleTexture2D(
            mat.ext1_tex, uv,
            mat.is_ext1_tex_differentiable()) .r;

        weight = mat.floatvec_0.w * ext_tex_r;
        alpha = 0.2f;
        eta = 2.0f;
    }
    
    [BackwardDifferentiable]
    static MixtureMaterial load(no_diff MaterialData data, no_diff float2 uv) {
        const float ext_tex_r = SampleTexture2D(
            data.ext1_tex, uv,
            data.is_ext1_tex_differentiable()).r;
        
        MixtureMaterial material = no_diff MixtureMaterial();
        material.weight = data.floatvec_0.w * ext_tex_r;
        material.alpha = 0.2f;
        material.eta = 2.0f;
        return material;
    }
};

struct MixtureBRDF : IBxDF {
    typedef MixtureMaterial TParam;

    // Evaluate the BSDF
    [Differentiable]
    static float3 eval(no_diff ibsdf::eval_in i, MixtureMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = no_diff i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        if (wo.z < 0) return float3(0, 0, 0);

        // Evaluate rough dilectric BRDF
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        const float3 spec_contrib = eval_isotropic_ggx_dielectric(wi, wo, wh, material.eta, params);

        // diffuse layer:
        // In order to reflect from the diffuse layer,
        float3 diffuse_contrib = 1. / k_pi;
        diffuse_contrib *= theta_phi_coord::AbsCosTheta(wo);

        return spec_contrib * (1 - material.weight)
             + diffuse_contrib * material.weight;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, MixtureMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        ibsdf::sample_out o;

        float lS = 1 - material.weight;
        float lR = material.weight;
        float diffuse_prob = lR / (lR + lS);
        float spec_prob = lS / (lR + lS);

        if (i.u.z < lR / (lR + lS)) {
            // Sample diffuse BRDF
            o.wo = frame.to_world(sample_cos_hemisphere(i.u.xy));
            o.pdf = abs(dot(frame.n, o.wo)) * k_inv_pi;

            const float3 wo = i.shading_frame.to_local(o.wo);
            const float3 wh = normalize(wi + wo);

            IsotropicTrowbridgeReitzParameter params;
            params.alpha = material.alpha;
            const float pdf = IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wi, wh, params);
            const float VdotH = abs(dot(wi, wh));
            float spec_pdf = pdf / (4 * abs(VdotH));
            o.pdf = o.pdf * diffuse_prob + spec_pdf * spec_prob;
        }
        else {
            // Sample rough conductor BRDF
            // Sample microfacet normal wm and reflected direction wi
            IsotropicTrowbridgeReitzParameter params;
            params.alpha = material.alpha;
            o = microfacet_reflection::sample_vnormal<
                IsotropicTrowbridgeReitzDistribution>(i, params);

            float diffuse_pdf = max(dot(frame.n, o.wo), 0) * k_inv_pi;
            o.pdf = o.pdf * spec_prob + diffuse_pdf * diffuse_prob;
        }

        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, MixtureMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        const float3 wh = normalize(wi + wo);

        float lS = 1 - material.weight;
        float lR = material.weight;
        float diffuse_prob = lR / (lR + lS);
        float spec_prob = lS / (lR + lS);

        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        const float pdf = IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wi, wh, params);
        const float VdotH = abs(dot(wi, wh));
        float spec_pdf = pdf / (4 * abs(VdotH));
        float diffuse_pdf = max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
        
        return spec_pdf * spec_prob + diffuse_pdf * diffuse_prob;
    }
    
    static void backward_grad(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval)(i.eval, material_pair, dL);
        float dweight = material_pair.d.weight;
        
        MixtureMaterial.Differential d_material;
        d_material.alpha = 0.f;
        d_material.eta = 0.f;
        d_material.weight = dweight;

        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_grad_w_aux(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        // MixtureMaterial material = MixtureMaterial(mat, texcoord);
        // float2 dd_dg_dweight = manual_diff_weight(i.eval, material, dL);
        // // float dalpha = dd_dg_dweight.x;

        // MixtureMaterial.Differential d_material;
        // d_material.alpha = 0.f;
        // d_material.eta = 0.f;
        // d_material.weight = dweight;
        
        // bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    // -------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------
    [Differentiable]
    static float3 eval_isotropic_ggx_dielectric(
        no_diff float3 wi,
        no_diff float3 wo,
        no_diff float3 wh,
        float eta,
        IsotropicTrowbridgeReitzParameter params
    ) {
        // Evaluate Fresnel factor F for conductor BRDF
        float3 F = FresnelDielectric(abs(dot(wi, wh)), eta);
        float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                    * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        return f;
    }

    static float2 manual_diff_weight(
        no_diff ibsdf::eval_in i, MixtureMaterial material, float3 dL
    ) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        
        // Evaluate rough dilectric BRDF
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        const float3 spec_contrib = eval_isotropic_ggx_dielectric(wi, wo, wh, material.eta, params);

        // diffuse layer:
        // In order to reflect from the diffuse layer,
        float3 diffuse_contrib = 1. / k_pi;
        diffuse_contrib *= theta_phi_coord::AbsCosTheta(wo);
        
        return float2(
            sum(diffuse_contrib * dL),
            - sum(spec_contrib * dL)
        );
    }

    static void backward_alpha_derivative_diffuse(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        float2 dd_dg_dweight = manual_diff_weight(i.eval, material, float3(1, 1, 1));
        float dalpha = dd_dg_dweight.x;
        
        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.weight = dalpha * average(dL);
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_alpha_derivative_spec(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        float2 dd_dg_dweight = manual_diff_weight(i.eval, material, float3(1, 1, 1));
        float dalpha = dd_dg_dweight.y;

        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.weight = dalpha * average(dL);
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }
    
    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_alpha_derivative_diffuse(
        const ibsdf::sample_in i, MixtureMaterial material) {
        // Sample diffuse BRDF
        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(sample_cos_hemisphere(i.u.xy));
        o.pdf = abs(dot(i.shading_frame.n, o.wo)) * k_inv_pi;
        return o;
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_alpha_derivative_spec(
        const ibsdf::sample_in i, MixtureMaterial material) {
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        ibsdf::sample_out o = microfacet_reflection::sample_vnormal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
        return o;
    }
};

#endif // _SRENDERER_GGX_DIFFUSE_MIXTURE_MATERIAL_