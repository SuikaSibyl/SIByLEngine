#ifndef _SRENDERER_GGX_DIFFUSE_MIXTURE_MATERIAL_
#define _SRENDERER_GGX_DIFFUSE_MIXTURE_MATERIAL_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/scene-binding.hlsli"
#include "srenderer/spt.hlsli"
#include "srenderer/materials/orennayar.hlsli"

struct MixtureMaterial : IBxDFParameter {
    float3 R;       // Reflectance
    float weight;   // weight of the diffuse (weight) over the specular (1 - weight)
    float sigma;    // sigma of the diffuse layer (Oren-Nayar)
    float alpha;    // roughness
    float eta;      // eta of the dielectric layer
    __init() {}
    __init(MaterialData mat, float2 uv) {
        const float3 ext_tex = SampleTexture2D(
            mat.ext1_tex, uv,
            mat.is_ext1_tex_differentiable()).xyz;
        R = mat.floatvec_0.xyz * SampleTexture2D(mat.albedo_tex, uv,
            mat.is_albedo_tex_differentiable()) .rgb;

        weight = mat.floatvec_0.w * ext_tex.x;
        sigma = ext_tex.y;
        alpha = ext_tex.z;
        eta = 2.0f;
    }
    
    [BackwardDifferentiable]
    static MixtureMaterial load(no_diff MaterialData data, no_diff float2 uv) {
        const float3 ext_tex = SampleTexture2D(
            data.ext1_tex, uv,
            data.is_ext1_tex_differentiable()).xyz;
        
        MixtureMaterial material = no_diff MixtureMaterial();
        material.R = data.floatvec_0.xyz * SampleTexture2D(data.albedo_tex, uv,
            data.is_albedo_tex_differentiable()) .rgb;
        material.weight = data.floatvec_0.w * ext_tex.x;
        material.sigma = ext_tex.y;
        material.alpha = ext_tex.z;
        material.eta = 2.0f;
        return material;
    }

    [BackwardDifferentiable]
    static MixtureMaterial load_w_aux(
        no_diff MaterialData data, no_diff float2 uv,
        no_diff float5 weight_aux, no_diff float5 sigma_aux,
        no_diff float5 alpha_aux
    ) {
        const float ext_tex_r = SampleTexture2D_one_channel_ratio(
            data.ext1_tex, uv, 0, weight_aux,
            data.is_ext1_tex_differentiable());
        const float ext_tex_g = SampleTexture2D_one_channel_ratio(
            data.ext1_tex, uv, 1, sigma_aux,
            data.is_ext1_tex_differentiable());
        const float ext_tex_b = SampleTexture2D_one_channel_ratio(
            data.ext1_tex, uv, 2, alpha_aux,
            data.is_ext1_tex_differentiable());
        
        MixtureMaterial material = no_diff MixtureMaterial();
        material.R = data.floatvec_0.xyz * SampleTexture2D(data.albedo_tex, uv,
            data.is_albedo_tex_differentiable()) .rgb;
        material.weight = data.floatvec_0.w * ext_tex_r;
        material.sigma = ext_tex_g;
        material.alpha = ext_tex_b;
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
        OrenNayarMaterial on = { material.R, material.sigma };
        float3 diffuse_contrib = OrenNayarBRDF::eval(i, on);

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
            o.wh = i.shading_frame.to_world(wh);
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
        eval_in.wh = o.wh;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample_safe(ibsdf::sample_in i, MixtureMaterial material) {
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
            const float pdf = IsotropicTrowbridgeReitzDistribution::pdf_normal(wi, wh, params);
            const float VdotH = abs(dot(wi, wh));
            float spec_pdf = pdf / (4 * abs(VdotH));
            o.pdf = o.pdf * diffuse_prob + spec_pdf * spec_prob;
            o.wh = i.shading_frame.to_world(wh);
        }
        else {
            // Sample rough conductor BRDF
            // Sample microfacet normal wm and reflected direction wi
            IsotropicTrowbridgeReitzParameter params;
            params.alpha = material.alpha;
            o = microfacet_reflection::sample_normal<
                IsotropicTrowbridgeReitzDistribution>(i, params);

            float diffuse_pdf = max(dot(frame.n, o.wo), 0) * k_inv_pi;
            o.pdf = o.pdf * spec_prob + diffuse_pdf * diffuse_prob;
        }

        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.wh = o.wh;
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
        bwd_diff(MixtureMaterial::load)(mat, texcoord, material_pair.d);
    }
    
    static void backward_grad_ratio(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);

        ibsdf::pdf_in pi;
        pi.wi = i.eval.wi;
        pi.wo = i.eval.wo;
        pi.wh = i.eval.wh;
        pi.geometric_normal = i.eval.geometric_normal;
        pi.shading_frame = i.eval.shading_frame;

        float5 weight_aux;
        // weight aux 0: the gradient of the first term
        var material_pair = diffPair(material);
        bwd_diff(eval_diffuse_lobe)(i.eval, material_pair, dL);
        weight_aux.data[0] = material_pair.d.weight;
        // weight aux 1: the gradient of the second term
        material_pair = diffPair(material);
        bwd_diff(eval_spec_lobe)(i.eval, material_pair, dL);
        weight_aux.data[1] = material_pair.d.weight;
        // weight aux 2: the PDF of sampling the first term
        weight_aux.data[2] = pdf_alpha_derivative_diffuse(pi) / i.pdf;
        // weight aux 3: the PDF of sampling the second term
        weight_aux.data[3] = pdf_alpha_derivative_spec(pi, material) / i.pdf;
        // weight aux 4: estimate the actual H
        weight_aux.data[4] = 1;

        float5 sigma_aux;
        // sigma aux 0: the gradient of the first term
        material_pair = diffPair(material);
        bwd_diff(eval_diffuse_lobe_1st)(i.eval, material_pair, dL);
        sigma_aux.data[0] = material_pair.d.sigma;
        // sigma aux 1: the gradient of the second term
        material_pair = diffPair(material);
        bwd_diff(eval_diffuse_lobe_2nd)(i.eval, material_pair, dL);
        sigma_aux.data[1] = material_pair.d.sigma;
        // sigma aux 2: the PDF of sampling the first term
        sigma_aux.data[2] = OrenNayarBRDF::pdf_term1(pi) / i.pdf;
        // sigma aux 3: the PDF of sampling the second term
        sigma_aux.data[3] = OrenNayarBRDF::pdf_term2(pi) / i.pdf;
        // sigma aux 4: estimate the actual H
        sigma_aux.data[4] = 1;
        
        float5 alpha_aux;
        // alpha aux 0: the gradient of the first term
        float2 dd_dg_dalpha = manual_backward_dalpha(i.eval, material, float3(1, 1, 1));
        alpha_aux.data[0] = (max(dd_dg_dalpha.x, 0)) * average(dL);
        // sigma aux 1: the gradient of the second term
        alpha_aux.data[1] = (min(dd_dg_dalpha.x, 0)) * average(dL);
        // sigma aux 2: the PDF of sampling the first term
        alpha_aux.data[2] = pdf_ggx_alpha_derivative_pos(pi, material) / i.pdf;
        // sigma aux 3: the PDF of sampling the second term
        alpha_aux.data[3] = pdf_ggx_alpha_derivative_neg(pi, material) / i.pdf;
        // sigma aux 4: estimate the actual H
        alpha_aux.data[4] = 1;
        
        MixtureMaterial.Differential d_material;
        d_material.alpha = dd_dg_dalpha.y * average(dL);

        bwd_diff(MixtureMaterial::load_w_aux)(
            mat, texcoord, weight_aux, sigma_aux, alpha_aux, d_material);
    }

    // -------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------

    // Evaluate the BSDF
    [Differentiable]
    static float3 eval_spec_lobe(no_diff ibsdf::eval_in i, MixtureMaterial material) {
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

        return spec_contrib * (1 - material.weight);
    }

    [Differentiable]
    static float3 eval_diffuse_lobe(no_diff ibsdf::eval_in i, MixtureMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = no_diff i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        if (wo.z < 0) return float3(0, 0, 0);

        // diffuse layer:
        // In order to reflect from the diffuse layer,
        OrenNayarMaterial on = { material.R, material.sigma };
        float3 diffuse_contrib = OrenNayarBRDF::eval(i, on);

        return diffuse_contrib * material.weight;
    }

    [Differentiable]
    static float3 eval_diffuse_lobe_1st(no_diff ibsdf::eval_in i, MixtureMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = no_diff i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        if (wo.z < 0) return float3(0, 0, 0);

        // diffuse layer:
        // In order to reflect from the diffuse layer,
        OrenNayarMaterial on = { material.R, material.sigma };
        float3 diffuse_contrib = OrenNayarBRDF::eval_term1(i, on);

        return diffuse_contrib * material.weight;
    }

    [Differentiable]
    static float3 eval_diffuse_lobe_2nd(no_diff ibsdf::eval_in i, MixtureMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = no_diff i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = no_diff i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        if (wo.z < 0) return float3(0, 0, 0);

        // diffuse layer:
        // In order to reflect from the diffuse layer,
        OrenNayarMaterial on = { material.R, material.sigma };
        float3 diffuse_contrib = OrenNayarBRDF::eval_term2(i, on);

        return diffuse_contrib * material.weight;
    }

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

    static void backward_weight_derivative_diffuse(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval_diffuse_lobe)(i.eval, material_pair, dL);
        float dweight = material_pair.d.weight;

        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.weight = dweight;
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_weight_derivative_spec(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval_spec_lobe)(i.eval, material_pair, dL);
        float dweight = material_pair.d.weight;
        
        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.weight = dweight;
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_sigma_derivative_lambert(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval_diffuse_lobe_1st)(i.eval, material_pair, dL);

        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.sigma = material_pair.d.sigma;
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_sigma_derivative_nonlambert(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval_diffuse_lobe_2nd)(i.eval, material_pair, dL);
        
        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.sigma = material_pair.d.sigma;
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static float2 manual_backward_dalpha(
        ibsdf::eval_in i, MixtureMaterial material, float3 dL) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        if (wo.z < 0) return {};

        //
        dL *= 1 - material.weight;

        // Evaluate rough dilectric BRDF
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        float d = IsotropicTrowbridgeReitzDistribution::D(wh, params);
        float g = IsotropicTrowbridgeReitzDistribution::G(wo, wi, params);
        float3 F = FresnelDielectric(abs(dot(wi, wh)), material.eta);
        float3 weight = F / (4 * theta_phi_coord::AbsCosTheta(wi));

        var pair_params_d = diffPair(params);
        bwd_diff(IsotropicTrowbridgeReitzDistribution::D)(wh, pair_params_d, sum(dL * weight * g));
        
        var pair_params_g = diffPair(params);
        bwd_diff(IsotropicTrowbridgeReitzDistribution::G)(wo, wi, pair_params_g, sum(dL * weight * d));
        
        return { pair_params_d.d.alpha, pair_params_g.d.alpha };
    }

    static void backward_ggx_alpha_derivative_pos(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        float2 dd_dg_dalpha = manual_backward_dalpha(i.eval, material, float3(1, 1, 1));
        float dalpha = max(dd_dg_dalpha.x, 0) + dd_dg_dalpha.y;
        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.alpha = dalpha * average(dL);
        bwd_diff(MixtureMaterial::load)(mat, texcoord, d_material);
    }

    static void backward_ggx_alpha_derivative_neg(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        MixtureMaterial material = MixtureMaterial(mat, texcoord);
        float2 dd_dg_dalpha = manual_backward_dalpha(i.eval, material, float3(1, 1, 1));
        float dalpha = min(dd_dg_dalpha.x, 0) + dd_dg_dalpha.y;
        // postprocess the derivatives
        MixtureMaterial.Differential d_material;
        d_material.alpha = dalpha * average(dL);
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
    static ibsdf::sample_out sample_sigma_derivative_lambert(
        const ibsdf::sample_in i, MixtureMaterial material) {
        // Sample diffuse BRDF
        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(sample_cos_hemisphere(i.u.xy));
        o.pdf = abs(dot(i.shading_frame.n, o.wo)) * k_inv_pi;
        return o;
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_sigma_derivative_nonlambert(
        const ibsdf::sample_in i, MixtureMaterial material) {
        return OrenNayarBRDF::sample_sigma_derivative_spec(i, { float3(1, 1, 1), material.sigma });
    }

    // evaluate the PDF of the first term of the BSDF
    // here we simply use cosine-weighted hemisphere sampling
    static float pdf_alpha_derivative_diffuse(ibsdf::pdf_in i) {
        Frame frame = i.shading_frame;
        return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
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

    // evaluate the PDF of the first term of the BSDF
    // here we simply use cosine-weighted hemisphere sampling
    static float pdf_alpha_derivative_spec(ibsdf::pdf_in i, MixtureMaterial material) {
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        return microfacet_reflection::pdf_vnormal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::sample_out sample_ggx_alpha_derivative_pos(
        const ibsdf::sample_in i, MixtureMaterial material) {
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
    static ibsdf::sample_out sample_ggx_alpha_derivative_neg(
        const ibsdf::sample_in i, MixtureMaterial material) {
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

    /** The pdf of postivized derivative sampling, positive part */
    static float pdf_ggx_alpha_derivative_pos(const ibsdf::pdf_in i, MixtureMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        return microfacet_reflection::pdf_pos<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param) / (4 * abs(dot(i.wh, i.wi)));
    }

    /** The pdf of postivized derivative sampling, negative part */
    static float pdf_ggx_alpha_derivative_neg(const ibsdf::pdf_in i, MixtureMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        return microfacet_reflection::pdf_neg<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param) / (4 * abs(dot(i.wh, i.wi)));
    }
};

#endif // _SRENDERER_GGX_DIFFUSE_MIXTURE_MATERIAL_