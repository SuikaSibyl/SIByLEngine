#ifndef _SRENDERER_CHROMATIC_FICTION_BRDF_
#define _SRENDERER_CHROMATIC_FICTION_BRDF_

#include "bxdf.hlsli"
#include "srenderer/scene-binding.hlsli"
#include "srenderer/spt.hlsli"

struct ChromaFictionMaterial : IBxDFParameter {
    float3 eta;     // real component of IoR
    float3 k;       // imaginary component of IoR
    float3 alpha;   // roughness across channels

    __init() {}
    __init(MaterialData data, float2 uv) {
        k = data.floatvec_0.xyz;
        eta = data.floatvec_2.xyz;
        alpha = { data.floatvec_0.w, data.floatvec_1.w, data.floatvec_2.w };
    }
};

struct ChromaFictionBRDF : IBxDF {
    typedef ChromaFictionMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, ChromaFictionMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; }

        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = i.shading_frame.to_local(i.wh);
        if (wo.z < 0 || wh.z < 0) return float3(0, 0, 0);

        float3 f = float3(0, 0, 0);
        // {   // evaluate RED channel as Oren-Nayar
        //     const float sinThetaI = theta_phi_coord::SinTheta(wi);
        //     const float sinThetaO = theta_phi_coord::SinTheta(wo);
        //     // Compute cosine term of Oren–Nayar model
        //     float maxCos = 0;
        //     if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
        //         float sinPhiI = theta_phi_coord::SinPhi(wi);
        //         float cosPhiI = theta_phi_coord::CosPhi(wi);
        //         float sinPhiO = theta_phi_coord::SinPhi(wo);
        //         float cosPhiO = theta_phi_coord::CosPhi(wo);
        //         float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
        //         maxCos = max(0.0, dCos);
        //     }
        //     // Compute sine and tangent terms of Oren–Nayar model
        //     float sinAlpha; float tanBeta;
        //     if (theta_phi_coord::AbsCosTheta(wi) >
        //         theta_phi_coord::AbsCosTheta(wo)) {
        //         sinAlpha = sinThetaO;
        //         tanBeta = sinThetaI / theta_phi_coord::AbsCosTheta(wi);
        //     } else {
        //         sinAlpha = sinThetaI;
        //         tanBeta = sinThetaO / theta_phi_coord::AbsCosTheta(wo);
        //     }
        //     // Compute A and B terms of Oren–Nayar model
        //     const float3 sigma = material.alpha * k_2pi;
        //     const float3 sigma2 = sigma * sigma;
        //     const float3 A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        //     const float3 B = 0.45 * sigma2 / (sigma2 + 0.09);
        //     f = k_inv_pi * (A + B * maxCos * sinAlpha * tanBeta) * abs(wo.z);
        // }
        // Sample rough conductor BRDF
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
    static ibsdf::sample_out sample(ibsdf::sample_in i, ChromaFictionMaterial material) {
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
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
        // ibsdf::sample_out o;
        // // Flip the shading frame if it is
        // // inconsistent with the geometry normal.
        // Frame frame = i.shading_frame;
        // o.wo = frame.to_world(sample_cos_hemisphere(i.u.xy));
        // o.pdf = abs(dot(frame.n, o.wo)) * k_inv_pi;

        // // evaluate the BSDF
        // ibsdf::eval_in eval_in;
        // eval_in.wi = i.wi;
        // eval_in.wo = o.wo;
        // eval_in.geometric_normal = i.geometric_normal;
        // eval_in.shading_frame = i.shading_frame;
        // o.bsdf = eval(eval_in, material) / o.pdf;
        // return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, ChromaFictionMaterial material) {
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
        // Frame frame = i.shading_frame;
        // return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
    }
    
    // Evaluate the PDF of the BSDF sampling
    static float3 perchannel_pdf(ibsdf::pdf_in i, ChromaFictionMaterial material) {
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
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        return f;
    }
}

#endif // _SRENDERER_CHROMATIC_FICTION_BRDF_