#ifndef _SRENDERER_GGX_BRDF_MATERIAL_
#define _SRENDERER_GGX_BRDF_MATERIAL_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/scene-binding.hlsli"
#include "srenderer/spt.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Plastic Material
// ----------------------------------------------------------------------------------------
// Plastic can be modeled as a mixture of a diffuse and glossy scattering
// function with parameters controlling the particular colors and
// specular highlight size. The parameters to PlasticMaterial are
// two reflectivities, Kd and Ks, which respectively control the
// amounts of diffuse reflection and glossy specular reflection.
///////////////////////////////////////////////////////////////////////////////////////////

struct PlasticMaterial : IBxDFParameter {
    float3 Kd;
    float3 Ks;
    float alpha; // roughness
    float eta;   // eta of the dielectric layer
    
    __init() {}
    __init(MaterialData mat, float2 uv) {
        Kd = mat.floatvec_0.xyz * sampleTexture(mat.albedo_tex, uv).rgb;
        Ks = mat.floatvec_2.xyz * sampleTexture(mat.ext1_tex, uv).rgb;
        alpha = mat.floatvec_0.w;
        eta = mat.floatvec_2.w;
    }
};

struct PlasticBRDF : IBxDF {
    typedef PlasticMaterial TParam;
    
    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, PlasticMaterial material) {
        const Frame frame = i.shading_frame;
        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        if (wo.z < 0 || wi.z < 0) return float3(0);
        
        // Evaluate rough dilectric BRDF
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        const float3 ggx_contrib = eval_isotropic_ggx_dielectric(wi, wo, wh, material.eta, params);
        const float3 spec_contrib = material.Ks * ggx_contrib;

        // diffuse layer:
        // In order to reflect from the diffuse layer,
        // the photon needs to bounce through the dielectric layers twice.
        // The transmittance is computed by 1 - fresnel.
        const float F_o = FresnelDielectric(abs(dot(wi, wh)), average(material.eta));
        const float F_i = FresnelDielectric(abs(dot(wo, wh)), average(material.eta));

        float3 diffuse_contrib = (1.f - F_o) * (1.f - F_i) / k_pi;
        diffuse_contrib *= theta_phi_coord::AbsCosTheta(wi) * material.Kd;

        return spec_contrib + diffuse_contrib;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, PlasticMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        ibsdf::sample_out o;

        float lS = luminance(material.Ks);
        float lR = luminance(material.Kd);
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
    static float pdf(ibsdf::pdf_in i, PlasticMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        const float3 wh = normalize(wi + wo);

        float lS = luminance(material.Ks);
        float lR = luminance(material.Kd);
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
};

// struct PlasticBRDFDerivative : IBxDFDerivative {
//     typedef PlasticMaterial TParam;

//     /** Backward derivative of bxdf evaluation */
//     static void bwd_eval(const ibsdf::eval_in i, inout DifferentialPair<PlasticMaterial> param) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = param.p.alpha;
//         DielectricFresnel fresnel = DielectricFresnel(1.0, 2.0);
//         var d_ggx_param = microfacet_reflection::bwd_eval<IsotropicTrowbridgeReitzDistribution>(
//             i, fresnel, ggx_param, float3(1.f));
//         // accumulate the derivatives
//         PlasticMaterial.Differential dparam = param.d;
//         dparam.alpha += d_ggx_param.alpha;
//         param = diffPair(param.p, dparam);
//     }
//     /** sample and compute brdfd with primal importance sample sampling */
//     static ibsdf::dsample_out<PlasticMaterial> sample_primal(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         // sample the primal BRDF
//         ibsdf::sample_out sample_o = PlasticBRDF::sample(i, material);
//         ibsdf::dsample_out<PlasticMaterial> o;
//         o.wo = sample_o.wo;
//         o.pdf = sample_o.pdf;
//         // evaluate the BRDF derivative
//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;
//         DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
//         PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
//         o.dparam = material_pair.d;
//         o.dparam.alpha /= o.pdf; // divide by the pdf
//         if (o.pdf == 0) o.dparam.alpha = 0.f;
//         if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;
//         // reject samples below the surface
//         const float3 wo = i.shading_frame.to_local(o.wo);
//         if (wo.z < 0.f || o.pdf == 0.f) {
//             o.dparam.alpha = 0.f;
//         }
//         return o;
//     }

//     /** sample but not compute brdfd with primal importance sample sampling */
//     static ibsdf::dsample_noeval_out sample_noeval_primal(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         // sample the primal BRDF
//         IsotropicTrowbridgeReitzParameter params;
//         params.alpha = material.alpha;
//         ibsdf::sample_out sample_o = microfacet_reflection::sample_normal<
//             IsotropicTrowbridgeReitzDistribution>(i, params);
//         ibsdf::dsample_noeval_out o;
//         o.wo = sample_o.wo;
//         o.pdf = sample_o.pdf;
//         o.wh = sample_o.wh;
//         o.pdf_wh = sample_o.pdf_wh;
//         return o;
//     }

//     /** The pdf of postivized derivative sampling, positive part */
//     static ibsdf::dsample_out<PlasticMaterial> sample_pos_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
        
//         float4 sample_pdf = microfacet_reflection::sample_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_out<PlasticMaterial> o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;
        
//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;

//         DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
//         PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
        
//         o.dparam = material_pair.d;
//         o.dparam.alpha /= o.pdf;
//         if (o.pdf == 0) o.dparam.alpha = 0.f;
//         if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;

//         // reject samples below the surface
//         const float3 wo = i.shading_frame.to_local(o.wo);
//         if (wo.z < 0.f || o.pdf == 0.f) {
//             o.dparam.alpha = 0.f;
//         }

//         return o;
//     }

//     /** sample but not compute brdfd with postivized derivative sampling, positive */
//     static ibsdf::dsample_noeval_out sample_noeval_pos_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
        
//         float4 sample_pdf = microfacet_reflection::sample_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_noeval_out o;
        
//         const float3 wi = i.shading_frame.to_local(i.wi);
//         const float3 wh = sample_pdf.xyz;
//         o.wo = i.shading_frame.to_world(reflect(-wi, wh));
//         o.pdf = sample_pdf.w / (4 * abs(dot(wi, wh)));
//         o.wh = i.shading_frame.to_world(wh);
//         o.pdf_wh = sample_pdf.w;
//         return o;
//     }

//     /** The pdf of postivized derivative sampling, negative part */
//     static ibsdf::dsample_out<PlasticMaterial> sample_neg_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_out<PlasticMaterial> o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;

//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;

//         DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
//         PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
        
//         o.dparam = material_pair.d;
//         o.dparam.alpha /= o.pdf;
//         if (o.pdf == 0) o.dparam.alpha = 0.f;
//         if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;
        
//         // reject samples below the surface
//         const float3 wo = i.shading_frame.to_local(o.wo);
//         if (wo.z < 0.f || o.pdf == 0.f) {
//             o.dparam.alpha = 0.f;
//         }
        
//         return o;
//     }

//     /** sample but not compute brdfd with postivized derivative sampling, negative */
//     static ibsdf::dsample_noeval_out sample_noeval_neg_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
        
//         float4 sample_pdf = microfacet_reflection::sample_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_noeval_out o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;
//         return o;
//     }

//     /** The pdf of postivized derivative sampling, positive part */
//     static float pdf_pos_derivative(const ibsdf::pdf_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
//         return microfacet_reflection::pdf_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//     }
//     /** The pdf of postivized derivative sampling, negative part */
//     static float pdf_neg_derivative(const ibsdf::pdf_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
//         return microfacet_reflection::pdf_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//     }
// };

#endif // _SRENDERER_GGX_BRDF_MATERIAL_