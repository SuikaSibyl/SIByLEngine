#ifndef _SRENDERER_PLASTIC_MATERIAL_
#define _SRENDERER_PLASTIC_MATERIAL_

#include "common/sampling.slang"
#include "srenderer/materials/bxdf-microfacet.slang"
#include "srenderer/spt-bindings.slang"
#include "srenderer/spt-definition.slang"

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

    __init() {
        Kd = float3(0.5, 0.5, 0.5);
        Ks = float3(0.5, 0.5, 0.5);
        alpha = 0.1; // roughness
        eta = 1.5;   // eta of the dielectric layer
    }
    __init(MaterialData mat, float2 uv) {
        Kd = mat.floatvec_0.xyz; // * sampleTexture(mat.albedo_tex, uv).rgb;
        Ks = mat.floatvec_2.xyz; // * sampleTexture(mat.albedo_tex, uv).rgb;
        alpha = mat.floatvec_1.w;
        eta = mat.floatvec_2.w;
        if (mat.albedo_tex >= 0) {
            Sampler2D albedo_tex = scene_read_texture(mat.albedo_tex);
            Kd *= albedo_tex.Sample(uv).rgb;
        }
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
        const float F_o = fresnel_dielectric(abs(dot(wi, wh)), material.eta);
        const float F_i = fresnel_dielectric(abs(dot(wo, wh)), material.eta);

        float3 diffuse_contrib = (1.f - F_o) * (1.f - F_i) / M_PI;
        diffuse_contrib *= theta_phi_coord::abs_cos_theta(wi) * material.Kd;

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
            o.pdf = abs(dot(frame.n, o.wo)) * M_INV_PI;

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

            float diffuse_pdf = max(dot(frame.n, o.wo), 0) * M_INV_PI;
            o.pdf = o.pdf * spec_prob + diffuse_pdf * diffuse_prob;
        }

        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material);
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
        float diffuse_pdf = max(dot(frame.n, i.wo), float(0)) * M_INV_PI;

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
        float3 F = fresnel_dielectric(abs(dot(wi, wh)), eta);
        float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                    * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::abs_cos_theta(wi));
        return f;
    }
};

#endif // _SRENDERER_PLASTIC_MATERIAL_