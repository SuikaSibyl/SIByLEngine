#ifndef _SRENDERER_DIELECTRIC_BRDF_
#define _SRENDERER_DIELECTRIC_BRDF_

#include "bxdf.hlsli"
#include "srenderer/scene-binding.hlsli"
#include "srenderer/spt.hlsli"

struct DielectricMaterial : IBxDFParameter {
    float eta; // IoR
    float alpha;

    __init() {
        alpha = 0.1f;
        eta = 1.5f;
    }
    __init(MaterialData data, float2 uv) {
        alpha = data.floatvec_0.w;
        eta = data.floatvec_2.w;
    }
};

struct DielectricBRDF : IBxDF {
    typedef DielectricMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, DielectricMaterial material) {
        if (material.eta == 1 || IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha))
            return float3(0.f, 0.f, 0.f);
        
        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        
        // Evaluate rough dielectric BSDF
        // Compute generalized half vector _wm_
        float cosTheta_o = theta_phi_coord::CosTheta(wo);
        float cosTheta_i = theta_phi_coord::CosTheta(wi);
        bool reflect = cosTheta_i * cosTheta_o > 0;
        float etap = 1;
        if (!reflect) etap = cosTheta_o > 0 ? material.eta : (1 / material.eta);

        float3 wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || length_squared(wm) == 0)
            return float3(0.f, 0.f, 0.f);
        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        // Discard backfacing microfacets
        if (dot(wm, wi) * cosTheta_i < 0 || dot(wm, wo) * cosTheta_o < 0)
            return float3(0.f, 0.f, 0.f);

        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        float F = FresnelDielectric(dot(wo, wm), material.eta);
        if (reflect) {
            // Compute reflection at rough dielectric interface
            return float3(IsotropicTrowbridgeReitzDistribution::D(wm, params) *
                          IsotropicTrowbridgeReitzDistribution::G(wo, wi, params) * F /
                        abs(4 * cosTheta_i * cosTheta_o));
        } else {
            // Compute transmission at rough dielectric interface
            float denom = sqr(dot(wi, wm) + dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
            float ft = IsotropicTrowbridgeReitzDistribution::D(wm, params) * (1 - F) *
                       IsotropicTrowbridgeReitzDistribution::G(wo, wi, params) *
                       abs(dot(wi, wm) * dot(wo, wm) / denom);
            return ft;
        }
    }

    static float3 FaceForward(float3 v, float3 n2) { return (dot(v, n2) < 0.f) ? -v : v; }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, DielectricMaterial material) {
        ibsdf::sample_out o;
        o.bsdf = float3(0, 0, 0);
        o.wo = float3(0, 0, 0);
        o.pdf = 0;

        if (material.eta == 1 || IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            const Frame frame = i.shading_frame;
            const float3 wi = i.shading_frame.to_local(i.wi);
            // Sample perfect specular dielectric BSDF
            const float R = FresnelDielectric(theta_phi_coord::CosTheta(wi), material.eta);
            const float T = 1 - R;
            // Compute probabilities pr and pt for sampling reflection and transmission
            const float pr = R; const float pt = T;
            if (pr == 0 && pt == 0) return o;

            if (i.u.z < pr / (pr + pt)) {
                // Sample perfect specular dielectric BRDF
                o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
                const float3 fr = R;
                o.pdf = pr / (pr + pt);
                o.bsdf = fr / o.pdf;
                return o;
            } else {
                // Sample perfect specular dielectric BTDF
                // Compute ray direction for specular transmission
                float3 wo = safe_refract(wi, float3(0, 0, 1), material.eta);
                if (all(wo == 0)) return o;

                const float3 ft = T;
                // Account for non-symmetry with transmission to different medium
                o.wo = frame.to_world(wo);
                o.pdf = pt / (pr + pt);
                o.bsdf = ft / o.pdf;
                return o;
            }
        } else {
            // Sample rough dielectric BSDF
            IsotropicTrowbridgeReitzParameter params;
            params.alpha = material.alpha;
            ibsdf::sample_out o = microfacet_reflection::sample_vnormal<
                IsotropicTrowbridgeReitzDistribution>(i, params);
            
            const float3 wi = i.shading_frame.to_local(i.wi);
            const float R = FresnelDielectric(dot(i.wi, o.wh), material.eta);
            const float T = 1 - R;
            const float pr = R; const float pt = T;
            if (i.u.z < pr / (pr + pt)) {
                // Sample reflection at rough dielectric interface
                float3 wh = i.shading_frame.to_local(o.wh);
                float3 wo = i.shading_frame.to_local(o.wo);
                if(wo.z < 0) {
                    o.bsdf = float3(0);
                    o.wo = float3(0);
                    o.pdf = 0;
                    return o;
                }
                float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                     * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                     * R / (4 * theta_phi_coord::AbsCosTheta(wi));
                o.pdf = o.pdf_wh / (4 * abs(dot(wi, wh))) * pr / (pr + pt);
                o.bsdf = f / o.pdf;
                return o;
            } else {
                // Sample transmission at rough dielectric interface
                float eta = material.eta;
                float3 wh = i.shading_frame.to_local(o.wh);
                float3 wo = safe_refract(wi, wh, eta);
                if (all(wo == 0)) {
                    o.bsdf = float3(0);
                    o.wo = float3(0);
                    o.pdf = 0;
                    return o;
                }
                o.wo = i.shading_frame.to_world(wo);
                // Compute PDF of rough dielectric transmission
                float denom = sqr(dot(wo, wh) + dot(wi, wh) / eta);
                float dwm_dwi = abs(dot(wo, wh)) / denom;
                o.pdf = o.pdf_wh * dwm_dwi * pt / (pr + pt);
                // Evaluate BRDF and return BSDFSample for rough transmission
                float3 ft = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                     * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                     * T * abs(dot(wi, wh) * dot(wo, wh) / (theta_phi_coord::AbsCosTheta(wi) * denom));
                // Account for non-symmetry with transmission to different medium
                ft /= sqr(eta);
                o.bsdf = ft / o.pdf;
            }
            return o;
        }
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, DielectricMaterial material) {
        if (material.eta == 1 || IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha))
            return 0.f;

        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);

        // Evaluate rough dielectric BSDF
        // Compute generalized half vector _wm_
        float cosTheta_o = theta_phi_coord::CosTheta(wo);
        float cosTheta_i = theta_phi_coord::CosTheta(wi);
        bool reflect = cosTheta_i * cosTheta_o > 0;
        float etap = 1;
        if (!reflect) etap = cosTheta_o > 0 ? material.eta : (1 / material.eta);

        float3 wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || length_squared(wm) == 0)
            return 0.f;
        wm = FaceForward(normalize(wm), float3(0, 0, 1));

        // Discard backfacing microfacets
        if (dot(wm, wi) * cosTheta_i < 0 || dot(wm, wo) * cosTheta_o < 0)
            return 0.f;

        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        // Determine Fresnel reflectance of rough dielectric boundary
        float R = FresnelDielectric(dot(wo, wm), material.eta);
        float T = 1 - R;

        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        float pr = R; float pt = T;
        // if (!(sampleFlags & BxDFReflTransFlags::Reflection))
        //     pr = 0;
        // if (!(sampleFlags & BxDFReflTransFlags::Transmission))
        //     pt = 0;
        // if (pr == 0 && pt == 0)
        //     return {};

        // Return PDF for rough dielectric
        float pdf;
        if (reflect) {
            // Compute PDF of rough dielectric reflection
            pdf = IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wo, wm, params) / (4 * abs(dot(wo, wm))) * pr / (pr + pt);
        } else {
            // Compute PDF of rough dielectric transmission
            float denom = sqr(dot(wi, wm) + dot(wo, wm) / etap);
            float dwm_dwi = abs(dot(wi, wm)) / denom;
            pdf = IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wo, wm, params) * dwm_dwi * pt / (pr + pt);
        }
        return pdf;
    }
}

#endif // !_SRENDERER_DIELECTRIC_BRDF_