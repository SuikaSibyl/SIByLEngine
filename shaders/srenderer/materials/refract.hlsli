#ifndef _SRENDERER_REFRACT_MATERIAL_BRDF_
#define _SRENDERER_REFRACT_MATERIAL_BRDF_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/scene-binding.hlsli"

struct RefractMaterial : IBxDFParameter {
    float eta;
    
    __init() { eta = 2.f; }
    __init(MaterialData mat, float2 uv) {
        eta = mat.floatvec_0.w;
    }
};

struct RefractBRDF : IBxDF {
    typedef RefractMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, RefractMaterial material) {
        if (dot(i.geometric_normal, i.wi) < 0 ||
            dot(i.geometric_normal, i.wo) < 0) {
            // No light below the surface
            return float3(0);
        }
        Frame frame = i.shading_frame;
        // Lambertian BRDF
        return 0.f;
    }
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, RefractMaterial material) {
        ibsdf::sample_out o;

        const Frame frame = i.shading_frame;
        const float3 wi = i.shading_frame.to_local(i.wi);
        // Sample perfect specular dielectric BSDF
        const float R = FresnelDielectric(theta_phi_coord::CosTheta(wi), material.eta);
        const float T = 1 - R;
        // // Compute probabilities pr and pt for sampling reflection and transmission
        // const float pr = R; const float pt = T;
        // if (pr == 0 && pt == 0) return o;

        // Sample perfect specular dielectric BTDF
        // Compute ray direction for specular transmission
        float3 wo = safe_refract(wi, float3(0, 0, 1), material.eta);
        if (all(wo == 0)) return o;
        
        const float3 ft = T;
        // Account for non-symmetry with transmission to different medium
        o.wo = frame.to_world(wo);
        // o.pdf = pt / (pr + pt);
        o.pdf = 1.f;
        o.bsdf = ft / o.pdf;
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, RefractMaterial material) {
        return 0.f;
    }
}

#endif // !_SRENDERER_REFRACT_MATERIAL_BRDF_