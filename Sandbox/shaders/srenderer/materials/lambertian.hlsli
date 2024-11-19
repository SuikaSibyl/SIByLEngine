#ifndef _SRENDERER_LAMBERTIAN_MATERIAL_
#define _SRENDERER_LAMBERTIAN_MATERIAL_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"
#include "srenderer/scene-binding.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Lambertian Diffuse Material
// ----------------------------------------------------------------------------------------
// A very simple diffuse material that follows the Lambertian reflectance model.
///////////////////////////////////////////////////////////////////////////////////////////

struct LambertMaterial : IBxDFParameter {
    __init() { R = float3(1.f); }
    __init(float3 r) { R = r; }
    __init(MaterialData mat, float2 uv) {
        R = mat.floatvec_0.xyz * SampleTexture2D(mat.albedo_tex, uv,
            mat.is_albedo_tex_differentiable()).rgb;
    }

    float3 R; // reflectance
    
    [BackwardDifferentiable]
    static LambertMaterial load(no_diff MaterialData mat, no_diff float2 uv) {
        LambertMaterial material = no_diff LambertMaterial();
        material.R = mat.floatvec_0.xyz * 
            SampleTexture2D(mat.albedo_tex, uv,
                mat.is_albedo_tex_differentiable()).rgb;
        return material;
    }
};

struct LambertianBRDF : IBxDF {
    typedef LambertMaterial TParam;
    
    // Evaluate the BSDF
    [Differentiable]
    static float3 eval(no_diff ibsdf::eval_in i, LambertMaterial material) {
        Frame frame = i.shading_frame;
        // Lambertian BRDF
        return max(dot(frame.n, i.wo), 0.f) *
               material.R * k_inv_pi;
    }

    static void backward_grad(
        ibsdf::bwd_in i, float3 dL,
        MaterialData mat,
        float2 texcoord
    ) {
        LambertMaterial material = LambertMaterial(mat, texcoord);
        var material_pair = diffPair(material);
        bwd_diff(eval)(i.eval, material_pair, dL);
        bwd_diff(LambertMaterial::load)(mat, texcoord, material_pair.d);
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, LambertMaterial material) {
        ibsdf::sample_out o;
        // Flip the shading frame if it is
        // inconsistent with the geometry normal.
        Frame frame = i.shading_frame;
        o.wo = frame.to_world(sample_cos_hemisphere(i.u.xy));
        o.pdf = abs(dot(frame.n, o.wo)) * k_inv_pi;
        
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
    static float pdf(ibsdf::pdf_in i, LambertMaterial material) {
        Frame frame = i.shading_frame;
        return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
    }
}

#endif // !_SRENDERER_LAMBERTIAN_MATERIAL_