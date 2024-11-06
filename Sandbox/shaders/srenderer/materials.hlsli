#ifndef _SRENDERER_MATERIAL_HEADER_
#define _SRENDERER_MATERIAL_HEADER_

#include "materials/lambertian.hlsli"
#include "materials/conductor.hlsli"
#include "materials/plastic.hlsli"

namespace materials {
ibsdf::sample_out bsdf_sample(ibsdf::sample_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::sample(i, ConductorMaterial(material));
    case 2: return PlasticBRDF::sample(i, PlasticMaterial(material, uv));
    }
    return o;
}

float bsdf_sample_pdf(ibsdf::pdf_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::pdf(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::pdf(i, ConductorMaterial(material));
    case 2: return PlasticBRDF::pdf(i, PlasticMaterial(material, uv));
    }
    return 0.f;
}

float3 bsdf_eval(ibsdf::eval_in i, MaterialData material, float2 uv) {
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::eval(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::eval(i, ConductorMaterial(material));
    case 2: return PlasticBRDF::eval(i, PlasticMaterial(material, uv));
    }
    return float3(0, 0, 0);
}

void bsdf_backward_grad(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: LambertianBRDF::backward_grad(i, dL, material, uv); break;
    }
}

float3 albedo(MaterialData material, float2 uv) {
    return material.floatvec_0.xyz *
           SampleTexture2D(material.albedo_tex, uv, 
            material.is_albedo_tex_differentiable()).rgb; 
}
float3 emission(MaterialData material) { return material.floatvec_1.xyz; }
}

#endif // _SRENDERER_MATERIAL_HEADER_