#ifndef _SRENDERER_MATERIAL_HEADER_
#define _SRENDERER_MATERIAL_HEADER_

#include "materials/lambertian.hlsli"
#include "materials/conductor.hlsli"
#include "materials/plastic.hlsli"
#include "materials/orennayar.hlsli"
#include "materials/mixture.hlsli"
#include "materials/refract.hlsli"
#include "materials/rglbrdf.hlsli"

/**
 * "materials" namespace contains unified interface to evaluate and sample BSDFs.
 *  The list of supported BSDFs are:
 *   * 0: Lambertian
 *   * 1: Conductor (one GGX lobe with complex fresnel)
 *   * 2: Plastic
 *   * 3: Oren-Nayar (Lambertian with roughness)
 */

namespace materials {
/**
 * Evaluate the primal BSDF
 * @param i: input evaluate information
 * @param material: material data
 * @param uv: texture coordinate
 */
float3 bsdf_eval(ibsdf::eval_in i, MaterialData material, float2 uv) {
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::eval(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::eval(i, ConductorMaterial(material, uv));
    case 2: return PlasticBRDF::eval(i, PlasticMaterial(material, uv));
    case 3: return OrenNayarBRDF::eval(i, OrenNayarMaterial(material, uv));
    case 4: return MixtureBRDF::eval(i, MixtureMaterial(material, uv));
    case 5: return RefractBRDF::eval(i, RefractMaterial(material, uv));
    }
    return float3(0, 0, 0);
}

/**
 * Sample the primal BSDF
 * @param i: input sample information
 * @param material: material data
 * @param uv: texture coordinate
 */
ibsdf::sample_out bsdf_sample(ibsdf::sample_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::sample(i, ConductorMaterial(material, uv));
    case 2: return PlasticBRDF::sample(i, PlasticMaterial(material, uv));
    case 3: return OrenNayarBRDF::sample(i, OrenNayarMaterial(material, uv));
    case 4: return MixtureBRDF::sample(i, MixtureMaterial(material, uv));
    case 5: return RefractBRDF::sample(i, RefractMaterial(material, uv));
    }
    return o;
}

/**
 * Sample the primal BSDF
 * @param i: input sample information
 * @param material: material data
 * @param uv: texture coordinate
 */
ibsdf::sample_out bsdf_sample_with_perchannel_cv(
    ibsdf::sample_in i, MaterialData material, float2 uv, out float3 cv) {
    cv = float3(1, 1, 1);
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::sample(i, ConductorMaterial(material, uv));
    case 2: return PlasticBRDF::sample(i, PlasticMaterial(material, uv));
    case 3: return OrenNayarBRDF::sample(i, OrenNayarMaterial(material, uv));
    case 4: return MixtureBRDF::sample(i, MixtureMaterial(material, uv));
    case 5: return RefractBRDF::sample(i, RefractMaterial(material, uv));
    case 10: {
        RGLMaterial material = RGLMaterial(0);
        ibsdf::sample_out sample_o = RGLBRDF::sample_with_perchannel_cv(i, material, cv);
        cv = discard_nan_inf(cv / sample_o.pdf);
        return sample_o;
    }
    }
    return o;
}

/**
 * Sample the primal BSDF
 * @param i: input sample information
 * @param material: material data
 * @param uv: texture coordinate
 */
ibsdf::sample_out bsdf_sample_safe(ibsdf::sample_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::sample(i, ConductorMaterial(material, uv));
    case 2: return PlasticBRDF::sample(i, PlasticMaterial(material, uv));
    case 3: return OrenNayarBRDF::sample(i, OrenNayarMaterial(material, uv));
    case 4: return MixtureBRDF::sample_safe(i, MixtureMaterial(material, uv));
    case 5: return RefractBRDF::sample(i, RefractMaterial(material, uv));
    }
    return o;
}

/**
 * Evaluate the PDF of the primal BSDF sampling
 * @param i: input sample information
 * @param material: material data
 * @param uv: texture coordinate
 */
float bsdf_sample_pdf(ibsdf::pdf_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::pdf(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::pdf(i, ConductorMaterial(material, uv));
    case 2: return PlasticBRDF::pdf(i, PlasticMaterial(material, uv));
    case 3: return OrenNayarBRDF::pdf(i, OrenNayarMaterial(material, uv));
    case 4: return MixtureBRDF::pdf(i, MixtureMaterial(material, uv));
    case 5: return RefractBRDF::pdf(i, RefractMaterial(material, uv));
    }
    return 0.f;
}

/**
 * Evaluate the backward BSDF
 * @param i: input evaluate information
 * @param dL: adjoint radiance contribution
 * @param material: material data
 * @param uv: texture coordinate
 */
void bsdf_backward_grad(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: LambertianBRDF::backward_grad(i, dL, material, uv); break;
    case 1: ConductorBRDF::backward_grad(i, dL, material, uv); break;
    case 3: OrenNayarBRDF::backward_grad(i, dL, material, uv); break;
    case 4: MixtureBRDF::backward_grad(i, dL, material, uv); break;
    }
}

/**
 * Evaluate the backward BSDF
 * @param i: input evaluate information
 * @param dL: adjoint radiance contribution
 * @param material: material data
 * @param uv: texture coordinate
 */
void bsdf_backward_grad_ratio(
    ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    // case 0: LambertianBRDF::backward_grad(i, dL, material, uv); break;
    // case 1: ConductorBRDF::backward_grad(i, dL, material, uv); break;
    case 3: OrenNayarBRDF::backward_grad_ratio(i, dL, material, uv); break;
    case 4: MixtureBRDF::backward_grad_ratio(i, dL, material, uv); break;
    }
}

/**
 * Evaluate the backward BSDF, with specialized techniques
 * @param i: input evaluate information
 * @param dL: adjoint radiance contribution
 * @param material: material data
 * @param uv: texture coordinate
 * @param index: specialized technique index, should start from 1
 */
void bsdf_backward_grad_nth(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv, int index) {
    switch (index) {
    case 1: brdfd_techniques::bsdf_backward_grad_1st(i, dL, material, uv); break;
    case 2: brdfd_techniques::bsdf_backward_grad_2nd(i, dL, material, uv); break;
    case 3: brdfd_techniques::bsdf_backward_grad_3rd(i, dL, material, uv); break;
    case 4: brdfd_techniques::bsdf_backward_grad_4th(i, dL, material, uv); break;
    case 5: brdfd_techniques::bsdf_backward_grad_5th(i, dL, material, uv); break;
    case 6: brdfd_techniques::bsdf_backward_grad_6th(i, dL, material, uv); break;
    }
}

/**
 * Sample the specialized BSDF derivative sampling technique
 * @param i: input sample information
 * @param dL: adjoint radiance contribution
 * @param material: material data
 * @param uv: texture coordinate
 * @param index: specialized technique index, should start from 1
 */
ibsdf::sample_out bsdf_sample_d_nth(
    ibsdf::sample_in i, MaterialData material, float2 uv, int index) {
    switch (index) {
    case 1: return brdfd_techniques::bsdf_sample_d_1st(i, material, uv);
    case 2: return brdfd_techniques::bsdf_sample_d_2nd(i, material, uv);
    case 3: return brdfd_techniques::bsdf_sample_d_3rd(i, material, uv);
    case 4: return brdfd_techniques::bsdf_sample_d_4th(i, material, uv);
    case 5: return brdfd_techniques::bsdf_sample_d_5th(i, material, uv);
    case 6: return brdfd_techniques::bsdf_sample_d_6th(i, material, uv);
    }
    return {};
}

/**
* Evaluate the PDF of the specialized BSDF derivative sampling lobe
* @param i: input evaluate information
* @param dL: adjoint radiance contribution
* @param material: material data
* @param uv: texture coordinate
* @param index: specialized technique index, should start from 1
*/
float bsdf_pdf_d_nth(
    ibsdf::pdf_in i, MaterialData material, float2 uv, int index) {
    switch (index) {
    case 1: return brdfd_techniques::bsdf_pdf_d_1st(i, material, uv);
    case 2: return brdfd_techniques::bsdf_pdf_d_2nd(i, material, uv);
    }
    return {};
}

namespace brdfd_techniques {
    /**
    * Evaluate the BSDF derivative, specialized for 1st derivative sampling lobe
    * @param i: input evaluate information
    * @param dL: adjoint radiance contribution
    * @param material: material data
    * @param uv: texture coordinate
    */
    void bsdf_backward_grad_1st(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 1: ConductorBRDF::backward_alpha_derivative_pos(i, dL, material, uv); break;
        case 3: OrenNayarBRDF::backward_sigma_derivative_diffuse(i, dL, material, uv); break;
        case 4: MixtureBRDF::backward_weight_derivative_diffuse(i, dL, material, uv); break;
        }
    }

    /**
     * Evaluate the BSDF derivative, specialized for 1st derivative sampling lobe
     * @param i: input evaluate information
     * @param dL: adjoint radiance contribution
     * @param material: material data
     * @param uv: texture coordinate
     */
    void bsdf_backward_grad_2nd(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 1: ConductorBRDF::backward_alpha_derivative_neg(i, dL, material, uv); break;
        case 3: OrenNayarBRDF::backward_sigma_derivative_spec(i, dL, material, uv); break;
        case 4: MixtureBRDF::backward_weight_derivative_spec(i, dL, material, uv); break;
        }
    }

    /**
     * Evaluate the BSDF derivative, specialized for 1st derivative sampling lobe
     * @param i: input evaluate information
     * @param dL: adjoint radiance contribution
     * @param material: material data
     * @param uv: texture coordinate
     */
    void bsdf_backward_grad_3rd(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: MixtureBRDF::backward_sigma_derivative_lambert(i, dL, material, uv); break;
        }
    }

    void bsdf_backward_grad_4th(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: MixtureBRDF::backward_sigma_derivative_nonlambert(i, dL, material, uv); break;
        }
    }

    void bsdf_backward_grad_5th(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: MixtureBRDF::backward_ggx_alpha_derivative_pos(i, dL, material, uv); break;
        }
    }
    
    void bsdf_backward_grad_6th(ibsdf::bwd_in i, float3 dL, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: MixtureBRDF::backward_ggx_alpha_derivative_neg(i, dL, material, uv); break;
        }
    }

    /**
    * Sample the 1st BSDF derivative sampling lobe
    * @param i: input sample information
    * @param dL: adjoint radiance contribution
    * @param material: material data
    * @param uv: texture coordinate
    */
    ibsdf::sample_out bsdf_sample_d_1st(
        ibsdf::sample_in i, MaterialData material, float2 uv) {
        switch (material.bxdf_type) {
        case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
        case 1: return ConductorBRDF::sample_alpha_derivative_pos(i, ConductorMaterial(material, uv));
        case 3: return OrenNayarBRDF::sample_sigma_derivative_diffuse(i, OrenNayarMaterial(material, uv));
        case 4: return MixtureBRDF::sample_alpha_derivative_diffuse(i, MixtureMaterial(material, uv));
        }
        return {};
    }

    /**
     * Sample the 2md BSDF derivative sampling lobe
     * @param i: input sample information
     * @param material: material data
     * @param uv: texture coordinate
     */
    ibsdf::sample_out bsdf_sample_d_2nd(ibsdf::sample_in i, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
        case 1: return ConductorBRDF::sample_alpha_derivative_neg(i, ConductorMaterial(material, uv));
        case 3: return OrenNayarBRDF::sample_sigma_derivative_spec(i, OrenNayarMaterial(material, uv));
        case 4: return MixtureBRDF::sample_alpha_derivative_spec(i, MixtureMaterial(material, uv));
        }
        return o;
    }

    /**
     * Sample the 3rd BSDF derivative sampling lobe
     * @param i: input sample information
     * @param material: material data
     * @param uv: texture coordinate
     */
    ibsdf::sample_out bsdf_sample_d_3rd(ibsdf::sample_in i, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: return MixtureBRDF::sample_sigma_derivative_lambert(i, MixtureMaterial(material, uv));
        }
        return o;
    }

    ibsdf::sample_out bsdf_sample_d_4th(ibsdf::sample_in i, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: return MixtureBRDF::sample_sigma_derivative_nonlambert(i, MixtureMaterial(material, uv));
        }
        return o;
    }

    ibsdf::sample_out bsdf_sample_d_5th(ibsdf::sample_in i, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: return MixtureBRDF::sample_ggx_alpha_derivative_pos(i, MixtureMaterial(material, uv));
        }
        return o;
    }

    ibsdf::sample_out bsdf_sample_d_6th(ibsdf::sample_in i, MaterialData material, float2 uv) {
        ibsdf::sample_out o;
        switch (material.bxdf_type) {
        case 4: return MixtureBRDF::sample_ggx_alpha_derivative_neg(i, MixtureMaterial(material, uv));
        }
        return o;
    }

    /**
    * Evaluate the PDF of the 1st BSDF derivative sampling lobe
    * @param i: input evaluate information
    * @param dL: adjoint radiance contribution
    * @param material: material data
    * @param uv: texture coordinate
    */
    float bsdf_pdf_d_1st(
        ibsdf::pdf_in i, MaterialData material, float2 uv) {
        switch (material.bxdf_type) {
        case 0: return LambertianBRDF::pdf(i, LambertMaterial(material, uv));
        case 1: return ConductorBRDF::pdf_alpha_derivative_pos(i, ConductorMaterial(material, uv));
        }
        return {};
    }

    /**
    * Evaluate the PDF of the 1st BSDF derivative sampling lobe
    * @param i: input evaluate information
    * @param dL: adjoint radiance contribution
    * @param material: material data
    * @param uv: texture coordinate
    */
    float bsdf_pdf_d_2nd(
        ibsdf::pdf_in i, MaterialData material, float2 uv) {
        switch (material.bxdf_type) {
        case 0: return LambertianBRDF::pdf(i, LambertMaterial(material, uv));
        case 1: return ConductorBRDF::pdf_alpha_derivative_neg(i, ConductorMaterial(material, uv));
        }
        return {};
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