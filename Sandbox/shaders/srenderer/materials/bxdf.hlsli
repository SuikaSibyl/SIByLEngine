#ifndef _SRENDERER_BSDF_HEADER_
#define _SRENDERER_BSDF_HEADER_

#include "common/geometry.hlsli"
#include "common/math.hlsli"
#include "common/microfacet.hlsli"

interface IBxDFParameter : IDifferentiable {};

namespace ibsdf {
struct eval_in {
    float3 wi;
    float3 wo;
    float3 geometric_normal;
    Frame shading_frame;
};

struct sample_in {
    float3 u;
    float3 wi;
    float3 geometric_normal;
    Frame shading_frame;
};

struct sample_out {
    float3 bsdf;
    float3 wo;
    float pdf;
    float3 wh; // microfacet orientation
    float pdf_wh; // microfacet pdf
};

struct pdf_in {
    float3 wi;
    float3 wo;
    float3 wh;
    float3 geometric_normal;
    Frame shading_frame;
};

struct bwd_in {
    eval_in eval;
    int hash_value;
};

struct dsample_out<TBxDFParameter : IBxDFParameter> {
    float3 wo;
    float pdf;
    TBxDFParameter.Differential dparam;
};
struct dsample_noeval_out {
    float3 wo;
    float pdf;
    float3 wh; // microfacet orientation
    float pdf_wh; // microfacet pdf
};
}

interface IBxDF {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : IBxDFParameter;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, TParam param);
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, TParam param);
    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, TParam param);
};

interface IBxDFDerivative {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : IBxDFParameter;

    /** Backward derivative of bxdf evaluation */
    static void bwd_eval(const ibsdf::eval_in i, DifferentialPair<TParam> param);
    
    /** The pdf of postivized derivative sampling, positive part */
    static ibsdf::dsample_out<TParam> sample_primal(const ibsdf::sample_in i, TParam param);
    /** The pdf of postivized derivative sampling, positive part */
    static ibsdf::dsample_out<TParam> sample_pos_derivative(const ibsdf::sample_in i, TParam param);
    /** The pdf of postivized derivative sampling, negative part */
    static ibsdf::dsample_out<TParam> sample_neg_derivative(const ibsdf::sample_in i, TParam param);

    /** The pdf of postivized derivative sampling, positive part; no derivative evauation */
    static ibsdf::dsample_noeval_out sample_noeval_primal(const ibsdf::sample_in i, TParam param);
    /** The pdf of postivized derivative sampling, positive part; no derivative evauation */
    static ibsdf::dsample_noeval_out sample_noeval_pos_derivative(const ibsdf::sample_in i, TParam param);
    /** The pdf of postivized derivative sampling, negative part; no derivative evauation */
    static ibsdf::dsample_noeval_out sample_noeval_neg_derivative(const ibsdf::sample_in i, TParam param);

    /** The pdf of postivized derivative sampling, positive part */
    static float pdf_pos_derivative(const ibsdf::pdf_in i, TParam param);
    /** The pdf of postivized derivative sampling, negative part */
    static float pdf_neg_derivative(const ibsdf::pdf_in i, TParam param);
};

namespace ibsdf {
float u2theta(float u) { return 
    sqr(u) * (k_pi / 2.f); }
float2 u2theta(float2 u) { return 
    float2(u2theta(u.x), u2theta(u.y)); }
float3 u2theta(float3 u) { return 
    float3(u2theta(u.xy), u2theta(u.z)); }
float4 u2theta(float4 u) { return 
    float4(u2theta(u.xy), u2theta(u.zw)); }

float u2phi(float u) { return 
    (2.f * u - 1.f) * k_pi; }
float2 u2phi(float2 u) { return 
    float2(u2phi(u.x), u2phi(u.y)); }
float3 u2phi(float3 u) { return 
    float3(u2phi(u.xy), u2phi(u.z)); }
float4 u2phi(float4 u) { return 
    float4(u2phi(u.xy), u2phi(u.zw)); }

float theta2u(float theta) { return 
    sqrt(theta * (2.f / k_pi)); }
float2 theta2u(float2 theta) { return 
    float2(theta2u(theta.x), theta2u(theta.y)); }
float3 theta2u(float3 theta) { return 
    float3(theta2u(theta.xy), theta2u(theta.z)); }
float4 theta2u(float4 theta) { return 
    float4(theta2u(theta.xy), theta2u(theta.zw)); }

float phi2u(float phi) { return 
    (phi + k_pi) / (2.f * k_pi); }
float2 phi2u(float2 phi) { return 
    float2(phi2u(phi.x), phi2u(phi.y)); }
float3 phi2u(float3 phi) { return 
    float3(phi2u(phi.xy), phi2u(phi.z)); }
float4 phi2u(float4 phi) { return 
    float4(phi2u(phi.xy), phi2u(phi.zw)); }
}

///////////////////////////////////////////////////////////////////////////////////////////
// General Microfacet BSDFs for further instantiation
// ----------------------------------------------------------------------------------------
// Basic and general microfacet BRDFs
///////////////////////////////////////////////////////////////////////////////////////////

namespace microfacet_reflection {
// Evaluate the BSDF
[Differentiable]
float3 eval<
    TMicrofacetDistribution : IMicrofacetDistribution,
    TFresnel : IFresnel
>(  no_diff ibsdf::eval_in i,
    no_diff TFresnel fresnel,
    TMicrofacetDistribution.TParam parameter,
    no_diff float3 R) 
{
    const float3 wi = no_diff i.shading_frame.to_local(i.wi);
    const float3 wo = no_diff i.shading_frame.to_local(i.wo);
    if (dot(i.geometric_normal, i.wi) < 0 ||
        dot(i.geometric_normal, i.wo) < 0) {
        // No light below the surface
        return float3(0);
    }

    const float cosThetaO = theta_phi_coord::AbsCosTheta(wo);
    const float cosThetaI = theta_phi_coord::AbsCosTheta(wi);
    float3 wh = normalize(wi + wo);
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return float3(0.);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return float3(0.);
    
    float3 F = fresnel.eval(dot(wi, wh));
    // float3 diffuse = R * k_inv_pi * (1 - F);
    // return specular + diffuse;
    // float3 specular = TMicrofacetDistribution::D(wh, parameter)
    //                 * TMicrofacetDistribution::G(wo, wi, parameter)
    //                 * F / (4 * cosThetaI);
    float3 specular = TMicrofacetDistribution::D(wh, parameter) / (4 * cosThetaI);
    return specular;
}

TMicrofacetDistribution.TParam.Differential bwd_eval<
    TMicrofacetDistribution : IMicrofacetDistribution,
    TFresnel : IFresnel
>(  ibsdf::eval_in i,
    TFresnel fresnel,
    TMicrofacetDistribution.TParam parameter,
    float3 R
) {
    var param_pair = diffPair(parameter);
    float3 d_out = float3(1);
    bwd_diff(eval<TMicrofacetDistribution, TFresnel>)(i, fresnel, param_pair, R, d_out);
    return param_pair.d;
    // TMicrofacetDistribution.TParam.Differential parameter_diff = 
    //     TMicrofacetDistribution.TParam::dzero();

    // const float3 wi = i.shading_frame.to_local(i.wi);
    // const float3 wo = i.shading_frame.to_local(i.wo);
    // if (dot(i.geometric_normal, i.wi) < 0 ||
    //     dot(i.geometric_normal, i.wo) < 0) {
    //     // No light below the surface
    //     return parameter_diff;
    // }

    // const float cosThetaO = theta_phi_coord::AbsCosTheta(wo);
    // const float cosThetaI = theta_phi_coord::AbsCosTheta(wi);
    // float3 wh = normalize(wi + wo);
    // // Handle degenerate cases for microfacet reflection
    // if (cosThetaI == 0 || cosThetaO == 0)
    //     return parameter_diff;
    // if (wh.x == 0 && wh.y == 0 && wh.z == 0)
    //     return parameter_diff;

    // float3 F = fresnel.eval(dot(wi, wh));
    // const float c = F.x / (4 * cosThetaI * cosThetaO) * max(wo.z, 0.f);

    // float D = TMicrofacetDistribution::D(wh, parameter);
    // float G = TMicrofacetDistribution::G(wo, wi, parameter);
    // var d_param_pair = diffPair(parameter);
    // var g_param_pair = diffPair(parameter);
    // bwd_diff(TMicrofacetDistribution::D)(wh, d_param_pair, c * G);
    // bwd_diff(TMicrofacetDistribution::G)(wo, wi, g_param_pair, c * D);
    // TMicrofacetDistribution.TParam.Differential dParam = 
    //     TMicrofacetDistribution.TParam.Differential::dadd(d_param_pair.d, g_param_pair.d);
    
    // float tan2Theta = theta_phi_coord::Tan2Theta(wh);
    // if (isinf(tan2Theta) || isnan(tan2Theta) || tan2Theta == 0) return parameter_diff;
    
    // return dParam;
}
// importance sample the BSDF
ibsdf::sample_out sample_vnormal<TMicrofacetDistribution : IMicrofacetDistribution>(
    ibsdf::sample_in i,
    TMicrofacetDistribution.TParam parameter) {
    ibsdf::sample_out o;
    // Sample microfacet orientation wh and reflected direction wi
    const float3 wi = i.shading_frame.to_local(i.wi);
    const float3 wh = TMicrofacetDistribution::sample_wh_vnormal(wi, i.u.xy, parameter);
    // const float3 wh = sample_cos_hemisphere(i.u);
    const float3 wo = reflect(-wi, wh);
    o.wh = i.shading_frame.to_world(wh);
    o.wo = i.shading_frame.to_world(wo);
    // Compute PDF of wi for microfacet reflection
    float VdotH = dot(wi, wh);
    const float pdf = TMicrofacetDistribution::pdf_vnormal(wi, wh, parameter);
    // const float pdf = pdf_cos_hemisphere(wh);
    o.pdf = pdf / (4 * abs(VdotH));
    o.pdf_wh = pdf;
    return o;
}

// importance sample the BSDF
ibsdf::sample_out sample_normal<TMicrofacetDistribution : IMicrofacetDistribution>(
    ibsdf::sample_in i,
    TMicrofacetDistribution.TParam parameter) {
    ibsdf::sample_out o;
    // Sample microfacet orientation wh and reflected direction wi
    float3 wi = i.shading_frame.to_local(i.wi);
    if (wi.z < 0) {
        wi.z = -wi.z;
        i.wi = i.shading_frame.to_world(wi);
    }
    const float3 wh = TMicrofacetDistribution::sample_wh_normal(wi, i.u.xy, parameter);
    // const float3 wh = sample_cos_hemisphere(i.u);
    const float3 wo = reflect(-wi, wh);
    o.wh = i.shading_frame.to_world(wh);
    o.wo = i.shading_frame.to_world(wo);
    // Compute PDF of wi for microfacet reflection
    float VdotH = dot(wi, wh);
    const float pdf = TMicrofacetDistribution::pdf_normal(wi, wh, parameter);
    // const float pdf = pdf_cos_hemisphere(wh);
    o.pdf = pdf / (4 * abs(VdotH));
    o.pdf_wh = pdf;
    return o;
}
// Evaluate the PDF of the BSDF sampling
float pdf<TMicrofacetDistribution : IMicrofacetDistribution>
    (ibsdf::pdf_in i,
    TMicrofacetDistribution distribution,
    TMicrofacetDistribution.TParam parameter) {
    return 1.f;
    // if (dot(i.geometric_normal, i.wi) < 0 ||
    //     dot(i.geometric_normal, i.wo) < 0) {
    //     // No light below the surface
    //     return float(0);
    // }
    // // Flip the shading frame if it is
    // // inconsistent with the geometry normal.
    // Frame frame = i.shading_frame;
    // float3 wi = frame.to_local(i.wi);
    // float3 wo = frame.to_local(i.wo);
    // float3 wh = normalize(wi + wo);

    // float VdotH = dot(wi, wh);
    // const float pdf = distribution.pdf(wi, wh, parameter);
    // return pdf / (4 * abs(VdotH));
}

// importance sample the BSDF Derivative, positive part
float4 sample_pos<TMicrofacetDerivative : IMicrofacetDerivative>(
    ibsdf::sample_in i,
    TMicrofacetDerivative.TParam parameter) {
    float4 o;
    // For Lambertian, we importance sample the cosine hemisphere domain.
    if (dot(i.geometric_normal, i.wi) < 0) {
        // Incoming direction is below the surface.
        o.rgb = float3(0);
        o.a = 0;
        return o;
    }

    // Sample microfacet orientation wh and reflected direction wi
    const float3 wi = i.shading_frame.to_local(i.wi);
    const float3 wh = TMicrofacetDerivative::sample_pos_wh(wi, i.u.xy, parameter);
    const float3 wo = reflect(-wi, wh);
    o.rgb = i.shading_frame.to_world(wo);
    // Compute PDF of wi for microfacet reflection
    float VdotH = dot(wi, wh);
    const float pdf = TMicrofacetDerivative::pdf_pos(wi, wh, parameter);
    o.a = pdf / (4 * abs(VdotH));

    // reject samples below the surface
    if (wo.z < 0.f) {
        o.rgb = normalize(float3(0.f));
        // o.a = 0.f;
    }
    return o;
}
// importance sample the BSDF Derivative, negative part
float4 sample_neg<TMicrofacetDerivative : IMicrofacetDerivative>(
    ibsdf::sample_in i,
    TMicrofacetDerivative.TParam parameter) {
    float4 o;
    // For Lambertian, we importance sample the cosine hemisphere domain.
    if (dot(i.geometric_normal, i.wi) < 0) {
        // Incoming direction is below the surface.
        o.rgb = float3(0);
        o.a = 0;
        return o;
    }

    // Sample microfacet orientation wh and reflected direction wi
    const float3 wi = i.shading_frame.to_local(i.wi);
    const float3 wh = TMicrofacetDerivative::sample_neg_wh(wi, i.u.xy, parameter);
    const float3 wo = reflect(-wi, wh);
    o.rgb = i.shading_frame.to_world(wo);
    // Compute PDF of wi for microfacet reflection
    float VdotH = dot(wi, wh);
    const float pdf = TMicrofacetDerivative::pdf_neg(wi, wh, parameter);
    o.a = pdf / (4 * abs(VdotH));

    // reject samples below the surface
    if (wo.z < 0.f) {
        o.rgb = normalize(float3(0.f));
        // o.a = 0.f;
    }

    return o;
}
float testtest(float theta, float alpha) {
    float x = acos(theta);
    float ct = cos(x);
    float ct_2 = ct * ct;
    float ct_3 = ct * ct_2;
    float tt_2 = tan(x) * tan(x);
    float alpha_2 = alpha * alpha;

    float numerator = 4 * alpha_2 * (tt_2 - alpha_2);
    float temp = alpha_2 + tt_2;
    float denominator = k_pi * ct_3 * temp * temp * temp;
    float result = numerator / denominator;
    return result;
}
// importance sample the BSDF Derivative, positive part
float pdf_pos<TMicrofacetDerivative : IMicrofacetDerivative>(
    ibsdf::pdf_in i,
    TMicrofacetDerivative.TParam parameter) {
    
    const float3 wi = i.shading_frame.to_local(i.wi);
    const float3 wh = i.shading_frame.to_local(i.wh);
    // Compute PDF of wi for microfacet reflection
    return TMicrofacetDerivative::pdf_pos(wi, wh, parameter);
}
// importance sample the BSDF Derivative, negative part
float pdf_neg<TMicrofacetDerivative : IMicrofacetDerivative>(
    ibsdf::pdf_in i,
    TMicrofacetDerivative.TParam parameter) {
    
    const float3 wi = i.shading_frame.to_local(i.wi);
    const float3 wh = i.shading_frame.to_local(i.wh);
    // Compute PDF of wi for microfacet reflection
    return TMicrofacetDerivative::pdf_neg(wi, wh, parameter);
}
};

#endif // _SRENDERER_BSDF_HEADER_