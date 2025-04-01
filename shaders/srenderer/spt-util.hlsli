#ifndef _SRENDERER_SPT_UTIL_HLSLI_
#define _SRENDERER_SPT_UTIL_HLSLI_

#include "srenderer/lights.hlsli"
#include "srenderer/materials.hlsli"

ibsdf::sample_out brdf_sample(
    const Ray ray,
    const MaterialData material,
    const PrimaryPayload payload,
    inout RandomSamplerState rng,
) {
    ibsdf::sample_in bs_i;
    bs_i.u = GetNextRandomFloat3(rng);
    bs_i.wi = -ray.direction;
    bs_i.geometric_normal = payload.hit.geometryNormal;
    float faceforward = IsFaceForward(payload.hit) ? 1.f : -1.f;
    bs_i.shading_frame = Frame(createFrame(payload.hit.shadingNormal * faceforward));
    return materials::bsdf_sample(bs_i, material, payload.hit.texcoord);
}

ibsdf::sample_out brdf_sample_rcv(
    const Ray ray,
    const MaterialData material,
    const PrimaryPayload payload,
    inout RandomSamplerState rng,
    out float3 channel_cv,
) {
    ibsdf::sample_in bs_i;
    bs_i.u = GetNextRandomFloat3(rng);
    bs_i.wi = -ray.direction;
    bs_i.geometric_normal = payload.hit.geometryNormal;
    float faceforward = IsFaceForward(payload.hit) ? 1.f : -1.f;
    bs_i.shading_frame = Frame(createFrame(payload.hit.shadingNormal * faceforward));
    return materials::bsdf_sample_with_perchannel_cv(bs_i, material, 
        payload.hit.texcoord, channel_cv);
}

ilight::sample_li_out light_sample(
    const Ray ray,
    const PrimaryPayload payload,
    inout RandomSamplerState rng,
) {
    ilight::sample_li_in nee_i;
    nee_i.p = payload.hit.position;
    nee_i.ns = payload.hit.shadingNormal;
    nee_i.uv = GetNextRandomFloat2(rng);
    return lights::nee_lbvh(
        nee_i, GetNextRandom(rng),
        (uint)ImportanceFacotr::Use_Power
        | (uint)ImportanceFacotr::Use_Distance
        | (uint)ImportanceFacotr::Use_Cone);
}

ilight::sample_li_out light_sample_rcv(
    const Ray ray,
    const PrimaryPayload payload,
    inout RandomSamplerState rng,
    out float3 channel_cv,
) {
    ilight::sample_li_in nee_i;
    nee_i.p = payload.hit.position;
    nee_i.ns = payload.hit.shadingNormal;
    nee_i.uv = GetNextRandomFloat2(rng);
    ilight::sample_li_out o = lights::nee_lbvh_with_aux(
        nee_i, GetNextRandom(rng),
        (uint)ImportanceFacotr::Use_Power
        | (uint)ImportanceFacotr::Use_Distance
        | (uint)ImportanceFacotr::Use_Cone, channel_cv);
    channel_cv /= o.pdf;
    if (isinf(o.pdf)) channel_cv = float3(1, 1, 1);
    return o;
}

ilight::sample_li_out light_sample_rcv(
    const Ray ray,
    const float3 position,
    inout RandomSamplerState rng,
    out float3 channel_cv,
) {
    ilight::sample_li_in nee_i;
    nee_i.p = position;
    nee_i.ns = float3(0, 0, 0);
    nee_i.uv = GetNextRandomFloat2(rng);
    ilight::sample_li_out o = lights::nee_lbvh_with_aux(
        nee_i, GetNextRandom(rng),
        (uint)ImportanceFacotr::Use_Power
        | (uint)ImportanceFacotr::Use_Distance
        | (uint)ImportanceFacotr::Use_Cone, channel_cv);
    channel_cv /= o.pdf;
    if (isinf(o.pdf)) channel_cv = float3(1, 1, 1);
    return o;
}

float3 eval_bsdf(
    const Ray ray,
    const float3 direction,
    const MaterialData material,
    const PrimaryPayload payload,
) {
    ibsdf::eval_in ev_i;
    ev_i.wi = -ray.direction;
    ev_i.wo = direction;
    ev_i.wh = normalize(ev_i.wi + ev_i.wo);
    ev_i.geometric_normal = payload.hit.geometryNormal;
    float faceforward = IsFaceForward(payload.hit) ? 1.f : -1.f;
    ev_i.shading_frame = Frame(createFrame(payload.hit.shadingNormal * faceforward));
    return materials::bsdf_eval(ev_i, material, payload.hit.texcoord);
}

float pdf_bsdf(
    const Ray ray,
    const float3 direction,
    const MaterialData material,
    const PrimaryPayload payload,
) {
    ibsdf::pdf_in pdf_in;
    pdf_in.wi = -ray.direction;
    pdf_in.wo = direction;
    pdf_in.wh = normalize(pdf_in.wi + pdf_in.wo);
    pdf_in.geometric_normal = payload.hit.geometryNormal;
    float faceforward = IsFaceForward(payload.hit) ? 1.f : -1.f;
    pdf_in.shading_frame = Frame(createFrame(payload.hit.shadingNormal * faceforward));
    float2 texcoord = payload.hit.texcoord;
    return materials::bsdf_sample_pdf(pdf_in, material, texcoord);
}

struct light_sample_ctx {
    float3 position;
    float3 geometry_normal;
    float4 aux_pdf;

    static light_sample_ctx from_payload(PrimaryPayload payload, float4 _aux_pdf) {
        light_sample_ctx ctx;
        ctx.position = payload.hit.position;
        ctx.geometry_normal = payload.hit.geometryNormal;
        ctx.aux_pdf = _aux_pdf;
        return ctx;
    }

    static light_sample_ctx from_poistion(float3 position) {
        light_sample_ctx ctx;
        ctx.position = position;
        ctx.geometry_normal = float3(0, 0, 0);
        ctx.aux_pdf = float4(1, 1, 1, 1);
        return ctx;
    }
};

float pdf_nee(
    const light_sample_ctx sample_ctx,
    const PrimaryPayload light_payload,
) {
    float nee_pdf = 0.f;
    int lightID = GPUScene_geometry[light_payload.hit.geometryID].lightID;
    if (lightID >= 0) {
        lightID += light_payload.hit.primitiveID;
        ilight::sample_li_pdf_in nee_pdf_i;
        nee_pdf_i.lightID = lightID;
        nee_pdf_i.ref_point = sample_ctx.position;
        nee_pdf_i.ref_normal = sample_ctx.geometry_normal;
        nee_pdf_i.light_point = light_payload.hit.position;
        nee_pdf_i.light_normal = light_payload.hit.geometryNormal;
        nee_pdf = lights::nee_lbvh_pdf(nee_pdf_i,
            (uint)ImportanceFacotr::Use_Power
            | (uint)ImportanceFacotr::Use_Distance
            | (uint)ImportanceFacotr::Use_Cone);
        nee_pdf = discard_nan_inf(nee_pdf);
    }
    return nee_pdf;
}

float4 pdf_aux_nee(
    const light_sample_ctx sample_ctx,
    const PrimaryPayload light_payload,
) {
    float4 nee_pdf = 0.f;
    int lightID = GPUScene_geometry[light_payload.hit.geometryID].lightID;
    if (lightID >= 0) {
        lightID += light_payload.hit.primitiveID;
        ilight::sample_li_pdf_in nee_pdf_i;
        nee_pdf_i.lightID = lightID;
        nee_pdf_i.ref_point = sample_ctx.position;
        nee_pdf_i.ref_normal = sample_ctx.geometry_normal;
        nee_pdf_i.light_point = light_payload.hit.position;
        nee_pdf_i.light_normal = light_payload.hit.geometryNormal;
        nee_pdf = lights::nee_lbvh_pdf_with_aux(nee_pdf_i,
            (uint)ImportanceFacotr::Use_Power
            | (uint)ImportanceFacotr::Use_Distance
            | (uint)ImportanceFacotr::Use_Cone);
        nee_pdf = discard_nan_inf(nee_pdf);
    }
    return nee_pdf;
}

float3 surface_emission(Optional<MaterialData> material, const float3 wo) {
    if (material.hasValue) {
        return materials::emission(material.value);
    }
    return float3(0, 0, 0);
}

#endif // _SRENDERER_SPT_UTIL_HLSLI_