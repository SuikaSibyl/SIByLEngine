#ifndef _SRENDERER_SPT_UTILS_SLANG_
#define _SRENDERER_SPT_UTILS_SLANG_

#include "srenderer/spt-bindings.slang"
#include "srenderer/spt-definition.slang"
#include "srenderer/materials.slang"
#include "srenderer/lights.slang"

extension IntersectionPayload {
    GeometryData geometry() { return scene_read_geometry(hit.geometryID); }
    Optional<MaterialData> material() {
        GeometryData geometry = scene_read_geometry(hit.geometryID);
        return scene_read_material(geometry);
    }

    /**
     * Sample the BSDF at the intersection point.
     * @param ri: incoming ray
     * @param u: random sample in [0, 1]^2
     * @return Optional containing the sampled BSDF information, or none if sampling failed.
     */
    Optional<ibsdf::sample_out> bsdf_sample(Ray ri, float3 u) {
        Frame shading_frame = hit.get_shading_frame();
        ibsdf::sample_in bs_i;
        bs_i.u = u;
        bs_i.wi = -ri.direction;
        bs_i.shading_frame = shading_frame;
        bs_i.geometric_normal = hit.geometryNormal;
        GeometryData geometry = scene_read_geometry(hit.geometryID);
        Optional<MaterialData> material = scene_read_material(geometry.materialID);
        if (!material.hasValue) return none;
        return materials::bsdf_sample(bs_i, material.value, hit.texcoord);
    }

    /**
     * Evaluate the BSDF at the intersection point.
     * @param wi: incoming direction
     * @param wo: outgoing direction
     * @return Evaluated BSDF value.
     */
    float3 bsdf_eval(float3 wi, float3 wo) {
        GeometryData geometry = scene_read_geometry(hit.geometryID);
        Optional<MaterialData> material = scene_read_material(geometry.materialID);
        ibsdf::eval_in ev_i;
        ev_i.wi = wi;
        ev_i.wo = wo;
        ev_i.wh = none;
        ev_i.shading_frame = hit.get_shading_frame();
        ev_i.geometric_normal = hit.geometryNormal;
        return materials::bsdf_eval(ev_i, material.value, hit.texcoord);
    }

    /**
     * Evaluate the PDF of BSDF sampling at the intersection point.
     * @param wi: incoming direction
     * @param wo: outgoing direction
     * @return PDF of the BSDF for the given directions.
     */
    float bsdf_pdf(float3 wi, float3 wo) {
        GeometryData geometry = scene_read_geometry(hit.geometryID);
        Optional<MaterialData> material = scene_read_material(geometry.materialID);
        ibsdf::pdf_in pdf_in;
        pdf_in.wi = wi;
        pdf_in.wo = wo;
        pdf_in.wh = normalize(pdf_in.wi + pdf_in.wo);
        pdf_in.shading_frame = hit.get_shading_frame();
        pdf_in.geometric_normal = hit.geometryNormal;
        return materials::bsdf_pdf(pdf_in, material.value, hit.texcoord);
    }

    /**
     * Sample the light at the intersection point.
     * @param ri: incoming ray
     * @param u: random sample in [0, 1]^2
     * @return Optional containing the sampled light information, or none if sampling failed.
     */
    Optional<ilight::sample_li_out> light_sample(Ray ri, float3 u, uint sceneID = 0) {
        Frame shading_frame = hit.get_shading_frame();
        ilight::sample_li_in nee_i;
        nee_i.p = hit.position;
        nee_i.ns = hit.shadingNormal;
        nee_i.uv = u.xy;
        return lights::nee_lbvh(nee_i, u.z,
            int(ImportanceFacotr::Use_Power 
              | ImportanceFacotr::Use_Distance 
              | ImportanceFacotr::Use_Cone), sceneID);
    }

    float light_pdf(IntersectionPayload next_payload, uint sceneID = 0) {
        GeometryData next_geometry = scene_read_geometry(next_payload.hit.geometryID);        
        float nee_pdf = 0.f;
        int lightID = next_geometry.lightID;
        if (lightID >= 0) {
            lightID += next_payload.hit.primitiveID;
            ilight::sample_li_pdf_in nee_pdf_i;
            nee_pdf_i.lightID = lightID;
            nee_pdf_i.ref_point = hit.position;
            nee_pdf_i.ref_normal = hit.geometryNormal;
            nee_pdf_i.light_point = next_payload.hit.position;
            nee_pdf_i.light_normal = next_payload.hit.geometryNormal;
            nee_pdf = lights::nee_lbvh_pdf(nee_pdf_i,
                  (uint)ImportanceFacotr::Use_Power
                | (uint)ImportanceFacotr::Use_Distance
                | (uint)ImportanceFacotr::Use_Cone, sceneID);
        }
        return nee_pdf;
    }
    
    float4 light_per_channel_pdf(IntersectionPayload next_payload, uint sceneID = 0) {
        GeometryData next_geometry = scene_read_geometry(next_payload.hit.geometryID);
        float4 nee_pdf = 0.f;
        int lightID = next_geometry.lightID;
        if (lightID >= 0) {
            lightID += next_payload.hit.primitiveID;
            ilight::sample_li_pdf_in nee_pdf_i;
            nee_pdf_i.lightID = lightID;
            nee_pdf_i.ref_point = hit.position;
            nee_pdf_i.ref_normal = hit.geometryNormal;
            nee_pdf_i.light_point = next_payload.hit.position;
            nee_pdf_i.light_normal = next_payload.hit.geometryNormal;
            nee_pdf = lights::nee_lbvh_pdf_with_per_channel_pdf(nee_pdf_i,
                (uint)ImportanceFacotr::Use_Power
                | (uint)ImportanceFacotr::Use_Distance
                | (uint)ImportanceFacotr::Use_Cone, sceneID);
        }
        return nee_pdf;
    }

    /**
     * Sample the BSDF at the intersection point.
     * Additionally, return the per-channel pdf for RCV.
     * @param ri: incoming ray
     * @param u: random sample in [0, 1]^2
     * @param cv: output per-channel pdf
     * @return Optional containing the sampled BSDF information, or none if sampling failed.
     */
    Optional<ibsdf::sample_out> bsdf_sample_with_perchannel_pdf(
        Ray ri, float3 u, out float3 cv
    ) {
        Frame shading_frame = hit.get_shading_frame();
        ibsdf::sample_in bs_i;
        bs_i.u = u;
        bs_i.wi = -ri.direction;
        bs_i.shading_frame = shading_frame;
        bs_i.geometric_normal = hit.geometryNormal;
        GeometryData geometry = scene_read_geometry(hit.geometryID);
        Optional<MaterialData> material = scene_read_material(geometry.materialID);
        if (!material.hasValue) { cv = float3(1, 1, 1); return none; }
        return materials::bsdf_sample_with_perchannel_pdf(bs_i, material.value, hit.texcoord, cv);
    }

    /**
     * Sample the light at the intersection point.
     * Additionally, return the per-channel pdf for RCV.
     * @param ri: incoming ray
     * @param u: random sample in [0, 1]^2
     * @param cv: output per-channel pdf
     * @return Optional containing the sampled light information, or none if sampling failed.
     */
    Optional<ilight::sample_li_out> light_sample_with_perchannel_pdf(Ray ri, float3 u, out float3 cv, int sceneID = 0) {
        Frame shading_frame = hit.get_shading_frame();
        ilight::sample_li_in nee_i;
        nee_i.p = hit.position;
        nee_i.ns = hit.shadingNormal;
        nee_i.uv = u.xy;
        return lights::nee_lbvh_with_perchannel_pdf(nee_i, u.z,
            int(ImportanceFacotr::Use_Power 
              | ImportanceFacotr::Use_Distance 
              | ImportanceFacotr::Use_Cone), cv, sceneID);
    }

    /**
     * Return the emission of the material at the intersection point.
     */
    float3 emission() {
        GeometryData geometry = scene_read_geometry(hit.geometryID);
        Optional<MaterialData> material = scene_read_material(geometry.materialID);
        if (material.hasValue) {
            return material.value.emission();
        }
        return float3(0.f, 0.f, 0.f);
    }
}

/**
 * Sample the light at the intersection point.
 * Additionally, return the per-channel pdf for RCV.
 * @param ri: incoming ray
 * @param u: random sample in [0, 1]^2
 * @param cv: output per-channel pdf
 * @return Optional containing the sampled light information, or none if sampling failed.
 */
Optional<ilight::sample_li_out> light_sample_with_perchannel_pdf(float3 position, Ray ri, float3 u, out float3 cv, uint sceneID = 0) {
    ilight::sample_li_in nee_i;
    nee_i.p = position;
    nee_i.ns = float3(0, 0, 0);
    nee_i.uv = u.xy;
    return lights::nee_lbvh_with_perchannel_pdf(nee_i, u.z,
            int(ImportanceFacotr::Use_Power 
              | ImportanceFacotr::Use_Distance 
              | ImportanceFacotr::Use_Cone), cv, sceneID);
}

struct light_sample_ctx {
    float3 position;
    float3 geometry_normal;
    float4 aux_pdf;

    static light_sample_ctx from_payload(IntersectionPayload payload, float4 _aux_pdf) {
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
    const IntersectionPayload light_payload,
) {
    float nee_pdf = 0.f;
    int lightID = se_geometry_buffers[light_payload.hit.geometryID].lightID;
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

float4 pdf_nee_with_per_channel_pdf(
    const light_sample_ctx sample_ctx,
    const IntersectionPayload light_payload,
) {
    float4 nee_pdf = 0.f;
    int lightID = se_geometry_buffers[light_payload.hit.geometryID].lightID;
    if (lightID >= 0) {
        lightID += light_payload.hit.primitiveID;
        ilight::sample_li_pdf_in nee_pdf_i;
        nee_pdf_i.lightID = lightID;
        nee_pdf_i.ref_point = sample_ctx.position;
        nee_pdf_i.ref_normal = sample_ctx.geometry_normal;
        nee_pdf_i.light_point = light_payload.hit.position;
        nee_pdf_i.light_normal = light_payload.hit.geometryNormal;
        nee_pdf = lights::nee_lbvh_pdf_with_per_channel_pdf(nee_pdf_i,
            (uint)ImportanceFacotr::Use_Power
            | (uint)ImportanceFacotr::Use_Distance
            | (uint)ImportanceFacotr::Use_Cone);
        nee_pdf = discard_nan_inf(nee_pdf);
    }
    return nee_pdf;
}

#endif // _SRENDERER_SPT_UTILS_SLANG_