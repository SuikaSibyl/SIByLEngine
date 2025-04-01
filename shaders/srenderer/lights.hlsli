#ifndef _SRENDERER_LIGHTS_HEADER_
#define _SRENDERER_LIGHTS_HEADER_

#include "lights/light.hlsli"
#include "lights/envmap.hlsli"
#include "shapes/shape.hlsli"
#include "shapes/triangles.hlsli"
#include "shapes/sphere.hlsli"
#include "shapes/rectangle.hlsli"
#include "common/sampling.hlsli"
#include "materials.hlsli"

namespace lights {
ilight::sample_li_out nee_given_light(ilight::sample_li_in i, LightPacket data) {
    switch (data.light_type) {
    case LightType::MESH_PRIMITIVE: {
        TriangleParameter triangle = TriangleParameter(data);
        ishape::sample_in ishape_i;
        ishape_i.position = i.p;
        ishape_i.normal = i.ns;
        ishape_i.uv = i.uv;
        ishape::sample ishape_o = Triangle::sample(ishape_i, triangle);
        ilight::sample_li_out o;
        o.wi = normalize(ishape_o.position - i.p);
        o.x = ishape_o.position;
        // Compute the corresponding emission
        const uint geometryID = data.uintscalar_1;
        const uint materialID = GPUScene_geometry[geometryID].materialID;
        MaterialData material = GPUScene_material[materialID];
        o.L = materials::emission(material);
        o.pdf = ishape_o.pdf;
        o.ns = ishape_o.normal;
        o.valid = true;
        o.isDelta = false;
        return o;
    }
    case LightType::SPHERE: {
        const uint geometryID = data.uintscalar_1;
        GeometryData geometry = GPUScene_geometry[geometryID];
        const float4x4 o2w = ObjectToWorld(geometry);
        const float4x4 o2wn = ObjectToWorldNormal(geometry);
        const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
        const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
        SphereParameter sphere = { sphere_center, sphere_radius };
        ishape::sample_in ishape_i;
        ishape_i.position = i.p;
        ishape_i.uv = i.uv;
        ishape::sample ishape_o = Sphere::sample(ishape_i, sphere);
        ilight::sample_li_out o;
        o.wi = normalize(ishape_o.position - i.p);
        o.x = ishape_o.position;
        const uint materialID = geometry.materialID;
        MaterialData material = GPUScene_material[materialID];
        o.L = materials::emission(material);
        o.pdf = ishape_o.pdf;
        o.ns = ishape_o.normal;
        o.valid = true;
        return o;
    }
    case LightType::RECTANGLE: {
        const uint geometryID = data.uintscalar_1;
        GeometryData geometry = GPUScene_geometry[geometryID];
        const float4x4 o2w = ObjectToWorld(geometry);
        const float4x4 o2wn = ObjectToWorldNormal(geometry);
        RectangleParameter rect = { o2w, o2wn };
        ishape::sample_in ishape_i;
        ishape_i.position = i.p;
        ishape_i.uv = i.uv;
        ishape::sample ishape_o = Rectangle::sample(ishape_i, rect);
        ilight::sample_li_out o;
        o.wi = normalize(ishape_o.position - i.p);
        o.x = ishape_o.position;
        const uint materialID = geometry.materialID;
        MaterialData material = GPUScene_material[materialID];
        o.L = materials::emission(material);
        o.pdf = ishape_o.pdf;
        o.ns = ishape_o.normal;
        o.valid = true;
        return o;
    }
    default: break;
    }
    ilight::sample_li_out o;
    return o;
}

float nee_given_light_pdf(ilight::sample_li_pdf_in i, LightPacket data) {
    switch (data.light_type) {
    case LightType::MESH_PRIMITIVE: {
    TriangleParameter triangle = TriangleParameter(data);
    ishape::pdf_in pdf_in;
    pdf_in.ref_point = i.ref_point;
    pdf_in.ref_normal = i.ref_normal;
    pdf_in.sample_point = i.light_point;
    pdf_in.sample_normal = i.light_normal;
    return Triangle::sample_pdf(pdf_in, triangle);
    }
    case LightType::SPHERE: {
    const uint geometryID = data.uintscalar_1;
    GeometryData geometry = GPUScene_geometry[geometryID];
    const float4x4 o2w = ObjectToWorld(geometry);
    const float4x4 o2wn = ObjectToWorldNormal(geometry);
    const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
    SphereParameter sphere = { sphere_center, sphere_radius };
    ishape::pdf_in pdf_in;
    pdf_in.ref_point = i.ref_point;
    pdf_in.ref_normal = i.ref_normal;
    pdf_in.sample_point = i.light_point;
    pdf_in.sample_normal = i.light_normal;
    return Sphere::sample_pdf(pdf_in, sphere);
    }
    case LightType::RECTANGLE: {
    const uint geometryID = data.uintscalar_1;
    GeometryData geometry = GPUScene_geometry[geometryID];
    const float4x4 o2w = ObjectToWorld(geometry);
    const float4x4 o2wn = ObjectToWorldNormal(geometry);
    RectangleParameter rect = { o2w, o2wn };
    ishape::pdf_in pdf_in;
    pdf_in.ref_point = i.ref_point;
    pdf_in.ref_normal = i.ref_normal;
    pdf_in.sample_point = i.light_point;
    pdf_in.sample_normal = i.light_normal;
    return Rectangle::sample_pdf(pdf_in, rect);
    }
    default : break;
    }
    return 0.f;
}

ilight::sample_li_out nee_uniform(ilight::sample_li_in i, float u) {
    int max_light_id = GPUScene_description[0].max_light_count;
    const int light_id = clamp(int(u * max_light_id), 0, max_light_id - 1);
    LightPacket data = GPUScene_light[light_id];
    ilight::sample_li_out o = nee_given_light(i, data);
    o.pdf *= 1.0f / max_light_id;
    o.lightID = light_id;
    return o;
}

float nee_uniform_pdf(ilight::sample_li_pdf_in i) {
    int max_light_id = GPUScene_description[0].max_light_count;
    LightPacket data = GPUScene_light[i.lightID];
    float pdf = nee_given_light_pdf(i, data);
    return pdf * 1.0f / max_light_id;
}

float nee_lbvh_pdf(ilight::sample_li_pdf_in ctx, uint factors) {
    float pmf = 1;
    int nodeIndex = 0;
    bounds3 allb = fetchAllLightBounds();
    // Initialize local variables for BVH traversal for PMF computation
    uint32_t bitTrail = GPUScene_light_trail[ctx.lightID];
    // Compute light’s PMF by walking down tree nodes to the light
    while (true) {
        LightBVHNode node = GPUScene_light_bvh[nodeIndex];
        if (node.is_leaf()) {
            LightPacket data = GPUScene_light[ctx.lightID];
            return pmf * nee_given_light_pdf(ctx, data);
        }
        // Compute child importances and update PMF for current node
        LightBVHNode child0 = GPUScene_light_bvh[nodeIndex + 1];
        LightBVHNode child1 = GPUScene_light_bvh[node.child_or_light_index()];
        float ci[2] = { child0.importance(ctx.ref_point, ctx.ref_normal, allb, factors),
                        child1.importance(ctx.ref_point, ctx.ref_normal, allb, factors) };
        pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);
        
        // Use bitTrail to find next node index and update its value
        nodeIndex = bool(bitTrail & 1) ? node.child_or_light_index() : (nodeIndex + 1);
        bitTrail >>= 1;
    }
}

float4 nee_lbvh_pdf_with_aux(ilight::sample_li_pdf_in ctx, uint factors) {
    int nodeIndex = 0;
    float4 pmf = { 1, 1, 1, 1 };
    bounds3 allb = fetchAllLightBounds();
    // Initialize local variables for BVH traversal for PMF computation
    uint32_t bitTrail = GPUScene_light_trail[ctx.lightID];
    // Compute light’s PMF by walking down tree nodes to the light
    while (true) {
        LightBVHNode node = GPUScene_light_bvh[nodeIndex];
        if (node.is_leaf()) {
            LightPacket data = GPUScene_light[ctx.lightID];
            return pmf * nee_given_light_pdf(ctx, data);
        }
        // Compute child importances and update PMF for current node
        LightBVHNode child0 = GPUScene_light_bvh[nodeIndex + 1];
        LightBVHNode child1 = GPUScene_light_bvh[node.child_or_light_index()];
        // float ci[2] = { child0.importance(ctx.ref_point, ctx.ref_normal, allb, factors),
        //                 child1.importance(ctx.ref_point, ctx.ref_normal, allb, factors) };
        primal_auxiliary<3> ci[2];
        ci[0] = child0.importance_with_aux(ctx.ref_point, ctx.ref_normal, allb, factors);
        ci[1] = child1.importance_with_aux(ctx.ref_point, ctx.ref_normal, allb, factors);
        ci[0] = max(discard_nan_inf(ci[0]), 0);
        ci[1] = max(discard_nan_inf(ci[1]), 0);
        pmf = as_float4(ci[bitTrail & 1]) / (as_float4(ci[0]) + as_float4(ci[1]));
        
        // Use bitTrail to find next node index and update its value
        nodeIndex = bool(bitTrail & 1) ? node.child_or_light_index() : (nodeIndex + 1);
        bitTrail >>= 1;
    }
}

ilight::sample_li_out nee_lbvh(ilight::sample_li_in i, float u, uint factors) {
    int nodeIndex = 0;
    float pmf = 1;
    bounds3 allb = fetchAllLightBounds();
    while (true) {
    // Process light BVH node for light sampling
    LightBVHNode node = GPUScene_light_bvh[nodeIndex];
    if (!node.is_leaf()) {
        // Compute light BVH child node importances
        LightBVHNode children[2] = {
            GPUScene_light_bvh[nodeIndex + 1],
            GPUScene_light_bvh[node.child_or_light_index()]
        };
        float ci[2] = { children[0].importance(i.p, i.ns, allb, factors),
                        children[1].importance(i.p, i.ns, allb, factors) };
        ci[0] = max(discard_nan_inf(ci[0]), 0);
        ci[1] = max(discard_nan_inf(ci[1]), 0);
        if (ci[0] == 0 && ci[1] == 0) return {};
        // Randomly sample light BVH child node
        float nodePMF;
        int child = sample_discrete(ci, u, nodePMF);
        pmf *= nodePMF;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node.child_or_light_index();
    }
    else {
        // Confirm light has nonzero importance before returning light sample
        if (nodeIndex > 0 || node.importance(i.p, i.ns, allb) > 0) {
            LightPacket data = GPUScene_light[node.child_or_light_index()];
            ilight::sample_li_out o = nee_given_light(i, data);
            o.pdf *= pmf;
            o.lightID = node.child_or_light_index();
            return o;
        }
        return {};
    }
    }
}

ilight::sample_li_out nee_lbvh_with_aux(
    ilight::sample_li_in i, float u, uint factors, out float3 aux
) {
    aux = { 0, 0, 0 };
    int nodeIndex = 0;
    primal_auxiliary<3> pmf = { 1, { 1, 1, 1 } };
    bounds3 allb = fetchAllLightBounds();
    while (true) {
    // Process light BVH node for light sampling
    LightBVHNode node = GPUScene_light_bvh[nodeIndex];
    if (!node.is_leaf()) {
        // Compute light BVH child node importances
        LightBVHNode children[2] = {
            GPUScene_light_bvh[nodeIndex + 1],
            GPUScene_light_bvh[node.child_or_light_index()]
        };

        primal_auxiliary<3> ci[2];
        ci[0] = children[0].importance_with_aux(i.p, i.ns, allb, factors);
        ci[1] = children[1].importance_with_aux(i.p, i.ns, allb, factors);
        ci[0] = max(discard_nan_inf(ci[0]), 0);
        ci[1] = max(discard_nan_inf(ci[1]), 0);
        if (ci[0].primal == 0 && ci[1].primal == 0) return {};
        // Randomly sample light BVH child node
        float nodePMF;
        // ci[0].primal = ci[0].auxiliaries[2];
        // ci[1].primal = ci[1].auxiliaries[2];
        float ci_primal[2] = { ci[0].primal, ci[1].primal };
        int child = sample_discrete(ci_primal, u, nodePMF);
        pmf.primal *= nodePMF;
        float3 auxiliary_sum = ci[0].auxiliaries + ci[1].auxiliaries;
        pmf.auxiliaries *= (child == 0) 
            ? ci[0].auxiliaries / auxiliary_sum 
            : ci[1].auxiliaries / auxiliary_sum;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node.child_or_light_index();
    }
    else {
        // Confirm light has nonzero importance before returning light sample
        if (nodeIndex > 0 || node.importance(i.p, i.ns, allb) > 0) {
            LightPacket data = GPUScene_light[node.child_or_light_index()];
            ilight::sample_li_out o = nee_given_light(i, data);
            aux = pmf.auxiliaries * o.pdf;
            o.pdf *= pmf.primal;
            o.lightID = node.child_or_light_index();
            return o;
        }
        return {};
    }
    }
}

enum PowerEnum {
    Luminance,
    RChannel,
    GChannel,
    BChannel
}

float power_from_node(LightBVHNode node, PowerEnum power) {
    switch (power) {
    case PowerEnum::Luminance: return node.power_luminance();
    case PowerEnum::RChannel: return node.power_r();
    case PowerEnum::GChannel: return node.power_g();
    case PowerEnum::BChannel: return node.power_b();
    }
    return 0;
}

ilight::sample_li_out nee_power(
    ilight::sample_li_in i, float u, 
    PowerEnum power = PowerEnum::Luminance
) {
    int nodeIndex = 0;
    float pmf = 1;
    bounds3 allb = fetchAllLightBounds();
    while (true) {
    // Process light BVH node for light sampling
    LightBVHNode node = GPUScene_light_bvh[nodeIndex];
    if (!node.is_leaf()) {
        // Compute light BVH child node importances
        LightBVHNode children[2] = {
            GPUScene_light_bvh[nodeIndex + 1],
            GPUScene_light_bvh[node.child_or_light_index()]
        };
        float ci[2] = { power_from_node(children[0], power),
                        power_from_node(children[1], power) };
        ci[0] = max(discard_nan_inf(ci[0]), 0);
        ci[1] = max(discard_nan_inf(ci[1]), 0);
        if (ci[0] == 0 && ci[1] == 0) return {};
        // Randomly sample light BVH child node
        float nodePMF;
        int child = sample_discrete(ci, u, nodePMF);
        pmf *= nodePMF;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node.child_or_light_index();
    }
    else {
        // Confirm light has nonzero importance before returning light sample
        if (nodeIndex > 0 || node.importance(i.p, i.ns, allb) > 0) {
            LightPacket data = GPUScene_light[node.child_or_light_index()];
            ilight::sample_li_out o = nee_given_light(i, data);
            o.lightID = node.child_or_light_index();
            o.pdf *= pmf;
            return o;
        }
        return {};
    }
    }
}
}
#endif // _SRENDERER_LIGHTS_HEADER_