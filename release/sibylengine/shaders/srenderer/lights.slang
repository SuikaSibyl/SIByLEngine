#ifndef _SRENDERER_LIGHTS_HEADER_
#define _SRENDERER_LIGHTS_HEADER_

#include "srenderer/lights/light.slang"
#include "srenderer/materials.slang"
#include "srenderer/shapes/sphere.slang"
#include "srenderer/shapes/triangle.slang"
#include "common/sampling.slang"

namespace lights {
Optional<ilight::sample_li_out> nee_given_light(ilight::sample_li_in i, LightData data) {
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
        const uint materialID = scene_read_geometry(geometryID).materialID;
        MaterialData material = scene_read_material(materialID);
        o.L = material.emission();
        o.pdf = ishape_o.pdf;
        o.ns = ishape_o.normal;
        o.valid = true;
        o.isDelta = false;
        return o;
    }
    case LightType::SPHERE: {
        const uint geometryID = data.uintscalar_1;
        GeometryData geometry = scene_read_geometry(geometryID);
        const float4x4 o2w = geometry.object_to_world();
        const float4x4 o2wn = geometry.object_to_world_normal();
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
        MaterialData material = scene_read_material(materialID);
        o.L = material.emission();
        o.pdf = ishape_o.pdf;
        o.ns = ishape_o.normal;
        o.valid = true;
        // Usually happens when a point on sphere light is the used as ishape_i.position
        if (distance(o.x, i.p) == 0) return none;
        return o;
    }
    // case LightType::RECTANGLE: {
    //     const uint geometryID = data.uintscalar_1;
    //     GeometryData geometry = GPUScene_geometry[geometryID];
    //     const float4x4 o2w = ObjectToWorld(geometry);
    //     const float4x4 o2wn = ObjectToWorldNormal(geometry);
    //     RectangleParameter rect = { o2w, o2wn };
    //     ishape::sample_in ishape_i;
    //     ishape_i.position = i.p;
    //     ishape_i.uv = i.uv;
    //     ishape::sample ishape_o = Rectangle::sample(ishape_i, rect);
    //     ilight::sample_li_out o;
    //     o.wi = normalize(ishape_o.position - i.p);
    //     o.x = ishape_o.position;
    //     const uint materialID = geometry.materialID;
    //     MaterialData material = GPUScene_material[materialID];
    //     o.L = materials::emission(material);
    //     o.pdf = ishape_o.pdf;
    //     o.ns = ishape_o.normal;
    //     o.valid = true;
    //     return o;
    // }
    default: break;
    }
    ilight::sample_li_out o;
    return o;
}

float nee_given_light_pdf(ilight::sample_li_pdf_in i, LightData data) {
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
    GeometryData geometry = scene_read_geometry(geometryID);
    const float4x4 o2w = geometry.object_to_world();
    const float4x4 o2wn = geometry.object_to_world_normal();
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
    // case LightType::RECTANGLE: {
    // const uint geometryID = data.uintscalar_1;
    // GeometryData geometry = scene_read_geometry(geometryID);
    // const float4x4 o2w = geometry.object_to_world();
    // const float4x4 o2wn = geometry.object_to_world_normal();
    // RectangleParameter rect = { o2w, o2wn };
    // ishape::pdf_in pdf_in;
    // pdf_in.ref_point = i.ref_point;
    // pdf_in.ref_normal = i.ref_normal;
    // pdf_in.sample_point = i.light_point;
    // pdf_in.sample_normal = i.light_normal;
    // return Rectangle::sample_pdf(pdf_in, rect);
    // }
    default : break;
    }
    return 0.f;
}

Optional<ilight::sample_li_out> nee_uniform(ilight::sample_li_in i, float u, uint sceneID) {
    int max_light_id = scene_read_scene_info(sceneID).nonDistantLightCount;
    const int light_id = clamp(int(u * max_light_id), 0, max_light_id - 1);
    LightData data = scene_read_light(light_id);
    if (let o = nee_given_light(i, data)) {
        ilight::sample_li_out lo = o;
        lo.pdf *= 1.0f / max_light_id;
        lo.lightID = light_id;
        return lo;
    }
    return none;
}

float nee_uniform_pdf(ilight::sample_li_pdf_in i, uint lbvhID) {
    int max_light_id = scene_read_scene_info(lbvhID).nonDistantLightCount;
    LightData data = scene_read_light(i.lightID);
    float pdf = nee_given_light_pdf(i, data);
    return pdf * 1.0f / max_light_id;
}

float nee_lbvh_pdf(ilight::sample_li_pdf_in ctx, uint factors, uint lbvhID = 0) {
    float pmf = 1;
    int nodeIndex = 0;
    bounds3 allb = scene_read_scene_info(lbvhID).light_bounds();
    // Initialize local variables for BVH traversal for PMF computation
    uint32_t bitTrail = scene_read_lbvh_trail(lbvhID, ctx.lightID);
    // Compute light’s PMF by walking down tree nodes to the light
    while (true) {
        LightBVHNode node = scene_read_lbvh_node(lbvhID, nodeIndex);
        if (node.is_leaf()) {
            LightData data = scene_read_light(ctx.lightID);
            return pmf * nee_given_light_pdf(ctx, data);
        }
        // Compute child importances and update PMF for current node
        LightBVHNode child0 = scene_read_lbvh_node(lbvhID, nodeIndex + 1);
        LightBVHNode child1 = scene_read_lbvh_node(lbvhID, node.child_or_light_index());
        float ci[2] = { child0.importance(ctx.ref_point, ctx.ref_normal, allb, factors),
                        child1.importance(ctx.ref_point, ctx.ref_normal, allb, factors) };
        pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);
        
        // Use bitTrail to find next node index and update its value
        nodeIndex = bool(bitTrail & 1) ? node.child_or_light_index() : (nodeIndex + 1);
        bitTrail >>= 1;
    }
}

float4 nee_lbvh_pdf_with_per_channel_pdf(ilight::sample_li_pdf_in ctx, uint factors, uint lbvhID = 0) {
    int nodeIndex = 0;
    float4 pmf = { 1, 1, 1, 1 };
    bounds3 allb = scene_read_scene_info(lbvhID).light_bounds();
    // Initialize local variables for BVH traversal for PMF computation
    uint32_t bitTrail = scene_read_lbvh_trail(lbvhID, ctx.lightID);
    // Compute light’s PMF by walking down tree nodes to the light
    while (true) {
        LightBVHNode node = scene_read_lbvh_node(lbvhID, nodeIndex);
        if (node.is_leaf()) {
            LightData data = scene_read_light(ctx.lightID);
            return pmf * nee_given_light_pdf(ctx, data);
        }
        // Compute child importances and update PMF for current node
        LightBVHNode child0 = scene_read_lbvh_node(lbvhID, nodeIndex + 1);
        LightBVHNode child1 = scene_read_lbvh_node(lbvhID, node.child_or_light_index());

        float4 ci[2];
        ci[0] = child0.importance_with_per_channel_pdf(ctx.ref_point, ctx.ref_normal, allb, factors);
        ci[1] = child1.importance_with_per_channel_pdf(ctx.ref_point, ctx.ref_normal, allb, factors);
        ci[0] = max(discard_nan_inf(ci[0]), 0);
        ci[1] = max(discard_nan_inf(ci[1]), 0);
        pmf = ci[bitTrail & 1] / (ci[0] + ci[1]);
        
        // Use bitTrail to find next node index and update its value
        nodeIndex = bool(bitTrail & 1) ? node.child_or_light_index() : (nodeIndex + 1);
        bitTrail >>= 1;
    }
}

Optional<ilight::sample_li_out> nee_lbvh(ilight::sample_li_in i, float u, uint factors, uint lbvhID = 0) {
    int nodeIndex = 0;
    float pmf = 1;
    bounds3 allb = scene_read_scene_info(lbvhID).light_bounds();
    while (true) {
    // Process light BVH node for light sampling
    LightBVHNode node = scene_read_lbvh_node(lbvhID, nodeIndex);
    if (!node.is_leaf()) {
        // Compute light BVH child node importances
        LightBVHNode children[2] = {
            scene_read_lbvh_node(lbvhID, nodeIndex + 1),
            scene_read_lbvh_node(lbvhID, node.child_or_light_index())
        };
        float ci[2] = { children[0].importance(i.p, i.ns, allb, factors),
                        children[1].importance(i.p, i.ns, allb, factors) };
        ci[0] = max(ci[0], 0);
        ci[1] = max(ci[1], 0);
        if (ci[0] == 0 && ci[1] == 0) return none;
        // Randomly sample light BVH child node
        float nodePMF;
        int child = sample_discrete(ci, u, nodePMF);
        pmf *= nodePMF;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node.child_or_light_index();
    }
    else {
        // Confirm light has nonzero importance before returning light sample
        if (nodeIndex > 0 || node.importance(i.p, i.ns, allb) > 0) {
            LightData data = scene_read_light(node.child_or_light_index());
            if (let o = nee_given_light(i, data)) {
                ilight::sample_li_out lo = o;
                lo.pdf *= pmf;
                lo.lightID = node.child_or_light_index();
                return lo;
            }
        }
        return none;
    }
    }
}

Optional<ilight::sample_li_out> nee_lbvh_with_perchannel_pdf(ilight::sample_li_in i, float u, uint factors, out float3 pdf, uint lbvhID = 0) {
    pdf = { 0, 0, 0 };
    int nodeIndex = 0;
    float4 pmf = { 1, 1, 1, 1 };
    bounds3 allb = scene_read_scene_info(lbvhID).light_bounds();
    while (true) {
    // Process light BVH node for light sampling
    LightBVHNode node = scene_read_lbvh_node(lbvhID, nodeIndex);
    if (!node.is_leaf()) {
        // Compute light BVH child node importances
        LightBVHNode children[2] = {
            scene_read_lbvh_node(lbvhID, nodeIndex + 1),
            scene_read_lbvh_node(lbvhID, node.child_or_light_index())
        };

        float4 ci[2];
        ci[0] = children[0].importance_with_per_channel_pdf(i.p, i.ns, allb, factors);
        ci[1] = children[1].importance_with_per_channel_pdf(i.p, i.ns, allb, factors);
        ci[0] = max(ci[0], 0);
        ci[1] = max(ci[1], 0);
        if (ci[0].x == 0 && ci[1].x == 0) return none;
        // Randomly sample light BVH child node
        float nodePMF;
        float ci_primal[2] = { ci[0].x, ci[1].x };
        int child = sample_discrete(ci_primal, u, nodePMF);
        pmf.x *= nodePMF;
        float3 auxiliary_sum = ci[0].yzw + ci[1].yzw;
        pmf.yzw *= (child == 0)
            ? ci[0].yzw / auxiliary_sum
            : ci[1].yzw / auxiliary_sum;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node.child_or_light_index();
    }
    else {
        // Confirm light has nonzero importance before returning light sample
        if (nodeIndex > 0 || node.importance(i.p, i.ns, allb) > 0) {
            LightData data = scene_read_light(node.child_or_light_index());
            if (let o = nee_given_light(i, data)) {
                ilight::sample_li_out lo = o;
                pdf = pmf.yzw * lo.pdf;
                lo.pdf *= pmf.x;
                lo.lightID = node.child_or_light_index();
                return lo;
            }
        }
        return none;
    }
    }
}

// enum PowerEnum {
//     Luminance,
//     RChannel,
//     GChannel,
//     BChannel
// }

// float power_from_node(LightBVHNode node, PowerEnum power) {
//     switch (power) {
//     case PowerEnum::Luminance: return node.power_luminance();
//     case PowerEnum::RChannel: return node.power_r();
//     case PowerEnum::GChannel: return node.power_g();
//     case PowerEnum::BChannel: return node.power_b();
//     }
//     return 0;
// }

// ilight::sample_li_out nee_power(
//     ilight::sample_li_in i, float u, 
//     PowerEnum power = PowerEnum::Luminance
// ) {
//     int nodeIndex = 0;
//     float pmf = 1;
//     bounds3 allb = fetchAllLightBounds();
//     while (true) {
//     // Process light BVH node for light sampling
//     LightBVHNode node = GPUScene_light_bvh[nodeIndex];
//     if (!node.is_leaf()) {
//         // Compute light BVH child node importances
//         LightBVHNode children[2] = {
//             GPUScene_light_bvh[nodeIndex + 1],
//             GPUScene_light_bvh[node.child_or_light_index()]
//         };
//         float ci[2] = { power_from_node(children[0], power),
//                         power_from_node(children[1], power) };
//         ci[0] = max(discard_nan_inf(ci[0]), 0);
//         ci[1] = max(discard_nan_inf(ci[1]), 0);
//         if (ci[0] == 0 && ci[1] == 0) return {};
//         // Randomly sample light BVH child node
//         float nodePMF;
//         int child = sample_discrete(ci, u, nodePMF);
//         pmf *= nodePMF;
//         nodeIndex = (child == 0) ? (nodeIndex + 1) : node.child_or_light_index();
//     }
//     else {
//         // Confirm light has nonzero importance before returning light sample
//         if (nodeIndex > 0 || node.importance(i.p, i.ns, allb) > 0) {
//             LightPacket data = GPUScene_light[node.child_or_light_index()];
//             ilight::sample_li_out o = nee_given_light(i, data);
//             o.lightID = node.child_or_light_index();
//             o.pdf *= pmf;
//             return o;
//         }
//         return {};
//     }
//     }
// }
}
#endif // _SRENDERER_LIGHTS_HEADER_