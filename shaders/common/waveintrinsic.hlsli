#ifndef _SRENDERER_WAVEINTRINSIC_HLSLI_
#define _SRENDERER_WAVEINTRINSIC_HLSLI_

// 
float glsl_subgroupClusteredAdd_IDX2(float value) {
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_basic : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_clustered : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_shuffle : enable)");
    __intrinsic_asm "subgroupClusteredAdd($0, 2)";
}

//
float glsl_subgroupClusteredAdd_ClusterSize4(float value) {
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_basic : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_clustered : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_shuffle : enable)");
    __intrinsic_asm "subgroupClusteredAdd($0, 4)";
}

float glsl_subgroupClusteredAdd_ClusterSize16(float value) {
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_basic : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_clustered : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_shuffle : enable)");
    __intrinsic_asm "subgroupClusteredAdd($0, 16)";
}

float2 glsl_subgroupClusteredAdd_IDX2_float2(float2 value) {
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_basic : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_clustered : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_shuffle : enable)");
    __intrinsic_asm "subgroupClusteredAdd($0, 2)";
}

#endif // _SRENDERER_WAVEINTRINSIC_HLSLI_