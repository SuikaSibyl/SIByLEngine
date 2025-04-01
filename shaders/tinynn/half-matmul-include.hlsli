#ifndef _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_
#define _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_

void __inline_set_half_shared_buffer(int i, float16_t value) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_set_half_shared_buffer($0, $1)";
}

float16_t __inline_get_half_shared_buffer(int i) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_get_half_shared_buffer($0)";
}

// ========================================================================
// 16x16 matrix multiplication
// ------------------------------------------------------------------------
// For 16x16 matrix multiplication, ...

void __inline_wmma_128_16_16() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_128_16_16()";
}

void __inline_wmma_16_128_16() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_16_128_16()";
}

// ========================================================================
// 32x32 matrix multiplication
// ------------------------------------------------------------------------
// For 32x32 matrix multiplication, ...

void __inline_wmma_128_32_32() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_128_32_32()";
}

void __inline_wmma_32_128_32() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_32_128_32()";
}

// ========================================================================
// 64x64 matrix multiplication
// ------------------------------------------------------------------------
// For 64x64 matrix multiplication, we load the weights from global memory
// directly to subgroup matrix object. This is because the weights are
// out of the shared memory size. Meanwhile, we use the shared memory to
// store the input tensor.
//
// The matrix weights should be gathered in a specified buffer:
// [[vk::binding(0, 1)]] StructuredBuffer<float16_t> u_weights;
// It is probably necessary to explicitly declare the above resource in Slang.
// And the matrix is a continuous trunks of 64x64 float16_t with 
// a global offset from the start.

// load an element from the u_weights buffer
float16_t __inline_load_weights_buffer(int index) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_load_weights_buffer($0)";
}

// matrix multiplication for 64x64 matrix and a 64x128 tensor
// assume the weights are already loaded into the shared memory
// and the matrix weights are residing in the u_weights buffer
void __inline_wmma_128_64_64(int weights_offset) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_128_64_64($0)";
}

// explicitly load matrix weights from u_weights buffer
// to the subgroup matrix objects, can be shared by multiple calls of
// "__inline_wmma_128_64_64_preload_weights", which might be useful in some cases.
void __inline_wmma_load_64_64_matrices(int weights_offset) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_load_64_64_matrices($0)";
}

// matrix multiplication for 64x64 matrix and a 64x128 tensor
// assume the weights are already loaded into the shared memory
// and the matrix weights are already on the subgroup matrix objects,
// which is loaded by "__inline_wmma_load_64_64_matrices"
void __inline_wmma_128_64_64_preload_weights() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "tinynn/half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_128_64_64_preload_weights()";
}

#endif // !_SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_