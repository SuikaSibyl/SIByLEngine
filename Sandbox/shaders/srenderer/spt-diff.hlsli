#ifndef _SRENDERER_SPT_DIFF_HEADER_
#define _SRENDERER_SPT_DIFF_HEADER_

#include "common/math.hlsli"

struct DifferentiableParameter {
    int dim_0;
    int dim_1;
    int dim_2;
    int grad_replica;
    int offset_primal;
    int offset_gradient;
    float default_value;
    int dim_aux;

    int flatten_index(int3 index) { return index.x * dim_1 * dim_2 + index.y * dim_2 + index.z; }
    int flatten_index_grad(int3 index) { return index.x * dim_1 * dim_2 + index.y * dim_2 + index.z; }
    int2 get_texture_dim() { return int2(dim_0, dim_1); }
    int pixel_to_index(int2 pixel, int channel = 0) { return flatten_index(int3(pixel.x, pixel.y, channel)); }
    int pixel_to_index_grad(int2 pixel, int channel = 0) { return flatten_index_grad(int3(pixel.x, pixel.y, channel)); }
    int pitch() { return dim_0 * dim_1 * dim_2; }
    Optional<int> offset(int replica = 0, int aux = 0) {
        if (replica >= grad_replica) return none;
        if (aux >= dim_aux) return none;
        return pitch() * (replica * (1 + dim_aux) + (1 + aux));
    }
};

RWStructuredBuffer<float> GPUScene_param_primal;
RWByteAddressBuffer GPUScene_param_gradient;
RWStructuredBuffer<DifferentiableParameter> GPUScene_param_packets;

/**
 * Load the primal value of a differentiable parameter.
 */
[BackwardDerivative(bwd_load_dparam)]
float load_dparam(no_diff DifferentiableParameter param, no_diff uint index)
{ return GPUScene_param_primal[param.offset_primal + index]; }

void bwd_load_dparam(
    no_diff DifferentiableParameter param,
    no_diff uint index,
    float.Differential d_output)
{ GPUScene_param_gradient.InterlockedAddF32((param.offset_gradient + index) * 4, discard_nan_inf(d_output)); }

/**
 * Load the primal value of a differentiable parameter.
 * with placeholder for auxiliary data want to splat in bwd pass.
 */
[BackwardDerivative(bwd_load_dparam_w_aux_float2)]
float load_dparam_w_aux_float2(
    no_diff DifferentiableParameter param, no_diff uint index,
    no_diff float2 aux)
{ return GPUScene_param_primal[param.offset_primal + index]; }

void bwd_load_dparam_w_aux_float2(
    no_diff DifferentiableParameter param,
    no_diff uint index,
    no_diff float2 aux,
    float.Differential d_output
) { 
    // splat the gradient
    GPUScene_param_gradient.InterlockedAddF32((param.offset_gradient + index) * 4, discard_nan_inf(d_output));
    // splat the aux data
    for (int i = 0; i < 1; ++i) {
        Optional<int> offset = param.offset(0, i);
        if (offset.hasValue) {
            GPUScene_param_gradient.InterlockedAddF32((param.offset_gradient + offset.value + index) * 4, discard_nan_inf(aux[i]));
        }
    }
}

float load_dparam_gradient(
    no_diff DifferentiableParameter param,
    no_diff uint index)
{ return GPUScene_param_gradient.Load<float>((param.offset_gradient + index) * 4); }

void set_dparam_gradient(
    no_diff DifferentiableParameter param,
    no_diff uint index, float value)
{ GPUScene_param_gradient.Store<float>((param.offset_gradient + index) * 4, value); }

float load_dparam_aux(
    no_diff DifferentiableParameter param,
    no_diff uint param_index, no_diff uint aux_idx, float value)
{
    Optional<int> offset = param.offset(0, aux_idx);
    if (offset.hasValue) {
        return GPUScene_param_gradient.Load<float>((param.offset_gradient + offset.value + param_index) * 4);
    }
    return float(0);
}

void set_dparam_aux(
    no_diff DifferentiableParameter param,
    no_diff uint param_index, no_diff uint aux_idx, float value)
{
    Optional<int> offset = param.offset(0, aux_idx);
    if (offset.hasValue) {
        GPUScene_param_gradient.Store<float>((param.offset_gradient + offset.value + param_index) * 4, value);
    }
}

float load_dparam_aux(
    no_diff DifferentiableParameter param,
    no_diff uint element_idx,
    no_diff uint aux_idx)
{
    Optional<int> offset = param.offset(0, aux_idx);
    if (offset.hasValue) return GPUScene_param_gradient.Load<float>((param.offset_gradient + offset.value + element_idx) * 4);
    else return 0.f;
}

// vector<float, N> load_dparam_aux<let N : int>(
//     no_diff DifferentiableParameter param,
//     no_diff uint index,
//     no_diff uint aux_index)
// {
//     // splat the aux data
//     for (int i = 0; i < N; ++i) {
//         Optional<int> offset = param.offset(0, i);
//         if (let ofst = offset as int) {
//             GPUScene_param_gradient.InterlockedAddF32((ofst + index) * 4, discard_nan_inf(aux[i]));
//         }
//     }
// }

[BackwardDifferentiable]
float software_bilinear_interpolation_load(
    no_diff float2 texcoord,
    no_diff uint channel,
    no_diff DifferentiableParameter param)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * param.get_texture_dim() + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    const float final_val =
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(0, 0), channel)) * neighbor_weights[0] +
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(0, 1), channel)) * neighbor_weights[1] +
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(1, 0), channel)) * neighbor_weights[2] +
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(1, 1), channel)) * neighbor_weights[3];
    return final_val;
}

[BackwardDerivative(bwd_software_bilinear_interpolation_load_w_aux_float2)]
float software_bilinear_interpolation_load_w_aux_float2(
    no_diff float2 texcoord,
    no_diff uint channel,
    no_diff DifferentiableParameter param,
    no_diff float2 aux)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * param.get_texture_dim() + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    const float final_val =
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(0, 0), channel)) * neighbor_weights[0] +
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(0, 1), channel)) * neighbor_weights[1] +
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(1, 0), channel)) * neighbor_weights[2] +
        load_dparam(param, param.pixel_to_index(lb_pixel + int2(1, 1), channel)) * neighbor_weights[3];
    return final_val;
}

void bwd_software_bilinear_interpolation_load_w_aux_float2(
    no_diff float2 texcoord,
    no_diff uint channel,
    no_diff DifferentiableParameter param,
    no_diff float2 aux,
    float.Differential d_output)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * param.get_texture_dim() + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    bwd_diff(load_dparam_w_aux_float2)(param, param.pixel_to_index(lb_pixel + int2(0, 0), channel), aux * neighbor_weights[0], d_output * neighbor_weights[0]);
    bwd_diff(load_dparam_w_aux_float2)(param, param.pixel_to_index(lb_pixel + int2(0, 1), channel), aux * neighbor_weights[1], d_output * neighbor_weights[1]);
    bwd_diff(load_dparam_w_aux_float2)(param, param.pixel_to_index(lb_pixel + int2(1, 0), channel), aux * neighbor_weights[2], d_output * neighbor_weights[2]);
    bwd_diff(load_dparam_w_aux_float2)(param, param.pixel_to_index(lb_pixel + int2(1, 1), channel), aux * neighbor_weights[3], d_output * neighbor_weights[3]);
}

float software_bilinear_interpolation_load_grad(
    no_diff float2 texcoord,
    no_diff uint channel,
    no_diff DifferentiableParameter param)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * param.get_texture_dim() + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    const float final_val =
        load_dparam_gradient(param, param.pixel_to_index_grad(lb_pixel + int2(0, 0), channel)) * neighbor_weights[0] +
        load_dparam_gradient(param, param.pixel_to_index_grad(lb_pixel + int2(0, 1), channel)) * neighbor_weights[1] +
        load_dparam_gradient(param, param.pixel_to_index_grad(lb_pixel + int2(1, 0), channel)) * neighbor_weights[2] +
        load_dparam_gradient(param, param.pixel_to_index_grad(lb_pixel + int2(1, 1), channel)) * neighbor_weights[3];
    return final_val;
}

float software_nearest_load_grad(
    no_diff float2 texcoord,
    no_diff uint channel,
    no_diff DifferentiableParameter param)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * param.get_texture_dim() + 0.5;
    return load_dparam_gradient(param, param.pixel_to_index_grad(int2(pixel), channel));
}

float software_nearest_load_aux(
    no_diff float2 texcoord,
    no_diff uint channel,
    no_diff uint aux_index,
    no_diff DifferentiableParameter param)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * param.get_texture_dim() + 0.5;
    return load_dparam_aux(param, param.pixel_to_index_grad(int2(pixel), channel), aux_index);
}

#endif // _SRENDERER_SPT_DIFF_HEADER_