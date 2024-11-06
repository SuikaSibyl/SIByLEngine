#ifndef _SRENDERER_SPT_DIFF_HEADER_
#define _SRENDERER_SPT_DIFF_HEADER_

struct DifferentiableParameter {
    int dim_0;
    int dim_1;
    int dim_2;
    int grad_replica;
    int offset_primal;
    int offset_gradient;
    float default_value;
    int padding;

    int flatten_index(int3 index) { return index.x * dim_1 * dim_2 + index.y * dim_2 + index.z; }
    int2 get_texture_dim() { return int2(dim_0, dim_1); }
    int pixel_to_index(int2 pixel, int channel = 0) { return flatten_index(int3(pixel.x, pixel.y, channel)); }
};

RWStructuredBuffer<float> GPUScene_param_primal;
RWByteAddressBuffer GPUScene_param_gradient;
RWStructuredBuffer<DifferentiableParameter> GPUScene_param_packets;

[BackwardDerivative(bwd_load_dparam)]
float load_dparam(no_diff uint index)
{ return GPUScene_param_primal[index]; }

void bwd_load_dparam(
    no_diff uint index,
    float.Differential d_output)
{ GPUScene_param_gradient.InterlockedAddF32(index * 4, d_output); }

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
        load_dparam(param.pixel_to_index(lb_pixel + int2(0, 0), channel)) * neighbor_weights[0] +
        load_dparam(param.pixel_to_index(lb_pixel + int2(0, 1), channel)) * neighbor_weights[1] +
        load_dparam(param.pixel_to_index(lb_pixel + int2(1, 0), channel)) * neighbor_weights[2] +
        load_dparam(param.pixel_to_index(lb_pixel + int2(1, 1), channel)) * neighbor_weights[3];
    return final_val;
}

#endif // _SRENDERER_SPT_DIFF_HEADER_