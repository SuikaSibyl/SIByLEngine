#ifndef _SRENDERER_DIFFERENTIABLE_RESOURCE_HEADER_
#define _SRENDERER_DIFFERENTIABLE_RESOURCE_HEADER_

[BackwardDerivative(bwd_diff_load_buffer)]
float load_buffer(
    no_diff RWByteAddressBuffer primal,
    no_diff RWByteAddressBuffer gradient,
    no_diff uint index)
{   return primal.Load<float>(index * 4); }

void bwd_diff_load_buffer(
    no_diff RWByteAddressBuffer primal,
    no_diff RWByteAddressBuffer gradient,
    no_diff uint index,
    float.Differential d_output)
{   gradient.InterlockedAddF32(index * 4, d_output); }

interface BufferBasedTexture {
    uint2 get_dimension();
    uint pixel_to_index(int2 pixel);
};

struct BufferBasedTexture_Linear {
    int2 dimension;
    __init(int width, int height) { dimension = int2(width, height); }
    __init(int2 resolution) { dimension = resolution; }
    uint2 get_dimension() { return dimension; }
    uint pixel_to_index(int2 pixel) { return pixel.y * dimension.x + pixel.x; };
};

[BackwardDifferentiable]
float software_bilinear_interpolation_buffer_load_r(
    no_diff RWByteAddressBuffer primal,
    no_diff RWByteAddressBuffer gradient,
    no_diff uint offset,
    no_diff float2 texcoord,
    no_diff BufferBasedTexture_Linear texture)
{
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * texture.get_dimension() + 0.5;
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
        load_buffer(primal, gradient, offset + texture.pixel_to_index(lb_pixel + int2(0, 0))) * neighbor_weights[0] +
        load_buffer(primal, gradient, offset + texture.pixel_to_index(lb_pixel + int2(0, 1))) * neighbor_weights[1] +
        load_buffer(primal, gradient, offset + texture.pixel_to_index(lb_pixel + int2(1, 0))) * neighbor_weights[2] +
        load_buffer(primal, gradient, offset + texture.pixel_to_index(lb_pixel + int2(1, 1))) * neighbor_weights[3];
    return final_val;
}


// struct BufferBasedTexture {
//     static const uint TEXMEM_BLOCK_SIZE = 16;
//     /** Flattens a 2D pixel position to a 1D index into a texture,
//      * so that neighboring pixels are locally coherent in memory.
//      * @param pixel The input 2D pixel position.
//      * @param texture The size of the texture.
//      * @return The flattened index. */
//     static uint pixel_to_index(int2 pixel, int data_pitch) {
//         // compute the block index and position within the block
//         const uint2 blockIdx = pixel / TEXMEM_BLOCK_SIZE;
//         const uint2 positionInBlock = pixel % TEXMEM_BLOCK_SIZE;
//         // compute the pitch of a row of blocks
//         const uint32_t renderWidthBlocks = (data_pitch + TEXMEM_BLOCK_SIZE - 1) / TEXMEM_BLOCK_SIZE;
//         const uint32_t reservoirBlockRowPitch = renderWidthBlocks * (TEXMEM_BLOCK_SIZE * TEXMEM_BLOCK_SIZE);
//         // return the flattened index
//         return blockIdx.y * reservoirBlockRowPitch
//         + blockIdx.x * (TEXMEM_BLOCK_SIZE * TEXMEM_BLOCK_SIZE)
//         + positionInBlock.y * TEXMEM_BLOCK_SIZE
//         + positionInBlock.x;
//     }
// };


#endif // _SRENDERER_DIFFERENTIABLE_RESOURCE_HEADER_