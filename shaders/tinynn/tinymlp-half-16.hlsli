#ifndef _SRENDERER_TINYMLP_16_HEADER_
#define _SRENDERER_TINYMLP_16_HEADER_

#include "tinynn-common.hlsli"
#include "tinynn-activations.hlsli"
#include "tinynn/half-matmul-include.hlsli"

struct Feature16_Half {
    float16_t vals[16];
};

struct Linear16_Half<let has_bias:bool> {
    typedef Feature16_Half Input;
    typedef Feature16_Half Output;
    typedef uint SharedMemRef;

    no_diff TensorView_Half weights_view;
    no_diff TensorView_Half bias_view;
    
    __init() {}
    __init(inout uint offset_prim,
           inout uint offset_grad) {
        this.weights_view = TensorView_Half(offset_prim, offset_grad, 16, 16);
        offset_prim += 16 * 16; offset_grad += 16 * 16;
        if (has_bias) {
            this.bias_view = TensorView_Half(offset_prim, offset_grad);
            offset_prim += 16; offset_grad += 16;
        }
    }
    
    Output forward(Input input, ThreadInfo thread_info) {
        // Move the input and weights to shared memory.
        move_input_to_smem(input, thread_info);
        move_weights_to_smem<false>(thread_info);
        // Do the matmul.
        __inline_wmma_128_16_16();
        // Move the output to local memory.
        Output out_feature;
        const SharedMemRef outPtr = 3072 + 
        calc_offset<32 * 16>(thread_info);
        load_output(outPtr, thread_info, out_feature);
        // output the result.
        return out_feature;
    }

    static Output forward(Linear16_Half<has_bias> layer, Input input, ThreadInfo thread_info) {
        return layer.forward(input, thread_info); }

    // Move the input array to the shared memory.
    void move_input_to_smem(
        Input input, ThreadInfo thread_info) {
        // Pack in row-major format.
        SharedMemRef inPtr = calc_offset<32 * 16>(thread_info); ;
        store_array<false>(inPtr, thread_info, input);
    }

    // move the weights from global memory to shared memory.
    void move_weights_to_smem<let colMajor : bool>(
        ThreadInfo thread_info) {
        const SharedMemRef wtPtr = 2048 + 
        calc_offset<16 * 16>(thread_info);
        const int2 threadIdx = thread_info.thread_idx;
        // Copy weights to shared memory.
        const int i_base = threadIdx.x % 16;
        const int j_base = (threadIdx.x / 16) * 8;
        [ForceUnroll] for (uint j = 0; j < 8; j++) {
            const float16_t w = float16_t(weights_view.
                load_prim(i_base, j + j_base));
            if (colMajor) __inline_set_half_shared_buffer(
                wtPtr + i_base * 16 + j + j_base, w);
            else __inline_set_half_shared_buffer(
                wtPtr + (j + j_base) * 16 + i_base, w);
        }
    }
    
    // Load the output array from the shared memory.
    void load_output(SharedMemRef outPtr, 
        ThreadInfo threadInfo, inout Output outputs) {
        load_array<false>(outPtr, threadInfo, outputs);
        if (has_bias) [ForceUnroll] for (int i = 0; i < 16; i++)
            outputs.vals[i] = outputs.vals[i] + 
            float16_t(bias_view.load_prim(i));
    }

    // Load the output array from shared memory.
    void load_array<let colMajor : bool>(SharedMemRef memptr, 
        ThreadInfo threadInfo, inout Output output) {
        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        // Each thread in the warp will move N contiguous elements 
        // from their corresponding shared memory.
        if (!colMajor) { [ForceUnroll] for (int i = 0; i < 16; i++)
            // rowMajor matrix loading
            output.vals[i] = __inline_get_half_shared_buffer(
                memptr + threadIdInWarp * 16 + i);
        } else { [ForceUnroll] for (int i = 0; i < 16; i++)
            // colMajor matrix loading
            output.vals[i] = __inline_get_half_shared_buffer(
                memptr + i * 32 + threadIdInWarp); }}

    // Store the input array to the shared memory.
    void store_array<let colMajor : bool>(
        SharedMemRef memptr, ThreadInfo threadInfo, Input input) {
        const uint threadIdInWarp = threadInfo.thread_idx.x % 32;
        // Each thread in the warp will move N contiguous 
        // elements to their corresponding shared memory.
        if (!colMajor) { [ForceUnroll] for (int i = 0; i < 16; i++) 
            // rowMajor matrix writing
            __inline_set_half_shared_buffer(memptr + 
            (threadIdInWarp * 16 + i), float16_t(input.vals[i]));
        } else { [ForceUnroll] for (int i = 0; i < 16; i++) 
            // rowMajor matrix writing
            __inline_set_half_shared_buffer(memptr + 
            (i * 32 + threadIdInWarp), float16_t(input.vals[i])); }}

    // Define how to arrange memory in shared memory.
    uint calc_offset<let N : int>(ThreadInfo threadInfo) {
        return uint(((threadInfo.thread_idx.x / 32) + 
            threadInfo.thread_idx.y * 
            (threadInfo.block_dim.x * 1.0 / 32)) * N); }
};

// // A standard MLP with 16 neurons in each layer.
// // N: the number of layers.
// // Act: the activation function.
// struct MLP16_Half<let N:int, Act:IActivationFn> {
//     typedef Feature16_Half Input;
//     typedef Feature16_Half Output;
//     Array<Linear16_Half, N> linears;

//     __init(inout uint offset_prim, inout uint offset_grad) {
//         [ForceUnroll] for (int i = 0; i < N; i++)
//         linears[i] = Linear16_Half(offset_prim, offset_grad); }

//     Output _foward(Input input, ThreadInfo thread_info) {
//         Output out_feature = input;
//         [ForceUnroll] for (int i = 0; i < N; i++) {
//             // Do the affine transformation.
//             out_feature = Linear16_Half.foward(linears[i], 
//             out_feature, thread_info);
//             // Do the activation.
//             if(i != N-1) [ForceUnroll] for (int j = 0; j < 16; j++)
//             out_feature.vals[j] = Act.eval(out_feature.vals[j]); }
//         return out_feature; }
// }


#endif // _SRENDERER_TINYMLP_16_HEADER_