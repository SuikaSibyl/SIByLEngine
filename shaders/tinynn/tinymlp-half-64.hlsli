#ifndef _SRENDERER_TINYMLP_64_HEADER_
#define _SRENDERER_TINYMLP_64_HEADER_

#include "tinynn-common.hlsli"
#include "tinynn-activations.hlsli"
#include "tinynn/half-matmul-include.hlsli"

struct HalfFeature64 {
    float16_t vals[64];
};

struct Linear64_Half {
    typedef HalfFeature64 Input;
    typedef HalfFeature64 Output;
    TensorView_Half weights_view;
    TensorView_Half bias_view;
    
    __init(inout uint primal_offset,
           inout uint grad_offset) {
        this.weights_view = TensorView_Half(primal_offset, grad_offset);
        primal_offset += 64 * 64; grad_offset += 64 * 64;
        this.bias_view = TensorView_Half(primal_offset, grad_offset);
        primal_offset += 64; grad_offset += 64; 
    }
};

struct MLPHalf64X64<let N : int> {
    typedef HalfFeature64 Input;
    typedef HalfFeature64 Output;
    int offset; // offset of the weights in the global buffer
    
    __init(int weights_offset) { this.offset = weights_offset; }
    
    Output foward(Input input, ThreadInfo thread_info) {
        Output out_feature = input;
        // Do MLP forwarding
        [ForceUnroll] for (int i = 0; i < N; i++) {
            // load inputs to register
            load_input(out_feature, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // X = W * X: weights multiplied by inputs
            __inline_wmma_128_64_64(offset + i * (64 * 64 + 64));
            GroupMemoryBarrierWithGroupSync();
            // load X to the register
            load_output(out_feature, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // load bias to shared memory
            load_bias(offset + i * (64 * 64 + 64) + 64 * 64, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // X = Act(X + b): add bias and apply activation function
            if (i < N - 1) {
                [ForceUnroll] for (int j = 0; j < 64; j++)
                    out_feature.vals[j] = SiLU.eval(out_feature.vals[j]
                    + __inline_get_half_shared_buffer(j));
            }
            else {
                [ForceUnroll] for (int j = 0; j < 64; j++)
                    out_feature.vals[j] = out_feature.vals[j]
                    + __inline_get_half_shared_buffer(j);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        return out_feature;
    }

    float2 foward_with_jacobian(
        Input input, 
        Input dual_0,
        Input dual_1,
        ThreadInfo thread_info, 
        out float2x2 Jacobian) 
    {
        Output out_feature = input;
        Output dual_0_vec = dual_0;
        Output dual_1_vec = dual_1;
        // Do MLP forwarding
        [ForceUnroll] for (int i = 0; i < N; i++) {
            // load inputs to register
            load_input(out_feature, thread_info);
            __inline_wmma_load_64_64_matrices(offset + i * (64 * 64 + 64));
            GroupMemoryBarrierWithGroupSync();
            // X = W * X: weights multiplied by inputs
            __inline_wmma_128_64_64_preload_weights();
            GroupMemoryBarrierWithGroupSync();
            // load X to the register
            load_output(out_feature, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // ------------------------------------
            // do the same for the dual vectors 0
            load_input(dual_0_vec, thread_info);
            GroupMemoryBarrierWithGroupSync();
            __inline_wmma_128_64_64_preload_weights();
            GroupMemoryBarrierWithGroupSync();
            load_output(dual_0_vec, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // do the same for the dual vectors 1
            load_input(dual_1_vec, thread_info);
            GroupMemoryBarrierWithGroupSync();
            __inline_wmma_128_64_64_preload_weights();
            GroupMemoryBarrierWithGroupSync();
            load_output(dual_1_vec, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // ------------------------------------
            // load bias to shared memory
            load_bias(offset + i * (64 * 64 + 64) + 64 * 64, thread_info);
            GroupMemoryBarrierWithGroupSync();
            // X = Act(X + b): add bias and apply activation function
            if (i < N - 1) {
                [ForceUnroll] for (int j = 0; j < 64; j++) {
                    const float x = float(out_feature.vals[j]
                    + __inline_get_half_shared_buffer(j));
                    const float sigmoid = Sigmoid.eval(x);
                    const float silu = x * sigmoid;
                    const float dsilu = sigmoid * (x * (1 - sigmoid) + 1);
                    out_feature.vals[j] = float16_t(silu);
                    dual_0_vec.vals[j] = float16_t(dsilu * dual_0_vec.vals[j]);
                    dual_1_vec.vals[j] = float16_t(dsilu * dual_1_vec.vals[j]);
                }
            }
            else {
                [ForceUnroll] for (int j = 0; j < 64; j++)
                    out_feature.vals[j] = out_feature.vals[j]
                    + __inline_get_half_shared_buffer(j);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        Jacobian[0][0] = float(dual_0_vec.vals[0]);
        Jacobian[0][1] = float(dual_0_vec.vals[1]);
        Jacobian[1][0] = float(dual_1_vec.vals[0]);
        Jacobian[1][1] = float(dual_1_vec.vals[1]);
        
        return float2(out_feature.vals[0], out_feature.vals[1]);
    }

    void load_input(Input inputs, ThreadInfo thread_info) {
        for (int i = 0; i < 64; i++)
            __inline_set_half_shared_buffer(
                i * 128 + thread_info.gid, inputs.vals[i]);
    }

    void load_output(inout Output outputs, ThreadInfo thread_info) {
        for (int i = 0; i < 64; i++)
            outputs.vals[i] = __inline_get_half_shared_buffer(
                i * 128 + thread_info.gid);
    }

    void load_bias(int bias_offset, ThreadInfo thread_info) {
        if (thread_info.gid < 64) {
            // load bias to shared memory
            __inline_set_half_shared_buffer(thread_info.gid,
                __inline_load_weights_buffer(bias_offset + thread_info.gid));
        }
    }
};

struct HalfFeature32 {
    float16_t vals[32];
};

struct AlbedoMLP {
    typedef HalfFeature32 Input;
    typedef HalfFeature32 Output;
    int offset; // offset of the weights in the global buffer

    __init(int weights_offset) { this.offset = weights_offset; }
    
    float foward(float2 input, ThreadInfo thread_info) {
        // load inputs to shared memory
        __inline_set_half_shared_buffer(thread_info.gid,
        __inline_load_weights_buffer(offset + thread_info.gid));
        const float bias = __inline_load_weights_buffer(offset + 32 * 4);
        GroupMemoryBarrierWithGroupSync();
        // Do MLP forwarding
        float output = 0.f;
        [ForceUnroll] for (int i = 0; i < 32; i++) {
            const float tmp = __inline_get_half_shared_buffer(i) * input.x
                + __inline_get_half_shared_buffer(i + 32) * input.y
                + __inline_get_half_shared_buffer(i + 64);
            output += __inline_get_half_shared_buffer(i + 96) * SiLU.eval(tmp);
        }
        return Sigmoid.eval(output + bias);
    }
};

#endif // _SRENDERER_TINYMLP_64_HEADER_