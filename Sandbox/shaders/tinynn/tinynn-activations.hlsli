#ifndef _SRENDERER_TINYNN_ACTIVATION_HLSLI_HEADER_
#define _SRENDERER_TINYNN_ACTIVATION_HLSLI_HEADER_

// Interace : Activation function
interface IActivationFn {
    [Differentiable] static float eval(float x);
    [Differentiable] static float16_t eval(float16_t x);
};

// relu
struct ReLU : IActivationFn {
    [Differentiable] static float eval(float x) { return max(0.0f, x); }
    [Differentiable] static float16_t eval(float16_t x) { return max(float16_t(0.0f), x); }
};

// leaky relu
struct LeakyReLU : IActivationFn {
    [Differentiable] static float eval(float x) {
        return max(0.0f, x) + 0.01f * min(0.0f, x); }
    [Differentiable] static float16_t eval(float16_t x) {
        return max(float16_t(0.0f), x) + float16_t(0.01f) * min(float16_t(0.0f), x); }
};

// exponential
struct Exponential : IActivationFn {
    [Differentiable] static float eval(float x) { return exp(x); }
    [Differentiable] static float2 eval(float2 x) { return exp(x); }
    [Differentiable] static float3 eval(float3 x) { return exp(x); }
    [Differentiable] static float4 eval(float4 x) { return exp(x); }
    [Differentiable] static float16_t eval(float16_t x) { return exp(x); }
};

// sigmoid
struct Sigmoid : IActivationFn {
    [Differentiable] static float16_t eval(float16_t x) { return float16_t(1.0f) / (float16_t(1.0f) + exp(-x)); }
    [Differentiable] static float eval(float x) { return 1.0f / (1.0f + exp(-x)); }
    [Differentiable] static float3 eval(float3 xyz) { return float3(eval(xyz.x), eval(xyz.y), eval(xyz.z)); }
};

// silu
struct SiLU : IActivationFn {
    [Differentiable] static float16_t eval(float16_t x) { return x * Sigmoid.eval(x); }
};

// softplus
struct Softplus : IActivationFn {
    [Differentiable] static float eval(float x) { return log(1.0f + exp(x)); }
};

// sine
struct Sine : IActivationFn {
    [Differentiable] static float eval(float x) { return sin(x); }
    [Differentiable] static float16_t eval(float16_t x) { return sin(x); }
};

// tanh
struct Tanh : IActivationFn {
    [Differentiable] static float eval(float x) { return tanh(x); }
    [Differentiable] static float16_t eval(float16_t x) { return tanh(x); }
};

#endif // _SRENDERER_TINYNN_ACTIVATION_HLSLI_HEADER_