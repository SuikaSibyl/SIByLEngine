#ifndef _SRENDERER_TINYNN_COMMON_HEADER_
#define _SRENDERER_TINYNN_COMMON_HEADER_

[[vk::binding(0, 1)]] StructuredBuffer<float16_t> primal_half_buffer;
[[vk::binding(1, 1)]] StructuredBuffer<float> primal_single_buffer;
[[vk::binding(2, 1)]] RWByteAddressBuffer gradient_buffer;

struct ThreadInfo {
    int2 thread_idx;
    int2 block_dim;
    int gid;
};

struct TensorView_Half {
    no_diff uint offset_prim;
    no_diff uint offset_grad;
    no_diff uint stride;
    no_diff uint pitch;

    __init(uint offset_prim, uint offset_grad) {
        this.offset_prim = offset_prim;
        this.offset_grad = offset_grad;
        this.stride = 1;
        this.pitch = 1;
    }

    __init(uint offset_prim, uint offset_grad, uint stride, uint pitch) {
        this.offset_prim = offset_prim;
        this.offset_grad = offset_grad;
        this.stride = stride;
        this.pitch = pitch;
    }

    // Load primal value
    [BackwardDerivative(load_prim_idx1_bwd)]
    float load_prim(int x) { 
    return float(primal_half_buffer[offset_prim + x]); }
    [BackwardDerivative(load_prim_idx2_bwd)]
    float load_prim(int x, int y) { 
    return load_prim(x * stride + y); }
    [BackwardDerivative(load_prim_idx3_bwd)]
    float load_prim(int x, int y, int z) { 
    return load_prim(x * stride + y * pitch + z); }
    
    // Interlocked add gradient to the buffer
    void interlocked_add_grad(int x, float val) { 
        gradient_buffer.InterlockedAddF32((offset_grad + x) * 4, val); }
    void interlocked_add_grad(int x, int y, float val) { 
        interlocked_add_grad(x * stride + y, val); }
    void interlocked_add_grad(int x, int y, int z, float val) { 
        interlocked_add_grad(x * stride + y * pitch + z, val); }
    
    // Backward derivative of primal value loading
    void load_prim_idx1_bwd(int x, float.Differential val) { 
        interlocked_add_grad(x, val); }
    void load_prim_idx2_bwd(int x, int y, float.Differential val) { 
        interlocked_add_grad(x, y, val); }
    void load_prim_idx3_bwd(int x, int y, int z, float.Differential val) { 
        interlocked_add_grad(x, y, z, val); }
};

struct TensorView_Single {
    no_diff uint offset_prim;
    no_diff uint offset_grad;
    no_diff uint stride;
    no_diff uint pitch;

    // Load primal value
    [BackwardDerivative(load_prim_idx1_bwd)]
    float load_prim(int x) {
        return primal_single_buffer[offset_prim + x]; }
    [BackwardDerivative(load_prim_idx2_bwd)]
    float load_prim(int x, int y) {
        return load_prim(x * stride + y); }
    [BackwardDerivative(load_prim_idx3_bwd)]
    float load_prim(int x, int y, int z) {
        return load_prim(x * stride + y * pitch + z); }

    // Interlocked add gradient to the buffer
    void interlocked_add_grad(int x, float val) {
        gradient_buffer.InterlockedAddF32((offset_grad + x) * 4, val); }
    void interlocked_add_grad(int x, int y, float val) {
        interlocked_add_grad(x * stride + y, val); }
    void interlocked_add_grad(int x, int y, int z, float val) {
        interlocked_add_grad(x * stride + y * pitch + z, val); }

    // Backward derivative of primal value loading
    void load_prim_idx1_bwd(int x, float.Differential val) {
        interlocked_add_grad(x, val); }
    void load_prim_idx2_bwd(int x, int y, float.Differential val) {
        interlocked_add_grad(x, y, val); }
    void load_prim_idx3_bwd(int x, int y, int z, float.Differential val) {
        interlocked_add_grad(x, y, z, val); }
};

#endif // !_SRENDERER_TINYNN_COMMON_HEADER_