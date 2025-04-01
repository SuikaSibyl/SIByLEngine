#ifndef _SRENDERER_RGLBRDF_MATERIAL_
#define _SRENDERER_RGLBRDF_MATERIAL_

#include "bxdf.hlsli"

struct RGLBRDFData {
    // ndf tensor // --- 0
    uint ndf_offset; 
    uint ndf_shape_0;
    uint ndf_shape_1;
    // sigma tensor
    uint sigma_offset; // --- 1
    uint sigma_shape_0;
    uint sigma_shape_1;
    // vndf tensor
    uint vndf_offset;
    uint vndf_shape_0; // --- 2
    uint vndf_shape_1;
    uint vndf_param_size_0;
    uint vndf_param_size_1;
    uint vndf_param_stride_0; // --- 3
    uint vndf_param_stride_1;
    uint vndf_param_offset_0;
    uint vndf_param_offset_1;
    uint vndf_marginal_offset; // --- 4
    uint vndf_conditional_offset;
    // luminance tensor
    uint luminance_offset;
    uint luminance_shape_0;
    uint luminance_shape_1; // --- 5
    uint luminance_param_size_0;
    uint luminance_param_size_1;
    uint luminance_param_stride_0;
    uint luminance_param_stride_1; // --- 6
    uint luminance_param_offset_0;
    uint luminance_param_offset_1;
    uint luminance_marginal_offset;
    uint luminance_conditional_offset; // --- 7
    // rgb tensor
    uint rgb_offset;
    uint rgb_shape_0;
    uint rgb_shape_1;
    uint rgb_param_size_0; // --- 8
    uint rgb_param_size_1;
    uint rgb_param_size_2;
    uint rgb_param_stride_0;
    uint rgb_param_stride_1; // --- 9
    uint rgb_param_stride_2;
    uint rgb_param_offset_0;
    uint rgb_param_offset_1;
    uint rgb_param_offset_2; // --- 10
    // other parameters
    bool isotropic;
    bool jacobian; // --- 6
    uint normalizer_offset;
    uint padding_1;
};

StructuredBuffer<RGLBRDFData> epfl_materials;
StructuredBuffer<float> epfl_data_tensors;

struct FloatStorage {
    __init(uint offset) { this.offset = offset; }
    uint offset;
    float load(uint index) { return epfl_data_tensors[offset + index]; }
};

namespace marginal2d {
struct invert_out {
    float2 sample;
    float pdf;
};

struct sample_out {
    float2 sample;
    float pdf;
};
}

float fma(float a, float b, float c) { return a * b + c; }

struct Marginal2D<let Dimension : int> {
#define ArraySize(Dimension) ((Dimension != 0) ? Dimension : 1)
    /// Resolution of the discretized density function
    uint2 m_size;
    /// Size of a bilinear patch in the unit square
    float2 m_patch_size; float2 m_inv_patch_size;
    /// Resolution of each parameter (optional)
    uint32_t m_param_size[ArraySize(Dimension)];
    /// Stride per parameter in units of sizeof(float)
    uint32_t m_param_strides[ArraySize(Dimension)];
    /// Discretization of each parameter domain
    FloatStorage m_param_values[ArraySize(Dimension)];
    /// Density values
    FloatStorage m_data;
    /// Marginal and conditional PDFs
    FloatStorage m_marginal_cdf;
    FloatStorage m_conditional_cdf;
    FloatStorage m_normalizer;

    bool pred_1(int dim, int idx, in vector<float, Dimension> param) {
        return m_param_values[dim].load(idx) <= param[dim];
    }

    // Find an interval in an ordered set
    int find_interval(uint32_t size_, int dim, 
        in vector<float, Dimension> param
    ) {
        int size = (int)size_ - 2;
        int first = 1;

        while (size > 0) {
            int half_ = (int)size >> 1;
            int middle = first + half_;
            // Evaluate the predicate */
            bool pred_result = pred_1(dim, middle, param);
            // .. and recurse into the left or right side
            first = pred_result ? middle + 1 : first;
            size = pred_result ? size - (half_ + 1) : half_;
        }

        return clamp((int)first - 1, (int)0, (int)size_ - 2);
    }

    float fetch_marginal(uint32_t idx, uint32_t offset,
        float param_weight[2 * ArraySize(Dimension)]) {
        return lookup<Dimension>(m_marginal_cdf, offset + idx,
                                 m_size.y, param_weight);
    };

    float fetch_conditional(uint32_t idx, uint32_t offset, uint32_t slice_size,
                            float param_weight[2 * ArraySize(Dimension)],
                            float2 sample) {
        float v0 = lookup<Dimension>(m_conditional_cdf, offset + idx,
                                     slice_size, param_weight);
        float v1 = lookup<Dimension>(m_conditional_cdf,
                                     offset + idx + m_size.x,
                                     slice_size, param_weight);
        return (1.f - sample.y) * v0 + sample.y * v1;
    };
    
    // Find an interval in an ordered set
    int find_interval(uint32_t size_, uint32_t offset,
        float param_weight[2 * ArraySize(Dimension)],
        float2 sample
    ) {
        int size = (int)size_ - 2;
        int first = 1;

        while (size > 0) {
            int half_ = (int)size >> 1;
            int middle = first + half_;
            // Evaluate the predicate */
            bool pred_result = fetch_marginal(middle, offset, param_weight) < sample.y;
            // .. and recurse into the left or right side
            first = pred_result ? middle + 1 : first;
            size = pred_result ? size - (half_ + 1) : half_;
        }

        return clamp((int)first - 1, (int)0, (int)size_ - 2);
    }

    // Find an interval in an ordered set
    int find_interval(uint32_t size_, uint32_t offset, uint32_t slice_size,
                      float param_weight[2 * ArraySize(Dimension)],
                      float2 sample
    ) {
        int size = (int)size_ - 2;
        int first = 1;

        while (size > 0) {
            int half_ = (int)size >> 1;
            int middle = first + half_;
            // Evaluate the predicate */
            bool pred_result = fetch_conditional(middle, offset, slice_size, param_weight, sample) < sample.x;
            // .. and recurse into the left or right side
            first = pred_result ? middle + 1 : first;
            size = pred_result ? size - (half_ + 1) : half_;
        }

        return clamp((int)first - 1, (int)0, (int)size_ - 2);
    }
    
    // Evaluate the density at position @pos.
    float eval(float2 pos) {
        /// Look up parameter-related indices and weights (if Dimension != 0)
        uint32_t slice_offset = 0u;
        /* Compute linear interpolation weights */
        pos *= m_inv_patch_size;
        uint2 offset = min(uint2(pos), m_size - 2u);

        float2 w1 = pos - float2(int2(offset));
        float2 w0 = float2(1.f) - w1;

        uint32_t index = offset.x + offset.y * m_size.x;

        uint32_t size = hprod(m_size);
        if (Dimension != 0)
            index += slice_offset * size;

        float v00 = lookup0(m_data, index, size);
        float v10 = lookup0(m_data, index + 1, size);
        float v01 = lookup0(m_data, index + m_size.x, size);
        float v11 = lookup0(m_data, index + m_size.x + 1, size);
        
        return fma(w0.y, fma(w0.x, v00, w1.x * v10),
                   w1.y * fma(w0.x, v01, w1.x * v11)) *
               hprod(m_inv_patch_size);
    }

    // Evaluate the density at position @pos.
    float eval(float2 pos, inout vector<float, Dimension> param) {
        /// Look up parameter-related indices and weights (if Dimension != 0)
        float param_weight[2 * ArraySize(Dimension)];
        uint32_t slice_offset = 0u;

        for (int dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = find_interval(
                m_param_size[dim], dim, param);

            float p0 = m_param_values[dim].load(param_index);
            float p1 = m_param_values[dim].load(param_index + 1);
            
            param_weight[2 * dim + 1] =
                clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }
        
        /* Compute linear interpolation weights */
        pos *= m_inv_patch_size;
        uint2 offset = min(uint2(pos), m_size - 2u);

        float2 w1 = pos - float2(int2(offset));
        float2 w0 = float2(1.f) - w1;

        uint32_t index = offset.x + offset.y * m_size.x;

        uint32_t size = hprod(m_size);
        if (Dimension != 0)
            index += slice_offset * size;

        float v00 = lookup<Dimension>(m_data, index, size, param_weight);
        float v10 = lookup<Dimension>(m_data, index + 1, size, param_weight);
        float v01 = lookup<Dimension>(m_data, index + m_size.x, size, param_weight);
        float v11 = lookup<Dimension>(m_data, index + m_size.x + 1, size, param_weight);
        
        return fma(w0.y, fma(w0.x, v00, w1.x * v10),
                   w1.y * fma(w0.x, v01, w1.x * v11)) *
               hprod(m_inv_patch_size);
    }

    // Given a uniformly distributed 2D sample, draw a sample from the
    // distribution(parameterized by \c param if applicable)
    marginal2d::sample_out sample(float2 sample,
        inout vector<float, Dimension> param) {
        marginal2d::sample_out o;
        /* Avoid degeneracies at the extrema */
        sample = clamp(sample, 1.f - k_inf, k_one_minus_epsilon);

        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize(Dimension)];
        uint32_t slice_offset = 0u;
        for (int dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = find_interval(
                m_param_size[dim], dim, param);

            float p0 = m_param_values[dim].load(param_index);
            float p1 = m_param_values[dim].load(param_index + 1);

            param_weight[2 * dim + 1] =
                clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Sample the row first */
        uint32_t offset = 0;
        if (Dimension != 0)
            offset = slice_offset * m_size.y;

        uint32_t row = find_interval(m_size.y, offset, param_weight, sample);

        sample.y -= fetch_marginal(row, offset, param_weight);

        uint32_t slice_size = hprod(m_size);
        offset = row * m_size.x;

        if (Dimension != 0)
            offset += slice_offset * slice_size;

        float r0 = lookup<Dimension>(m_conditional_cdf,
                                     offset + m_size.x - 1, 
                                     slice_size, param_weight);
        float r1 = lookup<Dimension>(m_conditional_cdf,
                                     offset + (m_size.x * 2 - 1), slice_size,
                                     param_weight);

        bool is_const = abs(r0 - r1) < 1e-4f * (r0 + r1);
        sample.y = is_const ? (2.f * sample.y) :
            (r0 - sqrt(r0 * r0 - 2.f * sample.y * (r0 - r1)));
        sample.y /= is_const ? (r0 + r1) : (r0 - r1);

        /* Sample the column next */
        sample.x *= (1.f - sample.y) * r0 + sample.y * r1;

        uint32_t col = find_interval(m_size.x, offset, slice_size, param_weight, sample);

        sample.x -= fetch_conditional(col, offset, slice_size, param_weight, sample);
        offset += col;
        
        float v00 = lookup<Dimension>(m_data, offset, slice_size, param_weight);
        float v10 = lookup<Dimension>(m_data, offset + 1, slice_size, param_weight);
        float v01 = lookup<Dimension>(m_data, offset + m_size.x, slice_size, param_weight);
        float v11 = lookup<Dimension>(m_data, offset + m_size.x + 1, slice_size, param_weight);
        float c0 = fma((1.f - sample.y), v00, sample.y * v01);
        float c1 = fma((1.f - sample.y), v10, sample.y * v11);

        is_const = abs(c0 - c1) < 1e-4f * (c0 + c1);
        sample.x = is_const ? (2.f * sample.x) :
            (c0 - sqrt(c0 * c0 - 2.f * sample.x * (c0 - c1)));
        sample.x /= is_const ? (c0 + c1) : (c0 - c1);

        o.sample = (float2(col, row) + sample) * m_patch_size;
        o.pdf = ((1.f - sample.x) * c0 + sample.x * c1) * hprod(m_inv_patch_size);
        return o;
    }

    /// Inverse of the mapping implemented in @sample()
    marginal2d::invert_out invert(float2 sample, inout vector<float, Dimension> param) {
        // Look up parameter-related indices and weights (if Dimension != 0)
        float param_weight[2 * ArraySize(Dimension)];
        uint32_t slice_offset = 0u;
        for (int dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = find_interval(
                m_param_size[dim], dim, param);

            float p0 = m_param_values[dim].load(param_index);
            float p1 = m_param_values[dim].load(param_index + 1);

            param_weight[2 * dim + 1] =
                clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Fetch values at corners of bilinear patch */
        sample *= m_inv_patch_size;
        uint2 pos = min(uint2(sample), m_size - 2u);
        sample -= float2(int2(pos));

        uint32_t offset = pos.x + pos.y * m_size.x;
        uint32_t slice_size = hprod(m_size);
        if (Dimension != 0)
            offset += slice_offset * slice_size;

        /* Invert the X component */
        float v00 = lookup<Dimension>(m_data, offset, slice_size, param_weight);
        float v10 = lookup<Dimension>(m_data, offset + 1, slice_size, param_weight);
        float v01 = lookup<Dimension>(m_data, offset + m_size.x, slice_size, param_weight);
        float v11 = lookup<Dimension>(m_data, offset + m_size.x + 1, slice_size, param_weight);

        float2 w1 = sample;
        float2 w0 = float2(1.f) - w1;

        float c0 = fma(w0.y, v00, w1.y * v01);
        float c1 = fma(w0.y, v10, w1.y * v11);
        float pdf = fma(w0.x, c0, w1.x * c1);

        sample.x *= c0 + .5f * sample.x * (c1 - c0);

        float v0 = lookup<Dimension>(m_conditional_cdf, offset, slice_size, param_weight);
        float v1 = lookup<Dimension>(m_conditional_cdf, offset + m_size.x, slice_size, param_weight);

        sample.x += (1.f - sample.y) * v0 + sample.y * v1;

        offset = pos.y * m_size.x;
        if (Dimension != 0) offset += slice_offset * slice_size;

        float r0 = lookup<Dimension>(m_conditional_cdf,
            offset + m_size.x - 1, slice_size, param_weight);
        float r1 = lookup<Dimension>(m_conditional_cdf,
            offset + (m_size.x * 2 - 1), slice_size, param_weight);

        sample.x /= (1.f - sample.y) * r0 + sample.y * r1;

        /* Invert the Y component */
        sample.y *= r0 + .5f * sample.y * (r1 - r0);

        offset = pos.y;
        if (Dimension != 0)
            offset += slice_offset * m_size.y;

        sample.y += lookup<Dimension>(m_marginal_cdf, offset, m_size.y, param_weight);

        marginal2d::invert_out o;
        o.sample = sample;
        o.pdf = pdf * hprod(m_inv_patch_size);
        return o;
    }

    float lookup<let Dim : int>(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        if(Dim == 3) return lookup3(data, i0, size, param_weight);
        else if(Dim == 2) return lookup2(data, i0, size, param_weight);
        else if(Dim == 1) return lookup1(data, i0, size, param_weight);
        else return lookup0(data, i0, size);
    }

    float lookup3(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        uint32_t i1 = i0 + m_param_strides[3 - 1] * size;
        float w0 = param_weight[2 * 3 - 2];
        float w1 = param_weight[2 * 3 - 1];
        float v0 = lookup2(data, i0, size, param_weight);
        float v1 = lookup2(data, i1, size, param_weight);
        return fma(v0, w0, v1 * w1);
    }

    float lookup2(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        uint32_t i1 = i0 + m_param_strides[2 - 1] * size;
        float w0 = param_weight[2 * 2 - 2];
        float w1 = param_weight[2 * 2 - 1];
        float v0 = lookup1(data, i0, size, param_weight);
        float v1 = lookup1(data, i1, size, param_weight);
        return fma(v0, w0, v1 * w1);
    }

    float lookup1(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        uint32_t i1 = i0 + m_param_strides[1 - 1] * size;
        float w0 = param_weight[2 * 1 - 2];
        float w1 = param_weight[2 * 1 - 1];
        float v0 = lookup0(data, i0, size);
        float v1 = lookup0(data, i1, size);
        return fma(v0, w0, v1 * w1);
    }

    float lookup0(
        FloatStorage data,
        uint32_t i0, uint32_t size
    ) { return data.load(i0); }
    
    // Evaluate the density at position @pos, both normalized and unnormalized
    // only intended for use with 3D marginals RGB eval
    float2 eval_dual(float2 pos, inout vector<float, Dimension> param) {
        /// Look up parameter-related indices and weights (if Dimension != 0)
        float param_weight[2 * ArraySize(Dimension)];
        uint32_t slice_offset = 0u;

        for (int dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = find_interval(
                m_param_size[dim], dim, param);

            float p0 = m_param_values[dim].load(param_index);
            float p1 = m_param_values[dim].load(param_index + 1);

            param_weight[2 * dim + 1] =
                clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Compute linear interpolation weights */
        pos *= m_inv_patch_size;
        uint2 offset = min(uint2(pos), m_size - 2u);

        float2 w1 = pos - float2(int2(offset));
        float2 w0 = float2(1.f) - w1;

        uint32_t index = offset.x + offset.y * m_size.x;

        uint32_t size = hprod(m_size);
        if (Dimension != 0)
            index += slice_offset * size;

        float2 v00 = lookup_dual<Dimension>(m_data, index, size, param_weight);
        float2 v10 = lookup_dual<Dimension>(m_data, index + 1, size, param_weight);
        float2 v01 = lookup_dual<Dimension>(m_data, index + m_size.x, size, param_weight);
        float2 v11 = lookup_dual<Dimension>(m_data, index + m_size.x + 1, size, param_weight);

        return float2(fma(w0.y, float2(fma(w0.x, v00, w1.x * v10)),
                   w1.y * float2(fma(w0.x, v01, w1.x * v11)))) *
               hprod(m_inv_patch_size);
    }

    float2 lookup_dual<let Dim : int>(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        if (Dim == 3) return lookup3_dual(data, i0, size, param_weight);
        else if (Dim == 2) return lookup2_dual(data, i0, size, param_weight);
        else return float2(0.);
    }

    // lookup both normalized and unnormalized values
    float2 lookup3_dual(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        uint32_t i1 = i0 + m_param_strides[3 - 1] * size;
        float w0 = param_weight[2 * 3 - 2];
        float w1 = param_weight[2 * 3 - 1];
        float2 v0 = lookup2_dual(data, i0, size, param_weight);
        float2 v1 = lookup2_dual(data, i1, size, param_weight);
        return float2(fma(v0, w0, v1 * w1));
    }

    // lookup both normalized and unnormalized values
    float2 lookup2_dual(
        FloatStorage data,
        uint32_t i0, uint32_t size,
        float param_weight[2 * ArraySize(Dimension)]) {
        uint32_t i1 = i0 + m_param_strides[2 - 1] * size;
        float w0 = param_weight[2 * 2 - 2];
        float w1 = param_weight[2 * 2 - 1];
        float n0 = m_normalizer.load(int(i0 / size));
        float n1 = m_normalizer.load(int(i1 / size));
        float v0 = lookup1(data, i0, size, param_weight);
        float v1 = lookup1(data, i1, size, param_weight);
        float2 v0_dual = float2(v0, v0 * n0);
        float2 v1_dual = float2(v1, v1 * n1);
        // float n0
        return float2(fma(v0_dual, w0, v1_dual * w1));
    }
#undef ArraySize
}

void initializeMarginal2D0(
    inout Marginal2D<0> marginal,
    uint2 size, uint offset
) {
    marginal.m_size = size;
    marginal.m_patch_size = float2(1.f) / float2(size - 1u);
    marginal.m_inv_patch_size = float2(size - 1u);
    marginal.m_data = FloatStorage(offset);
    marginal.m_param_size[0] = 1;
    marginal.m_param_strides[0] = 1;
}

void initializeMarginal2D2(
    inout Marginal2D<2> marginal,
    uint2 size, uint offset,
    uint2 param_size, uint2 param_stride,
    uint2 param_offset, uint marginal_offset,
    uint conditional_offset
) {
    marginal.m_size = size;
    marginal.m_patch_size = float2(1.f) / float2(size - 1u);
    marginal.m_inv_patch_size = float2(size - 1u);

    marginal.m_param_size[0] = param_size.x;
    marginal.m_param_size[1] = param_size.y;

    marginal.m_param_strides[0] = param_stride.x;
    marginal.m_param_strides[1] = param_stride.y;
    
    marginal.m_data = FloatStorage(offset);
    marginal.m_param_values[0] = FloatStorage(param_offset.x);
    marginal.m_param_values[1] = FloatStorage(param_offset.y);
    marginal.m_marginal_cdf = FloatStorage(marginal_offset);
    marginal.m_conditional_cdf = FloatStorage(conditional_offset);
}

void initializeMarginal2D3(
    inout Marginal2D<3> marginal,
    uint2 size, uint offset,
    uint3 param_size, uint3 param_stride,
    uint3 param_offset, uint normalizer_offset
) {
    marginal.m_size = size;
    marginal.m_patch_size = float2(1.f) / float2(size - 1u);
    marginal.m_inv_patch_size = float2(size - 1u);

    marginal.m_param_size[0] = param_size.x;
    marginal.m_param_size[1] = param_size.y;
    marginal.m_param_size[2] = param_size.z;

    marginal.m_param_strides[0] = param_stride.x;
    marginal.m_param_strides[1] = param_stride.y;
    marginal.m_param_strides[2] = param_stride.z;

    marginal.m_data = FloatStorage(offset);
    marginal.m_param_values[0] = FloatStorage(param_offset.x);
    marginal.m_param_values[1] = FloatStorage(param_offset.y);
    marginal.m_param_values[2] = FloatStorage(param_offset.z);

    marginal.m_normalizer = FloatStorage(normalizer_offset);
}

struct RGLMaterial : IBxDFParameter {
    // EPFL RGL BRDF data
    no_diff Marginal2D<0> ndf;
    no_diff Marginal2D<0> sigma;
    no_diff Marginal2D<2> vndf;
    no_diff Marginal2D<2> luminance;
    no_diff Marginal2D<3> rgb;
    no_diff bool isotropic;
    no_diff bool jacobian;
    
    __init(int index) {
        RGLBRDFData data = epfl_materials[index];
        isotropic = data.isotropic;
        jacobian = data.jacobian;
        initializeMarginal2D0(ndf, uint2(data.ndf_shape_0, 
        data.ndf_shape_1), data.ndf_offset);
        initializeMarginal2D0(sigma, uint2(data.sigma_shape_0, 
        data.sigma_shape_1), data.sigma_offset);
        initializeMarginal2D2(vndf, uint2(data.vndf_shape_0,
        data.vndf_shape_1), data.vndf_offset,
        uint2(data.vndf_param_size_0, data.vndf_param_size_1),
        uint2(data.vndf_param_stride_0, data.vndf_param_stride_1),
        uint2(data.vndf_param_offset_0, data.vndf_param_offset_1), 
        data.vndf_marginal_offset, data.vndf_conditional_offset);
        initializeMarginal2D2(luminance, uint2(data.luminance_shape_0,
        data.luminance_shape_1), data.luminance_offset,
        uint2(data.luminance_param_size_0, data.luminance_param_size_1),
        uint2(data.luminance_param_stride_0, data.luminance_param_stride_1),
        uint2(data.luminance_param_offset_0, data.luminance_param_offset_1),
        data.luminance_marginal_offset, data.luminance_conditional_offset);
        initializeMarginal2D3(rgb, uint2(data.rgb_shape_0,
        data.rgb_shape_1), data.rgb_offset,
        uint3(data.rgb_param_size_0, data.rgb_param_size_1, data.rgb_param_size_2),
        uint3(data.rgb_param_stride_0, data.rgb_param_stride_1, data.rgb_param_stride_2),
        uint3(data.rgb_param_offset_0, data.rgb_param_offset_1, data.rgb_param_offset_2),
        data.normalizer_offset);
    }
};

struct RGLBRDF : IBxDF {
    typedef RGLMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, RGLMaterial material) {
        const float3 wo = i.shading_frame.to_local(i.wo);
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi = float3(wi.x, wi.y, -wi.z);
            i.wi = i.shading_frame.to_world(wi); }
        if (wo.z < 0) { return float3(0); }
        
        float3 wm = normalize(wi + wo);
        
        // Cartesian -> spherical coordinates
        float theta_i = elevation(wi);
        float phi_i = atan2(wi.y, wi.x);
        float theta_m = elevation(wm);
        float phi_m = atan2(wm.y, wm.x);

        // Spherical coordinates -> unit coordinate system
        float2 u_wi = float2(ibsdf::theta2u(theta_i), ibsdf::phi2u(phi_i));
        float2 u_wm = float2(ibsdf::theta2u(theta_m),
            ibsdf::phi2u(material.isotropic ? (phi_m - phi_i) : phi_m));
        u_wm.y = u_wm.y - floor(u_wm.y);

        float2 params = float2(phi_i, theta_i);
        marginal2d::invert_out invert_o = material.vndf.invert(u_wm, params);
        float2 sample = invert_o.sample;
        float vndf_pdf = invert_o.pdf;
        
        float3 fr = float3(0.f);
        for (int i = 0; i < 3; ++i) {
            float3 params_fr = float3(phi_i, theta_i, float(i));
            fr[i] = material.rgb.eval(sample, params_fr);
            // clamp the value to zero 
            // (negative values occur when the original
            // spectral data goes out of gamut)
            fr[i] = max(0.f, fr[i]);
        }

        fr = fr * material.ndf.eval(u_wm) / (4 * material.sigma.eval(u_wi));

        return fr;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, RGLMaterial material) {
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { 
            wi = float3(wi.x, wi.y, -wi.z); 
            i.wi = i.shading_frame.to_world(wi); 
        }

        const float2 u = i.u.xy;
        float theta_i = elevation(wi);
        float phi_i = atan2(wi.y, wi.x);
        
        float2 params = float2(phi_i, theta_i);
        float2 u_wi = float2(ibsdf::theta2u(theta_i), ibsdf::phi2u(phi_i));
        float2 sample = float2(u.y, u.x);
        float lum_pdf = 1.f;

        marginal2d::sample_out lum_o = material.luminance.sample(sample, params);
        sample = lum_o.sample;
        lum_pdf = lum_o.pdf;

        marginal2d::sample_out vndf_o = material.vndf.sample(sample, params);
        float2 u_wm = vndf_o.sample;
        float ndf_pdf = vndf_o.pdf;

        float phi_m = ibsdf::u2phi(u_wm.y);
        float theta_m = ibsdf::u2theta(u_wm.x);

        if (material.isotropic)
            phi_m += phi_i;
        
        /* Spherical -> Cartesian coordinates */
        float sin_phi_m = sin(phi_m);
        float cos_phi_m = cos(phi_m);
        float sin_theta_m = sin(theta_m);
        float cos_theta_m = cos(theta_m);

        float3 wm = float3(
            cos_phi_m * sin_theta_m,
            sin_phi_m * sin_theta_m,
            cos_theta_m);
        
        float3 wo = wm * 2.f * dot(wm, wi) - wi;
        
        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(wo);

        ibsdf::eval_in i_eval;
        i_eval.wi = i.wi;
        i_eval.wo = o.wo;
        i_eval.geometric_normal = i.geometric_normal;
        i_eval.shading_frame = i.shading_frame;
        
        float3 fr = float3(0.f);
        for (int i = 0; i < 3; ++i) {
            float3 params_fr = float3(phi_i, theta_i, float(i));
            fr[i] = material.rgb.eval(sample, params_fr);
            /* clamp the value to zero (negative values occur when the original
               spectral data goes out of gamut) */
            fr[i] = max(0.f, fr[i]);
        }

        fr = fr * material.ndf.eval(u_wm) /
             (4 * material.sigma.eval(u_wi));

        float jacobian = max(2.f * sqr(k_pi) * u_wm.x *
            sin_theta_m, 1e-6f) * 4.f * dot(wi, wm);

        float pdf = ndf_pdf * lum_pdf / jacobian;
        
        o.pdf = pdf;
        o.bsdf = fr / pdf;
        if (wo.z <= 0) o.bsdf = float3(0.f);
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, RGLMaterial material) {
        const float3 wo = i.shading_frame.to_local(i.wo);
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) {
            wi = float3(wi.x, wi.y, -wi.z);
            i.wi = i.shading_frame.to_world(wi);
        }
        
        float3 wm = normalize(wi + wo);

        // Cartesian -> spherical coordinates
        float theta_i = elevation(wi);
        float phi_i = atan2(wi.y, wi.x);
        float theta_m = elevation(wm);
        float phi_m = atan2(wm.y, wm.x);

        // Spherical coordinates -> unit coordinate system
        float2 u_wm = float2(
            ibsdf::theta2u(theta_m),
            ibsdf::phi2u(material.isotropic ? (phi_m - phi_i) : phi_m));

        u_wm.y = u_wm.y - floor(u_wm.y);

        float2 params = float2(phi_i, theta_i);
        marginal2d::invert_out invert_o = material.vndf.invert(u_wm, params);
        float2 sample = invert_o.sample;
        float vndf_pdf = invert_o.pdf;

        float pdf = material.luminance.eval(sample, params);
        
        float sin_theta_m = sqrt(sqr(wm.x) + sqr(wm.y));
        float jacobian = max(2.f * sqr(k_pi) * u_wm.x *
            sin_theta_m, 1e-6f) * 4.f * dot(wi, wm);
        
        return vndf_pdf * pdf / jacobian;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample_with_perchannel_cv(
        ibsdf::sample_in i, RGLMaterial material,
        out float3 per_channel_pdf) {
        
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) {
            wi = float3(wi.x, wi.y, -wi.z);
            i.wi = i.shading_frame.to_world(wi);
        }
        
        const float2 u = i.u.xy;
        float theta_i = elevation(wi);
        float phi_i = atan2(wi.y, wi.x);

        float2 params = float2(phi_i, theta_i);
        float2 u_wi = float2(ibsdf::theta2u(theta_i), ibsdf::phi2u(phi_i));
        float2 sample = float2(u.y, u.x);
        float lum_pdf = 1.f;

        marginal2d::sample_out lum_o = material.luminance.sample(sample, params);
        sample = lum_o.sample;
        lum_pdf = lum_o.pdf;

        marginal2d::sample_out vndf_o = material.vndf.sample(sample, params);
        float2 u_wm = vndf_o.sample;
        float ndf_pdf = vndf_o.pdf;

        float phi_m = ibsdf::u2phi(u_wm.y);
        float theta_m = ibsdf::u2theta(u_wm.x);

        if (material.isotropic)
            phi_m += phi_i;

        /* Spherical -> Cartesian coordinates */
        float sin_phi_m = sin(phi_m);
        float cos_phi_m = cos(phi_m);
        float sin_theta_m = sin(theta_m);
        float cos_theta_m = cos(theta_m);

        float3 wm = float3(
            cos_phi_m * sin_theta_m,
            sin_phi_m * sin_theta_m,
            cos_theta_m);

        float3 wo = wm * 2.f * dot(wm, wi) - wi;

        ibsdf::sample_out o;
        o.wo = i.shading_frame.to_world(wo);

        ibsdf::eval_in i_eval;
        i_eval.wi = i.wi;
        i_eval.wo = o.wo;
        i_eval.geometric_normal = i.geometric_normal;
        i_eval.shading_frame = i.shading_frame;

        float3 rgb = float3(0.f);
        float3 rgb_normalized = float3(0.f);
        for (int i = 0; i < 3; ++i) {
            float3 params_fr = float3(phi_i, theta_i, float(i));
            float2 rgb_dual = material.rgb.eval_dual(sample, params_fr);
            rgb[i] = rgb_dual.x; rgb_normalized[i] = rgb_dual.y;
            /* clamp the value to zero (negative values occur when the original
               spectral data goes out of gamut) */
            rgb[i] = max(0.f, rgb[i]);
            rgb_normalized[i] = max(0.f, rgb_normalized[i]);
        }

        float3 fr = rgb * material.ndf.eval(u_wm) /
             (4 * material.sigma.eval(u_wi));

        float jacobian = max(2.f * sqr(k_pi) * u_wm.x *
                                 sin_theta_m, 1e-6f) * 4.f * dot(wi, wm);

        float pdf = ndf_pdf * lum_pdf / jacobian;
        per_channel_pdf = ndf_pdf * rgb_normalized / jacobian;
        
        o.pdf = pdf;
        o.bsdf = fr / pdf;
        if (wo.z <= 0) o.bsdf = float3(0.f);
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float3 rgb_per_channel_pdf(ibsdf::pdf_in i, RGLMaterial material) {
        const float3 wo = i.shading_frame.to_local(i.wo);
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) {
            wi = float3(wi.x, wi.y, -wi.z);
            i.wi = i.shading_frame.to_world(wi);
        }

        float3 wm = normalize(wi + wo);

        // Cartesian -> spherical coordinates
        float theta_i = elevation(wi);
        float phi_i = atan2(wi.y, wi.x);
        float theta_m = elevation(wm);
        float phi_m = atan2(wm.y, wm.x);

        // Spherical coordinates -> unit coordinate system
        float2 u_wm = float2(
            ibsdf::theta2u(theta_m),
            ibsdf::phi2u(material.isotropic ? (phi_m - phi_i) : phi_m));

        u_wm.y = u_wm.y - floor(u_wm.y);

        float2 params = float2(phi_i, theta_i);
        marginal2d::invert_out invert_o = material.vndf.invert(u_wm, params);
        float2 sample = invert_o.sample;
        float vndf_pdf = invert_o.pdf;

        float3 pdf = float3(0.f);
        for (int i = 0; i < 3; ++i) {
            float3 params_fr = float3(phi_i, theta_i, float(i));
            pdf[i] = material.rgb.eval(sample, params_fr);
            /* clamp the value to zero (negative values occur when the original
               spectral data goes out of gamut) */
            pdf[i] = max(0.f, pdf[i]);
        }
        
        float sin_theta_m = sqrt(sqr(wm.x) + sqr(wm.y));
        float jacobian = max(2.f * sqr(k_pi) * u_wm.x *
            sin_theta_m, 1e-6f) * 4.f * dot(wi, wm);
        
        return vndf_pdf * pdf / jacobian;
    }
};

#endif // _SRENDERER_RGLBRDF_MATERIAL_