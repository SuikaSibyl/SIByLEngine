#ifndef _SRENDERER_WOS_SOURCES_HLSLI_
#define _SRENDERER_WOS_SOURCES_HLSLI_

#include "common/sampling.hlsli"
#include "pde.hlsli"

struct SphereSource2D {
    float2 center;
    float radius;
    float3 source;

    source_sample<2, 3> sample_with_rcv(float2 uv) {
        const float pdf = 1. / area();
        source_sample<2, 3> sample;
        sample.position = center + radius * uniform_in_disk(uv);
        sample.pdf_sample = pdf;
        sample.pdf_channels = pdf;
        return sample;
    }

    bool is_inside(float2 x) { return length(x - center) < radius; }
    float area() { return k_pi * radius * radius; }
    float power() { return area() * luminance(source); }
    float3 power_rgb() { return area() * source; }
};

struct SphereSource2DList<let N:int> {
    Array<SphereSource2D, N> sources;

    source_sample<2, 3> sample_source(float3 uvw) {
        Array<float, N> pdf;
        Array<float3, N> pdf_rgb;

        float3 rgb_sum = float3(0, 0, 0);
        float luminance_sum = 0;

        for (int i = 0; i < N; ++i) {
            const float3 weights = sources[i].power_rgb();
            pdf_rgb[i] = weights;
            pdf[i] = luminance(weights);
            // try to sum up in all channels
            rgb_sum += pdf_rgb[i];
            luminance_sum += pdf[i];
        }

        for (int i = 0; i < N; ++i) {
            pdf[i] /= luminance_sum;
            pdf_rgb[i] /= rgb_sum;
        }

        float pmf; int index = sample_discrete(pdf, uvw.z, pmf);
        source_sample<2, 3> samples = sources[index].sample_with_rcv(uvw.xy);
        samples.pdf_sample *= pmf;
        samples.pdf_channels *= pdf_rgb[index];
        return samples;
    }

    float pdf_source(float2 p) {
        Array<float, N> pdf;
        Array<float3, N> pdf_rgb;

        float3 rgb_sum = float3(0, 0, 0);
        float luminance_sum = 0;

        for (int i = 0; i < N; ++i) {
            const float3 weights = sources[i].power_rgb();
            pdf_rgb[i] = weights;
            pdf[i] = luminance(weights);
            // try to sum up in all channels
            rgb_sum += pdf_rgb[i];
            luminance_sum += pdf[i];
        }

        for (int i = 0; i < N; ++i) {
            pdf[i] /= luminance_sum;
            pdf_rgb[i] /= rgb_sum;
        }
        
        float pdf_return = 0;
        for (int i = 0; i < N; ++i) {
            if (sources[i].is_inside(p)) {
                pdf_return += pdf[i] / sources[i].area();
            }
        }
        return pdf_return;
    }

    float3 source(float2 p) {
        float3 result = float3(0, 0, 0);
        for (int i = 0; i < N; ++i) {
            if (sources[i].is_inside(p)) {
                result += sources[i].source;
            }
        }
        return result;
    }

    float pdf(float2 pdf) {
        return 0.f;
    }

};

#endif // _SRENDERER_WOS_SOURCES_HLSLI_