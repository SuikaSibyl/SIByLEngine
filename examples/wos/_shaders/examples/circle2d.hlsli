#ifndef _WOS_CIRCLE_EXAMPLE_2D_HLSLI_
#define _WOS_CIRCLE_EXAMPLE_2D_HLSLI_

#include "../core/pde.hlsli"
#include "../core/source.hlsli"
#include "common/math.hlsli"

struct Circle2D_PDEImpl : IPDEImpl<2, 3> {
    typealias input_t = vector<float, 2>;

    // returns the dirichlet boundary value at the input point
    static float3 dirichlet_impl(input_t x) { return 0.8; }

    // returns the source value at the input point
    static float3 source_impl(input_t x) { return float3(0, 0, 0); }

    static float diffusion_impl(
        input_t x,
        bool computeGradient, bool computeLaplacian,
        out input_t gradient, out float laplacian
    ) {
        if (computeGradient || computeLaplacian) {
            gradient = float2(0, 0);
        }

        if (computeLaplacian) {
            laplacian = 0.f;
        }

        return 1.f;
    }

    static source_sample<2, 3> sample_source_impl(uint seed) {
        // generate a sample from the source
        RandomSamplerState rng; rng.state = seed;
        source_sample<2, 3> sample;
        sample.position = float2(0, 0);
        return sample;
    }
    
    static boundary_sample<2, 3> sample_boundary_impl(uint seed) {
        boundary_sample<2, 3> sample;
        sample.position = float2(0, 0);
        sample.normal = float2(0, 0);
        sample.values = float3(0, 0, 0);
        sample.pdf = 1.0f;
        return sample;
    }

    static float pdf_source_impl(float2 p) { return 0.f; }

    // returns the absorption coefficient value at the input point
    static float3 absorption_impl(input_t x) { return 0.0f; }

    static float diffusion_gradient_impl(input_t x, out input_t gradient) {
        gradient = 0.f;
        return 0.f; 
    }

    static float get_bound_impl() { return 1.0f; }

    static bool interior_only_impl() { return true; }
};

struct CircleBoundary2D : IBoundary<2, 3> {
    // return the radius of the boundary
    static float radius() { return 0.953615; }
    // returns the distance from the point x to the boundary
    static float distance(float2 x) { return radius() - length(x); }
    // return the length of the boundary
    static float length() { return 2 * 3.14159265359f * radius(); }
    
    // sample the boundary
    static boundary_sample<2, 3> sample_boundary(uint seed) {
        // initialize the random number generator
        RandomSamplerState rng; rng.state = seed;
        // generate a point on the boundary
        const float2 pos = uniform_on_disk(GetNextRandom(rng));
        boundary_sample<2, 3> sample;
        sample.position = pos * radius();
        sample.normal = pos;
        sample.values = float3(1, 1, 1);
        sample.pdf = 1.0f / length();
        return sample;
    }

    // return the closest point on the boundary to the point x
    static float2 closest_point(float2 x) {
        return radius() * normalize(x);
    }
}

#endif // _WOS_CIRCLE_EXAMPLE_2D_HLSLI_