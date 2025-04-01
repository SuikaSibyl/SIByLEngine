#ifndef _WOS_POLY2D_EXAMPLE_HLSLI
#define _WOS_POLY2D_EXAMPLE_HLSLI

#include "../core/pde.hlsli"
#include "../core/primitives.hlsli"
#include "common/math.hlsli"

struct Polygon2D_PDEImpl : IPDEImpl<2, 1> {
    typealias input_t = vector<float, 2>;

    // returns the dirichlet boundary value at the input point
    static float dirichlet_impl(input_t x) { return 1; }
    // returns the source value at the input point
    static float source_impl(input_t x) { return 0; }
    // returns the absorption coefficient value at the input point
    static float absorption_impl(input_t x) { return 0.f; }

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

    static float diffusion_gradient_impl(input_t x, out input_t gradient) {
        float laplacianStub;
        return diffusion_impl(x, true, false, gradient, laplacianStub);
    }
}

struct PolygonBoundary2D : IBoundary<2> {
    static const Polyline<3> boundary_dirichlet_0 = { float2(0.2, 0.1), float2(0.4, 0.3), float2(0.6, 0.5) };
    
    static float distance(float2 x) {
        float d = k_inf;
        d = min(d, distance_polyline(x, boundary_dirichlet_0));
        return d;   
    }
}

#endif // _WOS_POLY2D_EXAMPLE_HLSLI