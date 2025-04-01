#include "common/random.hlsli"
#include "core/solver.hlsli"
#include "examples/bunny2d.hlsli"
#include "examples/circle2d.hlsli"
#include "examples/triangle2d.hlsli"

enum RenderMode {
    SDF_PREVIEW,
    COEF_PREVIEW,
    SOLUTION,
    SOLUTION_RCV,
};

enum PreviewEnum {
    DIRICHLET,
    SOURCE,
}

[[vk::push_constant]] cbuffer PushConstants {
    int random_seed;
    RenderMode render_mode;
    int spp;
    int max_steps;
    PreviewEnum preview_enum;
};

RWTexture2D<float4> output;

typealias PDE_IMPL_TYPE  = Bunny2D_PDEImpl;
typealias BOUNDARY_TYPE  = BunnyBoundary2D;
typealias GREENS_FN_TYPE = YukawaGreensFnBall2;
// typealias PDE_IMPL_TYPE = Circle2D_PDEImpl;
// typealias BOUNDARY_TYPE = TriangleBoundary2D;
// typealias GREENS_FN_TYPE = HarmonicGreensFnBall2;
static const int RCV_MAX_BOUNCE = 5;

// typealias PDE_IMPL_TYPE = Polygon2D_PDEImpl;
// typealias BOUNDARY_TYPE = PolygonBoundary2D;
// typealias GREENS_FN_TYPE = YukawaGreensFnBall2;

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(1024, 1024);
    const int2 pixel = int2(dtid.xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;

    const float2 uv = (pixel + 0.5) / float2(resolution.x, resolution.y);
    const float2 coord = lerp(float2(-1, 1), float2(1, -1), uv);
    
    RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);

    PDE_WRAPPER<2, 3, PDE_IMPL_TYPE> pde;   // define the PDE
    BOUNDARY_TYPE boundary;                 // define the boundary
    GREENS_FN_TYPE greens_fn = YukawaGreensFnBall2(0.01f); // define the Green's function
    // HarmonicGreensFnBall2 greens_fn = HarmonicGreensFnBall2();

    // const float Epsilon = 1e-6;
    // const float Xi = 1e-3;
    // const float ScreenSize = 1.0;

    // float3 v = float3(0);
    // float2 p = coord;
    // for (int i = 0; i < max_steps; ++i) {
    //     const float2 cp = boundary.closest_point(p);
    //     const float r = distance(cp, p);

    //     if (r < Epsilon || r > 5.0 * ScreenSize) {
    //         v += pde.dirichlet(p);
    //         break;
    //     }

    //     p = p + r * uniform_on_disk(GetNextRandom(RNG));
    // }
    // float3 color = v;

    float3 color = float3(0, 0, 0);
    switch (render_mode) {
    case RenderMode::SDF_PREVIEW: {
        float distance = boundary.distance(coord);
        color = sdf_colormap(distance);
        break;
    }
    case RenderMode::COEF_PREVIEW: {
        float3 coef_val = 0.f;
        switch (preview_enum) {
        case PreviewEnum::DIRICHLET:
            coef_val = pde.dirichlet(coord);
            break;
        case PreviewEnum::SOURCE:
            coef_val = pde.source(coord);
            break;
        }
        color = coef_val;
        break;
    }
    case RenderMode::SOLUTION:
    case RenderMode::SOLUTION_RCV: {
        SampleStatistics<2, 3, RCV_MAX_BOUNCE> statistics = SampleStatistics<2, 3, RCV_MAX_BOUNCE>(0.f);
        WoS<2, 3, RCV_MAX_BOUNCE, PDE_WRAPPER<2, 3, PDE_IMPL_TYPE>,
            GREENS_FN_TYPE, BOUNDARY_TYPE>.solve(
                pde, boundary, greens_fn, 1e-4, max_steps,
                coord, spp, 0.f, statistics, RNG);
                
        if (render_mode == RenderMode::SOLUTION) {
            color = float3(statistics.getSolution());
            if (any(isnan(color))) color = float3(1, 0, 1);
        } else if (render_mode == RenderMode::SOLUTION_RCV) {
            color = float3(statistics.getSolution_rcv());
        }

        float distance = boundary.distance(coord);
        if (abs(distance) < 0.005) color = float3(0.8, 0.8, 0.8);
        
        // float boundary = pde.absorption(float3(coord, 0));
        // color = float3(boundary, boundary, boundary);
        // color = checkerboard_pattern(coord);
        // if (insideDomain(coord)) color *= 0.2;
        // else color *= 0.1;
        // float4 boundary = boundary_preview(coord);
        // if (boundary.w == 1.0) color = boundary.rgb;
        break;
    }
    }

    output[pixel] = float4(color, 1);
}