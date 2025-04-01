#include "common/random.hlsli"
#include "core/solver.hlsli"
#include "examples/circle2d.hlsli"
#include "reverse-wos.hlsli"
#include "examples/triangle2d.hlsli"

[[vk::push_constant]] cbuffer PushConstants {
    int random_seed;
    int spp;
    int threads;
    int max_steps;
};

RWStructuredBuffer<GreenSphere2D> rw_spheres;

int flatten_index(int tid, int i_sample, int i_step) { 
    return tid * spp * max_steps + i_sample * max_steps + i_step; 
}

#define DIM 2
#define CHANNEL 3
typealias PDE_IMPL_TYPE = Triangle2D_PDEImpl;
typealias BOUNDARY_TYPE = TriangleBoundary2D;
typealias GREENS_FN_TYPE = HarmonicGreensFnBall2;
typealias vector_t = vector<float, DIM>;
typealias value_t = vector<float, CHANNEL>;

[shader("compute")]
[numthreads(128, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    RandomSamplerState RNG = InitRandomSampler(dtid.x, random_seed);

    PDE_WRAPPER<DIM, CHANNEL, PDE_IMPL_TYPE> pde;   // define the PDE
    BOUNDARY_TYPE boundary;                 // define the boundary
    GREENS_FN_TYPE greens_fn;               // define the Green's function
    
    const float Epsilon = 1e-6;
    const float Xi = 1e-3;
    const float ScreenSize = 1.0;

    for (int i = 0; i < spp; ++i) {
        boundary_sample<DIM, CHANNEL> bs = boundary.sample_boundary(
            GetNextRandom(RNG)
        );

        bs.pdf = 1.0;
        vector_t dp = bs.normal * Xi;

        float2 p;
        if (pde.interior_only()){
            p = bs.position - dp;

            if (!boundary.insideDomain(p)) {
                GreenSphere2D sphere;
                sphere.center = float2(1e9, 1e9);
                sphere.radius = 0;
                sphere.weight = 0;
                sphere.source = float3(0, 0, 0);
                sphere.type = dtid.x;
                for (int j = 0; j < max_steps; ++j) {
                    rw_spheres[flatten_index(dtid.x, i, j)] = sphere;
                }
                return;
            }
        }
        else {
            const float r = GetNextRandom(RNG);
            if (r < 0.5) p = bs.position - dp;
            else p = bs.position + dp;
        }

        for (int j = 0; j < max_steps; ++j) {
            vector_t cp = boundary.closest_point(p);
            const float r = distance(cp, p);
            // else, early return
            if (r < Epsilon || r > 5.0 * ScreenSize) {
                GreenSphere2D sphere;
                sphere.center = float2(1e9, 1e9);
                sphere.radius = 0;
                sphere.weight = 0;
                sphere.source = float3(0, 0, 0);
                sphere.type = dtid.x;
                for (; j < max_steps; ++j) {
                    rw_spheres[flatten_index(dtid.x, i, j)] = sphere;
                }
                break;
            }
            
            // Divide Xi for the finite difference method
            float weight = 1.0 / bs.pdf / (Xi * spp * threads);
            // Reuse the same walk to all pixels by drawing the Green's disk
            GreenSphere2D sphere;
            sphere.center = p;
            sphere.radius = r;
            sphere.weight = weight;
            sphere.source = bs.values;
            sphere.type = dtid.x;
            rw_spheres[flatten_index(dtid.x, i, j)] = sphere;

            // continue the walk
            p = p + r * uniform_on_disk(GetNextRandom(RNG));
        }
    }
}