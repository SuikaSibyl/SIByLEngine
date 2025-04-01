#ifndef _FORWARD_WOS_HLSLI_
#define _FORWARD_WOS_HLSLI_

#include "core/distributions.hlsli"
#include "core/pde.hlsli"

struct ForwardWoS
<let DimIn : int,
 let DimOut : int,
 PDE_t : IPde<DimIn, DimOut>,
 Green_t : IGreensFnBall<DimIn>,
 Boundary_t : IBoundary<DimIn, DimOut>>
{
    typealias vector_t = vector<float, DimIn>;
    typealias return_t = vector<float, DimOut>;

    static void solve(
        PDE_t pde,
        Boundary_t boundary,
        inout Green_t greensFn,
        float Epsilon, float ScreenSize,
        int maxWalkLength,
        vector_t p, int nWalks,
        return_t initVal,
        inout RandomSamplerState rng
    ) {
        return_t v = return_t(0);

        for (int i = 0; i < maxWalkLength; ++i) {
            const vector_t cp = boundary.closest_point(p);
            const float r = distance(cp, p);

            if (r < Epsilon || r > 5.0 * ScreenSize) {
                v += pde.dirichlet(p);
                break;
            }

            p = p + r * uniform_on_disk(GetNextRandom(rng));
        }
    }
};

#endif // _FORWARD_WOS_HLSLI_