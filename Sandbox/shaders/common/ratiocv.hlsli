#ifndef _SRENDERER_COMMON_RATIO_CV_HEADER_
#define _SRENDERER_COMMON_RATIO_CV_HEADER_

#include "math.hlsli"

struct primal_auxiliary<let N : int> {
    float primal;
    vector<float, N> auxiliaries;
};

primal_auxiliary<N> discard_nan_inf<let N : int>(primal_auxiliary<N> x) {
    x.primal = discard_nan_inf(x.primal);
    for (int i = 0; i < N; i++)
        x.auxiliaries[i] = discard_nan_inf(x.auxiliaries[i]);
    return x;
}

primal_auxiliary<N> max<let N : int>(primal_auxiliary<N> x, float y) {
    x.primal = max(x.primal, y);
    for (int i = 0; i < N; i++)
        x.auxiliaries[i] = max(x.auxiliaries[i], y);
    return x;
}

float4 as_float4(primal_auxiliary<3> x) {
    return float4(x.auxiliaries, x.primal);
}

#endif // _SRENDERER_COMMON_RATIO_CV_HEADER_