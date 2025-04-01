#ifndef _SRENDERER_MEDIUM_PHASE_HLSLI_
#define _SRENDERER_MEDIUM_PHASE_HLSLI_

#include "common/math.hlsli"
#include "common/geometry.hlsli"
#include "common/sampling.hlsli"

namespace iphase {
struct sample_o {
    float3 wi;
    float pdf;
    float3 p;
    bool valid;
};
};

struct PhasePacket {
    float3 g;
    float padding;
};

// Interface for phase functions parameters
interface IPhaseParameter : IDifferentiable {};

// Interface for phase functions
interface IPhaseFunction {
    // Associated a parameter type for each phase function
    associatedtype TParam : IPhaseParameter;

    // returns the value of the phase function
    // for the given pair of directions
    static float3 p(float3 wo, float3 wi, TParam param);

    // draw samples from the distribution
    // described by a phase function
    static iphase::sample_o sample_p(float3 wo, float2 u, TParam param);

    // pdf of the phase function sampling
    static float pdf(float3 wo, float3 wi, TParam param);
};


// Henyey-Greenstein phase function parameters
struct HGParam : IPhaseParameter{
    float3 g; // asymmetry parameter
    
    __init() { g = 0; }
    __init(PhasePacket data) { g = data.g; }
};

// Henyey-Greenstein phase function
struct HGPhaseFunction : IPhaseFunction {
    typedef HGParam TParam;

    static float3 p(float3 wo, float3 wi, TParam param) {
        return henyey_greenstein(dot(wo, wi), param.g);
    }

    static iphase::sample_o sample_p(float3 wo, float2 u, TParam param) {
        float3 p; float pdf;
        float3 wi = sample_henyey_greenstein(wo, param.g, u, p, pdf);
        return { wi, pdf, p, true };
    }

    static iphase::sample_o sample_p_wcv(float3 wo, float2 u, TParam param, out float3 cv) {
        float3 p; float pdf;
        float3 wi = sample_henyey_greenstein(wo, param.g, u, p, pdf);
        cv = p; return { wi, pdf, p, true };
    }

    static float pdf(float3 wo, float3 wi, TParam param) {
        return average(HGPhaseFunction::p(wo, wi, param));
    };
    
    static float3 henyey_greenstein(float cosTheta, float3 g) {
        const float3 denom = 1 + sqr(g) + 2 * g * cosTheta;
        return k_inv_4_pi * (1 - sqr(g)) / (denom * safe_sqrt(denom));
    }

    static float3 sample_henyey_greenstein(
        float3 wo, float3 g_chromatic, float2 u, out float3 p, out float pdf) {
        const int channel = sample_discrete<3>( { 1, 1, 1 }, u[0]);
        const float g = g_chromatic[channel];
        // Compute cos_theta for Henyey-Greenstein sample
        float cosTheta;
        if (abs(g) < 1e-3f) cosTheta = 1 - 2 * u[0];
        else cosTheta = -1 / (2 * g) *
                       (1 + sqr(g) - sqr((1 - sqr(g)) / (1 + g - 2 * g * u[0])));
        float sinTheta = safe_sqrt(1 - sqr(cosTheta));
        float phi = 2 * k_pi * u[1];
        Frame wFrame = Frame(createFrame(wo));
        float3 wi = wFrame.to_world(
            theta_phi_coord::SphericalDirection(sinTheta, cosTheta, phi));

        // Compute direction wi for Henyey–Greenstein sample
        p = henyey_greenstein(cosTheta, g_chromatic);
        pdf = average(p);
        return wi;
    }
};

namespace phases {
iphase::sample_o sample_p(float3 wo, float2 u, PhasePacket param) {
    return HGPhaseFunction::sample_p(wo, u, HGParam(param));
}

iphase::sample_o sample_p_wcv(float3 wo, float2 u, PhasePacket param, out float3 cv) {
    return HGPhaseFunction::sample_p_wcv(wo, u, HGParam(param), cv);
}

float3 p(float3 wo, float3 wi, PhasePacket param) {
    return HGPhaseFunction::p(wo, wi, HGParam(param));
}

float pdf(float3 wo, float3 wi, PhasePacket param) {
    return HGPhaseFunction::pdf(wo, wi, HGParam(param));
}
};

#endif // _SRENDERER_MEDIUM_PHASE_HLSLI_