#ifndef _SRENDERER_LIGHT_HEADER_
#define _SRENDERER_LIGHT_HEADER_

interface ILightParameter : IDifferentiable {};

namespace ilight {
struct eval_le_in {
    float3 p;
    float3 dir;
};

struct sample_li_in {
    float3 p;
    float3 ns;
    float2 uv;
};

struct sample_li_out {
    float3 L;
    float pdf;
    float3 wi;
    bool valid;
    float3 x;
    bool isDelta;
    float3 ns;
    int lightID;
    __init() { valid = false; }
};

struct sample_li_pdf_in {
    float3 ref_point;
    float3 ref_normal;
    float3 light_point;
    float3 light_normal;
    int lightID; 
};
}

interface ILight {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : ILightParameter;

};

#endif // _SRENDERER_LIGHT_HEADER_