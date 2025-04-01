#ifndef _SRENDERER_SHAPE_HEADER_
#define _SRENDERER_SHAPE_HEADER_

#include "common/math.hlsli"
#include "srenderer/spt.hlsli"

interface IShapeParameter : IDifferentiable {};

namespace ishape {
struct sample_in {
    // the position of the shading point
    float3 position;
    // bit flag, indicating:
    // - the type of sampling
    uint32_t flag;
    // the normal of the shading point
    float3 normal;
    // the random number used for sampling
    float2 uv;
};

struct sample {
    float3 position;
    float pdf;
    float3 normal;
};

struct pdf_in {
    float3 ref_point;
    float3 ref_normal;
    float3 sample_point;
    float3 sample_normal;
};

float inv_geometry_term(
    float3 sample_position,
    float3 sample_normal,
    float3 shading_position,
) {
    float3 direction = shading_position - sample_position;
    float dist = length(direction);
    direction /= dist;
    return (dist * dist) / abs(dot(sample_normal, direction));
}
}

interface IShape {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : IShapeParameter;

    // // Check intersection between a ray and the shape.
    // bool intersect(Ray ray, inout PrimaryPayload payload);
    // Sample a point on the shape.
    static ishape::sample sample(ishape::sample_in i, TParam param);
};

#endif // _SRENDERER_SHAPE_HEADER_