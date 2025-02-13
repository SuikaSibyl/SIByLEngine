#ifndef _SRENDERER_SHAPES_TRIANGLE_HEADER_
#define _SRENDERER_SHAPES_TRIANGLE_HEADER_

#include "shape.hlsli"
#include "srenderer/scene-binding.hlsli"

struct TriangleParameter : IShapeParameter {
    float3 v0; float3 n0;
    float3 v1; float3 n1;
    float3 v2; float3 n2;
    float4x4 o2w; float4x4 o2wn;
    
    __init() {
        v0 = float3(0, 0, 0);
        v1 = float3(0, 0, 0);
        v2 = float3(0, 0, 0);
        n0 = float3(0, 0, 0);
        n1 = float3(0, 0, 0);
        n2 = float3(0, 0, 0);
        o2w = float4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
        o2wn = float4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }
    __init(LightPacket light) {
        // sample a triangle in the geometry
        const uint primitiveID = light.uintscalar_0;
        const uint geometryID = light.uintscalar_1;
        const GeometryData geometry = GPUScene_geometry[geometryID];
        // Get the indices of the vertices of the triangle
        const int3 index = int3(fetchTriangleIndices(geometry, int(primitiveID)));
        // Get the vertices / normals of the triangle
        v0 = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
        v1 = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
        v2 = fetchVertexPosition(index[2] + int(geometry.vertexOffset));
        n0 = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
        n1 = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
        n2 = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
        // Get the transforms
        o2w = ObjectToWorld(geometry);
        o2wn = ObjectToWorldNormal(geometry);
    }
};

struct Triangle : IShape {
    typedef TriangleParameter TParam;

    static ishape::sample sample(
        ishape::sample_in i, TriangleParameter param) {
        // Sample on the triangle
        const float3 e1 = param.v1 - param.v0;
        const float3 e2 = param.v2 - param.v0;
        const float a = sqrt(clamp(i.uv[0], 0., 1.));
        const float b1 = 1 - a;
        const float b2 = a * i.uv[1];
        const float3 v0_ws = mul(float4(param.v0, 1), param.o2w).xyz;
        const float3 v1_ws = mul(float4(param.v1, 1), param.o2w).xyz;
        const float3 v2_ws = mul(float4(param.v2, 1), param.o2w).xyz;
        const float3 e1_ws = v1_ws - v0_ws;
        const float3 e2_ws = v2_ws - v0_ws;
        float3 geometric_normal = normalize(cross(e1, e2));
        const float area = length(cross(e1_ws, e2_ws)) / 2;
        // Flip the geometric normal to the same side as the shading normal
        const float3 shading_normal = normalize(
            (1 - b1 - b2) * param.n0 + b1 * param.n1 + b2 * param.n2);
        if (dot(geometric_normal, shading_normal) < 0) {
            geometric_normal = -geometric_normal;
        }
        // fill the output structure
        ishape::sample o = {};
        o.position = mul(float4(param.v0 + (e1 * b1) + (e2 * b2), 1), param.o2w).xyz;
        o.normal = -normalize(mul(float4(geometric_normal, 0.0), param.o2wn).xyz);
        const float3 direction = normalize(o.position - i.position);
        o.pdf = length_squared(o.position - i.position) 
        / (area * abs(dot(o.normal, direction)));
        return o;
    }

    static float sample_pdf(ishape::pdf_in i, TriangleParameter param) {
        // Sample on the triangle
        const float3 e1 = param.v1 - param.v0;
        const float3 e2 = param.v2 - param.v0;
        const float3 v0_ws = mul(float4(param.v0, 1), param.o2w).xyz;
        const float3 v1_ws = mul(float4(param.v1, 1), param.o2w).xyz;
        const float3 v2_ws = mul(float4(param.v2, 1), param.o2w).xyz;
        const float3 e1_ws = v1_ws - v0_ws;
        const float3 e2_ws = v2_ws - v0_ws;
        float3 geometric_normal = normalize(cross(e1, e2));
        const float area = length(cross(e1_ws, e2_ws)) / 2;
        const float3 direction = normalize(i.sample_point - i.ref_point);
        return length_squared(i.ref_point - i.sample_point) 
        / (area * abs(dot(i.sample_normal, direction)));
    }
};

#endif // _SRENDERER_SHAPES_TRIANGLE_HEADER_