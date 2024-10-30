#ifndef _SRENDERER_SHAPE_RECTANGLE_HEADER_
#define _SRENDERER_SHAPE_RECTANGLE_HEADER_

#include "common/rotation.hlsli"
#include "sampling/warps.hlsli"
#include "shape.hlsli"

struct SphericalRectangle {
    float3 o, x, y, z;     // local reference system 'R'
    float z0, z0sq;        //
    float x0, y0, y0sq;    // rectangle coords in 'R'
    float x1, y1, y1sq;    //
    float b0, b1, b0sq, k; // misc precomputed constants
    float S;               // solid angle of 'Q'

    __init(float3 s, float3 ex, float3 ey, float3 o) {
        this.o = o;
        float exl = length(ex);
        float eyl = length(ey);
        // compute local reference system ’R’
        this.x = ex / exl;
        this.y = ey / eyl;
        this.z = cross(this.x, this.y);
        // compute rectangle coords in local reference system
        float3 d = s - o;
        this.z0 = dot(d, this.z);
        // flip ’z’ to make it point against ’Q’
        if (this.z0 > 0.) {
            this.z *= -1.;
            this.z0 *= -1.;
        }
        this.z0sq = this.z0 * this.z0;
        this.x0 = dot(d, this.x);
        this.y0 = dot(d, this.y);
        this.x1 = this.x0 + exl;
        this.y1 = this.y0 + eyl;
        this.y0sq = this.y0 * this.y0;
        this.y1sq = this.y1 * this.y1;
        // create vectors to four vertices
        float3 v00 = float3(this.x0, this.y0, this.z0);
        float3 v01 = float3(this.x0, this.y1, this.z0);
        float3 v10 = float3(this.x1, this.y0, this.z0);
        float3 v11 = float3(this.x1, this.y1, this.z0);
        // compute normals to edges
        float3 n0 = normalize(cross(v00, v10));
        float3 n1 = normalize(cross(v10, v11));
        float3 n2 = normalize(cross(v11, v01));
        float3 n3 = normalize(cross(v01, v00));
        // compute internal angles (gamma_i)
        float g0 = acos(-dot(n0, n1));
        float g1 = acos(-dot(n1, n2));
        float g2 = acos(-dot(n2, n3));
        float g3 = acos(-dot(n3, n0));
        // compute predefined constants
        this.b0 = n0.z;
        this.b1 = n2.z;
        this.b0sq = this.b0 * this.b0;
        this.k = 2. * k_pi - g2 - g3;
        // compute solid angle from internal angles
        this.S = g0 + g1 - this.k;
    }

    ishape::sample sample(ishape::sample_in i) {
        ishape::sample sample_o;
        // 1. compute ’cu’
        float au = i.uv.x * S + k;
        float fu = (cos(au) * b0 - b1) / sin(au);
        float cu = 1 / sqrt(fu * fu + b0sq) * (fu > 0 ? +1 : -1);
        cu = clamp(cu, -1, 1); // avoid NaNs
        // 2. compute ’xu’
        float xu = -(cu * z0) / sqrt(1 - cu * cu);
        xu = clamp(xu, x0, x1); // avoid Infs
        // 3. compute ’yv’
        float d = sqrt(xu * xu + z0sq);
        float h0 = y0 / sqrt(d * d + y0sq);
        float h1 = y1 / sqrt(d * d + y1sq);
        float hv = h0 + i.uv.y * (h1 - h0);
        float hv2 = hv * hv;
        float yv = (hv2 < 1 - 1e-4) ? (hv * d) / sqrt(1 - hv2) : y1;
        // 4. transform (xu,yv,z0) to world coords
        sample_o.position = (o + xu * x + yv * y + z0 * z);
        sample_o.pdf = 1.f / S;
        return sample_o;
    }
};

struct RectangleParameter {
    float4x4 o2w;
    float4x4 w2o;
};

struct Rectangle {
    typedef RectangleParameter TParam;

    // from iq:
    // https://iquilezles.org/articles/boxfunctions/
    static float hit(Ray ray, RectangleParameter param) {
        // convert from world to box space
        float3 rd = mul(float4(ray.direction, 0.0), param.w2o).xyz;
        float3 ro = mul(float4(ray.origin, 1.0), param.w2o).xyz;
        float t = -ro.z / rd.z;
        float3 local = ro + t * rd;
        if (abs(local.x) <= 1.f && abs(local.y) <= 1.f) return t;
        return -1.f;
    }

    static ishape::sample sample(ishape::sample_in i, RectangleParameter param) {
        float3 local_position = float3(i.uv * 2 - 1, 0);
        float3 world_position = mul(float4(local_position, 1.0), param.o2w).xyz;
        float3 world_0 = mul(float4(0, 0, 0, 1), param.o2w).xyz;
        float3 world_x = mul(float4(1, 0, 0, 1), param.o2w).xyz;
        float3 world_y = mul(float4(0, 1, 0, 1), param.o2w).xyz;
        float3 world_z = mul(float4(0, 0, 1, 1), param.o2w).xyz;
        float edge_1 = length(world_x - world_0);
        float edge_2 = length(world_y - world_0);
        float area = edge_1 * edge_2 * 4;
        float3 direction = normalize(world_position - i.position);

        ishape::sample sample_o;
        sample_o.position = world_position;
        sample_o.normal = normalize(world_z - world_0);
        sample_o.pdf = length_squared(i.position - sample_o.position) 
        / (area * abs(dot(sample_o.normal, -direction)));

        return sample_o;
    }

    static float sample_pdf(ishape::pdf_in i, RectangleParameter param) {
        float3 world_0 = mul(float4(0, 0, 0, 1), param.o2w).xyz;
        float3 world_x = mul(float4(1, 0, 0, 1), param.o2w).xyz;
        float3 world_y = mul(float4(0, 1, 0, 1), param.o2w).xyz;
        float3 world_z = mul(float4(0, 0, 1, 1), param.o2w).xyz;
        float edge_1 = length(world_x - world_0);
        float edge_2 = length(world_y - world_0);
        float area = edge_1 * edge_2 * 4;
        float3 direction = normalize(i.sample_point - i.ref_point);
        
        return length_squared(i.sample_point - i.ref_point) 
        / (area * abs(dot(i.sample_normal, -direction)));
    }
};

GeometryHit fetchRectangleGeometryHit(GeometryData geometry, Ray ray, float t) {
    // convert from world to box space
    float4x4 o2w = ObjectToWorld(geometry);
    float4x4 w2o = WorldToObject(geometry);
    float3 rd = mul(float4(ray.direction, 0.0), w2o).xyz;
    float3 ro = mul(float4(ray.origin, 1.0), w2o).xyz;
    float t = -ro.z / rd.z;
    GeometryHit hit;
    hit.shadingNormal = mul(float4(0, 0, 1, 0), transpose(w2o)).xyz;
    hit.shadingNormal = normalize(hit.shadingNormal);
    hit.texcoord = (ro + t * rd).xy * 0.5 + 0.5;
    hit.position = ray.origin + t * ray.direction;
    hit.geometryNormal = hit.shadingNormal;
    hit.barycentric = float2(0, 0);
    hit.tangent = float4(0);
    hit.barycentric = float2(0.333);
    
    if (ro.z > 0) SetFaceForward(hit, true);
    else {
        SetFaceForward(hit, false);
        hit.shadingNormal = -hit.shadingNormal;
        hit.geometryNormal = -hit.geometryNormal;
    }

    return hit;
}

// struct Rectangle : IShape {
//     float3 p;   // The left-bottom corner
//     float3 n;   // The normal of the rectangle
//     float3 u;   // The first axis of the rectangle
//     float3 v;   // The second axis of the rectangle

//     __init(float3 p, float3 n, float3 u, float3 v) {
//         this.p = p;
//         this.n = n;
//         this.u = u;
//         this.v = v;
//     }

//     __init(float3 position, float3 euler,
//            float width, float height) {
//         float4x4 rot = rotate_euler(euler);
//         float3 u = mul(transpose(rot), float4(1, 0, 0, 0)).xyz * width;
//         float3 v = mul(transpose(rot), float4(0, 1, 0, 0)).xyz * height;
//         float3 n = mul(transpose(rot), float4(0, 0, 1, 0)).xyz;
//         return Rectangle(
//             position - 0.5 * u - 0.5 * v,
//             n, u, v);
//     }

//     bool intersect(Ray ray, inout PrimaryPayload payload) {
//         SetHit(payload.hit, false);
//         const float3 pro = ray.origin - p;
//         const float nrd = -dot(ray.direction, n);
//         const float t = dot(pro, n) / nrd;
//         if (t < ray.tMin || t > ray.tMax) return false;

//         const float3 hitpoint = ray.origin + t * ray.direction;
//         const float3 hpdir = hitpoint - p;

//         const float minU = 0.;
//         const float maxU = 1.;

//         float _u = dot(u, hpdir) / dot(u, u);
//         if (_u < minU || _u > maxU) return false;
//         float _v = dot(v, hpdir) / dot(v, v);
//         if (_v < minU || _v > maxU) return false;

//         SetHit(payload.hit, true);
//         const float3 p = ray.origin + t * ray.direction;
//         // vec3 p = plane.p + u * plane.u + v * plane.v;
//         payload.hit.position = p;
//         payload.hit.shadingNormal = (nrd < 0.) ? -n : n;
//         payload.hit.geometryNormal = (nrd < 0.) ? -n : n;
//         payload.hit.texcoord = float2(_u, _v);
//         return true;
//     }
    
//     ishape::sample sample(ishape::sample_in i) {
//         ishape::sample o;
//         const uint32_t method = i.flag;
//         switch(method) {
//             case 0: {
//                 // Uniform Area Sampling
//                 // Sample a point on the rectangle
//                 o.position = p + i.uv.x * u + i.uv.y * v;
//                 o.normal = n;
//                 o.pdf = ishape::inv_geometry_term(
//                     o.position, o.normal, i.position) 
//                     / length(cross(u, v));
//                 break;
//             }
//             case 1: {
//                 // Uniform Solid Angle Sampling
//                 let sr = SphericalRectangle(p, u, v, i.position);
//                 o = sr.sample(i);
//                 o.normal = n;
//                 break;
//             }
//             case 2: {
//                 // Bilinear Cosine Warp X Uniform Solid Angle Sampling
//                 float3 wp[4];
//                 wp[0] = p;
//                 wp[1] = p + u;
//                 wp[2] = p + v;
//                 wp[3] = p + u + v;
//                 float4 prob;
//                 for (int j = 0; j < 4; j++) {
//                     float3 wpd = wp[j] - i.position;
//                     float3 wpn = normalize(wpd);
//                     // cosine term of ray with illuminated surface
//                     prob[j] = max(0., dot(wpn, i.normal));
//                 }
//                 let bw = BilinearWarp(prob[0], prob[2], prob[1], prob[3]);
//                 iwarp::warp2d_out wo = bw.inverse(i.uv);
//                 let sr = SphericalRectangle(p, u, v, i.position);
//                 i.uv = wo.sample;
//                 o = sr.sample(i);
//                 o.pdf = o.pdf * wo.pdf;
//                 o.normal = n;
//                 break;
//             }
//         }
//         return o;
//     }

//     float3[4] corners() {
//         float3 positions[4];
//         positions[0] = p;
//         positions[1] = p + u;
//         positions[2] = p + u + v;
//         positions[3] = p + v;
//         return positions;
//     }
// };

#endif // _SRENDERER_SHAPE_RECTANGLE_HEADER_