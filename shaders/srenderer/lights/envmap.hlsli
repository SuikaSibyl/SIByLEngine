#ifndef _SRENDERER_LIGHTS_ENVMAP_HLSLI_
#define _SRENDERER_LIGHTS_ENVMAP_HLSLI_

#include "light.hlsli"
#include "common/math.hlsli"
#include "sampling/pmf.hlsli"

struct EnvmapParameter : ILightParameter {
    uint32_t width;
    uint32_t height;
    uint32_t condition_offset;
    uint32_t marginal_offset;
    float sceneRadius;
    Texture2D<float4> envmap;
    // condition_size = width + 1
    // marginal_size = height + 1
};

struct EnvmapLight : ILight {
    typedef EnvmapParameter TParam;

    static float3 eval_le(ilight::eval_le_in i, EnvmapParameter param) {
        float3 wLight = normalize(i.dir);
        float2 uv = equal_area_sphere_to_square(wLight);
        int2 pixel = int2(uv * float2(param.width, param.height));
        return param.envmap[pixel].xyz;
    }

    static ilight::sample_li_out sample_li(ilight::sample_li_in i, EnvmapParameter param) {
        // Find (u,v) sample coordinates in infinite light texture
        piecewise_sample::sample_out_2d sample = sample_pmf_2d(
            i.uv, param.marginal_offset, param.height + 1,
            param.condition_offset, param.width + 1);
        
        // Convert infinite light sample point to direction
        float3 wLight = equal_area_square_to_sphere(sample.x);
        float3 wi = wLight;
        // Compute PDF for sampled infinite light direction
        float pdf = sample.pdf / (4 * k_pi);
        // Return radiance value for infinite light direction
        ilight::sample_li_out o;
        o.wi = wi; o.pdf = pdf;
        int2 pixel = int2(sample.x * float2(param.width, param.height));
        o.L = param.envmap[pixel].xyz;
        o.x = i.p + wi * (2 * param.sceneRadius);
        return o;
    }

    static float3 equal_area_square_to_sphere(float2 p) {
        // Transform p to and compute absolute values
        float u = 2 * p.x - 1; float v = 2 * p.y - 1;
        float up = abs(u); float vp = abs(v);
        // Compute radius r as signed distance from diagonal
        float signedDistance = 1 - (up + vp);
        float d = abs(signedDistance);
        float r = 1 - d;
        // Compute angle  for square to sphere mapping
        float phi = (r == 0 ? 1 : (vp - up) / r + 1) * k_pi / 4;
        // Find z coordinate for spherical direction
        float z = copysign(1 - sqr(r), signedDistance);
        // Compute  and  for original quadrant and return vector
        float cosPhi = copysign(cos(phi), u);
        float sinPhi = copysign(sin(phi), v);
        return float3(cosPhi * r * safe_sqrt(2 - sqr(r)), z,
                      sinPhi * r * safe_sqrt(2 - sqr(r)));
    }

    // Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
    static float2 equal_area_sphere_to_square(float3 d) {
        d = float3(d.x, d.z, d.y);
        float x = abs(d.x);
        float y = abs(d.y); 
        float z = abs(d.z);
        // Compute the radius r
        float r = safe_sqrt(1 - z); // r = sqrt(1-|z|)
        // Compute the argument to atan (detect a=0 to avoid div-by-zero)
        float a = max(x, y);
        float b = min(x, y);
        b = a == 0 ? 0 : b / a;
        // Polynomial approximation of atan(x)*2/pi, x=b
        // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
        // x=[0,1].
        const float t1 = 0.406758566246788489601959989e-5;
        const float t2 = 0.636226545274016134946890922156;
        const float t3 = 0.61572017898280213493197203466e-2;
        const float t4 = -0.247333733281268944196501420480;
        const float t5 = 0.881770664775316294736387951347e-1;
        const float t6 = 0.419038818029165735901852432784e-1;
        const float t7 = -0.251390972343483509333252996350e-1;
        float phi = evaluate_polynomial_6(b, t1, t2, t3, t4, t5, t6, t7);
        // Extend phi if the input is in the range 45-90 degrees (u<v)
        if (x < y) phi = 1 - phi;
        // Find (u,v) based on (r,phi)
        float v = phi * r;
        float u = r - v;
        if (d.z < 0) {
            // southern hemisphere -> mirror u,v
            swap(u, v);
            u = 1 - u;
            v = 1 - v;
        }
        // Move (u,v) to the correct quadrant based on the signs of (x,y)
        u = copysign(u, d.x);
        v = copysign(v, d.y);
        // Transform (u,v) from [-1,1] to [0,1]
        return float2(0.5f * (u + 1), 0.5f * (v + 1));
    }
};

#endif // _SRENDERER_LIGHTS_ENVMAP_HLSLI_