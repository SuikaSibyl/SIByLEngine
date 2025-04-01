#ifndef _SRENDERER_COMMON_MAPPING_HEADER_
#define _SRENDERER_COMMON_MAPPING_HEADER_

/*
 * Includes various mapping between various domains.
 * A slang version of repo "mappings-sphere-disc-square"
 * @url: https://github.com/marlam/mappings-sphere-disc-square/blob/master/disk-square-common.hpp
 * which is the code for paper "Mappings between Sphere, Disc, and Square"
 * @url: https://jcgt.org/published/0005/02/01/paper.pdf 
 */

#include "math.hlsli"
namespace mapping {

/**
 * Radius Stretching
 * ✘ equal-area
 * ✘ conformal
 */
float2 square_to_disk_stretching(float2 p) {
    const float t = hypot(p);
    float u; float v;
    if (t <= 0.0) {
        u = 0.0;
        v = 0.0;
    } else {
        const float c = (p.x * p.x >= p.y * p.y ? p.x : p.y);
        u = sign(p.x) * c * p.x / t;
        v = sign(p.y) * c * p.y / t;
    }
    return float2(u, v);
}

float2 disk_to_square_stretching(float2 p) {
    const float u = p.x;
    const float v = p.y;
    const float r = hypot(p);
    float x; float y;
    if (r <= 0.0) {
        x = 0.0;
        y = 0.0;
    } else {
        x = sign(u) * r;
        y = sign(v) * r;
        if (u * u >= v * v)
            y *= v / u;
        else
            x *= u / v;
    }
    return float2(x, y);
}

/**
 * Disk Polar
 * ✘ equal-area
 * ✘ conformal
 */
[Differentiable]
float2 square_to_disk_polar(float2 p) {
    const float r = sqrt(p.x);
    const float theta = k_2pi * p.y;
    float sin_theta; float cos_theta;
    sincos(theta, sin_theta, cos_theta);
    return float2(r * cos(theta), r * sin(theta));
}

float2 disk_to_square_polar(float2 p) {
    float phi = atan2(p.y, p.x);
    if (phi < 0) phi += k_2pi;
    return float2(sqr(p.x) + sqr(p.y), phi / k_2pi);
}

/**
 * Shirley's equal-area mapping [Shirley and Chiu 1997].
 * ✔ equal-area
 * ✘ conformal: suffers from strong angular distortions, especially
 *               around the discontinuities at the diagonals of the square.
 * Notes from "mappings-sphere-disc-square" repo:
 * -------------------------------------------------------------------------
 * Shirley's equal-area method, described in Shirley, P., & Chiu, K. (1997).
 * A low distortion map between disk and square. Journal of graphics tools,
 * 2(3), 45-52.
 *
 * Note that this is equivalent to the independently derived mapping described
 * in Roşca, D. "New uniform grids on the sphere." Astronomy & Astrophysics
 * 520 (2010): A63.
 *
 * For the square-to-disk functino, we use a trick to simplify the original
 * equations, first introduced by Dave Cline here:
 * http://psgraphics.blogspot.de/2011/01/improved-code-for-concentric-map.html
 */
float2 square_to_disk_shirley(float2 u) {
    float phi; float r;
    float a = 2 * u.x - 1;
    float b = 2 * u.y - 1;
    // use squares instead of absolute values
    if (a * a > b * b) {
        r = a;
        phi = k_pi_over_4 * (b / a);
    } else {
        r = b;
        if (abs(b) > 0.0)
            phi = k_pi_over_2 - k_pi_over_4 * (a / b);
        else phi = 0.0f;
    }
    float cos_phi; float sin_phi;
    sincos(phi, sin_phi, cos_phi);
    return float2(r * cos_phi, r * sin_phi);
}

float2 disk_to_square_shirley(float2 p) {
    const float r = length(p);
    const float atan2_vu = atan2(p.y, p.x);
    float phi = 0;
    if (atan2_vu >= -k_pi_over_4) {
        phi = atan2_vu;
    } else {
        phi = atan2_vu + k_2pi;
    }
    float x; float y;
    if (phi < k_pi_over_4) {
        x = r;
        y = r * phi * (4. / k_pi);
    } else if (phi < 3 * k_pi_over_4) {
        x = -(4. / k_pi) * r * (phi - k_pi_over_2);
        y = r;
    } else if (phi < 5 * k_pi_over_4) {
        x = -r;
        y = -(4. / k_pi) * r * (phi - k_pi);
    } else {
        x = (4. / k_pi) * r * (phi - 3 * k_pi_over_2);
        y = -r;
    }
    return float2(x, y) * 0.5 + 0.5;
}

/**
 * ✘ equal-area
 * ✘ conformal
 * Notes from "mappings-sphere-disc-square" repo:
 * -------------------------------------------------------------------------
 * Method using Guasti's Squircle: map concentric circles to concentric
 * squircles that become more square-like with growing size.
 * Details of the mapping are described in Fong, C. (2014). An Indoor
 * Alternative to Stereographic Spherical Panoramas. In Proceedings of
 * Bridges 2014: Mathematics, Music, Art, Architecture, Culture
 * (pp. 103-110).
 *
 * This mapping is neither conformal nor equal-area.
 */
float2 square_to_disk_squircle(float2 p) {
    const float x = p.x;
    const float y = p.y;
    const float x2 = x * x;
    const float y2 = y * y;
    const float t = x2 + y2;
    float u; float v;
    if (t <= 0) {
        u = v = 0;
    } else {
        const float r = sqrt(max(t - x2 * y2, 0.0));
        const float q = r / sqrt(t);
        u = x * q;
        v = y * q;
    }
    return float2(u, v);
}

float2 disk_to_square_squircle(float2 p) {
    const float u = p.x;
    const float v = p.y;
    const float u2 = u * u;
    const float v2 = v * v;
    const float w = sign(u * v) / k_sqrt2 *
                    sqrt(max(u2 + v2 - sqrt((u2 + v2) * 
                    (u2 + v2 - 4 * u2 * v2)), 0.0));
    float x; float y;
    if (abs(w) <= 0) {
        x = u;
        y = v;
    } else {
        x = w / v;
        y = w / u;
    }
    return float2(x, y);
}

/**
 * Method using elliptical arc mapping.
 * ✘ equal-area
 * ✘ conformal
 */
float2 square_to_disk_elliptical(float2 p) {
    const float x = p.x;
    const float y = p.y;
    const float u = x * sqrt(1 - y * y / 2);
    const float v = y * sqrt(1 - x * x / 2);
    return float2(u, v);
}

float2 disk_to_square_elliptical(float2 p) {
    const float u = p.x;
    const float v = p.y;
    const float x = 0.5 * sqrt(max(2 + u * u - v * v + u * 2 * k_sqrt2, 0.0))
        - 0.5 * sqrt(max(2 + u * u - v * v - u * 2 * k_sqrt2, 0.0));
    const float y = 0.5 * sqrt(max(2 - u * u + v * v + v * 2 * k_sqrt2, 0.0))
        - 0.5 * sqrt(max(2 - u * u + v * v - v * 2 * k_sqrt2, 0.0));
    return float2(x, y);
}

/**
 * Method using elliptical arc mapping.
 * ✔ equal-area
 * ✘ conformal
 */
float2 hemisphere_to_disc_lambert(float3 p) {
    const float r = k_sqrt2 * sqrt((1 - p.z) / 2);
    const float2 dir = normalize(p.xy);
    return dir * r;
}

/**
 * Method using elliptical arc mapping.
 * ✔ equal-area
 * ✘ conformal
 */
float3 disc_to_hemisphere_lambert(float2 p) {
    if(all(p == float2(0, 0))) return float3(0, 0, 1);
    const float r = length(p);
    const float z = 1 - 2 * sqr(r / k_sqrt2);
    const float2 xy = normalize(p) * sqrt(1 - z * z);
    return float3(xy, z);
}
} // end namespace mapping
#endif // !_SRENDERER_COMMON_MAPPING_HEADER_