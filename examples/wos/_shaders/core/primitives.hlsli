#ifndef _SRENDERER_WOS_PRIMITIVE_2D_HLSLI_
#define _SRENDERER_WOS_PRIMITIVE_2D_HLSLI_

#include "common/math.hlsli"
#include "common/sampling.hlsli"

float arg(float2 z) { return atan2(z.y, z.x); }
float2 rotate90(float2 u) { return float2(-u.y, u.x); }
float cross_2d(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
float2 complex_divide(float2 a, float2 b) { return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }

struct Polyline<let N : int> {
    float2 points[N];
    __init(Array<float2, N> initial) {
        for (int i = 0; i < N; ++i) {
            points[i] = initial[i];
        }
    }
}

// returns the closest point to x on a segment with endpoints a and b
float2 closet_point_to_line(float2 x, float2 a, float2 b) {
    const float2 u = b - a;
    const float t = clamp(dot(x - a, u) / dot(u, u), 0, 1);
    return (1. - t) * a + t * b;
}

float distance_polyline<let N : int>(float2 x, Polyline<N> polyline) {
    float d = k_inf; // minimum distance so far
    for (int i = 0; i < N - 1; ++i) {
        // distance to segment
        const float2 y = closet_point_to_line(x,
            polyline.points[i], polyline.points[i + 1]);
        d = min(d, length(x - y)); // update minimum distance
    }
    return d;
}

float2 sample_polyline<let N : int>(float u, Polyline<N> polyline, out float2 normal) {
    Array<float, N-1> pmf;
    for (int i = 0; i < N - 1; ++i) {
        pmf[i] = length(polyline.points[i + 1] - polyline.points[i]);
    }
    int i = sample_discrete(pmf, u);
    normal = normalize(rotate90(polyline.points[i + 1] - polyline.points[i]));
    return lerp(polyline.points[i], polyline.points[i + 1], u);
}

float2 closest_point_polyline<let N : int>(float2 x, Polyline<N> polyline) {
    float d = k_inf; // minimum distance so far
    float2 closest = float2(k_inf, k_inf);
    for (int i = 0; i < N - 1; ++i) {
        // distance to segment
        const float2 y = closet_point_to_line(x,
            polyline.points[i], polyline.points[i + 1]);
        const float dist = length(x - y);
        if (dist < d) {
            d = dist;
            closest = y;
        }
    }
    return closest;
}

float length_polyline<let N : int>(Polyline<N> polyline) {
    float l = 0; // minimum distance so far
    for (int i = 0; i < N - 1; ++i) {
        l += length(polyline.points[i + 1] - polyline.points[i]);
    }
    return l;
}

// returns true if the point b on the polyline abc is a silhoutte relative to x
bool is_silhouette(float2 x, float2 a, float2 b, float2 c) {
    return cross_2d(b - a, x - a) * cross_2d(c - b, x - b) < 0;
}

// returns distance from x to closest silhouette point on the given polylines P
float silhouette_distance_polyline<let N : int>(float2 x, Polyline<N> polyline) {
    float d = k_inf; // minimum distance so far
    for (int i = 1; i < N - 1; i++) { // iterate over polylines
        if (is_silhouette(x, polyline.points[i - 1],
                          polyline.points[i], polyline.points[i + 1])) {
            d = min(d, length(x - polyline.points[i])); // update minimum distance
        }
    }
    return d;
}

// these routines are not used by WoSt itself, but are rather used to check
// whether a given evaluation point is actually inside the domain
float signed_angle_polyline<let N : int>(float2 x, Polyline<N> polyline) {
    float Theta = 0.;
    // iterate over polylinese
    for (int i = 0; i < N - 1; i++)
        Theta += arg(complex_divide(polyline.points[i + 1] - x, polyline.points[i] - x));
    return Theta;
}

// returns the time t at which the ray x+tv intersects segment ab,
// or infinity if there is no intersection
float ray_intersection(float2 x, float2 v, float2 a, float2 b) {
    float2 u = b - a;
    float2 w = x - a;
    float d = cross_2d(v, u);
    float s = cross_2d(v, w) / d;
    float t = cross_2d(u, w) / d;
    if (t > 0. && 0. <= s && s <= 1.) { return t;}
    return k_inf;
}

struct Intersection2D {
    float2 point;
    float2 normal;
    bool on_boundary;

    __init() {
        point = float2(k_inf, k_inf);
        normal = float2(0, 0);
        on_boundary = false;
    }
};

// finds the first intersection y of the ray x+tv with the given polylines P,
// restricted to a ball of radius r around x. The flag onBoundary indicates
// whether the first hit is on a boundary segment (rather than the sphere), and
// if so sets n to the normal at the hit point.
Intersection2D intersect_polyline<let N : int>(float2 x, float2 v, float r, Polyline<N> P) {
    float tMin = r;              // smallest hit time so far
    float2 n = float2(0.0, 0.0); // first hit normal
    bool onBoundary = false;     // will be true only if the first hit is on a segment
    float2 result = float2(k_inf, k_inf);
    for (int i = 0; i < N - 1; ++i) {
        const float c = 1e-5; // ray offset (to avoid self-intersection)
        float t = ray_intersection(x + c * v, v, P.points[i], P.points[i + 1]);
        if (t < tMin) { // closest hit so far
            tMin = t;
            n = rotate90(P.points[i + 1] - P.points[i]); // get normal
            n /= length(n);                              // make normal unit length
            onBoundary = true;
        }
    }
    Intersection2D i2d;
    i2d.point = x + tMin * v;     // first hit location
    i2d.normal = n;               // first hit normal
    i2d.on_boundary = onBoundary; // first hit is on a boundary
    return i2d;
}

// only for visualization purposes
float checkerboard_pattern(float2 x) {
    float Size = 20.0;
    x = select(x < 0, x + 1, x);
    float2 Pos = floor(x * Size);
    float PatternMask = fmod(Pos.x + fmod(Pos.y, 2.0), 2.0);
    return lerp(0.5, 0.7, PatternMask);
}

struct nee_output {
    float2 point;
    float inv_pdf;
    float2 point_oppo;
    float inv_pdf_oppo;
};

nee_output nee_line(float2 x, float r, float2 point_a, float2 point_b, float eps, float2 uv) {
    // first, we use a local frame that map point a to the origin and point b to (1, 0)
    float2 u = point_b - point_a;
    float2 v = u / length_squared(u);
    float2x2 A = float2x2(v, rotate90(v));
    float2 b = -mul(A, point_a);
    point_a = float2(0, 0);
    point_b = float2(1, 0);
    x = mul(A, x) + b;
    const float r_new = length(mul(A, float2(r, 0)));
    eps = length(mul(A, float2(eps, 0)));
    bool flip = x.y < 0.0f; x.y = abs(x.y);

    // now we can compute the distance to the line
    float x_dist = max(max(0.0 - x.x, 0), max(x.x - 1.0, 0));
    float sin_a = x_dist / r_new; float cos_a = sqrt(1 - sin_a * sin_a);
    float a = asin(sin_a);
    float y_dist = r_new * cos(a);
    float theta = 0.f;
    float theta_min = 0.f;
    float theta_max = 0.f;
    // if the point is inside the band of the line
    if (x_dist == 0) {
        float line_theta = acos(((r_new - eps)) / r_new);
        float required_offset = sin(line_theta) * r_new;
        float offset_left = x.x - 0.0;
        float offset_right = 1.0 - x.x;
        // if the point is inside the left boundary of the sphere
        if (offset_left < required_offset) {
            float edge2 = r_new * r_new + offset_left * offset_left;
            float theta_left_2 = safe_acos((edge2 + r_new * r_new - eps * eps) 
                / (2 * r_new * sqrt(edge2)));
            float theta_left_1 = atan2(offset_left, r_new);
            theta += theta_left_1 + theta_left_2;
            theta_min = -k_pi / 2 - theta_left_1 - theta_left_2;
        } else {
            theta += line_theta;
            theta_min = -k_pi / 2 - line_theta;
        }

        if (offset_right < required_offset) {
            float edge2 = r_new * r_new + offset_right * offset_right;
            float theta_left_2 = safe_acos((edge2 + r_new * r_new - eps * eps) / (2 * r_new * sqrt(edge2)));
            float theta_left_1 = atan2(offset_right, r_new);
            theta += theta_left_1 + theta_left_2;
            theta_max = -k_pi / 2 + theta_left_1 + theta_left_2;
        } else {
            theta += line_theta;
            theta_max = -k_pi / 2 + line_theta;
        }
    }
    else if (y_dist >= eps) {
        theta = (acos((y_dist - eps) / r_new) - a)
            + 2 * asin(eps / (2 * r_new));
        if (x.x < 0.5) {
            float theta_center = atan2(-x.y, -x.x);
            theta_min = theta_center - (2 * asin(eps / (2 * r_new)));
            theta_max = theta_center + (acos((y_dist - eps) / r_new) - a);
        } else {
            float theta_center = atan2(0.0 - x.y, 1.0 - x.x);
            theta_min = theta_center - (acos((y_dist - eps) / r_new) - a);
            theta_max = theta_center + (2 * asin(eps / (2 * r_new)));
        }
    }
    else {
        theta = 4 * asin(eps / (2 * r_new));
        if (x.x < 0.5) {
            float theta_center = atan2(-x.y, -x.x);
            theta_min = theta_center - theta / 2;
            theta_max = theta_center + theta / 2;
        } else {
            float theta_center = atan2(0.0 - x.y, 1.0 - x.x);
            theta_min = theta_center - theta / 2;
            theta_max = theta_center + theta / 2;
        }
    }

    // sample a angle between theta_min and theta_max
    float theta_sample = lerp(theta_min, theta_max, uv.x);
    float2 new_x = x + float2(cos(theta_sample), sin(theta_sample)) * r_new;
    if (flip) new_x.y = -new_x.y;
    float2x2 A_inv = transpose(float2x2(u, rotate90(u)));
    new_x = mul(A_inv, (new_x - b));

    // sample another angle not in theta_min and theta_max
    if (theta_min < 0) {
        theta_min += k_2pi;
        theta_max += k_2pi;
    }
    if (theta_max > k_2pi) {
        theta_max -= k_2pi;
        theta_min -= k_2pi;
    }
    float arc_0_min = theta_min;
    float arc_max_1 = k_2pi - theta_max;
    int arc_sample = (uv.y < arc_0_min / (arc_0_min + arc_max_1)) ? 0 : 1;
    float theta_sample_2;
    if (arc_sample == 0) {
        theta_sample_2 = lerp(0, theta_min, uv.y * (arc_0_min + arc_max_1) / arc_0_min);
    } else {
        theta_sample_2 = lerp(theta_max, k_2pi, (uv.y - arc_0_min / (arc_0_min + arc_max_1)) * (arc_0_min + arc_max_1) / arc_max_1);
    }
    float2 new_x_oppo = x + float2(cos(theta_sample_2), sin(theta_sample_2)) * r_new;
    if (flip) new_x_oppo.y = -new_x_oppo.y;
    new_x_oppo = mul(A_inv, (new_x_oppo - b));

    nee_output output;
    output.point = new_x;
    output.inv_pdf = theta / (2 * k_pi);
    output.point_oppo = new_x_oppo;
    output.inv_pdf_oppo = (k_2pi - theta) / (2 * k_pi);
    return output;
}

#endif // _SRENDERER_WOS_PRIMITIVE_2D_HLSLI_