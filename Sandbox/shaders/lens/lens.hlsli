#ifndef _SRENDERER_LENS_HEADER_HLSLI_
#define _SRENDERER_LENS_HEADER_HLSLI_

namespace lens {
struct ray {
    float3 pos;
    float3 dir;
};

struct intersection {
    float3 pos;
    float3 norm;
    float theta;
    bool hit;
    bool inverted;
};

struct lens_interface {
    float3 center;
    float radius;
};

intersection test_flat(ray r, lens_interface F) {
    intersection i;
    i.pos = r.pos + r.dir * ((F.center.z - r.pos.z) / r.dir.z);
    i.norm = r.dir.z > 0 ? float3(0, 0, -1) : float3(0, 0, 1);
    i.theta = 0;  // meaningless
    i.hit = true;
    i.inverted = false;
    return i;
}

intersection test_sphere(ray r, lens_interface F) {
    intersection i;
    float3 D = r.pos - F.center;
    float B = dot(D, r.dir);
    float C = dot(D, D) - F.radius * F.radius;
    float B2_C = B * B - C;
    if (B2_C < 0) {
        // no intersection
        i.hit = false;
        return i;
    }
    float sgn = (F.radius * r.dir.z) > 0 ? 1 : -1;
    float t = sqrt(B2_C) * sgn-B;
    i.pos = r.dir * t + r.pos;
    i.norm = normalize(i.pos - F.center);
    if (dot(i.norm, r.dir) > 0) i.norm = -i.norm;
    i.theta = acos(dot(-r.dir, i.norm));
    i.hit = true;
    i.inverted = t < 0; // mark an inverted ray
    return i;
}

ray trace(
    int bid,        // index of current bounce / ghost
    ray r,          // input ray from the entrance plane
    float lambda,   // wavelength of r
) {
    int2 STR = {0, 1}; // start and end of interfaces
    int LEN = 2;    // length of interfaces

    // initialization
    int PHASE = 0; // ray-tracing phase
    int DELTA = 1; // delta for for-loop
    int T     = 1; // index of target interface to test

    int k;
    for (k = 0; k < LEN; k++, T += DELTA) {
        lens_interface F; // current interface
        if (T == STR.x) {
            F.center = float3(0, 0, 0);
            F.radius = 1;
        } else if (T == STR.y) {
            F.center = float3(0, 0, 2);
            F.radius = 1;
        }

        intersection i = test_sphere(r, F);
        if (!i.hit) {
            // no intersection
            return r;
        }

        // calculate the next ray
        float3 N = i.norm;
        float3 D = r.dir;
        float3 R = D - 2 * dot(D, N) * N;
        r.pos = i.pos;
        r.dir = R;

    }
}

};

#endif // _SRENDERER_LENS_HEADER_HLSLI_