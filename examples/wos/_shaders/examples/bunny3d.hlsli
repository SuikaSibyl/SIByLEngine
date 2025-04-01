#ifndef _WOS_BUNNY_EXAMPLE_HLSLI
#define _WOS_BUNNY_EXAMPLE_HLSLI

#include "../core/pde.hlsli"
#include "common/math.hlsli"

struct BunnyPDEImpl : IPDEImpl<3, 1> {
    static const float absorptionMin = 10;
    static const float absorptionMax = 100;
    static const float diffusionFreq = 0.5;
    static const float dirichletFreq = 1.5;

    // returns the dirichlet boundary value at the input point
    static float dirichlet_impl(vector<float, 3> x) {
        float k = k_pi * dirichletFreq;
        float sinAx = sin(k * x.x);
        float cosAx = cos(k * x.x);
        float sinBy = sin(2.0f * k * x.y);
        float cosBy = cos(2.0f * k * x.y);
        float sinCz = sin(3.0f * k * x.z);
        return sinAx * cosBy + (1.0f - cosAx) * (1.0f - sinBy) + sinCz * sinCz;
    }

    // returns the source value at the input point
    static float source_impl(vector<float, 3> x) {
        float3 dAlpha = float3(0, 0, 0);
        float alpha = diffusion_gradient_impl(x, dAlpha);
        float sigma = absorption_impl(x);
        float a = k_pi * dirichletFreq;
        float b = 2.0f * a;
        float c = 3.0f * a;
        float sinAx = sin(a * x[0]);
        float cosAx = cos(a * x[0]);
        float sinBy = sin(b * x[1]);
        float cosBy = cos(b * x[1]);
        float sinCz = sin(c * x[2]);
        float cosCz = cos(c * x[2]);

        float u = sinAx * cosBy + (1.0f - cosAx) * (1.0f - sinBy) + sinCz * sinCz;
        float3 du = float3(0, 0, 0);
        du[0] = (cosAx * cosBy + sinAx * (1.0f - sinBy)) * a;
        du[1] = -(sinAx * sinBy + (1.0f - cosAx) * cosBy) * b;
        float d2udx2 = (cosAx * (1.0f - sinBy) - sinAx * cosBy) * a * a;
        float d2udy2 = ((1.0f - cosAx) * sinBy - sinAx * cosBy) * b * b;
        float d2u = d2udx2 + d2udy2;
        du[2] = 2.0f * sinCz * cosCz * c;
        float d2udz2 = 2.0f * (cosCz * cosCz - sinCz * sinCz) * c * c;
        d2u += d2udz2;
        return -alpha * d2u - dot(dAlpha, du) + sigma * u;
    }

    static float diffusion_impl(
        vector<float, 3> x,
        bool computeGradient, bool computeLaplacian, 
        out vector<float, 3> gradient, out float laplacian
    ) {
        float a = 4.0f * k_pi * diffusionFreq;
        float b = 3.0f * k_pi*diffusionFreq;
        float sinAx = sin(a * x.x);
        float cosAx = cos(a * x.x);
        float sinBy = sin(b * x.y);
        float cosBy = cos(b * x.y);
        float alpha = exp(-x.y * x.y + cosAx * sinBy);

        if (computeGradient || computeLaplacian) {
            gradient.x = alpha * (-sinAx * sinBy * a);
            gradient.y = alpha * (-2.0f * x.y + cosAx * cosBy * b);
        }

        if (computeLaplacian) {
            float d2Alphadx2 = gradient.x * (-sinAx * sinBy * a) + alpha * (-cosAx * sinBy * a * a);
            float d2Alphady2 = gradient.y * (-2.0f * x.y + cosAx * cosBy * b) + alpha * (-2.0f - cosAx * sinBy * b * b);
            laplacian = d2Alphadx2 + d2Alphady2;
        }

        return alpha;
    }

    // returns the absorption coefficient value at the input point
    static float absorption_impl(vector<float, 3> x) {
        return absorptionMin + (absorptionMax - absorptionMin) *
            (1.0f + 0.5f * sin(2.0f * k_pi * x.x) * cos(0.5f * k_pi * x.y));
    }
    
    static float diffusion_gradient_impl(vector<float, 3> x, out vector<float, 3> gradient) {
        float laplacianStub;
        return diffusion_impl(x, true, false, gradient, laplacianStub);
    }
}

struct BunnyBoundary : IBoundary<3> {
    static float distance(float3 x) {
        return domain_bunny(x);
    }
}

float4 transpose_mul(float4x4 mat, float4 vec) { return mul(vec, mat); }

float domain_bunny(float3 p) {
    float3 op = p;
    p = op.zxy * 0.75;

    // sdf is undefined outside the unit sphere,
    // uncomment to witness the abominations
    if (length(p) > 1.) {
        return -length(p) / 0.75f;
    }
    // neural networks can be really compact... when they want to be
    float4 f00 = sin(p.y * float4(-3.02f, 1.95f, -3.42f, -.60f) + p.z * float4(3.08f, .85f, -2.25f, -.24f) - p.x * float4(-.29f, 1.16f, -3.74f, 2.89f) + float4(-.71f, 4.50f, -3.24f, -3.50f));
    float4 f01 = sin(p.y * float4(-.40f, -3.61f, 3.23f, -.14f) + p.z * float4(-.36f, 3.64f, -3.91f, 2.66f) - p.x * float4(2.90f, -.54f, -2.75f, 2.71f) + float4(7.02f, -5.41f, -1.12f, -7.41f));
    float4 f02 = sin(p.y * float4(-1.77f, -1.28f, -4.29f, -3.20f) + p.z * float4(-3.49f, -2.81f, -.64f, 2.79f) - p.x * float4(3.15f, 2.14f, -3.85f, 1.83f) + float4(-2.07f, 4.49f, 5.33f, -2.17f));
    float4 f03 = sin(p.y * float4(-.49f, .68f, 3.05f, .42f) + p.z * float4(-2.87f, .78f, 3.78f, -3.41f) - p.x * float4(-2.65f, .33f, .07f, -.64f) + float4(-3.24f, -5.90f, 1.14f, -4.71f));
    float4 f10 = sin(transpose_mul(float4x4(-.34, .06, -.59, -.76, .10, -.19, -.12, .44, .64, -.02, -.26, .15, -.16, .21, .91, .15), f00) +
                     transpose_mul(float4x4(.01, .54, -.77, .11, .06, -.14, .43, .51, -.18, .08, .39, .20, .33, -.49, -.10, .19), f01) +
                     transpose_mul(float4x4(.27, .22, .43, .53, .18, -.17, .23, -.64, -.14, .02, -.10, .16, -.13, -.06, -.04, -.36), f02) +
                     transpose_mul(float4x4(-.13, .29, -.29, .08, 1.13, .02, -.83, .32, -.32, .04, -.31, -.16, .14, -.03, -.20, .39), f03) +
                     float4(.73f, -4.28f, -1.56f, -1.80f)) / 1.0f + f00;
    float4 f11 = sin(transpose_mul(float4x4(-1.11, .55, -.12, -1.00, .16, .15, -.30, .31, -.01, .01, .31, -.42, -.29, .38, -.04, .71), f00) +
                     transpose_mul(float4x4(.96, -.02, .86, .52, -.14, .60, .44, .43, .02, -.15, -.49, -.05, -.06, -.25, -.03, -.22), f01) +
                     transpose_mul(float4x4(.52, .44, -.05, -.11, -.56, -.10, -.61, -.40, -.04, .55, .32, -.07, -.02, .28, .26, -.49), f02) +
                     transpose_mul(float4x4(.02, -.32, .06, -.17, -.59, .00, -.24, .60, -.06, .13, -.21, -.27, -.12, -.14, .58, -.55), f03) +
                     float4(-2.24f, -3.48f, -.80f, 1.41f)) / 1.0f + f01;
    float4 f12 = sin(transpose_mul(float4x4(.44, -.06, -.79, -.46, .05, -.60, .30, .36, .35, .12, .02, .12, .40, -.26, .63, -.21), f00) +
                     transpose_mul(float4x4(-.48, .43, -.73, -.40, .11, -.01, .71, .05, -.25, .25, -.28, -.20, .32, -.02, -.84, .16), f01) +
                     transpose_mul(float4x4(.39, -.07, .90, .36, -.38, -.27, -1.86, -.39, .48, -.20, -.05, .10, -.00, -.21, .29, .63), f02) +
                     transpose_mul(float4x4(.46, -.32, .06, .09, .72, -.47, .81, .78, .90, .02, -.21, .08, -.16, .22, .32, -.13), f03) +
                     float4(3.38f, 1.20f, .84f, 1.41f)) / 1.0f + f02;
    float4 f13 = sin(transpose_mul(float4x4(-.41, -.24, -.71, -.25, -.24, -.75, -.09, .02, -.27, -.42, .02, .03, -.01, .51, -.12, -1.24), f00) +
                     transpose_mul(float4x4(.64, .31, -1.36, .61, -.34, .11, .14, .79, .22, -.16, -.29, -.70, .02, -.37, .49, .39), f01) +
                     transpose_mul(float4x4(.79, .47, .54, -.47, -1.13, -.35, -1.03, -.22, -.67, -.26, .10, .21, -.07, -.73, -.11, .72), f02) +
                     transpose_mul(float4x4(.43, -.23, .13, .09, 1.38, -.63, 1.57, -.20, .39, -.14, .42, .13, -.57, -.08, -.21, .21), f03) +
                     float4(-.34f, -3.28f, .43f, -.52f)) / 1.0f + f03;
    f00 = sin(transpose_mul(float4x4(-.72, .23, -.89, .52, .38, .19, -.16, -.88, .26, -.37, .09, .63, .29, -.72, .30, -.95), f10) +
              transpose_mul(float4x4(-.22, -.51, -.42, -.73, -.32, .00, -1.03, 1.17, -.20, -.03, -.13, -.16, -.41, .09, .36, -.84), f11) +
              transpose_mul(float4x4(-.21, .01, .33, .47, .05, .20, -.44, -1.04, .13, .12, -.13, .31, .01, -.34, .41, -.34), f12) +
              transpose_mul(float4x4(-.13, -.06, -.39, -.22, .48, .25, .24, -.97, -.34, .14, .42, -.00, -.44, .05, .09, -.95), f13) +
              float4(.48f, .87f, -.87f, -2.06f)) / 1.4f + f10;
    f01 = sin(transpose_mul(float4x4(-.27, .29, -.21, .15, .34, -.23, .85, -.09, -1.15, -.24, -.05, -.25, -.12, -.73, -.17, -.37), f10) +
              transpose_mul(float4x4(-1.11, .35, -.93, -.06, -.79, -.03, -.46, -.37, .60, -.37, -.14, .45, -.03, -.21, .02, .59), f11) +
              transpose_mul(float4x4(-.92, -.17, -.58, -.18, .58, .60, .83, -1.04, -.80, -.16, .23, -.11, .08, .16, .76, .61), f12) +
              transpose_mul(float4x4(.29, .45, .30, .39, -.91, .66, -.35, -.35, .21, .16, -.54, -.63, 1.10, -.38, .20, .15), f13) +
              float4(-1.72f, -.14f, 1.92f, 2.08f)) / 1.4f + f11;
    f02 = sin(transpose_mul(float4x4(1.00, .66, 1.30, -.51, .88, .25, -.67, .03, -.68, -.08, -.12, -.14, .46, 1.15, .38, -.10), f10) +
              transpose_mul(float4x4(.51, -.57, .41, -.09, .68, -.50, -.04, -1.01, .20, .44, -.60, .46, -.09, -.37, -1.30, .04), f11) +
              transpose_mul(float4x4(.14, .29, -.45, -.06, -.65, .33, -.37, -.95, .71, -.07, 1.00, -.60, -1.68, -.20, -.00, -.70), f12) +
              transpose_mul(float4x4(-.31, .69, .56, .13, .95, .36, .56, .59, -.63, .52, -.30, .17, 1.23, .72, .95, .75), f13) +
              float4(-.90f, -3.26f, -.44f, -3.11f)) / 1.4f + f12;
    f03 = sin(transpose_mul(float4x4(.51, -.98, -.28, .16, -.22, -.17, -1.03, .22, .70, -.15, .12, .43, .78, .67, -.85, -.25), f10) +
              transpose_mul(float4x4(.81, .60, -.89, .61, -1.03, -.33, .60, -.11, -.06, .01, -.02, -.44, .73, .69, 1.02, .62), f11) +
              transpose_mul(float4x4(-.10, .52, .80, -.65, .40, -.75, .47, 1.56, .03, .05, .08, .31, -.03, .22, -1.63, .07), f12) +
              transpose_mul(float4x4(-.18, -.07, -1.22, .48, -.01, .56, .07, .15, .24, .25, -.09, -.54, .23, -.08, .20, .36), f13) +
              float4(-1.11f, -4.28f, 1.02f, -.23f)) / 1.4f + f13;
    return -(dot(f00, float4(.09f, .12f, -.07f, -.03f)) + dot(f01, float4(-.04f, .07f, -.08f, .05f)) +
             dot(f02, float4(-.01f, .06f, -.02f, .07f)) + dot(f03, float4(-.05f, .07f, .03f, .04f)) - 0.16f) / 0.75f;
}

#endif // _WOS_BUNNY_EXAMPLE_HLSLI