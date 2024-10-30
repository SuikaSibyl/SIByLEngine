#ifndef _SRENDERER_COMMMON_MATH_HEADER_
#define _SRENDERER_COMMMON_MATH_HEADER_

static const float k_pi         = 3.1415926535897932f;
static const float k_2pi        = 6.2831853071795864f;
static const float k_pi_over_2  = k_pi / 2;
static const float k_pi_over_4  = k_pi / 4;
static const float k_inv_pi     = 1. / k_pi;
static const float k_inv_2_pi = 1. / (2 * k_pi);
static const float k_inv_4_pi = 0.07957747154594766788;

static const float k_inf        = 1.0f / 0.0f;

static const float k_numeric_limits_float_min = 1.0f / exp2(126);
static const float k_numeric_limits_float_max = 3.402823466e+38;

static const float k_one_minus_epsilon = 0.999999940395355225f;
static const float k_float_epsilon = 1.192092896e-07f;

static const float k_sqrt2 = 1.41421356237309504880f;

/** Returns the smallest component of the vector. */
float minComponent(in const float3 v) {
    return min(v.x, min(v.y, v.z)); }
/** Returns the largest component of the vector. */
float maxComponent(in const float3 v) {
    return max(v.x, max(v.y, v.z)); }
/** Returns the largest component of the vector. */
float maxComponent(in const float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w)); }

/** Returns the index of the smallest component of the vector. */
int minDimension(in const float3 v) {
    return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : 
           ((v.y < v.z) ? 1 : 2); }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float2 v) {
    return (v.x >= v.y) ? 0 : 1; }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float3 v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : 
           ((v.y > v.z) ? 1 : 2); }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float4 v) {
    int4 isMax = select(v == maxComponent(v), int4(1), int4(0));
    return dot(isMax, int4(0, 1, 2, 3)); }

// permute a float3 vector
float3 permute(in const float3 v, int x, int y, int z) {
    return float3(v[x], v[y], v[z]); }

/** Returns either -1 or 1 based on the sign of the input value.
* If the input is zero, 1 is returned. */
float signNotZero(float x) {
    return (x >= 0.f) ? 1.f : -1.f; }
/** 2-component version of signNotZero. */
float2 signNotZero(float2 v) {
    return float2(signNotZero(v.x), signNotZero(v.y)); }
/** 3-component version of signNotZero. */
float3 signNotZero(float3 v) {
    return float3(signNotZero(v.x), signNotZero(v.y), signNotZero(v.z)); }
/** 4-component version of signNotZero. */
float4 signNotZero(float4 v) {
    return float4(signNotZero(v.x), signNotZero(v.y), signNotZero(v.z), signNotZero(v.w)); }

/** Returns the luminance of the input rgb. */
float luminance(in const float3 rgb) {
    return dot(rgb, float3(0.212671, 0.715160, 0.072169));
}

float2 interpolate(float2 vertices[3], float3 bary) {
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}
float3 interpolate(float3 vertices[3], float3 bary) {
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}
float4 interpolate(float4 vertices[3], float3 bary) {
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}

[Differentiable] float length_squared(in float2 v) { return dot(v, v); }
[Differentiable] float length_squared(in float3 v) { return dot(v, v); }

float distance_squared(in float3 v0, in float3 v1) {
    return dot(v0 - v1, v0 - v1);
}

float square(in float v) {
    return v * v;
}

[Differentiable] float sqr(float v) { return v * v; }
[Differentiable] float2 sqr(float2 v) { return float2(sqr(v.x), sqr(v.y)); }
[Differentiable] float3 sqr(float3 v) { return float3(sqr(v.x), sqr(v.y), sqr(v.z)); }
[Differentiable] float4 sqr(float4 v) { return float4(sqr(v.x), sqr(v.y), sqr(v.z), sqr(v.w)); }
[Differentiable] float safe_sqrt(float v) { return sqrt(max(v, 0.f)); }
[Differentiable] float2 safe_sqrt(float2 v) { return float2(safe_sqrt(v.x), safe_sqrt(v.y)); }
[Differentiable] float3 safe_sqrt(float3 v) { return float3(safe_sqrt(v.x), safe_sqrt(v.y), safe_sqrt(v.z)); }
[Differentiable] float4 safe_sqrt(float4 v) { return float4(safe_sqrt(v.x), safe_sqrt(v.y), safe_sqrt(v.z), safe_sqrt(v.w)); }

float elevation(float3 d) {
    return 2.f * asin(.5f * sqrt(sqr(d.x) + sqr(d.y) + sqr(d.z - 1.f)));
}

uint hprod<let Dim : int>(vector<uint, Dim> v) {
    uint result = v[0];
    for (int i = 1; i < Dim; ++i)
        result = result * v[i];
    return result;
}

float hprod<let Dim : int>(vector<float, Dim> v) {
    float result = v[0];
    for (int i = 1; i < Dim; ++i)
        result = result * v[i];
    return result;
}

void swap(inout float a, inout float b) {
    float temp = a; a = b; b = temp; }
void swap(inout float2 a, inout float2 b) {
    float2 temp = a; a = b; b = temp; }
void swap(inout float3 a, inout float3 b) {
    float3 temp = a; a = b; b = temp; }
void swap(inout float4 a, inout float4 b) {
    float4 temp = a; a = b; b = temp; }

float hypot(float2 v) { return length(v); }
float hypot(float3 v) { return length(v); }
float hypot(float4 v) { return length(v); }
float hypot(float x, float y) { return length(float2(x, y)); }
float hypot(float x, float y, float z) { return length(float3(x, y, z)); }
float hypot(float x, float y, float z, float w) { return length(float4(x, y, z, w)); }

float copysign(float mag, float sign) { return (sign >= 0) ? abs(mag) : -abs(mag); }

float discard_nan_inf(float v) { return (isinf(v) || isnan(v)) ? 0 : v; }
float2 discard_nan_inf(float2 v) { return select(isinf(v) || isnan(v), 0, v); }
float3 discard_nan_inf(float3 v) { return select(isinf(v) || isnan(v), 0, v); }
float4 discard_nan_inf(float4 v) { return select(isinf(v) || isnan(v), 0, v); }

bool any_nan_inf(float v) { return isinf(v) || isnan(v); }
bool any_nan_inf(float2 v) { return any(isinf(v) || isnan(v)); }
bool any_nan_inf(float3 v) { return any(isinf(v) || isnan(v)); }
bool any_nan_inf(float4 v) { return any(isinf(v) || isnan(v)); }

bool is_nan_or_inf(float v) { return isinf(v) || isnan(v); }
bool2 is_nan_or_inf(float2 v) { return isinf(v) || isnan(v); }
bool3 is_nan_or_inf(float3 v) { return isinf(v) || isnan(v); }
bool4 is_nan_or_inf(float4 v) { return isinf(v) || isnan(v); }

/**
 * floating-point complex structure based on pbrt-v4.
 */
struct complex {
    float re;
    float im;

    __init(float re) { this.re = re; this.im = 0; }
    __init(float re, float im) { this.re = re; this.im = im; }
};
[Differentiable] complex operator+(complex a, complex b) { return complex(a.re + b.re, a.im + b.im); }
[Differentiable] complex operator+(float value, complex z) { return complex(value) + z; }
[Differentiable] complex operator-(complex a, complex b) { return complex(a.re - b.re, a.im - b.im); }
[Differentiable] complex operator-(float value, complex z) { return complex(value) - z; }
[Differentiable] complex operator*(complex a, complex b) { 
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
[Differentiable] complex operator*(float value, complex z) { return complex(value) * z; }
[Differentiable] complex operator*(complex z, float value) { return complex(value) * z; }
[Differentiable] complex operator/(complex a, complex b) {
    float scale = 1 / (b.re * b.re + b.im * b.im);
    return complex(scale * (a.re * b.re + a.im * b.im),
                   scale * (a.im * b.re - a.re * b.im)); }
complex operator /(float value, complex z) { return complex(value) / z; }
[Differentiable] float real(complex z) { return z.re; }
[Differentiable] float imag(complex z) { return z.im; }
[Differentiable] float norm(complex z) { return z.re * z.re + z.im * z.im; }
[Differentiable] float abs(complex z) { return sqrt(norm(z)); }
[Differentiable] complex sqr(complex x) { return x * x; }
[Differentiable] complex sqrt(complex z) {
    const float n = abs(z);
    const float t1 = sqrt(.5 * (n + abs(z.re)));
    const float t2 = .5 * z.im / t1;

    if (n == 0) return complex(0);
    if (z.re >= 0) return { t1, t2 };
    else return complex(abs(t2), copysign(t1, z.im));
}

struct complex3 {
    complex x, y, z;
    __init(float r) { x = complex(r); y = complex(r); z = complex(r); }
    __init(complex x, complex y, complex z) { this.x = x; this.y = y; this.z = z; }
    __init(float3 r, float3 i) { x = complex(r.x, i.x); y = complex(r.y, i.y); z = complex(r.z, i.z); }
}

[Differentiable] complex3 operator +(complex3 a, complex3 b) { return complex3(a.x + b.x, a.y + b.y, a.z + b.z); }
[Differentiable] complex3 operator -(complex3 a, complex3 b) { return complex3(a.x - b.x, a.y - b.y, a.z - b.z); }
[Differentiable] complex3 operator *(complex3 a, complex3 b) { return complex3(a.x * b.x, a.y * b.y, a.z * b.z); }
[Differentiable] complex3 operator /(complex3 a, complex3 b) { return complex3(a.x / b.x, a.y / b.y, a.z / b.z); }
[Differentiable] complex3 sqr(complex3 c) { return complex3(sqr(c.x), sqr(c.y), sqr(c.z)); }
[Differentiable] complex3 sqrt(complex3 c) { return complex3(sqrt(c.x), sqrt(c.y), sqrt(c.z)); }
[Differentiable] float3 norm(complex3 c) { return float3(norm(c.x), norm(c.y), norm(c.z)); }

[Differentiable] float mse(float3 input, no_diff float3 ref) {
    float3 error = input - ref; return dot(error, error);
}

[Differentiable] float safe_acos(float x) {
    return acos(clamp(x, -1, 1));
}

[Differentiable] float angle_between(float3 v1, float3 v2) {
    if (dot(v1, v2) < 0) return k_pi - 2 * safe_acos(length(v1 + v2) / 2);
    else return 2 * safe_acos(length(v2 - v1) / 2);
}

[Differentiable] float evaluate_polynomial_0(float t, float c) { return c; }

[Differentiable] float ffma(float a, float b, float c) { return a * b + c; }

float evaluate_polynomial_1(
    float t, float c1, float c0) {
    return ffma(t, evaluate_polynomial_0(t, c0), c1);
}

float evaluate_polynomial_2(
    float t, float c2, float c1, float c0) {
    return ffma(t, evaluate_polynomial_1(t, c1, c0), c2);
}

float evaluate_polynomial_3(
    float t, float c3, float c2, float c1, float c0) {
    return ffma(t, evaluate_polynomial_2(t, c2, c1, c0), c3);
}

float evaluate_polynomial_4(
    float t, float c4, float c3, float c2, float c1, float c0) {
    return ffma(t, evaluate_polynomial_3(t, c3, c2, c1, c0), c4);
}

float evaluate_polynomial_5(
    float t, float c5, float c4, float c3, float c2, float c1, float c0) {
    return ffma(t, evaluate_polynomial_4(t, c4, c3, c2, c1, c0), c5);
}

float evaluate_polynomial_6(
    float t, float c6, float c5, float c4, float c3, float c2, float c1, float c0) {
    return ffma(t, evaluate_polynomial_5(t, c5, c4, c3, c2, c1, c0), c6);
}

float erf(float x) {
    // Early return for large |x|.
    if (abs(x) >= 4.0) {
        return asfloat((asuint(x) & 0x80000000) ^ asuint(1.0));
    }
    // Polynomial approximation based on https://forums.developer.nvidia.com/t/optimized 
    // -version -of-single -precision -error -function - erff / 40977
    if (abs(x) > 1.0) {
        float A1 = 1.628459513;
        float A2 = 9.15674746e-1;
        float A3 = 1.54329389e-1;
        float A4 = -3.51759829e-2;
        float A5 = 5.66795561e-3;
        float A6 = -5.64874616e-4;
        float A7 = 2.58907676e-5;
        float a = abs(x);
        float y = 1.0 - exp2(-(((((((A7 * a + A6) * a + A5) * a + A4) * a + A3) * a + A2) * a + A1) * a));
        return asfloat((asuint(x) & 0x80000000) ^ asuint(y));
    } else {
        float A1 = 1.128379121;
        float A2 = -3.76123011e-1;
        float A3 = 1.12799220e-1;
        float A4 = -2.67030653e-2;
        float A5 = 4.90735564e-3;
        float A6 = -5.58853149e-4;
        float x2 = x * x;
        return (((((A6 * x2 + A5) * x2 + A4) * x2 + A3) * x2 + A2) * x2 + A1) * x;
    }
}

float unpack_cpu_half(uint16_t hdata) {
    int s = (hdata >> 15) & 0x00000001;
    int e = (hdata >> 10) & 0x0000001f;
    int m = hdata & 0x000003ff;

    if (e == 0) {
        if (m == 0) {
            uint32_t result = (uint32_t)(s << 31);
            return asfloat(result);
        }
        else {
            while (!bool(m & 0x00000400)) {
                m <<= 1;
                e -= 1;
            }

            e += 1;
            m &= ~0x00000400;
        }
    }
    else if (e == 31) {
        if (m == 0) {
            uint32_t result = (uint32_t)((s << 31) | 0x7f800000);
            return asfloat(result);
        } else {
            uint32_t result = (uint32_t)((s << 31) | 0x7f800000 | (m << 13));
            return asfloat(result);
        }
    }

    e = e + (127 - 15);
    m = m << 13;

    uint32_t result = (uint32_t)((s << 31) | (e << 23) | m);
    return asfloat(result);
}

float3 yuv2rgb(float3 yuv) {
    float3 rgb;
    rgb.x = yuv.x + 1.13983f * yuv.z;
    rgb.y = yuv.x - 0.39465f * yuv.y - 0.58060f * yuv.z;
    rgb.z = yuv.x + 2.03211f * yuv.y;
    return rgb;
}

[Differentiable] float average(float3 v) { return (v.x + v.y + v.z) / 3; }

// decompose a floating-point number into its components
int exponent(float v) { return (asuint(v) >> 23) - 127; }
int significand(float v) { return asuint(v) & ((1 << 23) - 1); }
uint32_t sign_bit(float v) { return asuint(v) & 0x80000000; }

float min(float4 v) { return min(v.x, min(v.y, min(v.z, v.w))); }
float max(float4 v) { return max(v.x, max(v.y, max(v.z, v.w))); }
float fast_exp(float x) {
    // Compute x' such that e^x = 2^(x')
    float xp = x * 1.442695041f;
    // Find integer and fractional components of x'
    float fxp = floor(xp);
    float f = xp - fxp;
    int i = (int)fxp;
    // <Evaluate polynomial approximation of 2^f
    float twoToF = evaluate_polynomial_3(
        f, 1.f, 0.695556856f,
        0.226173572f, 0.0781455737f);
    // Scale 2^f by 2^i and return final result
    int exponent = exponent(twoToF) + i;
    if (exponent < -126) return 0;
    if (exponent > 127) return k_inf;
    uint32_t bits = asuint(twoToF);
    bits &= 0b10000000011111111111111111111111u;
    bits |= (exponent + 127) << 23;
    return asfloat(bits);
}

float sample_exponential(float u, float a) { return -log(1 - u) / a; }

float2 fast_exp(float2 v) { return float2(exp(v.x), exp(v.y)); }
float3 fast_exp(float3 v) { return float3(exp(v.x), exp(v.y), exp(v.z)); }

#endif // !_SRENDERER_COMMMON_MATH_HEADER_