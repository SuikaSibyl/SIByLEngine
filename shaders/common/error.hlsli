#ifndef _SRENDERER_COMMON_ERROR_HLSLI_
#define _SRENDERER_COMMON_ERROR_HLSLI_

#include "math.hlsli"
//
float next_float_up(float v) {
    // Handle infinity and negative zero for next_float_up()
    if (isinf(v) && v > 0.f) return v;
    if (v == -0.f) v = 0.f;
    // Advance v to next higher float
    uint32_t ui = asuint(v);
    if (v >= 0) ++ui;
    else --ui;
    return asfloat(ui);
}

float next_float_down(float v) {
    // Handle infinity and positive zero for _NextFloatDown()_
    if (isinf(v) && v < 0.) return v;
    if (v == 0.f) v = -0.f;
    uint32_t ui = asuint(v);
    if (v > 0) --ui;
    else ++ui;
    return asfloat(ui);
}

float add_round_up(float a, float b) { return next_float_up(a + b); }
float add_round_down(float a, float b) { return next_float_down(a + b); }
float sub_round_up(float a, float b) { return add_round_up(a, -b); }
float sub_round_down(float a, float b) { return add_round_down(a, -b); }
float mul_round_up(float a, float b) { return next_float_up(a * b); }
float mul_round_down(float a, float b) { return next_float_down(a * b); }
float div_round_up(float a, float b) { return next_float_up(a / b); }
float div_round_down(float a, float b) { return next_float_down(a / b); }
float sqrt_round_up(float a) { return next_float_up(sqrt(a)); }
float sqrt_round_down(float a) { return next_float_down(sqrt(a)); }

struct interval {
    float low;
    float high;

    __init() { low = 0; high = 0; }
    __init(float v) { low = v; high = v; }
    __init(float l, float h) { low = l; high = h; }

    static interval from_value_and_error(float v, float err) {
        interval i;
        if (err == 0) i.low = i.high = v;
        else {
            i.low = sub_round_down(v, err);
            i.high = add_round_up(v, err);
        }
        return i;
    }

    float upper_bound() { return high; }
    float lower_bound() { return low; }
    float midpoint() { return 0.5f * (low + high); }
    float width() { return high - low; }
    float as_float() { return midpoint(); }

    static bool in_range(float v, interval i) {
        return v >= i.lower_bound() && v <= i.upper_bound(); }
    static bool in_range(interval a, interval b) {
        return a.lower_bound() <= b.upper_bound() &&
               a.upper_bound() >= b.lower_bound(); }
    
};

interval operator +(interval a, interval b) {
    return { add_round_down(a.low, b.low), add_round_up(a.high, b.high) }; }

interval operator -(interval a, interval b) {
    return { sub_round_down(a.low, b.high), sub_round_up(a.high, b.low) }; }

interval operator *(interval a, interval b) {
    float4 lp = { mul_round_down(a.low, b.low), mul_round_down(a.high, b.low),
                  mul_round_down(a.low, b.high), mul_round_down(a.high, b.high) };
    float4 hp = { mul_round_up(a.low, b.low), mul_round_up(a.high, b.low),
                  mul_round_up(a.low, b.high), mul_round_up(a.high, b.high) };
    return { min(lp), max(hp) };
}

interval sqr(interval a, interval i) {
    float alow = abs(i.lower_bound());
    float ahigh = abs(i.upper_bound());
    if (alow > ahigh) swap(alow, ahigh);
    if (interval::in_range(0, i))
        return { 0, mul_round_up(ahigh, ahigh) };
    return { mul_round_down(alow, alow), mul_round_up(ahigh, ahigh) };
}

struct float3i {
    interval x, y, z;

    __init() {}
    __init(float3 v) { x = interval(v.x); y = interval(v.y); z = interval(v.z); }
    __init(interval x, interval y, interval z) { this.x = x; this.y = y; this.z = z; }

    float3 error() { return float3(x.width(), y.width(), z.width()); }
    bool is_exact() { return x.low == x.high && y.low == y.high && z.low == z.high; }
    float3 as_float3() { return float3(x.as_float(), y.as_float(), z.as_float()); }
};

#endif // _SRENDERER_COMMON_ERROR_HLSLI_