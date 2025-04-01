#ifndef _SRENDERER_COMMMON_PACKING_HEADER_
#define _SRENDERER_COMMMON_PACKING_HEADER_

#include "cpp_compatible.hlsli"
#include "octahedral.hlsli"

inline void packFloatHigh(float v, inout_ref(uint) u) {
    u &= 0x0000FFFF;
    u |= (f32tof16(v) << 16);
}

inline void packFloatLow(float v, inout_ref(uint) u) {
    u &= 0xFFFF0000;
    u |= f32tof16(v);
}

inline void packFloat2(float2 v, inout_ref(uint) u) {
    uint l = f32tof16(v.x);
    uint h = f32tof16(v.y);
    u = (h << 16) + l;
}

inline void packFloat3(float3 v, inout_ref(uint2) u) {
    packFloat2(v.xy, u.x);
    packFloatHigh(v.z, u.y);
}

inline void packFloat4(float4 v, inout_ref(uint2) u) {
    packFloat2(float2(v.x, v.y), u.x);
    packFloat2(float2(v.y, v.z), u.y);
}

inline uint2 PackFloat4ToUint2(float4 v) {
    uint2 u;
    packFloat2(float2(v.x, v.y), u.x);
    packFloat2(float2(v.y, v.z), u.y);
    return u;
}

inline float unpackFloatHigh(uint u) {
    uint h = (u >> 16) & 0xffff;
    return f16tof32(h);
}

inline float unpackFloatLow(uint u) {
    uint l = u & 0xffff;
    return f16tof32(l);
}

inline float2 unpackFloat2(uint u) {
    uint l = u & 0xffff;
    uint h = (u >> 16) & 0xffff;
    return float2(f16tof32(l), f16tof32(h));
}

inline float3 unpackFloat3(uint2 u) {
    return float3(unpackFloat2(u.x), unpackFloatHigh(u.y));
}

inline float4 unpackFloat4(uint2 u) {
    return float4(unpackFloat2(u.x), unpackFloat2(u.y));
}

// Return the given float value as an unsigned integer within the given numerical scale.
inline uint PACK_FLOAT_UINT(float v, float scale) {
    return (uint)floor(v * scale + 0.5f);
}

inline uint Pack_R10_UFLOAT(float r, float d = 0.5f) {
    const uint mask = (1U << 10) - 1U;
    return (uint)floor(r * mask + d) & mask;
}
inline uint Pack_R11_UFLOAT(float r, float d = 0.5f) {
    const uint mask = (1U << 11) - 1U;
    return (uint)floor(r * mask + d) & mask;
}

inline float Unpack_R10_UFLOAT(uint r) {
    const uint mask = (1U << 10) - 1U;
    return (float)(r & mask) / (float)mask;
}
inline  float Unpack_R11_UFLOAT(uint r) {
    const uint mask = (1U << 11) - 1U;
    return (float)(r & mask) / (float)mask;
}

/** Pack a float3 into a 32-bit unsigned integer.
 * All channels use 10 bits and 2 bits are unused.
 * Compliment of UNPACK_FLOAT3_UINT(). */
inline uint PACK_FLOAT3_UINT(float3 input) {
    return (PACK_FLOAT_UINT(input.r, 1023.f)) 
         | (PACK_FLOAT_UINT(input.g, 1023.f) << 10) 
         | (PACK_FLOAT_UINT(input.b, 1023.f) << 20);
}
/** Unpack a packed 32-bit unsigned integer to a float3.
 * Compliment of PACK_FLOAT3_UINT().  */
inline float3 UNPACK_FLOAT3_UINT(uint input) {
    float3 output;
    output.x = (float)(input & 0x000003FF) / 1023.f;
    output.y = (float)((input >> 10) & 0x000003FF) / 1023.f;
    output.z = (float)((input >> 20) & 0x000003FF) / 1023.f;
    return output;
}

// Packs a normalized float value into an 8-bit unsigned integer representation
inline uint Pack_R8_UFLOAT(float value) {
    // Clamp the value to the valid range [0, 1]
    value = clamp(value, 0.0, 1.0);
    // Convert the float value to an 8-bit integer value
    return uint(value * 255.0f + 0.5f);
}

// Unpacks an 8-bit value into a normalized float value
inline float Unpack_R8_UFLOAT(uint packedValue) {
    // Convert the packed value to a normalized float
    return clamp(packedValue & 0xff, 0, 255) / 255.0f;
}

inline uint Pack_R8G8B8_UFLOAT(float3 rgb) {
    uint r = Pack_R8_UFLOAT(rgb.r);
    uint g = Pack_R8_UFLOAT(rgb.g) << 8;
    uint b = Pack_R8_UFLOAT(rgb.b) << 16;
    return r | g | b;
}

inline float3 Unpack_R8G8B8_UFLOAT(uint rgb) {
    float r = Unpack_R8_UFLOAT(rgb);
    float g = Unpack_R8_UFLOAT(rgb >> 8);
    float b = Unpack_R8_UFLOAT(rgb >> 16);
    return float3(r, g, b);
}

inline uint Pack_R8G8B8A8_Gamma_UFLOAT(float4 rgba, float gamma = 2.2) {
    rgba = pow(clamp(rgba, 0.0f, 1.0f), float4(1.0 / gamma));
    uint r = Pack_R8_UFLOAT(rgba.r);
    uint g = Pack_R8_UFLOAT(rgba.g) << 8;
    uint b = Pack_R8_UFLOAT(rgba.b) << 16;
    uint a = Pack_R8_UFLOAT(rgba.a) << 24;
    return r | g | b | a;
}

inline float4 Unpack_R8G8B8A8_Gamma_UFLOAT(uint rgba, float gamma = 2.2) {
    float r = Unpack_R8_UFLOAT(rgba);
    float g = Unpack_R8_UFLOAT(rgba >> 8);
    float b = Unpack_R8_UFLOAT(rgba >> 16);
    float a = Unpack_R8_UFLOAT(rgba >> 24);
    float4 v = float4(r, g, b, a);
    v = pow(clamp(v, 0.0f, 1.0f), float4(gamma));
    return v;
}

inline uint Pack_R11G11B10_UFLOAT(float3 rgb, float3 d = float3(0.5f, 0.5f, 0.5f)) {
    uint r = Pack_R11_UFLOAT(rgb.r, d.r);
    uint g = Pack_R11_UFLOAT(rgb.g, d.g) << 11;
    uint b = Pack_R10_UFLOAT(rgb.b, d.b) << 22;
    return r | g | b;
}

inline float3 Unpack_R11G11B10_UFLOAT(uint rgb) {
    float r = Unpack_R11_UFLOAT(rgb);
    float g = Unpack_R11_UFLOAT(rgb >> 11);
    float b = Unpack_R10_UFLOAT(rgb >> 22);
    return float3(r, g, b);
}

// Unpack two 16-bit snorm values from the lo/hi bits of a dword.
//  - packed: Two 16-bit snorm in low/high bits.
//  - returns: Two float values in [-1,1].
inline float2 UnpackSnorm2x16(uint packed) {
    int2 bits = int2(int(packed << 16), int(packed)) >> 16;
    float2 unpacked = max(float2(bits) / 32767.0, float2(-1.0));
    return unpacked;
}

// Pack two floats into 16-bit snorm values in the lo/hi bits of a dword.
//  - returns: Two 16-bit snorm in low/high bits.
inline uint PackSnorm2x16(float2 v) {
    v = ternary(any(isnan(v)), float2(0, 0), clamp(v, -1.0, 1.0));
    int2 iv = int2(round(v * 32767.0));
    uint packed = (iv.x & 0x0000ffff) | (iv.y << 16);

    return packed;
}

// Encode a normal packed as 2x 16-bit snorms in the octahedral mapping.
inline uint EncodeNormalizedVectorToSnorm2x16(float3 normal) {
    float2 octNormal = UnitVectorToSignedOctahedron(normal);
    return PackSnorm2x16(octNormal);
}

// Decode a normal packed as 2x 16-bit snorms in the octahedral mapping.
inline float3 DecodeNormalizedVectorFromSnorm2x16(uint packedNormal) {
    float2 octNormal = UnpackSnorm2x16(packedNormal);
    return SignedOctahedronToUnitVector(octNormal);
}

/**
 * Pack a float3 into a 32-bit RGBE format.
 * Using a shared exponent allows for a wider range of values to be represented.
 * The exponent is determined by the maximum absolute value of the input vector.
 * The RGBE format is proposed by Greg Ward in his paper "Real Pixels" (1994).
 * @ref: G. Ward, ``Real Pixels'', Graphics Gems II, Ed. by J. Arvo, Academic Press, 1992
 * The code is adapted from the Q2VKPT project:
 * @url: https://github.com/NVIDIA/Q2RTX/blob/a45c9357658e9f15ce2f6216d61a6693d22ba186/src/refresh/vkpt/shader/utils.glsl#L465
 */
inline uint PackRGBE(in_ref(float3) v) {
    const float3 va = max(float3(0), v);
    const float max_abs = max(va.r, max(va.g, va.b));
    if (max_abs == 0) return 0;
    const float exponent = floor(log2(max_abs));
    uint result;
    result = uint(clamp(exponent + 20, 0, 31)) << 27;
    const float scale = pow(2, -exponent) * 256.0;
    const uint3 vu = min(uint3(511), uint3(round(va * scale)));
    result |= vu.r;
    result |= vu.g << 9;
    result |= vu.b << 18;
    return result;
}

/**
 * Unpack a float3 into a 32-bit RGBE format.
 * Using a shared exponent allows for a wider range of values to be represented.
 * The RGBE format is proposed by Greg Ward in his paper "Real Pixels" (1994).
 * @ref: G. Ward, ``Real Pixels'', Graphics Gems II, Ed. by J. Arvo, Academic Press, 1992
 * The code is adapted from the Q2VKPT project:
 * @url: https://github.com/NVIDIA/Q2RTX/blob/a45c9357658e9f15ce2f6216d61a6693d22ba186/src/refresh/vkpt/shader/utils.glsl#L465
 */
inline float3 UnpackRGBE(uint x) {
    const int exponent = int(x >> 27) - 20;
    const float scale = pow(2, exponent) / 256.0;
    float3 v;
    v.r = float(x & 0x1ff) * scale;
    v.g = float((x >> 9) & 0x1ff) * scale;
    v.b = float((x >> 18) & 0x1ff) * scale;
    return v;
}

/**
 * Transforms an RGB color in Rec.709 to CIE XYZ.
 * The code is adapted from the RTXDI project:
 * @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/rtxdi-sdk/include/rtxdi/RtxdiMath.hlsli#L233
 */
inline float3 RGBToXYZInRec709(float3 c) {
    static const float3x3 M = float3x3(
        0.4123907992659595, 0.3575843393838780, 0.1804807884018343,
        0.2126390058715104, 0.7151686787677559, 0.0721923153607337,
        0.0193308187155918, 0.1191947797946259, 0.9505321522496608
    );
    return mul(M, c);
}

/**
 * Transforms an XYZ color to RGB in Rec.709.
 * The code is adapted from the RTXDI project:
 * @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/rtxdi-sdk/include/rtxdi/RtxdiMath.hlsli#L233
 */
inline float3 XYZToRGBInRec709(float3 c) {
    static const float3x3 M = float3x3(
        3.240969941904522, -1.537383177570094, -0.4986107602930032,
        -0.9692436362808803, 1.875967501507721, 0.04155505740717569,
        0.05563007969699373, -0.2039769588889765, 1.056971514242878
    );
    return mul(M, c);
}

/**
 * Encode an RGB color into a 32-bit LogLuv HDR format.
 * The supported luminance range is roughly 10^-6..10^6 in 0.17% steps.
 * The log-luminance is encoded with 14 bits and chroma with 9 bits each.
 * This was empirically more accurate than using 8 bit chroma.
 * Black (all zeros) is handled exactly.
 * The code is adapted from the RTXDI project:
 * @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/rtxdi-sdk/include/rtxdi/RtxdiMath.hlsli#L233
 */
inline uint EncodeRGBToLogLuv(float3 color) {
    // Convert RGB to XYZ.
    float3 XYZ = RGBToXYZInRec709(color);
    // Encode log2(Y) over the range [-20,20) in 14 bits (no sign bit).
    // TODO: Fast path that uses the bits from the fp32 representation directly.
    float logY = 409.6 * (log2(XYZ.y) + 20.0); // -inf if Y==0
    uint Le = uint(clamp(logY, 0.0, 16383.0));
    // Early out if zero luminance to avoid NaN in chroma computation.
    // Note Le==0 if Y < 9.55e-7. We'll decode that as exactly zero.
    if (Le == 0) return 0;
    // Compute chroma (u,v) values by:
    //  x = X / (X + Y + Z)
    //  y = Y / (X + Y + Z)
    //  u = 4x / (-2x + 12y + 3)
    //  v = 9y / (-2x + 12y + 3)
    //
    // These expressions can be refactored to avoid a division by:
    //  u = 4X / (-2X + 12Y + 3(X + Y + Z))
    //  v = 9Y / (-2X + 12Y + 3(X + Y + Z))
    //
    float invDenom = 1.0 / (-2.0 * XYZ.x + 12.0 * XYZ.y + 3.0 * (XYZ.x + XYZ.y + XYZ.z));
    float2 uv = float2(4.0, 9.0) * float2(XYZ.x, XYZ.y) * invDenom;
    // Encode chroma (u,v) in 9 bits each.
    // The gamut of perceivable uv values is roughly [0,0.62], so scale by 820 to get 9-bit values.
    uint2 uve = uint2(clamp(820.0f * uv, 0.0, 511.0));
    return (Le << 18) | (uve.x << 9) | uve.y;
}

/**
 * Decode an RGB color stored in a 32-bit LogLuv HDR format.
 * The code is adapted from the RTXDI project:
 * @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/rtxdi-sdk/include/rtxdi/RtxdiMath.hlsli#L233
 */
inline float3 DecodeLogLuvToRGB(uint packedColor) {
    // Decode luminance Y from encoded log-luminance.
    uint Le = packedColor >> 18;
    if (Le == 0) return float3(0, 0, 0);
    float logY = (float(Le) + 0.5) / 409.6 - 20.0;
    float Y = pow(2.0, logY);
    // Decode normalized chromaticity xy from encoded chroma (u,v).
    //
    //  x = 9u / (6u - 16v + 12)
    //  y = 4v / (6u - 16v + 12)
    //
    uint2 uve = uint2(packedColor >> 9, packedColor) & 0x1ff;
    float2 uv = (float2(uve) + 0.5f) / 820.0f;
    float invDenom = 1.0f / (6.0f * uv.x - 16.0f * uv.y + 12.0f);
    float2 xy = float2(9.0, 4.0) * uv * invDenom;
    // Convert chromaticity to XYZ and back to RGB.
    //  X = Y / y * x
    //  Z = Y / y * (1 - x - y)
    //
    float s = Y / xy.y;
    float3 XYZ = float3(s * xy.x, Y, s * (1.f - xy.x - xy.y));
    // Convert back to RGB and clamp to avoid out-of-gamut colors.
    return max(XYZToRGBInRec709(XYZ), float3(0.0f));
}

#endif // !_SRENDERER_COMMMON_PACKING_HEADER_