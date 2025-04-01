#ifndef _SRENDERER_COMMMON_RANDOM_HEADER_
#define _SRENDERER_COMMMON_RANDOM_HEADER_

#define RANDOM_SAMPLER_IMPL_PCG32I 0
#define RANDOM_SAMPLER_IMPL_MURMUR3 1

// By default we use pcg32_i implementation
#ifndef RANDOM_SAMPLER_IMPL
#define RANDOM_SAMPLER_IMPL RANDOM_SAMPLER_IMPL_PCG32I
#endif

#if RANDOM_SAMPLER_IMPL == RANDOM_SAMPLER_IMPL_MURMUR3
#include "space_filling_curve.hlsli"
#endif

/**
 * CRNG
 * @url: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 */
uint Hash_CRNG(uint seed) {
    const uint state = seed * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
* Jenkins
// @url: http://burtleburtle.net/bob/hash/integer.html
*/
uint Hash_Jenkins(uint a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

/**
* murmur
*/
uint Hash_murmur(uint a) {
    uint m = 0x5bd1e995;
    uint r = 24;
    uint h = 64684;
    uint k = a;
    k *= m;
    k ^= (k >> r);
    k *= m;
    h *= m;
    h ^= k;
    return h;
}

#if RANDOM_SAMPLER_IMPL == RANDOM_SAMPLER_IMPL_PCG32I

/**
 * Random Number generator
 * pcg32i_random_t based.
 * @ref: https://www.pcg-random.org/index.html
 * @ref: https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#perfectlyspecularreflections/refactoringintersectioninformation
 */
struct RandomSamplerState { uint state; };

RandomSamplerState InitRandomSampler(uint threadIndex, uint frameIndex) {
    RandomSamplerState r;
    r.state = frameIndex + Hash_CRNG(threadIndex);
    return r;
}

RandomSamplerState InitRandomSampler(uint2 pixelPos, uint frameIndex) {
    RandomSamplerState r;
    r.state = frameIndex + Hash_CRNG((pixelPos.x << 16) | pixelPos.y);
    return r;
}
// Random number generation using pcg32i_random_t, using inc = 1.
uint StepRNG(uint rngState) {
    return rngState * 747796405 + 1;
}
// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float StepAndOutputRNGFloat(inout RandomSamplerState rngState) {
    // Condensed version of pcg_output_rxs_m_xs_32_32,
    // with simple conversion to floating-point [0,1].
    rngState.state = StepRNG(rngState.state);
    uint word = ((rngState.state >> ((rngState.state >> 28) + 4)) ^ rngState.state) * 277803737;
    word = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

uint SampleUint(inout RandomSamplerState rngState) {
    rngState.state = StepRNG(rngState.state);
    uint word = ((rngState.state >> ((rngState.state >> 28) + 4)) ^ rngState.state) * 277803737;
    word = (word >> 22) ^ word;
    return word;
}

// Sample a uniform float in [0,1]
float SampleUniformFloat(inout RandomSamplerState r) {
    return StepAndOutputRNGFloat(r);
}

#elif RANDOM_SAMPLER_IMPL == RANDOM_SAMPLER_IMPL_MURMUR3

struct RandomSamplerState {
    uint seed;
    uint index;
};

RandomSamplerState InitRandomSampler(uint2 pixelPos, uint frameIndex) {
    RandomSamplerState state;
    const uint linearPixelIndex = ZCurve2DToMortonCode(pixelPos);
    state.index = 1;
    state.seed = Hash_Jenkins(linearPixelIndex) + frameIndex;
    return state;
}

uint murmur3(inout RandomSamplerState r) {
#define ROT32(x, y) ((x << y) | (x >> (32 - y)))
    // https://en.wikipedia.org/wiki/MurmurHash
    uint c1 = 0xcc9e2d51;
    uint c2 = 0x1b873593;
    uint r1 = 15;
    uint r2 = 13;
    uint m = 5;
    uint n = 0xe6546b64;
    uint hash = r.seed;
    uint k = r.index++;
    k *= c1;
    k = ROT32(k, r1);
    k *= c2;
    hash ^= k;
    hash = ROT32(hash, r2) * m + n;
    hash ^= 4;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);
#undef ROT32
    return hash;
}

uint SampleUint(inout RandomSamplerState r) {
    return v = murmur3(r);
}

float SampleUniformFloat(inout RandomSamplerState r) {
    uint v = murmur3(r);
    const uint one = floatBitsToUint(1.f);
    const uint mask = (1 << 23) - 1;
    return uintBitsToFloat((mask & v) | one) - 1.f;
}

#endif

RandomSamplerState InitRandomSampler(uint2 index, uint frameIndex, uint pass) {
    return InitRandomSampler(index, frameIndex + pass * 13);
}

float GetNextRandom(inout RandomSamplerState r) {
    return SampleUniformFloat(r);
}

float2 GetNextRandomFloat2(inout RandomSamplerState r) {
    return float2(GetNextRandom(r), GetNextRandom(r));
}

float3 GetNextRandomFloat3(inout RandomSamplerState r) {
    return float3(GetNextRandom(r), GetNextRandom(r), GetNextRandom(r));
}

Array<float, n> GetNextNRandomFloat<let n : int>(inout RandomSamplerState r) {
    Array<float, n> result;
    [ForceUnroll]
    for (int i = 0; i < n; i++) {
        result[i] = GetNextRandom(r);
    }
    return result;
}

uint GetNextRandomUint(inout RandomSamplerState r) {
    return SampleUint(r);
}

#endif // !_SRENDERER_COMMMON_RANDOM_HEADER_