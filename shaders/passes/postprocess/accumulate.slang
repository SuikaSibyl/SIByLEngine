

[[vk::push_constant]] cbuffer PushConstants {
    uint2 resolution;
    uint gAccumCount;
    bool gAccumulate;
    bool gMovingAverageMode;
};
Texture2D<float4> u_input;
RWTexture2D<float4> u_lastSum;
RWTexture2D<float4> u_output;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 pixel = int2(dtid.xy);
    if (any(pixel >= resolution)) return;

    const float4 curColor = u_input[pixel];
    if (gAccumCount == 0) {
        u_lastSum[pixel] = float4(0);
    }

    float4 outputColor;
    if (gAccumulate) {
        float curWeight = 1.0f / (gAccumCount + 1);
        if (gMovingAverageMode) {
            // Exponential weighted moving average mode.
            // Blend the accumulated frame with the current frame.
            float4 sum = u_lastSum[pixel];
            outputColor = lerp(sum, curColor, curWeight);
            u_lastSum[pixel] = outputColor;
        }
        else {
            // High precision mode with no frame limit.
            // Fetch previous sum and compute the new sum.
            float4 sum = u_lastSum[pixel] + curColor;
            outputColor = sum * curWeight;
            u_lastSum[pixel] = sum;
        }
    }
    else {
        outputColor = curColor;
    }

    // gamma correction
    // outputColor = pow(outputColor, 1.0f / 2.2f);
    u_output[pixel] = outputColor;
}