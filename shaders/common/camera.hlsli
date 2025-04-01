#ifndef _SRENDERER_COMMMON_CAMERA_HEADER_
#define _SRENDERER_COMMMON_CAMERA_HEADER_

#include "cpp_compatible.hlsli"

struct CameraData {
    float4x4 viewMat;                 ///< Camera view matrix.
    float4x4 invViewMat;              ///< Camera inverse view matrix.
    float4x4 projMat;                 ///< Camera projection matrix.
    float4x4 invProjMat;              ///< Camera inverse projection matrix.
    float4x4 viewProjMat;             ///< Camera view-projection matrix.
    float4x4 invViewProj;             ///< Camera inverse view-projection matrix.
    float4x4 viewProjMatNoJitter;     ///< Camera view-projection matrix. No jittering is applied!
    float4x4 projMatNoJitter;         ///< Camera projection matrix. No jittering is applied!
    
    float3 posW;         ///< Camera world-space position.
    float focalLength;   ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
    float3 prevPosW;     ///< Camera world-space position associated to previous frame.
    float rectArea;      ///< Recrtangles area A at z=1 plane.
    float3 up;           ///< Camera world-space up vector.
    float aspectRatio;   ///< Camera film frame aspect ratio, equal to frameWidth / frameHeight
    float3 target;       ///< Camera target point in world-space.
    float nearZ;         ///< Camera near plane.
    float3 cameraU;      ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.
    float farZ;          ///< Camera far plane.
    float3 cameraV;      ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float jitterX;       ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float3 cameraW;      ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float jitterY;     ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    float frameHeight;    ///< Camera film frame height in mm. 24 is the height of a 35mm film
    float frameWidth;     ///< Camera film frame width in mm.  42 2/3 is the width assuming 24mm height and a 16:9 aspect ratio
    float focalDistance;  ///< Camera focal distance in scene units.
    float apertureRadius; ///< Camera aperture radius in scene units.
    float shutterSpeed;   ///< Camera shutter speed in seconds.
    float ISOSpeed;       ///< Camera film speed based on ISO standards.
    int mediumID;
    float _padding2;
    
    float2 clipToWindowScale;
    float2 clipToWindowBias;
};

int2 getViewportSize(in_ref(CameraData) data) {
    return int2(data.clipToWindowScale * float2(2, -2));
}

float2 getInvViewportSize(in_ref(CameraData) data) {
    return float2(1.f) / float2(data.clipToWindowScale * float2(2, -2));
}

float2 GetScreenPixelPos(in_ref(CameraData) data, in_ref(float3) posW) {
    const float4 clip = mul(float4(posW, 1.0f), data.viewProjMat);
    return getViewportSize(data) * (0.5 * clip.xy / clip.w + 0.5);
}

#endif // !_SRENDERER_COMMMON_CAMERA_HEADER_