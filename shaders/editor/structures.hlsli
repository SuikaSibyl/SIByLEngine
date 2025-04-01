#ifndef _SEDITOR_STRUCTURES_HLSLI_
#define _SEDITOR_STRUCTURES_HLSLI_

struct Line3DData {
    float3 position_1;
    uint padding_1;
    float3 position_2;
    uint padding_2;
    float3 color;
    float width;
};

struct BillboardData {
    float3 position;
    int iconIDX;
    float2 size;
    int recordID;
    int padding;
};

#endif // _SEDITOR_STRUCTURES_HLSLI_