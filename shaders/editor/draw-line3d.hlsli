#ifndef _SRENDERER_EDITOR_DRAW_LINE3D_HLSLI_
#define _SRENDERER_EDITOR_DRAW_LINE3D_HLSLI_

#include "structures.hlsli"
#include "common/geometry.hlsli"

void InsertLine3D(
    RWStructuredBuffer<Line3DData> lines, int index,
    float3 start, float3 end, float3 color, float width) {
    Line3DData line;
    line.position_1 = start;
    line.position_2 = end;
    line.color = color;
    line.width = width;
    lines[index] = line;
}

void InsertBBOX(
    RWStructuredBuffer<Line3DData> lines, int index,
    AABB bbox, float3 color, float width) {
    const float3 min = bbox.min;
    const float3 max = bbox.max;
    InsertLine3D(lines, index +  0, min, float3(min.x, min.y, max.z), color, width);
    InsertLine3D(lines, index +  1, min, float3(min.x, max.y, min.z), color, width);
    InsertLine3D(lines, index +  2, min, float3(max.x, min.y, min.z), color, width);
    InsertLine3D(lines, index +  3,float3(max.x, max.y, max.z), float3(min.x, max.y, max.z), color, width);
    InsertLine3D(lines, index +  4,float3(max.x, max.y, max.z), float3(max.x, min.y, max.z), color, width);
    InsertLine3D(lines, index +  5,float3(max.x, max.y, max.z), float3(max.x, max.y, min.z), color, width);
    InsertLine3D(lines, index +  6,float3(min.x, max.y, min.z), float3(min.x, max.y, max.z), color, width);
    InsertLine3D(lines, index +  7,float3(min.x, max.y, min.z), float3(max.x, max.y, min.z), color, width);
    InsertLine3D(lines, index +  8,float3(max.x, min.y, min.z), float3(min.x, min.y, min.z), color, width);
    InsertLine3D(lines, index +  9,float3(max.x, min.y, min.z), float3(max.x, max.y, min.z), color, width);
    InsertLine3D(lines, index + 10,float3(min.x, min.y, max.z), float3(min.x, max.y, max.z), color, width);
    InsertLine3D(lines, index + 11,float3(min.x, min.y, max.z), float3(max.x, min.y, max.z), color, width);
}

#endif _SRENDERER_EDITOR_DRAW_LINE3D_HLSLI_