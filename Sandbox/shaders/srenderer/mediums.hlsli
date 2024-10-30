#ifndef _SRENDERER_MEDIUMS_HLSLI_
#define _SRENDERER_MEDIUMS_HLSLI_

#include "common/math.hlsli"
#include "srenderer/mediums/medium.hlsli"


namespace mediums {
imedium::RayMajorantIterator sample_ray(Ray ray, float tMax, MediumPacket medium) {
    switch (medium.get_medium_type()) {
    case MediumPacket::MediumType::HOMOGENEOUS:
        return HomogeneousMedium::sample_ray(ray, tMax, HomogeneousMediumParameter(medium));
    case MediumPacket::MediumType::GRID_MEDIUM:
        return GridMedium::sample_ray(ray, tMax, GridMediumParameter(medium));
    case MediumPacket::MediumType::RGBGRID_MEDIUM:
        return RGBGridMedium::sample_ray(ray, tMax, RGBGridMediumParameter(medium));
    default: return {};
    }
}

imedium::MediumProperties sample_point(float3 p, MediumPacket medium) {
    switch (medium.get_medium_type()) {
    case MediumPacket::MediumType::HOMOGENEOUS:
        return HomogeneousMedium::sample_point(p, HomogeneousMediumParameter(medium));
    case MediumPacket::MediumType::GRID_MEDIUM:
        return GridMedium::sample_point(p, GridMediumParameter(medium));
    case MediumPacket::MediumType::RGBGRID_MEDIUM:
        return RGBGridMedium::sample_point(p, RGBGridMediumParameter(medium));
    default: return {};
    }
}
}

struct HomogeneousMediumData {
    float3 sigma_a;
    float3 sigma_s;
    float3 sigma_t;
    float g;

    __init() {}
    __init(MediumPacket data) {
        sigma_a = data.sigma_a;
        sigma_s = data.sigma_s;
    }
};

[Differentiable] float sample_transmittance(float sigma_t, float u) {
    return -log(1 - u) / sigma_t;
}

float sample_transmittance(float3 sigma_t, float u, int channel) {
    return -log(1 - u) / sigma_t[channel];
}

float sample_transmittance_pdf(float3 sigma_t, float t) {
    return average(exp(-sigma_t * t) * sigma_t);
}

float sample_transmittance_pdf_above_distance(float3 sigma_t, float t) {
    return average(exp(-sigma_t * t));
}

float sample_equiangular(
    Ray ray,
    float3 light_position,
    float max_distance,
    float u,
) {
    // get coord of closest point to light along (infinite) ray
    float delta = dot(light_position - ray.origin, ray.direction);
    // get distance this point is from light
    float D = length(ray.origin + delta * ray.direction - light_position);
    // get angle of endpoints
    float thetaA = atan2(float(0.0 - delta), D);
    float thetaB = atan2(max_distance - delta, D);
    // take sample
    float t = D * tan(lerp(thetaA, thetaB, u));
    return delta + t;
}

float pdf_sample_equiangular(
    Ray ray,
    float3 light_position,
    float dist,
    float max_distance,
) {
    // get coord of closest point to light along (infinite) ray
    float delta = dot(light_position - ray.origin, ray.direction);
    // get distance this point is from light
    float D = length(ray.origin + delta * ray.direction - light_position);
    // get angle of endpoints
    float thetaA = atan2(float(0.0 - delta), D);
    float thetaB = atan2(max_distance - delta, D);
    // take sample
    float t = dist - delta;
    return D / ((thetaB - thetaA) * (D * D + t * t));
}

float pdf_sample_equiangular_above_dist(
    Ray ray,
    float3 light_position,
    float dist,
    float max_distance,
) {
    // get coord of closest point to light along (infinite) ray
    float delta = dot(light_position - ray.origin, ray.direction);
    // get distance this point is from light
    float D = length(ray.origin + delta * ray.direction - light_position);
    // get angle of endpoints
    float thetaA = atan2(float(0.0 - delta), D);
    float thetaB = atan2(max_distance - delta, D);
    // take sample
    float thetaMix = atan2(dist - delta, D);
    return (thetaB - thetaMix) / (thetaB - thetaA);
}

#endif // _SRENDERER_MEDIUMS_HLSLI_