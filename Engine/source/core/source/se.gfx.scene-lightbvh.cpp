#include <se.gfx.hpp>
#include <span>

namespace se {
namespace gfx{
	struct DirectionCone {
		vec3 w;
		float cosTheta = std::numeric_limits<float>::infinity();

		DirectionCone() = default;
		DirectionCone(vec3 w, float cosTheta = 1.f) :w(w), cosTheta(cosTheta) {}

		auto is_empty() const noexcept -> bool { return cosTheta == std::numeric_limits<float>::infinity(); }

    static auto bound_subtended_directions(bounds3 const& b, point3 p) noexcept ->DirectionCone {
      // Compute bounding sphere for b and check if p is inside
      float radius; point3 pCenter;
      b.bounding_sphere(&pCenter, &radius);
      if (distance_squared(p, pCenter) < sqr(radius))
        return DirectionCone::entire_sphere();
      // Compute and return DirectionCone for bounding sphere
      vec3 w = normalize(pCenter - p);
      float sin2ThetaMax = sqr(radius) / distance_squared(pCenter, p);
      float cosThetaMax = safe_sqrt(1 - sin2ThetaMax);
      return DirectionCone(w, cosThetaMax);
    }
		static auto entire_sphere() noexcept -> DirectionCone { return DirectionCone(vec3(0, 0, 1), -1); }
	};

	auto union_direction_cone(DirectionCone const& a, DirectionCone const& b) noexcept -> DirectionCone {
		// Handle the cases where one or both cones are empty
		if (a.is_empty()) return b;
		if (b.is_empty()) return a;

		// Handle the cases where one cone is inside the other
		float theta_a = safe_acos(a.cosTheta);
		float theta_b = safe_acos(b.cosTheta);
		float theta_d = angle_between(a.w, b.w);
		if (std::min(theta_d + theta_b, M_FLOAT_PI) <= theta_a) return a;
		if (std::min(theta_d + theta_a, M_FLOAT_PI) <= theta_b) return b;

		// Compute the spread angle of the merged cone
		float theta_o = (theta_a + theta_d + theta_b) / 2;
		if (theta_o >= M_FLOAT_PI) return DirectionCone::entire_sphere();

		// Find the merged cone’s axis and return cone union
		float theta_r = theta_o - theta_a;
		vec3 wr = cross(a.w, b.w);
		if (wr.length_squared() == 0)
			return DirectionCone::entire_sphere();
		vec3 w = rotate(degrees(theta_r), wr)(a.w);
		return DirectionCone(w, std::cos(theta_o));
	}


	struct LightBounds {
		bounds3 bounds;
		vec3 w;
		vec3 rgb;
		float phi = 0.f;
		float cosTheta_o;
		float cosTheta_e;
		bool twoSided;

		vec3 centroid() const { return (bounds.pMin + bounds.pMax) / 2; }

	};

	auto union_light_bounds(LightBounds const& a, LightBounds const& b) noexcept -> LightBounds {
		// If one LightBounds has zero power, return the other
		if (a.phi == 0) return b;
		if (b.phi == 0) return a;

		// Find average direction and updated angles for LightBounds
		DirectionCone cone = union_direction_cone(DirectionCone(a.w, a.cosTheta_o), DirectionCone(b.w, b.cosTheta_o));
		float cosTheta_o = cone.cosTheta;
		float cosTheta_e = std::min(a.cosTheta_e, b.cosTheta_e);

		// Return final LightBounds union
		return { union_bounds(a.bounds, b.bounds), cone.w, a.rgb + b.rgb, 
			a.phi + b.phi, cosTheta_o, cosTheta_e, bool(a.twoSided | b.twoSided) };
	}

	auto cos_sub_clamped(
		float sinTheta_a, float cosTheta_a,
		float sinTheta_b, float cosTheta_b) -> float {
		if (cosTheta_a > cosTheta_b) return 1;
		return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
	};

  float sin_sub_clamped(
    float sinTheta_a, float cosTheta_a,
    float sinTheta_b, float cosTheta_b) {
    if (cosTheta_a > cosTheta_b) return 0;
    return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
  };

	inline vec2 octWrap(vec2 const& v) {
		return (vec2(1.f) - abs(vec2(v.y, v.x))) *
			select(vec2(v.x, v.y) >= vec2(0), vec2(1), vec2(-1));
	}

	inline vec2 UnitVectorToSignedOctahedron(vec3 normal) {
		// Project the sphere onto the octahedron (|x|+|y|+|z| = 1)
		normal.xy /= dot(vec3(1), abs(normal));
		// Then project the octahedron onto the xy-plane
		return (normal.z < 0) ? octWrap(normal.xy) : normal.xy;
	}

	inline uint32_t UnitVectorToUnorm32Octahedron(vec3 normal) {
		vec2 p = UnitVectorToSignedOctahedron(normal);
		p = clamp_vec2(vec2(p.x, p.y) * 0.5f + vec2(0.5f), vec2(0.f), vec2(1.f));
		return uint32_t(p.x * 0xfffe) | (uint32_t(p.y * 0xfffe) << 16);
	}

  inline vec3 SignedOctahedronToUnitVector(vec2 oct) {
    vec3 normal = vec3(oct, 1 - dot(vec2(1), abs(oct)));
    const float t = std::max(-normal.z, 0.f);
    normal.xy += select(normal.xy >= vec2(0), vec2(-t), vec2(t));
    return normalize(normal);
  }

  inline vec3 Unorm32OctahedronToUnitVector(uint32_t pUnorm) {
    vec2 p;
    p.x = std::clamp(float(pUnorm & 0xffff) / 0xfffe, 0.f, 1.f);
    p.y = std::clamp(float(pUnorm >> 16) / 0xfffe, 0.f, 1.f);
    p = p * 2.0 - vec2(1.0);
    return SignedOctahedronToUnitVector(p);
  }

	struct CompactLightBounds {
		float phi;
		uint32_t w;
		uint32_t bitfield;
		uint32_t qb_0;
		uint32_t qb_1;
		uint32_t qb_2;
		half coloru;
		half colorv;

		static uint32_t quantize_cos(float c) { return uint32_t(std::floor(32767.f * ((c + 1) / 2))); }

		// remaps a coordinate value c between min and max to the range
		// [0, 2^16 - 1] range of values that an unsigned 16-bit integer can store.
		static float quantize_bounds(float c, float min, float max) {
			if (min == max) return 0;
			return 65535.f * clamp((c - min) / (max - min), 0.f, 1.f);
		}

		CompactLightBounds() = default;
		CompactLightBounds(LightBounds const& lb, bounds3 const& allb) {
			uint32_t qCosTheta_o = quantize_cos(lb.cosTheta_o);
			uint32_t qCosTheta_e = quantize_cos(lb.cosTheta_e);
			uint32_t twoSided = lb.twoSided ? 1 : 0;

			bitfield = (qCosTheta_o << 17) | (qCosTheta_e << 2) | twoSided;
			phi = lb.phi;
			w = UnitVectorToUnorm32Octahedron(normalize(lb.w));
			// Quantize bounding box into qb
			uint32_t qb[3];
			for (int c = 0; c < 3; ++c) {
				uint32_t qb_lc = uint32_t(std::floor(quantize_bounds(
					lb.bounds.pMin[c], allb.pMin[c], allb.pMax[c])));
				uint32_t qb_rc = uint32_t(std::ceil(quantize_bounds(
					lb.bounds.pMax[c], allb.pMin[c], allb.pMax[c])));
				qb[c] = (qb_lc << 16) | qb_rc;
			}
			coloru = half(lb.rgb.g);
			colorv = half(lb.rgb.b);
			qb_0 = qb[0];
			qb_1 = qb[1];
			qb_2 = qb[2];
		}
	};

  struct LightBVHNode {
    CompactLightBounds cb;
    struct {
      unsigned int childOrLightIndex : 31;
      unsigned int isLeaf : 1;
    };

    static auto makeLeaf(unsigned int lightIndex,
      const CompactLightBounds& cb) -> LightBVHNode;
    static auto makeInterior(unsigned int child1Index,
      const CompactLightBounds& cb) -> LightBVHNode;


    bounds3 bounds(bounds3 allb) const {
      bounds3 b;
      b.pMin = vec3(
        lerp((cb.qb_0 >> 16) / 65535.f, allb.pMin.x, allb.pMax.x),
        lerp((cb.qb_1 >> 16) / 65535.f, allb.pMin.y, allb.pMax.y),
        lerp((cb.qb_2 >> 16) / 65535.f, allb.pMin.z, allb.pMax.z));
      b.pMax = vec3(
        lerp((cb.qb_0 & 0xffff) / 65535.f, allb.pMin.x, allb.pMax.x),
        lerp((cb.qb_1 & 0xffff) / 65535.f, allb.pMin.y, allb.pMax.y),
        lerp((cb.qb_2 & 0xffff) / 65535.f, allb.pMin.z, allb.pMax.z));
      return b;
    }

    float cos_theta_o() const { return 2 * ((cb.bitfield >> 17) / 32767.f) - 1; }
    float cos_theta_e() const { return 2 * (((cb.bitfield >> 2) & 0x7FFF) / 32767.f) - 1; }
    bool two_sided() const { return (cb.bitfield & 1) != 0; }

    float importance(vec3 p, vec3 n, bounds3 allb) const {
      bounds3 b = bounds(allb);
      float cosTheta_o = cos_theta_o();
      float cosTheta_e = cos_theta_e();
      // Return importance for light bounds at reference point
      // Compute clamped squared distance to reference point
      vec3 pc = (b.pMin + b.pMax) / 2;
      float d2 = distance_squared(p, pc);
      d2 = std::max(d2, length(b.diagonal()) / 2);
      // Compute sine and cosine of angle to vector w, theta_w
      vec3 wi = normalize(p - pc);
      vec3 w = Unorm32OctahedronToUnitVector(cb.w);
      float cosTheta_w = dot(w, wi);
      if (two_sided()) cosTheta_w = std::abs(cosTheta_w);
      float sinTheta_w = safe_sqrt(1 - sqr(cosTheta_w));
      // Compute cos theta_b for reference point
      float cosTheta_b = DirectionCone::bound_subtended_directions(b, p).cosTheta;
      float sinTheta_b = safe_sqrt(1 - sqr(cosTheta_b));
      // Compute cos theta' and test against cos theta_e
      float sinTheta_o = safe_sqrt(1 - sqr(cosTheta_o));
      float cosTheta_x = cos_sub_clamped(
        sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
      float sinTheta_x = sin_sub_clamped(
        sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
      float cosThetap = cos_sub_clamped(
        sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
      if (cosThetap <= cosTheta_e) return 0;
      // Return final importance at reference point
      float importance = cb.phi * cosThetap / d2;
      // Account for cos theta_i in importance at surfaces
      if (n.x != 0 || n.y != 0 || n.z != 0) {
        float cosTheta_i = std::abs(dot(wi, n));
        float sinTheta_i = safe_sqrt(1 - sqr(cosTheta_i));
        float cosThetap_i = cos_sub_clamped(
          sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
        importance *= cosThetap_i;
      }
      return importance;
    }
  };

  auto LightBVHNode::makeLeaf(unsigned int lightIndex,
    const CompactLightBounds& cb) -> LightBVHNode {
    return LightBVHNode{ cb, {lightIndex, 1} };
  }

  auto LightBVHNode::makeInterior(unsigned int child1Index,
    const CompactLightBounds& cb) -> LightBVHNode {
    return LightBVHNode{ cb, {child1Index, 0} };
  }

  struct BVHLightSampler :ILightSampler {
    std::vector<Light> lights;
    std::vector<Light> infiniteLights;
    bounds3 allLightBounds;
    std::vector<LightBVHNode> nodes;
    std::vector<uint32_t> lightToBitTrail;

    auto build_bvh(std::vector<std::pair<int, LightBounds>>& bvhLights,
      int start, int end, uint32_t bitTrail, int depth
    ) noexcept -> std::pair<int, LightBounds>;
    auto evaluateCost(const LightBounds& b,
      const bounds3& bounds, int dim) const -> float;
  };

  auto BVHLightSampler::build_bvh(
    std::vector<std::pair<int, LightBounds>>& bvhLights,
    const int start, const int end,
    uint32_t bitTrail, int depth
  ) noexcept -> std::pair<int, LightBounds> {
    // Initialize leaf node if only a single light remains
    if (end - start == 1) {
      int nodeIndex = nodes.size();
      CompactLightBounds cb(bvhLights[start].second, allLightBounds);
      int lightIndex = bvhLights[start].first;
      nodes.push_back(LightBVHNode::makeLeaf(lightIndex, cb));
      lightToBitTrail[lightIndex] = bitTrail;
      return { nodeIndex, bvhLights[start].second };
    }

    // Choose split dimension and position using modified SAH
    // Compute bounds and centroid bounds for lights
    bounds3 bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
      const LightBounds& lb = bvhLights[i].second;
      bounds = unionBounds(bounds, lb.bounds);
      centroidBounds = unionPoint(centroidBounds, point3(lb.centroid()));
    }

    float minCost = std::numeric_limits<float>::max();
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;
    for (int dim = 0; dim < 3; ++dim) {
      // Compute minimum cost bucket for splitting along dimension _dim_
      if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
        continue;
      // Compute _LightBounds_ for each bucket
      LightBounds bucketLightBounds[nBuckets];
      for (int i = start; i < end; ++i) {
        point3 pc = bvhLights[i].second.centroid();
        int b = nBuckets * centroidBounds.offset(pc)[dim];
        if (b == nBuckets)
          b = nBuckets - 1;
        bucketLightBounds[b] = union_light_bounds(bucketLightBounds[b], bvhLights[i].second);
      }

      // Compute costs for splitting lights after each bucket
      float cost[nBuckets - 1];
      for (int i = 0; i < nBuckets - 1; ++i) {
        // Find _LightBounds_ for lights below and above bucket split
        LightBounds b0, b1;
        for (int j = 0; j <= i; ++j)
          b0 = union_light_bounds(b0, bucketLightBounds[j]);
        for (int j = i + 1; j < nBuckets; ++j)
          b1 = union_light_bounds(b1, bucketLightBounds[j]);

        // Compute final light split cost for bucket
        cost[i] = evaluateCost(b0, bounds, dim) + evaluateCost(b1, bounds, dim);
      }

      // Find light split that minimizes SAH metric
      for (int i = 1; i < nBuckets - 1; ++i) {
        if (cost[i] > 0 && cost[i] < minCost) {
          minCost = cost[i];
          minCostSplitBucket = i;
          minCostSplitDim = dim;
        }
      }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1)
      mid = (start + end) / 2;
    else {
      const auto* pmid = std::partition(
        &bvhLights[start], &bvhLights[end - 1] + 1,
        [=](const std::pair<int, LightBounds>& l) {
          int b = nBuckets *
          centroidBounds.offset(l.second.centroid())[minCostSplitDim];
      if (b == nBuckets)
        b = nBuckets - 1;
      return b <= minCostSplitBucket;
        });
      mid = pmid - &bvhLights[0];
      if (mid == start || mid == end)
        mid = (start + end) / 2;
    }

    // Allocate interior _LightBVHNode_ and recursively initialize children
    int nodeIndex = nodes.size();
    nodes.push_back(LightBVHNode());
    std::pair<int, LightBounds> child0 =
      build_bvh(bvhLights, start, mid, bitTrail, depth + 1);
    std::pair<int, LightBounds> child1 =
      build_bvh(bvhLights, mid, end, bitTrail | (1u << depth), depth + 1);

    // Initialize interior node and return node index and bounds
    LightBounds lb = union_light_bounds(child0.second, child1.second);
    CompactLightBounds cb(lb, allLightBounds);
    nodes[nodeIndex] = LightBVHNode::makeInterior(child1.first, cb);
    return { nodeIndex, lb };
  }

  // evaluate the cost model for the two LightBounds for each split candidate
  auto BVHLightSampler::evaluateCost(const LightBounds& b,
    const bounds3& bounds, int dim) const -> float {
    // Evaluate direction bounds measure for LightBounds
    float theta_o = std::acos(b.cosTheta_o), theta_e = std::acos(b.cosTheta_e);
    float theta_w = std::min(theta_o + theta_e, M_FLOAT_PI);
    float sinTheta_o = safe_sqrt(1 - sqr(b.cosTheta_o));
    float M_omega = 2 * M_FLOAT_PI * (1 - b.cosTheta_o) +
      M_FLOAT_PI / 2 * (2 * theta_w * sinTheta_o - std::cos(theta_o - 2 * theta_w) -
        2 * theta_o * sinTheta_o + b.cosTheta_o);
    // Return complete cost estimate for LightBounds
    float Kr = maxComponent(bounds.diagonal()) / bounds.diagonal()[dim];
    return b.phi * M_omega * Kr * b.bounds.surfaceArea();
  }

  static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;

  inline auto SampleDiscrete(float weights[2], float u, float& pmf, float& uRemapped) -> int {
    return 1;
  }

  auto Scene::update_gpu_lightbvh() noexcept -> void {
    if (m_perSceneGPUInfo.lightSampler.sampler == nullptr) {
      m_perSceneGPUInfo.lightSampler.sampler = std::make_unique<BVHLightSampler>();
    }
    BVHLightSampler* sampler = (BVHLightSampler*)m_perSceneGPUInfo.lightSampler.sampler.get();
    sampler->allLightBounds.pMin = vec3(+1e9);
    sampler->allLightBounds.pMax = vec3(-1e9);

    std::vector<std::pair<int, LightBounds>> bvhLights;
    for (auto& entity_lights : m_gpuScene.lightList) {
      for (auto& lights_index : entity_lights.second) {
        for (size_t i = 0; i < lights_index.length; ++i) {
          int32_t index = i + lights_index.assignedIndex;
          // Store th light in either infiniteLights or bvhLights
          LightData light = m_gpuScene.lightBuffer[index];
          // handle light bounds
          LightBounds lb;
          lb.bounds = { light.floatvec_1.xyz(), light.floatvec_2.xyz() };
          lb.phi = light.floatvec_0.x;
          lb.w = { light.floatvec_0.w, light.floatvec_1.w, light.floatvec_2.w };
          lb.cosTheta_o = 1;
          lb.rgb = light.floatvec_0.xyz();
          lb.cosTheta_e = std::cos(M_FLOAT_PI / 2);
          lb.twoSided = false;

          if (lb.phi > 0) {
            bvhLights.push_back(std::make_pair(index, lb));
            sampler->allLightBounds = union_bounds(sampler->allLightBounds, lb.bounds);
          }
        }
      }
    }

    sampler->lightToBitTrail.resize(bvhLights.size());
    if (!bvhLights.empty())
      sampler->build_bvh(bvhLights, 0, bvhLights.size(), 0, 0);

    m_perSceneGPUInfo.lightSampler.treeBuffer->m_host.resize(sampler->nodes.size() * sizeof(LightBVHNode));
    memcpy((LightBVHNode*)m_perSceneGPUInfo.lightSampler.treeBuffer->m_host.data(), sampler->nodes.data(),
      sampler->nodes.size() * sizeof(LightBVHNode));

    m_perSceneGPUInfo.lightSampler.trailBuffer->m_host.resize(sampler->lightToBitTrail.size() * sizeof(uint32_t));
    memcpy((LightBVHNode*)m_perSceneGPUInfo.lightSampler.trailBuffer->m_host.data(), sampler->lightToBitTrail.data(),
      sampler->lightToBitTrail.size() * sizeof(uint32_t));

    m_perSceneGPUInfo.lightSampler.treeBuffer->m_hostStamp++;
    m_perSceneGPUInfo.lightSampler.trailBuffer->m_hostStamp++;
    m_perSceneGPUInfo.lightSampler.treeBuffer->host_to_device();
    m_perSceneGPUInfo.lightSampler.trailBuffer->host_to_device();
    m_perSceneGPUInfo.lightSampler.allLightBounds = sampler->allLightBounds;
  }
}
}