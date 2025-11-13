#pragma once
#include "se.rdg.hpp"
#include "../../source/se.editor.helper.hpp"

using namespace se;
struct InspectorPass : public rdg::ComputePass {
	struct PushConstant {
		int displayID = 0;
		int randomSeed = 0;
		int mouseState = 0;
		int hightlightGeometry = -1;
		ivec2 mousePixelX;
	} pConst;

	struct Interaction {
		int32_t     primitiveType;
		uint32_t    primitiveID;
		uint32_t    geometryID;
		uint32_t    padding;
		vec2			  barycentrics;
	};
	Interaction* m_interaction;

	struct Helper {
		int releaseCountDown = -1;
	} m_helper;

	// initialize pass, by defining the shader
	InspectorPass();

	virtual auto reflect(rdg::PassReflection& reflector) noexcept -> rdg::PassReflection;

	virtual auto execute(
		rdg::RenderContext* rdrCtx,
		rdg::RenderData const& rdrDat
	) noexcept -> void;

	virtual auto render_ui() noexcept -> void;
};

struct SecondaryInspectorPass : public rdg::ComputePass {
	struct PushConstant {
		int displayID = 0;
		int randomSeed = 0;
	} pConst;
	
	// initialize pass, by defining the shader
	SecondaryInspectorPass() { init("./shaders/editor/geometry-viewer-2nd.slang"); }

	virtual auto reflect(rdg::PassReflection& reflector) noexcept -> rdg::PassReflection;

	virtual auto execute(
		rdg::RenderContext* rdrCtx,
		rdg::RenderData const& rdrDat
	) noexcept -> void;

	virtual auto render_ui() noexcept -> void;
};