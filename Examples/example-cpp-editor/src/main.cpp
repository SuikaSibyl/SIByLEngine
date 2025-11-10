#include <spdlog/spdlog.h>
#include "se.utils.hpp"
#include "se.math.hpp"
#include "se.rhi.hpp"
#include "se.gfx.hpp"
#include "se.editor.hpp"
#include "se.rdg.hpp"
#include <../addon/pass-editor/ex.pass.editor.hpp>
#include <../addon/pass-postprocess/ex.pass.postprocess.hpp>

using namespace se;

struct BarPass : public se::rdg::RenderPass {
	// initialize pass, by defining the shader
	BarPass() { init("./shaders/editor/geometry-viewer.slang"); }

	virtual auto reflect(rdg::PassReflection& reflector) noexcept -> se::rdg::PassReflection {
		reflector.add_output("Color").is_texture()
			.with_format(se::rhi::TextureFormat::RGBA32_FLOAT)
			.consume_as_color_attachment_at(0);
		reflector.add_output("Depth").is_texture()
			.with_format(se::rhi::TextureFormat::DEPTH32_FLOAT)
			.consume_as_depth_stencil_attachment_at(0);
		return reflector;
	}

	virtual auto execute(
		se::rdg::RenderContext* rdrCtx,
		se::rdg::RenderData const& rdrDat
	) noexcept -> void {
		set_render_pass_descriptor({
			{se::rhi::RenderPassColorAttachment{rdrDat.get_texture("Color")->get_rtv(0, 0, 1),
				{0, 0, 0, 1}, se::rhi::LoadOp::CLEAR, se::rhi::StoreOp::STORE}},
			se::rhi::RenderPassDepthStencilAttachment{ rdrDat.get_texture("Depth")->get_dsv(0, 0, 1),
				1, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE, false,
				0, rhi::LoadOp::DONT_CARE, rhi::StoreOp::DONT_CARE, false,
			}
		});

		auto scene = rdrDat.get_scene();
		update_binding_scene(rdrCtx, scene);

		auto encoder = begin_pass(rdrCtx, rdrDat.get_texture("Color").get());
		scene->draw_meshes(encoder);
		encoder->end();
	}
};

struct FooGraph: public rdg::Graph {
	InspectorPass foo_pass;
	SecondaryInspectorPass sec_pass;
	AccumulatePass accum_pass;
	AccumulatePass accum_2nd_pass;

	FooGraph() {
		add_pass(&foo_pass, "Foo Pass");
		add_pass(&accum_pass, "Accum Pass");
		add_edge("Foo Pass", "Color", "Accum Pass", "Input");
		mark_output("Accum Pass", "Output");
	}
};

int main() {
	se::Configuration::set_config_file(se::Filesys::get_parent_path(__FILE__) + "/../runtime.config");

	PROFILE_BEGIN_SESSION("Init", "/profile/init.profile");

	// build the context
	PROFILE_SCOPE_NAME(InitContext);
	se::Window window = se::Window(1280, 720, L"Hello, World!");
	se::gfx::GFXContext::initialize(&window,
		(se::rhi::ContextExtensionEnum::DEBUG_UTILS
	 | se::rhi::ContextExtensionEnum::CUDA_INTEROPERABILITY
	 | se::rhi::ContextExtensionEnum::USE_AFTERMATH
	 | se::rhi::ContextExtensionEnum::FRAGMENT_BARYCENTRIC
	 | se::rhi::ContextExtensionEnum::COOPERATIVE_MATRIX
	 | se::rhi::ContextExtensionEnum::RAY_TRACING));
	se::editor::EditorContext::initialize();
	se::rhi::Device* device = se::gfx::GFXContext::device();
	PROFILE_SCOPE_STOP(InitContext);

	// build the render graph
	PROFILE_SCOPE_NAME(InitRenderGraph);
	std::unique_ptr<FooGraph> foo_graph = std::make_unique<FooGraph>();
	foo_graph->m_standardSize = { 1024,1024,1 };
	foo_graph->build();
	PROFILE_SCOPE_STOP(InitRenderGraph);

	// build the scene
	PROFILE_SCOPE_NAME(InitScene);
	gfx::SceneHandle scene = gfx::GFXContext::load_scene_gltf("/home/haolin/Projects/neural_variance_reduction/example-di/data/room.glb");
	editor::EditorContext::set_scene_display(scene);
	editor::EditorContext::set_graph_display(foo_graph.get());
	scene->update_gpu_scene();
	PROFILE_SCOPE_STOP(InitScene);

	PROFILE_END_SESSION()

	// main loop
	while (window.is_running()) {
		window.fetch_events();
		if (window.is_resized() || se::editor::ImGuiContext::need_recreate()) {
			if (window.get_width() == 0 || window.get_height() == 0) continue;
			se::editor::ImGuiContext::recreate(window.get_width(), window.get_height());
		}
		if (window.is_iconified()) continue;

		se::gfx::GFXContext::get_flights()->frame_start();
		se::editor::ImGuiContext::start_new_frame();

		// Updating
		// -------------------------------------
		scene->update_scripts();
		scene->update_gpu_scene();

		// create a command encoder
		std::unique_ptr<se::rhi::CommandEncoder> encoder = device->
			create_command_encoder(se::gfx::GFXContext::get_flights()->get_command_buffer());

		foo_graph->m_renderData.set_scene(scene);
		foo_graph->execute(encoder.get());

		auto output = foo_graph->get_output();
		if (output.has_value()) {
			editor::EditorContext::set_viewport_texture(output.value());
		}

		//	# start record the gui
		se::editor::EditorContext::begin_frame(encoder.get());

		//	# submit the command
		device->get_graphics_queue().submit(
			{ encoder->finish() },
			se::gfx::GFXContext::get_flights()->get_image_available_semaphore(),
			se::gfx::GFXContext::get_flights()->get_render_finished_semaphore(),
			se::gfx::GFXContext::get_flights()->get_fence());


		se::editor::EditorContext::end_frame(se::gfx::GFXContext::get_flights()->get_render_finished_semaphore());
		se::gfx::GFXContext::frame_end();
	}

	// release the scene and graph
	device->wait_idle();
	foo_graph = nullptr;
	scene.release();

	// release the context
	se::editor::EditorContext::finalize();
	se::gfx::GFXContext::finalize();
	window.destroy();
}