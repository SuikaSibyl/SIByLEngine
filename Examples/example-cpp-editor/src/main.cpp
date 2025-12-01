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
	foo_graph->m_standardSize = { 512,512,1 };
	foo_graph->build();
	PROFILE_SCOPE_STOP(InitRenderGraph);

	// build the scene
	PROFILE_SCOPE_NAME(InitScene);

	gfx::SceneBatchHandle sceneBatch = gfx::GFXContext::create_scene_batch();
	
	std::vector<std::string> scene_names = {
		"//home/haolin/Projects/gilo/prt/_pretrain/buddha/scene/buddha-simple.gltf",
		"/home/haolin/Projects/neural_variance_reduction/example-gi/data/grid_gi_1.glb",
		"/home/haolin/Projects/neural_variance_reduction/example-gi/data/grid_gi_2.glb",
		"/home/haolin/Projects/neural_variance_reduction/example-gi/data/grid_gi_3.glb",
	};
	gfx::SceneHandle scene = gfx::GFXContext::load_scene_gltf(scene_names[0]);
	scene->set_viewport_size({ 256, 256 });
	editor::EditorContext::set_scene_display(scene);
	editor::EditorContext::set_graph_display(foo_graph.get());
	scene->update_gpu_scene();
	
	// sceneBatch->update_gpu_scene_batch();
	// scene->update_gpu_scene();
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
		// scene->update_scripts();
		// scene->update_gpu_scene();

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
	sceneBatch = nullptr;

	// release the context
	se::editor::EditorContext::finalize();
	se::gfx::GFXContext::finalize();
	window.destroy();
}