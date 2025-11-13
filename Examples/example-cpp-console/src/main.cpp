#include "se.utils.hpp"
#include "se.math.hpp"
#include "se.rhi.hpp"
#include "se.gfx.hpp"
#include "se.editor.hpp"
#include "se.rdg.hpp"

int main() {
    se::Configuration::set_config_file(se::Filesys::get_parent_path(__FILE__) + "/../runtime.config");

	se::gfx::GFXContext::initialize(nullptr,
		(se::rhi::ContextExtensionEnum::DEBUG_UTILS
	 | se::rhi::ContextExtensionEnum::CUDA_INTEROPERABILITY
	 | se::rhi::ContextExtensionEnum::USE_AFTERMATH
	 | se::rhi::ContextExtensionEnum::COOPERATIVE_MATRIX
	 | se::rhi::ContextExtensionEnum::RAY_TRACING));

    se::rhi::Device* device = se::gfx::GFXContext::device();

    // create a scene batch
    {
        se::gfx::SceneBatchHandle sceneBatch = se::gfx::GFXContext::create_scene_batch();

        auto scene_names = std::vector<std::string>{
            "/home/haolin/Projects/neural_variance_reduction/example-di/data/grid_di_0.glb",
            "/home/haolin/Projects/neural_variance_reduction/example-di/data/grid_di_1.glb",
            "/home/haolin/Projects/neural_variance_reduction/example-di/data/grid_di_2.glb",
            "/home/haolin/Projects/neural_variance_reduction/example-di/data/grid_di_3.glb",
        };
        for (int i = 0; i < 3; ++i) {
        se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf(scene_names[i]);
        sceneBatch->emplace_scene(scene);
        }

        sceneBatch->update_gpu_scene_batch();
    }
	se::gfx::GFXContext::finalize();
    return 0;
}