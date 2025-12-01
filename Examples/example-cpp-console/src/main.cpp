#include "se.utils.hpp"
#include "se.math.hpp"
#include "se.rhi.hpp"
#include "se.gfx.hpp"
#include "se.editor.hpp"
#include "se.rdg.hpp"
#include <spdlog/spdlog.h>

int main() {
    se::Configuration::set_config_file(se::Filesys::get_parent_path(__FILE__) + "/../runtime.config");

	se::gfx::GFXContext::initialize(nullptr,
		(se::rhi::ContextExtensionEnum::DEBUG_UTILS
	 | se::rhi::ContextExtensionEnum::CUDA_INTEROPERABILITY
	 | se::rhi::ContextExtensionEnum::USE_AFTERMATH
	 | se::rhi::ContextExtensionEnum::COOPERATIVE_MATRIX
	 | se::rhi::ContextExtensionEnum::RAY_TRACING));

    se::rhi::Device* device = se::gfx::GFXContext::device();
    for (int i = 0; i < 5; ++i) {
      se::gfx::SceneBatchHandle sceneBatch = se::gfx::GFXContext::create_scene_batch();

      for (int j = 0; j < 4; ++j) {
        std::string scene_name = "/home/haolin/Projects/neural_variance_reduction/example-di/data/grid_di_" + std::to_string(i + j) + ".glb";
        se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf(scene_name);
        scene->invalid_gpu_resources();
        sceneBatch->emplace_scene(scene);
      }

      sceneBatch->update_gpu_scene_batch();
      se::gfx::GFXContext::clean_cache();
      const int buffer_size = se::gfx::GFXContext::number_of_cached_buffers();
      const int scene_size = se::Singleton<se::gfx::GFXContext>::instance()->m_scenes.size();
      se::info("Number of cached buffers: " + std::to_string(buffer_size) + ", number of scenes: " + std::to_string(scene_size));
    }
	se::gfx::GFXContext::finalize();
    return 0;
}