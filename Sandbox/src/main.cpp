#include <SE.GFX-Utils.h>
#include <crtdbg.h>
#include <glad/glad.h>
#include <imgui.h>
#include <tinygltf/tiny_gltf.h>

#include <IO/SE.Core.IO.hpp>
#include <Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarGTPass.hpp>
#include <Pipeline/SE.SRendere-ForwardPipeline.hpp>
#include <Pipeline/SE.SRendere-GeoInspectPipeline.hpp>
#include <Pipeline/SE.SRendere-RTGIPipeline.hpp>
#include <Pipeline/SE.SRendere-SSRXPipeline.hpp>
#include <Pipeline/SE.SRendere-UDPTPipeline.hpp>
#include <Plugins/SE.SRendererExt.GeomTab.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Addon.SLC.hpp>
#include <SE.Addon.SST.hpp>
#include <SE.Addon.VPL.hpp>
#include <SE.Addon.SSPM.hpp>
#include <SE.Addon.PSFiltering.hpp>
#include <SE.Addon.Fluid.hpp>
#include <SE.Application.hpp>
#include <SE.Editor.Config.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Editor.DebugDraw.hpp>
#include <SE.RHI.Profiler.hpp>
#include <SE.SRenderer.hpp>
#include <SE.Addon.Lightmap.hpp>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <stack>
#include <typeinfo>
#include <SE.GFX-SceneUpdate.h>

#include "CustomPipeline.hpp"

using namespace SIByL::Math;

struct SandBoxApplication : public Application::ApplicationBase {
  Video::VideoDecoder decoder;

  /** Initialize the application */
  virtual auto Init() noexcept -> void override {
    // create optional layers: rhi, imgui, editor
    rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
        RHI::RHIBackend::Vulkan,
        RHI::ContextExtensionsFlags(
            RHI::ContextExtension::RAY_TRACING |
            RHI::ContextExtension::BINDLESS_INDEXING |
            RHI::ContextExtension::FRAGMENT_BARYCENTRIC |
            RHI::ContextExtension::CONSERVATIVE_RASTERIZATION |
            RHI::ContextExtension::COOPERATIVE_MATRIX |
            RHI::ContextExtension::ATOMIC_FLOAT),
        mainWindow.get(),
        true,  // use imgui
        true,  // use aftermath
    });
    GFX::GFXManager::get()->rhiLayer = rhiLayer.get();
    GFX::GFXManager::get()->mTimeline.timer = &timer;
    imguiLayer = std::make_unique<Editor::ImGuiLayer>(rhiLayer.get());
    editorLayer = std::make_unique<Editor::EditorLayer>();
    Editor::Config::buildEditorLayer(editorLayer.get());

    // bind editor layer
    editorLayer->getWidget<Editor::SceneWidget>()->bindScene(&scene);
    editorLayer->getWidget<Editor::StatusWidget>()->timer = &timer;

    RHI::Device* device = rhiLayer->getDevice();
    RHI::SwapChain* swapchain = rhiLayer->getSwapChain();

    Singleton<RHI::DeviceProfilerManager>::instance()->initialize(
        device, "device_something");

    GFX::GFXManager::get()->config.meshLoaderConfig = SRenderer::meshLoadConfig;

    //scene.deserialize("../Sandbox/scenes/ajar.scene");
    //scene.deserialize("../Sandbox/scenes/fireplace.scene");
    scene.deserialize("../Sandbox/scenes/staircase.scene");
    
    InvalidScene();
    device->waitIdle();

    pipeline1 = std::make_unique<GTPipeline>();
    //pipeline1 = std::make_unique<SRP::GeoInspectPipeline>();
    //pipeline1 = std::make_unique<GTPipeline>();
    ////pipeline2 = std::make_unique<Addon::Lightmap::LightmapVisualizePipeline>();
    ////rtgi_pipeline = std::make_unique<Addon::Differentiable::NeuralRadiosityPipeline>();
    ////// pipeline1 = std::make_unique<CustomPipeline>();
    ////// pipeline2 = std::make_unique<VXPGReSTIRPipeline>();
    ////// pipeline2 = std::make_unique<SSPGReSTIRPipeline>();
    //pipeline2 = std::make_unique<Addon::SSPM::SSPMGPipeline>();
    ////pipeline2 = std::make_unique<Addon::Fluid::LBMPipeline>();
    //pipeline2 = std::make_unique<Addon::SLC::SLCTestPipeline>();
    //pipeline2 = std::make_unique<RestirGIPipeline>();
    //pipeline2 = std::make_unique<Addon::SLC::SLCTestPipeline>();
    pipeline2 = std::make_unique<SSPGP_GMM_Pipeline>();
    //////rtgi_pipeline = std::make_unique<Addon::VXGuiding::GeometryPrebakePipeline>();
    ////rtgi_pipeline = std::make_unique<Addon::SST::SSTTestPipeline>();
    //////rtgi_pipeline = std::make_unique<VXPGASVGFPipeline>();
    ////////      rtgi_pipeline = std::make_unique<RestirGIPipeline>();
    //////rtgi_pipeline = std::make_unique<RestirGIPipeline>();

    // //geoinsp_pipeline = std::make_unique<SSPGP_GMM_Pipeline>();
    ////geoinsp_pipeline = std::make_unique<GTPipeline>();
    geoinsp_pipeline = std::make_unique<SRP::GeoInspectPipeline>();
    //geoinsp_pipeline = std::make_unique<CustomPipeline>();
    //vxgi_pipeline = std::make_unique<SSPGPipeline>();
    //////vxdi_pipeline = std::make_unique<VXPGPipeline>();
    rtgi_pipeline = std::make_unique<VXPGReSTIRPipeline>();
    vxgi_pipeline = std::make_unique<VXPGASVGFPipeline>();
    vxdi_pipeline = std::make_unique<VXPGPipeline>();

    pipeline1->build();
    pipeline2->build();
    geoinsp_pipeline->build();
    rtgi_pipeline->build();
    vxgi_pipeline->build();
    vxdi_pipeline->build();


    pipeline = pipeline1.get();

    Editor::DebugDraw::Init(pipeline->getOutput(), nullptr);

    editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;

    // std::unique_ptr<Image::Texture_Host> dds_tex =
    // Image::DDS::fromDDS("D:/Art/Scenes/Bistro_v5_2/Bistro_v5_2/Textures/Bollards_BaseColor.dds");
    // Core::GUID guid_dds =
    // Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    // GFX::GFXManager::get()->registerTextureResource(guid_dds, dds_tex.get());

    cameraController.init(mainWindow.get()->getInput(), &timer,
                          editorLayer->getWidget<Editor::ViewportWidget>());
  };

  void InvalidScene() {
    srenderer = std::make_unique<SRenderer>();
    srenderer->init(scene);
    srenderer->timer = &timer;
  }

  /** Update the application every loop */
  virtual auto Update(double deltaTime) noexcept -> void override {
    // start new frame
    imguiLayer->startNewFrame();
    Editor::DebugDraw::Clear();
      RHI::Device* device = rhiLayer->getDevice();
    RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
    RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
    { 
      PROFILE_SCOPE("wait_idle");
      multiFrameFlights->frameStart();
      device->waitIdle();
    }
    {
      PROFILE_SCOPE("gfx::onupdate");
      GFX::GFXManager::get()->onUpdate();
    }

    if (scene.isDirty == true) {
      cameraController.forceReset = true;
    }
    // update camera
    {
      PROFILE_SCOPE("update_camera");
      int width = 1280; int height = 720;
      auto view = Core::ComponentManager::get()->view<GFX::CameraComponent>();
      for (auto& [entity, camera] : view) {
        if (camera.isPrimaryCamera) {
          camera.aspect = 1.f * width / height;
          editorLayer->getWidget<Editor::ViewportWidget>()->camera = &camera;
          cameraController.bindTransform(
              scene.getGameObject(entity)
                  ->getEntity()
                  .getComponent<GFX::TransformComponent>());
          srenderer->updateCamera(
              *(scene.getGameObject(entity)
                    ->getEntity()
                    .getComponent<GFX::TransformComponent>()),
              camera, {width, height});
          break;
        }
      }
      cameraController.onUpdate();
      // Core::LogManager::Log(std::to_string(timer.deltaTime()));
    }
    {
      PROFILE_SCOPE("update anim/transform");
      GFX::update_animation(scene);
      GFX::update_transforms(scene);
    }
    
    std::unique_ptr<RHI::CommandEncoder> commandEncoder =
        device->createCommandEncoder({multiFrameFlights->getCommandBuffer()});

    if (scene.isDirty == true) {
      device->waitIdle();
      InvalidScene();
      device->waitIdle();
      scene.isDirty = false;
    } else {
      PROFILE_SCOPE("invalid scene");
      srenderer->invalidScene(scene);
    }
    
    // GUI Recording

    static int frames2capture = 0;
    imguiLayer->startGuiRecording();
    bool show_demo_window = true;
    ImGui::ShowDemoWindow(&show_demo_window);
    {
      ImGui::Begin("Device-Profiler");
      RHI::DeviceProfilerManager* profiler =
          Singleton<RHI::DeviceProfilerManager>::instance();
      ImGui::Checkbox("Enable Profiler", &profiler->enabled);
      bool clearProfiler = false;
      ImGui::Checkbox("Clear Profiler", &clearProfiler);
      if (clearProfiler) {
        profiler->clear();
      }
      if (profiler->enabled) {
        profiler->flushResults();
        profiler->reset(commandEncoder.get());
      }
      for (auto& pair : profiler->statistics) {
        double time =
            double(pair.second.accumulation) / (pair.second.count * 1000000);
        if (ImGui::TreeNode(
                (pair.first + ": " + std::to_string(time)).c_str())) {
          ImGui::TreePop();
        }
      }
      ImGui::End();
    }

    {
      ImGui::Begin("Host Profiler");
      if (ImGui::Button("Begin")) {
        SIByL::Core::ProfileSession::get().beginSession("hello");
      }
      if (ImGui::Button("End")) {
        SIByL::Core::ProfileSession::get().endSession();
      }
      ImGui::End();
    }

    static int pipeline_id = 0;
    ImGui::Begin("Pipeline Choose");
    {  // Select an item type
      const char* item_names[] = {"Path Tracing",   "SSPG",
                                  "Geo Inspector",  "VXPG + ReSTIR",
                                  "VXPG + ASVGF",   "Vanilla VXPG"};
      bool reselect = ImGui::Combo("Mode", &pipeline_id, item_names, IM_ARRAYSIZE(item_names),
                   IM_ARRAYSIZE(item_names));
      if (reselect) {
        frames2capture = 6;
      }
      if (pipeline_id == 0) {
        pipeline = pipeline1.get();
        editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;
      } else if (pipeline_id == 1) {
        pipeline = pipeline2.get();
        editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;
      } else if (pipeline_id == 2) {
        pipeline = geoinsp_pipeline.get();
        editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;
      } else if (pipeline_id == 3) {
        pipeline = rtgi_pipeline.get();
        editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;
      } else if (pipeline_id == 4) {
        pipeline = vxgi_pipeline.get();
        editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;
      } else if (pipeline_id == 5) {
        pipeline = vxdi_pipeline.get();
        editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;
      }
    }
    ImGui::End();

    std::vector<RDG::Graph*> graphs = pipeline->getActiveGraphs();

    for (auto* graph : graphs) srenderer->updateRDGData(graph);

    for (auto* graph : graphs)
      for (size_t i : graph->getFlattenedPasses()) {
        graph->getPass(i)->onInteraction(
            mainWindow.get()->getInput(),
            &(editorLayer->getWidget<Editor::ViewportWidget>()->info));
      }
    { PROFILE_SCOPE("execute pipeline");
      Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
          commandEncoder.get(), RHI::PipelineStages::TOP_OF_PIPE_BIT,
          "total_pipe");
      pipeline->execute(commandEncoder.get());

      Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
          commandEncoder.get(), RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
          "total_pipe");
    }
    // Editor::DebugDraw::DrawAABB(srenderer->statisticsData.aabb, 5., 5.);
    // auto editor_graphs =
    // Editor::DebugDraw::get()->pipeline->getActiveGraphs(); for (auto* graph :
    // editor_graphs) 	srenderer->updateRDGData(graph);
    // Editor::DebugDraw::Draw(commandEncoder.get());

    device->getGraphicsQueue()->submit(
        {commandEncoder->finish({})},
        multiFrameFlights->getImageAvailableSeamaphore(),
        multiFrameFlights->getRenderFinishedSeamaphore(),
        multiFrameFlights->getFence());

    editorLayer->getWidget<Editor::ViewportWidget>()->setTarget(
        "Main Viewport", pipeline->getOutput());
    editorLayer->onDrawGui();
    imguiLayer->render(multiFrameFlights->getRenderFinishedSeamaphore());

    multiFrameFlights->frameEnd();

    pipeline->readback();
    if (frames2capture > 0) {
      auto& timeline = GFX::GFXManager::get()->mTimeline;
      int currentFrame = timeline.currentSec * timeline.step_per_sec;
      static bool should_capture_next = false;
      if (currentFrame == 155) {
      //if (currentFrame == 105) {
        should_capture_next = false;
        device->waitIdle();
        GFX::CaptureImage(pipeline->getOutput(),
            "D:/Art/Objects/hip_hop_dancing_women_gltf/pipeline-new-" +
            //"D:/Art/Scenes/BrainStem-gltf/glTF/dynamic_capture/pipeline-" +
            //"D:/Art/Scenes/BrainStem-gltf/glTF/zeroday/pipeline-" +
            std::to_string(pipeline_id));
        frames2capture = 0;
      }
     //   if (frames2capture == 1) {
     //       device->waitIdle();
     //       GFX::CaptureImage(pipeline->getOutput(),
     //           "D:/data/adaptation/new/bsdf-" + std::to_string(pipeline_id));            
     //   }
    	//frames2capture--;
    }
  };

  /** Update the application every fixed update timestep */
  virtual auto FixedUpdate() noexcept -> void override{

  };

  virtual auto Exit() noexcept -> void override {
    decoder.close();

    rhiLayer->getDevice()->waitIdle();
    srenderer = nullptr;
    pipeline1 = nullptr;
    pipeline2 = nullptr;
    rtgi_pipeline = nullptr;
    geoinsp_pipeline = nullptr;
    vxdi_pipeline = nullptr;
    vxgi_pipeline = nullptr;
    Singleton<RHI::DeviceProfilerManager>::instance()->finalize();
    Editor::DebugDraw::Destroy();

    editorLayer = nullptr;
    imguiLayer = nullptr;

    Core::ResourceManager::get()->clear();

    rhiLayer = nullptr;
  }

 private:
  Core::GUID rtTarget;

  uint32_t indexCount = 0;

  std::unique_ptr<RHI::RHILayer> rhiLayer = nullptr;
  std::unique_ptr<Editor::ImGuiLayer> imguiLayer = nullptr;
  std::unique_ptr<Editor::EditorLayer> editorLayer = nullptr;
  std::unique_ptr<SRenderer> srenderer = nullptr;

  RDG::Pipeline* pipeline = nullptr;
  std::unique_ptr<RDG::Pipeline> pipeline1 = nullptr;
  std::unique_ptr<RDG::Pipeline> pipeline2 = nullptr;
  std::unique_ptr<RDG::Pipeline> geoinsp_pipeline = nullptr;
  std::unique_ptr<RDG::Pipeline> rtgi_pipeline = nullptr;
  std::unique_ptr<RDG::Pipeline> vxgi_pipeline = nullptr;
  std::unique_ptr<RDG::Pipeline> vxdi_pipeline = nullptr;

  // the embedded scene, which should be removed in the future
  GFX::Scene scene;

  Editor::SimpleCameraController cameraController;
  GFX::TransformComponent cameraTransform;
};

int main() {
  // application root, control all managers
  Application::Root root;

  // run app
  SandBoxApplication app;
  app.createMainWindow({Platform::WindowVendor::GLFW, L"SIByL Engine 2023.1",
                        1920, 1080, Platform::WindowProperties::VULKAN_CONTEX});
  app.run();
}