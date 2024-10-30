#define SIByL_API __declspec(dllimport)
#include <se.editor.hpp>
#include <se.gfx.hpp>
#undef SIByL_API
#define SIByL_API __declspec(dllexport)
#include <seditor-base.hpp>
#undef SIByL_API
#include <imgui.h>

namespace se::editor {  
  auto EditorContext::initialize() noexcept -> void {
    ImGui::SetCurrentContext(se::editor::ImGuiContext::imContext);
  }

  auto EditorContext::onDrawGui() noexcept -> void {
    for (auto& iter : widgets) {
      iter.second->onDrawGui();
    }
  }

  std::unordered_map<std::string, std::unique_ptr<Widget>> EditorContext::widgets = {};

  
  auto EditorBase::onImGuiDraw() noexcept -> void {
    inspectorWidget.onDrawGui();
    sceneWidget.onDrawGui();
    viewportWidget.onDrawGui();
    rdgViewerWidget.onDrawGui();
    statusWidget.onDrawGui();
  }
  
  auto EditorBase::onUpdate() noexcept -> void {
    viewportWidget.onUpdate();
    if (se::editor::EditorBase::sceneWidget.scene.get() != nullptr)
    se::editor::EditorBase::sceneWidget.scene->editorInfo.active_camera_index 
      = se::editor::EditorBase::viewportWidget.camera_index;
  }

  auto EditorBase::finalize() noexcept -> void {
    se::editor::EditorBase::sceneWidget.scene = {};
    se::editor::EditorBase::viewportWidget.texture = {};
  }
  
  auto EditorBase::bindInput(se::input* input) noexcept -> void {
    viewportWidget.bindInput(input);
  }
  
  auto EditorBase::fetchInput() noexcept -> se::input* {
    return viewportWidget.controller.input;
  }

  auto EditorBase::bindTimer(se::timer* timer) noexcept -> void {
    viewportWidget.bindTimer(timer);
    statusWidget.timer = timer;
  }
  
  auto EditorBase::bindPipeline(se::rdg::Pipeline* pipeline) noexcept -> void {
    viewportWidget.texture = pipeline->getOutput();
    rdgViewerWidget.pipeline = pipeline;
  }
  
  auto EditorBase::bindScene(se::gfx::SceneHandle scene) noexcept -> void {
    // bind the scene to scene pannel
    getSceneWidget()->bindScene(scene);
    // bind the viewport camera to the scene
    scene->useEditorCameraView(
      &viewportWidget.editor_camera_transform, 
      &viewportWidget.editor_camera);
  }

  InspectorWidget EditorBase::inspectorWidget;
  SceneWidget EditorBase::sceneWidget;
  ViewportWidget EditorBase::viewportWidget;
  RDGViewerWidget EditorBase::rdgViewerWidget;
  StatusWidget EditorBase::statusWidget;


  auto EditorBase::getInspectorWidget() noexcept -> InspectorWidget* { return &inspectorWidget; }
  auto EditorBase::getSceneWidget() noexcept -> SceneWidget* { return &sceneWidget; }
  auto EditorBase::getViewportWidget() noexcept -> ViewportWidget* { return &viewportWidget; }
  auto EditorBase::getRDGViewerWidget() noexcept -> RDGViewerWidget* { return &rdgViewerWidget; }
  auto EditorBase::getStatusWidget() noexcept -> StatusWidget* { return &statusWidget; }
}