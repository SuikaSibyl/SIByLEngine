#pragma once
#include "seditor.hpp"
#include "seditor-gfx.hpp"
#include "seditor-rdg.hpp"

namespace se::editor {
struct SIByL_API EditorBase {
  static InspectorWidget	inspectorWidget;
  static SceneWidget		sceneWidget;
  static ViewportWidget		viewportWidget;
  static RDGViewerWidget	rdgViewerWidget;
  static StatusWidget		statusWidget;
  static auto onImGuiDraw() noexcept -> void;
  static auto onUpdate() noexcept -> void;
  static auto finalize() noexcept -> void;
  static auto bindInput(se::input* input) noexcept -> void;
  static auto fetchInput() noexcept -> se::input*;
  static auto bindTimer(se::timer* timer) noexcept -> void;
  static auto bindPipeline(se::rdg::Pipeline* pipeline) noexcept -> void;
  static auto bindScene(se::gfx::SceneHandle scene) noexcept -> void;

  static auto getInspectorWidget() noexcept -> InspectorWidget*;
  static auto getSceneWidget() noexcept -> SceneWidget*;
  static auto getViewportWidget() noexcept -> ViewportWidget*;
  static auto getRDGViewerWidget() noexcept -> RDGViewerWidget*;
  static auto getStatusWidget() noexcept -> StatusWidget*;
};
}