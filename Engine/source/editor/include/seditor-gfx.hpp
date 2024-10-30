#pragma once
#include <memory>
#include <string>
#include <imgui.h>
#include <unordered_map>
#include "seditor.hpp"
#include <se.gfx.hpp>

namespace se::editor {
struct SIByL_API InspectorWidget : public Widget {
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  /** set custom draw */
  auto setCustomDraw(std::function<void()> func) noexcept -> void;
  /** set empty */
  auto setEmpty() noexcept -> void;
  /** custom draw on inspector widget */
  std::function<void()> customDraw;
};

struct SIByL_API SceneWidget : public Widget {
  SceneWidget();
  ~SceneWidget();
  /** virtual draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  ///** draw a scene gameobject and its descents, return whether is deleted */
  //auto drawNode(gfx::Node const& node) -> bool;
  /** bind scene to the widget */
  auto bindScene(gfx::SceneHandle scene) noexcept -> void { this->scene = scene; }
  /** scene binded to be shown on widget */
  gfx::SceneHandle scene;
  /** the scene node that should be opened */
  gfx::Node forceNodeOpen;
  /** the scene node that is inspected */
  gfx::Node inspected;
  ///** viewport widget to show gameobject gizmos */
  //ViewportWidget* viewportWidget = nullptr;
  ///** game object inspector */
  //GameObjectInspector gameobjectInspector = {};
};

struct SIByL_API CameraState {
  float yaw = 0;
  float pitch = 0;
  float roll = 0;
  float x = 0;
  float y = 0;
  float z = 0;
  auto setFromTransform(gfx::Transform const& transform) noexcept -> void;
  auto lerpTowards(CameraState const& target, float positionLerpPct,
    float rotationLerpPct) noexcept -> void;
  auto updateTransform(gfx::Transform& transform) noexcept -> void;
};

struct SIByL_API SimpleCameraController {
  SimpleCameraController() : input(nullptr), timer(nullptr) {}
  auto onEnable(gfx::Transform const& transform) noexcept -> void;
  auto getInputTranslationDirection() noexcept -> se::vec3;
  auto bindTransform(gfx::Transform* transform) noexcept -> void;
  auto onUpdate() noexcept -> void;

  float mouseSensitivityMultiplier = 0.01f;
  CameraState targetCameraState;
  CameraState interpolatingCameraState;
  float boost = 3.5f;
  float positionLerpTime = 0.2f;
  float mouseSensitivity = 60.0f;
  se::AnimationCurve mouseSensitivityCurve = {{0, 0.5, 0, 5}, {1, 2.5, 0, 0}};
  float rotationLerpTime = 0.01f;
  bool invertY = true;
  bool forceReset = false;
  float scaling = 1.f;
  se::input* input;
  se::timer* timer;
  gfx::Transform* transform = nullptr;
  bool use2dmode = false;
};

struct SIByL_API ViewportWidget : public Widget {
  auto bindInput(se::input* input) noexcept -> void;
  auto bindTimer(se::timer* timer) noexcept -> void;
  auto setTarget(std::string const& name, gfx::TextureHandle tex) noexcept -> void;
  auto onUpdate() -> void;
  auto setCameraIndex(int index) noexcept -> void;
  auto use2DCamera() noexcept -> void;
  auto use3DCamera() noexcept -> void;

  auto addCustomDrawFn(std::function<void()>& fn) noexcept -> void;
  auto claerCustomDrawFn() noexcept -> void;

  auto getWindowPosition() noexcept -> vec2;
  auto getMousePosition() noexcept -> vec2;
  auto isWindowHovered() noexcept -> bool;
  auto isWindowFocused() noexcept -> bool;

  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  int camera_index = 0;
  gfx::TextureHandle texture;
  //gfx::Camera camera;
  gfx::Transform* camera_transform = nullptr;
  gfx::Transform* camera_transform_ref = nullptr;
  //ImGuizmoState gizmoState;
  //bool* forceReset;
  //std::optional<gfx::GameObjectHandle> selectedGO = std::nullopt;
  //gfx::Scene* selectedScene;
  gfx::Camera editor_camera = {};
  gfx::Transform editor_camera_transform = {};
  se::editor::SimpleCameraController controller;
  std::vector<std::function<void()>> custom_imgui_draws = {};
};

struct SIByL_API StatusWidget : public Widget {
  se::timer* timer;
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;

  auto getBindedTimer() noexcept -> se::timer*;
};

SIByL_API auto drawTextureViewer(gfx::TextureHandle texture) noexcept -> void;

SIByL_API auto drawCustomColume(const std::string& label, float columeWidth,
  std::function<void()> const& func) noexcept -> void;

template <class T, class UIFunction>
auto drawComponent(gfx::Node& node, std::string const& name,
  UIFunction uiFunction, bool couldRemove = true) noexcept -> void {
  const ImGuiTreeNodeFlags treeNodeFlags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
      ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth |
      ImGuiTreeNodeFlags_AllowItemOverlap;
  T* component = node.getComponent<T>();
  if (component) {
    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
    float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    ImGui::Separator();
    bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());
    bool removeComponent = false;
    if (couldRemove) {
      ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5);
      if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight })) {
        ImGui::OpenPopup("ComponentSettings");
      }
      if (ImGui::BeginPopup("ComponentSettings")) {
        if (ImGui::MenuItem("Remove Component")) removeComponent = true;
        ImGui::EndPopup();
      }
    }
    if (open) {
      T* component = node.getComponent<T>();
      uiFunction(component);
      ImGui::Dummy(ImVec2(0.0f, 20.0f));
      ImGui::TreePop();
    } 
    if (couldRemove && removeComponent)
      node.removeComponent<T>();
  }
}
}