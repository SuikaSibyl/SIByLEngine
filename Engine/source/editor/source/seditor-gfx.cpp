#define SIByL_API __declspec(dllexport)
#include <seditor-gfx.hpp>
#undef SIByL_API
#define SIByL_API __declspec(dllimport)
#include <se.editor.hpp>
#include <seditor-base.hpp>
#undef SIByL_API

namespace se::editor {
auto InspectorWidget::onDrawGui() noexcept -> void {
  ImGui::Begin("Inspector", 0, ImGuiWindowFlags_MenuBar);
  if (customDraw) customDraw();
  ImGui::End();
}

auto InspectorWidget::setCustomDraw(std::function<void()> func) noexcept -> void {
  customDraw = func;
}

auto InspectorWidget::setEmpty() noexcept -> void {
  setCustomDraw([]() {});
}

auto ViewportWidget::bindInput(se::input* input) noexcept -> void {
  controller.input = input;
}

auto ViewportWidget::bindTimer(se::timer* timer) noexcept -> void {
  controller.timer = timer;
}

auto ViewportWidget::onUpdate() noexcept -> void {
  controller.bindTransform(se::editor::EditorBase::viewportWidget.camera_transform);
  controller.onUpdate();
}

/** draw gui*/
auto ViewportWidget::onDrawGui() noexcept -> void {
   ImGui::Begin("Viewport", 0, ImGuiWindowFlags_MenuBar);
   static bool draw_gizmo = true;
   ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
   if (ImGui::BeginMenuBar()) {
    if (ImGui::Button("capture")) {
      if (texture.get()) {
        gfx::captureImage(texture);
      }
    }

    ImGui::Checkbox("Gizmo", &draw_gizmo);
    // menuBarSize = ImGui::GetWindowSize();
    //ImGui::list
    if (ImGui::BeginMenu("Cameras"))
    { // check all cameras
      gfx::SceneHandle& scene = se::editor::EditorBase::sceneWidget.scene;
      auto camera_view = scene->registry.view<gfx::Camera, gfx::NodeProperty, gfx::Transform>();
      int i = 0;
      bool use_editor_camera = false;
      // the editor camera
      if (true) {
        bool selected = false;
        ImGui::MenuItem("Editor View", NULL, &selected);
        //ImGui::SameLine();
        if (ImGui::Button("+")) {
          gfx::Node node = scene->createNode("new camera");
          gfx::Transform* transform = node.getComponent<gfx::Transform>();
          *transform = editor_camera_transform;
          gfx::Camera& camera = node.addComponent<gfx::Camera>();
          camera = editor_camera;
        }
        if (selected) use_editor_camera = true;
      }
      for (auto entity : camera_view) {
        auto [se_camera, se_node, se_trans] = camera_view.get<gfx::Camera, gfx::NodeProperty, gfx::Transform>(entity);
        bool selected = false;
        ImGui::MenuItem((se_node.name+std::to_string(uint32_t(entity))).c_str(), NULL, &selected);
        if (selected) camera_index = i;
        if (camera_index == i) {
          camera_transform = &se_trans;
        }
        i++;
      }
      if (use_editor_camera) {
        camera_index = i;
        camera_transform = &editor_camera_transform;
      }
      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
   }
   ImGui::PopItemWidth();
   commonOnDrawGui();
   auto currPos = ImGui::GetCursorPos();
   info.mousePos = ImGui::GetMousePos();
   info.mousePos.x -= info.windowPos.x + currPos.x;
   info.mousePos.y -= info.windowPos.y + currPos.y;

   ImVec2 p = ImGui::GetCursorScreenPos();

   if (texture.get()) {
      ImGui::Image(
        se::editor::ImGuiContext::getImGuiTexture(texture)->getTextureID(),
        {(float)texture->texture->width(), (float)texture->texture->height()},
        {0, 0}, {1, 1});
   } else {
    ImGui::End();
    return;
   }

   //float width = (float)texture->texture->width();
   //float height = (float)texture->texture->height();

   //Math::vec3 posW = camera_transform.translation;
   //Math::vec3 target =
   //    camera_transform.translation + camera_transform.getRotatedForward();

   //Math::mat4 view = Math::transpose(Math::lookAt(posW, target, Math::vec3(0, 1, 0)).m);
   //Math::mat4 proj = Math::transpose(Math::perspective(camera->fovy, camera->aspect,
   //                                      camera->near, camera->far)
   //        .m);

   //for (int i = 0; i < 4; i++)
   // for (int j = 0; j < 4; j++) {
   //   float neg_view = (j == 0 || (i == 3 && j == 3)) ? 1 : -1;
   //   view.data[i][j] *= neg_view;
   //   float neg_proj = (i == 0 || i == 1) ? 1 : ((i == 2) ? -1 : 2);
   //   proj.data[i][j] *= neg_proj;
   // }

   //if (draw_gizmo) {
   // ImGuizmo::SetOrthographic(false);
   // ImGuizmo::SetDrawlist();

   // float windowWidth = (float)ImGui::GetWindowWidth();
   // float windowHeight = (float)ImGui::GetWindowHeight();
   // ImVec2 wp = ImGui::GetWindowPos();

   // ImGuizmo::SetRect(wp.x + currPos.x, wp.y + currPos.y, width, height);

   // // Math::mat4 transform = tc.getAccumulativeTransform();
   // Math::mat4 transform;

   // ImGuiWindow* window = ImGui::GetCurrentWindow();
   // // ImGuizmo::DrawGrid(&view.data[0][0], &proj.data[0][0], identityMatrix,
   // //                    100.f);
   // bool test = false;
   // {
   //   if (ImGui::IsKeyPressed(ImGuiKey_R))
   //     gizmoState.mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
   //   if (ImGui::IsKeyPressed(ImGuiKey_T))
   //     gizmoState.mCurrentGizmoOperation = ImGuizmo::ROTATE;
   //   if (ImGui::IsKeyPressed(ImGuiKey_Y))  // r Key
   //     gizmoState.mCurrentGizmoOperation = ImGuizmo::SCALE;
   //   if (ImGui::IsKeyPressed(ImGuiKey_U))
   //     gizmoState.mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;
   // }
   // Math::mat4 objectTransform;
   // Math::mat4 objectTransformPrecursor;
   // if (selectedGO.has_value() && selectedScene != nullptr) {
   //   {  // first compute the transform
   //      //   get transform
   //     float oddScaling = 1.f;
   //     Math::vec3 scaling = Math::vec3{1, 1, 1};
   //     {  // get mesh transform matrix
   //       GFX::GameObject* go =
   //           selectedScene->getGameObject(selectedGO.value());
   //       GFX::TransformComponent* transform =
   //           go->getEntity().getComponent<GFX::TransformComponent>();
   //       objectTransform = transform->getTransform() * objectTransform;
   //       oddScaling *=
   //           transform->scale.x * transform->scale.y * transform->scale.z;
   //       scaling *= transform->scale;
   //       while (go->parent != Core::NULL_ENTITY) {
   //         test = true;
   //         go = selectedScene->getGameObject(go->parent);
   //         GFX::TransformComponent* transform =
   //             go->getEntity().getComponent<GFX::TransformComponent>();
   //         objectTransform = transform->getTransform() * objectTransform;
   //         objectTransformPrecursor =
   //             transform->getTransform() * objectTransformPrecursor;
   //         oddScaling *=
   //             transform->scale.x * transform->scale.y * transform->scale.z;
   //         scaling *= transform->scale;
   //       }
   //     }
   //   }
   //   objectTransform = Math::transpose(objectTransform);
   //   ImGuizmo::Manipulate(
   //       &view.data[0][0], &proj.data[0][0], gizmoState.mCurrentGizmoOperation,
   //       gizmoState.mCurrentGizmoMode, &objectTransform.data[0][0], NULL,
   //       gizmoState.useSnap ? &gizmoState.snap[0] : NULL, NULL, NULL);

   //   if (test) {
   //     float a = 1.f;
   //   }
   //   float matrixTranslation[3], matrixRotation[3], matrixScale[3];

   //   Math::mat4 inv = Math::transpose(Math::inverse(objectTransformPrecursor));
   //   Math::mat4 thisMatrix = objectTransform * inv;
   //   ImGuizmo::DecomposeMatrixToComponents(&thisMatrix.data[0][0],
   //                                         matrixTranslation, matrixRotation,
   //                                         matrixScale);
   //   // ImGuizmo::RecomposeMatrixFromComponents(
   //   //     matrixTranslation, matrixRotation, matrixScale,
   //   //     &thisMatrix.data[0][0]);

   //   GFX::GameObject* go = selectedScene->getGameObject(selectedGO.value());
   //   GFX::TransformComponent* transform =
   //       go->getEntity().getComponent<GFX::TransformComponent>();
   //   transform->translation = Math::vec3{
   //       matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]};
   //   transform->eulerAngles =
   //       Math::vec3{matrixRotation[0], matrixRotation[1], matrixRotation[2]};
   //   transform->scale =
   //       Math::vec3{matrixScale[0], matrixScale[1], matrixScale[2]};
   // }
   // float viewManipulateRight = wp.x + currPos.x + width;
   // float viewManipulateTop = wp.y + currPos.y;
   // Math::mat4 view_origin = view;
   // ImGuizmo::ManipulateResult result = ImGuizmo::ViewManipulate_Custom(
   //     &view.data[0][0], 5,
   //     ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128),
   //     0x10101010);

   // if (selectedGO.has_value() && selectedScene != nullptr && result.edited) {
   //   view = Math::transpose(view);
   //   for (int i = 0; i < 4; i++)
   //     for (int j = 0; j < 4; j++) {
   //       float neg_view = (j == 0 || (i == 3 && j == 3)) ? 1 : -1;
   //       view.data[i][j] *= neg_view;
   //     }
   //   Math::vec4 target =
   //       Math::transpose(objectTransform) * Math::vec4(0, 0, 0, 1);

   //   // Math::mat4::rotateZ(eulerAngles.z) * Math::mat4::rotateY(eulerAngles.y)
   //   // *
   //   //     Math::mat4::rotateX(eulerAngles.x)
   //   Math::vec3 forward =
   //       Math::vec3{result.newDir[0], result.newDir[1], result.newDir[2]};
   //   Math::vec3 cameraPos =
   //       Math::vec3{target.x, target.y, target.z} + forward * 5;
   //   camera_transform_ref->translation =
   //       Math::vec3{cameraPos.data[0], cameraPos.data[1], cameraPos.data[2]};

   //   float pitch = std::atan2(
   //       forward.z, sqrt(forward.x * forward.x + forward.y * forward.y));
   //   float yaw = std::atan2(forward.x, forward.y);
   //   pitch *= 180. / IM_PI;
   //   yaw *= 180. / IM_PI;
   //   yaw = 180. - yaw;
   //   camera_transform_ref->eulerAngles = {pitch, yaw, 0};

   //   *forceReset = true;
   // }
   //}

   ImGui::End();
}

auto StatusWidget::onDrawGui() noexcept -> void {
  ImGui::Begin("Status");
  ImGui::Text(("fps:\t" + std::to_string(1. / timer->deltaTime())).c_str());
  ImGui::Text(("time:\t" + std::to_string(timer->deltaTime())).c_str());
  ImGui::End();
}

SceneWidget::SceneWidget() {

}

SceneWidget::~SceneWidget() {

}

auto drawCustomColume(const std::string& label, float columeWidth,
  std::function<void()> const& func) noexcept -> void {
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts->Fonts[0];
  ImGui::PushID(label.c_str());
  ImGui::Columns(2);
  { // First Column
    ImGui::SetColumnWidth(0, columeWidth);
    ImGui::Text(label.c_str());
    ImGui::NextColumn();
  }
  { // Second Column
    int width = (ImGui::GetContentRegionAvail().x - 20 + 5);
    ImGui::PushItemWidth(width);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
    func();
    ImGui::PopStyleVar();
    ImGui::PopItemWidth();
  }
  ImGui::Columns(1);
  ImGui::PopID();
}

void drawNodeInspector(gfx::Node& node) {
  gfx::NodeProperty* nodeprop = node.getComponent<gfx::NodeProperty>();
  if (nodeprop) {
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    strcpy_s(buffer, nodeprop->name.c_str());
    if (ImGui::InputText(" ", buffer, sizeof(buffer)))
      nodeprop->name = std::string(buffer);
  }

  gfx::Camera* cameraComp = node.getComponent<gfx::Camera>();
  if (cameraComp) {
    drawComponent<gfx::Camera>(node, "Camera Component", [](gfx::Camera* component) {
      bool isDirty = false;
      //drawCustomColume("Project", 100, [&]() {
      //  int item_current = static_cast<int>(component->projectType);
      //  ImGui::Combo("##proj", &item_current, "Perspective\0Orthogonal\0");
      //  if (item_current != static_cast<int>(component->projectType))
      //    component->projectType =
      //    GFX::CameraComponent::ProjectType(item_current);
      //});
      drawCustomColume("yFoV", 100, [&]() {
        float fovy = component->yfov;
        ImGui::DragFloat("##yFoV", &fovy, 1);
        if (fovy != component->yfov) {
          component->yfov = fovy;
          isDirty = true;
        }
      });
      drawCustomColume("zNear", 100, [&]() {
        float znear = component->znear;
        ImGui::DragFloat("##near", &znear, 0.001f);
        if (znear != component->znear) {
          component->znear = znear;
          isDirty = true;
        }
      });
      drawCustomColume("zFar", 100, [&]() {
         float zfar = component->zfar;
         ImGui::DragFloat("##far", &zfar, 10.f);
        if (zfar != component->zfar) {
          component->zfar = zfar;
          isDirty = true;
        }
      });
    });
  }

}

auto drawNode(gfx::Node const& node, gfx::Scene* scene, SceneWidget* widget) -> bool {
   ImGui::PushID(uint32_t(node.entity));
   ImGuiTreeNodeFlags node_flags = 0;
   gfx::NodeProperty* nodeprop = node.getComponent<gfx::NodeProperty>();
   if (nodeprop->children.size() == 0)
    node_flags |= ImGuiTreeNodeFlags_Leaf;
   if (node.entity == widget->forceNodeOpen.entity && node.registry == widget->forceNodeOpen.registry) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    widget->forceNodeOpen = {};
   }
   std::string name = nodeprop->name.c_str();
   if (name == "") name = "[NAMELESS NODE]";
   bool opened = ImGui::TreeNodeEx(name.c_str(), node_flags);
   ImGuiID uid = ImGui::GetID((name + std::to_string(std::uint32_t(node.entity))).c_str());
   ImGui::TreeNodeBehaviorIsOpen(uid);
   // Clicked
   if (ImGui::IsItemClicked()) {
    widget->inspected = node;
    EditorBase::inspectorWidget.setCustomDraw(std::bind(drawNodeInspector, node));

    //if (inspectorWidget)
    //  gameobjectInspector.setInspectorWidget(inspectorWidget);
    //if (viewportWidget) {
    //  viewportWidget->selectedScene = scene;
    //  viewportWidget->selectedGO = node;
    //}
   }
   //// Right-click on blank space
   //bool entityDeleted = false;
   //if (ImGui::BeginPopupContextItem()) {
   // if (ImGui::MenuItem("Create Empty Entity")) {
   //   scene->createGameObject(node);
   //   forceNodeOpen = node;
   // }
   // if (ImGui::MenuItem("Delete Entity")) entityDeleted = true;
   // ImGui::EndPopup();
   //}
   //// If draged
   //if (ImGui::BeginDragDropSource()) {
   // ImGui::Text(tag->name.c_str());
   // ImGui::SetDragDropPayload("SceneEntity", &node, sizeof(node));
   // ImGui::EndDragDropSource();
   //}
   //// If dragged to
   //if (ImGui::BeginDragDropTarget()) {
   // if (const ImGuiPayload* payload =
   //         ImGui::AcceptDragDropPayload("SceneEntity")) {
   //   GFX::GameObjectHandle* dragged_handle = (uint64_t*)payload->Data;
   //   // binded_scene->tree.moveNode(*dragged_handle, node);
   // }
   // ImGui::EndDragDropTarget();
   //}
   // Opened
   if (opened) {
    ImGui::NextColumn();
    for (int i = 0; i < nodeprop->children.size(); i++) {
      drawNode(nodeprop->children[i], scene, widget);
    }
    ImGui::TreePop();
   }
   ImGui::PopID();
   //if (entityDeleted) {
   // bool isParentOfInspected = false;
   // GFX::GameObjectHandle inspected_parent = inspected;
   // while (inspected_parent != GFX::NULL_GO) {
   //   inspected_parent = scene->getGameObject(inspected_parent)->parent;
   //   if (node == node) {
   //     isParentOfInspected = true;
   //     break;
   //   }  // If set ancestor as child, no movement;
   // }
   // if (node == inspected || isParentOfInspected) {
   //   if (inspectorWidget) inspectorWidget->setEmpty();
   //   if (viewportWidget) {
   //     viewportWidget->selectedScene = nullptr;
   //     viewportWidget->selectedGO = std::nullopt;
   //   }
   //   // if (viewport) viewport->selectedEntity = {};
   //   inspected = 0;
   // }
   // scene->removeGameObject(node);
   // return true;
   //}
   return false;
}

auto SceneWidget::onDrawGui() noexcept -> void {
  ImGui::Begin("Scene", 0, ImGuiWindowFlags_MenuBar);
  ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
  auto save_scene = [&scene = this->scene]() {
    std::string name = scene->name + ".scene";
    std::string path = se::gfx::GFXContext::device->fromAdapter()->fromContext()->getBindedWindow()->saveFile(nullptr, name);
    if (path != "") {
      scene->serialize(path);
      scene->isDirty = false;
    }
  };
  // draw the menubar
  if (ImGui::BeginMenuBar()) {
    if (ImGui::Button("New")) {
    }
    if (ImGui::Button("Save")) {
      if (scene.get() != nullptr) {
        save_scene();
      }
    }

    ImGui::EndMenuBar();
  }
  ImGui::PopItemWidth();

  // Left-clock on blank space
  if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
    EditorBase::inspectorWidget.setEmpty();
    //if (viewportWidget) {
    //  viewportWidget->selectedScene = nullptr;
    //  viewportWidget->selectedGO = std::nullopt;
    //}
  }
  //// Right-click on blank space
  //if (ImGui::BeginPopupContextWindow(0, 1, false)) {
  //  if (ImGui::MenuItem("Create Empty Entity") && scene) {
  //    scene->createGameObject(GFX::NULL_GO);
  //    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
  //  }
  //  ImGui::EndPopup();
  //}

  // Draw scene hierarchy
  if (scene.get() != nullptr) {
    for (auto& node : scene->roots) drawNode(node, scene.get(), this);
  }

  ImGui::End();
}

auto CameraState::setFromTransform(
  gfx::Transform const& transform) noexcept -> void {
  auto rotationMatrix = transform.rotation.toMat3();
  se::vec3 eulerAngles = se::rotationMatrixToEulerAngles(rotationMatrix);
  se::vec3 translation = transform.translation;

  pitch = eulerAngles.x * 180 / float_Pi;
  yaw = eulerAngles.y * 180 / float_Pi;
  roll = eulerAngles.z * 180 / float_Pi;

  if (std::abs(roll + 180) < 1.f) {
    pitch = -(180.f - std::abs(pitch)) * pitch / std::abs(pitch);
    yaw = (180.f - std::abs(yaw)) * yaw / std::abs(yaw);
    roll = 0.f;
  }
  if (std::abs(roll - 180) < 1.f) {
    pitch = -(180.f - std::abs(pitch)) * pitch / std::abs(pitch);
    yaw = (180.f - std::abs(yaw)) * yaw / std::abs(yaw);
    roll = 0.f;
  }

  x = translation.x;
  y = translation.y;
  z = translation.z;
}

auto CameraState::lerpTowards(CameraState const& target, float positionLerpPct,
                              float rotationLerpPct) noexcept -> void {
   yaw = std::lerp(yaw, target.yaw, rotationLerpPct);
   pitch = std::lerp(pitch, target.pitch, rotationLerpPct);
   roll = std::lerp(roll, target.roll, rotationLerpPct);

   x = std::lerp(x, target.x, positionLerpPct);
   y = std::lerp(y, target.y, positionLerpPct);
   z = std::lerp(z, target.z, positionLerpPct);
}

auto CameraState::updateTransform(gfx::Transform& transform) noexcept
    -> void {
  transform.rotation = se::Quaternion(se::eulerAngleToRotationMatrix(se::vec3(pitch, yaw, roll)));
  transform.translation = se::vec3(x, y, z);
}

auto SimpleCameraController::onEnable(
  gfx::Transform const& transform) noexcept -> void {
   targetCameraState.setFromTransform(transform);
   interpolatingCameraState.setFromTransform(transform);
}

auto SimpleCameraController::getInputTranslationDirection() noexcept
    -> se::vec3 {
    se::vec3 direction(0.0f, 0.0f, 0.0f);
   if (input->isKeyPressed(se::input::key_w)) {
    direction += se::vec3(0, 0, +1);  // forward
   }
   if (input->isKeyPressed(se::input::key_s)) {
    direction += se::vec3(0, 0, -1);  // back
   }
   if (input->isKeyPressed(se::input::key_a)) {
    direction += se::vec3(-1, 0, 0);  // left
   }
   if (input->isKeyPressed(se::input::key_d)) {
    direction += se::vec3(1, 0, 0);  // right
   }
   if (input->isKeyPressed(se::input::key_q)) {
    direction += se::vec3(0, -1, 0);  // down
   }
   if (input->isKeyPressed(se::input::key_e)) {
    direction += se::vec3(0, 1, 0);  // up
   }
   return direction;
}

auto SimpleCameraController::bindTransform(
  gfx::Transform* transform) noexcept -> void {
   ViewportWidget* viewport = &EditorBase::viewportWidget;
   //viewport->camera_transform = *transform;
   //viewport->camera_transform_ref = transform;
   //viewport->forceReset = &forceReset;
   if (this->transform != transform) {
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    this->transform = transform;
   }
}

auto SimpleCameraController::onUpdate() noexcept -> void {
  if (transform == nullptr) return;
   if (forceReset) {
    forceReset = false;
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    return;
   }
   // check the viewport is hovered
   ViewportWidget* viewport = &EditorBase::viewportWidget;
   bool hovered = viewport->info.isHovered;
   // rotation
   static bool justPressedMouse = true;
   static float last_x = 0;
   static float last_y = 0;
   static bool inRotationMode = false;

   if (input->isMouseButtonPressed(se::input::mouse_button_2) &&
       viewport->info.isHovered && viewport->info.isFocused)
    inRotationMode = true;
   if (!input->isMouseButtonPressed(se::input::mouse_button_2)) {
    inRotationMode = false;
   }

   bool isDirty = false;

   if (input->isMouseButtonPressed(se::input::mouse_button_2)) {
    if (inRotationMode) {
      input->disableCursor();
      float x = input->getMouseX();
      float y = input->getMouseY();
      if (justPressedMouse) {
        last_x = x;
        last_y = y;
        justPressedMouse = false;
      } else {
        se::vec2 mouseMovement = se::vec2(x - last_x, y - last_y) *
                                   0.0005f * mouseSensitivityMultiplier *
                                   mouseSensitivity;
        if (invertY) mouseMovement.y = -mouseMovement.y;
        last_x = x;
        last_y = y;

        float mouseSensitivityFactor =
            mouseSensitivityCurve.evaluate(mouseMovement.length()) * 180. /
            3.1415926;

        targetCameraState.yaw -= mouseMovement.x * mouseSensitivityFactor;
        targetCameraState.pitch += mouseMovement.y * mouseSensitivityFactor;
        isDirty = true;
      }
    }
   } else if (!justPressedMouse) {
    input->enableCursor();
    justPressedMouse = true;
   }

   // translation
   se::vec3 translation = getInputTranslationDirection();
   translation *= timer->deltaTime() * 0.1;
   if (translation.x != 0 && translation.y != 0 && translation.z != 0) {
     isDirty = true;
   }

   // speed up movement when shift key held
   if (input->isKeyPressed(se::input::key_left_shift)) {
    translation *= 10.0f;
   }

   // modify movement by a boost factor ( defined in Inspector and modified in
   // play mode through the mouse scroll wheel)
   float y = input->getMouseScrollY();
   boost += y * 0.01f;
   translation *= powf(2.0f, boost);

   se::vec4 rotatedFoward4 = se::mat4::rotateZ(targetCameraState.roll) *
       se::mat4::rotateY(targetCameraState.yaw) *
       se::mat4::rotateX(targetCameraState.pitch) *
       se::vec4(0, 0, -1, 0);
   se::vec3 rotatedFoward =
       se::vec3(rotatedFoward4.x, rotatedFoward4.y, rotatedFoward4.z);
   se::vec3 up = se::vec3(0.0f, 1.0f, 0.0f);
   se::vec3 cameraRight = se::normalize(se::cross(rotatedFoward, up));
   se::vec3 cameraUp = se::cross(cameraRight, rotatedFoward);
   se::vec3 movement = translation.z * rotatedFoward +
                         translation.x * cameraRight + translation.y * cameraUp;

   targetCameraState.x += movement.x;
   targetCameraState.y += movement.y;
   targetCameraState.z += movement.z;

   // targetCameraState.translate(translation);

   // Framerate-independent interpolation
   // calculate the lerp amount, such that we get 99% of the way to our target
   // in the specified time
   float positionLerpPct =
       1.f - expf(log(1.f - 0.99f) / positionLerpTime * timer->deltaTime());
   float rotationLerpPct =
       1.f - expf(log(1.f - 0.99f) / rotationLerpTime * timer->deltaTime());
   interpolatingCameraState.lerpTowards(targetCameraState, positionLerpPct,
                                        rotationLerpPct);

   if (isDirty) {
    EditorBase::getSceneWidget()->scene->dirtyFlags = 
      (uint64_t)se::gfx::Scene::DirtyFlagBit::Camera;
   }

   if (transform != nullptr)
    interpolatingCameraState.updateTransform(*transform);
}

}