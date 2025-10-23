#include "se.gfx.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "se.editor.helper.hpp"

namespace se {
namespace gfx {
  auto ComponentManager::draw_all_components(Node& node) noexcept -> void {
    for (auto& iter : Singleton<ComponentManager>::instance()->m_components) {
      void* component = iter.second.retrival(node);
      if (component == nullptr) continue;

      ImGui::PushID(iter.first);
      const ImGuiTreeNodeFlags treeNodeFlags =
        ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
        ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth |
        ImGuiTreeNodeFlags_AllowItemOverlap;

      ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
      float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
      ImGui::Separator();
      bool open = ImGui::TreeNodeEx(iter.second.name.c_str(), treeNodeFlags);
      bool removeComponent = false;

      float boxSize = lineHeight * 0.25f;
      ImVec2 baseMin = ImGui::GetItemRectMin();
      ImVec2 baseMax = ImGui::GetItemRectMax();
      float verticalOffset = (baseMax.y - baseMin.y - boxSize) * 0.5f;
      float spacing = lineHeight * 0.6f;

      // Start from the right edge
      ImVec2 box1Pos = ImVec2(baseMax.x - boxSize - lineHeight * 2, baseMin.y + verticalOffset);
      ImVec2 box2Pos = ImVec2(baseMax.x - boxSize - lineHeight * 2 - spacing, baseMin.y + verticalOffset);

      if (iter.second.dirtyToGPU(component)) {
        ImGui::SameLine();
        helper::DrawColoredBox("##dirty_gpu", boxSize, box2Pos, IM_COL32(255, 165, 0, 255));
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup)) {
          ImGui::SetTooltip("This component has changes not updated to GPU");
        }
      }

      if (iter.second.dirtyToFile(component)) {
        ImGui::SameLine();
        helper::DrawColoredBox("##dirty_file", boxSize, box1Pos, IM_COL32(50, 163, 255, 255));
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup)) {
          ImGui::SetTooltip("This component has changes not saved to file");
        }
      }


      if (iter.second.couldRemove) {
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
        iter.second.draw(component);
        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::TreePop();
      }
      if (iter.second.couldRemove && removeComponent)
        iter.second.remove(node);

      ImGui::PopID();
    }
  }

	auto NodeProperty::draw_component(void* component) noexcept -> void {
    NodeProperty* node_property = (NodeProperty*)component;
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    strncpy(buffer, node_property->name.c_str(), sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0'; // Ensure null-termination
    if (ImGui::InputText(" ", buffer, sizeof(buffer))) {
      node_property->name = std::string(buffer);
      node_property->m_dirtyToFile = true;
    }
	}

  auto NodeProperty::serialize(SerializeData& data) noexcept -> void {
    auto node_view = data.gfx_scene->m_registry.view<NodeProperty>();
    for (auto [entity, _property] : node_view.each()) {
      size_t node_index = data.nodes.size();
      data.nodes.emplace(entity, node_index);
      tinygltf::Node node;
      node.name = _property.name;
      data.model->nodes.emplace_back(node);
    }
    for (auto [entity, _property] : node_view.each()) {
      size_t node_index = data.nodes[entity];
      for (auto& child : _property.children)
        data.model->nodes[node_index].children.push_back(data.nodes[child.m_entity]);
    }
  }

  auto Transform::local() const noexcept -> se::mat4 {
    se::mat4 trans = se::mat4::translate(translation);
    se::mat4 rotat = rotation.toMat4();
    se::mat4 scal = se::mat4::scale(scale);
    se::mat4 rotscal = rotat * scal;
    return trans * rotscal;
  }

  auto Transform::forward() const noexcept -> se::vec3 {
    se::vec4 rotated = rotation.toMat4() * se::vec4(0, 0, -1, 0);
    return se::vec3(rotated.x, rotated.y, rotated.z);
  }

  auto Transform::draw_component(void* _component) noexcept -> void {
    Transform* component = (Transform*)_component;

    if (ImGui::BeginTable("CameraTable", 2, ImGuiTableFlags_None)) {
      // Column setup MUST be here, before any rows
        // First column (labels) - fixed width
      ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
      // Second column (widgets) - stretch to remaining width
      ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

      // Projection Type (consistent left label)
      helper::DrawLabeledControl("Translation", [&]() {
        // set translation
        se::vec3 translation = component->translation;
        helper::DrawVec3Control("##Translation", translation, 0, 0.1);
        bool translation_modified = (component->translation != translation);
        if (translation_modified) {
          component->translation = translation;
          component->m_dirtyToFile = true;
          component->m_dirtyToGPU = true;
        }
        return translation_modified;
      });

      // Projection Type (consistent left label)
      helper::DrawLabeledControl("        Scaling", [&]() {
        se::vec3 scaling = component->scale;
        helper::DrawVec3Control("##Scaling", scaling, 1, 0.1);
        bool scale_modified = (component->scale != scaling);
        if (scale_modified) {
          component->scale = scaling;
          component->m_dirtyToFile = true;
          component->m_dirtyToGPU = true;
        }
        return scale_modified;
      });
      
      // Projection Type (consistent left label)
      static int rotation_mode = 0;
      const char* labels[] = {
        "Quaternion",
        "XYZ Euler",
      };
      helper::DrawLabeledControl("           Mode", [&]() {
      const char* currentLabel = labels[rotation_mode];
      if (ImGui::BeginCombo("##rotmode", currentLabel)) {
        for (int i = 0; i < 2; ++i) {
          bool isSelected = (i == rotation_mode);
          if (ImGui::Selectable(labels[i], isSelected)) {
            rotation_mode = i;
          }
          if (isSelected) {
            ImGui::SetItemDefaultFocus(); // Optional: focuses the selected item
          }
        }
        ImGui::EndCombo();
      }

        return false;
      });
      
    
      // Projection Type (consistent left label)
      helper::DrawLabeledControl("        Rotation", [&]() {
        if (rotation_mode == 0) {
          se::vec4 quaternion = { component->rotation.x, component->rotation.y, 
            component->rotation.z, component->rotation.w };
          helper::DrawVec4Control("##Rotation", quaternion, 0, 0.1);
          bool rotate_modified = 
            (component->rotation.x != quaternion.x) ||
            (component->rotation.y != quaternion.y) ||
            (component->rotation.z != quaternion.z) ||
            (component->rotation.w != quaternion.w);
          if (rotate_modified) {
            component->rotation.x = quaternion.x;
            component->rotation.y = quaternion.y;
            component->rotation.z = quaternion.z;
            component->rotation.w = quaternion.w;
          }
          return rotate_modified;
        }
        else if (rotation_mode == 1) {
          se::vec3 euler = rotation_matrix_to_euler_angles(component->rotation.toMat3());
          euler.x = degrees(euler.x);
          euler.y = degrees(euler.y);
          euler.z = degrees(euler.z);
          se::vec3 euler_old = euler;
          helper::DrawVec3Control("##Euler", euler, 1, 0.1);
          bool euler_modified = (euler_old != euler);
          if (euler_modified) {
            euler.x = radians(euler.x);
            euler.y = radians(euler.y);
            euler.z = radians(euler.z);
            component->rotation = euler_angle_to_quaternion(euler);
            component->m_dirtyToFile = true;
            component->m_dirtyToGPU = true;
          }
          return euler_modified;
        }
        return false;
      });
      
      ImGui::EndTable();
    }
  }
  
  auto Transform::serialize(SerializeData& data) noexcept -> void {
    auto node_view = data.gfx_scene->m_registry.view<Transform>();
    for (auto [entity, _transform] : node_view.each()) {
      size_t node_index = data.nodes[entity];
      tinygltf::Node& node = data.model->nodes[node_index];
      node.translation = { _transform.translation.x,_transform.translation.y, _transform.translation.z };
      node.scale = { _transform.scale.x,_transform.scale.y, _transform.scale.z };
      node.rotation = { _transform.rotation.x,_transform.rotation.y, _transform.rotation.z, _transform.rotation.w };
    }
  }

  auto MeshRenderer::draw_component(void* component) noexcept -> void {
    MeshRenderer* mr = (MeshRenderer*)component;
    mr->m_mesh->draw_gui(nullptr);
  }

  auto MeshRenderer::serialize(SerializeData& data) noexcept -> void {
    tinygltf::Model* m = data.model;
    auto node_view = data.gfx_scene->m_registry.view<MeshRenderer>();
    for (auto [entity, _meshRender] : node_view.each()) {
      int const node_id = data.nodes[entity];
      int const mesh_id = m->meshes.size();
      m->meshes.emplace_back(tinygltf::Mesh{}); 
      auto& gltf_mesh = m->meshes.back();

      int32_t position_buffer = data.add_buffer(
        _meshRender.m_mesh->m_positionBuffer->get_host(), "Position Buffer");
      int32_t index_buffer = data.add_buffer(
        _meshRender.m_mesh->m_indexBuffer->get_host(), "Index Buffer");
      int32_t vertex_buffer = data.add_buffer(
        _meshRender.m_mesh->m_vertexBuffer->get_host(), "Vertex Buffer");

      // triangle primitives
      if (_meshRender.m_mesh->m_customPrimitives.size() == 0) {
        for (auto& primitive : _meshRender.m_mesh->m_primitives) {
          tinygltf::Primitive gltf_primitive;
          { // position buffer
            tinygltf::BufferView bufferView;
            bufferView.buffer = position_buffer;
            bufferView.byteOffset = primitive.baseVertex * sizeof(float) * 3;
            bufferView.byteLength = primitive.numVertex * sizeof(float) * 3;
            bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
            tinygltf::Accessor accessor;
            accessor.byteOffset = 0;
            accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
            accessor.count = primitive.numVertex;
            accessor.type = TINYGLTF_TYPE_VEC3;
            accessor.maxValues = { primitive.max.x, primitive.max.y, primitive.max.z };
            accessor.minValues = { primitive.min.x, primitive.min.y, primitive.min.z };
            gltf_primitive.attributes["POSITION"] = data.add_view_accessor(bufferView, accessor);
          }
          { // index buffer
            tinygltf::BufferView bufferView;
            bufferView.buffer = index_buffer;
            bufferView.byteOffset = primitive.offset * sizeof(uint32_t);
            bufferView.byteLength = primitive.size * sizeof(uint32_t);
            bufferView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
            tinygltf::Accessor accessor;
            accessor.byteOffset = 0;
            accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
            accessor.count = primitive.size;
            accessor.type = TINYGLTF_TYPE_SCALAR;
            gltf_primitive.indices = data.add_view_accessor(bufferView, accessor);
          }
          { // vertex buffer
            tinygltf::BufferView bufferView;
            bufferView.buffer = vertex_buffer;
            size_t vertexByteOffset = primitive.baseVertex * sizeof(float) * 3;
            bufferView.byteOffset = vertexByteOffset / 3 * 8;
            bufferView.byteLength = primitive.numVertex * sizeof(float) * 8;
            bufferView.byteStride = 8 * sizeof(float);
            bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
            int view_id = data.model->bufferViews.size();
            data.model->bufferViews.push_back(bufferView);
            // normal
            tinygltf::Accessor accessor;
            accessor.bufferView = view_id;
            accessor.byteOffset = 0;
            accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
            accessor.count = primitive.numVertex;
            accessor.type = TINYGLTF_TYPE_VEC3;
            gltf_primitive.attributes["NORMAL"] = data.add_accessor(accessor);
            //// tangent
            //accessor.byteOffset = sizeof(float) * 3;
            //gltf_primitive.attributes["TANGENT"] = data.add_accessor(accessor);
            // coord
            accessor.byteOffset = sizeof(float) * 6;
            accessor.type = TINYGLTF_TYPE_VEC2;
            gltf_primitive.attributes["TEXCOORD_0"] = data.add_accessor(accessor);
          }
          //// texcoord1 buffer
          //if (m.buffers.size() >= 4 && m.buffers[3].data.size() > 0) {
          //  tinygltf::BufferView bufferView;
          //  bufferView.buffer = 3;
          //  bufferView.byteOffset = (primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset) * 2 / 3;
          //  bufferView.byteLength = primitive.numVertex * sizeof(float) * 2;
          //  bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
          //  tinygltf::Accessor accessor;
          //  accessor.byteOffset = 0;
          //  accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
          //  accessor.count = primitive.numVertex;
          //  accessor.type = TINYGLTF_TYPE_VEC2;
          //  gltf_primitive.attributes["TEXCOORD_1"] = add_view_accessor(bufferView, accessor);
          //}
          gltf_primitive.mode = TINYGLTF_MODE_TRIANGLES;
          gltf_primitive.material = data.add_material(primitive.material.get());

          //tinygltf::Value::Object primitive_extra;
          //primitive_extra["exterior"] = tinygltf::Value(add_medium(primitive.exterior));
          //primitive_extra["interior"] = tinygltf::Value(add_medium(primitive.interior));
          //gltf_primitive.extras = tinygltf::Value(primitive_extra);

          gltf_mesh.primitives.emplace_back(gltf_primitive);
        }
        m->nodes[node_id].mesh = mesh_id;
      }
      // custom primitives
      else {
        se::error("Custom mesh primitive serialization not implemented yet.");
      }
    }
  }

  auto Camera::get_viewMat() noexcept -> se::mat4 {
    return se::mat4{};
  }

  auto Camera::get_projection_mat() const noexcept -> se::mat4 {
    se::mat4 projection;
    if (projectType == ProjectType::PERSPECTIVE) {
      projection = se::perspective(yfov, aspectRatio, znear, zfar).m;
    }
    else if (projectType == ProjectType::ORTHOGONAL) {
      projection = se::ortho(-aspectRatio * bottom_top, aspectRatio * bottom_top,
        -bottom_top, bottom_top, znear, zfar).m;
    }
    return projection;
  }

  auto Camera::draw_component(void* _component) noexcept -> void {
    Camera* component = (Camera*)_component;
    if (ImGui::BeginTable("CameraTable", 2, ImGuiTableFlags_None)) {
      // Column setup MUST be here, before any rows
        // First column (labels) - fixed width
      ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
      // Second column (widgets) - stretch to remaining width
      ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

      // Projection Type (consistent left label)
      helper::DrawLabeledControl("Projection", [&]() {
        const char* types[] = { "Perspective", "Orthogonal" };
      int current = static_cast<int>(component->projectType);
      bool changed = ImGui::Combo("##ProjectionType", &current, types, IM_ARRAYSIZE(types));
      if (changed) {
        component->projectType = static_cast<ProjectType>(current);
        component->m_dirtyToFile = true;
        component->m_dirtyToGPU = true;
      }
      return changed;
        });

      // Clipping Planes
      se::vec2 clipPlanes{ component->znear, component->zfar };
      if (helper::DrawLabeledControl("Clipping", [&]() {
        return helper::DrawVec2Control("##Clip", clipPlanes);
        })) {
        component->znear = std::max(0.0001f, clipPlanes.x);
        component->zfar = std::max(component->znear + 0.01f, clipPlanes.y);
        component->m_dirtyToFile = true;
        component->m_dirtyToGPU = true;
      }

      // Perspective settings
      if (component->projectType == ProjectType::PERSPECTIVE) {
        if (helper::DrawLabeledControl("AspectRatio", [&]() {
          return ImGui::DragFloat("##AspectRatio", &component->aspectRatio, 0.01f, 0.01f, 10.0f);
          })) {
          component->m_dirtyToFile = true;
          component->m_dirtyToGPU = true;
        }

        if (helper::DrawLabeledControl("Y-FOV", [&]() {
          return ImGui::DragFloat("##YFOV", &component->yfov, 0.1f, 1.0f, 180.0f);
          })) {
          component->m_dirtyToFile = true;
          component->m_dirtyToGPU = true;
        }
      }

      // Orthogonal settings
      if (component->projectType == ProjectType::ORTHOGONAL) {
        se::vec2 ortho{ component->left_right, component->bottom_top };
        if (helper::DrawLabeledControl("OrthoRegion", [&]() {
          return helper::DrawVec2Control("##Ortho", ortho);
          })) {
          component->left_right = ortho.x;
          component->bottom_top = ortho.y;
          component->m_dirtyToFile = true;
          component->m_dirtyToGPU = true;
        }
      }

      ImGui::EndTable();
    }

  }

  auto Camera::serialize(SerializeData& data) noexcept -> void {
    tinygltf::Model* m = data.model;
    auto node_view = data.gfx_scene->m_registry.view<Camera>();
    for (auto [entity, _camera] : node_view.each()) {
      tinygltf::Camera gltf_camera = {};
      int const node_id = data.nodes[entity];
      if (_camera.projectType == Camera::ProjectType::PERSPECTIVE) {
        gltf_camera.type = "perspective";
        gltf_camera.perspective.aspectRatio = _camera.aspectRatio;
        gltf_camera.perspective.yfov = se::radians(_camera.yfov);
        gltf_camera.perspective.znear = _camera.znear;
        gltf_camera.perspective.zfar = _camera.zfar;
      }
      int const camera_id = m->cameras.size();
      m->cameras.emplace_back(gltf_camera);
      m->nodes[node_id].camera = camera_id;
    }
  }

  CameraData::CameraData(gfx::Camera const& camera, gfx::Transform const& transform) {
    nearZ = camera.znear;
    farZ = camera.zfar;
    posW = transform.translation;
    target = transform.translation + transform.forward();
    viewMat = se::transpose(se::look_at(posW, target, se::vec3(0, 1, 0)).m);
    invViewMat = se::inverse(viewMat);
    projMat = se::transpose(camera.get_projection_mat());
    invProjMat = se::inverse(projMat);
    viewProjMat = viewMat * projMat;
    invViewProj = se::inverse(viewProjMat);
    // Ray tracing related vectors
    focalDistance = 1;
    aspectRatio = camera.aspectRatio;
    up = se::vec3(0, 1, 0);
    cameraW = se::normalize(target - posW) * focalDistance;
    cameraU = se::normalize(se::cross(cameraW, up));
    cameraV = se::normalize(se::cross(cameraU, cameraW));
    const float ulen = focalDistance * std::tan(se::radians(camera.yfov) * 0.5f) * aspectRatio;
    cameraU *= ulen;
    const float vlen = focalDistance * std::tan(se::radians(camera.yfov) * 0.5f);
    cameraV *= vlen;
    jitterX = 0;
    jitterY = 0;
    //clipToWindowScale = se::vec2(0.5f * camera.width, -0.5f * camera.height);
    //clipToWindowBias = se::vec2(0.f) + se::vec2(camera.width, camera.height) * 0.5f;
    rectArea = 4 * ulen * vlen / (focalDistance * focalDistance);
  }

  auto Light::draw_component(void* _component) noexcept -> void {
    Light* component = (Light*)_component;
    if (ImGui::BeginTable("CameraTable", 2, ImGuiTableFlags_None)) {
      // Column setup MUST be here, before any rows
        // First column (labels) - fixed width
      ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
      // Second column (widgets) - stretch to remaining width
      ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);
      
      // Projection Type (consistent left label)
      static int light_type = (int)component->light.light_type;
      const char* type_labels[] = {
        "Directional",
        "Point",
        "Spot",
        "Mesh Primitive"
      };
      helper::DrawLabeledControl("           Type", [&]() {
        const char* currentLabel = type_labels[light_type];
        if (ImGui::BeginCombo("##type", currentLabel)) {
          for (int i = 0; i < 4; ++i) {
            bool isSelected = (i == light_type);
            if (ImGui::Selectable(type_labels[i], isSelected)) {
              component->light.light_type = LightTypeEnum(i);
            }
            if (isSelected) {
              ImGui::SetItemDefaultFocus(); // Optional: focuses the selected item
            }
          }
          ImGui::EndCombo();
        }
        return false;
      });

      ImGui::EndTable();
    }

  }

  auto Light::serialize(SerializeData& data) noexcept -> void {
    tinygltf::Model* m = data.model;
    auto node_view = data.gfx_scene->m_registry.view<Light>();
    for (auto [entity, _light] : node_view.each()) {
      //tinygltf::Camera gltf_camera = {};
      //int const node_id = data.nodes[entity];
      //if (_camera.projectType == Camera::ProjectType::PERSPECTIVE) {
      //  gltf_camera.type = "perspective";
      //  gltf_camera.perspective.aspectRatio = _camera.aspectRatio;
      //  gltf_camera.perspective.yfov = se::radians(_camera.yfov);
      //  gltf_camera.perspective.znear = _camera.znear;
      //  gltf_camera.perspective.zfar = _camera.zfar;
      //}
      //int const camera_id = m->cameras.size();
      //m->cameras.emplace_back(gltf_camera);
      //m->nodes[node_id].camera = camera_id;
    }
  }
  
  auto Light::deserialize(DeserializeData& data) noexcept -> void {

  };

  auto Script::draw_component(void* component) noexcept -> void {
    Script* script = (Script*)component;
    if (ImGui::BeginTable("AttachedScripts", 2,
      ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
      // Header
      ImGui::TableSetupColumn("Attached Scripts");
      ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 20.0f); // Narrow column for "×"
      ImGui::TableHeadersRow();

      // Rows
      int id = 0;
      auto iter = script->m_scripts.begin();
      while (iter != script->m_scripts.end()) {
        ImGui::TableNextRow();

        // Column 0: Script name
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", iter->first.c_str());

        // Column 1: Delete button ("×")
        ImGui::TableSetColumnIndex(1);
        ImGui::PushID(id++);
        if (ImGui::SmallButton("×")) {
          iter = script->m_scripts.erase(iter); // Remove from std::list
        }
        else {
          ++iter;
        }
        ImGui::PopID();
      }

      ImGui::EndTable();
    }


    {  // add component
      ImGui::Separator();
      ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
      ImVec2 buttonSize(200, 30);
      ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 100 + 20);
      if (ImGui::Button(" Add Scripts", buttonSize))
        ImGui::OpenPopup("AddScripts");
      if (ImGui::BeginPopup("AddScripts")) {
        auto& instantiate = Singleton<ScriptManager>::instance()->m_instaniator;
        for (auto& iter : instantiate) {
          if (ImGui::MenuItem(iter.first.c_str())) {
            script->m_scripts.emplace_back(iter.first, iter.second());
            ImGui::CloseCurrentPopup();
          }
        }
        ImGui::EndPopup();
      }
    }
  }

  auto Script::serialize(SerializeData& data) noexcept -> void {
    tinygltf::Model* m = data.model;
    auto node_view = data.gfx_scene->m_registry.view<Script>();
    for (auto [entity, _script] : node_view.each()) {
      int const node_id = data.nodes[entity];

      if (!m->nodes[node_id].extras.IsObject())
        m->nodes[node_id].extras = tinygltf::Value(tinygltf::Value::Object());
      auto& extra = m->nodes[node_id].extras.Get<tinygltf::Value::Object>();
      std::vector<tinygltf::Value> arrays(_script.m_scripts.size());
      size_t offset = 0;
      for (auto& iter: _script.m_scripts) {
        arrays[offset++] = tinygltf::Value(iter.first);
      }
      extra["scripts"] = tinygltf::Value(arrays);
    }
  }

  auto Script::update(Node& node, double delta) noexcept -> void {
    for (auto& iter : m_scripts) {
      if (iter.second->m_initialized == false) {
        iter.second->on_init(node);
        iter.second->m_initialized = true;
      }
      iter.second->on_update(node, delta);
    }
  }

  auto Script::deserialize(DeserializeData& data) noexcept -> void {
    for (size_t i = 0; i < data.model->nodes.size(); ++i) {
      auto& gltfNode = data.model->nodes[i];
      auto& seNode = data.nodes[i];

      if (gltfNode.extras.IsObject()) {
        if (gltfNode.extras.Has("scripts")) {
          auto& script = seNode.add_component<Script>();
          auto& extra = gltfNode.extras.Get<tinygltf::Value::Object>();
          auto& extra_strings = extra["scripts"];
          for (size_t j = 0; j < extra_strings.ArrayLen(); ++j) {
            std::string const& script_name = extra_strings.Get(j).Get<std::string>();
            auto& instantiate = Singleton<ScriptManager>::instance()->m_instaniator;
            for (auto& iter : instantiate) {
              if (iter.first == script_name)
                script.m_scripts.emplace_back(iter.first, iter.second());
            }
          }
        }
      }
    }
  }

}
}