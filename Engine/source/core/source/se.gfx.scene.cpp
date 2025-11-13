#include "se.gfx.hpp"
#include "se.editor.hpp"
#include <imgui.h>

namespace se {
  namespace gfx {
    Scene::Scene() { reset(); }

    auto Scene::create_node(std::string const& name) noexcept -> Node {
      auto entity = m_registry.create();
      auto node = Node{ entity, &m_registry };
      m_registry.emplace<NodeProperty>(entity, name);
      auto& transform = m_registry.emplace<Transform>(entity);
      transform.m_dirtyToFile = false; transform.m_dirtyToGPU = true;
      return node;
    }

    auto Scene::create_node(Node parent, std::string const& name) noexcept -> Node {
      auto entity = m_registry.create();
      auto node = Node{ entity, &m_registry };
      m_registry.emplace<NodeProperty>(entity, name);
      m_registry.get<NodeProperty>(parent.m_entity).children.push_back(node);
      return node;
    }

    auto Scene::set_viewport_size(ivec2 size) noexcept -> void {
      m_viewportSize = size;
    }
    
    auto Scene::reset() noexcept -> void {
      m_registry = ex::registry{};
      m_roots.clear();
      m_filepath = "";
      m_name = "";
      m_gpuScene = {};

      m_perSceneGPUInfo.reset();
      m_gpuScene.reset();
      m_timer.update();
    }

    auto draw_scene_node(gfx::Scene* scene, gfx::Node node, editor::IFragment* fragment) {
      NodeProperty* _property = node.get_component<NodeProperty>();
      gfx::ComponentManager::draw_all_components(node);

      {  // add component
        ImGui::Separator();
        ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
        ImVec2 buttonSize(200, 30);
        ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 100 + 20);
        if (ImGui::Button(" Add Component", buttonSize))
          ImGui::OpenPopup("AddComponent");
        if (ImGui::BeginPopup("AddComponent")) {

          for (auto& pair : Singleton<ComponentManager>::instance()->m_components) {
            void* comp_ = pair.second.retrival(node);
            if (comp_ == nullptr) {
              if (ImGui::MenuItem(pair.second.name.c_str())) {
                pair.second.add(node);
                ImGui::CloseCurrentPopup();
              }
            }
          }
          ImGui::EndPopup();
        }
      }
    }

    auto drawNode(gfx::Node const& node, gfx::Scene* scene) -> bool {
      ImGui::PushID(uint32_t(node.m_entity));
      ImGuiTreeNodeFlags node_flags = 0;
      gfx::NodeProperty* nodeprop = node.get_component<gfx::NodeProperty>();
      if (nodeprop->children.size() == 0)
        node_flags |= ImGuiTreeNodeFlags_Leaf;
      //if (node.entity == widget->forceNodeOpen.entity && node.registry == widget->forceNodeOpen.registry) {
      //  ImGui::SetNextItemOpen(true, ImGuiCond_Always);
      //  widget->forceNodeOpen = {};
      //}
      std::string name = nodeprop->name.c_str();
      if (name == "") name = "$NAMELESS NODE$";
      bool opened = ImGui::TreeNodeEx(name.c_str(), node_flags);
      ImGuiID uid = ImGui::GetID((name + std::to_string(std::uint32_t(node.m_entity))).c_str());
      //ImGui::TreeNodeBehaviorIsOpen(uid);
      // Clicked
      if (ImGui::IsItemClicked()) {
        std::function<void()> fn = std::bind(&draw_scene_node, scene, node, nullptr);
        se::editor::EditorContext::set_inspector_callback(fn);
      }
      // Opened
      if (opened) {
        ImGui::NextColumn();
        for (int i = 0; i < nodeprop->children.size(); i++) {
          drawNode(nodeprop->children[i], scene);
        }
        ImGui::TreePop();
      }
      ImGui::PopID();
      return false;
    }

    auto Scene::draw_gui(editor::IFragment* fragment) noexcept -> void {
      ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
      ImGui::SeparatorText("Statistics ");
      ImGui::Text("FPS: %.2f", 1. / m_timer.delta_time());
      ImGui::SeparatorText("scene hierarchy");
      
      auto save_scene = [&]() {
        std::string name = m_name + ".gltf";
        std::string path = Platform::save_file(nullptr, name);
        if (path != "") {
          //scene->serialize(path);
          //scene->isDirty = false;
        }
      };
      // draw the menubar
      if (ImGui::BeginMenuBar()) {
        if (ImGui::Button("Load")) {
          std::string load_path = Platform::open_file("",
            Configuration::string_property("project_path"));
          if (load_path != "") {
            reset();
            std::string extension = Filesys::get_extension(load_path);
            if (extension == ".gltf")
              load_gltf(load_path);
            else {
              se::error("Reload scene with unknown file extension {}", extension);
            }
            update_gpu_scene();
          };
        }
        if (ImGui::Button("Save")) {
          std::string save_path = Platform::save_file("", m_filepath);
          save(save_path);
        }

        ImGui::EndMenuBar();
      }
      ImGui::PopItemWidth();

      // Left-clock on blank space
      if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
        se::editor::EditorContext::clear_inspector_callback();
      }

      // Detect right-click on window background (but not on items/widgets)
      if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup) &&
        !ImGui::IsAnyItemHovered() &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        // Your right-click background handler
        ImGui::OpenPopup("MyBackgroundPopup");
      }

      if (ImGui::BeginPopup("MyBackgroundPopup")) {
        if (ImGui::MenuItem("Create Empty Entity")) {
          m_roots.push_back(create_node("new node"));
        }
        ImGui::EndPopup();
      }


      for (auto& node : m_roots)
        drawNode(node, this);
    }

    auto Scene::open_node_with_geometry_index(int32_t index) noexcept -> void {
      for (auto& iter : m_gpuScene.geometryList) {
        for (auto& index_info : iter.second) {
          int geometryID = index_info.assignedIndex;
          if (geometryID == index) {
            Node node = { iter.first.entity, &m_registry };
            std::function<void()> fn = std::bind(&draw_scene_node, this, node, nullptr);
            se::editor::EditorContext::set_inspector_callback(fn);
          } 
        }
      }
    }
}
}