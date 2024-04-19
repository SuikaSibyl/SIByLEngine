#include <se.core.hpp>
#include <se.rhi.hpp>
#include <se.rhi.torch.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
//#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
//#include "include.hpp"
//#include <cuda_runtime.h>
namespace py = pybind11;
using namespace se;

#include <imgui.h>
#include <se.editor.hpp>
#include <seditor.hpp>
#include <seditor-base.hpp>
#include <seditor-gfx.hpp>
#include <seditor-rdg.hpp>
#include <../pyscript/py.define.hpp>

struct rhi_namescope {};
struct gfx_namescope {};
struct rdg_namescope {};
struct imgui_namescope {};

enum ImGuiDockRequestType {
    ImGuiDockRequestType_None = 0,
    ImGuiDockRequestType_Dock,
    ImGuiDockRequestType_Undock,
    ImGuiDockRequestType_Split                  // Split is the same as Dock but without a DockPayload
};

struct ImGuiDockRequest {
    ImGuiDockRequestType    Type;
    ImGuiWindow* DockTargetWindow;   // Destination/Target Window to dock into (may be a loose window or a DockNode, might be NULL in which case DockTargetNode cannot be NULL)
    ImGuiDockNode* DockTargetNode;     // Destination/Target Node to dock into
    ImGuiWindow* DockPayload;        // Source/Payload window to dock (may be a loose window or a DockNode), [Optional]
    ImGuiDir                DockSplitDir;
    float                   DockSplitRatio;
    bool                    DockSplitOuter;
    ImGuiWindow* UndockTargetWindow;
    ImGuiDockNode* UndockTargetNode;
};

struct ImGuiDockNodeSettings {
    ImGuiID             ID;
    ImGuiID             ParentNodeId;
    ImGuiID             ParentWindowId;
    ImGuiID             SelectedTabId;
    signed char         SplitAxis;
    char                Depth;
    ImGuiDockNodeFlags  Flags;                  // NB: We save individual flags one by one in ascii format (ImGuiDockNodeFlags_SavedFlagsMask_)
    ImVec2ih            Pos;
    ImVec2ih            Size;
    ImVec2ih            SizeRef;
};

namespace ImGui {
bool DragInt(const char* label, CPPType<int32_t>& v, float v_speed, int v_min, 
  int v_max, const char* format, ImGuiSliderFlags flags) {
  return DragInt(label, &v.get(), v_speed, v_min, v_max, format, flags);
}

bool DragFloat(const char* label, CPPType<float>& v, float v_speed, float v_min, 
  float v_max, const char* format, ImGuiSliderFlags flags) {
  return DragFloat(label, &v.get(), v_speed, v_min, v_max, format, flags);
}

bool Button(const char* label, se::vec2 size) {
  root::print::debug("hello");
  return Button(label, ImVec2(size.x, size.y));
}

bool Combo(const char* label, CPPType<int32_t>& current_item, 
  std::vector<std::string> const& items, int popup_max_height_in_items = -1) {
  std::vector<const char*> items_char(items.size());
  for (size_t i = 0; i < items.size(); ++i)
    items_char[i] = items[i].c_str();
  return Combo(label, &current_item.get(), items_char.data(), items_char.size(), popup_max_height_in_items);
}
}

PYBIND11_MODULE(pyeditor, m) {
    m.doc() = "SIByLpy :: editor module";

    // Export ImGuiContext struct
    // ------------------------------------------------------------------------
    py::class_<editor::RawImGuiCtx> class_RawImGuiCtx(m, "RawImGuiCtx");
    py::class_<editor::ImGuiContext> class_ImGuiContext(m, "ImGuiContext");
    class_ImGuiContext.def(py::init<>())
      .def_static("initialize", &editor::ImGuiContext::initialize)
      .def_static("finalize", &editor::ImGuiContext::finalize)
      .def_static("startNewFrame", &editor::ImGuiContext::startNewFrame)
      .def_static("startGuiRecording", &editor::ImGuiContext::startGuiRecording)
      .def_static("render", &editor::ImGuiContext::render)
      .def_static("getRawCtx", &editor::ImGuiContext::getRawCtx, py::return_value_policy::reference);

    // Export EditorContext struct
    // ------------------------------------------------------------------------
    py::class_<editor::EditorContext> class_EditorContext(m, "EditorContext");
    class_EditorContext.def(py::init<>())
      .def_static("initialize", &editor::EditorContext::initialize);

    // Export ImGui struct
    // ------------------------------------------------------------------------
    py::class_<imgui_namescope> class_ImGui(m, "ImGui");
    class_ImGui
      .def_static("SetCurrentContext", &ImGui::SetCurrentContext)
      .def_static("Begin", &ImGui::Begin)
      .def_static("End", &ImGui::End)
      .def_static("Button",
       static_cast<bool(*)(const char*, vec2)>(&ImGui::Button),
       py::arg("label"), py::arg("size"))
      .def_static("DragInt",
       static_cast<bool(*)(const char*, CPPType<int32_t>&, float, int, int, const char*, ImGuiSliderFlags)>(&ImGui::DragInt),
       py::arg("label"), py::arg("v"), py::arg("v_speed") = 1.0f, py::arg("v_min") = 0, py::arg("v_max") = 0,
       py::arg("format") = "%d", py::arg("flags") = 0)
      .def_static("DragFloat",
       static_cast<bool(*)(const char*, CPPType<float>&, float, float, float, const char*, ImGuiSliderFlags)>(&ImGui::DragFloat),
       py::arg("label"), py::arg("v"), py::arg("v_speed") = 1.0f, py::arg("v_min") = 0.0f, py::arg("v_max") = 0.0f,
       py::arg("format") = "%.3f", py::arg("flags") = 0)
      .def_static("Combo",
       static_cast<bool(*)(const char*, CPPType<int32_t>&, std::vector<std::string> const&, int)>(&ImGui::Combo),
       py::arg("label"), py::arg("current_items"), py::arg("items"), py::arg("popup_max_height_in_items") = -1);

    // Export EditorBase struct
    // ------------------------------------------------------------------------
    py::class_<editor::InspectorWidget> class_InspectorWidget(m, "InspectorWidget");
    py::class_<editor::SceneWidget> class_SceneWidget(m, "SceneWidget");
    class_SceneWidget.def("bindScene", &editor::SceneWidget::bindScene);

    py::class_<editor::ViewportWidget> class_ViewportWidget(m, "ViewportWidget");
    class_ViewportWidget.def("bindInput", &editor::ViewportWidget::bindInput);
    class_ViewportWidget.def("bindTimer", &editor::ViewportWidget::bindTimer);

    py::class_<editor::RDGViewerWidget> class_RDGViewerWidget(m, "RDGViewerWidget");
    py::class_<editor::StatusWidget> class_StatusWidget(m, "StatusWidget");

    // Export EditorBase struct
    // ------------------------------------------------------------------------
    py::class_<editor::EditorBase> class_EditorBase(m, "EditorBase");
    class_EditorBase.def_static("onImGuiDraw", &editor::EditorBase::onImGuiDraw);
    class_EditorBase.def_static("onUpdate", &editor::EditorBase::onUpdate);
    class_EditorBase.def_static("finalize", &editor::EditorBase::finalize);
    class_EditorBase.def_static("getInspectorWidget", &editor::EditorBase::getInspectorWidget, py::return_value_policy::reference);
    class_EditorBase.def_static("getSceneWidget", &editor::EditorBase::getSceneWidget, py::return_value_policy::reference);
    class_EditorBase.def_static("getViewportWidget", &editor::EditorBase::getViewportWidget, py::return_value_policy::reference);
    class_EditorBase.def_static("getRDGViewerWidget", &editor::EditorBase::getRDGViewerWidget, py::return_value_policy::reference);
    class_EditorBase.def_static("getStatusWidget", &editor::EditorBase::getStatusWidget, py::return_value_policy::reference);
    class_EditorBase.def_static("bindInput", &editor::EditorBase::bindInput);
    class_EditorBase.def_static("bindTimer", &editor::EditorBase::bindTimer);
    class_EditorBase.def_static("bindPipeline", &editor::EditorBase::bindPipeline);
    class_EditorBase.def_static("bindScene", &editor::EditorBase::bindScene);

    //py::class_<root::print>(class_root, "print")
    //  .def(py::init<>())
    //  .def_static("debug", &root::print::debug)
    //  .def_static("log", &root::print::log)
    //  .def_static("warning", &root::print::warning)
    //  .def_static("error", &root::print::error)
    //  .def_static("correct", &root::print::correct);



    //// Export root struct
    //// ------------------------------------------------------------------------
    //// add root class definition
    //py::class_<root> class_root(m, "root");
    //// add print module to root
    //py::class_<root::print>(class_root, "print")
    //    .def(py::init<>())
    //    .def_static("debug", &root::print::debug)
    //    .def_static("log", &root::print::log)
    //    .def_static("warning", &root::print::warning)
    //    .def_static("error", &root::print::error)
    //    .def_static("correct", &root::print::correct);
}