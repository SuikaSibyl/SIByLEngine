#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/wstring.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/trampoline.h>
#include <nanobind/ndarray.h>
#include <string>
#include <se.utils.hpp>
#include <se.rhi.hpp>
#include "se.rhi.cuda.hpp"
#include <se.gfx.hpp>
#include <se.editor.hpp>
#include <imgui.h>
#include "../addon/pass-postprocess/ex.pass.postprocess.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace se {
  inline void info_str(std::string const& s) { return info(s); }
  inline void warn_str(std::string const& s) { return warn(s); }
  inline void trace_str(std::string const& s) { return trace(s); }
  inline void debug_str(std::string const& s) { return debug(s); }
  inline void error_str(std::string const& s) { return error(s); }
  inline void critical_str(std::string const& s) { return critical(s); }

  template<class Type>
  struct CPPType {
    Type value;
    CPPType() = default;
    CPPType(Type const& v) :value(v) {}
    auto get() noexcept -> Type& { return value; }
    auto set(Type const& v) noexcept -> void { value = v; }
  };
}

struct namespace_rhi {};
struct namespace_gfx {};
struct namespace_editor {};
struct namespace_rdg {};
struct namespace_imgui {};
struct namespace_addon {};

#define BIND_BITMASK_FLAGS(NameSpace, EnumType, EnumName)                                \
  nb::class_<se::Flags<EnumType>>(NameSpace, EnumName)                                   \
  .def(nb::init<EnumType>(), nb::is_implicit())                                          \
  .def("__or__", [](const se::Flags<EnumType>& a, const EnumType& b) {                   \
    return a | b; }, nb::is_operator())                                                  \
    .def("__or__", [](const se::Flags<EnumType>& a, const se::Flags<EnumType>& b) {      \
    return a | b; }, nb::is_operator());                                                 \


namespace se {
namespace rdg{
  template <class Base = Pass>
struct PyPass : Base {
    NB_TRAMPOLINE(Base, 6);  // Number of methods overrideable from Python
    auto reflect(PassReflection& reflector) noexcept -> PassReflection override {
        NB_OVERRIDE(reflect, reflector); }
    auto execute(RenderContext* context, RenderData const& renderData) noexcept -> void override {
        NB_OVERRIDE(execute, context, renderData); }
    auto readback(RenderData const& renderData) noexcept -> void override {
        NB_OVERRIDE(readback, renderData); }
    auto render_ui() noexcept -> void override {
        NB_OVERRIDE(render_ui); }
};

template <class PipelinePassBase = se::rdg::PipelinePass>
struct PyPipelinePass : PyPass<PipelinePassBase> {
  using PyPass<PipelinePassBase>::PyPass; // Inherit constructors
};

template <class RenderPassBase = se::rdg::RenderPass>
struct PyRenderPass : PyPipelinePass<RenderPassBase> {
  NB_TRAMPOLINE(PyPipelinePass<RenderPassBase>, 6);
};

 //template <class FullScreenPassBase = se::rdg::FullScreenPass>
 //struct PyFullScreenPass : PyRenderPass<FullScreenPassBase> {
 //  using PyRenderPass<FullScreenPassBase>::PyRenderPass; // Inherit constructors
 //};

template <class ComputePassBase = se::rdg::ComputePass>
struct PyComputePass : PyPipelinePass<ComputePassBase> {
  NB_TRAMPOLINE(PyPipelinePass<ComputePassBase>, 6);
};

template <class GraphBase = se::rdg::Graph>
struct PyGraph : GraphBase {
  NB_TRAMPOLINE(GraphBase, 3);
  virtual auto render_ui() noexcept -> void
  { NB_OVERRIDE(render_ui); };
};

//template <class PipelineBase = se::rdg::Pipeline>
//struct PyPipeline : PipelineBase {
//  using PipelineBase::PipelineBase; // Inherit constructors
//  virtual auto build() noexcept -> void
//  { PYBIND11_OVERRIDE_PURE(void, PipelineBase, build); };
//  virtual auto execute(rhi::CommandEncoder* encoder) noexcept -> void
//  { PYBIND11_OVERRIDE_PURE(void, PipelineBase, execute, encoder); };
//  virtual auto readback() noexcept -> void
//  { PYBIND11_OVERRIDE(void, PipelineBase, readback); };
//  virtual auto renderUI() noexcept -> void
//  { PYBIND11_OVERRIDE(void, PipelineBase, renderUI); };
//  virtual auto getAllGraphs() noexcept -> std::vector<Graph*>
//  { PYBIND11_OVERRIDE_PURE(std::vector<Graph*>, PipelineBase, getAllGraphs); };
//  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*>
//  { PYBIND11_OVERRIDE_PURE(std::vector<Graph*>, PipelineBase, getActiveGraphs); };
//  virtual auto getOutput() noexcept -> gfx::TextureHandle
//  { PYBIND11_OVERRIDE_PURE(gfx::TextureHandle, PipelineBase, getOutput); };
//};
//
//template <class SingleGraphPipelineBase = se::rdg::SingleGraphPipeline>
//struct PySingleGraphPipeline : SingleGraphPipelineBase {
//  using SingleGraphPipelineBase::SingleGraphPipelineBase; // Inherit constructors
//  virtual auto build() noexcept -> void
//  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, build); };
//  virtual auto execute(rhi::CommandEncoder* encoder) noexcept -> void
//  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, execute, encoder); };
//  virtual auto readback() noexcept -> void
//  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, readback); };
//  virtual auto renderUI() noexcept -> void
//  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, renderUI); };
//  virtual auto getAllGraphs() noexcept -> std::vector<Graph*>
//  { PYBIND11_OVERRIDE(std::vector<Graph*>, SingleGraphPipelineBase, getAllGraphs); };
//  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*>
//  { PYBIND11_OVERRIDE(std::vector<Graph*>, SingleGraphPipelineBase, getActiveGraphs); };
//  virtual auto getOutput() noexcept -> gfx::TextureHandle
//  { PYBIND11_OVERRIDE(gfx::TextureHandle, SingleGraphPipelineBase, getOutput); };
//};
}
}

NB_MODULE(pycore, m) {
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ log                                                                       ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  m.def("info", &se::info_str);
  m.def("warn", &se::warn_str);
  m.def("trace", &se::trace_str);
  m.def("debug", &se::debug_str);
  m.def("error", &se::error_str);
  m.def("critical", &se::critical_str);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ memory                                                                    ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::MiniBuffer>(m, "MiniBuffer");

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ file                                                                      ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::Filesys>(m, "Filesys")
    .def_static("sync_read_file", &se::Filesys::sync_read_file)
    .def_static("sync_write_file", &se::Filesys::sync_write_file)
    .def_static("get_executable_path", &se::Filesys::get_executable_path)
    .def_static("get_parent_path", &se::Filesys::get_parent_path)
    .def_static("get_stem", &se::Filesys::get_stem)
    .def_static("get_filename", &se::Filesys::get_filename)
    .def_static("get_absolute_path", &se::Filesys::get_absolute_path)
    .def_static("file_exists", &se::Filesys::file_exists)
    .def_static("resolve_path", &se::Filesys::resolve_path);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource                                                                  ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::Resources>(m, "Resources")
    .def_static("query_runtime_uid", &se::Resources::query_runtime_uid)
    .def_static("query_string_uid", (se::UID(*)(const std::string&)) & se::Resources::query_string_uid);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ platform                                                                  ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::Platform>(m, "Platform")
    .def_static("open_file", &se::Platform::open_file)
    .def_static("save_file", &se::Platform::save_file)
    .def_static("string_cast", (std::wstring(*)(const std::string&)) & se::Platform::string_cast)
    .def_static("string_cast", (std::string(*)(const std::wstring&)) & se::Platform::string_cast);

  nb::class_<se::timer>(m, "timer")
    .def(nb::init<>())
    .def("update", &se::timer::update)
    .def("delta_time", &se::timer::delta_time)
    .def("total_time", &se::timer::total_time);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ configuration                                                             ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::Configuration>(m, "Configuration")
    .def_static("set_macro", &se::Configuration::set_macro)
    .def_static("set_config_file", &se::Configuration::set_config_file)
    .def_static("string_property", &se::Configuration::string_property)
    .def_static("string_array_property", &se::Configuration::string_array_property)
    .def_static("on_draw_gui", &se::Configuration::on_draw_gui);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ window                                                                    ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::Window>(m, "Window")
    .def(nb::init<size_t, size_t, std::wstring const&>())
    .def("is_running", &se::Window::is_running)
    .def("fetch_events", &se::Window::fetch_events)
    .def("destroy", &se::Window::destroy)
    .def("is_resized", &se::Window::is_resized)
    .def("get_width", &se::Window::get_width)
    .def("get_height", &se::Window::get_height)
    .def("get_high_dpi", &se::Window::get_high_dpi)
    .def("is_iconified", &se::Window::is_iconified)
    .def("resize", &se::Window::resize);

  auto struct_input = nb::class_<se::Input>(m, "Input");

  auto input_enum_bind = nb::enum_<se::Input::CodeEnum>(struct_input, "CodeEnum");
  { constexpr auto entries = magic_enum::enum_entries<se::Input::CodeEnum>();
  for (auto& iter : entries) input_enum_bind.value(std::string(iter.second).c_str(), iter.first); }

  struct_input.def("is_key_pressed", &se::Input::is_key_pressed)
    .def("get_mouse_x", &se::Input::get_mouse_x)
    .def("get_mouse_y", &se::Input::get_mouse_y)
    .def("enable_cursor", &se::Input::enable_cursor)
    .def("disable_cursor", &se::Input::disable_cursor)
    .def("get_mouse_scroll_x", &se::Input::get_mouse_scroll_x)
    .def("get_mouse_scroll_y", &se::Input::get_mouse_scroll_y)
    .def("is_mouse_button_pressed", &se::Input::is_mouse_button_pressed);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ math                                                                      ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<se::ivec2>(m, "ivec2")
    .def(nb::init<int, int>())
    .def_rw("x", &se::ivec2::x)
    .def_rw("y", &se::ivec2::y);
  nb::class_<se::ivec3>(m, "ivec3")
    .def(nb::init<int, int, int>())
    .def_rw("x", &se::ivec3::x)
    .def_rw("y", &se::ivec3::y)
    .def_rw("z", &se::ivec3::z);
  nb::class_<se::ivec4>(m, "ivec4")
    .def(nb::init<int, int, int, int>())
    .def_rw("x", &se::ivec4::x)
    .def_rw("y", &se::ivec4::y)
    .def_rw("z", &se::ivec4::z)
    .def_rw("w", &se::ivec4::w);

  nb::class_<se::vec2>(m, "vec2")
    .def(nb::init<float, float>())
    .def_rw("x", &se::vec2::x)
    .def_rw("y", &se::vec2::y);
  nb::class_<se::vec3>(m, "vec3")
    .def(nb::init<float, float, float>())
    .def_rw("x", &se::vec3::x)
    .def_rw("y", &se::vec3::y)
    .def_rw("z", &se::vec3::z);
  nb::class_<se::vec4>(m, "vec4")
    .def(nb::init<float, float, float, float>())
    .def_rw("x", &se::vec4::x)
    .def_rw("y", &se::vec4::y)
    .def_rw("z", &se::vec4::z)
    .def_rw("w", &se::vec4::w);

  nb::class_<se::point3>(m, "point3")
    .def(nb::init<float, float, float>())
    .def(nb::init<se::vec3>());

  nb::class_<se::bounds3>(m, "bounds3")
    .def(nb::init<se::point3>())
    .def(nb::init<se::point3, se::point3>());

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ rhi                                                                       ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ Context Interface for Graphics-API.  							                       ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  auto ns_rhi = nb::class_<::namespace_rhi>(m, "rhi");

  nb::enum_<se::rhi::ContextExtensionEnum>(ns_rhi, "ContextExtensionEnum")
    .value("NONE", se::rhi::ContextExtensionEnum::NONE)
    .value("DEBUG_UTILS", se::rhi::ContextExtensionEnum::DEBUG_UTILS)
    .value("MESH_SHADER", se::rhi::ContextExtensionEnum::MESH_SHADER)
    .value("FRAGMENT_BARYCENTRIC", se::rhi::ContextExtensionEnum::FRAGMENT_BARYCENTRIC)
    .value("SAMPLER_FILTER_MIN_MAX", se::rhi::ContextExtensionEnum::SAMPLER_FILTER_MIN_MAX)
    .value("RAY_TRACING", se::rhi::ContextExtensionEnum::RAY_TRACING)
    .value("SHADER_NON_SEMANTIC_INFO", se::rhi::ContextExtensionEnum::SHADER_NON_SEMANTIC_INFO)
    .value("BINDLESS_INDEXING", se::rhi::ContextExtensionEnum::BINDLESS_INDEXING)
    .value("ATOMIC_FLOAT", se::rhi::ContextExtensionEnum::ATOMIC_FLOAT)
    .value("CONSERVATIVE_RASTERIZATION", se::rhi::ContextExtensionEnum::CONSERVATIVE_RASTERIZATION)
    .value("COOPERATIVE_MATRIX", se::rhi::ContextExtensionEnum::COOPERATIVE_MATRIX)
    .value("CUDA_INTEROPERABILITY", se::rhi::ContextExtensionEnum::CUDA_INTEROPERABILITY)
    .value("USE_AFTERMATH", se::rhi::ContextExtensionEnum::USE_AFTERMATH)
    .def("__or__", [](se::rhi::ContextExtensionEnum a, se::rhi::ContextExtensionEnum b) {
    return se::Flags<se::rhi::ContextExtensionEnum>(a | b);
      }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::ContextExtensionEnum, "ContextExtensions");

  nb::enum_<se::rhi::PowerPreferenceEnum>(ns_rhi, "PowerPreferenceEnum")
    .value("LOW_POWER", se::rhi::PowerPreferenceEnum::LOW_POWER)
    .value("HIGH_PERFORMANCE", se::rhi::PowerPreferenceEnum::HIGH_PERFORMANCE);

  nb::class_<se::rhi::Context>(ns_rhi, "Context")
    .def(nb::init<se::Window*, se::Flags<se::rhi::ContextExtensionEnum>>())
    .def("request_adapter", &se::rhi::Context::request_adapter,
      "pp"_a = se::rhi::PowerPreferenceEnum::HIGH_PERFORMANCE)
    .def("get_binded_window", &se::rhi::Context::get_binded_window,
      nb::rv_policy::reference);

  nb::class_<se::rhi::Adapter>(ns_rhi, "Adapter")
    .def("request_device", &se::rhi::Adapter::request_device);

  nb::enum_<se::rhi::PipelineStageEnum>(ns_rhi, "PipelineStageEnum")
    .value("TOP_OF_PIPE_BIT", se::rhi::PipelineStageEnum::TOP_OF_PIPE_BIT)
    .value("DRAW_INDIRECT_BIT", se::rhi::PipelineStageEnum::DRAW_INDIRECT_BIT)
    .value("VERTEX_INPUT_BIT", se::rhi::PipelineStageEnum::VERTEX_INPUT_BIT)
    .value("VERTEX_SHADER_BIT", se::rhi::PipelineStageEnum::VERTEX_SHADER_BIT)
    .value("TESSELLATION_CONTROL_SHADER_BIT", se::rhi::PipelineStageEnum::TESSELLATION_CONTROL_SHADER_BIT)
    .value("TESSELLATION_EVALUATION_SHADER_BIT", se::rhi::PipelineStageEnum::TESSELLATION_EVALUATION_SHADER_BIT)
    .value("GEOMETRY_SHADER_BIT", se::rhi::PipelineStageEnum::GEOMETRY_SHADER_BIT)
    .value("FRAGMENT_SHADER_BIT", se::rhi::PipelineStageEnum::FRAGMENT_SHADER_BIT)
    .value("EARLY_FRAGMENT_TESTS_BIT", se::rhi::PipelineStageEnum::EARLY_FRAGMENT_TESTS_BIT)
    .value("LATE_FRAGMENT_TESTS_BIT", se::rhi::PipelineStageEnum::LATE_FRAGMENT_TESTS_BIT)
    .value("COLOR_ATTACHMENT_OUTPUT_BIT", se::rhi::PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT)
    .value("COMPUTE_SHADER_BIT", se::rhi::PipelineStageEnum::COMPUTE_SHADER_BIT)
    .value("TRANSFER_BIT", se::rhi::PipelineStageEnum::TRANSFER_BIT)
    .value("BOTTOM_OF_PIPE_BIT", se::rhi::PipelineStageEnum::BOTTOM_OF_PIPE_BIT)
    .value("HOST_BIT", se::rhi::PipelineStageEnum::HOST_BIT)
    .value("ALL_GRAPHICS_BIT", se::rhi::PipelineStageEnum::ALL_GRAPHICS_BIT)
    .value("ALL_COMMANDS_BIT", se::rhi::PipelineStageEnum::ALL_COMMANDS_BIT)
    .value("TRANSFORM_FEEDBACK_BIT_EXT", se::rhi::PipelineStageEnum::TRANSFORM_FEEDBACK_BIT_EXT)
    .value("CONDITIONAL_RENDERING_BIT_EXT", se::rhi::PipelineStageEnum::CONDITIONAL_RENDERING_BIT_EXT)
    .value("ACCELERATION_STRUCTURE_BUILD_BIT_KHR", se::rhi::PipelineStageEnum::ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
    .value("RAY_TRACING_SHADER_BIT_KHR", se::rhi::PipelineStageEnum::RAY_TRACING_SHADER_BIT_KHR)
    .value("TASK_SHADER_BIT_NV", se::rhi::PipelineStageEnum::TASK_SHADER_BIT_NV)
    .value("MESH_SHADER_BIT_NV", se::rhi::PipelineStageEnum::MESH_SHADER_BIT_NV)
    .value("FRAGMENT_DENSITY_PROCESS_BIT", se::rhi::PipelineStageEnum::FRAGMENT_DENSITY_PROCESS_BIT)
    .value("FRAGMENT_SHADING_RATE_ATTACHMENT_BIT", se::rhi::PipelineStageEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_BIT)
    .value("COMMAND_PREPROCESS_BIT", se::rhi::PipelineStageEnum::COMMAND_PREPROCESS_BIT)
    .def("__or__", [](se::rhi::PipelineStageEnum a, se::rhi::PipelineStageEnum b) {
    return se::Flags<se::rhi::PipelineStageEnum>(a | b); }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::PipelineStageEnum, "PipelineStages");

  nb::class_<se::rhi::Queue>(ns_rhi, "Queue")
    .def("submit", nb::overload_cast<std::vector<se::rhi::CommandBuffer*> const&,
      se::rhi::Semaphore*, se::rhi::Semaphore*, se::rhi::Fence*>(&se::rhi::Queue::submit),
      "commandBuffers"_a, "wait"_a = nullptr,
      "signal"_a = nullptr, "fence"_a = nullptr)
    .def("submit", nb::overload_cast<std::vector<se::rhi::CommandBuffer*> const&,
      std::vector<se::rhi::Semaphore*> const&, std::vector<size_t> const&,
      std::vector<se::Flags<se::rhi::PipelineStageEnum>> const&,
      std::vector<se::rhi::Semaphore*> const&, std::vector<size_t> const&, 
      se::rhi::Fence*>(&se::rhi::Queue::submit), "commandBuffers"_a, 
      "wait_semaphores"_a = std::vector<se::rhi::Semaphore*>{}, "wait_indices"_a = std::vector<size_t>{},
      "wait_stages"_a = std::vector < se::Flags<se::rhi::PipelineStageEnum>>{},
      "signal_semaphores"_a = std::vector<se::rhi::Semaphore*>{}, "signal_indices"_a = std::vector<size_t>{},
      nb::arg("fence").none() = nb::none());

  auto rhi_device = nb::class_<se::rhi::Device>(ns_rhi, "Device")
    .def("wait_idle", &se::rhi::Device::wait_idle)
    .def("from_which_adapter", &se::rhi::Device::from_which_adapter, nb::rv_policy::reference)
    .def("get_graphics_queue", &se::rhi::Device::get_graphics_queue, nb::rv_policy::reference)
    .def("get_compute_queue", &se::rhi::Device::get_compute_queue, nb::rv_policy::reference)
    .def("get_present_queue", &se::rhi::Device::get_present_queue, nb::rv_policy::reference)
    .def("create_swapchain", &se::rhi::Device::create_swapchain)
    .def("wait_idle", &se::rhi::Device::wait_idle)
    .def("create_shader_module", &se::rhi::Device::create_shader_module)
    .def("create_frame_resources", &se::rhi::Device::create_frame_resources)
    .def("allocate_command_buffer", &se::rhi::Device::allocate_command_buffer);

  nb::class_<se::rhi::CommandPool>(ns_rhi, "CommandPool")
    .def("allocate_command_buffer", &se::rhi::CommandPool::allocate_command_buffer);

  nb::class_<se::rhi::CommandBuffer>(ns_rhi, "CommandBuffer");

  nb::enum_<se::rhi::BufferUsageEnum>(ns_rhi, "BufferUsageEnum")
    .value("MAP_READ", se::rhi::BufferUsageEnum::MAP_READ)
    .value("MAP_WRITE", se::rhi::BufferUsageEnum::MAP_WRITE)
    .value("COPY_SRC", se::rhi::BufferUsageEnum::COPY_SRC)
    .value("COPY_DST", se::rhi::BufferUsageEnum::COPY_DST)
    .value("INDEX", se::rhi::BufferUsageEnum::INDEX)
    .value("VERTEX", se::rhi::BufferUsageEnum::VERTEX)
    .value("UNIFORM", se::rhi::BufferUsageEnum::UNIFORM)
    .value("STORAGE", se::rhi::BufferUsageEnum::STORAGE)
    .value("INDIRECT", se::rhi::BufferUsageEnum::INDIRECT)
    .value("QUERY_RESOLVE", se::rhi::BufferUsageEnum::QUERY_RESOLVE)
    .value("SHADER_DEVICE_ADDRESS", se::rhi::BufferUsageEnum::SHADER_DEVICE_ADDRESS)
    .value("ACCELERATION_STRUCTURE_STORAGE", se::rhi::BufferUsageEnum::ACCELERATION_STRUCTURE_STORAGE)
    .value("ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY", se::rhi::BufferUsageEnum::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
    .value("SHADER_BINDING_TABLE", se::rhi::BufferUsageEnum::SHADER_BINDING_TABLE)
    .value("CUDA_ACCESS", se::rhi::BufferUsageEnum::CUDA_ACCESS)
    .def("__or__", [](se::rhi::BufferUsageEnum a, se::rhi::BufferUsageEnum b) {
    return se::Flags<se::rhi::BufferUsageEnum>(a | b);
      }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::BufferUsageEnum, "BufferUsages");

  nb::enum_<se::rhi::BufferShareMode>(ns_rhi, "BufferShareMode")
    .value("CONCURRENT", se::rhi::BufferShareMode::CONCURRENT)
    .value("EXCLUSIVE", se::rhi::BufferShareMode::EXCLUSIVE);

  nb::enum_<se::rhi::MemoryPropertyEnum>(ns_rhi, "MemoryPropertyEnum")
    .value("DEVICE_LOCAL_BIT", se::rhi::MemoryPropertyEnum::DEVICE_LOCAL_BIT)
    .value("HOST_VISIBLE_BIT", se::rhi::MemoryPropertyEnum::HOST_VISIBLE_BIT)
    .value("HOST_COHERENT_BIT", se::rhi::MemoryPropertyEnum::HOST_COHERENT_BIT)
    .value("HOST_CACHED_BIT", se::rhi::MemoryPropertyEnum::HOST_CACHED_BIT)
    .value("LAZILY_ALLOCATED_BIT", se::rhi::MemoryPropertyEnum::LAZILY_ALLOCATED_BIT)
    .value("PROTECTED_BIT", se::rhi::MemoryPropertyEnum::PROTECTED_BIT)
    .value("FLAG_BITS_MAX_ENUM", se::rhi::MemoryPropertyEnum::FLAG_BITS_MAX_ENUM)
    .def("__or__", [](se::rhi::MemoryPropertyEnum a, se::rhi::MemoryPropertyEnum b) {
    return se::Flags<se::rhi::MemoryPropertyEnum>(a | b); }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::MemoryPropertyEnum, "MemoryPropertys");

  nb::class_<se::rhi::Buffer>(ns_rhi, "Buffer")
    .def("size", &se::rhi::Buffer::size);

  nb::class_<se::rhi::BufferDescriptor>(ns_rhi, "BufferDescriptor")
    .def(nb::init<size_t, se::Flags<se::rhi::BufferUsageEnum>,
      se::rhi::BufferShareMode, se::Flags<se::rhi::MemoryPropertyEnum>, bool, int>(),
      nb::arg("size"), nb::arg("usage") = se::Flags<se::rhi::BufferUsageEnum>(0), nb::arg("shareMode") = se::rhi::BufferShareMode::EXCLUSIVE,
      nb::arg("memoryProperties") = se::Flags<se::rhi::MemoryPropertyEnum>(0),
      nb::arg("mappedAtCreation") = false, nb::arg("minimumAlignment") = -1);
  rhi_device.def("create_buffer", &se::rhi::Device::create_buffer);

  nb::class_<se::rhi::SwapChain>(ns_rhi, "SwapChain");

  nb::class_<se::rhi::Semaphore>(ns_rhi, "Semaphore")
    .def("current_host", &se::rhi::Semaphore::current_host)
    .def("current_device", &se::rhi::Semaphore::current_device)
    .def("signal", &se::rhi::Semaphore::signal)
    .def("wait", &se::rhi::Semaphore::wait)
    .def("get_handle", &se::rhi::Semaphore::get_handle);
  rhi_device.def("create_semaphore", &se::rhi::Device::create_semaphore);

  nb::class_<se::rhi::Fence>(ns_rhi, "Fence")
    .def("wait", &se::rhi::Fence::wait)
    .def("reset", &se::rhi::Fence::reset);

  nb::class_<se::rhi::FrameResources>(ns_rhi, "FrameResources")
    .def("frame_start", &se::rhi::FrameResources::frame_start)
    .def("frame_end", &se::rhi::FrameResources::frame_end)
    .def("get_flight_index", &se::rhi::FrameResources::get_flight_index)
    .def("get_swapchain_index", &se::rhi::FrameResources::get_swapchain_index)
    .def("get_command_buffer", &se::rhi::FrameResources::get_command_buffer, nb::rv_policy::reference)
    .def("get_image_available_semaphore", &se::rhi::FrameResources::get_image_available_semaphore, nb::rv_policy::reference)
    .def("get_render_finished_semaphore", &se::rhi::FrameResources::get_render_finished_semaphore, nb::rv_policy::reference)
    .def("get_fence", &se::rhi::FrameResources::get_fence, nb::rv_policy::reference)
    .def("reset", &se::rhi::FrameResources::reset);

  nb::enum_<se::rhi::AccessFlagEnum>(ns_rhi, "AccessFlagEnum")
    .value("INDIRECT_COMMAND_READ_BIT", se::rhi::AccessFlagEnum::INDIRECT_COMMAND_READ_BIT)
    .value("INDEX_READ_BIT", se::rhi::AccessFlagEnum::INDEX_READ_BIT)
    .value("VERTEX_ATTRIBUTE_READ_BIT", se::rhi::AccessFlagEnum::VERTEX_ATTRIBUTE_READ_BIT)
    .value("UNIFORM_READ_BIT", se::rhi::AccessFlagEnum::UNIFORM_READ_BIT)
    .value("INPUT_ATTACHMENT_READ_BIT", se::rhi::AccessFlagEnum::INPUT_ATTACHMENT_READ_BIT)
    .value("SHADER_READ_BIT", se::rhi::AccessFlagEnum::SHADER_READ_BIT)
    .value("SHADER_WRITE_BIT", se::rhi::AccessFlagEnum::SHADER_WRITE_BIT)
    .value("COLOR_ATTACHMENT_READ_BIT", se::rhi::AccessFlagEnum::COLOR_ATTACHMENT_READ_BIT)
    .value("COLOR_ATTACHMENT_WRITE_BIT", se::rhi::AccessFlagEnum::COLOR_ATTACHMENT_WRITE_BIT)
    .value("DEPTH_STENCIL_ATTACHMENT_READ_BIT", se::rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    .value("DEPTH_STENCIL_ATTACHMENT_WRITE_BIT", se::rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
    .value("TRANSFER_READ_BIT", se::rhi::AccessFlagEnum::TRANSFER_READ_BIT)
    .value("TRANSFER_WRITE_BIT", se::rhi::AccessFlagEnum::TRANSFER_WRITE_BIT)
    .value("HOST_READ_BIT", se::rhi::AccessFlagEnum::HOST_READ_BIT)
    .value("HOST_WRITE_BIT", se::rhi::AccessFlagEnum::HOST_WRITE_BIT)
    .value("MEMORY_READ_BIT", se::rhi::AccessFlagEnum::MEMORY_READ_BIT)
    .value("MEMORY_WRITE_BIT", se::rhi::AccessFlagEnum::MEMORY_WRITE_BIT)
    .value("TRANSFORM_FEEDBACK_WRITE_BIT", se::rhi::AccessFlagEnum::TRANSFORM_FEEDBACK_WRITE_BIT)
    .value("TRANSFORM_FEEDBACK_COUNTER_READ_BIT", se::rhi::AccessFlagEnum::TRANSFORM_FEEDBACK_COUNTER_READ_BIT)
    .value("TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT", se::rhi::AccessFlagEnum::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT)
    .value("CONDITIONAL_RENDERING_READ_BIT", se::rhi::AccessFlagEnum::CONDITIONAL_RENDERING_READ_BIT)
    .value("COLOR_ATTACHMENT_READ_NONCOHERENT_BIT", se::rhi::AccessFlagEnum::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT)
    .value("ACCELERATION_STRUCTURE_READ_BIT", se::rhi::AccessFlagEnum::ACCELERATION_STRUCTURE_READ_BIT)
    .value("ACCELERATION_STRUCTURE_WRITE_BIT", se::rhi::AccessFlagEnum::ACCELERATION_STRUCTURE_WRITE_BIT)
    .value("FRAGMENT_DENSITY_MAP_READ_BIT", se::rhi::AccessFlagEnum::FRAGMENT_DENSITY_MAP_READ_BIT)
    .value("FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT", se::rhi::AccessFlagEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT)
    .value("COMMAND_PREPROCESS_READ_BIT", se::rhi::AccessFlagEnum::COMMAND_PREPROCESS_READ_BIT)
    .value("COMMAND_PREPROCESS_WRITE_BIT", se::rhi::AccessFlagEnum::COMMAND_PREPROCESS_WRITE_BIT)
    .value("NONE", se::rhi::AccessFlagEnum::NONE)
    .def("__or__", [](se::rhi::AccessFlagEnum a, se::rhi::AccessFlagEnum b) {
    return se::Flags<se::rhi::AccessFlagEnum>(a | b); }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::AccessFlagEnum, "AccessFlags");

  nb::enum_<se::rhi::CompareFunction>(ns_rhi, "CompareFunction")
    .value("NEVER", se::rhi::CompareFunction::NEVER)
    .value("LESS", se::rhi::CompareFunction::LESS)
    .value("EQUAL", se::rhi::CompareFunction::EQUAL)
    .value("LESS_EQUAL", se::rhi::CompareFunction::LESS_EQUAL)
    .value("GREATER", se::rhi::CompareFunction::GREATER)
    .value("NOT_EQUAL", se::rhi::CompareFunction::NOT_EQUAL)
    .value("GREATER_EQUAL", se::rhi::CompareFunction::GREATER_EQUAL)
    .value("ALWAYS", se::rhi::CompareFunction::ALWAYS);

  nb::enum_<se::rhi::ShaderStageEnum>(ns_rhi, "ShaderStageEnum")
    .value("VERTEX", se::rhi::ShaderStageEnum::VERTEX)
    .value("FRAGMENT", se::rhi::ShaderStageEnum::FRAGMENT)
    .value("COMPUTE", se::rhi::ShaderStageEnum::COMPUTE)
    .value("GEOMETRY", se::rhi::ShaderStageEnum::GEOMETRY)
    .value("RAYGEN", se::rhi::ShaderStageEnum::RAYGEN)
    .value("MISS", se::rhi::ShaderStageEnum::MISS)
    .value("CLOSEST_HIT", se::rhi::ShaderStageEnum::CLOSEST_HIT)
    .value("INTERSECTION", se::rhi::ShaderStageEnum::INTERSECTION)
    .value("ANY_HIT", se::rhi::ShaderStageEnum::ANY_HIT)
    .value("CALLABLE", se::rhi::ShaderStageEnum::CALLABLE)
    .value("TASK", se::rhi::ShaderStageEnum::TASK)
    .value("MESH", se::rhi::ShaderStageEnum::MESH)
    .def("__or__", [](se::rhi::ShaderStageEnum a, se::rhi::ShaderStageEnum b) {
    return se::Flags<se::rhi::ShaderStageEnum>(a | b); }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::ShaderStageEnum, "ShaderStages");

  nb::enum_<se::rhi::IndexFormat>(ns_rhi, "IndexFormat")
    .value("UINT16_t", se::rhi::IndexFormat::UINT16_t)
    .value("UINT32_T", se::rhi::IndexFormat::UINT32_T);

  nb::enum_<se::rhi::LoadOp>(ns_rhi, "LoadOp")
    .value("DONT_CARE", se::rhi::LoadOp::DONT_CARE)
    .value("LOAD", se::rhi::LoadOp::LOAD)
    .value("CLEAR", se::rhi::LoadOp::CLEAR);

  nb::enum_<se::rhi::StoreOp>(ns_rhi, "StoreOp")
    .value("DONT_CARE", se::rhi::StoreOp::DONT_CARE)
    .value("STORE", se::rhi::StoreOp::STORE)
    .value("DISCARD", se::rhi::StoreOp::DISCARD);

  nb::enum_<se::rhi::BlendOperation>(ns_rhi, "BlendOperation")
    .value("ADD", se::rhi::BlendOperation::ADD)
    .value("SUBTRACT", se::rhi::BlendOperation::SUBTRACT)
    .value("REVERSE_SUBTRACT", se::rhi::BlendOperation::REVERSE_SUBTRACT)
    .value("MIN", se::rhi::BlendOperation::MIN)
    .value("MAX", se::rhi::BlendOperation::MAX);

  nb::enum_<se::rhi::BlendFactor>(ns_rhi, "BlendFactor")
    .value("ZERO", se::rhi::BlendFactor::ZERO)
    .value("ONE", se::rhi::BlendFactor::ONE)
    .value("SRC", se::rhi::BlendFactor::SRC)
    .value("ONE_MINUS_SRC", se::rhi::BlendFactor::ONE_MINUS_SRC)
    .value("SRC_ALPHA", se::rhi::BlendFactor::SRC_ALPHA)
    .value("ONE_MINUS_SRC_ALPHA", se::rhi::BlendFactor::ONE_MINUS_SRC_ALPHA)
    .value("DST", se::rhi::BlendFactor::DST)
    .value("ONE_MINUS_DST", se::rhi::BlendFactor::ONE_MINUS_DST)
    .value("DST_ALPHA", se::rhi::BlendFactor::DST_ALPHA)
    .value("ONE_MINUS_DST_ALPHA", se::rhi::BlendFactor::ONE_MINUS_DST_ALPHA)
    .value("SRC_ALPHA_SATURATED", se::rhi::BlendFactor::SRC_ALPHA_SATURATED)
    .value("CONSTANT", se::rhi::BlendFactor::CONSTANT)
    .value("ONE_MINUS_CONSTANT", se::rhi::BlendFactor::ONE_MINUS_CONSTANT);

  nb::enum_<se::rhi::TextureAspectEnum>(ns_rhi, "TextureAspectEnum")
    .value("COLOR_BIT", se::rhi::TextureAspectEnum::COLOR_BIT)
    .value("STENCIL_BIT", se::rhi::TextureAspectEnum::STENCIL_BIT)
    .value("DEPTH_BIT", se::rhi::TextureAspectEnum::DEPTH_BIT);

  nb::enum_<se::rhi::TextureUsageEnum>(ns_rhi, "TextureUsageEnum")
    .value("COPY_SRC", se::rhi::TextureUsageEnum::COPY_SRC)
    .value("COPY_DST", se::rhi::TextureUsageEnum::COPY_DST)
    .value("TEXTURE_BINDING", se::rhi::TextureUsageEnum::TEXTURE_BINDING)
    .value("STORAGE_BINDING", se::rhi::TextureUsageEnum::STORAGE_BINDING)
    .value("COLOR_ATTACHMENT", se::rhi::TextureUsageEnum::COLOR_ATTACHMENT)
    .value("DEPTH_ATTACHMENT", se::rhi::TextureUsageEnum::DEPTH_ATTACHMENT)
    .value("TRANSIENT_ATTACHMENT", se::rhi::TextureUsageEnum::TRANSIENT_ATTACHMENT)
    .value("INPUT_ATTACHMENT", se::rhi::TextureUsageEnum::INPUT_ATTACHMENT)
    .def("__or__", [](se::rhi::TextureUsageEnum a, se::rhi::TextureUsageEnum b) {
    return se::Flags<se::rhi::TextureUsageEnum>(a | b); }, nb::is_operator());
  BIND_BITMASK_FLAGS(ns_rhi, se::rhi::TextureUsageEnum, "TextureUsages");

  nb::enum_<se::rhi::TextureFormat>(ns_rhi, "TextureFormat")
    .value("UNKOWN", se::rhi::TextureFormat::UNKOWN)
    .value("R8_UNORM", se::rhi::TextureFormat::R8_UNORM)
    .value("R8_SNORM", se::rhi::TextureFormat::R8_SNORM)
    .value("R8_UINT", se::rhi::TextureFormat::R8_UINT)
    .value("R8_SINT", se::rhi::TextureFormat::R8_SINT)
    .value("R16_UINT", se::rhi::TextureFormat::R16_UINT)
    .value("R16_SINT", se::rhi::TextureFormat::R16_SINT)
    .value("R16_FLOAT", se::rhi::TextureFormat::R16_FLOAT)
    .value("RG8_UNORM", se::rhi::TextureFormat::RG8_UNORM)
    .value("RG8_SNORM", se::rhi::TextureFormat::RG8_SNORM)
    .value("RG8_UINT", se::rhi::TextureFormat::RG8_UINT)
    .value("RG8_SINT", se::rhi::TextureFormat::RG8_SINT)
    .value("R32_UINT", se::rhi::TextureFormat::R32_UINT)
    .value("R32_SINT", se::rhi::TextureFormat::R32_SINT)
    .value("R32_FLOAT", se::rhi::TextureFormat::R32_FLOAT)
    .value("RG16_UINT", se::rhi::TextureFormat::RG16_UINT)
    .value("RG16_SINT", se::rhi::TextureFormat::RG16_SINT)
    .value("RG16_FLOAT", se::rhi::TextureFormat::RG16_FLOAT)
    .value("RGBA8_UNORM", se::rhi::TextureFormat::RGBA8_UNORM)
    .value("RGBA8_UNORM_SRGB", se::rhi::TextureFormat::RGBA8_UNORM_SRGB)
    .value("RGBA8_SNORM", se::rhi::TextureFormat::RGBA8_SNORM)
    .value("RGBA8_UINT", se::rhi::TextureFormat::RGBA8_UINT)
    .value("RGBA8_SINT", se::rhi::TextureFormat::RGBA8_SINT)
    .value("BGRA8_UNORM", se::rhi::TextureFormat::BGRA8_UNORM)
    .value("BGRA8_UNORM_SRGB", se::rhi::TextureFormat::BGRA8_UNORM_SRGB)
    .value("RGB9E5_UFLOAT", se::rhi::TextureFormat::RGB9E5_UFLOAT)
    .value("RG11B10_UFLOAT", se::rhi::TextureFormat::RG11B10_UFLOAT)
    .value("RG32_UINT", se::rhi::TextureFormat::RG32_UINT)
    .value("RG32_SINT", se::rhi::TextureFormat::RG32_SINT)
    .value("RG32_FLOAT", se::rhi::TextureFormat::RG32_FLOAT)
    .value("RGBA16_UINT", se::rhi::TextureFormat::RGBA16_UINT)
    .value("RGBA16_SINT", se::rhi::TextureFormat::RGBA16_SINT)
    .value("RGBA16_FLOAT", se::rhi::TextureFormat::RGBA16_FLOAT)
    .value("RGBA32_UINT", se::rhi::TextureFormat::RGBA32_UINT)
    .value("RGBA32_SINT", se::rhi::TextureFormat::RGBA32_SINT)
    .value("RGBA32_FLOAT", se::rhi::TextureFormat::RGBA32_FLOAT)
    .value("STENCIL8", se::rhi::TextureFormat::STENCIL8)
    .value("DEPTH16_UNORM", se::rhi::TextureFormat::DEPTH16_UNORM)
    .value("DEPTH24", se::rhi::TextureFormat::DEPTH24)
    .value("DEPTH24STENCIL8", se::rhi::TextureFormat::DEPTH24STENCIL8)
    .value("DEPTH32_FLOAT", se::rhi::TextureFormat::DEPTH32_FLOAT)
    .value("COMPRESSION", se::rhi::TextureFormat::COMPRESSION)
    .value("RGB10A2_UNORM", se::rhi::TextureFormat::RGB10A2_UNORM)
    .value("DEPTH32STENCIL8", se::rhi::TextureFormat::DEPTH32STENCIL8)
    .value("BC1_RGB_UNORM_BLOCK", se::rhi::TextureFormat::BC1_RGB_UNORM_BLOCK)
    .value("BC1_RGB_SRGB_BLOCK", se::rhi::TextureFormat::BC1_RGB_SRGB_BLOCK)
    .value("BC1_RGBA_UNORM_BLOCK", se::rhi::TextureFormat::BC1_RGBA_UNORM_BLOCK)
    .value("BC1_RGBA_SRGB_BLOCK", se::rhi::TextureFormat::BC1_RGBA_SRGB_BLOCK)
    .value("BC2_UNORM_BLOCK", se::rhi::TextureFormat::BC2_UNORM_BLOCK)
    .value("BC2_SRGB_BLOCK", se::rhi::TextureFormat::BC2_SRGB_BLOCK)
    .value("BC3_UNORM_BLOCK", se::rhi::TextureFormat::BC3_UNORM_BLOCK)
    .value("BC3_SRGB_BLOCK", se::rhi::TextureFormat::BC3_SRGB_BLOCK)
    .value("BC4_UNORM_BLOCK", se::rhi::TextureFormat::BC4_UNORM_BLOCK)
    .value("BC4_SNORM_BLOCK", se::rhi::TextureFormat::BC4_SNORM_BLOCK)
    .value("BC5_UNORM_BLOCK", se::rhi::TextureFormat::BC5_UNORM_BLOCK)
    .value("BC5_SNORM_BLOCK", se::rhi::TextureFormat::BC5_SNORM_BLOCK)
    .value("BC6H_UFLOAT_BLOCK", se::rhi::TextureFormat::BC6H_UFLOAT_BLOCK)
    .value("BC6H_SFLOAT_BLOCK", se::rhi::TextureFormat::BC6H_SFLOAT_BLOCK)
    .value("BC7_UNORM_BLOCK", se::rhi::TextureFormat::BC7_UNORM_BLOCK)
    .value("BC7_SRGB_BLOCK", se::rhi::TextureFormat::BC7_SRGB_BLOCK);

  nb::enum_<se::rhi::TextureLayoutEnum>(ns_rhi, "TextureLayoutEnum")
    .value("UNDEFINED", se::rhi::TextureLayoutEnum::UNDEFINED)
    .value("GENERAL", se::rhi::TextureLayoutEnum::GENERAL)
    .value("COLOR_ATTACHMENT_OPTIMAL", se::rhi::TextureLayoutEnum::COLOR_ATTACHMENT_OPTIMAL)
    .value("DEPTH_STENCIL_ATTACHMENT_OPTIMA", se::rhi::TextureLayoutEnum::DEPTH_STENCIL_ATTACHMENT_OPTIMA)
    .value("DEPTH_STENCIL_READ_ONLY_OPTIMAL", se::rhi::TextureLayoutEnum::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    .value("SHADER_READ_ONLY_OPTIMAL", se::rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL)
    .value("TRANSFER_SRC_OPTIMAL", se::rhi::TextureLayoutEnum::TRANSFER_SRC_OPTIMAL)
    .value("TRANSFER_DST_OPTIMAL", se::rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL)
    .value("PREINITIALIZED", se::rhi::TextureLayoutEnum::PREINITIALIZED)
    .value("DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL", se::rhi::TextureLayoutEnum::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
    .value("DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL", se::rhi::TextureLayoutEnum::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
    .value("DEPTH_ATTACHMENT_OPTIMAL", se::rhi::TextureLayoutEnum::DEPTH_ATTACHMENT_OPTIMAL)
    .value("DEPTH_READ_ONLY_OPTIMAL", se::rhi::TextureLayoutEnum::DEPTH_READ_ONLY_OPTIMAL)
    .value("STENCIL_ATTACHMENT_OPTIMAL", se::rhi::TextureLayoutEnum::STENCIL_ATTACHMENT_OPTIMAL)
    .value("STENCIL_READ_ONLY_OPTIMAL", se::rhi::TextureLayoutEnum::STENCIL_READ_ONLY_OPTIMAL)
    .value("PRESENT_SRC", se::rhi::TextureLayoutEnum::PRESENT_SRC)
    .value("SHARED_PRESENT", se::rhi::TextureLayoutEnum::SHARED_PRESENT)
    .value("FRAGMENT_DENSITY_MAP_OPTIMAL", se::rhi::TextureLayoutEnum::FRAGMENT_DENSITY_MAP_OPTIMAL)
    .value("FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL", se::rhi::TextureLayoutEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL)
    .value("READ_ONLY_OPTIMAL", se::rhi::TextureLayoutEnum::READ_ONLY_OPTIMAL)
    .value("ATTACHMENT_OPTIMAL", se::rhi::TextureLayoutEnum::ATTACHMENT_OPTIMAL);

  nb::enum_<se::rhi::AddressMode>(ns_rhi, "AddressMode")
    .value("CLAMP_TO_EDGE", se::rhi::AddressMode::CLAMP_TO_EDGE)
    .value("REPEAT", se::rhi::AddressMode::REPEAT)
    .value("MIRROR_REPEAT", se::rhi::AddressMode::MIRROR_REPEAT);

  nb::enum_<se::rhi::FilterMode>(ns_rhi, "FilterMode")
    .value("NEAREST", se::rhi::FilterMode::NEAREST)
    .value("LINEAR", se::rhi::FilterMode::LINEAR);

  nb::enum_<se::rhi::MipmapFilterMode>(ns_rhi, "MipmapFilterMode")
    .value("NEAREST", se::rhi::MipmapFilterMode::NEAREST)
    .value("LINEAR", se::rhi::MipmapFilterMode::LINEAR);

  nb::class_<se::rhi::BarrierDescriptor>(ns_rhi, "BarrierDescriptor")
    .def(nb::init<se::Flags<se::rhi::PipelineStageEnum>, se::Flags<se::rhi::PipelineStageEnum>,
      se::Flags<se::rhi::DependencyTypeEnum>, std::vector<se::rhi::MemoryBarrier*>,
      std::vector<se::rhi::BufferMemoryBarrierDescriptor>, std::vector<se::rhi::TextureMemoryBarrierDescriptor>>());

  nb::class_<se::rhi::CommandEncoder>(ns_rhi, "CommandEncoder")
    .def("finish", &se::rhi::CommandEncoder::finish, nb::rv_policy::reference)
    .def("pipeline_barrier", &se::rhi::CommandEncoder::pipeline_barrier);
  rhi_device.def("create_command_encoder", &se::rhi::Device::create_command_encoder,
      nb::arg("external").none() = nb::none());

  nb::class_<se::rhi::RenderPassEncoder>(ns_rhi, "RenderPassEncoder")
    .def("push_constants", [](se::rhi::RenderPassEncoder& encoder,
      uintptr_t address, se::Flags<se::rhi::ShaderStageEnum> stage, uint32_t offset, uint32_t size
      ) { encoder.push_constants(reinterpret_cast<void*>(address), stage, offset, size); })
    .def("push_constants", [](se::rhi::RenderPassEncoder& encoder,
      uintptr_t address, se::rhi::ShaderStageEnum stage, uint32_t offset, uint32_t size
      ) { encoder.push_constants(reinterpret_cast<void*>(address), stage, offset, size); })
    .def("set_index_buffer", &se::rhi::RenderPassEncoder::set_index_buffer)
    .def("draw", &se::rhi::RenderPassEncoder::draw)
    .def("draw_indexed", &se::rhi::RenderPassEncoder::draw_indexed)
    .def("end", &se::rhi::RenderPassEncoder::end);

  nb::class_<se::rhi::ComputePassEncoder>(ns_rhi, "ComputePassEncoder")
    .def("push_constants", [](se::rhi::ComputePassEncoder& encoder,
      uintptr_t address, se::Flags<se::rhi::ShaderStageEnum> stage, uint32_t offset, uint32_t size
      ) { encoder.push_constants(reinterpret_cast<void*>(address), stage, offset, size); })
    .def("push_constants", [](se::rhi::ComputePassEncoder& encoder,
      uintptr_t address, se::rhi::ShaderStageEnum stage, uint32_t offset, uint32_t size
      ) { encoder.push_constants(reinterpret_cast<void*>(address), stage, offset, size); })
    .def("dispatch_workgroups", &se::rhi::ComputePassEncoder::dispatch_workgroups)
    .def("dispatch_workgroups_indirect", &se::rhi::ComputePassEncoder::dispatch_workgroups_indirect)
    .def("end", &se::rhi::ComputePassEncoder::end);

  nb::class_<se::rhi::Texture> class_Texture(ns_rhi, "Texture");
  nb::class_<se::rhi::TextureView> class_TextureView(ns_rhi, "TextureView");
  nb::class_<se::rhi::Sampler> class_Sampler(ns_rhi, "Sampler");
  nb::class_<se::rhi::TLAS> class_TLAS(ns_rhi, "TLAS");

  nb::class_<se::rhi::BufferBinding>(ns_rhi, "BufferBinding")
    .def(nb::init<se::rhi::Buffer*, size_t, size_t>());

  nb::class_<se::rhi::BindingResource> class_BindingResource(ns_rhi, "BindingResource");
  class_BindingResource
    .def(nb::init<>())
    .def(nb::init<se::rhi::TextureView*, se::rhi::Sampler*>())
    .def(nb::init<se::rhi::Sampler*>())
    .def(nb::init<se::rhi::TextureView*>())
    .def(nb::init<se::rhi::BufferBinding>())
    .def(nb::init<se::rhi::TLAS*>())
    .def(nb::init<std::vector<se::rhi::TextureView*>>())
    .def(nb::init<std::vector<se::rhi::TextureView*>, se::rhi::Sampler*>())
    .def(nb::init<std::vector<se::rhi::TextureView*>, std::vector<se::rhi::Sampler*>>());

  nb::class_<se::rhi::RenderPassColorAttachment>(ns_rhi, "RenderPassColorAttachment")
    .def(nb::init<>())
    .def(nb::init<se::rhi::TextureView*, se::vec4, se::rhi::LoadOp, se::rhi::StoreOp>());

  nb::class_<se::rhi::RenderPassDepthStencilAttachment>(ns_rhi, "RenderPassDepthStencilAttachment")
    .def(nb::init<>())
    .def(nb::init<se::rhi::TextureView*, float, se::rhi::LoadOp, se::rhi::StoreOp, bool,
      uint32_t, se::rhi::LoadOp, se::rhi::StoreOp, bool>());

  nb::enum_<se::rhi::DataType>(ns_rhi, "DataType")
    .value("Float16", se::rhi::DataType::Float16)
    .value("Float32", se::rhi::DataType::Float32)
    .value("Float64", se::rhi::DataType::Float64)
    .value("UINT8", se::rhi::DataType::UINT8)
    .value("INT8", se::rhi::DataType::INT8)
    .value("INT16", se::rhi::DataType::INT16)
    .value("INT32", se::rhi::DataType::INT32)
    .value("INT64", se::rhi::DataType::INT64);

  nb::class_<se::rhi::RenderPassDescriptor>(ns_rhi, "RenderPassDescriptor")
    .def(nb::init<>())
    .def(nb::init<std::vector<se::rhi::RenderPassColorAttachment> const&>())
    .def(nb::init<std::vector<se::rhi::RenderPassColorAttachment> const&,
      se::rhi::RenderPassDepthStencilAttachment const&>())
    .def_rw("maxDrawCount", &se::rhi::RenderPassDescriptor::maxDrawCount);

  nb::class_<se::rhi::CUDASemaphore> class_cudaSemaphore(ns_rhi, "CUDASemaphore");
  class_cudaSemaphore.def("signal", nb::overload_cast<uintptr_t>(&se::rhi::CUDASemaphore::signal))
    .def("signal", nb::overload_cast<uintptr_t, size_t>(&se::rhi::CUDASemaphore::signal))
    .def("wait", nb::overload_cast<uintptr_t>(&se::rhi::CUDASemaphore::wait))
    .def("wait", nb::overload_cast<uintptr_t, size_t>(&se::rhi::CUDASemaphore::wait));

  nb::class_<se::rhi::CUDAExternalBuffer> class_cudaBuffer(ns_rhi, "CUDAExternalBuffer");
  nb::class_<se::rhi::CUDAContext> class_cudacontext(ns_rhi, "CUDAContext");
  class_cudacontext.def_static("initialize", nb::overload_cast<se::rhi::Device*>(&se::rhi::CUDAContext::initialize));
  class_cudacontext.def_static("export_to_cuda", nb::overload_cast<se::rhi::Buffer*>(&se::rhi::CUDAContext::export_to_cuda));
  class_cudacontext.def_static("export_to_cuda", nb::overload_cast<se::rhi::Semaphore*>(&se::rhi::CUDAContext::export_to_cuda));
  class_cudacontext.def_static("to_tensor", [](
    se::rhi::CUDAExternalBuffer* cudaBuffer,
    std::vector<size_t> const& shapes,
    se::rhi::DataType type
  ) {
    // Get raw pointer and analyze data type
    void* data_ptr = cudaBuffer->ptr();
    size_t itemsize = 0;
    // Get dtype
    nb::dlpack::dtype dtype;
    switch (type) {
    case se::rhi::DataType::Float32: 
      dtype = nb::dtype<float>(); 
      itemsize = sizeof(float);
      break;
    case se::rhi::DataType::Float64: 
      dtype = nb::dtype<double>();
      itemsize = sizeof(double);
      break;
    case se::rhi::DataType::INT32:   
      dtype = nb::dtype<int32_t>();
      itemsize = sizeof(int32_t);
      break;
    case se::rhi::DataType::INT64:   
      dtype = nb::dtype<int64_t>();
      itemsize = sizeof(int64_t);
      break;
    case se::rhi::DataType::UINT8:   
      dtype = nb::dtype<uint8_t>();
      itemsize = sizeof(uint8_t);
      break;
    default: se::error("Unsupported tensor data type");
    }
    
    // Delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data_ptr, [](void* p) noexcept {});

    // Construct ndarray dynamically
    return nb::ndarray<nb::pytorch, nb::device::cuda>(
      data_ptr,
      shapes.size(),
      shapes.data(),
      owner,
      {},
      dtype
    );
    });

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ gfx                                                                       ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ Graphics layer above rhi and supprt further dev.		                       ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  auto ns_gfx = nb::class_<::namespace_gfx>(m, "gfx");

  auto gfx_buffer = nb::class_<se::gfx::Buffer>(ns_gfx, "Buffer");

  nb::class_<se::gfx::Buffer::ConsumeEntry>(gfx_buffer, "ConsumeEntry")
    .def(nb::init<>())
    .def("add_stage", &se::gfx::Buffer::ConsumeEntry::add_stage, nb::rv_policy::reference)
    .def("set_access", &se::gfx::Buffer::ConsumeEntry::set_access, nb::rv_policy::reference)
    .def("set_subresource", &se::gfx::Buffer::ConsumeEntry::set_subresource, nb::rv_policy::reference);

  auto gfx_texture = nb::class_<se::gfx::Texture>(ns_gfx, "Texture");

  nb::class_<se::gfx::ShaderModule>(ns_gfx, "ShaderModule");
  nb::class_<se::gfx::ShaderHandle>(ns_gfx, "ShaderHandle")
    .def("get", [](se::gfx::ShaderHandle& self) { return self.m_handle.handle().get(); }, nb::rv_policy::reference);

  nb::enum_<se::gfx::Texture::ConsumeType>(gfx_texture, "ConsumeType")
    .value("ColorAttachment", se::gfx::Texture::ConsumeType::ColorAttachment)
    .value("DepthStencilAttachment", se::gfx::Texture::ConsumeType::DepthStencilAttachment)
    .value("TextureBinding", se::gfx::Texture::ConsumeType::TextureBinding)
    .value("StorageBinding", se::gfx::Texture::ConsumeType::StorageBinding);

  nb::class_<se::gfx::Texture::ConsumeEntry>(gfx_texture, "ConsumeEntry")
    .def(nb::init<>())
    .def(nb::init<se::gfx::Texture::ConsumeType>())
    .def(nb::init<se::gfx::Texture::ConsumeType, se::Flags<se::rhi::AccessFlagEnum>, se::Flags<se::rhi::PipelineStageEnum>,
      uint32_t, uint32_t, uint32_t, uint32_t, se::rhi::TextureLayoutEnum, bool, se::rhi::CompareFunction, uint32_t>(),
      nb::arg("type"), nb::arg("access"), nb::arg("stages") = 0,
      nb::arg("level_beg") = 0, nb::arg("level_end") = 1, nb::arg("mip_beg") = 0, nb::arg("mip_end") = 1,
      nb::arg("layout") = se::rhi::TextureLayoutEnum::UNDEFINED, nb::arg("depthWrite") = false,
      nb::arg("depthCmp") = se::rhi::CompareFunction::ALWAYS, nb::arg("attachLoc") = uint32_t(-1))
    .def("add_stage", &se::gfx::Texture::ConsumeEntry::add_stage, nb::rv_policy::reference)
    .def("add_stage", [](se::gfx::Texture::ConsumeEntry& entry, se::rhi::PipelineStageEnum stage
      )->se::gfx::Texture::ConsumeEntry { return entry.add_stage(stage); }, nb::rv_policy::reference)
    .def("set_layout", &se::gfx::Texture::ConsumeEntry::set_layout, nb::rv_policy::reference)
    .def("enable_depth_write", &se::gfx::Texture::ConsumeEntry::enable_depth_write, nb::rv_policy::reference)
    .def("set_depth_compare_fn", &se::gfx::Texture::ConsumeEntry::set_depth_compare_fn, nb::rv_policy::reference)
    .def("set_subresource", &se::gfx::Texture::ConsumeEntry::set_subresource, nb::rv_policy::reference)
    .def("set_attachment_loc", &se::gfx::Texture::ConsumeEntry::set_attachment_loc, nb::rv_policy::reference)
    .def("set_blend_operation", &se::gfx::Texture::ConsumeEntry::set_blend_operation, nb::rv_policy::reference)
    .def("set_source_blender_factor", &se::gfx::Texture::ConsumeEntry::set_source_blender_factor, nb::rv_policy::reference)
    .def("set_target_blender_factor", &se::gfx::Texture::ConsumeEntry::set_target_blender_factor, nb::rv_policy::reference)
    .def("set_access", &se::gfx::Texture::ConsumeEntry::set_access, nb::rv_policy::reference);

  nb::class_<se::gfx::TextureHandle>(ns_gfx, "TextureHandle")
    .def("get", [](se::gfx::TextureHandle& self) { return self.m_handle.handle().get(); }, nb::rv_policy::reference)
    .def("get_uav", [](se::gfx::TextureHandle& self, uint32_t mipLevel, uint32_t firstArraySlice,
      uint32_t arraySize) { return self->get_uav(mipLevel, firstArraySlice, arraySize); }, nb::rv_policy::reference)
    .def("get_rtv", [](se::gfx::TextureHandle& self, uint32_t mipLevel, uint32_t firstArraySlice,
      uint32_t arraySize) { return self->get_rtv(mipLevel, firstArraySlice, arraySize); }, nb::rv_policy::reference)
    .def("get_dsv", [](se::gfx::TextureHandle& self, uint32_t mipLevel, uint32_t firstArraySlice,
      uint32_t arraySize) { return self->get_dsv(mipLevel, firstArraySlice, arraySize); }, nb::rv_policy::reference)
    .def("get_srv", [](se::gfx::TextureHandle& self, uint32_t mostDetailedMip, uint32_t mipCount,
      uint32_t firstArraySlice, uint32_t arraySize) { return self->get_srv(mostDetailedMip, mipCount,
        firstArraySlice, arraySize); }, nb::rv_policy::reference)
    .def("width", [](se::gfx::TextureHandle& self) { return self->width(); })
    .def("height", [](se::gfx::TextureHandle& self) { return self->height(); });

  auto gfx_scene = nb::class_<se::gfx::Scene>(ns_gfx, "Scene");
  gfx_scene.def("gpu_scene", &se::gfx::Scene::gpu_scene, nb::rv_policy::reference);

  nb::class_<se::gfx::Scene::GPUScene>(ns_gfx, "GPUScene")
    .def("binding_resource_index", &se::gfx::Scene::GPUScene::binding_resource_index)
    .def("binding_resource_position", &se::gfx::Scene::GPUScene::binding_resource_position)
    .def("binding_resource_vertex", &se::gfx::Scene::GPUScene::binding_resource_vertex)
    .def("binding_resource_geometry", &se::gfx::Scene::GPUScene::binding_resource_geometry)
    .def("binding_resource_tlas", &se::gfx::Scene::GPUScene::binding_resource_tlas)
    .def("binding_resource_medium", &se::gfx::Scene::GPUScene::binding_resource_medium)
    .def("binding_resource_medium_grid", &se::gfx::Scene::GPUScene::binding_resource_medium_grid)
    .def("binding_resource_camera", &se::gfx::Scene::GPUScene::binding_resource_camera);

  nb::class_<se::gfx::SceneBatch>(ns_gfx, "SceneBatch")
    .def("emplace_scene", &se::gfx::SceneBatch::emplace_scene)
    .def("update_gpu_scene_batch", &se::gfx::SceneBatch::update_gpu_scene_batch)
    .def("gpu_scene", [](se::gfx::SceneBatch& self) { return self.gpu_scene(); }, nb::rv_policy::reference);
  
  nb::class_<se::gfx::SceneHandle>(ns_gfx, "SceneHandle")
    .def("update_scripts", [](se::gfx::SceneHandle& self) { return self->update_scripts(); })
    .def("update_transform", [](se::gfx::SceneHandle& self) { return self->update_transform(); })
    .def("update_gpu_scene", [](se::gfx::SceneHandle& self) { return self->update_gpu_scene(); })
    .def("set_viewport_size", [](se::gfx::SceneHandle& self, se::ivec2 size) { return self->set_viewport_size(size); })
    .def("load_gltf", [](se::gfx::SceneHandle& self, std::string const& path) { return self->load_gltf(path); })
    .def("gpu_scene", [](se::gfx::SceneHandle& self) { return self->gpu_scene(); }, nb::rv_policy::reference)
    .def("draw_meshes", [](se::gfx::SceneHandle& self, se::rhi::RenderPassEncoder* encoder, int32_t geometryIDOffset)
      { return self->draw_meshes(encoder, geometryIDOffset); });

  nb::class_<se::gfx::GFXContext>(ns_gfx, "GFXContext")
    .def_static("initialize", &se::gfx::GFXContext::initialize,
      nb::arg("window").none() = nb::none(), nb::arg("ext") = 0)
    .def_static("device", &se::gfx::GFXContext::device, nb::rv_policy::reference)
    .def_static("create_flights", &se::gfx::GFXContext::create_flights,
      nb::arg("maxFlightNum") = 2, nb::arg("swapchain").none() = nb::none())
    .def_static("get_flights", &se::gfx::GFXContext::get_flights, nb::rv_policy::reference)
    .def_static("load_shader_slang",
      static_cast<std::vector<se::gfx::ShaderHandle>(*)(std::string const&,
        std::vector<std::pair<std::string, se::rhi::ShaderStageEnum>> const&,
        std::vector<std::pair<char const*, char const*>> const&,
        bool)>(&se::gfx::GFXContext::load_shader_slang),
      nb::arg("path"), nb::arg("entrypoints"), nb::arg("macros") =
      std::vector<std::pair<char const*, char const*>>{}, nb::arg("glsl_intermediate") = false,
      nb::rv_policy::reference)
    .def_static("load_scene_gltf", &se::gfx::GFXContext::load_scene_gltf)
    .def_static("load_scene_xml", &se::gfx::GFXContext::load_scene_xml)
    .def_static("load_scene_pbrt", &se::gfx::GFXContext::load_scene_pbrt)
    .def_static("create_scene_batch", &se::gfx::GFXContext::create_scene_batch)
    .def_static("clean_texture_cache", &se::gfx::GFXContext::clean_texture_cache)
    .def_static("clean_cache", &se::gfx::GFXContext::clean_cache)
    .def_static("frame_end", &se::gfx::GFXContext::frame_end)
    .def_static("finalize", &se::gfx::GFXContext::finalize);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ rdg                                                                       ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ render dependency graph.                                 		             ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  nb::class_<namespace_rdg> ns_rdg(m, "rdg");

  nb::class_<se::rdg::PassReflection>(ns_rdg, "PassReflection")
    .def(nb::init<>())
    .def("add_input", &se::rdg::PassReflection::add_input, nb::rv_policy::reference)
    .def("add_output", &se::rdg::PassReflection::add_output, nb::rv_policy::reference)
    .def("add_input_output", &se::rdg::PassReflection::add_input_output, nb::rv_policy::reference)
    .def("add_internal", &se::rdg::PassReflection::add_internal, nb::rv_policy::reference);

  nb::class_<se::rdg::ResourceInfo>(ns_rdg, "ResourceInfo")
    .def("is_buffer", &se::rdg::ResourceInfo::is_buffer, nb::rv_policy::reference)
    .def("is_texture", &se::rdg::ResourceInfo::is_texture, nb::rv_policy::reference);

  nb::class_<se::rdg::BufferInfo>(ns_rdg, "BufferInfo")
    .def("with_size", &se::rdg::BufferInfo::with_size, nb::rv_policy::reference)
    .def("with_usages", &se::rdg::BufferInfo::with_usages, nb::rv_policy::reference)
    .def("with_usages", [](se::rdg::BufferInfo& info, se::rhi::BufferUsageEnum usage
      )->se::rdg::BufferInfo& { return  info.with_usages(usage); }, nb::rv_policy::reference)
    .def("consume", &se::rdg::BufferInfo::consume, nb::rv_policy::reference);

  nb::class_<se::rdg::TextureInfo>(ns_rdg, "TextureInfo")
    .def("consume", &se::rdg::TextureInfo::consume, nb::rv_policy::reference)
    .def("set_info", &se::rdg::TextureInfo::set_info, nb::rv_policy::reference)
    .def("with_size", nb::overload_cast<se::ivec3>(&se::rdg::TextureInfo::with_size), nb::rv_policy::reference)
    .def("with_size", nb::overload_cast<se::vec3>(&se::rdg::TextureInfo::with_size), nb::rv_policy::reference)
    .def("with_size_relative", &se::rdg::TextureInfo::with_size_relative, nb::rv_policy::reference)
    .def("with_levels", &se::rdg::TextureInfo::with_levels, nb::rv_policy::reference)
    .def("with_layers", &se::rdg::TextureInfo::with_layers, nb::rv_policy::reference)
    .def("with_samples", &se::rdg::TextureInfo::with_samples, nb::rv_policy::reference)
    .def("with_format", &se::rdg::TextureInfo::with_format, nb::rv_policy::reference)
    .def("with_stages", &se::rdg::TextureInfo::with_stages, nb::rv_policy::reference)
    .def("with_usages", &se::rdg::TextureInfo::with_usages, nb::rv_policy::reference)
    .def("consume_as_storage_binding_in_compute", &se::rdg::TextureInfo::consume_as_storage_binding_in_compute, nb::rv_policy::reference)
    .def("consume_as_color_attachment_at", &se::rdg::TextureInfo::consume_as_color_attachment_at, nb::rv_policy::reference)
    .def("consume_as_depth_stencil_attachment_at", &se::rdg::TextureInfo::consume_as_depth_stencil_attachment_at,
      "loc"_a, "depth_write"_a = true, "depth_compare"_a = se::rhi::CompareFunction::LESS_EQUAL,  nb::rv_policy::reference)
    .def("with_usages", [](se::rdg::TextureInfo& info, se::rhi::TextureUsageEnum usage
      )->se::rdg::TextureInfo& { return  info.with_usages(usage); }, nb::rv_policy::reference)
    .def("get_size", &se::rdg::TextureInfo::get_size, nb::rv_policy::reference);

  nb::class_<se::rdg::RenderContext>(ns_rdg, "RenderContext")
    .def(nb::init<>())
    .def(nb::init<se::rhi::CommandEncoder*, size_t>(), "encoder"_a, "idx"_a = 0)
    .def_rw("cmdEncoder", &se::rdg::RenderContext::cmdEncoder)
    .def_rw("flightIdx", &se::rdg::RenderContext::flightIdx);

  nb::class_<se::rdg::RenderData>(ns_rdg, "RenderData")
    .def(nb::init<>())
    .def("set_scene", &se::rdg::RenderData::set_scene)
    .def("set_scene_batch", &se::rdg::RenderData::set_scene_batch)
    .def("get_texture", &se::rdg::RenderData::get_texture)
    .def("get_buffer", &se::rdg::RenderData::get_buffer)
    .def("get_scene", &se::rdg::RenderData::get_scene)
    .def("get_scene_batch", &se::rdg::RenderData::get_scene_batch, nb::rv_policy::reference);

  nb::class_<se::rdg::Pass, se::rdg::PyPass<>>(ns_rdg, "Pass")
    .def("pass", &se::rdg::Pass::pass)
    .def("render_ui", &se::rdg::Pass::render_ui);

  //py::class_<se::rdg::DummyPass, se::rdg::Pass, se::rdg::PyDummyPass<>>(namespace_rdg, "DummyPass")
  //  .def(py::init<>())
  //  .def("execute", &se::rdg::RenderPass::execute);

  nb::class_<se::rdg::PipelinePass, se::rdg::Pass, se::rdg::PyPipelinePass<>>(ns_rdg, "PipelinePass");

  nb::class_<se::rdg::RenderPass, se::rdg::PipelinePass, se::rdg::PyRenderPass<>>(ns_rdg, "RenderPass")
    .def(nb::init<>())
    .def("reflect", &se::rdg::RenderPass::reflect)
    .def("execute", &se::rdg::RenderPass::execute)
    .def("update_bindings", &se::rdg::RenderPass::update_bindings)
    .def("update_binding_scene_batch", &se::rdg::RenderPass::update_binding_scene_batch)
    .def("set_render_pass_descriptor", &se::rdg::RenderPass::set_render_pass_descriptor)
    .def("init", static_cast<void(se::rdg::RenderPass::*)(
      std::string const&)>(&se::rdg::RenderPass::init))
    .def("init", static_cast<void(se::rdg::RenderPass::*)(
      se::gfx::ShaderModule*, se::gfx::ShaderModule*)>(&se::rdg::RenderPass::init))
    .def("init", static_cast<void(se::rdg::RenderPass::*)(
      se::gfx::ShaderModule*, se::gfx::ShaderModule*, se::gfx::ShaderModule*)>(&se::rdg::RenderPass::init))
    .def("begin_pass", static_cast<se::rhi::RenderPassEncoder* (se::rdg::RenderPass::*)(
      se::rdg::RenderContext*, se::gfx::Texture*)>(&se::rdg::RenderPass::begin_pass),
      nb::rv_policy::reference)
    .def("begin_pass", static_cast<se::rhi::RenderPassEncoder* (se::rdg::RenderPass::*)(
      se::rdg::RenderContext*, uint32_t, uint32_t)>(&se::rdg::RenderPass::begin_pass),
      nb::rv_policy::reference);

  // //py::class_<se::rdg::FullScreenPass, se::rdg::RenderPass, se::rdg::PyFullScreenPass<>>(namespace_rdg, "FullScreenPass")
  // //  .def(py::init<>())
  // //  .def("init", &se::rdg::FullScreenPass::init)
  // //  .def("reflect", &se::rdg::FullScreenPass::reflect)
  // //  .def("execute", &se::rdg::FullScreenPass::execute)
  // //  .def("updateBindings", &se::rdg::FullScreenPass::updateBindings)
  // //  .def("dispatchFullScreen", &se::rdg::FullScreenPass::dispatchFullScreen);

   nb::class_<se::rdg::ComputePass, se::rdg::PipelinePass, se::rdg::PyComputePass<>>(ns_rdg, "ComputePass")
     .def(nb::init<>())
     .def(nb::init<std::string const&>())
     .def(nb::init<std::string const&,
        std::vector<std::pair<char const*, char const*>> const&>())
     .def("reflect", &se::rdg::ComputePass::reflect)
     .def("execute", &se::rdg::ComputePass::execute)
     .def("update_binding_scene", &se::rdg::RenderPass::update_binding_scene)
     .def("update_binding_scene_batch", &se::rdg::RenderPass::update_binding_scene_batch)
     .def("update_bindings", &se::rdg::ComputePass::update_bindings)
     .def("begin_pass", &se::rdg::ComputePass::begin_pass, nb::rv_policy::reference)
     .def("init", nb::overload_cast<std::string const&>(&se::rdg::ComputePass::init))
     .def("init", nb::overload_cast<se::gfx::ShaderModule*>(&se::rdg::ComputePass::init))
     .def("init", nb::overload_cast<std::string const&,
        std::vector<std::pair<char const*, char const*>> const&>(&se::rdg::ComputePass::init));

  nb::class_<se::rdg::Graph, se::rdg::PyGraph<>>(ns_rdg, "Graph")
    .def(nb::init<>())
    .def("build", &se::rdg::Graph::build)
    .def("execute", &se::rdg::Graph::execute)
    //.def("readback", &se::rdg::Graph::readback)
    //.def("renderUI", &se::rdg::Graph::renderUI)
    //.def("addEdge", static_cast<void(se::rdg::Graph::*)(std::string const&,
    //  std::string const&, std::string const&, std::string const&)>(&se::rdg::Graph::addEdge))
    //.def("addEdge", static_cast<void(se::rdg::Graph::*)(std::string const&,
    //  std::string const&)>(&se::rdg::Graph::addEdge))
    .def("mark_output", &se::rdg::Graph::mark_output)
    .def("get_output", &se::rdg::Graph::get_output)
    .def("set_standard_size", &se::rdg::Graph::set_standard_size)
    .def("render_ui", &se::rdg::Graph::render_ui)
    .def("get_render_data", &se::rdg::Graph::get_render_data, nb::rv_policy::reference)
    //.def("getBufferResource", &se::rdg::Graph::getBufferResource)
    //.def("getTextureResource", &se::rdg::Graph::getTextureResource)
    //.def("getPass", static_cast<se::rdg::Pass* (se::rdg::Graph::*)(size_t)>(&se::rdg::Graph::getPass), nb::rv_policy::reference)
    //.def("getPass", static_cast<se::rdg::Pass* (se::rdg::Graph::*)(std::string const&)>(&se::rdg::Graph::getPass), nb::rv_policy::reference)
    .def("add_pass", &se::rdg::Graph::add_pass)
    .def("add_edge", &se::rdg::Graph::add_edge);

  //py::class_<se::rdg::Pipeline, se::rdg::PyPipeline<>>(namespace_rdg, "Pipeline")
  //  .def(py::init<>())
  //  .def("build", &se::rdg::Pipeline::build)
  //  .def("execute", &se::rdg::Pipeline::execute)
  //  .def("readback", &se::rdg::Pipeline::readback)
  //  .def("renderUI", &se::rdg::Pipeline::renderUI)
  //  .def("getActiveGraphs", &se::rdg::Pipeline::getActiveGraphs)
  //  .def("setStandardSize", &se::rdg::Pipeline::setStandardSize)
  //  .def("bindScene", &se::rdg::Pipeline::bindScene)
  //  .def("getOutput", &se::rdg::Pipeline::getOutput)
  //  .def("pipeline", &se::rdg::Pipeline::pipeline, py::return_value_policy::reference);

  //py::class_<se::rdg::SingleGraphPipeline, se::rdg::Pipeline, se::rdg::PySingleGraphPipeline<>>(namespace_rdg, "SingleGraphPipeline")
  //  .def(py::init<>())
  //  .def("execute", &se::rdg::SingleGraphPipeline::execute)
  //  .def("getActiveGraphs", &se::rdg::SingleGraphPipeline::getActiveGraphs)
  //  .def("getOutput", &se::rdg::SingleGraphPipeline::getOutput)
  //  .def("readback", &se::rdg::SingleGraphPipeline::readback)
  //  .def("build", &se::rdg::SingleGraphPipeline::build)
  //  .def("setGraph", &se::rdg::SingleGraphPipeline::setGraph);


  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ editor                                                                    ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ Editor layer for ImGui and other editor related features.		             ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  auto ns_editor = nb::class_<::namespace_editor>(m, "editor");
  
  nb::class_<se::editor::IFragment>(ns_editor, "IFragment");
  nb::class_<se::editor::FragmentPool>(ns_editor, "FragmentPool");

  nb::class_<se::editor::EditorContext>(ns_editor, "EditorContext")
    .def_static("initialize", &se::editor::EditorContext::initialize)
    .def_static("finalize", &se::editor::EditorContext::finalize)
    .def_static("set_scene_display", &se::editor::EditorContext::set_scene_display)
    .def_static("set_graph_display", &se::editor::EditorContext::set_graph_display)
    .def_static("set_viewport_texture", &se::editor::EditorContext::set_viewport_texture)
    .def_static("begin_frame", &se::editor::EditorContext::begin_frame)
    .def_static("end_frame", &se::editor::EditorContext::end_frame);

  nb::class_<se::editor::ImGuiContext>(ns_editor, "ImGuiContext")
    .def_static("need_recreate", &se::editor::ImGuiContext::need_recreate)
    .def_static("recreate", &se::editor::ImGuiContext::recreate, nb::arg("width"), nb::arg("height"))
    .def_static("start_new_frame", &se::editor::ImGuiContext::start_new_frame)
    .def_static("start_gui_recording", &se::editor::ImGuiContext::start_gui_recording)
    .def_static("render", &se::editor::ImGuiContext::render,
        nb::arg("waitSemaphore").none() = nb::none())
    .def_static("get_dpi", &se::editor::ImGuiContext::get_dpi);

  // Export ImGui struct
  // ------------------------------------------------------------------------


  // Export CPP type struct
  // ------------------------------------------------------------------------
  nb::class_<se::CPPType<int32_t>>(m, "Int32")
    .def(nb::init<>()).def(nb::init<int32_t>())
    .def("set", &se::CPPType<int32_t>::set)
    .def("get", &se::CPPType<int32_t>::get, nb::rv_policy::reference);
  nb::class_<se::CPPType<uint32_t>>(m, "UInt32")
    .def(nb::init<>()).def(nb::init<uint32_t>())
    .def("set", &se::CPPType<uint32_t>::set)
    .def("get", &se::CPPType<uint32_t>::get, nb::rv_policy::reference);
  nb::class_<se::CPPType<float>>(m, "Float32")
    .def(nb::init<>()).def(nb::init<float>())
    .def("set", &se::CPPType<float>::set)
    .def("get", &se::CPPType<float>::get, nb::rv_policy::reference);
  nb::class_<se::CPPType<bool>>(m, "Bool")
    .def(nb::init<>()).def(nb::init<bool>())
    .def("set", &se::CPPType<bool>::set)
    .def("get", &se::CPPType<bool>::get, nb::rv_policy::reference);

  nb::class_<namespace_imgui> class_ImGui(m, "imgui");
  class_ImGui
    .def_static("set_current_context", &ImGui::SetCurrentContext)
    .def_static("begin", &ImGui::Begin)
    .def_static("end", &ImGui::End)
    .def_static("set_cursor_pos", [](se::vec2 v) {
      ImGui::SetCursorPos(ImVec2{ float(v.x),float(v.y) });
    })
    .def_static("get_cursor_pos", []() {
      ImVec2 cursor_pos = ImGui::GetCursorPos();
      return se::vec2{ cursor_pos.x, cursor_pos.y }; 
    })
    .def_static("checkbox", [](const char* label, se::CPPType<bool>& item)->bool{
      return ImGui::Checkbox(label, &item.get());
    })
    .def_static("text_colored", [](se::vec4 v, std::string const& text){
      ImGui::TextColored(ImVec4{ v.x, v.y, v.z, v.w }, text.c_str());
    })
    .def_static("set_window_font_scale", &ImGui::SetWindowFontScale)
    .def_static("push_item_width", &ImGui::PushItemWidth)
    .def_static("pop_item_width", &ImGui::PopItemWidth)
    .def_static("tree_node", [](const char* label){ return ImGui::TreeNode(label); })
    .def_static("tree_pop", &ImGui::TreePop)
    .def_static("color_edit_vec3", [](const char* label, se::vec3& color){ 
      return ImGui::ColorEdit3(label, &color[0]); })
    .def_static("text", [](std::string const& text) { ImGui::Text(text.c_str()); })
    .def_static("same_line",
      static_cast<void(*)(float, float)>(&ImGui::SameLine),
      nb::arg("offset_from_start_x") = 0.0f, nb::arg("spacing") = -1.0f)
    .def_static("button",[](const char* label, se::vec2 size){
      return ImGui::Button(label, ImVec2(size.x, size.y)); })
    .def_static("drag_int", [](const char* label, se::CPPType<int32_t>& v, float v_speed, int v_min,
      int v_max, const char* format, ImGuiSliderFlags flags) {
      return ImGui::DragInt(label, &v.get(), v_speed, v_min, v_max, format, flags);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0f, nb::arg("v_min") = 0, nb::arg("v_max") = 0,
      nb::arg("format") = "%d", nb::arg("flags") = 0)
    .def_static("drag_float", [](const char* label, se::CPPType<float>& v, float v_speed, float v_min,
      float v_max, const char* format, ImGuiSliderFlags flags) {
        return ImGui::DragFloat(label, &v.get(), v_speed, v_min, v_max, format, flags); },
      nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0f, nb::arg("v_min") = 0.0f, nb::arg("v_max") = 0.0f,
      nb::arg("format") = "%.3f", nb::arg("flags") = 0)
    .def_static("combo", [](const char* label, se::CPPType<int32_t>& current_item,
      std::vector<std::string> const& items, int popup_max_height_in_items) {
        std::vector<const char*> items_char(items.size());
        for (size_t i = 0; i < items.size(); ++i)
          items_char[i] = items[i].c_str();
        return ImGui::Combo(label, &current_item.get(), items_char.data(), items_char.size(), popup_max_height_in_items); },
      nb::arg("label"), nb::arg("current_items"), nb::arg("items"), nb::arg("popup_max_height_in_items") = -1);



    // Export addon pass
    // ------------------------------------------------------------------------
    auto ns_addon = nb::class_<::namespace_addon>(m, "addon");
    nb::class_<AccumulatePass, se::rdg::ComputePass>(ns_addon, "AccumulatePass")
      .def(nb::init<>())
      .def("reflect", &AccumulatePass::reflect)
      .def("execute", &AccumulatePass::execute)
      .def("render_ui", &AccumulatePass::update_bindings);

}