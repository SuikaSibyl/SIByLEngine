#include <se.core.hpp>
#include <se.rhi.hpp>
#include <se.rhi.torch.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>
#include <se.editor.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
namespace py = pybind11;
using namespace se;
#pragma once
#undef MemoryBarrier

#include <passes/se.pass.postprocess.hpp>
#include <passes/se.pass.editor.hpp>
#include <passes/se.pass.rasterizer.hpp>
#include <passes/se.pass.cbt.hpp>
#include "py.define.hpp"

#include <bxdfs/se.bxdf.epflbrdf.hpp>
#include <lights/se.lights.envmap.hpp>

struct rhi_namescope {};
struct gfx_namescope {};
struct rdg_namescope {};
struct pass_namescope {};
struct bxdf_namescope {};
struct light_namescope {};
struct cbt_namescope {};

namespace se::rdg {
template <class PassBase = se::rdg::Pass>
struct PyPass : PassBase {
  using PassBase::PassBase; // Inherit constructors
  virtual auto reflect() noexcept -> PassReflection 
  { PYBIND11_OVERRIDE_PURE(PassReflection, PassBase, reflect); };
  virtual auto execute(RenderContext* context, RenderData const& renderData) noexcept -> void 
  { PYBIND11_OVERRIDE_PURE(void, PassBase, execute, context, renderData); };
  virtual auto readback(RenderData const& renderData) noexcept -> void
  { PYBIND11_OVERRIDE(void, PassBase, readback, renderData); }
  virtual auto renderUI() noexcept -> void 
  { PYBIND11_OVERRIDE(void, PassBase, renderUI, ); }
  virtual auto onInteraction(se::input* ipt, editor::Widget::WidgetInfo* info) noexcept -> void
  { PYBIND11_OVERRIDE(void, PassBase, onInteraction, ipt, info); }
  virtual auto generateMarker() noexcept -> void
  { PYBIND11_OVERRIDE(void, PassBase, generateMarker, ); }
};

template <class DummyPassBase = se::rdg::DummyPass>
struct PyDummyPass : PyPass<DummyPassBase> {
  using PyPass<DummyPassBase>::PyPass; // Inherit constructors
  virtual auto execute(RenderContext* context, RenderData const& renderData) noexcept -> void 
  { PYBIND11_OVERRIDE_PURE(void, DummyPassBase, execute, context, renderData); };
};

template <class PipelinePassBase = se::rdg::PipelinePass>
struct PyPipelinePass : PyPass<PipelinePassBase> {
  using PyPass<PipelinePassBase>::PyPass; // Inherit constructors
};

template <class RenderPassBase = se::rdg::RenderPass>
struct PyRenderPass : PyPipelinePass<RenderPassBase> {
  using PyPipelinePass<RenderPassBase>::PyPipelinePass; // Inherit constructors
  virtual auto beforeDirectDrawcall(rhi::RenderPassEncoder* encoder, int geometry_idx,
      gfx::Scene::GeometryDrawData const& data) noexcept -> void
  { PYBIND11_OVERRIDE(void, RenderPassBase, beforeDirectDrawcall, encoder, geometry_idx, data); }
};

template <class FullScreenPassBase = se::rdg::FullScreenPass>
struct PyFullScreenPass : PyRenderPass<FullScreenPassBase> {
  using PyRenderPass<FullScreenPassBase>::PyRenderPass; // Inherit constructors
};

template <class ComputePassBase = se::rdg::ComputePass>
struct PyComputePass : PyPipelinePass<ComputePassBase> {
  using PyPipelinePass<ComputePassBase>::PyPipelinePass; // Inherit constructors
};

template <class GraphBase = se::rdg::Graph>
struct PyGraph : GraphBase {
  using GraphBase::GraphBase; // Inherit constructors
  virtual auto readback() noexcept -> void
  { PYBIND11_OVERRIDE(void, GraphBase, readback); };
  virtual auto renderUI() noexcept -> void
  { PYBIND11_OVERRIDE(void, GraphBase, renderUI); };
};

template <class PipelineBase = se::rdg::Pipeline>
struct PyPipeline : PipelineBase {
  using PipelineBase::PipelineBase; // Inherit constructors
  virtual auto build() noexcept -> void
  { PYBIND11_OVERRIDE_PURE(void, PipelineBase, build); };
  virtual auto execute(rhi::CommandEncoder* encoder) noexcept -> void
  { PYBIND11_OVERRIDE_PURE(void, PipelineBase, execute, encoder); };
  virtual auto readback() noexcept -> void
  { PYBIND11_OVERRIDE(void, PipelineBase, readback); };
  virtual auto renderUI() noexcept -> void
  { PYBIND11_OVERRIDE(void, PipelineBase, renderUI); };
  virtual auto getAllGraphs() noexcept -> std::vector<Graph*>
  { PYBIND11_OVERRIDE_PURE(std::vector<Graph*>, PipelineBase, getAllGraphs); };
  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*>
  { PYBIND11_OVERRIDE_PURE(std::vector<Graph*>, PipelineBase, getActiveGraphs); };
  virtual auto getOutput() noexcept -> gfx::TextureHandle
  { PYBIND11_OVERRIDE_PURE(gfx::TextureHandle, PipelineBase, getOutput); };
};

template <class SingleGraphPipelineBase = se::rdg::SingleGraphPipeline>
struct PySingleGraphPipeline : SingleGraphPipelineBase {
  using SingleGraphPipelineBase::SingleGraphPipelineBase; // Inherit constructors
  virtual auto build() noexcept -> void
  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, build); };
  virtual auto execute(rhi::CommandEncoder* encoder) noexcept -> void
  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, execute, encoder); };
  virtual auto readback() noexcept -> void
  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, readback); };
  virtual auto renderUI() noexcept -> void
  { PYBIND11_OVERRIDE(void, SingleGraphPipelineBase, renderUI); };
  virtual auto getAllGraphs() noexcept -> std::vector<Graph*>
  { PYBIND11_OVERRIDE(std::vector<Graph*>, SingleGraphPipelineBase, getAllGraphs); };
  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*>
  { PYBIND11_OVERRIDE(std::vector<Graph*>, SingleGraphPipelineBase, getActiveGraphs); };
  virtual auto getOutput() noexcept -> gfx::TextureHandle
  { PYBIND11_OVERRIDE(gfx::TextureHandle, SingleGraphPipelineBase, getOutput); };
};
}

namespace se::gfx {  
template <class BxDFBase = se::gfx::IBxDF>
struct PyBxDF : BxDFBase {
  using BxDFBase::BxDFBase; // Inherit constructors
};
}

void set_null(void* ptr) {
  ptr = nullptr;
}

py::capsule create_capsule_from_address(uintptr_t address) {
    void* ptr = reinterpret_cast<void*>(address);
    return py::capsule(ptr, PyCapsule_Destructor(nullptr));
}

PYBIND11_MODULE(pycore, m) {
  m.doc() = "SIByLpy :: core module";
  m.def("SetNullptr", &set_null);
  m.def("ptr_from_address", &create_capsule_from_address);

  // Export CPP type struct
  // ------------------------------------------------------------------------
  py::class_<CPPType<int32_t>>(m, "Int32")
    .def(py::init<>())
    .def(py::init<int32_t>())
    .def("set", &CPPType<int32_t>::set)
    .def("get", &CPPType<int32_t>::get, py::return_value_policy::reference);
  py::class_<CPPType<uint32_t>>(m, "UInt32")
    .def(py::init<>())
    .def(py::init<uint32_t>())
    .def("set", &CPPType<uint32_t>::set)
    .def("get", &CPPType<uint32_t>::get, py::return_value_policy::reference);
  py::class_<CPPType<float>>(m, "Float32")
    .def(py::init<>())
    .def(py::init<float>())
    .def("set", &CPPType<float>::set)
    .def("get", &CPPType<float>::get, py::return_value_policy::reference);
  py::class_<CPPType<bool>>(m, "Bool")
    .def(py::init<>())
    .def(py::init<bool>())
    .def("set", &CPPType<bool>::set)
    .def("get", &CPPType<bool>::get, py::return_value_policy::reference);

  // Export root struct
  // ------------------------------------------------------------------------
  // add root class definition
  py::class_<root> class_root(m, "root");
  // add print module to root
  py::class_<root::print>(class_root, "print")
    .def(py::init<>())
    .def_static("debug", &root::print::debug)
    .def_static("log", &root::print::log)
    .def_static("warning", &root::print::warning)
    .def_static("error", &root::print::error)
    .def_static("correct", &root::print::correct);

  // Export input struct
  // ------------------------------------------------------------------------
  py::class_<se::input> class_input(m, "input");
  py::class_<se::input::CodeEnum> class_CodeEnum(class_input, "CodeEnum");
  class_input
    .def_property_readonly_static("mouse_button_left", [](py::object const&) { return se::input::mouse_button_left; })
    .def_property_readonly_static("mouse_button_right", [](py::object const&) { return se::input::mouse_button_right; })
    .def_property_readonly_static("mouse_button_middle", [](py::object const&) { return se::input::mouse_button_middle; })
    .def_property_readonly_static("mouse_button_last", [](py::object const&) { return se::input::mouse_button_last; })
    .def("isMouseButtonPressed", &se::input::isMouseButtonPressed);

  // Export timer struct
  // ------------------------------------------------------------------------
  py::class_<se::timer> class_timer(m, "timer");
  class_timer.def(py::init<>())
    .def("update", &se::timer::update)
    .def("deltaTime", &se::timer::deltaTime)
    .def("totalTime", &se::timer::totalTime);

  // Export window struct
  // ------------------------------------------------------------------------
  py::class_<se::window::WindowOptions> class_WindowOption(m, "WindowOption");
  class_WindowOption.def(py::init<se::window::Vendor, 
    std::wstring const&, uint32_t, uint32_t, se::window::Properties>())
    .def_readwrite("vendor", &se::window::WindowOptions::vendor)
    .def_readwrite("title", &se::window::WindowOptions::title)
    .def_readwrite("width", &se::window::WindowOptions::width)
    .def_readwrite("height", &se::window::WindowOptions::height)
    .def_readwrite("properties", &se::window::WindowOptions::properties);
  py::class_<se::window> class_window(m, "window");
  py::enum_<se::window::Vendor>(class_window, "EnumVendor")
    .value("GLFW", se::window::Vendor::GLFW)
    .value("Win64", window::Vendor::WIN_64);
  py::enum_<se::window::Properties>(class_window, "EnumProperty")
    .value("None", se::window::Properties::NONE)
    .value("VulkanContext", se::window::Properties::VULKAN_CONTEX)
    .value("OpenGLContext", se::window::Properties::OPENGL_CONTEX);
  class_window.def_static("create", &se::window::create);
  class_window.def("fetchEvents", &se::window::fetchEvents)
    .def("endFrame", &se::window::endFrame)
    .def("destroy", &se::window::destroy)
    .def("getInput", &se::window::getInput, py::return_value_policy::reference)
    .def("isRunning", &se::window::isRunning);

  // Export math namespace
  // ------------------------------------------------------------------------
  py::class_<ivec2>(m, "ivec2")
    .def(py::init<int, int>())
    .def_readwrite("x", &ivec2::x)
    .def_readwrite("y", &ivec2::y);
  py::class_<ivec3>(m, "ivec3")
    .def(py::init<int, int, int>())
    .def_readwrite("x", &ivec3::x)
    .def_readwrite("y", &ivec3::y)
    .def_readwrite("z", &ivec3::z);
  py::class_<ivec4>(m, "ivec4")
    .def(py::init<int, int, int, int>())
    .def_readwrite("x", &ivec4::x)
    .def_readwrite("y", &ivec4::y)
    .def_readwrite("z", &ivec4::z)
    .def_readwrite("w", &ivec4::w);

  py::class_<vec2>(m, "vec2")
    .def(py::init<float, float>())
    .def_readwrite("x", &vec2::x)
    .def_readwrite("y", &vec2::y);
  py::class_<vec3>(m, "vec3")
    .def(py::init<float, float, float>())
    .def_readwrite("x", &vec3::x)
    .def_readwrite("y", &vec3::y)
    .def_readwrite("z", &vec3::z);
  py::class_<vec4>(m, "vec4")
    .def(py::init<float, float, float, float>())
    .def_readwrite("x", &vec4::x)
    .def_readwrite("y", &vec4::y)
    .def_readwrite("z", &vec4::z)
    .def_readwrite("w", &vec4::w);
  
  py::class_<point3>(m, "point3")
    .def(py::init<float, float, float>())
    .def(py::init<vec3>());
  
  py::class_<bounds3>(m, "bounds3")
    .def(py::init<point3>())
    .def(py::init<point3, point3>());
  
  // Export rhi namespace
  // ------------------------------------------------------------------------
  py::class_<rhi_namescope> namespace_rhi(m, "rhi");
  py::enum_<se::rhi::ContextExtensionBit>(namespace_rhi, "EnumContextExtension")
    .value("NONE", se::rhi::ContextExtensionBit::NONE)
    .value("DEBUG_UTILS", se::rhi::ContextExtensionBit::DEBUG_UTILS)
    .value("MESH_SHADER", se::rhi::ContextExtensionBit::MESH_SHADER)
    .value("FRAGMENT_BARYCENTRIC", se::rhi::ContextExtensionBit::FRAGMENT_BARYCENTRIC)
    .value("SAMPLER_FILTER_MIN_MAX", se::rhi::ContextExtensionBit::SAMPLER_FILTER_MIN_MAX)
    .value("RAY_TRACING", se::rhi::ContextExtensionBit::RAY_TRACING)
    .value("SHADER_NON_SEMANTIC_INFO", se::rhi::ContextExtensionBit::SHADER_NON_SEMANTIC_INFO)
    .value("BINDLESS_INDEXING", se::rhi::ContextExtensionBit::BINDLESS_INDEXING)
    .value("ATOMIC_FLOAT", se::rhi::ContextExtensionBit::ATOMIC_FLOAT)
    .value("CONSERVATIVE_RASTERIZATION", se::rhi::ContextExtensionBit::CONSERVATIVE_RASTERIZATION)
    .value("COOPERATIVE_MATRIX", se::rhi::ContextExtensionBit::COOPERATIVE_MATRIX)
    .value("CUDA_INTEROPERABILITY", se::rhi::ContextExtensionBit::CUDA_INTEROPERABILITY)
    .value("USE_AFTERMATH", se::rhi::ContextExtensionBit::USE_AFTERMATH);
  py::enum_<se::rhi::MemoryPropertyBit>(namespace_rhi, "EnumMemoryProperty")
    .value("DEVICE_LOCAL_BIT", se::rhi::MemoryPropertyBit::DEVICE_LOCAL_BIT)
    .value("HOST_VISIBLE_BIT", se::rhi::MemoryPropertyBit::HOST_VISIBLE_BIT)
    .value("HOST_COHERENT_BIT", se::rhi::MemoryPropertyBit::HOST_COHERENT_BIT)
    .value("HOST_CACHED_BIT", se::rhi::MemoryPropertyBit::HOST_CACHED_BIT)
    .value("LAZILY_ALLOCATED_BIT", se::rhi::MemoryPropertyBit::LAZILY_ALLOCATED_BIT)
    .value("PROTECTED_BIT", se::rhi::MemoryPropertyBit::PROTECTED_BIT)
    .value("FLAG_BITS_MAX_ENUM", se::rhi::MemoryPropertyBit::FLAG_BITS_MAX_ENUM);
  py::enum_<se::rhi::AccessFlagBits>(namespace_rhi, "AccessFlagBits")
    .value("INDIRECT_COMMAND_READ_BIT", se::rhi::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
    .value("INDEX_READ_BIT", se::rhi::AccessFlagBits::INDEX_READ_BIT)
    .value("VERTEX_ATTRIBUTE_READ_BIT", se::rhi::AccessFlagBits::VERTEX_ATTRIBUTE_READ_BIT)
    .value("UNIFORM_READ_BIT", se::rhi::AccessFlagBits::UNIFORM_READ_BIT)
    .value("INPUT_ATTACHMENT_READ_BIT", se::rhi::AccessFlagBits::INPUT_ATTACHMENT_READ_BIT)
    .value("SHADER_READ_BIT", se::rhi::AccessFlagBits::SHADER_READ_BIT)
    .value("SHADER_WRITE_BIT", se::rhi::AccessFlagBits::SHADER_WRITE_BIT)
    .value("COLOR_ATTACHMENT_READ_BIT", se::rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT)
    .value("COLOR_ATTACHMENT_WRITE_BIT", se::rhi::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT)
    .value("DEPTH_STENCIL_ATTACHMENT_READ_BIT", se::rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    .value("DEPTH_STENCIL_ATTACHMENT_WRITE_BIT", se::rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
    .value("TRANSFER_READ_BIT", se::rhi::AccessFlagBits::TRANSFER_READ_BIT)
    .value("TRANSFER_WRITE_BIT", se::rhi::AccessFlagBits::TRANSFER_WRITE_BIT)
    .value("HOST_READ_BIT", se::rhi::AccessFlagBits::HOST_READ_BIT)
    .value("HOST_WRITE_BIT", se::rhi::AccessFlagBits::HOST_WRITE_BIT)
    .value("MEMORY_READ_BIT", se::rhi::AccessFlagBits::MEMORY_READ_BIT)
    .value("MEMORY_WRITE_BIT", se::rhi::AccessFlagBits::MEMORY_WRITE_BIT)
    .value("TRANSFORM_FEEDBACK_WRITE_BIT", se::rhi::AccessFlagBits::TRANSFORM_FEEDBACK_WRITE_BIT)
    .value("TRANSFORM_FEEDBACK_COUNTER_READ_BIT", se::rhi::AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_READ_BIT)
    .value("TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT", se::rhi::AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT)
    .value("CONDITIONAL_RENDERING_READ_BIT", se::rhi::AccessFlagBits::CONDITIONAL_RENDERING_READ_BIT)
    .value("COLOR_ATTACHMENT_READ_NONCOHERENT_BIT", se::rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT)
    .value("ACCELERATION_STRUCTURE_READ_BIT", se::rhi::AccessFlagBits::ACCELERATION_STRUCTURE_READ_BIT)
    .value("ACCELERATION_STRUCTURE_WRITE_BIT", se::rhi::AccessFlagBits::ACCELERATION_STRUCTURE_WRITE_BIT)
    .value("FRAGMENT_DENSITY_MAP_READ_BIT", se::rhi::AccessFlagBits::FRAGMENT_DENSITY_MAP_READ_BIT)
    .value("FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT", se::rhi::AccessFlagBits::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT)
    .value("COMMAND_PREPROCESS_READ_BIT", se::rhi::AccessFlagBits::COMMAND_PREPROCESS_READ_BIT)
    .value("COMMAND_PREPROCESS_WRITE_BIT", se::rhi::AccessFlagBits::COMMAND_PREPROCESS_WRITE_BIT)
    .value("NONE", se::rhi::AccessFlagBits::NONE);
  py::enum_<se::rhi::CompareFunction>(namespace_rhi, "CompareFunction")
    .value("NEVER", se::rhi::CompareFunction::NEVER)
    .value("LESS", se::rhi::CompareFunction::LESS)
    .value("EQUAL", se::rhi::CompareFunction::EQUAL)
    .value("LESS_EQUAL", se::rhi::CompareFunction::LESS_EQUAL)
    .value("GREATER", se::rhi::CompareFunction::GREATER)
    .value("NOT_EQUAL", se::rhi::CompareFunction::NOT_EQUAL)
    .value("GREATER_EQUAL", se::rhi::CompareFunction::GREATER_EQUAL)
    .value("ALWAYS", se::rhi::CompareFunction::ALWAYS);
  py::enum_<se::rhi::ShaderStageBit>(namespace_rhi, "EnumShaderStage")
    .value("VERTEX", se::rhi::ShaderStageBit::VERTEX)
    .value("FRAGMENT", se::rhi::ShaderStageBit::FRAGMENT)
    .value("COMPUTE", se::rhi::ShaderStageBit::COMPUTE)
    .value("GEOMETRY", se::rhi::ShaderStageBit::GEOMETRY)
    .value("RAYGEN", se::rhi::ShaderStageBit::RAYGEN)
    .value("MISS", se::rhi::ShaderStageBit::MISS)
    .value("CLOSEST_HIT", se::rhi::ShaderStageBit::CLOSEST_HIT)
    .value("INTERSECTION", se::rhi::ShaderStageBit::INTERSECTION)
    .value("ANY_HIT", se::rhi::ShaderStageBit::ANY_HIT)
    .value("CALLABLE", se::rhi::ShaderStageBit::CALLABLE)
    .value("TASK", se::rhi::ShaderStageBit::TASK)
    .value("MESH", se::rhi::ShaderStageBit::MESH);
  py::enum_<se::rhi::IndexFormat>(namespace_rhi, "IndexFormat")
    .value("UINT16_t", se::rhi::IndexFormat::UINT16_t)
    .value("UINT32_T", se::rhi::IndexFormat::UINT32_T);
  py::enum_<se::rhi::LoadOp>(namespace_rhi, "LoadOp")
    .value("DONT_CARE", se::rhi::LoadOp::DONT_CARE)
    .value("LOAD", se::rhi::LoadOp::LOAD)
    .value("CLEAR", se::rhi::LoadOp::CLEAR);
  py::enum_<se::rhi::StoreOp>(namespace_rhi, "StoreOp")
    .value("DONT_CARE", se::rhi::StoreOp::DONT_CARE)
    .value("STORE", se::rhi::StoreOp::STORE)
    .value("DISCARD", se::rhi::StoreOp::DISCARD);
  py::enum_<se::rhi::BlendOperation>(namespace_rhi, "BlendOperation")
    .value("ADD", se::rhi::BlendOperation::ADD)
    .value("SUBTRACT", se::rhi::BlendOperation::SUBTRACT)
    .value("REVERSE_SUBTRACT", se::rhi::BlendOperation::REVERSE_SUBTRACT)
    .value("MIN", se::rhi::BlendOperation::MIN)
    .value("MAX", se::rhi::BlendOperation::MAX);
  py::enum_<se::rhi::BlendFactor>(namespace_rhi, "BlendFactor")
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

  // Export rhi::context structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::Context> class_rhi_context(namespace_rhi, "Context");
  py::enum_<se::rhi::Context::Backend>(class_rhi_context, "EnumBackend")
    .value("Vulkan", se::rhi::Context::Backend::Vulkan);
  class_rhi_context.def_static("create", &se::rhi::Context::create);
  class_rhi_context.def("init", &se::rhi::Context::init)
    .def("requestAdapter", &se::rhi::Context::requestAdapter);
  // Export rhi::adapter structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::AdapterDescriptor> class_rhi_adapter_options(
    namespace_rhi, "RequestAdapterOptions");
  py::enum_<se::rhi::AdapterDescriptor::PowerPreference>(class_rhi_adapter_options, "EnumPowerPreference")
    .value("LOW_POWER", se::rhi::AdapterDescriptor::PowerPreference::LOW_POWER)
    .value("HIGH_PERFORMANCE", se::rhi::AdapterDescriptor::PowerPreference::HIGH_PERFORMANCE);
  class_rhi_adapter_options.def(py::init<se::rhi::AdapterDescriptor::PowerPreference, bool>(),
    py::arg("powerPerference")= se::rhi::AdapterDescriptor::PowerPreference::HIGH_PERFORMANCE,
    py::arg("forceFallbackAdapter")=false)
    .def_readwrite("powerPerference", &se::rhi::AdapterDescriptor::powerPerference)
    .def_readwrite("forceFallbackAdapter", &se::rhi::AdapterDescriptor::forceFallbackAdapter);
  py::class_<se::rhi::Adapter> class_rhi_adapter(namespace_rhi, "Adapter");
  class_rhi_adapter.def("requestDevice", &se::rhi::Adapter::requestDevice);
  // Export rhi::device structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::Queue> class_rhi_queue(namespace_rhi, "Queue");
  class_rhi_queue.def("submit", py::overload_cast<std::vector<se::rhi::CommandBuffer*> const&>(&se::rhi::Queue::submit));
  class_rhi_queue.def("submit", py::overload_cast<std::vector<se::rhi::CommandBuffer*> const&,
      se::rhi::Semaphore*, se::rhi::Semaphore*, se::rhi::Fence*>(&se::rhi::Queue::submit));

  py::class_<se::rhi::Semaphore> class_rhi_semaphore(namespace_rhi, "Semaphore");
  py::class_<se::rhi::Fence> class_rhi_fence(namespace_rhi, "Fence");


  py::class_<se::rhi::RenderPassColorAttachment> class_rhi_RenderPassColorAttachment(
    namespace_rhi, "RenderPassColorAttachment");
  class_rhi_RenderPassColorAttachment.def(py::init<>())
    .def(py::init<rhi::TextureView*, rhi::TextureView*, se::vec4, rhi::LoadOp, rhi::StoreOp>());
  py::class_<se::rhi::RenderPassDepthStencilAttachment> class_rhi_RenderPassDepthStencilAttachment(
    namespace_rhi, "RenderPassDepthStencilAttachment");
  class_rhi_RenderPassDepthStencilAttachment.def(py::init<>())
    .def(py::init<rhi::TextureView*, float, rhi::LoadOp, rhi::StoreOp, bool,
    rhi::StencilValue, rhi::LoadOp, rhi::StoreOp, bool>());

  py::class_<se::rhi::RenderPassTimestampWrite> class_rhi_RenderPassTimestampWrite(
    namespace_rhi, "RenderPassTimestampWrite");
  class_rhi_RenderPassTimestampWrite.def(py::init<>());

  py::class_<se::rhi::RenderPassDescriptor> class_rhi_RenderPassDescriptor(namespace_rhi, "RenderPassDescriptor");
  class_rhi_RenderPassDescriptor.def(py::init<>())
    .def(py::init<std::vector<se::rhi::RenderPassColorAttachment> const&, 
      se::rhi::RenderPassDepthStencilAttachment, se::rhi::RenderPassTimestampWrites, uint64_t>(),
    py::arg("colorAttachments"), py::arg("depthStencilAttachment"), 
    py::arg("timestampWrites")= se::rhi::RenderPassTimestampWrites{},
    py::arg("timestampWrites") = 50000000)
    .def_readwrite("timestampWrites", &se::rhi::RenderPassDescriptor::timestampWrites)
    .def_readwrite("maxDrawCount", &se::rhi::RenderPassDescriptor::maxDrawCount);

  py::class_<se::rhi::SwapChainDescriptor> class_rhi_swapchaindesc(namespace_rhi, "SwapChainDescriptor");
  class_rhi_swapchaindesc.def(py::init<>());
  py::class_<se::rhi::MultiFrameFlightsDescriptor> class_rhi_mfddesc(namespace_rhi, "MultiFrameFlightsDescriptor");
  class_rhi_mfddesc.def(py::init<int, se::rhi::SwapChain*>());

  py::class_<se::rhi::SwapChain> class_rhi_swapchain(namespace_rhi, "SwapChain");
  class_rhi_swapchain.def("getTexture", &se::rhi::SwapChain::getTexture, py::return_value_policy::reference);

  py::class_<se::rhi::MultiFrameFlights> class_rhi_mff(namespace_rhi, "MultiFrameFlights");
  class_rhi_mff.def("frameStart", &se::rhi::MultiFrameFlights::frameStart);
  class_rhi_mff.def("frameEnd", &se::rhi::MultiFrameFlights::frameEnd);
  class_rhi_mff.def("getCommandBuffer", &se::rhi::MultiFrameFlights::getCommandBuffer, py::return_value_policy::reference);
  class_rhi_mff.def("getImageAvailableSeamaphore", &se::rhi::MultiFrameFlights::getImageAvailableSeamaphore, py::return_value_policy::reference);
  class_rhi_mff.def("getRenderFinishedSeamaphore", &se::rhi::MultiFrameFlights::getRenderFinishedSeamaphore, py::return_value_policy::reference);
  class_rhi_mff.def("getFence", &se::rhi::MultiFrameFlights::getFence, py::return_value_policy::reference);

  py::class_<se::rhi::Device> class_rhi_device(namespace_rhi, "Device");
  class_rhi_device.def("createBuffer", &se::rhi::Device::createBuffer);
  class_rhi_device.def("waitIdle", &se::rhi::Device::waitIdle);
  class_rhi_device.def("createSwapChain", &se::rhi::Device::createSwapChain);
  class_rhi_device.def("copyBufferToBuffer", &se::rhi::Device::copyBufferToBuffer);
  class_rhi_device.def("createMultiFrameFlights", &se::rhi::Device::createMultiFrameFlights);
  class_rhi_device.def("trainsitionTextureLayout", &se::rhi::Device::trainsitionTextureLayout);
  class_rhi_device.def("getGraphicsQueue", &se::rhi::Device::getGraphicsQueue, py::return_value_policy::reference);

  // Export rhi::buffer structures
  // ------------------------------------------------------------------------
  py::enum_<se::rhi::BufferUsageBit>(namespace_rhi, "EnumBufferUsage")
    .value("MAP_READ", se::rhi::BufferUsageBit::MAP_READ)
    .value("MAP_WRITE", se::rhi::BufferUsageBit::MAP_WRITE)
    .value("COPY_SRC", se::rhi::BufferUsageBit::COPY_SRC)
    .value("COPY_DST", se::rhi::BufferUsageBit::COPY_DST)
    .value("INDEX", se::rhi::BufferUsageBit::INDEX)
    .value("VERTEX", se::rhi::BufferUsageBit::VERTEX)
    .value("UNIFORM", se::rhi::BufferUsageBit::UNIFORM)
    .value("STORAGE", se::rhi::BufferUsageBit::STORAGE)
    .value("INDIRECT", se::rhi::BufferUsageBit::INDIRECT)
    .value("QUERY_RESOLVE", se::rhi::BufferUsageBit::QUERY_RESOLVE)
    .value("SHADER_DEVICE_ADDRESS", se::rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS)
    .value("ACCELERATION_STRUCTURE_STORAGE", se::rhi::BufferUsageBit::ACCELERATION_STRUCTURE_STORAGE)
    .value("ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY", se::rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
    .value("SHADER_BINDING_TABLE", se::rhi::BufferUsageBit::SHADER_BINDING_TABLE)
    .value("CUDA_ACCESS", se::rhi::BufferUsageBit::CUDA_ACCESS);
  py::enum_<se::rhi::BufferShareMode>(namespace_rhi, "EnumBufferShareMode")
    .value("CONCURRENT", se::rhi::BufferShareMode::CONCURRENT)
    .value("EXCLUSIVE", se::rhi::BufferShareMode::EXCLUSIVE);
  py::class_<se::rhi::BufferDescriptor> class_BufferDescriptor(namespace_rhi, "BufferDescriptor");
  class_BufferDescriptor.def(py::init<size_t,
    se::rhi::BufferUsages, se::rhi::BufferShareMode, se::rhi::MemoryProperties, bool, int>(),
    py::arg("size"), py::arg("usage"), py::arg("shareMode") = se::rhi::BufferShareMode::EXCLUSIVE, 
    py::arg("memoryProperties") = 0, py::arg("mappedAtCreation") = false, py::arg("minimumAlignment") = -1)
    .def_readwrite("size", &se::rhi::BufferDescriptor::size)
    .def_readwrite("usage", &se::rhi::BufferDescriptor::usage)
    .def_readwrite("shareMode", &se::rhi::BufferDescriptor::shareMode)
    .def_readwrite("memoryProperties", &se::rhi::BufferDescriptor::memoryProperties)
    .def_readwrite("mappedAtCreation", &se::rhi::BufferDescriptor::mappedAtCreation)
    .def_readwrite("minimumAlignment", &se::rhi::BufferDescriptor::minimumAlignment);
  py::class_<se::rhi::Buffer> class_rhiBuffer(namespace_rhi, "Buffer");
  class_rhiBuffer.def("size", &se::rhi::Buffer::size);

  py::class_<se::rhi::Texture> class_Texture(namespace_rhi, "Texture");
  py::class_<se::rhi::TextureView> class_TextureView(namespace_rhi, "TextureView");
  py::class_<se::rhi::Sampler> class_Sampler(namespace_rhi, "Sampler");
  py::class_<se::rhi::TLAS> class_TLAS(namespace_rhi, "TLAS");

  py::class_<se::rhi::RenderCommandsMixin> class_RenderCommandsMixin(namespace_rhi, "RenderCommandsMixin");
  py::class_<se::rhi::BindingCommandMixin> class_BindingCommandMixin(namespace_rhi, "BindingCommandMixin");
  class_BindingCommandMixin.def("pushConstants", &se::rhi::BindingCommandMixin::pushConstants);

  py::class_<se::rhi::RenderPassEncoder, se::rhi::RenderCommandsMixin,
    se::rhi::BindingCommandMixin> class_RenderPassEncoder(
    namespace_rhi, "RenderPassEncoder");
  class_RenderPassEncoder
    .def("setIndexBuffer", &se::rhi::RenderPassEncoder::setIndexBuffer)
    .def("draw", &se::rhi::RenderPassEncoder::draw)
    .def("drawIndexed", &se::rhi::RenderPassEncoder::drawIndexed)
    .def("end", &se::rhi::RenderPassEncoder::end);

  py::class_<se::rhi::ComputePassEncoder, se::rhi::BindingCommandMixin> class_ComputePassEncoder(
    namespace_rhi, "ComputePassEncoder");
  class_ComputePassEncoder
    .def("dispatchWorkgroups", &se::rhi::ComputePassEncoder::dispatchWorkgroups)
    .def("dispatchWorkgroupsIndirect", &se::rhi::ComputePassEncoder::dispatchWorkgroupsIndirect)
    .def("end", &se::rhi::ComputePassEncoder::end);

  py::class_<se::rhi::BufferBinding> class_BufferBinding(namespace_rhi, "BufferBinding");
  class_BufferBinding.def(py::init<se::rhi::Buffer*, size_t, size_t>());
  py::class_<se::rhi::BindingResource> class_BindingResource(namespace_rhi, "BindingResource");
  class_BindingResource
    .def(py::init<>())
    .def(py::init<se::rhi::TextureView*, se::rhi::Sampler*>())
    .def(py::init<se::rhi::Sampler*>())
    .def(py::init<se::rhi::TextureView*>())
    .def(py::init<se::rhi::BufferBinding>())
    .def(py::init<se::rhi::TLAS*>())
    .def(py::init<std::vector<se::rhi::TextureView*>>())
    .def(py::init<std::vector<se::rhi::TextureView*>, se::rhi::Sampler*>())
    .def(py::init<std::vector<se::rhi::TextureView*>, std::vector<se::rhi::Sampler*>>());

  // Export rhi::CommandEncoder structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::CommandEncoderDescriptor> class_CommandEncoderDescriptor(namespace_rhi, "CommandEncoderDescriptor");
  class_CommandEncoderDescriptor.def(py::init<>());
  class_CommandEncoderDescriptor.def(py::init<se::rhi::CommandBuffer*>());

  py::class_<se::rhi::BarrierDescriptor> class_BarrierDescriptor(namespace_rhi, "BarrierDescriptor");

  py::class_<se::rhi::ImageSubresourceRange> class_ImageSubresourceRange(namespace_rhi, "ImageSubresourceRange");
  class_ImageSubresourceRange.def(py::init<se::rhi::TextureAspects, uint32_t, uint32_t, uint32_t, uint32_t>());

  py::class_<se::rhi::MemoryBarrier> class_MemoryBarrier(namespace_rhi, "MemoryBarrier");
  py::class_<se::rhi::BufferMemoryBarrierDescriptor> class_BufferMemoryBarrierDescriptor(namespace_rhi, "BufferMemoryBarrierDescriptor");
  py::class_<se::rhi::TextureMemoryBarrierDescriptor> class_TextureMemoryBarrierDescriptor(namespace_rhi, "TextureMemoryBarrierDescriptor");
  class_TextureMemoryBarrierDescriptor.def(py::init<se::rhi::Texture*,
    se::rhi::ImageSubresourceRange, se::rhi::AccessFlags, se::rhi::AccessFlags,
    se::rhi::TextureLayout, se::rhi::TextureLayout, se::rhi::Queue*, se::rhi::Queue*>());

  py::enum_<se::rhi::TextureAspectBit>(namespace_rhi, "TextureAspectBit")
    .value("COLOR_BIT", se::rhi::TextureAspectBit::COLOR_BIT)
    .value("STENCIL_BIT", se::rhi::TextureAspectBit::STENCIL_BIT)
    .value("DEPTH_BIT", se::rhi::TextureAspectBit::DEPTH_BIT);
  
  py::enum_<se::rhi::TextureUsageBit>(namespace_rhi, "TextureUsageBit")
    .value("COPY_SRC", se::rhi::TextureUsageBit::COPY_SRC)
    .value("COPY_DST", se::rhi::TextureUsageBit::COPY_DST)
    .value("TEXTURE_BINDING", se::rhi::TextureUsageBit::TEXTURE_BINDING)
    .value("STORAGE_BINDING", se::rhi::TextureUsageBit::STORAGE_BINDING)
    .value("COLOR_ATTACHMENT", se::rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .value("DEPTH_ATTACHMENT", se::rhi::TextureUsageBit::DEPTH_ATTACHMENT)
    .value("TRANSIENT_ATTACHMENT", se::rhi::TextureUsageBit::TRANSIENT_ATTACHMENT)
    .value("INPUT_ATTACHMENT", se::rhi::TextureUsageBit::INPUT_ATTACHMENT);

  py::enum_<se::rhi::TextureFormat>(namespace_rhi, "TextureFormat")
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

  py::enum_<se::rhi::TextureLayout>(namespace_rhi, "TextureLayout")
    .value("UNDEFINED", se::rhi::TextureLayout::UNDEFINED)
    .value("GENERAL", se::rhi::TextureLayout::GENERAL)
    .value("COLOR_ATTACHMENT_OPTIMAL", se::rhi::TextureLayout::COLOR_ATTACHMENT_OPTIMAL)
    .value("DEPTH_STENCIL_ATTACHMENT_OPTIMA", se::rhi::TextureLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA)
    .value("DEPTH_STENCIL_READ_ONLY_OPTIMAL", se::rhi::TextureLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    .value("SHADER_READ_ONLY_OPTIMAL", se::rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL)
    .value("TRANSFER_SRC_OPTIMAL", se::rhi::TextureLayout::TRANSFER_SRC_OPTIMAL)
    .value("TRANSFER_DST_OPTIMAL", se::rhi::TextureLayout::TRANSFER_DST_OPTIMAL)
    .value("PREINITIALIZED", se::rhi::TextureLayout::PREINITIALIZED)
    .value("DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL", se::rhi::TextureLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
    .value("DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL", se::rhi::TextureLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
    .value("DEPTH_ATTACHMENT_OPTIMAL", se::rhi::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL)
    .value("DEPTH_READ_ONLY_OPTIMAL", se::rhi::TextureLayout::DEPTH_READ_ONLY_OPTIMAL)
    .value("STENCIL_ATTACHMENT_OPTIMAL", se::rhi::TextureLayout::STENCIL_ATTACHMENT_OPTIMAL)
    .value("STENCIL_READ_ONLY_OPTIMAL", se::rhi::TextureLayout::STENCIL_READ_ONLY_OPTIMAL)
    .value("PRESENT_SRC", se::rhi::TextureLayout::PRESENT_SRC)
    .value("SHARED_PRESENT", se::rhi::TextureLayout::SHARED_PRESENT)
    .value("FRAGMENT_DENSITY_MAP_OPTIMAL", se::rhi::TextureLayout::FRAGMENT_DENSITY_MAP_OPTIMAL)
    .value("FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL", se::rhi::TextureLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL)
    .value("READ_ONLY_OPTIMAL", se::rhi::TextureLayout::READ_ONLY_OPTIMAL)
    .value("ATTACHMENT_OPTIMAL", se::rhi::TextureLayout::ATTACHMENT_OPTIMAL);
  
  py::enum_<se::rhi::PipelineStageBit>(namespace_rhi, "PipelineStageBit")
    .value("TOP_OF_PIPE_BIT", se::rhi::PipelineStageBit::TOP_OF_PIPE_BIT)
    .value("DRAW_INDIRECT_BIT", se::rhi::PipelineStageBit::DRAW_INDIRECT_BIT)
    .value("VERTEX_INPUT_BIT", se::rhi::PipelineStageBit::VERTEX_INPUT_BIT)
    .value("VERTEX_SHADER_BIT", se::rhi::PipelineStageBit::VERTEX_SHADER_BIT)
    .value("TESSELLATION_CONTROL_SHADER_BIT", se::rhi::PipelineStageBit::TESSELLATION_CONTROL_SHADER_BIT)
    .value("TESSELLATION_EVALUATION_SHADER_BIT", se::rhi::PipelineStageBit::TESSELLATION_EVALUATION_SHADER_BIT)
    .value("GEOMETRY_SHADER_BIT", se::rhi::PipelineStageBit::GEOMETRY_SHADER_BIT)
    .value("FRAGMENT_SHADER_BIT", se::rhi::PipelineStageBit::FRAGMENT_SHADER_BIT)
    .value("EARLY_FRAGMENT_TESTS_BIT", se::rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT)
    .value("LATE_FRAGMENT_TESTS_BIT", se::rhi::PipelineStageBit::LATE_FRAGMENT_TESTS_BIT)
    .value("COLOR_ATTACHMENT_OUTPUT_BIT", se::rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT)
    .value("COMPUTE_SHADER_BIT", se::rhi::PipelineStageBit::COMPUTE_SHADER_BIT)
    .value("TRANSFER_BIT", se::rhi::PipelineStageBit::TRANSFER_BIT)
    .value("BOTTOM_OF_PIPE_BIT", se::rhi::PipelineStageBit::BOTTOM_OF_PIPE_BIT)
    .value("HOST_BIT", se::rhi::PipelineStageBit::HOST_BIT)
    .value("ALL_GRAPHICS_BIT", se::rhi::PipelineStageBit::ALL_GRAPHICS_BIT)
    .value("ALL_COMMANDS_BIT", se::rhi::PipelineStageBit::ALL_COMMANDS_BIT)
    .value("TRANSFORM_FEEDBACK_BIT_EXT", se::rhi::PipelineStageBit::TRANSFORM_FEEDBACK_BIT_EXT)
    .value("CONDITIONAL_RENDERING_BIT_EXT", se::rhi::PipelineStageBit::CONDITIONAL_RENDERING_BIT_EXT)
    .value("ACCELERATION_STRUCTURE_BUILD_BIT_KHR", se::rhi::PipelineStageBit::ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
    .value("RAY_TRACING_SHADER_BIT_KHR", se::rhi::PipelineStageBit::RAY_TRACING_SHADER_BIT_KHR)
    .value("TASK_SHADER_BIT_NV", se::rhi::PipelineStageBit::TASK_SHADER_BIT_NV)
    .value("MESH_SHADER_BIT_NV", se::rhi::PipelineStageBit::MESH_SHADER_BIT_NV)
    .value("FRAGMENT_DENSITY_PROCESS_BIT", se::rhi::PipelineStageBit::FRAGMENT_DENSITY_PROCESS_BIT)
    .value("FRAGMENT_SHADING_RATE_ATTACHMENT_BIT", se::rhi::PipelineStageBit::FRAGMENT_SHADING_RATE_ATTACHMENT_BIT)
    .value("COMMAND_PREPROCESS_BIT", se::rhi::PipelineStageBit::COMMAND_PREPROCESS_BIT);

  py::enum_<se::rhi::AddressMode>(namespace_rhi, "EnumAddressMode")
    .value("CLAMP_TO_EDGE", se::rhi::AddressMode::CLAMP_TO_EDGE)
    .value("REPEAT", se::rhi::AddressMode::REPEAT)
    .value("MIRROR_REPEAT", se::rhi::AddressMode::MIRROR_REPEAT);
  
  py::enum_<se::rhi::FilterMode>(namespace_rhi, "EnumFilterMode")
    .value("NEAREST", se::rhi::FilterMode::NEAREST)
    .value("LINEAR", se::rhi::FilterMode::LINEAR);
  
  py::enum_<se::rhi::MipmapFilterMode>(namespace_rhi, "EnumMipmapFilterMode")
    .value("NEAREST", se::rhi::MipmapFilterMode::NEAREST)
    .value("LINEAR", se::rhi::MipmapFilterMode::LINEAR);

  class_BarrierDescriptor.def(py::init<se::rhi::PipelineStages, se::rhi::PipelineStages,
    se::rhi::DependencyTypeFlags, std::vector<se::rhi::MemoryBarrier*>,
    std::vector<se::rhi::BufferMemoryBarrierDescriptor>, std::vector<se::rhi::TextureMemoryBarrierDescriptor>>());

  py::class_<se::rhi::CommandEncoder> class_CommandEncoder(namespace_rhi, "CommandEncoder");
  class_rhi_device.def("createCommandEncoder", &se::rhi::Device::createCommandEncoder);
  class_CommandEncoder.def("finish", &se::rhi::CommandEncoder::finish, py::return_value_policy::reference);
  class_CommandEncoder.def("pipelineBarrier", &se::rhi::CommandEncoder::pipelineBarrier);
  
  py::class_<se::rhi::CommandBuffer> class_CommandBuffer(namespace_rhi, "CommandBuffer");

  // Export rhi::cudacontext structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::CUDABuffer> class_cudaBuffer(namespace_rhi, "CUDABuffer");
  py::class_<se::rhi::CUDAContext> class_cudacontext(namespace_rhi, "CUDAContext");
  class_cudacontext.def_static("initialize", py::overload_cast<se::rhi::Device*>(&se::rhi::CUDAContext::initialize));
  class_cudacontext.def_static("synchronize", &se::rhi::CUDAContext::synchronize);
  class_cudacontext.def_static("toCUDABuffer", &se::rhi::CUDAContext::toCUDABuffer);
  class_cudacontext.def_static("toCUDABufferInterval", &se::rhi::CUDAContext::toCUDABufferInterval);
  class_cudacontext.def_static("allocCUDABuffer", &se::rhi::CUDAContext::allocCUDABuffer);

  // Export rhi::pytorch structures
  // ------------------------------------------------------------------------
  py::enum_<se::rhi::DataType>(namespace_rhi, "EnumDataType")
    .value("Float16", se::rhi::DataType::Float16)
    .value("Float32", se::rhi::DataType::Float32)
    .value("Float64", se::rhi::DataType::Float64)
    .value("UINT8", se::rhi::DataType::UINT8)
    .value("INT8", se::rhi::DataType::INT8)
    .value("INT16", se::rhi::DataType::INT16)
    .value("INT32", se::rhi::DataType::INT32)
    .value("INT64", se::rhi::DataType::INT64);
  namespace_rhi.def_static("toTensor", &se::rhi::toTensor,
    py::arg("cudaBuffer"), py::arg("dimension"), py::arg("type") = se::rhi::DataType::Float32);

  // Export gfx:: structures
  // ------------------------------------------------------------------------
  py::class_<gfx_namescope> namespace_gfx(m, "gfx");
  py::class_<se::gfx::GFXContext> class_gfx_context(namespace_gfx, "Context");
  class_gfx_context.def_static("initialize", py::overload_cast<se::rhi::Device*>(&se::gfx::GFXContext::initialize));
  class_gfx_context.def_static("captureImage", static_cast<void(*)(se::gfx::TextureHandle)>(&se::gfx::captureImage));
  class_gfx_context.def_static("captureImage", static_cast<void(*)(se::gfx::TextureHandle src, std::string path)>(&se::gfx::captureImage));
  class_gfx_context.def_static("finalize", &se::gfx::GFXContext::finalize);
  class_gfx_context.def_static("load_scene_gltf", &se::gfx::GFXContext::load_scene_gltf);
  class_gfx_context.def_static("load_scene_pbrt", &se::gfx::GFXContext::load_scene_pbrt);
  class_gfx_context.def_static("load_scene_xml", &se::gfx::GFXContext::load_scene_xml);
  class_gfx_context.def_static("create_texture_file", &se::gfx::GFXContext::create_texture_file);
  class_gfx_context.def_static("create_sampler_desc", static_cast<se::gfx::SamplerHandle(*)
      (rhi::AddressMode, rhi::FilterMode, rhi::MipmapFilterMode)>
      (&se::gfx::GFXContext::create_sampler_desc), 
      py::arg("address"), py::arg("filter"), py::arg("mipmap"),
      py::return_value_policy::reference);
  class_gfx_context.def_static("createFlights", &se::gfx::GFXContext::createFlights);
  class_gfx_context.def_static("getFlights", &se::gfx::GFXContext::getFlights, py::return_value_policy::reference);
  class_gfx_context.def_static("getDevice", &se::gfx::GFXContext::getDevice, py::return_value_policy::reference);
  class_gfx_context.def_static("create_scene", &se::gfx::GFXContext::create_scene);
  class_gfx_context.def_static("load_shader_slang", 
    static_cast<std::vector<se::gfx::ShaderHandle>(*)(std::string const&,
      std::vector<std::pair<std::string, rhi::ShaderStageBit>> const&,
      std::vector<std::pair<char const*, char const*>> const&,
      bool)>(&se::gfx::GFXContext::load_shader_slang),
    py::arg("path"), py::arg("entrypoints"), py::arg("macros") = 
    std::vector<std::pair<char const*, char const*>>{}, py::arg("glsl_intermediate") = false,
    py::return_value_policy::reference);

  py::class_<se::gfx::Buffer> class_gfx_buffer(namespace_gfx, "Buffer");
  class_gfx_buffer
    .def("getBindingResource", &se::gfx::Buffer::getBindingResource)
    .def("getDevice", &se::gfx::Buffer::getDevice, py::return_value_policy::reference);

  py::class_<se::gfx::Texture> class_gfx_texture(namespace_gfx, "Texture");
  class_gfx_texture
    .def("getUAV", &se::gfx::Texture::getUAV, py::return_value_policy::reference)
    .def("getRTV", &se::gfx::Texture::getRTV, py::return_value_policy::reference)
    .def("getDSV", &se::gfx::Texture::getDSV, py::return_value_policy::reference)
    .def("getSRV", &se::gfx::Texture::getSRV, py::return_value_policy::reference)
    .def("getWidth", &se::gfx::Texture::getWidth)
    .def("getHeight", &se::gfx::Texture::getHeight);

  py::class_<se::gfx::BufferHandle> class_gfx_bufferHandle(namespace_gfx, "BufferHandle");
  class_gfx_bufferHandle.def("get", &se::gfx::BufferHandle::get, py::return_value_policy::reference);
  
  py::class_<se::gfx::TextureHandle> class_gfx_textureHandle(namespace_gfx, "TextureHandle");
  class_gfx_textureHandle.def("get", &se::gfx::TextureHandle::get, py::return_value_policy::reference);

  py::class_<se::gfx::DifferentiableParameter::Packet> class_gfx_parampacket(namespace_gfx, "ParameterPacket");
  class_gfx_parampacket.def_readwrite("dim_0", &se::gfx::DifferentiableParameter::Packet::dim_0)
    .def_readwrite("dim_1", &se::gfx::DifferentiableParameter::Packet::dim_1)
    .def_readwrite("dim_2", &se::gfx::DifferentiableParameter::Packet::dim_2)
    .def_readwrite("offset_primal", &se::gfx::DifferentiableParameter::Packet::offset_primal)
    .def_readwrite("default_value", &se::gfx::DifferentiableParameter::Packet::default_value)
    .def_readwrite("offset_grad", &se::gfx::DifferentiableParameter::Packet::offset_grad);

  py::class_<se::gfx::Scene> class_gfx_scene(namespace_gfx, "Scene");
  class_gfx_scene.def("updateTransform", &se::gfx::Scene::updateTransform)
    .def("updateGPUScene", &se::gfx::Scene::updateGPUScene)
    .def("createTexcoord", &se::gfx::Scene::createTexcoord)
    .def("getSceneLightCounts", &se::gfx::Scene::getSceneLightCounts)
    .def("getEditorActiveCameraIndex", &se::gfx::Scene::getEditorActiveCameraIndex)
    .def("getDefaultResolution", &se::gfx::Scene::getDefaultResolution)
    .def("getGPUScene", &se::gfx::Scene::getGPUScene, py::return_value_policy::reference)
    .def("serialize", static_cast<void(se::gfx::Scene::*)(std::string const&)>(&se::gfx::Scene::serialize));
  

  py::class_<se::gfx::Scene::GeometryDrawData>(namespace_gfx, "GeometryDrawData");

  py::enum_<se::gfx::Scene::TexcoordKind>(namespace_gfx, "TexcoordKind")
    .value("CopyCoord0", se::gfx::Scene::TexcoordKind::CopyCoord0)
    .value("XAtlas", se::gfx::Scene::TexcoordKind::XAtlas);

  py::class_<se::gfx::Scene::GPUScene> class_gfx_gpuscene(class_gfx_scene, "GPUScene");
  class_gfx_gpuscene.def("bindingResourceCamera", &se::gfx::Scene::GPUScene::bindingResourceCamera)
    .def("bindingResourcePosition", &se::gfx::Scene::GPUScene::bindingResourcePosition)
    .def("bindingResourceMaterial", &se::gfx::Scene::GPUScene::bindingResourceMaterial)
    .def("bindingResourceMedium", &se::gfx::Scene::GPUScene::bindingResourceMedium)
    .def("bindingResourceTextures", &se::gfx::Scene::GPUScene::bindingResourceTextures)
    .def("bindingResourceLight", &se::gfx::Scene::GPUScene::bindingResourceLight)
    .def("bindingResourceLightBVH", &se::gfx::Scene::GPUScene::bindingResourceLightBVH)
    .def("bindingResourceLightTrail", &se::gfx::Scene::GPUScene::bindingResourceLightTrail)
    .def("bindingResourceParamPacket", &se::gfx::Scene::GPUScene::bindingResourceParamPacket)
    .def("bindingResourceParamPrimal", &se::gfx::Scene::GPUScene::bindingResourceParamPrimal)
    .def("bindingResourceParamGradient", &se::gfx::Scene::GPUScene::bindingResourceParamGradient)
    .def("bindingResourceGrids", &se::gfx::Scene::GPUScene::bindingResourceGrids)
    .def("bindingSceneDescriptor", &se::gfx::Scene::GPUScene::bindingSceneDescriptor)
    .def("bindingResourceIndex", &se::gfx::Scene::GPUScene::bindingResourceIndex)
    .def("bindingResourceVertex", &se::gfx::Scene::GPUScene::bindingResourceVertex)
    .def("bindingResourceGeometry", &se::gfx::Scene::GPUScene::bindingResourceGeometry)
    .def("bindingResourceTLAS", &se::gfx::Scene::GPUScene::bindingResourceTLAS)
    .def("bindingResourceTLASPrev", &se::gfx::Scene::GPUScene::bindingResourceTLASPrev)
    .def("bindingResourceUvTLAS", &se::gfx::Scene::GPUScene::bindingResourceUvTLAS)
    .def("getPositionBuffer", &se::gfx::Scene::GPUScene::getPositionBuffer)
    .def("getIndexBuffer", &se::gfx::Scene::GPUScene::getIndexBuffer)
    .def("export_param_primal", &se::gfx::Scene::GPUScene::export_param_primal)
    .def("export_param_gradient", &se::gfx::Scene::GPUScene::export_param_gradient)
    .def("export_texture_parameters", &se::gfx::Scene::GPUScene::export_texture_parameters);

  py::class_<se::gfx::SceneHandle> class_gfx_sceneHandle(namespace_gfx, "SceneHandle");
  class_gfx_sceneHandle.def("get", &se::gfx::SceneHandle::get, py::return_value_policy::reference);

  py::class_<se::gfx::ShaderModule> class_gfx_shaderModule(namespace_gfx, "ShaderModule");
  py::class_<se::gfx::ShaderHandle> class_gfx_shaderHandle(namespace_gfx, "ShaderHandle");
  class_gfx_shaderHandle.def("get", &se::gfx::ShaderHandle::get, py::return_value_policy::reference);

  py::class_<se::gfx::SamplerHandle> class_gfx_samplerHandle(namespace_gfx, "SamplerHandle");
  class_gfx_samplerHandle.def("get", &se::gfx::SamplerHandle::get, py::return_value_policy::reference);

  py::class_<se::gfx::PMFConstructor> class_gfx_PMFConstructor(namespace_gfx, "PMFConstructor");
  class_gfx_PMFConstructor.def("upload_datapack", &se::gfx::PMFConstructor::upload_datapack);
  class_gfx_PMFConstructor.def("clear_datapack", &se::gfx::PMFConstructor::clear_datapack);
  class_gfx_PMFConstructor.def("binding_resource_buffer", &se::gfx::PMFConstructor::binding_resource_buffer);

  // Export rdg:: structures
  // ------------------------------------------------------------------------
  py::class_<rdg_namescope> namespace_rdg(m, "rdg");



  py::class_<se::rdg::PassReflection> class_pass_reflection(namespace_rdg, "PassReflection");
  class_pass_reflection.def(py::init<>())
    .def("addInput", &se::rdg::PassReflection::addInput, py::return_value_policy::reference)
    .def("addOutput", &se::rdg::PassReflection::addOutput, py::return_value_policy::reference)
    .def("addInputOutput", &se::rdg::PassReflection::addInputOutput, py::return_value_policy::reference)
    .def("addInternal", &se::rdg::PassReflection::addInternal, py::return_value_policy::reference);

  py::class_<se::rdg::ResourceInfo> class_ResourceInfo(namespace_rdg, "ResourceInfo");
  class_ResourceInfo
    .def("isBuffer", &se::rdg::ResourceInfo::isBuffer, py::return_value_policy::reference)
    .def("isTexture", &se::rdg::ResourceInfo::isTexture, py::return_value_policy::reference);
  
  py::class_<se::rdg::BufferInfo> class_BufferInfo(namespace_rdg, "BufferInfo");
  class_BufferInfo
    .def("withSize", &se::rdg::BufferInfo::withSize, py::return_value_policy::reference)
    .def("withUsages", &se::rdg::BufferInfo::withUsages, py::return_value_policy::reference)
    .def("withFlags", &se::rdg::BufferInfo::withFlags, py::return_value_policy::reference)
    .def("consume", &se::rdg::BufferInfo::consume, py::return_value_policy::reference);

  py::class_<se::rdg::BufferInfo::ConsumeEntry> class_BufferConsumeEntry(class_BufferInfo, "ConsumeEntry");
  class_BufferConsumeEntry
    .def(py::init<>())
    .def("addStage", &se::rdg::BufferInfo::ConsumeEntry::addStage, py::return_value_policy::reference)
    .def("setAccess", &se::rdg::BufferInfo::ConsumeEntry::setAccess, py::return_value_policy::reference)
    .def("setSubresource", &se::rdg::BufferInfo::ConsumeEntry::setSubresource, py::return_value_policy::reference);

  py::class_<se::rdg::TextureInfo> class_TextureInfo(namespace_rdg, "TextureInfo");
  class_TextureInfo
    .def("consume", &se::rdg::TextureInfo::consume, py::return_value_policy::reference)
    .def("setInfo", &se::rdg::TextureInfo::setInfo, py::return_value_policy::reference)
    .def("withSize", py::overload_cast<se::ivec3>(&se::rdg::TextureInfo::withSize), py::return_value_policy::reference)
    .def("withSize", py::overload_cast<se::vec3>(&se::rdg::TextureInfo::withSize), py::return_value_policy::reference)
    .def("withSizeRelative", &se::rdg::TextureInfo::withSizeRelative, py::return_value_policy::reference)
    .def("withLevels", &se::rdg::TextureInfo::withLevels, py::return_value_policy::reference)
    .def("withLayers", &se::rdg::TextureInfo::withLayers, py::return_value_policy::reference)
    .def("withSamples", &se::rdg::TextureInfo::withSamples, py::return_value_policy::reference)
    .def("withFormat", &se::rdg::TextureInfo::withFormat, py::return_value_policy::reference)
    .def("withStages", &se::rdg::TextureInfo::withStages, py::return_value_policy::reference)
    .def("withUsages", &se::rdg::TextureInfo::withUsages, py::return_value_policy::reference)
    .def("withFlags", &se::rdg::TextureInfo::withFlags, py::return_value_policy::reference)
    .def("getSize", &se::rdg::TextureInfo::getSize, py::return_value_policy::reference);

  py::enum_<se::rdg::TextureInfo::ConsumeType>(class_TextureInfo, "ConsumeType")
    .value("ColorAttachment", se::rdg::TextureInfo::ConsumeType::ColorAttachment)
    .value("DepthStencilAttachment", se::rdg::TextureInfo::ConsumeType::DepthStencilAttachment)
    .value("TextureBinding", se::rdg::TextureInfo::ConsumeType::TextureBinding)
    .value("StorageBinding", se::rdg::TextureInfo::ConsumeType::StorageBinding);

  py::class_<se::rdg::TextureInfo::ConsumeEntry> class_TextureInfo_ConsumeEntry(class_TextureInfo, "ConsumeEntry");
  class_TextureInfo_ConsumeEntry
    .def(py::init<>())
    .def(py::init<se::rdg::TextureInfo::ConsumeType, rhi::AccessFlags, rhi::PipelineStages,
      uint32_t, uint32_t, uint32_t, uint32_t, rhi::TextureLayout, bool, rhi::CompareFunction, uint32_t>(),
      py::arg("type"), py::arg("access") = 0, py::arg("stages") = 0,
      py::arg("level_beg") = 0, py::arg("level_end") = 1, py::arg("mip_beg") = 0, py::arg("mip_end") = 1,
      py::arg("layout") = rhi::TextureLayout::UNDEFINED, py::arg("depthWrite") = false,
      py::arg("depthCmp") = rhi::CompareFunction::ALWAYS, py::arg("attachLoc") = uint32_t(-1))
    .def("addStage", &se::rdg::TextureInfo::ConsumeEntry::addStage, py::return_value_policy::reference)
    .def("setLayout", &se::rdg::TextureInfo::ConsumeEntry::setLayout, py::return_value_policy::reference)
    .def("enableDepthWrite", &se::rdg::TextureInfo::ConsumeEntry::enableDepthWrite, py::return_value_policy::reference)
    .def("setDepthCompareFn", &se::rdg::TextureInfo::ConsumeEntry::setDepthCompareFn, py::return_value_policy::reference)
    .def("setSubresource", &se::rdg::TextureInfo::ConsumeEntry::setSubresource, py::return_value_policy::reference)
    .def("setAttachmentLoc", &se::rdg::TextureInfo::ConsumeEntry::setAttachmentLoc, py::return_value_policy::reference)
    .def("setBlendOperation", &se::rdg::TextureInfo::ConsumeEntry::setBlendOperation, py::return_value_policy::reference)
    .def("setSourceBlenderFactor", &se::rdg::TextureInfo::ConsumeEntry::setSourceBlenderFactor, py::return_value_policy::reference)
    .def("setTargetBlenderFactor", &se::rdg::TextureInfo::ConsumeEntry::setTargetBlenderFactor, py::return_value_policy::reference)
    .def("setAccess", &se::rdg::TextureInfo::ConsumeEntry::setAccess, py::return_value_policy::reference);

    //.def(py::init<se::rdg::TextureInfo::ConsumeType, rhi::AccessFlags, rhi::PipelineStages,
    //  uint32_t, uint32_t, uint32_t, uint32_t, rhi::TextureLayout, bool, rhi::CompareFunction, uint32_t>(),
    //  py::arg("type"), py::arg("access") = 0, py::arg("stages") = 0,
    //  py::arg("level_beg") = 0, py::arg("level_end") = 1, py::arg("mip_beg") = 0, py::arg("mip_end") = 1,
    //  py::arg("layout") = rhi::TextureLayout::UNDEFINED, py::arg("depthWrite") = false,
    //  py::arg("depthCmp") = rhi::CompareFunction::ALWAYS, py::arg("attachLoc") = -1);

  py::class_<se::rdg::RenderContext> class_render_context(namespace_rdg, "RenderContext");
  class_render_context.def(py::init<>())
    .def_readwrite("cmdEncoder", &se::rdg::RenderContext::cmdEncoder)
    .def_readwrite("flightIdx", &se::rdg::RenderContext::flightIdx);

  py::class_<se::rdg::RenderData> class_render_data(namespace_rdg, "RenderData");
  class_render_data.def(py::init<>())
    .def("getTexture", &se::rdg::RenderData::getTexture)
    .def("getBuffer", &se::rdg::RenderData::getBuffer)
    .def("getScene", &se::rdg::RenderData::getScene);

  py::class_<se::rdg::Pass, se::rdg::PyPass<>>(namespace_rdg, "Pass")
    .def("pass", &se::rdg::Pass::pass)
    .def("renderUI", &se::rdg::Pass::renderUI);
  
  py::class_<se::rdg::DummyPass, se::rdg::Pass, se::rdg::PyDummyPass<>>(namespace_rdg, "DummyPass")
    .def(py::init<>())
    .def("execute", &se::rdg::RenderPass::execute);

  py::class_<se::rdg::PipelinePass, se::rdg::Pass, se::rdg::PyPipelinePass<>>(namespace_rdg, "PipelinePass");
    
  py::class_<se::rdg::RenderPass, se::rdg::PipelinePass, se::rdg::PyRenderPass<>>(namespace_rdg, "RenderPass")
    .def(py::init<>())
    .def("reflect", &se::rdg::RenderPass::reflect)
    .def("execute", &se::rdg::RenderPass::execute)
    .def("beforeDirectDrawcall", &se::rdg::RenderPass::beforeDirectDrawcall)
    .def("updateBindings", &se::rdg::RenderPass::updateBindings)
    .def("setRenderPassDescriptor", &se::rdg::RenderPass::setRenderPassDescriptor)
    .def("issueDirectDrawcalls", &se::rdg::RenderPass::issueDirectDrawcalls)
    .def("init", static_cast<void(se::rdg::RenderPass::*)(
      gfx::ShaderModule*, gfx::ShaderModule*)>(&se::rdg::RenderPass::init))
    .def("init", static_cast<void(se::rdg::RenderPass::*)(
      gfx::ShaderModule*, gfx::ShaderModule*, gfx::ShaderModule*)>(&se::rdg::RenderPass::init))
    .def("beginPass", static_cast<rhi::RenderPassEncoder* (se::rdg::RenderPass::*)(
      rdg::RenderContext*, gfx::Texture*)>(&se::rdg::RenderPass::beginPass), 
      py::return_value_policy::reference)
    .def("beginPass", static_cast<rhi::RenderPassEncoder* (se::rdg::RenderPass::*)(
      rdg::RenderContext*, uint32_t, uint32_t)>(&se::rdg::RenderPass::beginPass), 
      py::return_value_policy::reference);
  
  py::class_<se::rdg::FullScreenPass, se::rdg::RenderPass, se::rdg::PyFullScreenPass<>>(namespace_rdg, "FullScreenPass")
    .def(py::init<>())
    .def("init", &se::rdg::FullScreenPass::init)
    .def("reflect", &se::rdg::FullScreenPass::reflect)
    .def("execute", &se::rdg::FullScreenPass::execute)
    .def("updateBindings", &se::rdg::FullScreenPass::updateBindings)
    .def("dispatchFullScreen", &se::rdg::FullScreenPass::dispatchFullScreen);

  py::class_<se::rdg::ComputePass, se::rdg::PipelinePass, se::rdg::PyComputePass<>>(namespace_rdg, "ComputePass")
    .def(py::init<>())
    .def("reflect", &se::rdg::ComputePass::reflect)
    .def("execute", &se::rdg::ComputePass::execute)
    .def("updateBindings", &se::rdg::ComputePass::updateBindings)
    .def("beginPass", &se::rdg::ComputePass::beginPass, py::return_value_policy::reference)
    .def("init", &se::rdg::ComputePass::init);

  py::class_<se::rdg::Graph, se::rdg::PyGraph<>>(namespace_rdg, "Graph")
    .def(py::init<>())
    .def("build", &se::rdg::Graph::build)
    .def("execute", &se::rdg::Graph::execute)
    .def("readback", &se::rdg::Graph::readback)
    .def("renderUI", &se::rdg::Graph::renderUI)
    .def("addEdge", static_cast<void(se::rdg::Graph::*)(std::string const&, 
      std::string const&, std::string const&, std::string const&)>(&se::rdg::Graph::addEdge))
    .def("addEdge", static_cast<void(se::rdg::Graph::*)(std::string const&, 
      std::string const&)>(&se::rdg::Graph::addEdge))
    .def("markOutput", &se::rdg::Graph::markOutput)
    .def("getOutput", &se::rdg::Graph::getOutput)
    .def("getBufferResource", &se::rdg::Graph::getBufferResource)
    .def("getTextureResource", &se::rdg::Graph::getTextureResource)
    .def("getPass", static_cast<se::rdg::Pass*(se::rdg::Graph::*)(size_t)>(&se::rdg::Graph::getPass), py::return_value_policy::reference)
    .def("getPass", static_cast<se::rdg::Pass*(se::rdg::Graph::*)(std::string const&)>(&se::rdg::Graph::getPass), py::return_value_policy::reference)
    .def("addPass", static_cast<void(se::rdg::Graph::*)(se::rdg::Pass*, std::string const&)>(&se::rdg::Graph::addPass));

  py::class_<se::rdg::Pipeline, se::rdg::PyPipeline<>>(namespace_rdg, "Pipeline")
    .def(py::init<>())
    .def("build", &se::rdg::Pipeline::build)
    .def("execute", &se::rdg::Pipeline::execute)
    .def("readback", &se::rdg::Pipeline::readback)
    .def("renderUI", &se::rdg::Pipeline::renderUI)
    .def("getActiveGraphs", &se::rdg::Pipeline::getActiveGraphs)
    .def("setStandardSize", &se::rdg::Pipeline::setStandardSize)
    .def("bindScene", &se::rdg::Pipeline::bindScene)
    .def("getOutput", &se::rdg::Pipeline::getOutput)
    .def("pipeline", &se::rdg::Pipeline::pipeline, py::return_value_policy::reference);

  py::class_<se::rdg::SingleGraphPipeline, se::rdg::Pipeline, se::rdg::PySingleGraphPipeline<>>(namespace_rdg, "SingleGraphPipeline")
    .def(py::init<>())
    .def("execute", &se::rdg::SingleGraphPipeline::execute)
    .def("getActiveGraphs", &se::rdg::SingleGraphPipeline::getActiveGraphs)
    .def("getOutput", &se::rdg::SingleGraphPipeline::getOutput)
    .def("readback", &se::rdg::SingleGraphPipeline::readback)
    .def("build", &se::rdg::SingleGraphPipeline::build)
    .def("setGraph", &se::rdg::SingleGraphPipeline::setGraph);

  // Export :: predefined bxdfs
  // ------------------------------------------------------------------------
  py::class_<bxdf_namescope> namespace_bxdf(m, "bxdfs");
  py::class_<se::gfx::IBxDF, se::gfx::PyBxDF<>>(namespace_bxdf, "IBxDF");
  py::class_<se::EPFLBrdf, se::gfx::IBxDF>(namespace_bxdf, "EPFLBrdf")
    .def(py::init<std::string>())
    .def_static("updateGPUResource", &se::EPFLBrdf::updateGPUResource)
    .def_static("bindingResourceBuffer", &se::EPFLBrdf::bindingResourceBuffer)
    .def_static("bindingResourceBRDFs", &se::EPFLBrdf::bindingResourceBRDFs);
  
  py::class_<light_namescope> namespace_lights(m, "lights");
  py::class_<se::EnvmapLight> class_EnvmapLight(namespace_lights, "EnvmapLight");
  py::enum_<se::EnvmapLight::ImportanceType>(class_EnvmapLight, "ImportanceType")
      .value("Luminance", se::EnvmapLight::ImportanceType::Luminance)
      .value("Length", se::EnvmapLight::ImportanceType::Length);
  class_EnvmapLight.def(py::init<std::string, se::EnvmapLight::ImportanceType>());
  class_EnvmapLight.def("width", &se::EnvmapLight::width);
  class_EnvmapLight.def("height", &se::EnvmapLight::height);
  class_EnvmapLight.def("rgb_integrated", &se::EnvmapLight::rgb_integrated);
  class_EnvmapLight.def("condition_offset", &se::EnvmapLight::condition_offset);
  class_EnvmapLight.def("marginal_offset", &se::EnvmapLight::marginal_offset);
  class_EnvmapLight.def("get_texture", &se::EnvmapLight::get_texture, py::return_value_policy::reference);

  // Export :: predefined passes
  // ------------------------------------------------------------------------
  py::class_<pass_namescope> namespace_pass(m, "passes");
  // rasterizer passes
  py::class_<se::RasterizerShadePass, se::rdg::RenderPass>(namespace_pass, "RasterizerShadePass")
    .def(py::init<>());
  // editor passes
  py::class_<se::EditorInitPass, se::rdg::FullScreenPass>(namespace_pass, "EditorInitPass")
    .def(py::init<>());
  py::class_<se::BillboardPass, se::rdg::RenderPass>(namespace_pass, "BillboardPass")
    .def(py::init<>())
    .def("setExternalBuffer", &se::BillboardPass::setExternalBuffer);
  py::class_<se::Line3DPass, se::rdg::RenderPass>(namespace_pass, "Line3DPass")
    .def(py::init<>())
    .def("clear", &se::Line3DPass::clear)
    .def("addLine", &se::Line3DPass::addLine)
    .def("addAABB", &se::Line3DPass::addAABB)
    .def("setExternalBuffer", &se::Line3DPass::setExternalBuffer);
  // postprocss passes
  py::class_<se::AccumulatePass, se::rdg::ComputePass>(namespace_pass, "AccumulatePass")
    .def(py::init<se::ivec3>());

  // cbt passes
  py::class_<cbt_namescope> namespace_cbt(namespace_pass, "cbt");
  py::class_<se::cbt::CreateCBTPass, se::rdg::DummyPass>(namespace_cbt, "CreateCBTPass")
    .def(py::init<int, int>())
    .def("reflect", &se::cbt::CreateCBTPass::reflect)
    .def("execute", &se::cbt::CreateCBTPass::execute);
  py::class_<se::cbt::SumReductionFusedPass, se::rdg::ComputePass>(namespace_cbt, "SumReductionFusedPass")
    .def(py::init<int>())
    .def("reflect", &se::cbt::CreateCBTPass::reflect)
    .def("execute", &se::cbt::CreateCBTPass::execute);
  py::class_<se::cbt::SumReductionOneLayerPass, se::rdg::ComputePass>(namespace_cbt, "SumReductionOneLayerPass")
    .def(py::init<int>())
    .def("reflect", &se::cbt::CreateCBTPass::reflect)
    .def("execute", &se::cbt::CreateCBTPass::execute);
  py::class_<se::cbt::CBTSpatialTreeVisualizePass, se::rdg::RenderPass>(
    namespace_cbt, "CBTSpatialTreeVisualizePass").def(py::init<size_t>());
}