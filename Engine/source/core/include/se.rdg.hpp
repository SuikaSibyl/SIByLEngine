#pragma once
#include "se.rhi.hpp"
#include "se.gfx.hpp"
#include "se.math.hpp"

namespace se {
namespace rdg {
  /**
   * Information that describe a buffer resource in RDG
   * @param * size		 : size of the buffer
   * @param * usages;	 : usage of the texture};
   * @param * flags	 : some flags describe the resource
   */
  struct BufferInfo {
    uint32_t m_size = 0;
    Flags<rhi::BufferUsageEnum> m_usages = 0;
    gfx::BufferHandle m_reference;
    gfx::Buffer::ConsumeState m_consumeHistories;
    Flags<rhi::MemoryPropertyEnum> m_memoryProperties = 0;

    auto with_size(uint32_t size) noexcept -> BufferInfo&;
    auto with_usages(Flags<rhi::BufferUsageEnum> usages) noexcept -> BufferInfo&;
    auto with_memory_properties(Flags<rhi::MemoryPropertyEnum> prop) noexcept -> BufferInfo&;
    auto consume(gfx::Buffer::ConsumeEntry const& entry) noexcept -> BufferInfo&;
    auto to_descriptor() noexcept -> rhi::BufferDescriptor;
  };

  struct TextureInfo {
    using Size = std::variant<se::ivec3, se::vec3>;
    enum struct SizeDefine {
      Absolute,              // * define size by absolute value
      Relative,              // * define size by relative value
      RelativeToAnotherTex,  // * define size by relative value
    };

    Size m_size = { se::vec3{1.f} };
    SizeDefine m_sizeDef = SizeDefine::Relative;
    uint32_t m_levels = 1;
    uint32_t m_layers = 1;
    uint32_t m_samples = 1;
    rhi::TextureFormat m_format = rhi::TextureFormat::RGBA8_UNORM;
    Flags<rhi::TextureUsageEnum> m_usages = 0;
    Flags<rhi::TextureFeatureEnum> m_tflags = 0;
    Flags<rhi::PipelineStageEnum> m_stages = 0;
    Flags<rhi::AccessFlagEnum> m_access = 0;
    rhi::TextureLayoutEnum m_laytout = rhi::TextureLayoutEnum::GENERAL;
    Flags<rhi::ShaderStageEnum> m_sflags = 0;
    gfx::TextureHandle m_reference;
    std::string m_sizeRefName;
    gfx::Texture::ConsumeState m_consumeHistories;

    auto consume(gfx::Texture::ConsumeEntry const& entry) noexcept -> TextureInfo&;
    auto consume_as_storage_binding_in_compute() noexcept -> TextureInfo&;
    auto consume_as_color_attachment_at(uint32_t loc) noexcept -> TextureInfo&;
    auto consume_as_depth_stencil_attachment_at(
      uint32_t loc,
      bool depth_write = true,
      se::rhi::CompareFunction depth_compare = se::rhi::CompareFunction::LESS_EQUAL
    ) noexcept -> TextureInfo&;
    auto set_info(TextureInfo const& x) noexcept -> TextureInfo&;
    auto with_size(se::ivec3 absolute) noexcept -> TextureInfo&;
    auto with_size(se::vec3 relative) noexcept -> TextureInfo&;
    auto with_size_relative(std::string const& src, se::vec3 relative = { 1. }) noexcept -> TextureInfo&;
    auto with_levels(uint32_t levels) noexcept -> TextureInfo&;
    auto with_layers(uint32_t layers) noexcept -> TextureInfo&;
    auto with_samples(uint32_t samples) noexcept -> TextureInfo&;
    auto with_format(rhi::TextureFormat format) noexcept -> TextureInfo&;
    auto with_stages(Flags<rhi::ShaderStageEnum> flags) noexcept -> TextureInfo&;
    auto with_usages(Flags<rhi::TextureUsageEnum> usages) noexcept -> TextureInfo&;
    auto get_size(se::ivec3 ref) const noexcept -> uvec3;
  };

  /** Information that describe a resource in RDG */
  struct ResourceInfo {
    enum struct Type { 
      UNKNOWN, 
      Buffer, 
      Texture 
    } m_type = Type::UNKNOWN;

    struct Info {
      void* null;
      BufferInfo buffer;
      TextureInfo texture;
    } m_info;

    uint32_t m_resoureceID = 0;
    size_t m_devirtualizeID = (-1);
    ResourceInfo* m_prev = nullptr;

    ResourceInfo() : m_type(Type::UNKNOWN) {};
    /* declare the resource is a buffer */
    auto is_buffer() noexcept -> BufferInfo&;
    /* declare the resource is a texture */
    auto is_texture() noexcept -> TextureInfo&;
  };

  struct BufferResource {
    struct ConsumeHistory {
      size_t passID;
      gfx::Buffer::ConsumeState entries;
    };
    rhi::BufferDescriptor m_desc;
    gfx::BufferHandle m_buffer;
    std::vector<ConsumeHistory> m_cosumeHistories;
    std::string m_name;
    // the first state of the resource in the graph
    std::optional<gfx::Buffer::ResourceStateMachine> m_startState;
    // the last state of the resource in the graph
    std::optional<gfx::Buffer::ResourceStateMachine> m_endState;
  };

  struct TextureResource {
    struct ConsumeHistory {
      size_t passID;
      gfx::Texture::ConsumeState entry;
    };
    rhi::TextureDescriptor m_desc;
    gfx::TextureHandle m_texture;
    std::vector<ConsumeHistory> m_cosumeHistories;
    // the first state of the resource in the graph
    std::optional<gfx::Texture::ResourceStateMachine> m_startState;
    // the last state of the resource in the graph
    std::optional<gfx::Texture::ResourceStateMachine> m_endState;

    std::string m_name;
  };

  struct PassReflection {
    uint32_t m_indexOffset = 0;
    std::unordered_map<std::string, ResourceInfo> m_inputResources = {};
    std::unordered_map<std::string, ResourceInfo> m_outputResources = {};
    std::unordered_map<std::string, ResourceInfo> m_inputOutputResources = {};
    std::unordered_map<std::string, ResourceInfo> m_internalResources = {};

    PassReflection();
    auto add_input(std::string const& name) noexcept -> ResourceInfo&;
    auto add_output(std::string const& name) noexcept -> ResourceInfo&;
    auto add_input_output(std::string const& name) noexcept -> ResourceInfo&;
    auto add_internal(std::string const& name) noexcept -> ResourceInfo&;
    auto add_external(std::string const& name) noexcept -> ResourceInfo&;
    auto get_depth_stencil_state() noexcept -> rhi::DepthStencilState;
    auto get_color_target_state() noexcept -> std::vector<rhi::ColorTargetState>;
    auto get_resource_info(std::string const& name) noexcept -> ResourceInfo*;
  };

  auto to_buffer_descriptor(BufferInfo const& info) noexcept -> rhi::BufferDescriptor;
  auto to_texture_descriptor(TextureInfo const& info, se::ivec3 ref_size) noexcept -> rhi::TextureDescriptor;
  
  struct RenderContext {
    rhi::CommandEncoder* cmdEncoder;
    size_t flightIdx = 0;

    RenderContext() = default;
    RenderContext(rhi::CommandEncoder* encoder, size_t idx = 0)
      :cmdEncoder(encoder), flightIdx(idx) {};
  };

  struct Graph;
  struct Pass;
  struct PipelinePass;

  struct RenderData {
    struct DelegateData {
      rhi::CommandEncoder* cmdEncoder;
      union {
        rhi::RenderPassEncoder* render;
        rhi::ComputePassEncoder* compute;
        //rhi::RayTracingPassEncoder* trace;
      } passEncoder;
      PipelinePass* pipelinePass;
      void* customData;
    };

    Graph* m_graph;
    Pass* m_pass;
    std::unordered_map<std::string, std::vector<rhi::BindGroupEntry>*> m_bindGroups;
    std::unordered_map<std::string, rhi::BindingResource> m_bindingResources;
    std::unordered_map<std::string, se::uvec2> m_uvec2s;
    std::unordered_map<std::string, uint32_t> m_uints;
    //std::unordered_map<std::string, std::function<void(DelegateData const&)>> m_delegates;
    std::unordered_map<std::string, void*> m_ptrs;
    std::unordered_map<std::string, se::mat4> m_mat4s;
    std::optional<gfx::SceneHandle> m_scene;
    std::optional<gfx::SceneBatchHandle> m_sceneBatch;

    //auto setDelegate(
    //  std::string const& name,
    //  std::function<void(DelegateData const&)> const& fn
    //) noexcept -> void;
    //auto getDelegate(
    //  std::string const& name
    //) const noexcept -> std::function<void(DelegateData const&)> const&;
    //auto setBindingResource(
    //  std::string const& name,
    //  rhi::BindingResource const& bindGroupEntry
    //) noexcept -> void;
    //auto getBindingResource(std::string const& name) const noexcept
    //  -> std::optional<rhi::BindingResource>;
    //auto setBindGroupEntries(
    //  std::string const& name,
    //  std::vector<rhi::BindGroupEntry>* bindGroup) noexcept -> void;
    //auto getBindGroupEntries(std::string const& name) const noexcept
    //  -> std::vector<rhi::BindGroupEntry>*;
    auto get_texture(std::string const& name) const noexcept -> gfx::TextureHandle;
    auto get_texture_id(std::string const& name) const noexcept -> uint32_t;
    auto get_buffer(std::string const& name) const noexcept -> gfx::BufferHandle;

    //auto setUVec2(std::string const& name, se::uvec2 v) noexcept -> void;
    //auto getUVec2(std::string const& name) const noexcept -> se::uvec2;
    //auto setUInt(std::string const& name, uint32_t v) noexcept -> void;
    //auto getUInt(std::string const& name) const noexcept -> uint32_t;
    //auto setPtr(std::string const& name, void* v) noexcept -> void;
    //auto getPtr(std::string const& name) const noexcept -> void*;
    //auto setMat4(std::string const& name, se::mat4 m) noexcept -> void;
    //auto getMat4(std::string const& name) const noexcept -> se::mat4;

    auto set_scene(gfx::SceneHandle) noexcept -> void;
    auto set_scene_batch(gfx::SceneBatchHandle) noexcept -> void;
    auto get_scene() const noexcept -> gfx::SceneHandle;
    auto get_scene_batch() const noexcept -> gfx::SceneBatchHandle;
  };

  struct Pass {
    std::string m_identifier;
    PassReflection m_pReflection;
    std::vector<size_t> m_subgraphStack;
    rhi::DebugLabelDescriptor m_marker;

    Pass() = default;
    Pass(Pass&&) = default;
    Pass(Pass const&) = delete;
    virtual ~Pass() = default;
    auto operator=(Pass&&)->Pass & = default;
    auto operator=(Pass const&)->Pass & = delete;

    auto pass() noexcept -> Pass* { return this; }
    virtual auto reflect(PassReflection& reflect) noexcept -> PassReflection { return reflect; };
    virtual auto execute(RenderContext* context, RenderData const& renderData) noexcept -> void {};
    virtual auto readback(RenderData const& renderData) noexcept -> void {}
    virtual auto render_ui() noexcept -> void {}
    virtual auto generate_marker() noexcept -> void;
    virtual auto init() noexcept -> void;
  };

  struct PipelinePass : public Pass {
    gfx::ShaderReflection m_reflection;
    std::unique_ptr<rhi::PipelineLayout> m_pipelineLayout;
    std::vector<std::unique_ptr<rhi::BindGroupLayout>> m_bindgroupLayouts;
    std::vector<std::array<std::unique_ptr<rhi::BindGroup>, SE_FRAME_FLIGHTS_COUNT>> m_bindgroups;

    PipelinePass() = default;
    virtual ~PipelinePass() = default;
    //
    auto get_bindgroup(RenderContext* context, uint32_t size) noexcept -> rhi::BindGroup*;
    // Direct update binding with binding location find by string
    auto update_binding(RenderContext* context, std::string const& name,
      rhi::BindingResource const& resource) noexcept -> void;
    // Direct update binding of a scene
    auto update_binding_scene(RenderContext* context, gfx::SceneHandle scene) noexcept -> void;
    // Direct update binding of a scene batch
    auto update_binding_scene_batch(RenderContext* context, gfx::SceneBatchHandle sceneBatch) noexcept -> void;

    //
    auto update_bindings(
      RenderContext* context,
      std::vector<std::pair<std::string, rhi::BindingResource>> const& bindings) noexcept -> void;

    virtual auto init(std::vector<gfx::ShaderModule*> shaderModules) noexcept -> void;
  };

  struct RenderPass : public PipelinePass {
    rhi::RenderPassDescriptor m_renderPassDescriptor;
    std::array<std::unique_ptr<rhi::RenderPipeline>, SE_FRAME_FLIGHTS_COUNT> m_pipelines;
    std::array<std::unique_ptr<rhi::RenderPassEncoder>, SE_FRAME_FLIGHTS_COUNT> m_passEncoders;
    std::optional<se::gfx::ShaderHandle> m_vertexShader;
    std::optional<se::gfx::ShaderHandle> m_fragmentShader;
    std::optional<se::gfx::ShaderHandle> m_geometryShader;
    std::optional<se::gfx::ShaderHandle> m_taskShader;
    std::optional<se::gfx::ShaderHandle> m_meshShader;

    RenderPass() = default;
    virtual ~RenderPass() = default;
    auto begin_pass(RenderContext* context, gfx::Texture* target) noexcept -> rhi::RenderPassEncoder*;
    auto begin_pass(RenderContext* context, uint32_t width, uint32_t height) noexcept -> rhi::RenderPassEncoder*;
    auto prepare_delegate_data(RenderContext* context, RenderData const& renderData) noexcept -> RenderData::DelegateData;
    virtual auto generate_marker() noexcept -> void override;
    virtual auto init(std::string const& shader_path) noexcept -> void;
    virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment) noexcept -> void;
    virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment, gfx::ShaderModule* geometry) noexcept -> void;
    // utils functions
    auto set_render_pass_descriptor(rhi::RenderPassDescriptor const& renderPassDescriptor) noexcept -> void;
    //auto issueDirectDrawcalls(rhi::RenderPassEncoder* encoder, gfx::SceneHandle scene) noexcept -> void;
    //virtual auto beforeDirectDrawcall(rhi::RenderPassEncoder* encoder, int geometry_idx,
    //  gfx::Scene::GeometryDrawData const& data) noexcept -> void {}

    auto prepare_dispatch(RenderContext* context, gfx::Texture* target) noexcept -> void;
    auto prepare_dispatch(rdg::RenderContext* context, uint32_t width, uint32_t height) noexcept -> void;

    using RenderPipelineDescCallback = std::function<void(rhi::RenderPipelineDescriptor&)>;
    virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment,
      std::optional<RenderPipelineDescCallback> callback) noexcept -> void;
    virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* geometry,
      gfx::ShaderModule* fragment, std::optional<RenderPipelineDescCallback> callback = std::nullopt) noexcept -> void;
  };

  struct ComputePass : public PipelinePass {
    ComputePass() = default;
    ComputePass(std::string const& shader_path);
    ComputePass(std::string const& shader_path, std::vector<std::pair<char const*, char const*>> const& macros);
    virtual ~ComputePass() = default;
    std::unique_ptr<rhi::ComputePipeline> m_pipeline;
    std::array<std::unique_ptr<rhi::ComputePassEncoder>, SE_FRAME_FLIGHTS_COUNT> m_passEncoders;
    std::optional<se::gfx::ShaderHandle> m_computeShader;

    auto begin_pass(rdg::RenderContext* context) noexcept -> rhi::ComputePassEncoder*;
    auto prepare_dispatch(rdg::RenderContext* context) noexcept -> void;
    virtual auto generate_marker() noexcept -> void;
    virtual auto init(gfx::ShaderModule* comp) noexcept -> void;
    virtual auto init(std::string const& comp) noexcept -> void;
    virtual auto init(std::string const& comp, std::vector<std::pair<char const*, char const*>> const& macros) noexcept -> void;
  };

  struct DAG {
    std::unordered_map<uint32_t, std::set<uint32_t>> adj;
    auto add_edge(uint32_t src, uint32_t dst) noexcept -> void;
    auto reverse() const noexcept -> DAG;
  };

	struct Graph {
    RenderData m_renderData;
    std::string m_name = "unnamed graph";
    std::string m_outputPass;
    std::string m_outputResource;
    se::ivec3 m_standardSize = { 1280, 720, 1 };
    std::vector<size_t> m_flattenedPasses;
    std::unordered_map<size_t, Pass*> m_passes;
    std::unordered_map<size_t, TextureResource> m_textureResources;
    std::unordered_map<size_t, BufferResource> m_bufferResources;
    std::unordered_map<size_t, std::vector<rhi::BarrierDescriptor>> m_barriers;
    std::list<std::pair<uint32_t, uint32_t>> m_edges;
    bool m_openEditor = false;
    DAG m_dag;

    virtual ~Graph() = default;
    auto add_pass(Pass* pass, std::string const& identifier) noexcept -> void;
    auto add_edge(std::string const& src_pass, std::string const& src_resource,
      std::string const& dst_pass, std::string const& dst_resource) noexcept -> void;

    auto set_standard_size(int width, int height) noexcept -> void;
    auto get_output() noexcept -> std::optional<gfx::TextureHandle>;
    auto get_output_index() noexcept -> std::optional<uint32_t>;
    auto mark_output(std::string const& pass, std::string const& output) noexcept -> void;
    auto get_render_data() noexcept -> RenderData& { return m_renderData; }
    auto build() noexcept -> void;
		auto on_draw_gui() noexcept -> void;
    auto execute(rhi::CommandEncoder* encoder) noexcept -> void;
    auto on_draw_inspector() noexcept -> void;
    virtual auto render_ui() noexcept -> void {};
	};
}
}