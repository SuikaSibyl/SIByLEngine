#pragma once
#include <se.core.hpp>
#include <se.math.hpp>
#include <se.rhi.hpp>
#include <se.gfx.hpp>
#include <se.editor.hpp>

namespace se::editor {
struct RDGViewerWidget;
}

namespace se::rdg {
enum struct ResourceFlag : uint32_t {
  None = 0 << 0,
  Persistent = 1 << 0,
};
using ResourceFlags = uint32_t;

/**
 * Information that describe a buffer resource in RDG
 * @param * size		 : size of the buffer
 * @param * usages;	 : usage of the texture};
 * @param * flags	 : some flags describe the resource
 */
struct SIByL_API BufferInfo {
  uint32_t size = 0;
  rhi::BufferUsages usages = 0;
  ResourceFlags flags = 0;
  gfx::BufferHandle reference;

  struct SIByL_API ConsumeEntry {
    rhi::AccessFlags access;
    rhi::PipelineStages stages;
    uint64_t offset = 0;
    uint64_t size = uint64_t(-1);
    inline auto addStage(rhi::PipelineStages stage) noexcept -> ConsumeEntry&;
    inline auto setAccess(uint32_t acc) noexcept -> ConsumeEntry&;
    auto setSubresource(uint64_t offset, uint64_t size) noexcept -> ConsumeEntry&;
  };

  auto withSize(uint32_t size) noexcept -> BufferInfo&;
  auto withUsages(rhi::BufferUsages usages) noexcept -> BufferInfo&;
  auto withFlags(ResourceFlags flags) noexcept -> BufferInfo&;
  auto consume(ConsumeEntry const& entry) noexcept -> BufferInfo&;

  std::vector<ConsumeEntry> consumeHistories;
};

struct SIByL_API TextureInfo {
  TextureInfo() = default;
  ~TextureInfo() = default;

  enum struct SizeDefine {
    Absolute,              // * define size by absolute value
    Relative,              // * define size by relative value
    RelativeToAnotherTex,  // * define size by relative value
  };
  union Size {
    se::ivec3 absolute;  // * the absolute size
    se::vec3 relative;   // * the relative size
  };
  Size size = Size{.relative = se::vec3{1.f}};
  SizeDefine sizeDef = SizeDefine::Relative;
  uint32_t levels = 1;
  uint32_t layers = 1;
  uint32_t samples = 1;
  rhi::TextureFormat format = rhi::TextureFormat::RGBA8_UNORM;
  rhi::TextureUsages usages = 0;
  rhi::TextureFlags tflags = (uint32_t)rhi::TextureFlagBit::NONE;
  ResourceFlags flags = 0;
  rhi::PipelineStages stages = 0;
  rhi::AccessFlags access = 0;
  rhi::TextureLayout laytout = rhi::TextureLayout::GENERAL;
  rhi::ShaderStages sflags = 0;

  gfx::TextureHandle reference;
  std::string sizeRefName;

  enum struct ConsumeType {
    ColorAttachment,
    DepthStencilAttachment,
    TextureBinding,
    StorageBinding,
  };

  struct SIByL_API ConsumeEntry {
    ConsumeType type;
    rhi::AccessFlags access;
    rhi::PipelineStages stages;
    uint32_t level_beg = 0;
    uint32_t level_end = 1;
    uint32_t mip_beg = 0;
    uint32_t mip_end = 1;
    rhi::TextureLayout layout;
    bool depthWrite = false;
    rhi::CompareFunction depthCmp = rhi::CompareFunction::ALWAYS;
    uint32_t attachLoc = uint32_t(-1);
    rhi::BlendOperation bldOperation = rhi::BlendOperation::ADD;
    rhi::BlendFactor srcFactor = rhi::BlendFactor::ONE;
    rhi::BlendFactor dstFactor = rhi::BlendFactor::ZERO;

    ConsumeEntry() = default;
    ConsumeEntry(ConsumeType _type,
      rhi::AccessFlags _access = 0, rhi::PipelineStages _stages = 0,
      uint32_t _level_beg = 0, uint32_t _level_end = 1,
      uint32_t _mip_beg = 0, uint32_t _mip_end = 1,
      rhi::TextureLayout _layout = rhi::TextureLayout::UNDEFINED,
      bool _depthWrite = false, rhi::CompareFunction _depthCmp = rhi::CompareFunction::ALWAYS,
      uint32_t _attachLoc = uint32_t(-1)) : type(_type), access(_access), stages(_stages),
      level_beg(_level_beg), level_end(_level_end), mip_beg(_mip_beg), mip_end(_mip_end),
      layout(_layout), depthWrite(_depthWrite), depthCmp(_depthCmp), attachLoc(_attachLoc) {}

    auto addStage(rhi::PipelineStages stage) noexcept -> ConsumeEntry&;
    auto setLayout(rhi::TextureLayout _layout) noexcept -> ConsumeEntry&;
    auto enableDepthWrite(bool set) noexcept -> ConsumeEntry&;
    auto setDepthCompareFn(rhi::CompareFunction fn) noexcept -> ConsumeEntry&;
    auto setSubresource(uint32_t mip_beg, uint32_t mip_end,
        uint32_t level_beg, uint32_t level_end) noexcept -> ConsumeEntry&;
    auto setAttachmentLoc(uint32_t loc) noexcept -> ConsumeEntry&;
    auto setAccess(uint32_t acc) noexcept -> ConsumeEntry&;
    auto setBlendOperation(rhi::BlendOperation operation) noexcept -> ConsumeEntry&;
    auto setSourceBlenderFactor(rhi::BlendFactor factor) noexcept -> ConsumeEntry&;
    auto setTargetBlenderFactor(rhi::BlendFactor factor) noexcept -> ConsumeEntry&;
  };

  auto consume(ConsumeEntry const& entry) noexcept -> TextureInfo&;
  auto setInfo(TextureInfo const& x) noexcept -> TextureInfo&;
  auto withSize(se::ivec3 absolute) noexcept -> TextureInfo&;
  auto withSize(se::vec3 relative) noexcept -> TextureInfo&;
  auto withSizeRelative(std::string const& src, se::vec3 relative = {1.}) noexcept -> TextureInfo&;
  auto withLevels(uint32_t levels) noexcept -> TextureInfo&;
  auto withLayers(uint32_t layers) noexcept -> TextureInfo&;
  auto withSamples(uint32_t samples) noexcept -> TextureInfo&;
  auto withFormat(rhi::TextureFormat format) noexcept -> TextureInfo&;
  auto withStages(rhi::ShaderStages flags) noexcept -> TextureInfo&;
  auto withUsages(rhi::TextureUsages usages) noexcept -> TextureInfo&;
  auto withFlags(ResourceFlags flags) noexcept -> TextureInfo&;
  auto getSize(se::ivec3 ref) const noexcept -> rhi::Extend3D;

  std::vector<ConsumeEntry> consumeHistories;
};

/** Information that describe a resource in RDG */
struct SIByL_API ResourceInfo {
  ResourceInfo() : type(Type::UNKNOWN){};

  enum struct Type { UNKNOWN, Buffer, Texture } type = Type::UNKNOWN;
  struct Info {
    void* null;
    BufferInfo buffer;
    TextureInfo texture;
  } info;

  size_t devirtualizeID = (-1);
  /* declare the resource is a buffer */
  auto isBuffer() noexcept -> BufferInfo&;
  /* declare the resource is a texture */
  auto isTexture() noexcept -> TextureInfo&;

  ResourceInfo* prev = nullptr;
};

struct SIByL_API Resource { RUID ruid; };

struct SIByL_API BufferResource : public Resource {
  rhi::BufferDescriptor desc;
  gfx::BufferHandle buffer;
  struct ConsumeHistory {
    size_t passID;
    std::vector<BufferInfo::ConsumeEntry> entries; };
  std::vector<ConsumeHistory> cosumeHistories;
};

struct SIByL_API TextureResource : public Resource {
  rhi::TextureDescriptor desc;
  gfx::TextureHandle texture;
  std::string name;
  struct ConsumeHistory {
    size_t passID;
    std::vector<TextureInfo::ConsumeEntry> entries; };
  std::vector<ConsumeHistory> cosumeHistories;
};


struct SIByL_API PassReflection {
  PassReflection();

  auto addInput(std::string const& name) noexcept -> ResourceInfo&;
  auto addOutput(std::string const& name) noexcept -> ResourceInfo&;
  auto addInputOutput(std::string const& name) noexcept -> ResourceInfo&;
  auto addInternal(std::string const& name) noexcept -> ResourceInfo&;

  auto getDepthStencilState() noexcept -> rhi::DepthStencilState;
  auto getColorTargetState() noexcept -> std::vector<rhi::ColorTargetState>;
  auto getResourceInfo(std::string const& name) noexcept -> ResourceInfo*;

  std::unordered_map<std::string, ResourceInfo> inputResources = {};
  std::unordered_map<std::string, ResourceInfo> outputResources = {};
  std::unordered_map<std::string, ResourceInfo> inputOutputResources = {};
  std::unordered_map<std::string, ResourceInfo> internalResources = {};
};

auto SIByL_API toBufferDescriptor(BufferInfo const& info) noexcept -> rhi::BufferDescriptor;
auto SIByL_API toTextureDescriptor(TextureInfo const& info, se::ivec3 ref_size) noexcept -> rhi::TextureDescriptor;

struct SIByL_API RenderContext {
  size_t flightIdx = 0;
  rhi::CommandEncoder* cmdEncoder;
};

struct Graph;
struct Pass;
struct PipelinePass;

struct SIByL_API RenderData {
  struct DelegateData {
    rhi::CommandEncoder* cmdEncoder;
    union {
      rhi::RenderPassEncoder* render;
      rhi::ComputePassEncoder* compute;
      rhi::RayTracingPassEncoder* trace;
    } passEncoder;
    PipelinePass* pipelinePass;
    void* customData;
  };

  auto setDelegate(
    std::string const& name,
    std::function<void(DelegateData const&)> const& fn
  ) noexcept -> void;
  auto getDelegate(
    std::string const& name
  ) const noexcept -> std::function<void(DelegateData const&)> const&;
  auto setBindingResource(
    std::string const& name, 
    rhi::BindingResource const& bindGroupEntry
  ) noexcept -> void;
  auto getBindingResource(std::string const& name) const noexcept
    -> std::optional<rhi::BindingResource>;
  auto setBindGroupEntries(
    std::string const& name,
    std::vector<rhi::BindGroupEntry>* bindGroup) noexcept -> void;
  auto getBindGroupEntries(std::string const& name) const noexcept
    -> std::vector<rhi::BindGroupEntry>*;
  auto getTexture(std::string const& name) const noexcept -> gfx::TextureHandle;
  auto getBuffer(std::string const& name) const noexcept -> gfx::BufferHandle;

  auto setUVec2(std::string const& name, se::uvec2 v) noexcept -> void;
  auto getUVec2(std::string const& name) const noexcept -> se::uvec2;
  auto setUInt(std::string const& name, uint32_t v) noexcept -> void;
  auto getUInt(std::string const& name) const noexcept -> uint32_t;
  auto setPtr(std::string const& name, void* v) noexcept -> void;
  auto getPtr(std::string const& name) const noexcept -> void*;
  auto setMat4(std::string const& name, se::mat4 m) noexcept -> void;
  auto getMat4(std::string const& name) const noexcept -> se::mat4;

  auto getScene() const noexcept -> gfx::SceneHandle;

  Graph* graph;
  Pass* pass;
  gfx::SceneHandle scene = {};
  std::unordered_map<std::string, std::vector<rhi::BindGroupEntry>*> bindGroups;
  std::unordered_map<std::string, rhi::BindingResource> bindingResources;
  std::unordered_map<std::string, se::uvec2> uvec2s;
  std::unordered_map<std::string, uint32_t> uints;
  std::unordered_map<std::string, std::function<void(DelegateData const&)>> delegates;
  std::unordered_map<std::string, void*> ptrs;
  std::unordered_map<std::string, se::mat4> mat4s;
};

struct SIByL_API Pass {
  Pass() = default;
  Pass(Pass&&) = default;
  Pass(Pass const&) = delete;
  virtual ~Pass() = default;
  auto operator=(Pass&&)->Pass & = default;
  auto operator=(Pass const&)->Pass & = delete;

  virtual auto reflect() noexcept -> PassReflection = 0;
  virtual auto execute(RenderContext* context, RenderData const& renderData) noexcept -> void = 0;
  virtual auto readback(RenderData const& renderData) noexcept -> void {}
  virtual auto renderUI() noexcept -> void {}
  virtual auto onInteraction(se::input* input, editor::Widget::WidgetInfo* info) noexcept -> void {}
  virtual auto generateMarker() noexcept -> void;
  auto pass() noexcept -> Pass* { return this; }

  PassReflection pReflection;
  rhi::DebugUtilLabelDescriptor marker;
  std::string identifier;
  std::vector<size_t> subgraphStack;

 protected:
  virtual auto init() noexcept -> void;
};

struct SIByL_API DummyPass : public Pass {
  virtual auto reflect() noexcept -> PassReflection override = 0;
  virtual auto execute(RenderContext* context,
      RenderData const& renderData) noexcept -> void override {}
};

struct SIByL_API AliasDict {
  struct Value {
    std::string pass;
    std::string resource; };
  std::unordered_map<std::string, Value> dict;

  auto addAlias(std::string const& subgraph_resource,
    std::string const& pass,
    std::string const& pass_resource) noexcept
    -> AliasDict&;
};

struct SIByL_API Subgraph {
  virtual auto onRegister(Graph* graph) noexcept -> void = 0;
  virtual auto alias() noexcept -> AliasDict = 0;
  auto CONCAT(std::string const& name) noexcept -> std::string;
  virtual auto generateMarker() noexcept -> void;
  std::string identifier;
  rhi::DebugUtilLabelDescriptor marker;
};

struct SIByL_API PipelinePass : public Pass {
  PipelinePass() = default;
  virtual ~PipelinePass() = default;
  gfx::ShaderReflection reflection;
  std::vector<std::unique_ptr<rhi::BindGroupLayout>> bindgroupLayouts;
  std::vector<std::array<std::unique_ptr<rhi::BindGroup>, MULTIFRAME_FLIGHTS_COUNT>> bindgroups;
  std::unique_ptr<rhi::PipelineLayout> pipelineLayout;

  auto getBindGroup(RenderContext* context, uint32_t size) noexcept -> rhi::BindGroup*;
  // Direct update binding with binding location find by string
  auto updateBinding(RenderContext* context, std::string const& name,
                     rhi::BindingResource const& resource) noexcept -> void;
  auto updateBindings(
    RenderContext* context,
    std::vector<std::pair<std::string, rhi::BindingResource>> const& bindings) noexcept -> void;
 protected:
  virtual auto init(std::vector<gfx::ShaderModule*> shaderModules) noexcept -> void;
};

struct SIByL_API RenderPass : public PipelinePass {
  RenderPass() = default;
  virtual ~RenderPass() = default;
  rhi::RenderPassDescriptor renderPassDescriptor;
  std::array<std::unique_ptr<rhi::RenderPipeline>, MULTIFRAME_FLIGHTS_COUNT> pipelines;
  std::array<std::unique_ptr<rhi::RenderPassEncoder>, MULTIFRAME_FLIGHTS_COUNT> passEncoders;
  auto beginPass(RenderContext* context, gfx::Texture* target) noexcept -> rhi::RenderPassEncoder*;
  auto beginPass(RenderContext* context, uint32_t width, uint32_t height) noexcept -> rhi::RenderPassEncoder*;
  auto prepareDelegateData(RenderContext* context, RenderData const& renderData) noexcept -> RenderData::DelegateData;
  virtual auto generateMarker() noexcept -> void;
  virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment) noexcept -> void;
  virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment, gfx::ShaderModule* geometry) noexcept -> void;
  // utils functions
  auto setRenderPassDescriptor(rhi::RenderPassDescriptor const& renderPassDescriptor) noexcept -> void;
  auto issueDirectDrawcalls(rhi::RenderPassEncoder* encoder, gfx::SceneHandle scene) noexcept -> void;
  virtual auto beforeDirectDrawcall(rhi::RenderPassEncoder* encoder, int geometry_idx,
    gfx::Scene::GeometryDrawData const& data) noexcept -> void {}
 protected:
  auto prepareDispatch(RenderContext* context, gfx::Texture* target) noexcept -> void;
  auto prepareDispatch(rdg::RenderContext* context, uint32_t width, uint32_t height) noexcept -> void;

  using RenderPipelineDescCallback = std::function<void(rhi::RenderPipelineDescriptor&)>;
  virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void;
  virtual auto init(gfx::ShaderModule* vertex, gfx::ShaderModule* geometry,
    gfx::ShaderModule* fragment, std::optional<RenderPipelineDescCallback> callback = std::nullopt) noexcept -> void;
};

struct SIByL_API FullScreenPass : public RenderPass {
  FullScreenPass() = default;
  virtual ~FullScreenPass() = default;
  auto dispatchFullScreen(rdg::RenderContext* context) noexcept -> void;
  virtual auto init(gfx::ShaderModule* fragment) noexcept -> void;
  virtual auto generateMarker() noexcept -> void;
};

struct SIByL_API ComputePass : public PipelinePass {
  ComputePass() = default;
  virtual ~ComputePass() = default;
  std::array<std::unique_ptr<rhi::ComputePipeline>, MULTIFRAME_FLIGHTS_COUNT> pipelines;
  std::array<std::unique_ptr<rhi::ComputePassEncoder>, MULTIFRAME_FLIGHTS_COUNT> passEncoders;

  auto beginPass(rdg::RenderContext* context) noexcept -> rhi::ComputePassEncoder*;
  auto prepareDispatch(rdg::RenderContext* context) noexcept -> void;
  virtual auto generateMarker() noexcept -> void;
  virtual auto init(gfx::ShaderModule* comp) noexcept -> void;
};

struct SIByL_API RayTracingPass : public PipelinePass {
  RayTracingPass() = default;
  virtual ~RayTracingPass() = default;
  rhi::RayTracingPipelineDescriptor rtPipelineDescriptor;
  std::array<std::unique_ptr<rhi::RayTracingPipeline>, MULTIFRAME_FLIGHTS_COUNT> pipelines;
  std::array<std::unique_ptr<rhi::RayTracingPassEncoder>, MULTIFRAME_FLIGHTS_COUNT> passEncoders;

  auto beginPass(rdg::RenderContext* context) noexcept -> rhi::RayTracingPassEncoder*;
  inline auto prepareDispatch(rdg::RenderContext* context) noexcept -> void;
  virtual auto generateMarker() noexcept -> void;
  //virtual auto init(gfx::SBTsDescriptor const& sbt, uint32_t max_depth) noexcept -> void;
};

struct SIByL_API DAG {
  auto addEdge(uint32_t src, uint32_t dst) noexcept -> void;
  auto reverse() const noexcept -> DAG;
  std::unordered_map<uint32_t, std::set<uint32_t>> adj;
};

auto SIByL_API flatten_bfs(DAG const& g, size_t output) noexcept
-> std::optional<std::vector<size_t>>;

struct SIByL_API Graph {
  /** virtual destructor */
  virtual ~Graph() = default;
  /** virtual constructor */
  Graph() = default;
  Graph(Graph&&) = default;
  Graph(Graph const&) = delete;
  auto operator=(Graph&&) ->Graph & = default;
  auto operator=(Graph const&) ->Graph & = delete;

  /** build the graph */
  auto build() noexcept -> bool;
  /** execute the graph */
  auto execute(rhi::CommandEncoder* encoder) noexcept -> void;
  /** read back the information */
  virtual auto readback() noexcept -> void;
  /** render graph ui */
  virtual auto renderUI() noexcept -> void {}

  /** Add a RDPass to the RDGraph
   * @param pass		: the pass to add
   * @param identifier	: identifier of the pass added */
  auto addPass(std::unique_ptr<Pass>&& pass,
               std::string const& identifier) noexcept -> void;
  auto addPass(Pass* pass, std::string const& identifier) noexcept -> void;

  /** Add a Subgraph to the RDGraph
   * @param subgraph	: the subgraph to add
   * @param identifier	: identifier of the pass added */
  auto addSubgraph(std::unique_ptr<Subgraph>&& subgraph,
                   std::string const& identifier) noexcept -> void;

  /** Adding dependency from src_pass.src_resource to dst_pass.dst_resource.
   * @param src_pass: identifier of source pass
   * @param src_pass: identifier of the source output of source pass
   * @param dst_pass: identifier of destination pass
   * @param dst_pass: identifier of the destination input of destination pass */
  auto addEdge(std::string const& src_pass, std::string const& src_resource,
               std::string const& dst_pass,
               std::string const& dst_resource) noexcept -> void;
  auto addEdge(std::string const& src_pass,
               std::string const& dst_pass) noexcept -> void;

  /** Set external resource
   * @param pass		: the pass where the resource is in
   * @param resource	: the resource to set */
  auto setExternal(std::string const& pass, std::string const& resource,
                   gfx::TextureHandle tex) noexcept -> void;
  auto setExternal(std::string const& pass, std::string const& resource,
                   gfx::BufferHandle tex) noexcept -> void;

  /** Mark the main output of the render graph.
   * @param pass		: the pass produce the output
   * @param identifier	: identifier of the output resource */
  auto markOutput(std::string const& pass, std::string const& output) noexcept -> void;

  auto getOutput() noexcept -> gfx::TextureHandle;

  auto getTextureResource(std::string const& pass,
      std::string const& output) noexcept
      -> gfx::TextureHandle;
  
  auto getBufferResource(std::string const& pass,
      std::string const& output) noexcept
      -> gfx::BufferHandle;


  auto getFlattenedPasses() noexcept -> std::vector<size_t> const& { return flattenedPasses; }
  auto getPass(size_t i) noexcept -> Pass* { return passes[i]; }
  auto getPass(std::string const& name) noexcept -> Pass*;
  auto bindScene(gfx::SceneHandle scene) noexcept -> void;

  RenderData renderData;
  DAG dag;
  size_t passID = 0;
  size_t subgraphID = 0;
  std::unordered_map<std::string, size_t> passNameList;
  std::unordered_map<size_t, Pass*> passes;
  std::unordered_map<size_t, std::string> subgraphNameList;
  std::unordered_map<std::string, std::unique_ptr<Subgraph>> subgraphs;
  std::unordered_map<std::string, AliasDict> subgraphsAlias;
  std::unordered_map<size_t, std::vector<rhi::BarrierDescriptor>> barriers;
  std::vector<size_t> flattenedPasses;
  std::vector<size_t> subgraphStack;
  // contrain the pass objects if necessary
  // only useful for C++ case
  std::unordered_map<size_t, std::unique_ptr<Pass>> passesContainer;

  std::string output_pass;
  std::string output_resource;

  se::ivec3 standardSize = {1280, 720, 1};

  size_t resourceID = 0;
  std::unordered_map<size_t, TextureResource> textureResources;
  std::unordered_map<size_t, BufferResource> bufferResources;

  auto generateBufferBarriers() noexcept -> void;
  auto generateTextureBarriers() noexcept -> void;

  auto decodeAlias(std::string& pass, std::string& resource) noexcept -> void;
};

struct SIByL_API Pipeline {
  /** default ctor and dctor. */
  Pipeline() = default;
  virtual ~Pipeline() = default;
  /** build the render pipeline. */
  virtual auto build() noexcept -> void = 0;
  /** execute the pipeline on the command encoder. */
  virtual auto execute(rhi::CommandEncoder* encoder) noexcept -> void = 0;
  /** readback after the execution of the pipeline */
  virtual auto readback() noexcept -> void {}
  /** render pipeline ui. */
  virtual auto renderUI() noexcept -> void {}
  /** get all active graphs now within the pipeline. */
  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*> = 0;
  /** get all active graphs now within the pipeline. */
  virtual auto getAllGraphs() noexcept -> std::vector<Graph*> = 0;
  /** get the output of the pipeline. */
  virtual auto getOutput() noexcept -> gfx::TextureHandle = 0;
  /** set the standard size for all the graphs. */
  auto setStandardSize(se::ivec3 size) noexcept -> void;
  /** bind a scene to the pipeline.  */
  auto bindScene(gfx::SceneHandle scene) noexcept -> void;
  /** get the pointer of the pipeline. */
  auto pipeline() noexcept -> Pipeline* { return this; }
};

struct SIByL_API SingleGraphPipeline : public Pipeline {
  SingleGraphPipeline() = default;
  virtual ~SingleGraphPipeline() = default;

  virtual auto execute(rhi::CommandEncoder* encoder) noexcept -> void { pGraph->execute(encoder); }

  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*> { return {pGraph}; }

  virtual auto getAllGraphs() noexcept -> std::vector<Graph*> { return { pGraph }; }

  virtual auto getOutput() noexcept -> gfx::TextureHandle { return pGraph->getOutput(); }

  virtual auto readback() noexcept -> void { pGraph->readback(); }

  virtual auto build() noexcept -> void override { pGraph->build(); }

  auto setGraph(Graph* graph) noexcept -> void { pGraph = graph; }

  Graph* pGraph;
};

}