#define DLIB_EXPORT
#include <se.rdg.hpp>
#undef DLIB_EXPORT
#include <queue>
#include <stack>
#undef MemoryBarrier

namespace se::rdg {
auto BufferInfo::ConsumeEntry::addStage(rhi::PipelineStages stage) noexcept -> ConsumeEntry& {
  stages |= stage;
  return *this;
}

auto BufferInfo::ConsumeEntry::setAccess(uint32_t acc) noexcept -> ConsumeEntry& {
  access = acc;
  return *this;
}

auto BufferInfo::ConsumeEntry::setSubresource(uint64_t offset, uint64_t size
) noexcept -> BufferInfo::ConsumeEntry& {
  this->offset = offset;
  this->size = size;
  return *this;
}

auto BufferInfo::withSize(uint32_t _size) noexcept -> BufferInfo& {
  size = _size;
  return *this;
}

auto BufferInfo::withUsages(rhi::BufferUsages _usages) noexcept -> BufferInfo& {
  usages = _usages;
  return *this;
}

auto BufferInfo::withFlags(ResourceFlags _flags) noexcept -> BufferInfo& {
  flags = _flags;
  return *this;
}

auto BufferInfo::consume(ConsumeEntry const& entry) noexcept -> BufferInfo& {
  consumeHistories.emplace_back(entry);
  return *this;
}

auto TextureInfo::ConsumeEntry::addStage(rhi::PipelineStages stage) noexcept -> ConsumeEntry& {
    stages |= stage;
    return *this; }

auto TextureInfo::ConsumeEntry::setLayout(rhi::TextureLayout _layout) noexcept -> ConsumeEntry& {
    layout = _layout;
    return *this; }

auto TextureInfo::ConsumeEntry::enableDepthWrite(bool set) noexcept -> ConsumeEntry& {
    depthWrite = set;
    return *this; }

auto TextureInfo::ConsumeEntry::setDepthCompareFn(rhi::CompareFunction fn) noexcept
-> ConsumeEntry& {
    depthCmp = fn;
    return *this; }

auto TextureInfo::ConsumeEntry::setSubresource(uint32_t mip_beg, uint32_t mip_end,
  uint32_t level_beg, uint32_t level_end) noexcept
  -> ConsumeEntry& {
  this->mip_beg = mip_beg;
  this->mip_end = mip_end;
  this->level_beg = level_beg;
  this->level_end = level_end;
  return *this;
}

auto TextureInfo::ConsumeEntry::setAttachmentLoc(uint32_t loc) noexcept -> ConsumeEntry& {
  attachLoc = loc;
  return *this;
}

auto TextureInfo::ConsumeEntry::setAccess(uint32_t acc) noexcept -> ConsumeEntry& {
  access = acc;
  return *this;
}

auto TextureInfo::ConsumeEntry::setBlendOperation(rhi::BlendOperation operation) noexcept -> ConsumeEntry& {
  bldOperation = operation;
  return *this;
}

auto TextureInfo::ConsumeEntry::setSourceBlenderFactor(rhi::BlendFactor factor) noexcept -> ConsumeEntry& {
  srcFactor = factor;
  return *this;
}

auto TextureInfo::ConsumeEntry::setTargetBlenderFactor(rhi::BlendFactor factor) noexcept -> ConsumeEntry& {
  dstFactor = factor;
  return *this;
}

auto TextureInfo::consume(ConsumeEntry const& _entry) noexcept -> TextureInfo& {
  ConsumeEntry entry = _entry;
  if (entry.type == ConsumeType::ColorAttachment) {
    entry.access |= uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT) |
                    uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT);
    entry.stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT);
    entry.layout = rhi::TextureLayout::COLOR_ATTACHMENT_OPTIMAL;
  } else if (entry.type == ConsumeType::DepthStencilAttachment) {
    entry.access |=
        uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
        uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    entry.stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT) |
                    uint32_t(rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT) |
                    uint32_t(rhi::PipelineStageBit::LATE_FRAGMENT_TESTS_BIT);
    entry.layout = rhi::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL;
  } else if (entry.type == ConsumeType::TextureBinding) {
    entry.access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT);
    entry.layout = rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
  } else if (entry.type == ConsumeType::StorageBinding) {
    entry.access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT) |
                    uint32_t(rhi::AccessFlagBits::SHADER_WRITE_BIT);
    entry.layout = rhi::TextureLayout::GENERAL;
  }
  consumeHistories.emplace_back(entry);
  return *this;
}

auto TextureInfo::setInfo(TextureInfo const& x) noexcept
    -> TextureInfo& {
  sizeDef = x.sizeDef;
  if (sizeDef == SizeDefine::Absolute)
    size.absolute = x.size.absolute;
  else if (sizeDef == SizeDefine::Relative)
    size.relative = x.size.relative;
  format = x.format;
  levels = x.levels;
  return *this;
}

auto TextureInfo::withSize(se::ivec3 absolute) noexcept
    -> TextureInfo& {
  sizeDef = SizeDefine::Absolute;
  size.absolute = absolute;
  return *this;
}

auto TextureInfo::withSize(se::vec3 relative) noexcept -> TextureInfo& {
  sizeDef = SizeDefine::Relative;
  size.relative = relative;
  return *this;
}

auto TextureInfo::withSizeRelative(std::string const& src, se::vec3 relative) noexcept -> TextureInfo& {
  sizeDef = SizeDefine::RelativeToAnotherTex;
  sizeRefName = src;
  size.relative = relative;
  return *this;
}

auto TextureInfo::withLevels(uint32_t _levels) noexcept -> TextureInfo& {
  levels = _levels;
  return *this;
}

auto TextureInfo::withLayers(uint32_t _layers) noexcept -> TextureInfo& {
  layers = _layers;
  return *this;
}

auto TextureInfo::withSamples(uint32_t _samples) noexcept
    -> TextureInfo& {
  samples = _samples;
  return *this;
}

auto TextureInfo::withFormat(rhi::TextureFormat _format) noexcept -> TextureInfo& {
  format = _format;
  return *this;
}

auto TextureInfo::withStages(rhi::ShaderStages flags) noexcept -> TextureInfo& {
  sflags = flags;
  return *this;
}

auto TextureInfo::withUsages(rhi::TextureUsages _usages) noexcept -> TextureInfo& {
  usages = _usages;
  if (usages | uint32_t(rhi::TextureUsageBit::COLOR_ATTACHMENT)) {
    stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT) | sflags;
    access |= uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT) |
              uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT);
    laytout = rhi::TextureLayout::COLOR_ATTACHMENT_OPTIMAL;
  } else if (usages | uint32_t(rhi::TextureUsageBit::DEPTH_ATTACHMENT)) {
    stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT) |
              uint32_t(rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT) |
              uint32_t(rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT) | sflags;
    access |= uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
              uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    laytout = rhi::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL;
  } else if (usages | uint32_t(rhi::TextureUsageBit::TEXTURE_BINDING)) {
    stages |= sflags;
    access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT);
    laytout = rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
  } else if (usages | uint32_t(rhi::TextureUsageBit::STORAGE_BINDING)) {
    stages |= sflags;
    access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT) |
              uint32_t(rhi::AccessFlagBits::SHADER_WRITE_BIT);
    laytout = rhi::TextureLayout::GENERAL;
  }
  return *this;
}

auto TextureInfo::withFlags(ResourceFlags _flags) noexcept -> TextureInfo& {
  flags = _flags;
  return *this;
}

auto TextureInfo::getSize(se::ivec3 ref) const noexcept -> rhi::Extend3D {
  if (sizeDef == SizeDefine::Absolute)
    return rhi::Extend3D{uint32_t(size.absolute.x), uint32_t(size.absolute.y),
                         uint32_t(size.absolute.z)};
  else
    return rhi::Extend3D{uint32_t(std::ceil(size.relative.x * ref.x)),
                         uint32_t(std::ceil(size.relative.y * ref.y)),
                         uint32_t(std::ceil(size.relative.z * ref.z))};
}

auto ResourceInfo::isBuffer() noexcept -> BufferInfo& {
  type = Type::Buffer;
  info.buffer = BufferInfo{};
  return info.buffer;
}

auto ResourceInfo::isTexture() noexcept -> TextureInfo& {
  type = Type::Texture;
  info.texture = TextureInfo{};
  return info.texture;
}

PassReflection::PassReflection() {
  inputResources.clear();
  outputResources.clear();
  inputOutputResources.clear();
  internalResources.clear();
}

auto PassReflection::addInput(std::string const& name) noexcept
    -> ResourceInfo& {
  return inputResources[name];
}
auto PassReflection::addOutput(std::string const& name) noexcept
    -> ResourceInfo& {
  return outputResources[name];
}
auto PassReflection::addInputOutput(std::string const& name) noexcept
    -> ResourceInfo& {
  return inputOutputResources[name];
}
auto PassReflection::addInternal(std::string const& name) noexcept
    -> ResourceInfo& {
  return internalResources[name];
}
auto PassReflection::getDepthStencilState() noexcept -> rhi::DepthStencilState {
  std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &inputResources, &outputResources, &inputOutputResources,
          &internalResources};
  for (auto& pool : pools) {
    for (auto& pair : *pool) {
      if (pair.second.type != ResourceInfo::Type::Texture) continue;
      if (rhi::hasDepthBit(pair.second.info.texture.format)) {
        TextureInfo::ConsumeEntry entry;
        for (auto const& ch : pair.second.info.texture.consumeHistories) {
          if (ch.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
            entry = ch;
            break;
          }
        }
        return rhi::DepthStencilState{pair.second.info.texture.format,
                                      entry.depthWrite, entry.depthCmp};
      }
    }
  }
  return rhi::DepthStencilState{};
}
auto PassReflection::getColorTargetState() noexcept
    -> std::vector<rhi::ColorTargetState> {
  std::vector<rhi::ColorTargetState> cts;
  std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &inputResources, &outputResources, &inputOutputResources,
          &internalResources};
  for (auto& pool : pools) {
    for (auto& pair : *pool) {
      if (pair.second.type != ResourceInfo::Type::Texture) continue;
      if (!rhi::hasDepthBit(pair.second.info.texture.format)) {
        for (auto const& history : pair.second.info.texture.consumeHistories) {
          if (history.type != TextureInfo::ConsumeType::ColorAttachment)
            continue;
          cts.resize(history.attachLoc + 1);
          cts[history.attachLoc] = rhi::ColorTargetState{
            pair.second.info.texture.format,
            rhi::BlendState{history.bldOperation, history.srcFactor,history.dstFactor}
          };
        }
      }
    }
  }
  return cts;
}

auto PassReflection::getResourceInfo(std::string const& name) noexcept -> ResourceInfo* {
  if (inputResources.find(name) != inputResources.end()) {
    return &inputResources[name];
  } else if (outputResources.find(name) != outputResources.end()) {
    return &outputResources[name];
  } else if (inputOutputResources.find(name) != inputOutputResources.end()) {
    return &inputOutputResources[name];
  } else if (internalResources.find(name) != internalResources.end()) {
    return &internalResources[name];
  }
  root::print::error(std::format(
    "RDG::PassReflection::getResourceInfo Failed to find resource \"{0}\"", name));
  return nullptr;
}

auto toBufferDescriptor(BufferInfo const& info) noexcept -> rhi::BufferDescriptor {
  return rhi::BufferDescriptor{ info.size, info.usages };
}

auto toTextureDescriptor(TextureInfo const& info, se::ivec3 ref_size) noexcept -> rhi::TextureDescriptor {
  rhi::Extend3D size = info.getSize(ref_size);
  return rhi::TextureDescriptor{
    size,
    info.levels,
    info.layers,
    info.samples,
    size.depthOrArrayLayers == 1 ? rhi::TextureDimension::TEX2D
                                 : rhi::TextureDimension::TEX3D,
    info.format,
    info.usages | (uint32_t)rhi::TextureUsageBit::TEXTURE_BINDING |
        (uint32_t)rhi::TextureUsageBit::COPY_SRC,
    std::vector<rhi::TextureFormat>{info.format},
    info.tflags};
}

auto RenderData::setDelegate(
    std::string const& name,
    std::function<void(DelegateData const&)> const& fn) noexcept -> void {
    delegates[name] = fn;
}

auto RenderData::getDelegate(std::string const& name) const noexcept
-> std::function<void(DelegateData const&)> const& {
    auto const& iter = delegates.find(name);
    if (iter == delegates.end()) return nullptr;
    return iter->second;
}

auto RenderData::setBindingResource(
  std::string const& name,
  rhi::BindingResource const& bindGroupEntry) noexcept
  -> void {
  bindingResources[name] = bindGroupEntry;
}

auto RenderData::getBindingResource(std::string const& name) const noexcept
-> std::optional<rhi::BindingResource> {
    auto const& iter = bindingResources.find(name);
    if (iter == bindingResources.end()) return std::nullopt;
    return iter->second;
}

auto RenderData::setBindGroupEntries(
    std::string const& name,
    std::vector<rhi::BindGroupEntry>* bindGroup) noexcept -> void {
    bindGroups[name] = bindGroup;
}

auto RenderData::getBindGroupEntries(std::string const& name) const noexcept
-> std::vector<rhi::BindGroupEntry>* {
    auto const& iter = bindGroups.find(name);
    if (iter == bindGroups.end()) return nullptr;
    return iter->second;
}

auto RenderData::getTexture(std::string const& name) const noexcept -> gfx::TextureHandle {
  if (pass->pReflection.inputResources.find(name) !=
      pass->pReflection.inputResources.end()) {
      return graph->textureResources[pass->pReflection.inputResources[name]
          .devirtualizeID].texture;
  }
  if (pass->pReflection.outputResources.find(name) !=
      pass->pReflection.outputResources.end()) {
      return graph
          ->textureResources[pass->pReflection.outputResources[name]
          .devirtualizeID].texture;
  }
  if (pass->pReflection.inputOutputResources.find(name) !=
      pass->pReflection.inputOutputResources.end()) {
      return graph
          ->textureResources[pass->pReflection.inputOutputResources[name]
          .devirtualizeID]
          .texture;
  }
  if (pass->pReflection.internalResources.find(name) !=
      pass->pReflection.internalResources.end()) {
      return graph
          ->textureResources[pass->pReflection.internalResources[name]
          .devirtualizeID]
          .texture;
  }
  return gfx::TextureHandle{};
}

auto RenderData::getBuffer(std::string const& name) const noexcept -> gfx::BufferHandle {
  if (pass->pReflection.inputResources.find(name) !=
      pass->pReflection.inputResources.end()) {
      return graph
          ->bufferResources[pass->pReflection.inputResources[name].devirtualizeID]
          .buffer;
  }
  if (pass->pReflection.outputResources.find(name) !=
      pass->pReflection.outputResources.end()) {
      return graph
          ->bufferResources[pass->pReflection.outputResources[name]
          .devirtualizeID]
          .buffer;
  }
  if (pass->pReflection.inputOutputResources.find(name) !=
      pass->pReflection.inputOutputResources.end()) {
      return graph
          ->bufferResources[pass->pReflection.inputOutputResources[name]
          .devirtualizeID]
          .buffer;
  }
  if (pass->pReflection.internalResources.find(name) !=
      pass->pReflection.internalResources.end()) {
      return graph
          ->bufferResources[pass->pReflection.internalResources[name]
          .devirtualizeID]
          .buffer;
  }
}

auto RenderData::setUVec2(std::string const& name, se::uvec2 v) noexcept -> void {
  uvec2s[name] = v;
}

auto RenderData::getUVec2(std::string const& name) const noexcept -> se::uvec2 {
  auto const& iter = uvec2s.find(name);
  if (iter == uvec2s.end()) return se::uvec2{ 0 };
  return iter->second;
}

auto RenderData::setUInt(std::string const& name, uint32_t v) noexcept -> void {
  uints[name] = v;
}

auto RenderData::getUInt(std::string const& name) const noexcept -> uint32_t {
  auto const& iter = uints.find(name);
  if (iter == uints.end()) return 0;
  return iter->second;
}

auto RenderData::setPtr(std::string const& name, void* v) noexcept -> void {
  ptrs[name] = v;
}

auto RenderData::getPtr(std::string const& name) const noexcept -> void* {
    auto const& iter = ptrs.find(name);
    if (iter == ptrs.end()) return 0;
    return iter->second;
}

auto RenderData::setMat4(std::string const& name, se::mat4 m) noexcept -> void {
  mat4s[name] = m;
}

auto RenderData::getMat4(std::string const& name) const noexcept -> se::mat4 {
  auto const& iter = mat4s.find(name);
  if (iter == mat4s.end()) return se::mat4();
  return iter->second;
}

auto RenderData::getScene() const noexcept -> gfx::SceneHandle { return scene; }

auto Pass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.490, 0.607, 0.003, 1. };
}

auto Pass::init() noexcept -> void { pReflection = this->reflect(); }

auto AliasDict::addAlias(std::string const& subgraph_resource,
    std::string const& pass,
    std::string const& pass_resource) noexcept
    -> AliasDict& {
    dict[subgraph_resource] = Value{ pass, pass_resource };
    return *this;
}

auto Subgraph::CONCAT(std::string const& name) noexcept -> std::string {
  return identifier + "." + name;
}

auto Subgraph::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.235, 0.321, 0.2, 1. };
}

auto PipelinePass::getBindGroup(RenderContext* context, uint32_t size) noexcept -> rhi::BindGroup* {
  return bindgroups[size][context->flightIdx].get();
}

auto PipelinePass::init(std::vector<gfx::ShaderModule*> shaderModules) noexcept  -> void {
  Pass::init();
  rhi::Device* device = gfx::GFXContext::device;
  // create pipeline reflection
  for (auto& sm : shaderModules) reflection = reflection + sm->reflection;
  // create bindgroup layouts
  bindgroupLayouts.resize(reflection.bindings.size());
  for (int i = 0; i < reflection.bindings.size(); ++i) {
    rhi::BindGroupLayoutDescriptor desc =
        gfx::ShaderReflection::toBindGroupLayoutDescriptor(reflection.bindings[i]);
    bindgroupLayouts[i] = device->createBindGroupLayout(desc);
  }
  // create bindgroups
  bindgroups.resize(reflection.bindings.size());
  for (size_t i = 0; i < reflection.bindings.size(); ++i) {
    for (size_t j = 0; j < MULTIFRAME_FLIGHTS_COUNT; ++j) {
      bindgroups[i][j] = device->createBindGroup(rhi::BindGroupDescriptor{
          bindgroupLayouts[i].get(),
          std::vector<rhi::BindGroupEntry>{},
      });
    }
  }
  // create pipelineLayout
  rhi::PipelineLayoutDescriptor desc = {};
  desc.pushConstants.resize(reflection.pushConstant.size());
  for (int i = 0; i < reflection.pushConstant.size(); ++i)
    desc.pushConstants[i] = rhi::PushConstantEntry{
      reflection.pushConstant[i].stages,
      reflection.pushConstant[i].offset,
      reflection.pushConstant[i].range,
    };
  desc.bindGroupLayouts.resize(bindgroupLayouts.size());
  for (int i = 0; i < bindgroupLayouts.size(); ++i)
    desc.bindGroupLayouts[i] = bindgroupLayouts[i].get();
  pipelineLayout = device->createPipelineLayout(desc);
}

auto PipelinePass::updateBinding(
    RenderContext* context,
    std::string const& name,
    rhi::BindingResource const& resource) noexcept -> void {
  auto iter = reflection.bindingInfo.find(name);
  if (iter == reflection.bindingInfo.end()) {
    root::print::error("RDG::Binding Name " + name + " not found");
  }
  std::vector<rhi::BindGroupEntry> set_entries = {
    rhi::BindGroupEntry{iter->second.binding, resource}};
  getBindGroup(context, iter->second.set)->updateBinding(set_entries);
}

auto PipelinePass::updateBindings(
  RenderContext* context,
  std::vector<std::pair<std::string, rhi::BindingResource>> const& bindings) noexcept -> void {
  for (auto& pair : bindings) updateBinding(context, pair.first, pair.second);
}

auto RenderPass::beginPass(RenderContext* context, gfx::Texture* target) noexcept -> rhi::RenderPassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginRenderPass(renderPassDescriptor);
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context, target);
  return passEncoders[context->flightIdx].get();
}

auto RenderPass::beginPass(RenderContext* context, uint32_t width, uint32_t height) noexcept -> rhi::RenderPassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginRenderPass(renderPassDescriptor);
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context, width, height);
  return passEncoders[context->flightIdx].get();
}

auto RenderPass::prepareDelegateData(RenderContext* context,
  RenderData const& renderData) noexcept
  -> RenderData::DelegateData {
  RenderData::DelegateData data;
  data.cmdEncoder = context->cmdEncoder;
  data.passEncoder.render = passEncoders[context->flightIdx].get();
  data.pipelinePass = this;
  return data;
}

auto RenderPass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.764, 0.807, 0.725, 1. };
}

auto RenderPass::prepareDispatch(RenderContext* context, gfx::Texture* target) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
        i, bindgroups[i][context->flightIdx].get());
  passEncoders[context->flightIdx]->setViewport(
      0, 0, target->texture->width(), target->texture->height(), 0, 1);
  passEncoders[context->flightIdx]->setScissorRect(
      0, 0, target->texture->width(), target->texture->height());
}

auto RenderPass::prepareDispatch(rdg::RenderContext* context, uint32_t width, uint32_t height) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
        i, bindgroups[i][context->flightIdx].get());
  passEncoders[context->flightIdx]->setViewport(0, 0, width, height, 0, 1);
  passEncoders[context->flightIdx]->setScissorRect(0, 0, width, height);
}

auto RenderPass::init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void {
  PipelinePass::init(std::vector<gfx::ShaderModule*>{vertex, fragment});
  rhi::Device* device = gfx::GFXContext::device;
  rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
      pipelineLayout.get(),
      rhi::VertexState{// vertex shader
                       vertex->shaderModule.get(),
                       "main",
                       // vertex attribute layout
                       {}},
      rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                          rhi::IndexFormat::UINT16_t},
      pReflection.getDepthStencilState(),
      rhi::MultisampleState{},
      rhi::FragmentState{// fragment shader
                         fragment->shaderModule.get(), "main",
                         pReflection.getColorTargetState()}};
  if (callback.has_value()) {
    callback.value()(pipelineDesc);
  }
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(pipelineDesc);
  }
}

auto RenderPass::init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment) noexcept -> void {
  PipelinePass::init(std::vector<gfx::ShaderModule*>{vertex, fragment});
  rhi::Device* device = gfx::GFXContext::device;
  rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
      pipelineLayout.get(),
      rhi::VertexState{// vertex shader
                       vertex->shaderModule.get(),
                       "main",
                       // vertex attribute layout
                       {}},
      rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                          rhi::IndexFormat::UINT16_t},
      pReflection.getDepthStencilState(),
      rhi::MultisampleState{},
      rhi::FragmentState{// fragment shader
                         fragment->shaderModule.get(), "main",
                         pReflection.getColorTargetState()}};
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(pipelineDesc);
  }
}

auto RenderPass::init(gfx::ShaderModule* vertex,
  gfx::ShaderModule* geometry,
  gfx::ShaderModule* fragment) noexcept -> void {
  PipelinePass::init(
    std::vector<gfx::ShaderModule*>{vertex, geometry, fragment});
  rhi::Device* device = gfx::GFXContext::device;
  rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
      pipelineLayout.get(),
      rhi::VertexState{// vertex shader
                       vertex->shaderModule.get(),
                       "main",
                       // vertex attribute layout
                       {}},
      rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                          rhi::IndexFormat::UINT16_t},
      pReflection.getDepthStencilState(),
      rhi::MultisampleState{},
      rhi::FragmentState{// fragment shader
                         fragment->shaderModule.get(), "main",
                         pReflection.getColorTargetState()} };
  pipelineDesc.geometry = { geometry->shaderModule.get() };
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(pipelineDesc);
  }
}

auto RenderPass::init(gfx::ShaderModule* vertex,
    gfx::ShaderModule* geometry,
    gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void {
  PipelinePass::init(
      std::vector<gfx::ShaderModule*>{vertex, geometry, fragment});
  rhi::Device* device = gfx::GFXContext::device;
  rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
      pipelineLayout.get(),
      rhi::VertexState{// vertex shader
                       vertex->shaderModule.get(),
                       "main",
                       // vertex attribute layout
                       {}},
      rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                          rhi::IndexFormat::UINT16_t},
      pReflection.getDepthStencilState(),
      rhi::MultisampleState{},
      rhi::FragmentState{// fragment shader
                         fragment->shaderModule.get(), "main",
                         pReflection.getColorTargetState()}};
  pipelineDesc.geometry = {geometry->shaderModule.get()};
  if (callback.has_value()) {
    callback.value()(pipelineDesc);
  }
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(pipelineDesc);
  }
}

auto RenderPass::setRenderPassDescriptor(rhi::RenderPassDescriptor const& input) noexcept -> void {
  renderPassDescriptor = input;
}

auto RenderPass::issueDirectDrawcalls(
  rhi::RenderPassEncoder* encoder, gfx::SceneHandle scene) noexcept -> void {
  encoder->setIndexBuffer(
    scene->getGPUScene()->index_buffer->buffer.get(), rhi::IndexFormat::UINT32_T, 0,
    scene->getGPUScene()->index_buffer->buffer->size());
  std::span<gfx::Scene::GeometryDrawData> geometries =
    scene->gpuScene.geometry_buffer->getHostAsStructuredArray<gfx::Scene::GeometryDrawData>();
  for (size_t geometry_idx = 0; geometry_idx < geometries.size(); ++geometry_idx) {
    auto& geometry = geometries[geometry_idx];
    beforeDirectDrawcall(encoder, geometry_idx, geometry);
    encoder->drawIndexed(
      geometry.indexSize, 1, geometry.indexOffset,
      geometry.vertexOffset, 0);
  }
}

auto FullScreenPass::dispatchFullScreen(rdg::RenderContext* context) noexcept -> void {
    passEncoders[context->flightIdx]->draw(3, 1, 0, 0);
}

auto FullScreenPass::init(gfx::ShaderModule* fragment) noexcept -> void {
  gfx::ShaderHandle fullscreen_vertex;
  auto [vert] = gfx::GFXContext::load_shader_slang(
    "../shaders/passes/postprocess/fullscreen.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 1>{
      std::make_pair("vertexMain", rhi::ShaderStageBit::VERTEX),
    });
  RenderPass::init(vert.get(), fragment);
}

auto FullScreenPass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.682, 0.764, 0.752, 1. };
}

auto ComputePass::beginPass(rdg::RenderContext* context) noexcept -> rhi::ComputePassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginComputePass(rhi::ComputePassDescriptor{});
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context);
  return passEncoders[context->flightIdx].get();
}

auto ComputePass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.6, 0.721, 0.780, 1. };
}

auto ComputePass::prepareDispatch(rdg::RenderContext* context) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
      i, bindgroups[i][context->flightIdx].get());
}

auto ComputePass::init(gfx::ShaderModule* comp) noexcept -> void {
  PipelinePass::init({comp});
  rhi::Device* device = gfx::GFXContext::device;
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createComputePipeline(rhi::ComputePipelineDescriptor{
        pipelineLayout.get(), {comp->shaderModule.get(), "main"}});
  }
}

auto RayTracingPass::beginPass(rdg::RenderContext* context) noexcept -> rhi::RayTracingPassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginRayTracingPass(rhi::RayTracingPassDescriptor{});
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context);
  return passEncoders[context->flightIdx].get();
}

auto RayTracingPass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.517, 0.674, 0.807, 1. };
}

auto RayTracingPass::prepareDispatch(rdg::RenderContext* context) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
      i, bindgroups[i][context->flightIdx].get());
}

auto DAG::addEdge(uint32_t src, uint32_t dst) noexcept -> void {
  adj[src].insert(dst);
  if (adj.find(dst) == adj.end()) adj[dst] = {};
}

auto DAG::reverse() const noexcept -> DAG {
  DAG r;
  for (auto pair : adj) {
    uint32_t iv = pair.first;
    for (uint32_t const& iw : pair.second) r.addEdge(iw, iv);
  }
  return r;
}

auto flatten_bfs(DAG const& g, size_t output) noexcept
    -> std::optional<std::vector<size_t>> {
  DAG forward = g.reverse();
  DAG reverse = g;
  std::stack<size_t> revList;
  std::queue<size_t> waitingList;
  auto takeNode = [&](size_t node) noexcept -> void {
    revList.push(node);
    for (auto& pair : reverse.adj) pair.second.erase(node);
    reverse.adj.erase(node);
  };

  for (auto& node : g.adj) {
    if (node.second.size() == 0) {
      waitingList.push(node.first);
      break;
    }
  }
  //waitingList.push(output);
  while (!waitingList.empty()) {
    size_t front = waitingList.front();
    waitingList.pop();
    takeNode(front);
    for (auto& pair : reverse.adj)
      if (pair.second.size() == 0) {
        waitingList.push(pair.first);
        break;
      }
  }
  std::vector<size_t> flattened;
  while (!revList.empty()) {
    flattened.push_back(revList.top());
    revList.pop();
  }
  return flattened;
}

auto Graph::execute(rhi::CommandEncoder* encoder) noexcept -> void {
  renderData.graph = this;
  se::rhi::MultiFrameFlights* flights = gfx::GFXContext::flights.get();
  RenderContext renderContext;
  renderContext.cmdEncoder = encoder;
  renderContext.flightIdx = (flights == nullptr) ? 0 : flights->getFlightIndex();

  std::vector<size_t> marker_stack;

  for (size_t pass_id : flattenedPasses) {
    auto* pass = passes[pass_id];
    renderData.pass = pass;
    // insert barriers
    for (auto const& barriers : barriers[pass_id]) {
      renderContext.cmdEncoder->pipelineBarrier(barriers);
    }

    {  // deal with subgroup markers
      if (marker_stack.empty() && !pass->subgraphStack.empty()) {
        for (auto const& subgraph_id : pass->subgraphStack) {
          encoder->beginDebugUtilsLabelEXT(
              subgraphs[subgraphNameList[subgraph_id]]->marker);
          marker_stack.push_back(subgraph_id);
        }
      } else if (!marker_stack.empty() && pass->subgraphStack.empty()) {
        while (marker_stack.size() > 0) {
          marker_stack.pop_back();
          encoder->endDebugUtilsLabelEXT();
        }
      } else if (!marker_stack.empty() &&
                 marker_stack.back() != pass->subgraphStack.back()) {
        size_t offset = 0;
        while (offset < marker_stack.size() &&
               offset < pass->subgraphStack.size() &&
               marker_stack[offset] == pass->subgraphStack[offset]) {
          offset++;
        }
        while (marker_stack.size() > offset) {
          marker_stack.pop_back();
          encoder->endDebugUtilsLabelEXT();
        }
        for (size_t i = offset; i < pass->subgraphStack.size(); ++i) {
          encoder->beginDebugUtilsLabelEXT(
              subgraphs[subgraphNameList[pass->subgraphStack[i]]]->marker);
          marker_stack.push_back(pass->subgraphStack[i]);
        }
      }
    }

    encoder->beginDebugUtilsLabelEXT(pass->marker);
    pass->execute(&renderContext, renderData);
    encoder->endDebugUtilsLabelEXT();
  }

  while (marker_stack.size() > 0) {
    marker_stack.pop_back();
    encoder->endDebugUtilsLabelEXT();
  }

  {
    // insert barriers
    for (auto const& barriers : barriers[size_t(-1)])
      renderContext.cmdEncoder->pipelineBarrier(barriers);
  }
}

auto Graph::readback() noexcept -> void {
  renderData.graph = this;
  for (size_t pass_id : flattenedPasses) {
    auto* pass = passes[pass_id];
    renderData.pass = pass;
    pass->readback(renderData);
  }
}

auto tryMerge(rhi::ImageSubresourceRange const& x,
    rhi::ImageSubresourceRange const& y) noexcept
    -> std::optional<rhi::ImageSubresourceRange> {
  rhi::ImageSubresourceRange range;
  if (x.aspectMask != y.aspectMask) return std::nullopt;
  if (x.baseArrayLayer == y.baseArrayLayer && x.layerCount == y.layerCount) {
    if (x.baseMipLevel + x.levelCount == y.baseMipLevel) {
      return rhi::ImageSubresourceRange{x.aspectMask, x.baseMipLevel,
        x.levelCount + y.levelCount, x.baseArrayLayer, x.layerCount};
    } else if (y.baseMipLevel + y.levelCount == x.baseMipLevel) {
      return rhi::ImageSubresourceRange{x.aspectMask, y.baseMipLevel,
        x.levelCount + y.levelCount, x.baseArrayLayer, x.layerCount};
    } else return std::nullopt;
  } else if (x.baseMipLevel == y.baseMipLevel && x.levelCount == y.levelCount) {
    if (x.baseArrayLayer + x.layerCount == y.baseArrayLayer) {
      return rhi::ImageSubresourceRange{x.aspectMask, x.baseMipLevel,
        x.levelCount, x.baseArrayLayer, x.layerCount + y.layerCount};
    } else if (y.baseArrayLayer + y.layerCount == x.baseArrayLayer) {
      return rhi::ImageSubresourceRange{x.aspectMask, x.baseMipLevel,
        x.levelCount, y.baseArrayLayer, x.layerCount + y.layerCount};
    } else return std::nullopt;
  } else return std::nullopt;
}

auto tryMergeBarriers(std::vector<rhi::BarrierDescriptor>& barriers) noexcept
    -> void {
  // try merge macro barriers
  while (true) {
    bool should_continue = false;
    for (int i = 0; i < barriers.size(); ++i) {
      if (should_continue) break;
      for (int j = i + 1; j < barriers.size(); ++j) {
        if (should_continue) break;
        if ((barriers[i].srcStageMask == barriers[j].srcStageMask) &&
            (barriers[i].dstStageMask == barriers[j].dstStageMask)) {
          barriers[i].memoryBarriers.insert(barriers[i].memoryBarriers.end(),
            barriers[j].memoryBarriers.begin(), barriers[j].memoryBarriers.end());
          barriers[i].bufferMemoryBarriers.insert(
              barriers[i].bufferMemoryBarriers.end(),
              barriers[j].bufferMemoryBarriers.begin(),
              barriers[j].bufferMemoryBarriers.end());
          barriers[i].textureMemoryBarriers.insert(
              barriers[i].textureMemoryBarriers.end(),
              barriers[j].textureMemoryBarriers.begin(),
              barriers[j].textureMemoryBarriers.end());
          barriers.erase(barriers.begin() + j);
          should_continue = true;
        }
      }
    }
    if (should_continue) continue;
    break;
  }
  // try merge texture sub barriers
  for (int k = 0; k < barriers.size(); ++k) {
    auto& barrier = barriers[k];
    auto& textureBarriers = barrier.textureMemoryBarriers;
    while (true) {
      bool should_continue = false;
      for (int i = 0; i < textureBarriers.size(); ++i) {
        if (should_continue) break;
        for (int j = i + 1; j < textureBarriers.size(); ++j) {
          if (should_continue) break;
          if ((textureBarriers[i].texture == textureBarriers[j].texture) &&
              (textureBarriers[i].srcAccessMask ==
               textureBarriers[j].srcAccessMask) &&
              (textureBarriers[i].dstAccessMask ==
               textureBarriers[j].dstAccessMask) &&
              (textureBarriers[i].oldLayout == textureBarriers[j].oldLayout) &&
              (textureBarriers[i].newLayout == textureBarriers[j].newLayout)) {
            auto merged = tryMerge(textureBarriers[i].subresourceRange,
                                   textureBarriers[j].subresourceRange);
            if (merged.has_value()) {
              textureBarriers[i].subresourceRange = merged.value();
              textureBarriers.erase(textureBarriers.begin() + j);
              should_continue = true;
            }
          }
        }
      }
      if (should_continue) continue;
      break;
    }
  }
}

auto Graph::addPass(std::unique_ptr<Pass>&& pass,
                    std::string const& identifier) noexcept -> void {
  passNameList[identifier] = passID;
  pass->identifier = identifier;
  pass->subgraphStack = subgraphStack;
  pass->generateMarker();
  dag.adj[passID] = {};
  passesContainer[passID] = std::move(pass);
  passes[passID] = passesContainer[passID].get();
  passID++;
}

auto Graph::addPass(Pass* pass,
                    std::string const& identifier) noexcept -> void {
  passNameList[identifier] = passID;
  pass->identifier = identifier;
  pass->subgraphStack = subgraphStack;
  pass->generateMarker();
  dag.adj[passID] = {};
  passes[passID] = pass;
  passID++;
}

auto Graph::addSubgraph(std::unique_ptr<Subgraph>&& subgraph,
                        std::string const& identifier) noexcept -> void {
  uint32_t id = subgraphID++;
  subgraphNameList[id] = identifier;
  subgraph->identifier = identifier;
  subgraph->onRegister(this);
  subgraphsAlias[identifier] = subgraph->alias();
  subgraphStack.push_back(id);
  subgraphStack.pop_back();
  subgraph->generateMarker();
  subgraphs[identifier] = std::move(subgraph);
}

auto Graph::addEdge(std::string const& _src_pass,
                    std::string const& _src_resource,
                    std::string const& _dst_pass,
                    std::string const& _dst_resource) noexcept -> void {
  std::string src_pass = _src_pass;
  std::string dst_pass = _dst_pass;
  std::string src_resource = _src_resource;
  std::string dst_resource = _dst_resource;
  decodeAlias(src_pass, src_resource);
  decodeAlias(dst_pass, dst_resource);

  dag.addEdge(passNameList[src_pass], passNameList[dst_pass]);
  auto* dst_res = passes[passNameList[dst_pass]]->pReflection.getResourceInfo(dst_resource);
  auto* src_res = passes[passNameList[src_pass]]->pReflection.getResourceInfo(src_resource);
  if (dst_res == nullptr) {
    se::root::print::error(
      std::format("RDG::Graph::addEdge Failed to find dst resource \"{0}\" "
        "in pass \"{1}\"", dst_resource, dst_pass));
    return;
  }
  if (src_res == nullptr) {
    se::root::print::error(
      std::format("RDG::Graph::addEdge Failed to find src resource \"{0}\" "
        "in pass \"{1}\"", src_resource, src_pass));
    return;
  }
  dst_res->prev = src_res;
}

auto Graph::addEdge(std::string const& src_pass,
                    std::string const& dst_pass) noexcept -> void {
  dag.addEdge(passNameList[src_pass], passNameList[dst_pass]);
}

auto Graph::decodeAlias(std::string& pass, std::string& resource) noexcept
    -> void {
  auto findPass = subgraphsAlias.find(pass);
  if (findPass != subgraphsAlias.end()) {
    auto findResource = findPass->second.dict.find(resource);
    if (findResource != findPass->second.dict.end()) {
      pass = findResource->second.pass;
      resource = findResource->second.resource;
    }
  }
}

auto Graph::setExternal(std::string const& _pass, std::string const& _resource,
                        gfx::TextureHandle tex) noexcept -> void {
  std::string pass = _pass, res = _resource;
  decodeAlias(pass, res);
  passes[passNameList[pass]]
    ->pReflection.getResourceInfo(res)
    ->info.texture.reference = tex;
}

auto Graph::setExternal(std::string const& _pass, std::string const& _resource,
    gfx::BufferHandle buff) noexcept -> void {
  std::string pass = _pass, res = _resource;
  decodeAlias(pass, res);
  passes[passNameList[pass]]
      ->pReflection.getResourceInfo(res)
      ->info.buffer.reference = buff;
}

auto Graph::markOutput(std::string const& _pass,
                       std::string const& _output) noexcept -> void {
  std::string pass = _pass, output = _output;
  decodeAlias(pass, output);
  output_pass = pass;
  output_resource = output;

  auto const& iter_id = passNameList.find(output_pass);
  if (iter_id == passNameList.end()) return;
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return;
  // enable color attachment for editor debug draw
  iter_pass->second->pReflection.getResourceInfo(output_resource)
      ->info.texture.usages |= (uint32_t)(rhi::TextureUsageBit::COLOR_ATTACHMENT);
}

inline auto AccessIsWrite(rhi::AccessFlagBits bit) noexcept -> bool {
  switch (bit) {
    case rhi::AccessFlagBits::INDIRECT_COMMAND_READ_BIT:
    case rhi::AccessFlagBits::INDEX_READ_BIT:
    case rhi::AccessFlagBits::VERTEX_ATTRIBUTE_READ_BIT:
    case rhi::AccessFlagBits::UNIFORM_READ_BIT:
    case rhi::AccessFlagBits::INPUT_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagBits::SHADER_READ_BIT:
    case rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagBits::TRANSFER_READ_BIT:
    case rhi::AccessFlagBits::HOST_READ_BIT:
    case rhi::AccessFlagBits::MEMORY_READ_BIT:
    case rhi::AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_READ_BIT:
    case rhi::AccessFlagBits::CONDITIONAL_RENDERING_READ_BIT:
    case rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT:
    case rhi::AccessFlagBits::ACCELERATION_STRUCTURE_READ_BIT:
    case rhi::AccessFlagBits::FRAGMENT_DENSITY_MAP_READ_BIT:
    case rhi::AccessFlagBits::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagBits::COMMAND_PREPROCESS_READ_BIT:
    case rhi::AccessFlagBits::NONE: return false;
    case rhi::AccessFlagBits::SHADER_WRITE_BIT:
    case rhi::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT:
    case rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT:
    case rhi::AccessFlagBits::TRANSFER_WRITE_BIT:
    case rhi::AccessFlagBits::HOST_WRITE_BIT:
    case rhi::AccessFlagBits::MEMORY_WRITE_BIT:
    case rhi::AccessFlagBits::TRANSFORM_FEEDBACK_WRITE_BIT:
    case rhi::AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT:
    case rhi::AccessFlagBits::ACCELERATION_STRUCTURE_WRITE_BIT:
    case rhi::AccessFlagBits::COMMAND_PREPROCESS_WRITE_BIT:
      return true;
    default:
      return false;
  }
}

auto Graph::getOutput() noexcept -> gfx::TextureHandle {
  auto const& iter_id = passNameList.find(output_pass);
  if (iter_id == passNameList.end()) return gfx::TextureHandle{};
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return gfx::TextureHandle{};
  renderData.pass = iter_pass->second;
  return renderData.getTexture(output_resource);
}

inline auto ExtractWriteAccessFlags(rhi::AccessFlags flag) noexcept -> rhi::AccessFlags {
  rhi::AccessFlags eflag = 0;
  for (int i = 0; i < 32; ++i) {
    const uint32_t bit = flag & (0x1 << i);
    if (bit != 0 && AccessIsWrite(rhi::AccessFlagBits(bit))) {
      eflag |= bit;
    }
  }
  return eflag;
}

inline auto ExtractReadAccessFlags(rhi::AccessFlags flag) noexcept -> rhi::AccessFlags {
  rhi::AccessFlags eflag = 0;
  for (int i = 0; i < 32; ++i) {
    const uint32_t bit = flag & (0x1 << i);
    if (bit != 0 && !AccessIsWrite(rhi::AccessFlagBits(bit))) {
      eflag |= bit;
    }
  }
  return eflag;
}

struct BufferResourceVirtualMachine {
  struct BufferSubresourceRange {
    uint64_t range_beg;
    uint64_t range_end;
    auto operator==(BufferSubresourceRange const& x) noexcept -> bool {
      return range_beg == x.range_beg && range_end == x.range_end;
    }
  };

  struct BufferSubresourceState {
    rhi::PipelineStages stageMask;
    rhi::AccessFlags access;
    auto operator==(BufferSubresourceState const& x) noexcept -> bool {
      return stageMask == x.stageMask && access == x.access;
    }
  };

  struct BufferSubresourceEntry {
    BufferSubresourceRange range;
    BufferSubresourceState state;
  };

  BufferResourceVirtualMachine(
    rhi::Buffer* buff,
    std::vector<BufferResource::ConsumeHistory> const& cosumeHistories)
      : buffer(buff) {
    // init state
    write_states.emplace_back(
        BufferSubresourceEntry{BufferSubresourceRange{0, buffer->size()},
                               BufferSubresourceState{0, 0}});
    read_states.emplace_back(
        BufferSubresourceEntry{BufferSubresourceRange{0, buffer->size()},
                               BufferSubresourceState{0, 0}});
    for (auto const& hentry : cosumeHistories) {
      for (auto const& subentry : hentry.entries) {
        updateSubresource(
            BufferSubresourceRange{subentry.offset, subentry.offset + subentry.size},
            BufferSubresourceState{
                subentry.stages, subentry.access});
      }
    }
  }

  auto valid(BufferSubresourceRange const& x) noexcept -> bool {
    return (x.range_beg < x.range_beg);
  }

  auto intersect(BufferSubresourceRange const& x,
                 BufferSubresourceRange const& y) noexcept
      -> std::optional<BufferSubresourceRange> {
    BufferSubresourceRange isect;
    isect.range_beg = std::max(x.range_beg, y.range_beg);
    isect.range_end = std::min(x.range_end, y.range_end);
    if (valid(isect))
      return isect;
    else
      return std::nullopt;
  }

  auto diff(BufferSubresourceRange const& x,
            BufferSubresourceRange const& y) noexcept
      -> std::vector<BufferSubresourceRange> {
    std::vector<BufferSubresourceRange> diffs;
    if (x.range_beg == y.range_beg && x.range_end == y.range_end) {
    // do nothing
    } else if (x.range_beg == y.range_beg) {
    diffs.emplace_back(
        BufferSubresourceRange{y.range_end, x.range_end});
    } else if (x.range_end == y.range_end) {
    diffs.emplace_back(BufferSubresourceRange{x.range_beg, y.range_beg});
    } else {
    diffs.emplace_back(BufferSubresourceRange{x.range_beg, y.range_beg});
    diffs.emplace_back(BufferSubresourceRange{y.range_end, x.range_end});
    }
    return diffs;
  }

  auto toBarrierDescriptor(BufferSubresourceRange const& range,
                           BufferSubresourceState const& prev,
                           BufferSubresourceState const& next) {
    rhi::BarrierDescriptor desc = rhi::BarrierDescriptor {
      prev.stageMask, next.stageMask, uint32_t(rhi::DependencyType::NONE),
          std::vector<rhi::MemoryBarrier*>{},
          std::vector<rhi::BufferMemoryBarrierDescriptor>{
              rhi::BufferMemoryBarrierDescriptor{
                  buffer, prev.access, next.access, range.range_beg,
                  range.range_end - range.range_beg}},
        std::vector<rhi::TextureMemoryBarrierDescriptor>{}};
    return desc;
  }

  auto updateSubresource(BufferSubresourceRange const& range,
                         BufferSubresourceState const& state) noexcept
      -> std::vector<rhi::BarrierDescriptor> {
    std::vector<rhi::BarrierDescriptor> barriers;
    std::vector<BufferSubresourceEntry> addedEntries;

    // First check write access
    rhi::AccessFlags write_access = ExtractWriteAccessFlags(state.access);
    if (write_access != 0) {
      BufferSubresourceState target_state;
      target_state.stageMask = state.stageMask;
      target_state.access = write_access;
      // Write - Write hazard
      for (auto iter = write_states.begin();;) {
        if (iter == write_states.end()) break;
        if (iter->range == range) {
          if (iter->state.access != 0)
            barriers.emplace_back(
                toBarrierDescriptor(range, iter->state, target_state));
          break;
        }
        std::optional<BufferSubresourceRange> isect =
            intersect(iter->range, range);
        if (isect.has_value()) {
          BufferSubresourceRange const& isect_range = isect.value();
          if (iter->state.access != 0)
            barriers.emplace_back(
                toBarrierDescriptor(isect_range, iter->state, target_state));
          iter++;
        } else {
          iter++;
        }
      }
      // Read - Write hazard
      for (auto iter = read_states.begin();;) {
        if (iter == read_states.end()) break;
        if (iter->range == range) {
          if (iter->state.access != 0)
            barriers.emplace_back(
                toBarrierDescriptor(range, iter->state, target_state));
          break;
        }
        std::optional<BufferSubresourceRange> isect =
            intersect(iter->range, range);
        if (isect.has_value()) {
          BufferSubresourceRange const& isect_range = isect.value();
          if (iter->state.access != 0)
            barriers.emplace_back(
                toBarrierDescriptor(isect_range, iter->state, target_state));
          iter++;
        } else {
          iter++;
        }
      }
    }

    // Then check read access
    rhi::AccessFlags read_access = ExtractReadAccessFlags(state.access);
    if (read_access != 0) {
      BufferSubresourceState target_state;
      target_state.stageMask = state.stageMask;
      target_state.access = read_access;
      // Write - Read hazard
      for (auto iter = write_states.begin();;) {
        if (iter == write_states.end()) break;
        if (iter->range == range) {
          if (iter->state.access != 0)
            barriers.emplace_back(
                toBarrierDescriptor(range, iter->state, target_state));
          break;
        }
        std::optional<BufferSubresourceRange> isect =
            intersect(iter->range, range);
        if (isect.has_value()) {
          BufferSubresourceRange const& isect_range = isect.value();
          if (iter->state.access != 0)
            barriers.emplace_back(
                toBarrierDescriptor(isect_range, iter->state, target_state));
          iter++;
        } else {
          iter++;
        }
      }
    }

    if (write_access != 0) {
      // Update write state
      for (auto iter = write_states.begin();;) {
        if (iter == write_states.end()) break;
        if (iter->range == range) {
          iter->state.stageMask = state.stageMask;
          iter->state.access = write_access;
          break;
        }
        std::optional<BufferSubresourceRange> isect =
            intersect(iter->range, range);
        if (isect.has_value()) {
          // isect ranges
          BufferSubresourceRange const& isect_range = isect.value();
          BufferSubresourceState isect_state;
          isect_state.stageMask = state.stageMask;
          isect_state.access = write_access;
          addedEntries.emplace_back(
              BufferSubresourceEntry{isect_range, isect_state});
          // diff ranges
          std::vector<BufferSubresourceRange> diff_ranges =
              diff(iter->range, isect_range);
          for (auto const& drange : diff_ranges) {
            addedEntries.emplace_back(
                BufferSubresourceEntry{drange, iter->state});
          }
          iter = write_states.erase(iter);
        } else {
          iter++;
        }
      }
      // Clear read state
      for (auto iter = read_states.begin();;) {
        if (iter == read_states.end()) break;
        if (iter->range == range) {
          iter->state.stageMask = 0;
          iter->state.access = 0;
          break;
        }
        std::optional<BufferSubresourceRange> isect =
            intersect(iter->range, range);
        if (isect.has_value()) {
          // isect ranges
          BufferSubresourceRange const& isect_range = isect.value();
          BufferSubresourceState isect_state;
          isect_state.stageMask = 0;
          isect_state.access = 0;
          addedEntries.emplace_back(
              BufferSubresourceEntry{isect_range, isect_state});
          // diff ranges
          std::vector<BufferSubresourceRange> diff_ranges =
              diff(iter->range, isect_range);
          for (auto const& drange : diff_ranges) {
            addedEntries.emplace_back(
                BufferSubresourceEntry{drange, iter->state});
          }
          iter = read_states.erase(iter);
        } else {
          iter++;
        }
      }
    }

    if (read_access != 0) {
      // Update read state
      for (auto iter = read_states.begin();;) {
        if (iter == read_states.end()) break;
        if (iter->range == range) {
          iter->state.stageMask |= state.stageMask;
          iter->state.access |= read_access;
          break;
        }
        std::optional<BufferSubresourceRange> isect =
            intersect(iter->range, range);
        if (isect.has_value()) {
          // isect ranges
          BufferSubresourceRange const& isect_range = isect.value();
          BufferSubresourceState isect_state = iter->state;
          isect_state.stageMask |= state.stageMask;
          isect_state.access |= read_access;
          addedEntries.emplace_back(
              BufferSubresourceEntry{isect_range, isect_state});
          // diff ranges
          std::vector<BufferSubresourceRange> diff_ranges =
              diff(iter->range, isect_range);
          for (auto const& drange : diff_ranges) {
            addedEntries.emplace_back(
                BufferSubresourceEntry{drange, iter->state});
          }
          iter = read_states.erase(iter);
        } else {
          iter++;
        }
      }
      //// Clear write state
      //for (auto iter = write_states.begin();;) {
      //  if (iter == write_states.end()) break;
      //  if (iter->range == range) {
      //    iter->state.stageMask = 0;
      //    iter->state.access = 0;
      //    break;
      //  }
      //  std::optional<BufferSubresourceRange> isect =
      //      intersect(iter->range, range);
      //  if (isect.has_value()) {
      //    // isect ranges
      //    BufferSubresourceRange const& isect_range = isect.value();
      //    BufferSubresourceState isect_state;
      //    isect_state.stageMask = 0;
      //    isect_state.access = 0;
      //    addedEntries.emplace_back(
      //        BufferSubresourceEntry{isect_range, isect_state});
      //    // diff ranges
      //    std::vector<BufferSubresourceRange> diff_ranges =
      //        diff(iter->range, isect_range);
      //    for (auto const& drange : diff_ranges) {
      //      addedEntries.emplace_back(
      //          BufferSubresourceEntry{drange, iter->state});
      //    }
      //    iter = write_states.erase(iter);
      //  } else {
      //    iter++;
      //  }
      //}
    }

    //tryMerge();
    return barriers;
  }

  rhi::Buffer* buffer;
  std::vector<BufferSubresourceEntry> write_states;
  std::vector<BufferSubresourceEntry> read_states;
};

auto Graph::generateBufferBarriers() noexcept -> void {
  for (auto& res : bufferResources) {
    if (res.second.cosumeHistories.size() == 0) continue;

    // deal with all max-possbiel notations
    for (auto& hentry : res.second.cosumeHistories) {
      for (auto& subentry : hentry.entries) {
        if (subentry.size == uint64_t(-1))
          subentry.size = res.second.buffer->buffer->size();
      }
    }
    BufferResourceVirtualMachine vm(res.second.buffer->buffer.get(),
                                    res.second.cosumeHistories);
    for (auto const& hentry : res.second.cosumeHistories) {
      for (auto const& subentry : hentry.entries) {
        std::vector<rhi::BarrierDescriptor> decses = vm.updateSubresource(
            BufferResourceVirtualMachine::BufferSubresourceRange{
                subentry.offset, subentry.offset + subentry.size},
            BufferResourceVirtualMachine::BufferSubresourceState{
                subentry.stages, subentry.access});
        for (auto const& desc : decses)
          barriers[hentry.passID].emplace_back(desc);
      }
    }
  }
}


struct TextureResourceVirtualMachine {
  struct TextureSubresourceRange {
    uint32_t level_beg;
    uint32_t level_end;
    uint32_t mip_beg;
    uint32_t mip_end;

    auto operator==(TextureSubresourceRange const& x) noexcept -> bool {
      return level_beg == x.level_beg && level_end == x.level_end &&
             mip_beg == x.mip_beg && mip_end == x.mip_end;
    }
  };

  struct TextureSubresourceState {
    rhi::PipelineStages stageMask;
    rhi::AccessFlags access;
    rhi::TextureLayout layout;

    auto operator==(TextureSubresourceState const& x) noexcept -> bool {
      return stageMask == x.stageMask && access == x.access &&
             layout == x.layout;
    }
  };

  struct TextureSubresourceEntry {
    TextureSubresourceRange range;
    TextureSubresourceState state;
  };

  TextureResourceVirtualMachine(
      rhi::Texture* tex,
      std::vector<TextureResource::ConsumeHistory> const& cosumeHistories)
      : texture(tex) {
    bool depthBit = rhi::hasDepthBit(tex->format());
    bool stencilBit = rhi::hasStencilBit(tex->format());
    if (depthBit) aspects |= uint32_t(rhi::TextureAspectBit::DEPTH_BIT);
    if (stencilBit) aspects |= uint32_t(rhi::TextureAspectBit::STENCIL_BIT);
    if (!depthBit && !stencilBit)
      aspects |= uint32_t(rhi::TextureAspectBit::COLOR_BIT);

    // init state
    // TODO :: fix more complicated cases
    states.emplace_back(TextureSubresourceEntry{
        TextureSubresourceRange{0, tex->depthOrArrayLayers(), 0,
                                tex->mipLevelCount()},
        TextureSubresourceState{
            (uint32_t)rhi::PipelineStageBit::ALL_COMMANDS_BIT,
            (uint32_t)rhi::AccessFlagBits::MEMORY_READ_BIT |
                (uint32_t)rhi::AccessFlagBits::MEMORY_WRITE_BIT,
            rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL}});
    for (auto const& hentry : cosumeHistories) {
      for (auto const& subentry : hentry.entries) {
        updateSubresource(
            TextureResourceVirtualMachine::TextureSubresourceRange{
                subentry.level_beg, subentry.level_end, subentry.mip_beg,
                subentry.mip_end},
            TextureResourceVirtualMachine::TextureSubresourceState{
                subentry.stages, subentry.access, subentry.layout});
        if (subentry.type == TextureInfo::ConsumeType::ColorAttachment ||
            subentry.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
          updateSubresource(
              TextureResourceVirtualMachine::TextureSubresourceRange{
                  subentry.level_beg, subentry.level_end, subentry.mip_beg,
                  subentry.mip_end},
              TextureResourceVirtualMachine::TextureSubresourceState{
                  subentry.stages, subentry.access,
                  rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL});
        }
      }
    }
  }

  auto valid(TextureSubresourceRange const& x) noexcept -> bool {
    return (x.level_beg < x.level_end && x.mip_beg < x.mip_end);
  }

  auto intersect(TextureSubresourceRange const& x,
                 TextureSubresourceRange const& y) noexcept
      -> std::optional<TextureSubresourceRange> {
    TextureSubresourceRange isect;
    isect.level_beg = std::max(x.level_beg, y.level_beg);
    isect.level_end = std::min(x.level_end, y.level_end);
    isect.mip_beg = std::max(x.mip_beg, y.mip_beg);
    isect.mip_end = std::min(x.mip_end, y.mip_end);
    if (valid(isect))
      return isect;
    else
      return std::nullopt;
  }

  auto merge(TextureSubresourceRange const& x,
             TextureSubresourceRange const& y) noexcept
      -> std::optional<TextureSubresourceRange> {
    if (x.level_beg == y.level_beg && x.level_end == y.level_end) {
      if (x.mip_beg == y.mip_end)
        return TextureSubresourceRange{x.level_beg, x.level_end, y.mip_beg,
                                       x.mip_end};
      else if (x.mip_end == y.mip_beg)
        return TextureSubresourceRange{x.level_beg, x.level_end, x.mip_beg,
                                       y.mip_end};
      else
        return std::nullopt;
    } else if (x.mip_beg == y.mip_beg && x.mip_end == y.mip_end) {
      if (x.level_beg == y.level_end)
        return TextureSubresourceRange{y.level_beg, x.level_end, x.mip_beg,
                                       x.mip_end};
      else if (x.level_end == y.level_beg)
        return TextureSubresourceRange{x.level_beg, y.level_end, x.mip_beg,
                                       x.mip_end};
      else
        return std::nullopt;
    } else
      return std::nullopt;
  }

  auto tryMerge() noexcept -> void {
    if (states.size() <= 1) return;
    while (true) {
      for (size_t i = 1; i <= states.size(); ++i) {
        if (i == states.size()) return;
        if (states[i].state == states[i - 1].state) {
          std::optional<TextureSubresourceRange> merged =
              merge(states[i - 1].range, states[i].range);
          if (merged.has_value()) {
            states[i - 1].range = merged.value();
            states.erase(states.begin() + i);
            break;
          }
        }
      }
    }
  }

  auto diff(TextureSubresourceRange const& x,
            TextureSubresourceRange const& y) noexcept
      -> std::vector<TextureSubresourceRange> {
    std::vector<TextureSubresourceRange> diffs;
    auto fn_subdivide_mip = [&](uint32_t level_beg, uint32_t level_end) {
      if (x.mip_beg == y.mip_beg && x.mip_end == y.mip_end) {
        // do nothing
      } else if (x.mip_beg == y.mip_beg) {
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   y.mip_end, x.mip_end});
      } else if (x.mip_end == y.mip_end) {
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   x.mip_beg, y.mip_beg});
      } else {
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   x.mip_beg, y.mip_beg});
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   y.mip_end, x.mip_end});
      }
    };
    if (x.level_beg == y.level_beg && x.level_end == y.level_end) {
      fn_subdivide_mip(x.level_beg, x.level_end);
    } else if (x.level_beg == y.level_beg) {
      diffs.emplace_back(TextureSubresourceRange{y.level_end, x.level_end,
                                                 x.mip_beg, x.mip_end});
      fn_subdivide_mip(y.level_beg, y.level_end);
    } else if (x.level_end == y.level_end) {
      diffs.emplace_back(TextureSubresourceRange{x.level_beg, y.level_beg,
                                                 x.mip_beg, x.mip_end});
      fn_subdivide_mip(y.level_beg, y.level_end);
    } else {
      diffs.emplace_back(TextureSubresourceRange{x.level_beg, y.level_beg,
                                                 x.mip_beg, x.mip_end});
      diffs.emplace_back(TextureSubresourceRange{y.level_end, x.level_end,
                                                 x.mip_beg, x.mip_end});
      fn_subdivide_mip(y.level_beg, y.level_end);
    }
    return diffs;
  }

  auto toBarrierDescriptor(TextureSubresourceRange const& range,
                           TextureSubresourceState const& prev,
                           TextureSubresourceState const& next) {
    rhi::BarrierDescriptor desc = rhi::BarrierDescriptor{
        prev.stageMask,
        next.stageMask,
        uint32_t(rhi::DependencyType::NONE),
        std::vector<rhi::MemoryBarrier*>{},
        std::vector<rhi::BufferMemoryBarrierDescriptor>{},
        std::vector<rhi::TextureMemoryBarrierDescriptor>{
            rhi::TextureMemoryBarrierDescriptor{
                texture,
                rhi::ImageSubresourceRange{
                    aspects, range.mip_beg, range.mip_end - range.mip_beg,
                    range.level_beg, range.level_end - range.level_beg},
                prev.access, next.access, prev.layout, next.layout}}};
    return desc;
  }

  auto updateSubresource(TextureSubresourceRange const& range,
                         TextureSubresourceState const& state) noexcept
      -> std::vector<rhi::BarrierDescriptor> {
    std::vector<rhi::BarrierDescriptor> barriers;

    std::vector<TextureSubresourceEntry> addedEntries;
    for (auto iter = states.begin();;) {
      if (iter == states.end()) break;
      if (iter->range == range) {
        barriers.emplace_back(toBarrierDescriptor(range, iter->state, state));
        iter->state = state;
        return barriers;
      }
      std::optional<TextureSubresourceRange> isect =
          intersect(iter->range, range);
      if (isect.has_value()) {
        TextureSubresourceRange const& isect_range = isect.value();
        barriers.emplace_back(
            toBarrierDescriptor(isect_range, iter->state, state));
        addedEntries.emplace_back(TextureSubresourceEntry{isect_range, state});
        std::vector<TextureSubresourceRange> diff_ranges =
            diff(iter->range, isect_range);
        for (auto const& drange : diff_ranges)
          addedEntries.emplace_back(
              TextureSubresourceEntry{drange, iter->state});
        iter = states.erase(iter);
      } else {
        iter++;
      }
    }
    states.insert(states.end(), addedEntries.begin(), addedEntries.end());
    tryMerge();
    return barriers;
  }

  rhi::Texture* texture;
  rhi::TextureAspects aspects = 0;
  std::vector<TextureSubresourceEntry> states;
};

auto Graph::generateTextureBarriers() noexcept -> void {
  for (auto& res : textureResources) {
    if (res.second.cosumeHistories.size() == 0) continue;

    TextureResourceVirtualMachine vm(res.second.texture->texture.get(),
                                     res.second.cosumeHistories);
    // deal with all max-possbiel notations
    for (auto& hentry : res.second.cosumeHistories) {
      for (auto& subentry : hentry.entries) {
        if (subentry.mip_end == uint64_t(-1))
          subentry.mip_end = res.second.texture->texture->mipLevelCount();
      }
    }
    for (auto const& hentry : res.second.cosumeHistories) {
      for (auto const& subentry : hentry.entries) {
        std::vector<rhi::BarrierDescriptor> decses = vm.updateSubresource(
            TextureResourceVirtualMachine::TextureSubresourceRange{
                subentry.level_beg, subentry.level_end, subentry.mip_beg,
                subentry.mip_end},
            TextureResourceVirtualMachine::TextureSubresourceState{
                subentry.stages, subentry.access, subentry.layout});
        for (auto const& desc : decses)
          barriers[hentry.passID].emplace_back(desc);
        if (subentry.type == TextureInfo::ConsumeType::ColorAttachment ||
            subentry.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
          //vm.updateSubresource(
          //    TextureResourceVirtualMachine::TextureSubresourceRange{
          //        subentry.level_beg, subentry.level_end, subentry.mip_beg,
          //        subentry.mip_end},
          //    TextureResourceVirtualMachine::TextureSubresourceState{
          //        (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT |
          //        (uint32_t)RHI::PipelineStages::VERTEX_INPUT_BIT |
          //        (uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
          //        (uint32_t)RHI::PipelineStages::TESSELLATION_CONTROL_SHADER_BIT |
          //        (uint32_t)RHI::PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT |
          //        (uint32_t)RHI::PipelineStages::GEOMETRY_SHADER_BIT |
          //        (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT |
          //        (uint32_t)RHI::PipelineStages::EARLY_FRAGMENT_TESTS_BIT |
          //        (uint32_t)RHI::PipelineStages::LATE_FRAGMENT_TESTS_BIT |
          //        (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT, subentry.access,
          //        RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL});
        }
      }
    }
  }

  for (auto& barrier : barriers) {
    tryMergeBarriers(barrier.second);
  }
}

auto Graph::build() noexcept -> bool {
  renderData.graph = this;
  // Flatten the graph
  std::optional<std::vector<size_t>> flat =
      flatten_bfs(dag, passNameList[output_pass]);
  if (!flat.has_value()) return false;
  flattenedPasses = flat.value();

  // Find the resource
  for (size_t i = 0; i < flattenedPasses.size(); ++i) {
    // devirtualize internal resources
    for (auto& internal : passes[flattenedPasses[i]]->pReflection.internalResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        size_t rid = resourceID++;
        textureResources[rid] = TextureResource{};
        textureResources[rid].desc = toTextureDescriptor(internal.second.info.texture, standardSize);
        textureResources[rid].name = "RDG::" + passes[flattenedPasses[i]]->identifier + "::" + internal.first;
        textureResources[rid].cosumeHistories.push_back({flattenedPasses[i], internal.second.info.texture.consumeHistories});
        internal.second.devirtualizeID = rid;
      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = resourceID++;
        bufferResources[rid] = BufferResource{};
        bufferResources[rid].desc = toBufferDescriptor(internal.second.info.buffer);
        bufferResources[rid].cosumeHistories.push_back({flattenedPasses[i], internal.second.info.buffer.consumeHistories});
        internal.second.devirtualizeID = rid;
      }
    }
    // devirtualize output resources
    for (auto& output : passes[flattenedPasses[i]]->pReflection.outputResources) {
      if (output.second.type == ResourceInfo::Type::Texture) {
        size_t rid = resourceID++;
        textureResources[rid] = TextureResource{};
        textureResources[rid].desc = toTextureDescriptor(output.second.info.texture, standardSize);
        textureResources[rid].name = "RDG::" + passes[flattenedPasses[i]]->identifier + "::" + output.first;
        textureResources[rid].cosumeHistories.push_back({flattenedPasses[i], output.second.info.texture.consumeHistories});
        output.second.devirtualizeID = rid;
        if (output.second.info.texture.reference.get() != nullptr)
          textureResources[rid].texture = output.second.info.texture.reference;
      } else if (output.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = resourceID++;
        bufferResources[rid] = BufferResource{};
        bufferResources[rid].desc = toBufferDescriptor(output.second.info.buffer);
        bufferResources[rid].cosumeHistories.push_back({flattenedPasses[i], output.second.info.buffer.consumeHistories});
        output.second.devirtualizeID = rid;
        if (output.second.info.buffer.reference.get() != nullptr)
          bufferResources[rid].buffer = output.second.info.buffer.reference;
      }
    }
    // devirtualize input resources
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.inputResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (!internal.second.info.texture.reference.get()) {
          size_t rid = internal.second.prev->devirtualizeID;
          internal.second.devirtualizeID = rid;
          rhi::TextureDescriptor desc = toTextureDescriptor(internal.second.info.texture, standardSize);
          textureResources[rid].desc.usage |= desc.usage;
          textureResources[rid].cosumeHistories.push_back(
            {flattenedPasses[i], internal.second.info.texture.consumeHistories});
        } else {
          size_t rid = resourceID++;
          textureResources[rid] = TextureResource{};
          textureResources[rid].desc = toTextureDescriptor(internal.second.info.texture, standardSize);
          textureResources[rid].cosumeHistories.push_back(
            {flattenedPasses[i], internal.second.info.texture.consumeHistories});
          internal.second.devirtualizeID = rid;
          textureResources[rid].texture = internal.second.info.texture.reference;
        }
      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = internal.second.prev->devirtualizeID;
        internal.second.devirtualizeID = rid;
        rhi::BufferDescriptor desc = toBufferDescriptor(internal.second.info.buffer);
        bufferResources[rid].desc.usage |= desc.usage;
        bufferResources[rid].cosumeHistories.push_back(
          {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
      }
    }
    // devirtualize input-output resources
    for (auto& inout : passes[flattenedPasses[i]]->pReflection.inputOutputResources) {
      if (inout.second.type == ResourceInfo::Type::Texture) {
        if (inout.second.prev == nullptr) {
          se::root::print::error(std::format(
            "RDG::Graph::build() failed, input-output resource \"{0}\" in "
            "pass \"{1}\" has no source.",
            inout.first, passes[flattenedPasses[i]]->identifier));
        }
        size_t rid = inout.second.prev->devirtualizeID;
        inout.second.devirtualizeID = rid;
        rhi::TextureDescriptor desc = toTextureDescriptor(inout.second.info.texture, standardSize);
        textureResources[rid].desc.usage |= desc.usage;
        textureResources[rid].cosumeHistories.push_back(
          {flattenedPasses[i], inout.second.info.texture.consumeHistories});
      } else if (inout.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = inout.second.prev->devirtualizeID;
        inout.second.devirtualizeID = rid;
        rhi::BufferDescriptor desc = toBufferDescriptor(inout.second.info.buffer);
        bufferResources[rid].desc.usage |= desc.usage;
        bufferResources[rid].cosumeHistories.push_back(
          {flattenedPasses[i], inout.second.info.buffer.consumeHistories});
      }
    }
    // Deal with relative to another tex case
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.internalResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (internal.second.info.texture.sizeDef ==
            TextureInfo::SizeDefine::RelativeToAnotherTex) {
          auto const& refResourceInfo =
              passes[flattenedPasses[i]]->pReflection.getResourceInfo(
                  internal.second.info.texture.sizeRefName);
          textureResources[internal.second.devirtualizeID].desc.size =
              textureResources[refResourceInfo->devirtualizeID].desc.size;
        }
      }
    }
    // devirtualize output resources
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.outputResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (internal.second.info.texture.sizeDef ==
            TextureInfo::SizeDefine::RelativeToAnotherTex) {
          auto const& refResourceInfo =
              passes[flattenedPasses[i]]->pReflection.getResourceInfo(
                  internal.second.info.texture.sizeRefName);
          textureResources[internal.second.devirtualizeID].desc.size =
              textureResources[refResourceInfo->devirtualizeID].desc.size;
        }
      }
    }
  }

  // devirtualize all the resources
  std::unique_ptr<rhi::CommandEncoder> commandEncoder =
    gfx::GFXContext::device->createCommandEncoder({ nullptr });
  for (auto& res : textureResources) {
    if (res.second.desc.mipLevelCount == uint32_t(-1)) {
      res.second.desc.mipLevelCount =
          std::log2(std::max(res.second.desc.size.width,
                             res.second.desc.size.height)) + 1;
    }

    rdg::TextureInfo::ConsumeEntry final_consume =
      rdg::TextureInfo::ConsumeEntry{
        rdg::TextureInfo::ConsumeType::TextureBinding}
        .addStage((uint32_t)rhi::PipelineStageBit::FRAGMENT_SHADER_BIT)
        .setLayout(rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL)
        .setAccess(uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT))
        .setSubresource(0, res.second.desc.mipLevelCount, 0,
                           res.second.desc.arrayLayerCount);
    res.second.cosumeHistories.push_back({size_t(-1), {final_consume}});

    if (res.second.texture.get() == nullptr) {
      res.second.texture = gfx::GFXContext::create_texture_desc(res.second.desc);
      res.second.texture->texture->setName(res.second.name);

      commandEncoder->pipelineBarrier(se::rhi::BarrierDescriptor{
        (uint32_t)se::rhi::PipelineStageBit::BOTTOM_OF_PIPE_BIT,
        (uint32_t)se::rhi::PipelineStageBit::ALL_GRAPHICS_BIT, 0,
        {}, {},
        std::vector<se::rhi::TextureMemoryBarrierDescriptor>{
          se::rhi::TextureMemoryBarrierDescriptor{
            res.second.texture->texture.get(),
            se::rhi::ImageSubresourceRange {
              getTextureAspect(res.second.texture->texture->format()), 0, 1, 0, 1},
              // memory barrier mask
              (uint32_t)se::rhi::AccessFlagBits::NONE,
              (uint32_t)se::rhi::AccessFlagBits::NONE,
              // only if layout transition is need
              se::rhi::TextureLayout::UNDEFINED,
              se::rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL
            }
        }
      });
    }
  }
  gfx::GFXContext::device->getGraphicsQueue()->submit({ commandEncoder->finish() });
  gfx::GFXContext::device->waitIdle();

  for (auto& res : bufferResources) {
    if (res.second.buffer.get() == nullptr) {
      res.second.buffer = gfx::GFXContext::create_buffer_desc(res.second.desc);
    }
  }
  // create barriers
  generateTextureBarriers();
  generateBufferBarriers();
  return true;
}

auto Graph::getTextureResource(std::string const& _pass,
  std::string const& _output) noexcept -> gfx::TextureHandle {
  std::string pass = _pass, output = _output;
  decodeAlias(pass, output);
  auto const& iter_id = passNameList.find(pass);
  if (iter_id == passNameList.end()) return gfx::TextureHandle{};
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return gfx::TextureHandle{};
  renderData.pass = iter_pass->second;
  return renderData.getTexture(output);
}

auto Graph::getBufferResource(std::string const& _pass,
  std::string const& _output) noexcept
  -> gfx::BufferHandle {
  std::string pass = _pass, output = _output;
  decodeAlias(pass, output);
  auto const& iter_id = passNameList.find(pass);
  if (iter_id == passNameList.end()) return gfx::BufferHandle{};
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return gfx::BufferHandle{};
  renderData.pass = iter_pass->second;
  return renderData.getBuffer(output);
}

auto Graph::getPass(std::string const& name) noexcept -> Pass* {
  auto iter = passNameList.find(name);
  if (iter == passNameList.end()) return nullptr;
  else return passes[iter->second];
}

auto Graph::bindScene(gfx::SceneHandle scene) noexcept -> void {
  renderData.scene = scene;
}

auto Pipeline::setStandardSize(se::ivec3 size) noexcept -> void {
  auto graphs = getAllGraphs();
  for (auto& graph : graphs) {
      graph->standardSize = size;
  }
}

auto Pipeline::bindScene(gfx::SceneHandle scene) noexcept -> void {
  auto graphs = getActiveGraphs();
  for (auto graph : graphs) {
    graph->bindScene(scene);
  }
}
}