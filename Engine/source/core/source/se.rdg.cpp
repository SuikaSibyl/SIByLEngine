#include <se.rdg.hpp>
#include <imnodes/imnodes.h>
#include <stack>
#include <queue>
using namespace entt::literals;

namespace se {
namespace rdg {
  auto DAG::add_edge(uint32_t src, uint32_t dst) noexcept -> void {
    adj[src].insert(dst);
    if (adj.find(dst) == adj.end()) adj[dst] = {};
  }

  auto DAG::reverse() const noexcept -> DAG {
    DAG r;
    for (auto pair : adj) {
      uint32_t iv = pair.first;
      for (uint32_t const& iw : pair.second) r.add_edge(iw, iv);
    }
    return r;
  }

  auto flatten_bfs(DAG const& g) noexcept
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

	auto BufferInfo::with_size(uint32_t _size) noexcept -> BufferInfo& {
		m_size = _size;
		return *this;
	}

	auto BufferInfo::with_usages(Flags<rhi::BufferUsageEnum> _usages) noexcept -> BufferInfo& {
		m_usages = _usages;
		return *this;
	}

  auto BufferInfo::with_memory_properties(Flags<rhi::MemoryPropertyEnum> prop) noexcept -> BufferInfo& {
    m_memoryProperties |= prop;
    return *this;
  }

	auto BufferInfo::consume(gfx::Buffer::ConsumeEntry const& entry) noexcept -> BufferInfo& {
		m_consumeHistories.m_entries.emplace_back(entry);
		return *this;
	}

  auto BufferInfo::to_descriptor() noexcept -> rhi::BufferDescriptor {
    return rhi::BufferDescriptor{ m_size, m_usages,
      rhi::BufferShareMode::EXCLUSIVE, m_memoryProperties };
  }

  auto TextureInfo::consume(gfx::Texture::ConsumeEntry const& _entry) noexcept -> TextureInfo& {
    gfx::Texture::ConsumeEntry entry = _entry;
    if (entry.type == gfx::Texture::ConsumeType::ColorAttachment) {
      entry.access |= rhi::AccessFlagEnum::COLOR_ATTACHMENT_READ_BIT | 
        rhi::AccessFlagEnum::COLOR_ATTACHMENT_WRITE_BIT;
      entry.stages |= (rhi::PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT);
      entry.layout = rhi::TextureLayoutEnum::COLOR_ATTACHMENT_OPTIMAL;
    }
    else if (entry.type == gfx::Texture::ConsumeType::DepthStencilAttachment) {
      entry.access |=
        rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      entry.stages |= (rhi::PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT) |
        (rhi::PipelineStageEnum::EARLY_FRAGMENT_TESTS_BIT) |
        (rhi::PipelineStageEnum::LATE_FRAGMENT_TESTS_BIT);
      entry.layout = rhi::TextureLayoutEnum::DEPTH_ATTACHMENT_OPTIMAL;
    }
    else if (entry.type == gfx::Texture::ConsumeType::TextureBinding) {
      entry.access |= rhi::AccessFlagEnum::SHADER_READ_BIT;
      entry.layout = rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL;
    }
    else if (entry.type == gfx::Texture::ConsumeType::StorageBinding) {
      entry.access |= rhi::AccessFlagEnum::SHADER_READ_BIT |
        rhi::AccessFlagEnum::SHADER_WRITE_BIT;
      entry.layout = rhi::TextureLayoutEnum::GENERAL;
    }
    m_consumeHistories.m_entries.emplace_back(entry);
    return *this;
  }

  auto TextureInfo::consume_as_storage_binding_in_compute() noexcept -> TextureInfo& {
    return with_usages(rhi::TextureUsageEnum::STORAGE_BINDING)
      .consume(gfx::Texture::ConsumeEntry(gfx::Texture::ConsumeType::StorageBinding)
      .add_stage(rhi::PipelineStageEnum::COMPUTE_SHADER_BIT));
  }
  
  auto TextureInfo::consume_as_color_attachment_at(uint32_t loc) noexcept -> TextureInfo& {
    return with_usages(se::rhi::TextureUsageEnum::COLOR_ATTACHMENT)
      .consume(se::gfx::Texture::ConsumeEntry(se::gfx::Texture::ConsumeType::ColorAttachment)
      .add_stage(se::rhi::PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT)
      .set_attachment_loc(loc));
  }

  auto TextureInfo::consume_as_depth_stencil_attachment_at(
    uint32_t loc,
    bool depth_write,
    se::rhi::CompareFunction depth_compare
  ) noexcept -> TextureInfo& {
    return with_usages(se::rhi::TextureUsageEnum::DEPTH_ATTACHMENT)
      .consume(se::gfx::Texture::ConsumeEntry(se::gfx::Texture::ConsumeType::DepthStencilAttachment)
        .enable_depth_write(depth_write).set_attachment_loc(loc)
        .set_depth_compare_fn(depth_compare));
  }

  auto TextureInfo::set_info(TextureInfo const& x) noexcept
    -> TextureInfo& {
    m_sizeDef = x.m_sizeDef;
    m_size = x.m_size;
    m_format = x.m_format;
    m_levels = x.m_levels;
    return *this;
  }

  auto TextureInfo::with_size(se::ivec3 absolute) noexcept
    -> TextureInfo& {
    m_sizeDef = SizeDefine::Absolute;
    m_size = absolute;
    return *this;
  }

  auto TextureInfo::with_size(se::vec3 relative) noexcept -> TextureInfo& {
    m_sizeDef = SizeDefine::Relative;
    m_size = relative;
    return *this;
  }

  auto TextureInfo::with_size_relative(std::string const& src, se::vec3 relative) noexcept -> TextureInfo& {
    m_sizeDef = SizeDefine::RelativeToAnotherTex;
    m_sizeRefName = src;
    m_size = relative;
    return *this;
  }

  auto TextureInfo::with_levels(uint32_t _levels) noexcept -> TextureInfo& {
    m_levels = _levels;
    return *this;
  }

  auto TextureInfo::with_layers(uint32_t _layers) noexcept -> TextureInfo& {
    m_layers = _layers;
    return *this;
  }

  auto TextureInfo::with_samples(uint32_t _samples) noexcept
    -> TextureInfo& {
    m_samples = _samples;
    return *this;
  }

  auto TextureInfo::with_format(rhi::TextureFormat _format) noexcept -> TextureInfo& {
    m_format = _format;
    return *this;
  }

  auto TextureInfo::with_stages(Flags<rhi::ShaderStageEnum> flags) noexcept -> TextureInfo& {
    m_sflags = flags;
    return *this;
  }

  auto TextureInfo::with_usages(Flags<rhi::TextureUsageEnum> _usages) noexcept -> TextureInfo& {
    m_usages = _usages;
    if (m_usages | rhi::TextureUsageEnum::COLOR_ATTACHMENT) {
      m_stages |= rhi::PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT;
      m_access |= rhi::AccessFlagEnum::COLOR_ATTACHMENT_READ_BIT |
                  rhi::AccessFlagEnum::COLOR_ATTACHMENT_WRITE_BIT;
      m_laytout = rhi::TextureLayoutEnum::COLOR_ATTACHMENT_OPTIMAL;
    }
    else if (m_usages | rhi::TextureUsageEnum::DEPTH_ATTACHMENT) {
      m_stages |= rhi::PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT |
        rhi::PipelineStageEnum::EARLY_FRAGMENT_TESTS_BIT |
        rhi::PipelineStageEnum::EARLY_FRAGMENT_TESTS_BIT;
      m_access |= rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      m_laytout = rhi::TextureLayoutEnum::DEPTH_ATTACHMENT_OPTIMAL;
    }
    else if (m_usages | rhi::TextureUsageEnum::TEXTURE_BINDING) {
      m_access |= rhi::AccessFlagEnum::SHADER_READ_BIT;
      m_laytout = rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL;
    }
    else if (m_usages | rhi::TextureUsageEnum::STORAGE_BINDING) {
      m_access |= rhi::AccessFlagEnum::SHADER_READ_BIT |
        rhi::AccessFlagEnum::SHADER_WRITE_BIT;
      m_laytout = rhi::TextureLayoutEnum::GENERAL;
    }
    return *this;
  }

  auto TextureInfo::get_size(se::ivec3 ref) const noexcept -> uvec3 {
    if (m_sizeDef == SizeDefine::Absolute) {
      se::ivec3 size = std::get<se::ivec3>(m_size);  // OK
      return uvec3{ uint32_t(size.x), uint32_t(size.y), uint32_t(size.z) };
    }
    else {
      se::vec3 size = std::get<se::vec3>(m_size);  // OK
      return uvec3{ uint32_t(std::ceil(size.x * ref.x)),
                    uint32_t(std::ceil(size.y * ref.y)),
                    uint32_t(std::ceil(size.z * ref.z)) };
    }
  }

  auto ResourceInfo::is_buffer() noexcept -> BufferInfo& {
    m_type = Type::Buffer;
    m_info.buffer = BufferInfo{};
    return m_info.buffer;
  }

  auto ResourceInfo::is_texture() noexcept -> TextureInfo& {
    m_type = Type::Texture;
    m_info.texture = TextureInfo{};
    return m_info.texture;
  }

  PassReflection::PassReflection() {
    m_inputResources.clear();
    m_outputResources.clear();
    m_inputOutputResources.clear();
    m_internalResources.clear();
  }

  auto PassReflection::add_input(std::string const& name) noexcept
    -> ResourceInfo& {
    ResourceInfo& resource = m_inputResources[name];
    resource.m_resoureceID = se::Resources::query_string_uid(name) + m_indexOffset;
    return resource;
  }

  auto PassReflection::add_output(std::string const& name) noexcept
    -> ResourceInfo& {
    ResourceInfo& resource = m_outputResources[name];
    resource.m_resoureceID = se::Resources::query_string_uid(name) + m_indexOffset;
    return resource;
  }

  auto PassReflection::add_input_output(std::string const& name) noexcept
    -> ResourceInfo& {
    ResourceInfo& resource = m_inputOutputResources[name];
    resource.m_resoureceID = se::Resources::query_string_uid(name) + m_indexOffset;
    return resource;
  }

  auto PassReflection::add_internal(std::string const& name) noexcept
    -> ResourceInfo& {
    ResourceInfo& resource = m_internalResources[name];
    resource.m_resoureceID = se::Resources::query_string_uid(name) + m_indexOffset;
    return resource;
  }

  auto PassReflection::add_external(std::string const& name) noexcept
    -> ResourceInfo& {
    ResourceInfo& resource = m_internalResources[name];
    resource.m_resoureceID = se::Resources::query_string_uid(name) + m_indexOffset;
    return resource;
  }

  auto PassReflection::get_depth_stencil_state() noexcept -> rhi::DepthStencilState {
    std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &m_inputResources, &m_outputResources, &m_inputOutputResources,
          &m_internalResources };
    for (auto& pool : pools) {
      for (auto& pair : *pool) {
        if (pair.second.m_type != ResourceInfo::Type::Texture) continue;
        if (rhi::has_depth_bit(pair.second.m_info.texture.m_format)) {
          gfx::Texture::ConsumeEntry entry;
          for (auto const& ch : pair.second.m_info.texture.m_consumeHistories.m_entries) {
            if (ch.type == gfx::Texture::ConsumeType::DepthStencilAttachment) {
              entry = ch;
              break;
            }
          }
          return rhi::DepthStencilState{ pair.second.m_info.texture.m_format,
                                        entry.depthWrite, entry.depthCmp };
        }
      }
    }
    return rhi::DepthStencilState{};
  }

  auto PassReflection::get_color_target_state() noexcept
    -> std::vector<rhi::ColorTargetState> {
    std::vector<rhi::ColorTargetState> cts;
    std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &m_inputResources, & m_outputResources, & m_inputOutputResources,
          &m_internalResources };
    for (auto& pool : pools) {
      for (auto& pair : *pool) {
        if (pair.second.m_type != ResourceInfo::Type::Texture) continue;
        if (!rhi::has_depth_bit(pair.second.m_info.texture.m_format)) {
          for (auto const& history : pair.second.m_info.texture.m_consumeHistories.m_entries) {
            if (history.type != gfx::Texture::ConsumeType::ColorAttachment)
              continue;
            cts.resize(history.attachLoc + 1);
            cts[history.attachLoc] = rhi::ColorTargetState{
              pair.second.m_info.texture.m_format,
              rhi::BlendState{history.bldOperation, history.srcFactor,history.dstFactor}
            };
          }
        }
      }
    }
    return cts;
  }

  auto to_buffer_descriptor(BufferInfo const& info) noexcept -> rhi::BufferDescriptor {
    return rhi::BufferDescriptor{ info.m_size, info.m_usages };
  }

  auto to_texture_descriptor(TextureInfo const& info, se::ivec3 ref_size) noexcept -> rhi::TextureDescriptor {
    uvec3 size = info.get_size(ref_size);
    return rhi::TextureDescriptor{
      size,
      info.m_levels,
      info.m_layers,
      info.m_samples,
      size.z == 1 ? rhi::TextureDimension::TEX2D
                  : rhi::TextureDimension::TEX3D,
      info.m_format,
      info.m_usages | rhi::TextureUsageEnum::TEXTURE_BINDING | rhi::TextureUsageEnum::COPY_SRC,
      std::vector<rhi::TextureFormat>{info.m_format},
    };
  }

  auto RenderData::get_texture(std::string const& name) const noexcept -> gfx::TextureHandle {
    if (m_pass->m_pReflection.m_inputResources.find(name) !=
      m_pass->m_pReflection.m_inputResources.end()) {
      return m_graph->m_textureResources[m_pass->m_pReflection.m_inputResources[name]
        .m_devirtualizeID].m_texture;
    }
    if (m_pass->m_pReflection.m_outputResources.find(name) !=
      m_pass->m_pReflection.m_outputResources.end()) {
      return m_graph
        ->m_textureResources[m_pass->m_pReflection.m_outputResources[name]
        .m_devirtualizeID].m_texture;
    }
    if (m_pass->m_pReflection.m_inputOutputResources.find(name) !=
      m_pass->m_pReflection.m_inputOutputResources.end()) {
      return m_graph
        ->m_textureResources[m_pass->m_pReflection.m_inputOutputResources[name]
        .m_devirtualizeID]
        .m_texture;
    }
    if (m_pass->m_pReflection.m_internalResources.find(name) !=
      m_pass->m_pReflection.m_internalResources.end()) {
      return m_graph
        ->m_textureResources[m_pass->m_pReflection.m_internalResources[name]
        .m_devirtualizeID]
        .m_texture;
    }
    return gfx::TextureHandle{};
  }

  auto RenderData::get_buffer(std::string const& name) const noexcept -> gfx::BufferHandle {
    if (m_pass->m_pReflection.m_inputResources.find(name) !=
      m_pass->m_pReflection.m_inputResources.end()) {
      return m_graph->m_bufferResources[m_pass->m_pReflection.m_inputResources[name]
        .m_devirtualizeID].m_buffer;
    }
    if (m_pass->m_pReflection.m_outputResources.find(name) !=
      m_pass->m_pReflection.m_outputResources.end()) {
      return m_graph
        ->m_bufferResources[m_pass->m_pReflection.m_outputResources[name]
        .m_devirtualizeID].m_buffer;
    }
    if (m_pass->m_pReflection.m_inputOutputResources.find(name) !=
      m_pass->m_pReflection.m_inputOutputResources.end()) {
      return m_graph
        ->m_bufferResources[m_pass->m_pReflection.m_inputOutputResources[name]
        .m_devirtualizeID]
        .m_buffer;
    }
    if (m_pass->m_pReflection.m_internalResources.find(name) !=
      m_pass->m_pReflection.m_internalResources.end()) {
      return m_graph
        ->m_bufferResources[m_pass->m_pReflection.m_internalResources[name]
        .m_devirtualizeID]
        .m_buffer;
    }
    return gfx::BufferHandle{};
  }

  auto RenderData::get_texture_id(std::string const& name) const noexcept -> uint32_t {
    if (m_pass->m_pReflection.m_inputResources.find(name) !=
      m_pass->m_pReflection.m_inputResources.end()) {
      return m_pass->m_pReflection.m_inputResources[name].m_resoureceID;
    }
    if (m_pass->m_pReflection.m_outputResources.find(name) !=
      m_pass->m_pReflection.m_outputResources.end()) {
      return m_pass->m_pReflection.m_outputResources[name].m_resoureceID;
    }
    if (m_pass->m_pReflection.m_inputOutputResources.find(name) !=
      m_pass->m_pReflection.m_inputOutputResources.end()) {
      return m_pass->m_pReflection.m_inputOutputResources[name].m_resoureceID;
    }
    if (m_pass->m_pReflection.m_internalResources.find(name) !=
      m_pass->m_pReflection.m_internalResources.end()) {
      return m_pass->m_pReflection.m_internalResources[name].m_resoureceID;
    }
    return -1;
  }

  auto RenderData::set_scene(gfx::SceneHandle scene) noexcept -> void {
    m_scene = scene;
  }

  auto RenderData::set_scene_batch(gfx::SceneBatchHandle sceneBatch) noexcept -> void {
    m_sceneBatch = sceneBatch;
  }

  auto RenderData::get_scene() const noexcept -> gfx::SceneHandle {
    if (m_scene.has_value()) return m_scene.value();
    se::error("Scene is acquired from RenderData, but is not defined.");
    return {};
  }

  auto RenderData::get_scene_batch() const noexcept -> gfx::SceneBatchHandle {
    if (m_sceneBatch.has_value()) return m_sceneBatch.value();
    se::error("SceneBatch is acquired from RenderData, but is not defined.");
    return {};
  }

  auto PassReflection::get_resource_info(std::string const& name) noexcept -> ResourceInfo* {
    if (m_inputResources.find(name) != m_inputResources.end()) {
      return &m_inputResources[name];
    }
    else if (m_outputResources.find(name) != m_outputResources.end()) {
      return &m_outputResources[name];
    }
    else if (m_inputOutputResources.find(name) != m_inputOutputResources.end()) {
      return &m_inputOutputResources[name];
    }
    else if (m_internalResources.find(name) != m_internalResources.end()) {
      return &m_internalResources[name];
    }
    se::error("RDG::PassReflection::getResourceInfo Failed to find resource \"{0}\"", name);
    return nullptr;
  }

  auto Pass::generate_marker() noexcept -> void {
    m_marker.name = m_identifier;
    m_marker.color = { 0.490, 0.607, 0.003, 1. };
  }

  auto Pass::init() noexcept -> void { 
    PassReflection _reflect;
    _reflect.m_indexOffset = se::Resources::query_string_uid(m_identifier);
    m_pReflection = this->reflect(_reflect);
  }

  auto PipelinePass::get_bindgroup(RenderContext* context, uint32_t size) noexcept -> rhi::BindGroup* {
    if (m_bindgroups.size() == 0) {
      se::error("PipelinePass :: get_bindgroup() called, but no bind group is built.");
    }
    return m_bindgroups[size][context->flightIdx].get();
  }

  auto PipelinePass::init(std::vector<gfx::ShaderModule*> shaderModules) noexcept  -> void {
    Pass::init();
    rhi::Device* device = gfx::GFXContext::device();
    // create pipeline reflection
    for (auto& sm : shaderModules) m_reflection = m_reflection + sm->m_reflection;
    // create bindgroup layouts
    m_bindgroupLayouts.resize(m_reflection.bindings.size());
    for (int i = 0; i < m_reflection.bindings.size(); ++i) {
      rhi::BindGroupLayoutDescriptor desc =
        gfx::ShaderReflection::toBindGroupLayoutDescriptor(m_reflection.bindings[i]);
      m_bindgroupLayouts[i] = device->create_bindgroup_layout(desc);
    }
    // create bindgroups
    m_bindgroups.resize(m_reflection.bindings.size());
    for (size_t i = 0; i < m_reflection.bindings.size(); ++i) {
      for (size_t j = 0; j < SE_FRAME_FLIGHTS_COUNT; ++j) {
        m_bindgroups[i][j] = device->create_bindgroup(rhi::BindGroupDescriptor{
            m_bindgroupLayouts[i].get(),
            std::vector<rhi::BindGroupEntry>{},
          });
      }
    }
    // create pipelineLayout
    rhi::PipelineLayoutDescriptor desc = {};
    desc.pushConstants.resize(m_reflection.pushConstant.size());
    for (int i = 0; i < m_reflection.pushConstant.size(); ++i)
      desc.pushConstants[i] = rhi::PushConstantEntry{
        m_reflection.pushConstant[i].stages,
        m_reflection.pushConstant[i].offset,
        m_reflection.pushConstant[i].range,
    };
    desc.bindGroupLayouts.resize(m_bindgroupLayouts.size());
    for (int i = 0; i < m_bindgroupLayouts.size(); ++i)
      desc.bindGroupLayouts[i] = m_bindgroupLayouts[i].get();
    m_pipelineLayout = device->create_pipeline_layout(desc);
  }

  auto PipelinePass::update_binding(
    RenderContext* context,
    std::string const& name,
    rhi::BindingResource const& resource) noexcept -> void {
    auto iter = m_reflection.bindingInfo.find(name);
    if (iter == m_reflection.bindingInfo.end()) {
      se::error("RDG::Binding Name " + name + " not found");
    }
    std::vector<rhi::BindGroupEntry> set_entries = {
      rhi::BindGroupEntry{iter->second.binding, resource} };
    get_bindgroup(context, iter->second.set)->update_binding(set_entries);
  }
  
  auto PipelinePass::update_binding_scene(RenderContext* context, gfx::SceneHandle scene) noexcept -> void {
    update_bindings(context, {
			{ "se_index_buffers",			scene->gpu_scene()->binding_resource_index() },
			{ "se_position_buffers",	scene->gpu_scene()->binding_resource_position() },
			{ "se_vertex_buffers",		scene->gpu_scene()->binding_resource_vertex() },
			{ "se_camera_buffers",		scene->gpu_scene()->binding_resource_camera() },
			{ "se_geometry_buffers",  scene->gpu_scene()->binding_resource_geometry() },
      { "se_material_buffers",  scene->gpu_scene()->binding_resource_material() },
      { "se_light_buffer",      scene->gpu_scene()->binding_resource_light() },
      { "se_textures",          scene->gpu_scene()->binding_resource_textures() },
      { "se_lightbvh_nodes",    scene->gpu_scene()->binding_resource_lightbvh_tree() },
      { "se_lightbvh_trails",   scene->gpu_scene()->binding_resource_lightbvh_trail() },
      { "se_scene_buffer",      scene->gpu_scene()->binding_resource_sceneinfo() },
		});
  }

  auto PipelinePass::update_binding_scene_batch(RenderContext* context, gfx::SceneBatchHandle sceneBatch) noexcept -> void {
    update_bindings(context, {
			{ "se_index_buffers",			sceneBatch->gpu_scene()->binding_resource_index() },
			{ "se_position_buffers",	sceneBatch->gpu_scene()->binding_resource_position() },
			{ "se_vertex_buffers",		sceneBatch->gpu_scene()->binding_resource_vertex() },
			{ "se_camera_buffers",		sceneBatch->gpu_scene()->binding_resource_camera() },
			{ "se_geometry_buffers",  sceneBatch->gpu_scene()->binding_resource_geometry() },
      { "se_material_buffers",  sceneBatch->gpu_scene()->binding_resource_material() },
      { "se_light_buffer",      sceneBatch->gpu_scene()->binding_resource_light() },
      { "se_textures",          sceneBatch->gpu_scene()->binding_resource_textures() },
      { "se_lightbvh_nodes",    sceneBatch->gpu_scene()->binding_resource_lightbvh_tree() },
      { "se_lightbvh_trails",   sceneBatch->gpu_scene()->binding_resource_lightbvh_trail() },
      { "se_scene_buffer",      sceneBatch->gpu_scene()->binding_resource_sceneinfo() },
		});
  }

  auto PipelinePass::update_bindings(
    RenderContext* context,
    std::vector<std::pair<std::string, rhi::BindingResource>> const& bindings) noexcept -> void {
    for (auto& pair : bindings) update_binding(context, pair.first, pair.second);
  }

  auto RenderPass::begin_pass(RenderContext* context, gfx::Texture* target) noexcept -> rhi::RenderPassEncoder* {
    m_passEncoders[context->flightIdx] = context->cmdEncoder->begin_render_pass(m_renderPassDescriptor);
    m_passEncoders[context->flightIdx]->set_pipeline(m_pipelines[context->flightIdx].get());
    prepare_dispatch(context, target);
    return m_passEncoders[context->flightIdx].get();
  }

  auto RenderPass::begin_pass(RenderContext* context, uint32_t width, uint32_t height) noexcept -> rhi::RenderPassEncoder* {
    m_passEncoders[context->flightIdx] = context->cmdEncoder->begin_render_pass(m_renderPassDescriptor);
    m_passEncoders[context->flightIdx]->set_pipeline(m_pipelines[context->flightIdx].get());
    prepare_dispatch(context, width, height);
    return m_passEncoders[context->flightIdx].get();
  }

  auto RenderPass::prepare_delegate_data(RenderContext* context,
    RenderData const& renderData) noexcept
    -> RenderData::DelegateData {
    RenderData::DelegateData data;
    data.cmdEncoder = context->cmdEncoder;
    data.passEncoder.render = m_passEncoders[context->flightIdx].get();
    data.pipelinePass = this;
    return data;
  }

  auto RenderPass::generate_marker() noexcept -> void {
    m_marker.name = m_identifier;
    m_marker.color = { 0.764, 0.807, 0.725, 1. };
  }

  auto RenderPass::prepare_dispatch(RenderContext* context, gfx::Texture* target) noexcept -> void {
    for (size_t i = 0; i < m_bindgroups.size(); ++i)
      m_passEncoders[context->flightIdx]->set_bindgroup(
        i, m_bindgroups[i][context->flightIdx].get());
    m_passEncoders[context->flightIdx]->set_viewport(
      0, 0, target->m_texture->width(), target->m_texture->height(), 0, 1);
    m_passEncoders[context->flightIdx]->set_scissor_rect(
      0, 0, target->m_texture->width(), target->m_texture->height());
  }

  auto RenderPass::prepare_dispatch(rdg::RenderContext* context, uint32_t width, uint32_t height) noexcept -> void {
    for (size_t i = 0; i < m_bindgroups.size(); ++i)
      m_passEncoders[context->flightIdx]->set_bindgroup(
        i, m_bindgroups[i][context->flightIdx].get());
    m_passEncoders[context->flightIdx]->set_viewport(0, 0, width, height, 0, 1);
    m_passEncoders[context->flightIdx]->set_scissor_rect(0, 0, width, height);
  }

  auto RenderPass::init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void {
    PipelinePass::init(std::vector<gfx::ShaderModule*>{vertex, fragment});
    rhi::Device* device = gfx::GFXContext::device();
    rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
        m_pipelineLayout.get(),
        rhi::VertexState{// vertex shader
                         vertex->m_shaderModule.get(),
                         "main",
                         // vertex attribute layout
                         {}},
        rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                            rhi::IndexFormat::UINT16_t},
        m_pReflection.get_depth_stencil_state(),
        rhi::MultisampleState{},
        rhi::FragmentState{// fragment shader
                           fragment->m_shaderModule.get(), "main",
                           m_pReflection.get_color_target_state()} };
    if (callback.has_value()) {
      callback.value()(pipelineDesc);
    }
    for (int i = 0; i < SE_FRAME_FLIGHTS_COUNT; ++i) {
      m_pipelines[i] = device->create_render_pipeline(pipelineDesc);
    }
  }

  auto RenderPass::init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment) noexcept -> void {
    PipelinePass::init(std::vector<gfx::ShaderModule*>{vertex, fragment});
    rhi::Device* device = gfx::GFXContext::device();
    rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
        m_pipelineLayout.get(),
        rhi::VertexState{// vertex shader
                         vertex->m_shaderModule.get(),
                         "main",
                         // vertex attribute layout
                         {}},
        rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                            rhi::IndexFormat::UINT16_t},
        m_pReflection.get_depth_stencil_state(),
        rhi::MultisampleState{},
        rhi::FragmentState{// fragment shader
                           fragment->m_shaderModule.get(), "main",
                           m_pReflection.get_color_target_state()} };
    for (int i = 0; i < SE_FRAME_FLIGHTS_COUNT; ++i) {
      m_pipelines[i] = device->create_render_pipeline(pipelineDesc);
    }
  }
  
  auto RenderPass::init(std::string const& shader_path) noexcept -> void {
    auto [vert, frag] = se::gfx::GFXContext::load_shader_slang(shader_path,
      std::array<std::pair<std::string, se::rhi::ShaderStageEnum>, 2>{
      std::make_pair("VertexMain", se::rhi::ShaderStageEnum::VERTEX),
        std::make_pair("FragmentMain", se::rhi::ShaderStageEnum::FRAGMENT),
    });
    m_vertexShader = vert;
    m_fragmentShader = frag;
    se::rdg::RenderPass::init(m_vertexShader.value().get(), m_fragmentShader.value().get());
  }

  auto RenderPass::init(gfx::ShaderModule* vertex,
    gfx::ShaderModule* geometry,
    gfx::ShaderModule* fragment) noexcept -> void {
    PipelinePass::init(
      std::vector<gfx::ShaderModule*>{vertex, geometry, fragment});
    rhi::Device* device = gfx::GFXContext::device();
    rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
        m_pipelineLayout.get(),
        rhi::VertexState{// vertex shader
                         vertex->m_shaderModule.get(),
                         "main",
                         // vertex attribute layout
                         {}},
        rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                            rhi::IndexFormat::UINT16_t},
        m_pReflection.get_depth_stencil_state(),
        rhi::MultisampleState{},
        rhi::FragmentState{// fragment shader
                           fragment->m_shaderModule.get(), "main",
                           m_pReflection.get_color_target_state()} };
    pipelineDesc.geometry = { geometry->m_shaderModule.get() };
    for (int i = 0; i < SE_FRAME_FLIGHTS_COUNT; ++i) {
      m_pipelines[i] = device->create_render_pipeline(pipelineDesc);
    }
  }

  auto RenderPass::init(gfx::ShaderModule* vertex,
    gfx::ShaderModule* geometry,
    gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void {
    PipelinePass::init(
      std::vector<gfx::ShaderModule*>{vertex, geometry, fragment});
    rhi::Device* device = gfx::GFXContext::device();
    rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
        m_pipelineLayout.get(),
        rhi::VertexState{// vertex shader
                         vertex->m_shaderModule.get(),
                         "main",
                         // vertex attribute layout
                         {}},
        rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                            rhi::IndexFormat::UINT16_t},
        m_pReflection.get_depth_stencil_state(),
        rhi::MultisampleState{},
        rhi::FragmentState{// fragment shader
                           fragment->m_shaderModule.get(), "main",
                           m_pReflection.get_color_target_state()} };
    pipelineDesc.geometry = { geometry->m_shaderModule.get() };
    if (callback.has_value()) {
      callback.value()(pipelineDesc);
    }
    for (int i = 0; i < SE_FRAME_FLIGHTS_COUNT; ++i) {
      m_pipelines[i] = device->create_render_pipeline(pipelineDesc);
    }
  }

  auto RenderPass::set_render_pass_descriptor(rhi::RenderPassDescriptor const& input) noexcept -> void {
    m_renderPassDescriptor = input;
  }

  //auto RenderPass::issueDirectDrawcalls(
  //  rhi::RenderPassEncoder* encoder, gfx::SceneHandle scene) noexcept -> void {
  //  encoder->setIndexBuffer(
  //    scene->getGPUScene()->index_buffer->buffer.get(), rhi::IndexFormat::UINT32_T, 0,
  //    scene->getGPUScene()->index_buffer->buffer->size());
  //  std::span<gfx::Scene::GeometryDrawData> geometries =
  //    scene->gpuScene.geometry_buffer->getHostAsStructuredArray<gfx::Scene::GeometryDrawData>();
  //  for (size_t geometry_idx = 0; geometry_idx < geometries.size(); ++geometry_idx) {
  //    auto& geometry = geometries[geometry_idx];
  //    beforeDirectDrawcall(encoder, geometry_idx, geometry);
  //    encoder->drawIndexed(
  //      geometry.indexSize, 1, geometry.indexOffset,
  //      geometry.vertexOffset, 0);
  //  }
  //}

  ComputePass::ComputePass(std::string const& shader_path) {
    m_computeShader = gfx::GFXContext::load_shader_slang(
      shader_path,
      { {"ComputeMain", rhi::ShaderStageEnum::COMPUTE} },
      {}, false)[0];
    init(m_computeShader->get());
  }

  auto ComputePass::begin_pass(rdg::RenderContext* context) noexcept -> rhi::ComputePassEncoder* {
    m_passEncoders[context->flightIdx] = context->cmdEncoder->begin_compute_pass();
    m_passEncoders[context->flightIdx]->set_pipeline(m_pipeline.get());
    prepare_dispatch(context);
    return m_passEncoders[context->flightIdx].get();
  }

  auto ComputePass::generate_marker() noexcept -> void {
    m_marker.name = m_identifier;
    m_marker.color = { 0.6, 0.721, 0.780, 1. };
  }

  auto ComputePass::prepare_dispatch(rdg::RenderContext* context) noexcept -> void {
    for (size_t i = 0; i < m_bindgroups.size(); ++i)
      m_passEncoders[context->flightIdx]->set_bindgroup(
        i, m_bindgroups[i][context->flightIdx].get());
  }

  auto ComputePass::init(gfx::ShaderModule* comp) noexcept -> void {
    PipelinePass::init({ comp });
    rhi::Device* device = gfx::GFXContext::device();
    m_pipeline = device->create_compute_pipeline(rhi::ComputePipelineDescriptor{
        m_pipelineLayout.get(), {comp->m_shaderModule.get(), "main"} });
  }
  
  auto ComputePass::init(std::string const& comp) noexcept -> void {
    m_computeShader = gfx::GFXContext::load_shader_slang(
      comp, { {"ComputeMain", rhi::ShaderStageEnum::COMPUTE} },
      {}, false)[0];
    init(m_computeShader->get());
  }

  auto ComputePass::init(std::string const& comp,
    std::vector<std::pair<char const*, char const*>> const& macros) noexcept -> void {
    m_computeShader = gfx::GFXContext::load_shader_slang(
      comp, { {"ComputeMain", rhi::ShaderStageEnum::COMPUTE} },
      macros, false)[0];
    init(m_computeShader->get());
  }

  auto Graph::add_pass(Pass* pass,
    std::string const& identifier) noexcept -> void {
    uint32_t passID = se::Resources::query_string_uid(identifier);
    pass->m_identifier = identifier;
    //pass->m_subgraphStack = subgraphStack;
    pass->generate_marker();
    m_passes[passID] = pass;
    m_dag.adj[passID] = {};
  }

  //auto decodeAlias(Graph* graph, std::string& pass, std::string& resource) noexcept
  //  -> void {
  //  auto findPass = subgraphsAlias.find(pass);
  //  if (findPass != subgraphsAlias.end()) {
  //    auto findResource = findPass->second.dict.find(resource);
  //    if (findResource != findPass->second.dict.end()) {
  //      pass = findResource->second.pass;
  //      resource = findResource->second.resource;
  //    }
  //  }
  //}

  auto Graph::add_edge(std::string const& _src_pass, std::string const& _src_resource,
    std::string const& _dst_pass, std::string const& _dst_resource) noexcept -> void {
    std::string src_pass = _src_pass;
    std::string dst_pass = _dst_pass;
    std::string src_resource = _src_resource;
    std::string dst_resource = _dst_resource;
    //decodeAlias(src_pass, src_resource);
    //decodeAlias(dst_pass, dst_resource);
    uint32_t dst_id = se::Resources::query_string_uid(dst_pass);
    uint32_t src_id = se::Resources::query_string_uid(src_pass);

    m_dag.add_edge(src_id, dst_id);
    auto* dst_res = m_passes[dst_id]->m_pReflection.get_resource_info(dst_resource);
    auto* src_res = m_passes[src_id]->m_pReflection.get_resource_info(src_resource);
    if (dst_res == nullptr) {
      se::error("RDG::Graph::addEdge Failed to find dst resource \"{0}\" "
          "in pass \"{1}\"", dst_resource, dst_pass);
      return;
    }
    if (src_res == nullptr) {
      se::error("RDG::Graph::addEdge Failed to find src resource \"{0}\" "
          "in pass \"{1}\"", src_resource, src_pass);
      return;
    }
    dst_res->m_prev = src_res;

    m_edges.push_back({ src_res->m_resoureceID, dst_res->m_resoureceID });
  }

  auto Graph::mark_output(std::string const& _pass, std::string const& _output) noexcept -> void {
    std::string pass = _pass, output = _output;
    //decodeAlias(pass, output);
    m_outputPass = pass;
    m_outputResource = output;

    uint32_t id = se::Resources::query_string_uid(pass);
    auto const& iter_pass = m_passes.find(id);
    if (iter_pass == m_passes.end()) {
      se::error("RDG :: during mark output, the pass name, {}, is not found.", _pass);
      return;
    }
    // enable color attachment for editor debug draw
    iter_pass->second->m_pReflection.get_resource_info(m_outputResource)
      ->m_info.texture.m_usages |= rhi::TextureUsageEnum::COLOR_ATTACHMENT;
  }

  auto Graph::set_standard_size(int width, int height) noexcept -> void {
    m_standardSize = { width, height, 1 };
  }

  auto Graph::get_output() noexcept -> std::optional<gfx::TextureHandle> {
    uint32_t id = se::Resources::query_string_uid(m_outputPass);
    auto const& iter_pass = m_passes.find(id);
    if (iter_pass == m_passes.end()) return {};
    m_renderData.m_pass = iter_pass->second;
    return m_renderData.get_texture(m_outputResource);
  }

  auto Graph::get_output_index() noexcept -> std::optional<uint32_t> {
    uint32_t id = se::Resources::query_string_uid(m_outputPass);
    auto const& iter_pass = m_passes.find(id);
    if (iter_pass == m_passes.end()) return {};
    m_renderData.m_pass = iter_pass->second;
    return m_renderData.get_texture_id(m_outputResource);
  }

  auto toTextureDescriptor(TextureInfo const& info, se::ivec3 ref_size) noexcept -> rhi::TextureDescriptor {
    uvec3 size = info.get_size(ref_size);
    return rhi::TextureDescriptor{
      size,
      info.m_levels,
      info.m_layers,
      info.m_samples,
      size.z == 1 ? rhi::TextureDimension::TEX2D
                  : rhi::TextureDimension::TEX3D,
      info.m_format,
      info.m_usages | rhi::TextureUsageEnum::TEXTURE_BINDING |
          rhi::TextureUsageEnum::COPY_SRC,
      std::vector<rhi::TextureFormat>{info.m_format},
      info.m_tflags };
  }

  auto tryMerge(rhi::TextureRange const& x,
    rhi::TextureRange const& y) noexcept
    -> std::optional<rhi::TextureRange> {
    rhi::TextureRange range;
    if (x.aspectMask._mask != y.aspectMask._mask) return std::nullopt;
    if (x.baseArrayLayer == y.baseArrayLayer && x.layerCount == y.layerCount) {
      if (x.baseMipLevel + x.levelCount == y.baseMipLevel) {
        return rhi::TextureRange{ x.aspectMask, x.baseMipLevel,
          x.levelCount + y.levelCount, x.baseArrayLayer, x.layerCount };
      }
      else if (y.baseMipLevel + y.levelCount == x.baseMipLevel) {
        return rhi::TextureRange{ x.aspectMask, y.baseMipLevel,
          x.levelCount + y.levelCount, x.baseArrayLayer, x.layerCount };
      }
      else return std::nullopt;
    }
    else if (x.baseMipLevel == y.baseMipLevel && x.levelCount == y.levelCount) {
      if (x.baseArrayLayer + x.layerCount == y.baseArrayLayer) {
        return rhi::TextureRange{ x.aspectMask, x.baseMipLevel,
          x.levelCount, x.baseArrayLayer, x.layerCount + y.layerCount };
      }
      else if (y.baseArrayLayer + y.layerCount == x.baseArrayLayer) {
        return rhi::TextureRange{ x.aspectMask, x.baseMipLevel,
          x.levelCount, y.baseArrayLayer, x.layerCount + y.layerCount };
      }
      else return std::nullopt;
    }
    else return std::nullopt;
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
          if ((barriers[i].srcStageMask._mask == barriers[j].srcStageMask._mask) &&
            (barriers[i].dstStageMask._mask == barriers[j].dstStageMask._mask)) {
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
              (textureBarriers[i].srcAccessMask._mask ==
                textureBarriers[j].srcAccessMask._mask) &&
              (textureBarriers[i].dstAccessMask._mask ==
                textureBarriers[j].dstAccessMask._mask) &&
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

  auto generateBufferBarriers(Graph* graph) noexcept -> void {
    for (auto& res : graph->m_bufferResources) {
      if (res.second.m_cosumeHistories.size() == 0) continue;

      // deal with all max-possbiel notations
      auto& consume = res.second.m_cosumeHistories[0];
      for (auto& sub_entry : consume.entries.m_entries) {
        if (sub_entry.m_size == uint64_t(-1))
          sub_entry.m_size = res.second.m_buffer->m_buffer->size();
      }

      gfx::Buffer::ResourceStateMachine sm;
      sm.m_buffer = res.second.m_buffer->m_buffer.get();
      
      for (auto& sub_entry : consume.entries.m_entries) {
        sm.update_subresource(
          gfx::Buffer::ResourceStateMachine::SubresourceRange{
              sub_entry.m_offset, sub_entry.m_offset + sub_entry.m_size },
          gfx::Buffer::ResourceStateMachine::SubresourceState{
              sub_entry.m_stages, sub_entry.m_access });
      }
      res.second.m_startState = sm;

      for (size_t i = 1; i < res.second.m_cosumeHistories.size(); ++i) {
        for (auto& sub_entry : consume.entries.m_entries) {
          std::vector<rhi::BarrierDescriptor> decses = sm.update_subresource(
            gfx::Buffer::ResourceStateMachine::SubresourceRange{
              sub_entry.m_offset, sub_entry.m_offset + sub_entry.m_size },
            gfx::Buffer::ResourceStateMachine::SubresourceState{
              sub_entry.m_stages, sub_entry.m_access });
          for (auto const& desc : decses)
            graph->m_barriers[res.second.m_cosumeHistories[i].passID].emplace_back(desc);
        }
      }
      res.second.m_endState = sm;
    }
  }

  auto generateTextureBarriers(Graph* graph) noexcept -> void {
    // we handle each resource sequentially
    for (auto& res : graph->m_textureResources) {
      // if the resource is not consumed at all, just skip
      if (res.second.m_cosumeHistories.size() == 0) continue;

      // The first task, is finding out what is the initial status of the resource
      // which essentially tells what the status it should be when first consumed in the graph
      // For now, we simply think it is the first history;
      gfx::Texture::ResourceStateMachine sm = gfx::Texture::ResourceStateMachine(
        res.second.m_texture->m_texture.get());

      auto& consume = res.second.m_cosumeHistories[0];
      for (auto& sub_entry : consume.entry.m_entries) {
        sm.update_subresource(
          gfx::Texture::ResourceStateMachine::SubresourceRange{
              sub_entry.level_beg, sub_entry.level_end, sub_entry.mip_beg, sub_entry.mip_end },
          gfx::Texture::ResourceStateMachine::SubresourceState{
              sub_entry.stages, sub_entry.access, sub_entry.layout });
      }
      res.second.m_startState = sm;

      for (size_t i = 1; i < res.second.m_cosumeHistories.size(); ++i) {
        for (auto& sub_entry : res.second.m_cosumeHistories[i].entry.m_entries) {
          std::vector<rhi::BarrierDescriptor> decses = sm.update_subresource(
            gfx::Texture::ResourceStateMachine::SubresourceRange{
                sub_entry.level_beg, sub_entry.level_end, sub_entry.mip_beg, sub_entry.mip_end },
            gfx::Texture::ResourceStateMachine::SubresourceState{
                sub_entry.stages, sub_entry.access, sub_entry.layout });
          for (auto const& desc : decses)
            graph->m_barriers[res.second.m_cosumeHistories[i].passID].emplace_back(desc);
        }
      }
      res.second.m_endState = sm;
    }

    for (auto& barrier : graph->m_barriers) {
      tryMergeBarriers(barrier.second);
    }
  }

  auto Graph::build() noexcept -> void {
    m_renderData.m_graph = this;

    std::optional<std::vector<size_t>> flatten = flatten_bfs(m_dag);
    if (!flatten.has_value()) {
      se::error("SE::RDG build error.");
      return;
    }
    m_flattenedPasses = flatten.value();

    // Find the resource
    size_t resourceID = 0;
    for (size_t i = 0; i < m_flattenedPasses.size(); ++i) {
      Pass* pass = m_passes[m_flattenedPasses[i]];

      // devirtualize internal resources
      for (auto& internal: pass->m_pReflection.m_internalResources) {
        if (internal.second.m_type == ResourceInfo::Type::Texture) {
          size_t rid = resourceID++;
          m_textureResources[rid] = TextureResource{};
          m_textureResources[rid].m_desc = toTextureDescriptor(internal.second.m_info.texture, m_standardSize);
          m_textureResources[rid].m_name = "RDG::" + pass->m_identifier + "::" + internal.first;
          m_textureResources[rid].m_cosumeHistories.push_back({ m_flattenedPasses[i], internal.second.m_info.texture.m_consumeHistories });
          internal.second.m_devirtualizeID = rid;
        }
        else if (internal.second.m_type == ResourceInfo::Type::Buffer) {
          size_t rid = resourceID++;
          m_bufferResources[rid] = BufferResource{};
          m_bufferResources[rid].m_desc = internal.second.m_info.buffer.to_descriptor();
          m_bufferResources[rid].m_name = "RDG::" + pass->m_identifier + "::" + internal.first;
          m_bufferResources[rid].m_cosumeHistories.push_back({ m_flattenedPasses[i], internal.second.m_info.buffer.m_consumeHistories });
          internal.second.m_devirtualizeID = rid;
        }
      }

      // devirtualize output resources
      for (auto& output : pass->m_pReflection.m_outputResources) {
        if (output.second.m_type == ResourceInfo::Type::Texture) {
          size_t rid = resourceID++;
          m_textureResources[rid] = TextureResource{};
          m_textureResources[rid].m_desc = toTextureDescriptor(output.second.m_info.texture, m_standardSize);
          m_textureResources[rid].m_name = "RDG::" + pass->m_identifier + "::" + output.first;
          m_textureResources[rid].m_cosumeHistories.push_back({ m_flattenedPasses[i], output.second.m_info.texture.m_consumeHistories });
          output.second.m_devirtualizeID = rid;
          if (output.second.m_info.texture.m_reference.get() != nullptr)
            m_textureResources[rid].m_texture = output.second.m_info.texture.m_reference;
        }
        else if (output.second.m_type == ResourceInfo::Type::Buffer) {
          size_t rid = resourceID++;
          m_bufferResources[rid] = BufferResource{};
          m_bufferResources[rid].m_desc = output.second.m_info.buffer.to_descriptor();
          m_bufferResources[rid].m_name = "RDG::" + pass->m_identifier + "::" + output.first;
          m_bufferResources[rid].m_cosumeHistories.push_back({ m_flattenedPasses[i], output.second.m_info.buffer.m_consumeHistories });
          output.second.m_devirtualizeID = rid;
          if (output.second.m_info.buffer.m_reference.get() != nullptr)
            m_bufferResources[rid].m_buffer = output.second.m_info.buffer.m_reference;
        }
      }
      
      // devirtualize input resources
      for (auto& internal : pass->m_pReflection.m_inputResources) {
        if (internal.second.m_type == ResourceInfo::Type::Texture) {
          if (!internal.second.m_info.texture.m_reference.get()) {
            size_t rid = internal.second.m_prev->m_devirtualizeID;
            internal.second.m_devirtualizeID = rid;
            rhi::TextureDescriptor desc = toTextureDescriptor(internal.second.m_info.texture, m_standardSize);
            m_textureResources[rid].m_desc.usage |= desc.usage;
            m_textureResources[rid].m_cosumeHistories.push_back(
              { m_flattenedPasses[i], internal.second.m_info.texture.m_consumeHistories});
          } else {
            size_t rid = resourceID++;
            m_textureResources[rid] = TextureResource{};
            m_textureResources[rid].m_desc = toTextureDescriptor(internal.second.m_info.texture, m_standardSize);
            m_textureResources[rid].m_cosumeHistories.push_back(
              { m_flattenedPasses[i], internal.second.m_info.texture.m_consumeHistories});
            internal.second.m_devirtualizeID = rid;
            m_textureResources[rid].m_texture = internal.second.m_info.texture.m_reference;
          }
        } else if (internal.second.m_type == ResourceInfo::Type::Buffer) {
          size_t rid = internal.second.m_prev->m_devirtualizeID;
          internal.second.m_devirtualizeID = rid;
          m_bufferResources[rid].m_cosumeHistories.push_back({ m_flattenedPasses[i], internal.second.m_info.buffer.m_consumeHistories });
        }
      }
      
      // devirtualize input-output resources
      for (auto& inout : pass->m_pReflection.m_inputOutputResources) {
        if (inout.second.m_type == ResourceInfo::Type::Texture) {
          if (inout.second.m_prev == nullptr) {
            se::error("RDG::Graph::build() failed, input-output resource \"{0}\" in "
              "pass \"{1}\" has no source.",
              inout.first, m_passes[m_flattenedPasses[i]]->m_identifier);
          }
          size_t rid = inout.second.m_prev->m_devirtualizeID;
          inout.second.m_devirtualizeID = rid;
          rhi::TextureDescriptor desc = toTextureDescriptor(inout.second.m_info.texture, m_standardSize);
          m_textureResources[rid].m_desc.usage |= desc.usage;
          m_textureResources[rid].m_cosumeHistories.push_back(
            { m_flattenedPasses[i], inout.second.m_info.texture.m_consumeHistories});
        } else if (inout.second.m_type == ResourceInfo::Type::Buffer) {
          size_t rid = inout.second.m_prev->m_devirtualizeID;
          inout.second.m_devirtualizeID = rid;
          m_bufferResources[rid] = BufferResource{};
          m_bufferResources[rid].m_desc = inout.second.m_info.buffer.to_descriptor();
          m_bufferResources[rid].m_name = "RDG::" + pass->m_identifier + "::" + inout.first;
          m_bufferResources[rid].m_cosumeHistories.push_back({ m_flattenedPasses[i], inout.second.m_info.buffer.m_consumeHistories });
        }
      }
    }

    // devirtualize all the resources
    // 
    std::unique_ptr<rhi::CommandEncoder> commandEncoder =
      gfx::GFXContext::device()->create_command_encoder(nullptr);

    for (auto& res : m_bufferResources) {
      //
      if (res.second.m_buffer.get() == nullptr) {
        res.second.m_buffer = gfx::GFXContext::create_buffer_desc(res.second.m_desc);
        res.second.m_buffer->m_buffer->set_name(res.second.m_name);
        res.second.m_buffer->m_name = res.second.m_name;
        res.second.m_buffer->m_creator = "RDG \'" + m_name + "\'";
        res.second.m_buffer->m_job = "Buffer resource of RDG";
      }
    }
    generateBufferBarriers(this);

    for (auto& res : m_textureResources) {
      // initialize mip levels
      if (res.second.m_desc.mipLevelCount == uint32_t(-1)) {
        res.second.m_desc.mipLevelCount =
          std::log2(std::max(res.second.m_desc.size.x,
            res.second.m_desc.size.y)) + 1;
      }

      if (res.second.m_texture.get() == nullptr) {
        res.second.m_texture = gfx::GFXContext::create_texture_desc(res.second.m_desc);
        res.second.m_texture->m_texture->set_name(res.second.m_name);
        res.second.m_texture->m_name = res.second.m_name;
        res.second.m_texture->m_creator = "RDG \'" + m_name + "\'";
        res.second.m_texture->m_job = "Texture resource of RDG";
      }
    }

    generateTextureBarriers(this);

    // initialize resources
    for (auto& res : m_textureResources) {
      // transition from intiialize state to start state
      if (res.second.m_startState.has_value()) {
        auto const barriers = res.second.m_texture->m_stateMachine.transition(
          res.second.m_startState.value());
        for (auto& barrier : barriers) {
          commandEncoder->pipeline_barrier(barrier);
        }
      }
    }

    gfx::GFXContext::device()->get_graphics_queue().submit({ commandEncoder->finish() });
    gfx::GFXContext::device()->wait_idle();
  }

  auto Graph::execute(rhi::CommandEncoder* encoder) noexcept -> void {
    m_renderData.m_graph = this;
    se::rhi::FrameResources* flights = gfx::GFXContext::get_flights();
    RenderContext renderContext;
    renderContext.cmdEncoder = encoder;
    renderContext.flightIdx = (flights == nullptr) ? 0 : flights->get_flight_index();

    std::vector<size_t> marker_stack;
    for (auto& res : m_textureResources) {
      // transition from intiialize state to start state
      if (res.second.m_startState.has_value()) {
        auto const barriers = res.second.m_texture->m_stateMachine.transition(
          res.second.m_startState.value());
        for (auto& barrier : barriers) {
          encoder->pipeline_barrier(barrier);
        }
      }
    }

    for (size_t pass_id : m_flattenedPasses) {
      auto* pass = m_passes[pass_id];
      m_renderData.m_pass = pass;
      // insert barriers
      for (auto const& barriers : m_barriers[pass_id]) {
        renderContext.cmdEncoder->pipeline_barrier(barriers);
      }

      //{  
      //  // deal with subgroup markers
      //  if (marker_stack.empty() && !pass->m_subgraphStack.empty()) {
      //    for (auto const& subgraph_id : pass->m_subgraphStack) {
      //      encoder->end_debug_utils_label(
      //        subgraphs[subgraphNameList[subgraph_id]]->marker);
      //      marker_stack.push_back(subgraph_id);
      //    }
      //  }
      //  else if (!marker_stack.empty() && pass->subgraphStack.empty()) {
      //    while (marker_stack.size() > 0) {
      //      marker_stack.pop_back();
      //      encoder->endDebugUtilsLabelEXT();
      //    }
      //  }
      //  else if (!marker_stack.empty() &&
      //    marker_stack.back() != pass->subgraphStack.back()) {
      //    size_t offset = 0;
      //    while (offset < marker_stack.size() &&
      //      offset < pass->subgraphStack.size() &&
      //      marker_stack[offset] == pass->subgraphStack[offset]) {
      //      offset++;
      //    }
      //    while (marker_stack.size() > offset) {
      //      marker_stack.pop_back();
      //      encoder->endDebugUtilsLabelEXT();
      //    }
      //    for (size_t i = offset; i < pass->subgraphStack.size(); ++i) {
      //      encoder->beginDebugUtilsLabelEXT(
      //        subgraphs[subgraphNameList[pass->subgraphStack[i]]]->marker);
      //      marker_stack.push_back(pass->subgraphStack[i]);
      //    }
      //  }
      //}

      encoder->begin_debug_utils_label(pass->m_marker.name, pass->m_marker.color);
      pass->execute(&renderContext, m_renderData);
      encoder->end_debug_utils_label();
    }

    //while (marker_stack.size() > 0) {
    //  marker_stack.pop_back();
    //  encoder->endDebugUtilsLabelEXT();
    //}

    // setting the status of resources
    for (auto& res : m_textureResources) {
      // transition from intiialize state to start state
      if (res.second.m_endState.has_value()) {
        res.second.m_texture->m_stateMachine = res.second.m_endState.value();
      }
    }

    //{
    //  // insert barriers
    //  for (auto const& barriers : m_barriers[size_t(-1)])
    //    renderContext.cmdEncoder->pipeline_barrier(barriers);
    //}
  }

  auto Graph::on_draw_inspector() noexcept -> void {
    if (ImGui::Button("Open graph viewer")) {
      m_openEditor = true;
    }

    on_draw_gui();
    render_ui();
    for (auto index : m_flattenedPasses) {
      if (ImGui::TreeNodeEx(m_passes[index]->m_identifier.c_str())) {

        m_passes[index]->render_ui();
        ImGui::TreePop();
      }
    }
  }

	auto Graph::on_draw_gui() noexcept -> void {
    if (!m_openEditor) return;

    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_Once);
    ImGui::SetNextWindowCollapsed(false, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_Once);
		ImGui::Begin("RDG (render dependency graph) editor", &m_openEditor, ImGuiWindowFlags_MenuBar);
    // Menu
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("Add pass")) {
        if (ImGui::MenuItem("Compute pass")) {

        }
        if (ImGui::MenuItem("Raster pass")) {

        }
        ImGui::EndMenu();
      }

      ImGui::EndMenuBar();
    }

		ImNodes::BeginNodeEditor();

    for (auto& vertex: m_flattenedPasses) {
      Pass* pass = m_passes[vertex];
      ImNodes::BeginNode(vertex);
      ImNodes::BeginNodeTitleBar();
      ImGui::TextUnformatted(pass->m_identifier.c_str());
      ImNodes::EndNodeTitleBar();

      // show reflection
      auto& inputs = pass->m_pReflection.m_inputResources;
      for (auto& pair : inputs) {
        ImNodes::BeginInputAttribute(pair.second.m_resoureceID);
        ImGui::Text("%s", pair.first.c_str());
        ImNodes::EndInputAttribute();
      }

      auto& outputs = pass->m_pReflection.m_outputResources;
      for (auto& pair : outputs) {
        ImNodes::BeginOutputAttribute(pair.second.m_resoureceID);
        ImGui::Text("%s", pair.first.c_str());
        ImNodes::EndOutputAttribute();
      }

      ImNodes::EndNode();
    }

    size_t edge_index = 0;
    for (auto& edge : m_edges) {
      ImNodes::Link(edge_index++, edge.first, edge.second);
    }

    std::optional<uint32_t> output_idx = get_output_index();
    if (output_idx.has_value()) {
      ImNodes::BeginNode(uint32_t(-2));
      ImNodes::BeginNodeTitleBar();
      ImGui::TextUnformatted("Output");
      ImNodes::EndNodeTitleBar();

      ImNodes::BeginInputAttribute(uint32_t(-1));
      ImGui::Text("output texture");
      ImNodes::EndInputAttribute();

      ImNodes::EndNode();

      //gh_imgui.imnodes_link(link1_2, node1_output1, node2_input1)
      ImNodes::Link(uint32_t(-3), output_idx.value(), uint32_t(-1));
    }

    ImNodes::MiniMap();
		ImNodes::EndNodeEditor();

		ImGui::End();

	}
}
};