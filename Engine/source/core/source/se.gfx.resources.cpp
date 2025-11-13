#include "se.gfx.hpp"
#include "se.utils.hpp"
#include "spirvreflect/spirv_reflect.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <slang.h>
#include <slang-com-ptr.h>
#include <filesystem>
#include "se.editor.hpp"
#include <algorithm>
#include "se.rdg.hpp"

namespace slang_inline {
  using ::Slang::ComPtr;
  using namespace se;

  struct SlangSession {
    SlangSession(
      std::string const& filepath,
      std::vector<std::pair<char const*, char const*>> const& macros = {},
      bool use_glsl_intermediate = false);

    auto load(std::vector<std::pair<std::string, se::rhi::ShaderStageEnum>> const&
      entrypoints) noexcept -> std::vector<se::gfx::ShaderHandle>;

    std::unordered_map<std::string, se::gfx::ShaderReflection::BindingInfo> bindingInfo;
  private:
    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    slang::IModule* slangModule = nullptr;
    ComPtr<slang::ISession> session;
    bool use_glsl_intermediate;
    std::string filepath;
  };

  struct SlangManager {
    SINGLETON(SlangManager, {
      SlangResult result = slang::createGlobalSession(slangGlobalSession.writeRef());
      if (result != 0)
        se::error("GFX::SLANG::Global session create failed.");
      });
    auto getGlobalSession() noexcept -> slang::IGlobalSession* {
      return slangGlobalSession.get();
    }
    ComPtr<slang::IGlobalSession> slangGlobalSession;
  };

  inline void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob) {
    if (diagnosticsBlob != nullptr) {
      //if()
      std::string error_message = std::string((const char*)diagnosticsBlob->getBufferPointer());
      //if (error_message.find("capabilities: 'spvRayQueryKHR'")) return;
      se::error(error_message);
    }
  }

  SlangSession::SlangSession(
    std::string const& filepath,
    std::vector<std::pair<char const*, char const*>> const& macros,
    bool use_glsl_intermediate)
    : use_glsl_intermediate(use_glsl_intermediate), filepath(filepath) {
    std::filesystem::path path(filepath);
    SlangManager* manager = Singleton<SlangManager>::instance();
    slang::IGlobalSession* globalSession = manager->getGlobalSession();
    if (use_glsl_intermediate) {
      // set target to spirv glsl460
      targetDesc.format = SLANG_GLSL;
      targetDesc.profile = globalSession->findProfile("glsl460");
    }
    else {
      // set target to spirv glsl460
      targetDesc.format = SLANG_SPIRV;
      targetDesc.profile = globalSession->findProfile("spirv_1_5");
      targetDesc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    }

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    // set search path
    std::string parent_path = path.parent_path().string();
    auto const& engine_shader_path = Configuration::string_array_property("shader_path");
    std::string found_path = Filesys::get_parent_path(Filesys::resolve_path(filepath, engine_shader_path));
    std::vector<const char*> found_path_cstr = { found_path.c_str() };
    for (auto const& path : engine_shader_path)
      found_path_cstr.emplace_back(path.c_str());

    sessionDesc.searchPaths = found_path_cstr.data();
    sessionDesc.searchPathCount = found_path_cstr.size();

    // push pre-defined macros
    std::vector<slang::PreprocessorMacroDesc> macro_list;
    for (auto const& macro : macros)
      macro_list.emplace_back(
        slang::PreprocessorMacroDesc{ macro.first, macro.second });
    sessionDesc.preprocessorMacros = macro_list.data();
    sessionDesc.preprocessorMacroCount = macro_list.size();
    // create slang session
    SlangResult result =
      globalSession->createSession(sessionDesc, session.writeRef());
    if (result != 0) {
      se::error("GFX::SLANG::Session create failed.");
      return;
    }
    // load module
    ComPtr<slang::IBlob> diagnosticBlob;
    std::string stem = path.stem().string();
    slangModule = session->loadModule(stem.c_str(),
      diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
    if (!slangModule) return;

    slang::ShaderReflection* shaderReflection = slangModule->getLayout();

    unsigned parameterCount = shaderReflection->getParameterCount();
    for (unsigned pp = 0; pp < parameterCount; pp++) {
      se::gfx::ShaderReflection::BindingInfo bindinfo = {
          se::gfx::ShaderReflection::ResourceType::Undefined, 0, 0 };
      slang::VariableLayoutReflection* parameter =
        shaderReflection->getParameterByIndex(pp);
      char const* parameterName = parameter->getName();
      slang::ParameterCategory category = parameter->getCategory();
      unsigned index = parameter->getBindingIndex();
      unsigned space = parameter->getBindingSpace();
      bindinfo.binding = index;
      bindinfo.set = space;
      // get array size
      slang::TypeLayoutReflection* typeLayout = parameter->getTypeLayout();
      slang::TypeReflection::Kind kind = typeLayout->getKind();

      size_t arrayCount = 1;
      slang::TypeReflection* param_type = parameter->getTypeLayout()->getType();
      if (param_type->getKind() == slang::TypeReflection::Kind::Array) {
        arrayCount = param_type->getElementCount(); // <— this is what you want
      }

      switch (kind) {
      case slang::TypeReflection::Kind::None:
        break;
      case slang::TypeReflection::Kind::Struct:
        break;
      case slang::TypeReflection::Kind::Array:
        break;
      case slang::TypeReflection::Kind::Matrix:
        break;
      case slang::TypeReflection::Kind::Vector:
        break;
      case slang::TypeReflection::Kind::Scalar:
        break;
      case slang::TypeReflection::Kind::ConstantBuffer:
        break;
      case slang::TypeReflection::Kind::Resource:
        break;
      case slang::TypeReflection::Kind::SamplerState:
        break;
      case slang::TypeReflection::Kind::TextureBuffer:
        break;
      case slang::TypeReflection::Kind::ShaderStorageBuffer:
        break;
      case slang::TypeReflection::Kind::ParameterBlock:
        break;
      case slang::TypeReflection::Kind::GenericTypeParameter:
        break;
      case slang::TypeReflection::Kind::Interface:
        break;
      case slang::TypeReflection::Kind::OutputStream:
        break;
      case slang::TypeReflection::Kind::Specialized:
        break;
      case slang::TypeReflection::Kind::Feedback:
        break;
      default:
        break;
      }
      slang::BindingType type = typeLayout->getDescriptorSetDescriptorRangeType(0, 0);
      switch (type) {
      case slang::BindingType::PushConstant:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::Undefined;
        break;
      case slang::BindingType::Unknown:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::Undefined;
        break;
      case slang::BindingType::CombinedTextureSampler:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::SampledImages;
        break;
      case slang::BindingType::RayTracingAccelerationStructure:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::AccelerationStructure;
        bindinfo.count = arrayCount;
        break;
      case slang::BindingType::ConstantBuffer:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::UniformBuffer;
        break;
      case slang::BindingType::RawBuffer:
      case slang::BindingType::MutableRawBuffer:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::StorageBuffer;
        break;
      case slang::BindingType::MutableTexture:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::StorageImages;
        break;
      case slang::BindingType::Texture:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::ReadonlyImage;
        break;
      case slang::BindingType::Sampler:
        bindinfo.type = se::gfx::ShaderReflection::ResourceType::Sampler;
        break;
      case slang::BindingType::ParameterBlock:
      case slang::BindingType::TypedBuffer:
      case slang::BindingType::InputRenderTarget:
      case slang::BindingType::InlineUniformData:
      case slang::BindingType::VaryingInput:
      case slang::BindingType::VaryingOutput:
      case slang::BindingType::ExistentialValue:
      case slang::BindingType::MutableFlag:
      case slang::BindingType::MutableTypedBuffer:
      case slang::BindingType::BaseMask:
      case slang::BindingType::ExtMask:
      default: se::error("SLANG :: Binding not valid");
        break;
      }
      bindingInfo[std::string(parameterName)] = bindinfo;
    }
  }

  auto SlangSession::load(
    std::vector<std::pair<std::string, se::rhi::ShaderStageEnum>> const&
    entrypoints) noexcept -> std::vector<se::gfx::ShaderHandle> {
    std::vector<se::gfx::ShaderHandle> sms(entrypoints.size());
    // add all entrypoints
    std::vector<ComPtr<slang::IEntryPoint>> entryPointsPtrs(entrypoints.size());
    std::vector<slang::IComponentType*> componentTypes;
    componentTypes.push_back(slangModule);
    for (size_t i = 0; i < entrypoints.size(); ++i) {

      SlangInt32 w = slangModule->getDefinedEntryPointCount();
      SlangResult result = slangModule->findEntryPointByName(
        entrypoints[i].first.c_str(), entryPointsPtrs[i].writeRef());
      if (result != 0) {
        se::error("GFX::SLANG::cannot find entrypoint \"" + entrypoints[i].first + "\"");
        return sms;
      }
      componentTypes.push_back(entryPointsPtrs[i]);
    }

    // compile the session
    ComPtr<slang::IBlob> diagnosticBlob;
    ComPtr<slang::IComponentType> composedProgram;
    SlangResult result = session->createCompositeComponentType(
      componentTypes.data(), componentTypes.size(), composedProgram.writeRef(),
      diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
    if (result != 0) {
      se::error("GFX::SLANG::createCompositeComponentType() failed.");
      return sms;
    }
    ComPtr<slang::IBlob> compiledCode;
    for (size_t i = 0; i < entrypoints.size(); ++i) {
      ComPtr<slang::IBlob> diagnosticsBlob;

      SlangResult result = composedProgram->getEntryPointCode(
        i, 0, compiledCode.writeRef(), diagnosticsBlob.writeRef());
      diagnoseIfNeeded(diagnosticsBlob);
      if (result != 0) {
        se::error("GFX::SLANG::getEntryPointCode() failed.");
        return sms;
      }
      se::MiniBuffer spirvcode;
      if (use_glsl_intermediate) {
        // compile SPIR-V from glsl
        se::MiniBuffer glslcode;
        glslcode.m_isReference = true;
        glslcode.m_data = (void*)(compiledCode->getBufferPointer());
        glslcode.m_size = compiledCode->getBufferSize();
        std::string suffix = "glsl";
        switch (entrypoints[i].second) {
        case se::rhi::ShaderStageEnum::VERTEX:       suffix = "vert"; break;
        case se::rhi::ShaderStageEnum::FRAGMENT:     suffix = "frag"; break;
        case se::rhi::ShaderStageEnum::COMPUTE:      suffix = "comp"; break;
        case se::rhi::ShaderStageEnum::GEOMETRY:     suffix = "geom"; break;
        case se::rhi::ShaderStageEnum::RAYGEN:       suffix = "rgen"; break;
        case se::rhi::ShaderStageEnum::MISS:         suffix = "rmiss"; break;
        case se::rhi::ShaderStageEnum::CLOSEST_HIT:  suffix = "rchit"; break;
        case se::rhi::ShaderStageEnum::INTERSECTION: suffix = "rint"; break;
        case se::rhi::ShaderStageEnum::ANY_HIT:      suffix = "rahit"; break;
        case se::rhi::ShaderStageEnum::CALLABLE:     suffix = "rcall"; break;
        case se::rhi::ShaderStageEnum::TASK:         suffix = "task"; break;
        case se::rhi::ShaderStageEnum::MESH:         suffix = "mesh"; break;
        default: break;
        }
        std::string glsl_path = filepath.substr(0, filepath.find_last_of('.') + 1) + suffix;
        se::Filesys::sync_write_file(glsl_path.c_str(), glslcode);
        // create shader module
        //sms[i] = se::gfx::GFXContext::load_shader_glsl(glsl_path.c_str(), entrypoints[i].second);
      }
      else {
        // directly use the compiled SPIR-V
        spirvcode.m_isReference = true;
        spirvcode.m_data = (void*)(compiledCode->getBufferPointer());
        spirvcode.m_size = compiledCode->getBufferSize();
        // create shader module
        sms[i] = se::gfx::GFXContext::load_shader_spirv(&spirvcode, entrypoints[i].second);
      }
    }
    for (size_t i = 0; i < entrypoints.size(); ++i) {
      sms[i].get()->m_reflection.bindingInfo = bindingInfo;
    }
    return sms;
  }
}

namespace se {
namespace gfx {
  auto GFXContext::initialize(Window* window, Flags<rhi::ContextExtensionEnum> ext) noexcept -> void {
    se::init_extensions();
    Singleton<GFXContext>::instance()->m_ctx = std::make_unique<se::rhi::Context>(window, ext);
    Singleton<GFXContext>::instance()->m_adapter = Singleton<GFXContext>::instance()->m_ctx->request_adapter();
    Singleton<GFXContext>::instance()->m_device = Singleton<GFXContext>::instance()->m_adapter->request_device();
  }

  auto GFXContext::device() noexcept -> rhi::Device* {
    return Singleton<GFXContext>::instance()->m_device.get();
  }

  auto GFXContext::create_flights(int maxFlightNum, rhi::SwapChain* swapchain) -> void {
    Singleton<GFXContext>::instance()->m_flights = device()->create_frame_resources(
      maxFlightNum, swapchain);
  }

  auto GFXContext::get_flights() -> rhi::FrameResources* {
    return Singleton<GFXContext>::instance()->m_flights.get();
  }

  auto GFXContext::finalize() noexcept -> void {
    //buffers.clear();
    //meshs.clear();
    Singleton<GFXContext>::instance()->m_scenes.clear();
    Singleton<GFXContext>::instance()->m_textures.clear();
    Singleton<GFXContext>::instance()->m_samplers.clear();
    Singleton<GFXContext>::instance()->m_shaders.clear();
    Singleton<GFXContext>::instance()->m_meshs.clear();
    Singleton<GFXContext>::instance()->m_materials.clear();
    Singleton<GFXContext>::instance()->m_mediums.clear();
    Singleton<GFXContext>::instance()->m_buffers.clear();
    // release the base objects
    Singleton<GFXContext>::instance()->m_flights = nullptr;
    Singleton<GFXContext>::instance()->m_adapter = nullptr;
    Singleton<GFXContext>::instance()->m_device = nullptr;
  }

  template<class T>
  std::string enum_flags_to_string(Flags<T> entry) {
    constexpr auto entries = magic_enum::enum_entries<T>();
    std::string output = "";
    for (auto& iter : entries) {
      if (entry & iter.first) {
        output += "| " + std::string(iter.second);
      }
    }
    return (output == "") ? "None" : output;
  }

  TextureLoader::result_type TextureLoader::operator()(from_desc_tag, rhi::TextureDescriptor const& desc) {
    TextureLoader::result_type result = std::make_shared<Texture>();
    result->m_texture = GFXContext::device()->create_texture(desc);
    return result;
  }

  TextureLoader::result_type TextureLoader::operator()(from_file_tag, std::string const& path) {
    TextureLoader::result_type result = std::make_shared<Texture>();
    std::unique_ptr<image::Image> host_tex = image::load_image(path);
    // create staging buffer
    rhi::BufferDescriptor stagingBufferDescriptor;
    stagingBufferDescriptor.size = host_tex->m_dataSize;
    stagingBufferDescriptor.usage = rhi::BufferUsageEnum::COPY_SRC;
    stagingBufferDescriptor.memoryProperties =
      rhi::MemoryPropertyEnum::HOST_VISIBLE_BIT |
      rhi::MemoryPropertyEnum::HOST_COHERENT_BIT;
    stagingBufferDescriptor.mappedAtCreation = true;
    std::unique_ptr<rhi::Buffer> stagingBuffer = GFXContext::device()->create_buffer(stagingBufferDescriptor);
    std::future<bool> mapped = stagingBuffer->map_async(0, 0, stagingBufferDescriptor.size);
    if (mapped.get()) {
      void* mapdata = stagingBuffer->get_mapped_range(0);
      memcpy(mapdata, host_tex->get_data(), (size_t)stagingBufferDescriptor.size);
      stagingBuffer->unmap();
    }
    std::unique_ptr<rhi::CommandEncoder> commandEncoder = GFXContext::device()->create_command_encoder(nullptr);
    // create texture image
    result->m_texture = GFXContext::device()->create_texture(host_tex->get_descriptor());
    result->m_resourcePath = { path };
    // copy to target
    commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
      rhi::PipelineStageEnum::TOP_OF_PIPE_BIT,
      rhi::PipelineStageEnum::TRANSFER_BIT,
      rhi::DependencyTypeEnum::NONE,
      {}, {}, { rhi::TextureMemoryBarrierDescriptor{
        result->m_texture.get(),
        rhi::TextureRange{rhi::TextureAspectEnum::COLOR_BIT, 0,
                          host_tex->m_mipLevels, 0, host_tex->m_arrayLayers},
        rhi::AccessFlagEnum::NONE,
        rhi::AccessFlagEnum::TRANSFER_WRITE_BIT,
        rhi::TextureLayoutEnum::UNDEFINED,
        rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL}} });

    for (auto const& subresource : host_tex->m_subResources) {
      commandEncoder->copy_buffer_to_texture(
        { subresource.offset, 0, 0, stagingBuffer.get() },
        { result->m_texture.get(),
        subresource.mip,
        {},
        rhi::TextureAspectEnum::COLOR_BIT },
        { subresource.width, subresource.height, 1 });
    }

    commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
        rhi::PipelineStageEnum::TRANSFER_BIT,
        rhi::PipelineStageEnum::FRAGMENT_SHADER_BIT,
        rhi::DependencyTypeEnum::NONE,
        {}, {},
        {rhi::TextureMemoryBarrierDescriptor{
            result->m_texture.get(),
            rhi::TextureRange{rhi::TextureAspectEnum::COLOR_BIT, 0,
                                       host_tex->m_mipLevels, 0, host_tex->m_arrayLayers},
            rhi::AccessFlagEnum::TRANSFER_WRITE_BIT,
            rhi::AccessFlagEnum::SHADER_READ_BIT,
            rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL,
            rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL}} });

    GFXContext::device()->get_graphics_queue().submit({commandEncoder->finish()});
    GFXContext::device()->wait_idle();
    return result;
  }
  
  TextureLoader::result_type TextureLoader::operator()(TextureLoader::from_binary_tag, int width, int height, int channel, int bits, const char* data) {
    TextureLoader::result_type result = std::make_shared<Texture>();
    std::unique_ptr<image::Image> host_tex = image::Binary::from_binary(width, height, channel, bits, data);
    // create staging buffer
    rhi::BufferDescriptor stagingBufferDescriptor;
    stagingBufferDescriptor.size = host_tex->m_dataSize;
    stagingBufferDescriptor.usage = (uint32_t)rhi::BufferUsageEnum::COPY_SRC;
    stagingBufferDescriptor.memoryProperties =
      (uint32_t)rhi::MemoryPropertyEnum::HOST_VISIBLE_BIT |
      (uint32_t)rhi::MemoryPropertyEnum::HOST_COHERENT_BIT;
    stagingBufferDescriptor.mappedAtCreation = true;
    std::unique_ptr<rhi::Buffer> stagingBuffer = GFXContext::device()->create_buffer(stagingBufferDescriptor);
    std::future<bool> mapped = stagingBuffer->map_async(0, 0, stagingBufferDescriptor.size);
    if (mapped.get()) {
      void* mapdata = stagingBuffer->get_mapped_range(0);
      memcpy(mapdata, host_tex->get_data(), (size_t)stagingBufferDescriptor.size);
      stagingBuffer->unmap();
    }
    std::unique_ptr<rhi::CommandEncoder> commandEncoder = GFXContext::device()->create_command_encoder({ nullptr });
    // create texture image
    result->m_texture = GFXContext::device()->create_texture(host_tex->get_descriptor());
    // copy to target
    commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
      (uint32_t)rhi::PipelineStageEnum::TOP_OF_PIPE_BIT,
      (uint32_t)rhi::PipelineStageEnum::TRANSFER_BIT,
      (uint32_t)rhi::DependencyTypeEnum::NONE,
      {}, {}, { rhi::TextureMemoryBarrierDescriptor{
        result->m_texture.get(),
        rhi::TextureRange{(uint32_t)rhi::TextureAspectEnum::COLOR_BIT, 0,
          host_tex->m_mipLevels, 0, host_tex->m_arrayLayers},
        (uint32_t)rhi::AccessFlagEnum::NONE,
        (uint32_t)rhi::AccessFlagEnum::TRANSFER_WRITE_BIT,
        rhi::TextureLayoutEnum::UNDEFINED,
        rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL}} });

    for (auto const& subresource : host_tex->m_subResources) {
      commandEncoder->copy_buffer_to_texture(
        { subresource.offset, 0, 0, stagingBuffer.get() },
        { result->m_texture.get(),
        subresource.mip,
        {},
        (uint32_t)rhi::TextureAspectEnum::COLOR_BIT },
        { subresource.width, subresource.height, 1 });
    }

    commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
        (uint32_t)rhi::PipelineStageEnum::TRANSFER_BIT,
        (uint32_t)rhi::PipelineStageEnum::FRAGMENT_SHADER_BIT,
        (uint32_t)rhi::DependencyTypeEnum::NONE,
        {}, {},
        {rhi::TextureMemoryBarrierDescriptor{
            result->m_texture.get(),
            rhi::TextureRange{(uint32_t)rhi::TextureAspectEnum::COLOR_BIT, 0,
              host_tex->m_mipLevels, 0, host_tex->m_arrayLayers},
            (uint32_t)rhi::AccessFlagEnum::TRANSFER_WRITE_BIT,
            (uint32_t)rhi::AccessFlagEnum::SHADER_READ_BIT,
            rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL,
            rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL}} });

    GFXContext::device()->get_graphics_queue().submit({commandEncoder->finish()});
    GFXContext::device()->wait_idle();
    return result;
  }

  auto GFXContext::create_buffer_empty(
  ) noexcept -> BufferHandle {
    UID const ruid = Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_buffers.load(
      ruid, BufferLoader::from_empty_tag{});
    return BufferHandle{ ret.first->second };
  }

  auto GFXContext::create_buffer_desc(
    rhi::BufferDescriptor const& desc
  ) noexcept -> BufferHandle {
    UID const ruid = Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_buffers.load(
      ruid, BufferLoader::from_desc_tag{}, desc);
    return BufferHandle{ ret.first->second };
  }

  auto GFXContext::create_buffer_host(
    MiniBuffer const& buffer,
    Flags<rhi::BufferUsageEnum> usages
  ) noexcept -> BufferHandle {
    UID const ruid = Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_buffers.load(
      ruid, BufferLoader::from_host_tag{}, buffer, usages);
    return BufferHandle{ ret.first->second };
  }

  auto Texture::save_image(std::string const& path) noexcept -> void {
    size_t width = m_texture->width();
    size_t height = m_texture->height();

    rhi::TextureFormat format;
    size_t pixelSize;
    if (m_texture->format() == rhi::TextureFormat::RGBA32_FLOAT) {
      format = rhi::TextureFormat::RGBA32_FLOAT;
      pixelSize = sizeof(se::vec4);
    }
    else if (m_texture->format() == rhi::TextureFormat::RGBA8_UNORM) {
      format = rhi::TextureFormat::RGBA8_UNORM;
      pixelSize = sizeof(uint8_t) * 4;
    }
    else {
      se::error(
        "Editor :: ViewportWidget :: captureImage() :: Unsupported format to "
        "capture."); return;
    }

  std::unique_ptr<rhi::CommandEncoder> commandEncoder =
    gfx::GFXContext::device()->create_command_encoder({});

  TextureHandle copyDst{};
  if (copyDst.get() == nullptr) {
    rhi::TextureDescriptor desc{
      {uint32_t(width), uint32_t(height), 1}, 1, 1, 1,
      rhi::TextureDimension::TEX2D, format,
      (uint32_t)rhi::TextureUsageEnum::COPY_DST |
      (uint32_t)rhi::TextureUsageEnum::TEXTURE_BINDING,
      {format}, (uint32_t)rhi::TextureFeatureEnum::HOST_VISIBLE };
    copyDst = GFXContext::create_texture_desc(desc);
    commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
      (uint32_t)rhi::PipelineStageEnum::ALL_GRAPHICS_BIT,
      (uint32_t)rhi::PipelineStageEnum::TRANSFER_BIT,
      (uint32_t)rhi::DependencyTypeEnum::NONE,
      {}, {}, {rhi::TextureMemoryBarrierDescriptor{
        copyDst->m_texture.get(),
        rhi::TextureRange{(uint32_t)rhi::TextureAspectEnum::COLOR_BIT, 0, 1, 0, 1},
        (uint32_t)rhi::AccessFlagEnum::NONE,
        (uint32_t)rhi::AccessFlagEnum::TRANSFER_WRITE_BIT,
        rhi::TextureLayoutEnum::UNDEFINED,
        rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL}} });
  }
  gfx::GFXContext::device()->wait_idle();

  //gfx::Texture::ResourceStateMachine::SubresourceRange{
  //  entry.level_beg, entry.level_end, entry.mip_beg, entry.mip_end },
  //  gfx::Texture::ResourceStateMachine::SubresourceState{
  //      entry.stages, entry.access, entry.layout });

  std::vector<rhi::BarrierDescriptor> barriers = consume(ConsumeEntry()
    .add_stage((uint32_t)rhi::PipelineStageEnum::TRANSFER_BIT)
    .set_layout(rhi::TextureLayoutEnum::TRANSFER_SRC_OPTIMAL)
    .set_access((uint32_t)rhi::AccessFlagEnum::TRANSFER_READ_BIT)
  );
  for (auto& barrier : barriers)
    commandEncoder->pipeline_barrier(barrier);
  //commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
  //  (uint32_t)rhi::PipelineStageEnum::FRAGMENT_SHADER_BIT,
  //  (uint32_t)rhi::PipelineStageEnum::TRANSFER_BIT,
  //  (uint32_t)rhi::DependencyTypeEnum::NONE,
  //  {}, {}, { rhi::TextureMemoryBarrierDescriptor{
  //    m_texture.get(),
  //    rhi::TextureRange{(uint32_t)rhi::TextureAspectEnum::COLOR_BIT, 0, 1, 0, 1},
  //    (uint32_t)rhi::AccessFlagEnum::SHADER_READ_BIT |
  //    (uint32_t)rhi::AccessFlagEnum::SHADER_WRITE_BIT,
  //    (uint32_t)rhi::AccessFlagEnum::TRANSFER_READ_BIT,
  //    rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL,
  //    rhi::TextureLayoutEnum::TRANSFER_SRC_OPTIMAL}} });
  commandEncoder->copy_texture_to_texture(
    rhi::ImageCopyTexture{ m_texture.get() },
    rhi::ImageCopyTexture{ copyDst->m_texture.get() },
    se::uvec3{ uint32_t(width), uint32_t(height), 1 });
  commandEncoder->pipeline_barrier(rhi::BarrierDescriptor{
    (uint32_t)rhi::PipelineStageEnum::TRANSFER_BIT,
    (uint32_t)rhi::PipelineStageEnum::HOST_BIT,
    (uint32_t)rhi::DependencyTypeEnum::NONE,
    {}, {}, { rhi::TextureMemoryBarrierDescriptor{
      copyDst->m_texture.get(),
      rhi::TextureRange{(uint32_t)rhi::TextureAspectEnum::COLOR_BIT, 0, 1, 0, 1},
      (uint32_t)rhi::AccessFlagEnum::TRANSFER_WRITE_BIT,
      (uint32_t)rhi::AccessFlagEnum::HOST_READ_BIT,
      rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL,
      rhi::TextureLayoutEnum::TRANSFER_DST_OPTIMAL}} });

  gfx::GFXContext::device()->get_graphics_queue().submit(
    { commandEncoder->finish() });
  gfx::GFXContext::device()->wait_idle();
  std::future<bool> mapped = copyDst->m_texture->map_async((uint32_t)rhi::MapModeEnum::READ, 0,
    width * height * pixelSize);
  if (mapped.get()) {
    void* data = copyDst->m_texture->get_mapped_range(0, width * height * pixelSize);
    if (m_texture->format() == rhi::TextureFormat::RGBA32_FLOAT) {
      image::EXR::write_exr(path, width, height, 4,
        reinterpret_cast<float*>(data));
    }
    else if (m_texture->format() == rhi::TextureFormat::RGBA8_UNORM) {
      //std::string filepath = mainWindow->saveFile(
      //    "", Core::WorldTimePoint::get().to_string() + ".bmp");
      //Image::BMP::writeBMP(filepath, width, height, 4,
      //    reinterpret_cast<float*>(data));
    }
    copyDst->m_texture->unmap();
  }
  }

  auto GFXContext::create_texture_desc(
    rhi::TextureDescriptor const& desc
  ) noexcept -> TextureHandle {
    UID const ruid = Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_textures.load(
      ruid, TextureLoader::from_desc_tag{}, desc);
    entt::resource<Texture> res = ret.first->second;
    res->m_uid = ruid;
    res->init();
    return TextureHandle{ ret.first->second };
  }

  auto GFXContext::load_texture_file(
    std::string const& path
  ) noexcept -> TextureHandle {
    std::string abs_path = Filesys::resolve_path(path, {
      Configuration::string_property("engine_path"),
      Configuration::string_property("project_path")
    });
    UID const ruid = Resources::query_string_uid(abs_path);
    auto ret = Singleton<GFXContext>::instance()->m_textures.load(
      ruid, TextureLoader::from_file_tag{}, abs_path);
    // true only if the resource was not already present
    const bool loaded = ret.second;
    // takes the resource handle pointed to by the returned iterator
    entt::resource<Texture> res = ret.first->second;
    res->m_uid = ruid;
    res->init();
    return TextureHandle{ ret.first->second };
  }

  auto GFXContext::load_texture_binary(
    int width, int height, int channel,
    int bits, const char* data
  ) noexcept -> TextureHandle {
    UID const ruid = Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_textures.load(
      ruid, TextureLoader::from_binary_tag{},
      width, height, channel, bits, data);
    // true only if the resource was not already present
    const bool loaded = ret.second;
    // takes the resource handle pointed to by the returned iterator
    entt::resource<Texture> res = ret.first->second;
    res->m_uid = ruid;
    res->init();
    return TextureHandle{ ret.first->second };
  }

  inline auto combineResourceFlags(
    Flags<ShaderReflection::ResourceEnum> a,
    Flags<ShaderReflection::ResourceEnum> b) noexcept
    -> Flags<ShaderReflection::ResourceEnum> {
    Flags<ShaderReflection::ResourceEnum> r = 0;
    if ((a | (ShaderReflection::ResourceEnum::NotReadable)) &&
      (b | (ShaderReflection::ResourceEnum::NotReadable)))
      r |= (ShaderReflection::ResourceEnum::NotReadable);
    if ((a | (ShaderReflection::ResourceEnum::NotWritable)) &&
      (b | (ShaderReflection::ResourceEnum::NotWritable)))
      r |= (ShaderReflection::ResourceEnum::NotWritable);
    return r;
  }

  auto ShaderReflection::operator+(
    ShaderReflection const& reflection) const
    -> ShaderReflection {
    ShaderReflection added = *this;
    added.bindingInfo.insert(reflection.bindingInfo.begin(),
      reflection.bindingInfo.end());
    for (int set = 0; set < reflection.bindings.size(); ++set) {
      if (added.bindings.size() <= set) {
        added.bindings.resize(set + 1);
        added.bindings[set] = reflection.bindings[set];
      }
      else {
        for (int binding = 0; binding < reflection.bindings[set].size();
          ++binding) {
          if (added.bindings[set].size() <= binding) {
            added.bindings[set].resize(binding + 1);
            added.bindings[set][binding] = reflection.bindings[set][binding];
          }
          else if (reflection.bindings[set][binding].type ==
            ResourceType::Undefined) {
            // SKIP
          }
          else if (added.bindings[set][binding].type ==
            ResourceType::Undefined) {
            added.bindings[set][binding] = reflection.bindings[set][binding];
          }
          else {
            assert(added.bindings[set][binding].type ==
              reflection.bindings[set][binding].type);
            added.bindings[set][binding].stages |=
              reflection.bindings[set][binding].stages;
            added.bindings[set][binding].flags =
              combineResourceFlags(added.bindings[set][binding].flags,
                reflection.bindings[set][binding].flags);
          }
        }
      }
    }
    int this_id = 0;
    for (int i = 0; i < reflection.pushConstant.size(); ++i) {
      // if added_pushconstants is smaller
      if (this_id >= added.pushConstant.size())
        added.pushConstant.push_back(reflection.pushConstant[i]);
      else if (added.pushConstant[this_id].offset ==
        reflection.pushConstant[i].offset) {
        added.pushConstant[this_id].stages |= reflection.pushConstant[i].stages;
      }
      else {
        added.pushConstant.push_back(reflection.pushConstant[i]);
      }
    }
    return added;
  }

  auto ShaderReflection::toBindGroupLayoutDescriptor(
    std::vector<ResourceEntry> const& bindings) noexcept
    -> rhi::BindGroupLayoutDescriptor {
    rhi::BindGroupLayoutDescriptor descriptor;
    for (uint32_t i = 0; i < bindings.size(); ++i) {
      auto const& bind = bindings[i];
      if (bind.type == ResourceType::UniformBuffer)
        descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
            i, bind.stages,
            rhi::BufferBindingLayout{rhi::BufferBindingType::UNIFORM} });
      else if (bind.type == ResourceType::StorageBuffer)
        descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
            i, bind.stages,
            rhi::BufferBindingLayout{rhi::BufferBindingType::STORAGE} });
      else if (bind.type == ResourceType::StorageImages) {
        rhi::BindGroupLayoutEntry entry{ i, bind.stages, rhi::StorageTextureBindingLayout{} };
        entry.array_size = bind.arraySize;
        descriptor.entries.emplace_back(entry);
      }
      else if (bind.type == ResourceType::AccelerationStructure) {
        rhi::BindGroupLayoutEntry entry{ i, bind.stages, rhi::AccelerationStructureBindingLayout{} };
        entry.array_size = bind.arraySize;
        descriptor.entries.emplace_back(entry);
      }
      else if (bind.type == ResourceType::SampledImages)
        descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
            i, bind.stages, rhi::BindlessTexturesBindingLayout{} });
      else if (bind.type == ResourceType::ReadonlyImage) {
        rhi::BindGroupLayoutEntry entry{ i, bind.stages,
                                        rhi::TextureBindingLayout{} };
        entry.array_size = bind.arraySize;
        descriptor.entries.emplace_back(entry);
      }
      else if (bind.type == ResourceType::Sampler)
        descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
            i, bind.stages, rhi::SamplerBindingLayout{} });
    }
    return descriptor;
  }

  void ShaderReflection::on_draw_gui() noexcept {
    if (ImGui::TreeNode("Shader Reflection")) {
      // Push constants
      if (ImGui::TreeNode("Push Constants")) {
        for (size_t i = 0; i < pushConstant.size(); ++i) {
          const auto& entry = pushConstant[i];
          ImGui::PushID(static_cast<int>(i));
          ImGui::Text("Index: %u", entry.index);
          ImGui::Text("Offset: %u", entry.offset);
          ImGui::Text("Range: %u", entry.range);
          //ImGui::Text("Stages: 0x%X", static_cast<uint32_t>(entry.stages));
          ImGui::Separator();
          ImGui::PopID();
        }
        ImGui::TreePop();
      }

      // Resource bindings
      if (ImGui::TreeNode("Descriptor Sets")) {
        for (size_t set = 0; set < bindings.size(); ++set) {
          const auto& setBindings = bindings[set];
          std::string label = "Set " + std::to_string(set);
          if (ImGui::TreeNode(label.c_str())) {
            for (size_t binding = 0; binding < setBindings.size(); ++binding) {
              const auto& entry = setBindings[binding];
              ImGui::PushID(static_cast<int>(binding));
              ImGui::Text("Binding:  %zu", binding);
              ImGui::Text("  Type:   %s", std::string(magic_enum::enum_name(entry.type)).c_str());
              ImGui::Text("  Flags:  %s", enum_flags_to_string<ResourceEnum>(entry.flags).c_str());
              ImGui::Text("  Stages: %s", enum_flags_to_string<rhi::ShaderStageEnum>(entry.stages).c_str());
              ImGui::Text("  #Array: %u", entry.arraySize);
              ImGui::Separator();
              ImGui::PopID();
            }
            ImGui::TreePop();
          }
        }
        ImGui::TreePop();
      }

      // Named binding info
      if (ImGui::TreeNode("Named Binding Info")) {
        for (const auto& [name, info] : bindingInfo) {
          ImGui::Text("Name: %s", name.c_str());
          ImGui::Text("  Set: %u", info.set);
          ImGui::Text("  Binding: %u", info.binding);
          ImGui::Text("  Type: %s", std::string(magic_enum::enum_name(info.type)).c_str());
          ImGui::Separator();
        }
        ImGui::TreePop();
      }

      ImGui::TreePop();
    }
  }

  SamplerLoader::result_type SamplerLoader::operator()(
    SamplerLoader::from_desc_tag, rhi::SamplerDescriptor const& desc) {
    std::shared_ptr<Sampler> ret = std::make_shared<Sampler>();
    ret->m_sampler = GFXContext::device()->create_sampler(desc);
    return ret;
  }

  SamplerLoader::result_type SamplerLoader::operator()(
    SamplerLoader::from_mode_tag, rhi::AddressMode address, rhi::FilterMode filter, rhi::MipmapFilterMode mipmap) {
    rhi::SamplerDescriptor desc;
    desc.addressModeU = address;
    desc.addressModeV = address;
    desc.addressModeW = address;
    desc.magFilter = filter;
    desc.minFilter = filter;
    desc.mipmapFilter = mipmap;
    return operator()(SamplerLoader::from_desc_tag{}, desc);
  }

  auto findDimension(rhi::TextureDimension dim, uint32_t arraySize) noexcept
    -> rhi::TextureViewDimension {
    rhi::TextureViewDimension dimension;
    switch (dim) {
    case rhi::TextureDimension::TEX1D:
      dimension = (arraySize > 1) ? rhi::TextureViewDimension::TEX1D_ARRAY
        : rhi::TextureViewDimension::TEX1D;
      break;
    case rhi::TextureDimension::TEX2D:
      dimension = (arraySize > 1) ? rhi::TextureViewDimension::TEX2D_ARRAY
        : rhi::TextureViewDimension::TEX2D;
      break;
    case rhi::TextureDimension::TEX3D:
      dimension = (arraySize > 1) ? rhi::TextureViewDimension::TEX3D_ARRAY
        : rhi::TextureViewDimension::TEX3D;
      break;
    default:
      break;
    }
    return dimension;
  }

  auto Texture::ResourceStateMachine::SubresourceRange::operator==(SubresourceRange const& x) noexcept -> bool {
    return m_levelBeg == x.m_levelBeg && m_levelEnd == x.m_levelEnd &&
      m_mipBeg == x.m_mipBeg && m_mipEnd == x.m_mipEnd;
  }

  auto Texture::ResourceStateMachine::SubresourceRange::valid() noexcept -> bool {
    return (m_levelBeg < m_levelEnd&& m_mipBeg < m_mipEnd);
  }

  auto Texture::ResourceStateMachine::SubresourceState::operator==(SubresourceState const& x) noexcept -> bool {
    return (stageMask._mask == x.stageMask._mask)
      && (access._mask == x.access._mask) &&
      (layout == x.layout);
  }

  auto Texture::ResourceStateMachine::intersect(
    SubresourceRange const& x,
    SubresourceRange const& y
  ) noexcept -> std::optional<SubresourceRange> {
    SubresourceRange isect;
    isect.m_levelBeg = std::max(x.m_levelBeg, y.m_levelBeg);
    isect.m_levelEnd = std::min(x.m_levelEnd, y.m_levelEnd);
    isect.m_mipBeg = std::max(x.m_mipBeg, y.m_mipBeg);
    isect.m_mipEnd = std::min(x.m_mipEnd, y.m_mipEnd);
    if (isect.valid())
      return isect;
    else
      return std::nullopt;
  }

  auto Texture::ResourceStateMachine::merge(
    SubresourceRange const& x,
    SubresourceRange const& y
  ) noexcept -> std::optional<SubresourceRange> {
    if (x.m_levelBeg == y.m_levelBeg && x.m_levelEnd == y.m_levelEnd) {
      if (x.m_mipBeg == y.m_mipEnd)
        return SubresourceRange{ x.m_levelBeg, 
          x.m_levelEnd, y.m_mipBeg, x.m_mipEnd };
      else if (x.m_mipEnd == y.m_mipBeg)
        return SubresourceRange{ x.m_levelBeg, 
          x.m_levelEnd, x.m_mipBeg, y.m_mipEnd };
      else return std::nullopt;
    }
    else if (x.m_mipBeg == y.m_mipBeg && x.m_mipEnd == y.m_mipEnd) {
      if (x.m_levelBeg == y.m_levelEnd)
        return SubresourceRange{ y.m_levelBeg, 
          x.m_levelEnd, x.m_mipBeg, x.m_mipEnd };
      else if (x.m_levelEnd == y.m_levelBeg)
        return SubresourceRange{ x.m_levelBeg, 
          y.m_levelEnd, x.m_mipBeg, x.m_mipEnd };
      else return std::nullopt;
    }
    else return std::nullopt;
  }
  
  auto Texture::ResourceStateMachine::diff(
    SubresourceRange const& x,
    SubresourceRange const& y
  ) noexcept -> std::vector<SubresourceRange> {
    std::vector<SubresourceRange> diffs;
    auto fn_subdivide_mip = [&](uint32_t level_beg, uint32_t level_end) {
      if (x.m_mipBeg == y.m_mipBeg && x.m_mipEnd == y.m_mipEnd) {
        // do nothing
      }
      else if (x.m_mipBeg == y.m_mipBeg) {
        diffs.emplace_back(SubresourceRange{ 
          level_beg, level_end, y.m_mipEnd, x.m_mipEnd });
      }
      else if (x.m_mipEnd == y.m_mipEnd) {
        diffs.emplace_back(SubresourceRange{ 
          level_beg, level_end, x.m_mipBeg, y.m_mipBeg });
      }
      else {
        diffs.emplace_back(SubresourceRange{ 
          level_beg, level_end, x.m_mipBeg, y.m_mipBeg });
        diffs.emplace_back(SubresourceRange{ 
          level_beg, level_end, y.m_mipEnd, x.m_mipEnd });
      }
    };
    if (x.m_levelBeg == y.m_levelBeg && x.m_levelEnd == y.m_levelEnd) {
      fn_subdivide_mip(x.m_levelBeg, x.m_levelEnd);
    }
    else if (x.m_levelBeg == y.m_levelBeg) {
      diffs.emplace_back(SubresourceRange{ y.m_levelEnd, 
        x.m_levelEnd, x.m_mipBeg, x.m_mipEnd });
      fn_subdivide_mip(y.m_levelBeg, y.m_levelEnd);
    }
    else if (x.m_levelEnd == y.m_levelEnd) {
      diffs.emplace_back(SubresourceRange{ x.m_levelBeg, 
        y.m_levelBeg, x.m_mipBeg, x.m_mipEnd });
      fn_subdivide_mip(y.m_levelBeg, y.m_levelEnd);
    }
    else {
      diffs.emplace_back(SubresourceRange{ x.m_levelBeg, 
        y.m_levelBeg, x.m_mipBeg, x.m_mipEnd });
      diffs.emplace_back(SubresourceRange{ y.m_levelEnd, 
        x.m_levelEnd, x.m_mipBeg, x.m_mipEnd });
      fn_subdivide_mip(y.m_levelBeg, y.m_levelEnd);
    }
    return diffs;
  }

  auto Texture::ResourceStateMachine::to_barrier_descriptor(
    SubresourceRange const& range,
    SubresourceState const& prev,
    SubresourceState const& next
  ) noexcept -> rhi::BarrierDescriptor {
    rhi::BarrierDescriptor desc = rhi::BarrierDescriptor{
    prev.stageMask,
    next.stageMask,
    rhi::DependencyTypeEnum::NONE,
    std::vector<rhi::MemoryBarrier*>{},
    std::vector<rhi::BufferMemoryBarrierDescriptor>{},
    std::vector<rhi::TextureMemoryBarrierDescriptor>{
        rhi::TextureMemoryBarrierDescriptor{
            m_texture,
            rhi::TextureRange{
                m_aspects, range.m_mipBeg, range.m_mipEnd - range.m_mipBeg,
                range.m_levelBeg, range.m_levelEnd - range.m_levelBeg},
            prev.access, next.access, prev.layout, next.layout}} };
    return desc;
  }

  auto Texture::ResourceStateMachine::try_merge() noexcept -> void {
    if (m_states.size() <= 1) return;
    while (true) {
      for (size_t i = 1; i <= m_states.size(); ++i) {
        if (i == m_states.size()) return;
        if (m_states[i].state == m_states[i - 1].state) {
          std::optional<SubresourceRange> merged =
            merge(m_states[i - 1].range, m_states[i].range);
          if (merged.has_value()) {
            m_states[i - 1].range = merged.value();
            m_states.erase(m_states.begin() + i);
            break;
          }
        }
      }
    }
  }

  auto Texture::ResourceStateMachine::update_subresource(
    SubresourceRange const& range,
    SubresourceState const& state
  ) noexcept -> std::vector<rhi::BarrierDescriptor> {
    std::vector<rhi::BarrierDescriptor> barriers;
    std::vector<SubresourceEntry> addedEntries;
    for (auto iter = m_states.begin();;) {
      if (iter == m_states.end()) break;
      if (iter->range == range) {
        barriers.emplace_back(to_barrier_descriptor(range, iter->state, state));
        iter->state = state;
        return barriers;
      }
      std::optional<SubresourceRange> isect =
        intersect(iter->range, range);
      if (isect.has_value()) {
        SubresourceRange const& isect_range = isect.value();
        barriers.emplace_back(
          to_barrier_descriptor(isect_range, iter->state, state));
        addedEntries.emplace_back(SubresourceEntry{ isect_range, state });
        std::vector<SubresourceRange> diff_ranges =
          diff(iter->range, isect_range);
        for (auto const& drange : diff_ranges)
          addedEntries.emplace_back(SubresourceEntry{ drange, iter->state });
        iter = m_states.erase(iter);
      }
      else {
        iter++;
      }
    }
    m_states.insert(m_states.end(), addedEntries.begin(), addedEntries.end());
    try_merge();
    return barriers;
  }

  auto Texture::ResourceStateMachine::transition(ResourceStateMachine const& new_sm) noexcept ->
    std::vector<rhi::BarrierDescriptor> {
    std::vector<rhi::BarrierDescriptor> output;
    for (auto& entry : new_sm.m_states) {
      std::vector<rhi::BarrierDescriptor> barriers = 
        update_subresource(entry.range, entry.state);
      output.insert(output.end(), barriers.begin(), barriers.end());
    }
    return output;
  }

  auto width() noexcept -> size_t;
  auto height() noexcept -> size_t;

  Texture::ResourceStateMachine::ResourceStateMachine(rhi::Texture* tex) {
    m_texture = tex;
    // first, figure out what aspects are used in the texture
    bool depthBit = rhi::has_depth_bit(tex->format());
    bool stencilBit = rhi::has_stencil_bit(tex->format());
    if (depthBit) m_aspects |= rhi::TextureAspectEnum::DEPTH_BIT;
    if (stencilBit) m_aspects |= rhi::TextureAspectEnum::STENCIL_BIT;
    if (!depthBit && !stencilBit) m_aspects |= rhi::TextureAspectEnum::COLOR_BIT;

    // all subresources, are initialized as undefined layout
    m_states.emplace_back(SubresourceEntry{
    SubresourceRange{0, tex->depth_or_array_layers(), 0, tex->mip_level_count()},
    SubresourceState{rhi::PipelineStageEnum::ALL_COMMANDS_BIT,
        rhi::AccessFlagEnum::NONE,
        rhi::TextureLayoutEnum::UNDEFINED} });
  }

  auto Texture::init() noexcept -> void {
    m_stateMachine = ResourceStateMachine(m_texture.get());
  }

  auto Texture::consume(ConsumeEntry const& entry) noexcept -> std::vector<rhi::BarrierDescriptor> {
    return m_stateMachine.update_subresource(
      gfx::Texture::ResourceStateMachine::SubresourceRange{
          entry.level_beg, entry.level_end, entry.mip_beg, entry.mip_end },
          gfx::Texture::ResourceStateMachine::SubresourceState{
              entry.stages, entry.access, entry.layout });
  }

  /** Get the UAV of the texture */
  auto Texture::get_uav(uint32_t mipLevel, uint32_t firstArraySlice,
    uint32_t arraySize) noexcept -> rhi::TextureView* {
    rhi::TextureViewIndex idx = { rhi::TextureViewType::UAV, mipLevel, 0, firstArraySlice, arraySize };
    rhi::TextureViewDimension dimension =
      findDimension(m_texture->dimension(), arraySize);
    auto find = m_viewPool.find(idx);
    if (find == m_viewPool.end()) {
      m_viewPool[idx] = m_texture->create_view(rhi::TextureViewDescriptor{
          m_texture->format(), dimension, (uint32_t)rhi::TextureAspectEnum::COLOR_BIT,
          mipLevel, 1, firstArraySlice, arraySize });
      find = m_viewPool.find(idx);
    }
    return find->second.get();
  }

  auto Texture::get_rtv(uint32_t mipLevel, uint32_t firstArraySlice,
    uint32_t arraySize) noexcept -> rhi::TextureView* {
    rhi::TextureViewIndex idx = { rhi::TextureViewType::RTV, mipLevel, 0, firstArraySlice, arraySize };
    rhi::TextureViewDimension dimension =
      findDimension(m_texture->dimension(), arraySize);
    auto find = m_viewPool.find(idx);
    if (find == m_viewPool.end()) {
      m_viewPool[idx] = m_texture->create_view(rhi::TextureViewDescriptor{
          m_texture->format(), dimension, (uint32_t)rhi::TextureAspectEnum::COLOR_BIT,
          mipLevel, 1, firstArraySlice, arraySize });
      find = m_viewPool.find(idx);
    }
    return find->second.get();
  }

  auto Texture::get_dsv(uint32_t mipLevel, uint32_t firstArraySlice,
    uint32_t arraySize) noexcept -> rhi::TextureView* {
    rhi::TextureViewIndex idx = { rhi::TextureViewType::DSV, mipLevel, 0, firstArraySlice, arraySize };
    rhi::TextureViewDimension dimension =
      findDimension(m_texture->dimension(), arraySize);
    auto find = m_viewPool.find(idx);
    if (find == m_viewPool.end()) {
      m_viewPool[idx] = m_texture->create_view(rhi::TextureViewDescriptor{
          m_texture->format(), dimension, (uint32_t)rhi::TextureAspectEnum::DEPTH_BIT,
          mipLevel, 1, firstArraySlice, arraySize });
      find = m_viewPool.find(idx);
    }
    return find->second.get();
  }

  auto Texture::get_srv(uint32_t mostDetailedMip, uint32_t mipCount,
    uint32_t firstArraySlice, uint32_t arraySize) noexcept -> rhi::TextureView* {
    rhi::TextureViewIndex idx = { rhi::TextureViewType::RTV, mostDetailedMip, mipCount, firstArraySlice, arraySize };
    rhi::TextureViewDimension dimension =
      findDimension(m_texture->dimension(), arraySize);
    Flags<rhi::TextureAspectEnum> aspect = rhi::TextureAspectEnum::COLOR_BIT;
    if (rhi::has_depth_bit(m_texture->format()))
      aspect = (uint32_t)rhi::TextureAspectEnum::DEPTH_BIT;
    if (rhi::has_stencil_bit(m_texture->format()))
      aspect |= (uint32_t)rhi::TextureAspectEnum::STENCIL_BIT;

    auto find = m_viewPool.find(idx);
    if (find == m_viewPool.end()) {
      m_viewPool[idx] = m_texture->create_view(rhi::TextureViewDescriptor{
          m_texture->format(), dimension, aspect, mostDetailedMip, mipCount,
          firstArraySlice, arraySize });
      find = m_viewPool.find(idx);
    }
    return find->second.get();
  }

  inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) {
    return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y);
  }

  inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) {
    return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y);
  }

  inline ImVec2 operator*(const ImVec2& lhs, const float& rhs) {
    return ImVec2(lhs.x * rhs, lhs.y * rhs);
  }

  inline ImVec2 operator/(const ImVec2& lhs, const float& rhs) {
    return ImVec2(lhs.x / rhs, lhs.y / rhs);
  }

  void ShowTextureViewer(
    ImTextureID texture,          // The texture to display
    int texWidth, int texHeight,  // Size of the original texture (not the display window)
    int maxPanelWidth, int maxPanelHeight,    // Max size for the viewer panel
    ImVec4* outPickedColor = nullptr,
    ImVec2* outPickedCoord = nullptr
  ) {
    static float zoom = 1.0f;
    static ImVec2 panOffset = ImVec2(0, 0);
    static ImVec2 prevCanvasPos = ImVec2(0, 0);
    static bool firstFrame = true;

    //static std::unique_ptr<editor_inline::ImageInspectGraph> graph;

    ImGuiIO& io = ImGui::GetIO();

    // Layout sizing
    ImVec2 availableSize = ImGui::GetContentRegionAvail();
    float panelWidth = ImMin((float)maxPanelWidth, availableSize.x);
    float panelHeight = ImMin((float)maxPanelHeight, availableSize.y);
    ImVec2 panelSize(panelWidth, panelHeight);

    //if (graph == nullptr) {
    //  //graph = std::make_unique<editor_inline::ImageInspectGraph>();
    //  //graph->m_standardSize = { int32_t(availableSize.x), int32_t(availableSize.x), 1};
    //  //graph->build();
    //}

    if (ImGui::Button("Reset View")) {
      zoom = 1.0f;
      panOffset = ImVec2(0, 0);
      firstFrame = true;
    }

    ImVec2 canvasPos = ImGui::GetCursorScreenPos();

    // Fix panOffset if layout moves the canvas
    if (!firstFrame) {
      ImVec2 canvasDelta = canvasPos - prevCanvasPos;
      panOffset = panOffset - canvasDelta;
    }
    prevCanvasPos = canvasPos;
    firstFrame = false;

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(canvasPos, canvasPos + panelSize, IM_COL32(30, 30, 30, 255));

    ImVec2 imageSize = ImVec2(texWidth * zoom, texHeight * zoom);
    ImVec2 imagePos = canvasPos + panOffset;

    // Handle zooming
    if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f) {
      ImVec2 mousePos = io.MousePos;
      ImVec2 beforeZoom = (mousePos - imagePos) / zoom;

      float newZoom = ImClamp(zoom * (1.0f + io.MouseWheel * 0.1f), 0.1f, 20.0f);
      ImVec2 afterZoom = beforeZoom * newZoom;

      panOffset = panOffset + (beforeZoom * zoom - afterZoom);
      zoom = newZoom;

      imageSize = ImVec2(texWidth * zoom, texHeight * zoom);
      imagePos = canvasPos + panOffset;
    }

    // Handle panning
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      panOffset = panOffset + io.MouseDelta;
      imagePos = canvasPos + panOffset;
    }

    // Draw image
    ImVec2 uv0(0, 0), uv1(1, 1);
    drawList->AddImage(texture, imagePos, imagePos + imageSize, uv0, uv1);

    // Pixel picking
    ImVec2 mouseCanvasPos = io.MousePos - imagePos;
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      int px = (int)(mouseCanvasPos.x / zoom);
      int py = (int)(mouseCanvasPos.y / zoom);
      if (px >= 0 && px < texWidth && py >= 0 && py < texHeight) {
        if (outPickedCoord) *outPickedCoord = ImVec2((float)px, (float)py);
        if (outPickedColor) {
          *outPickedColor = ImVec4(
            (float)px / texWidth,
            (float)py / texHeight,
            1.0f - (float)px / texWidth,
            1.0f
          );
        }
      }
    }

    // Display debug info
    if (outPickedCoord && outPickedColor) {
      ImGui::Text("Picked Pixel: (%d, %d)", (int)outPickedCoord->x, (int)outPickedCoord->y);
      ImGui::ColorEdit4("Picked Color", (float*)outPickedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_DisplayRGB);
    }
  }

  auto Buffer::ConsumeEntry::add_stage(Flags<rhi::PipelineStageEnum> stage) noexcept -> ConsumeEntry& {
    m_stages |= stage;
    return *this;
  }

  auto Buffer::ConsumeEntry::set_access(Flags<rhi::AccessFlagEnum> acc) noexcept -> ConsumeEntry& {
    m_access = acc;
    return *this;
  }

  auto Buffer::ConsumeEntry::set_subresource(uint64_t offset, uint64_t size
  ) noexcept -> Buffer::ConsumeEntry& {
    this->m_offset = offset;
    this->m_size = size;
    return *this;
  }

  auto Buffer::ResourceStateMachine::SubresourceRange::operator==(SubresourceRange const& x) noexcept -> bool {
    return m_rangeBeg == x.m_rangeBeg && m_rangeEnd == x.m_rangeEnd;
  }

  auto Buffer::ResourceStateMachine::SubresourceRange::valid() noexcept -> bool {
    return (m_rangeBeg < m_rangeEnd);
  }

  auto Buffer::ResourceStateMachine::SubresourceState::operator==(SubresourceState const& x) noexcept -> bool {
    return m_stageMask._mask == x.m_stageMask._mask && m_access._mask == x.m_access._mask;
  }

  auto Buffer::ResourceStateMachine::intersect(
    SubresourceRange const& x,
    SubresourceRange const& y) noexcept
    -> std::optional<SubresourceRange> {
    SubresourceRange isect;
    isect.m_rangeBeg = std::max(x.m_rangeBeg, y.m_rangeBeg);
    isect.m_rangeEnd = std::min(x.m_rangeEnd, y.m_rangeEnd);
    if (isect.valid())
      return isect;
    else
      return std::nullopt;
  }

  auto Buffer::ResourceStateMachine::diff(
    SubresourceRange const& x,
    SubresourceRange const& y) noexcept
    -> std::vector<SubresourceRange> {
    std::vector<SubresourceRange> diffs;
    if (x.m_rangeBeg == y.m_rangeBeg && x.m_rangeEnd == y.m_rangeEnd) {
    // do nothing
    } else if (x.m_rangeBeg == y.m_rangeBeg) {
    diffs.emplace_back(SubresourceRange{y.m_rangeEnd, x.m_rangeEnd });
    } else if (x.m_rangeEnd == y.m_rangeEnd) {
    diffs.emplace_back(SubresourceRange{x.m_rangeBeg, y.m_rangeBeg });
    } else {
    diffs.emplace_back(SubresourceRange{x.m_rangeBeg, y.m_rangeBeg });
    diffs.emplace_back(SubresourceRange{y.m_rangeEnd, x.m_rangeEnd });
    }
    return diffs;
  }

  auto Buffer::ResourceStateMachine::to_barrier_descriptor(
    SubresourceRange const& range,
    SubresourceState const& prev,
    SubresourceState const& next) noexcept
    -> rhi::BarrierDescriptor {
      rhi::BarrierDescriptor desc = rhi::BarrierDescriptor {
      prev.m_stageMask, next.m_stageMask, rhi::DependencyTypeEnum::NONE,
          std::vector<rhi::MemoryBarrier*>{},
          std::vector<rhi::BufferMemoryBarrierDescriptor>{
              rhi::BufferMemoryBarrierDescriptor{
                  m_buffer, prev.m_access, next.m_access, range.m_rangeBeg,
                  range.m_rangeEnd - range.m_rangeBeg}},
        std::vector<rhi::TextureMemoryBarrierDescriptor>{}};
    return desc;
  }

  inline auto AccessIsWrite(rhi::AccessFlagEnum bit) noexcept -> bool {
    switch (bit) {
    case rhi::AccessFlagEnum::INDIRECT_COMMAND_READ_BIT:
    case rhi::AccessFlagEnum::INDEX_READ_BIT:
    case rhi::AccessFlagEnum::VERTEX_ATTRIBUTE_READ_BIT:
    case rhi::AccessFlagEnum::UNIFORM_READ_BIT:
    case rhi::AccessFlagEnum::INPUT_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagEnum::SHADER_READ_BIT:
    case rhi::AccessFlagEnum::COLOR_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagEnum::TRANSFER_READ_BIT:
    case rhi::AccessFlagEnum::HOST_READ_BIT:
    case rhi::AccessFlagEnum::MEMORY_READ_BIT:
    case rhi::AccessFlagEnum::TRANSFORM_FEEDBACK_COUNTER_READ_BIT:
    case rhi::AccessFlagEnum::CONDITIONAL_RENDERING_READ_BIT:
    case rhi::AccessFlagEnum::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT:
    case rhi::AccessFlagEnum::ACCELERATION_STRUCTURE_READ_BIT:
    case rhi::AccessFlagEnum::FRAGMENT_DENSITY_MAP_READ_BIT:
    case rhi::AccessFlagEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT:
    case rhi::AccessFlagEnum::COMMAND_PREPROCESS_READ_BIT:
    case rhi::AccessFlagEnum::NONE: return false;
    case rhi::AccessFlagEnum::SHADER_WRITE_BIT:
    case rhi::AccessFlagEnum::COLOR_ATTACHMENT_WRITE_BIT:
    case rhi::AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT:
    case rhi::AccessFlagEnum::TRANSFER_WRITE_BIT:
    case rhi::AccessFlagEnum::HOST_WRITE_BIT:
    case rhi::AccessFlagEnum::MEMORY_WRITE_BIT:
    case rhi::AccessFlagEnum::TRANSFORM_FEEDBACK_WRITE_BIT:
    case rhi::AccessFlagEnum::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT:
    case rhi::AccessFlagEnum::ACCELERATION_STRUCTURE_WRITE_BIT:
    case rhi::AccessFlagEnum::COMMAND_PREPROCESS_WRITE_BIT:
      return true;
    default:
      return false;
    }
  }

  inline auto ExtractWriteAccessFlags(Flags<rhi::AccessFlagEnum> flag) noexcept -> Flags<rhi::AccessFlagEnum> {
    Flags<rhi::AccessFlagEnum> eflag = 0;
    for (int i = 0; i < 32; ++i) {
      const uint32_t bit = flag.mask() & (0x1 << i);
      if (bit != 0 && AccessIsWrite(rhi::AccessFlagEnum(bit))) {
        eflag |= bit;
      }
    }
    return eflag;
  }

  inline auto ExtractReadAccessFlags(Flags<rhi::AccessFlagEnum> flag) noexcept -> Flags<rhi::AccessFlagEnum> {
    Flags<rhi::AccessFlagEnum> eflag = 0;
    for (int i = 0; i < 32; ++i) {
      const uint32_t bit = flag.mask() & (0x1 << i);
      if (bit != 0 && !AccessIsWrite(rhi::AccessFlagEnum(bit))) {
        eflag |= bit;
      }
    }
    return eflag;
  }

  auto Buffer::ResourceStateMachine::update_subresource(
    SubresourceRange const& range,
    SubresourceState const& state) noexcept
    -> std::vector<rhi::BarrierDescriptor> {
    std::vector<rhi::BarrierDescriptor> barriers;
    std::vector<SubresourceEntry> addedEntries;

    // First check write access
    Flags<rhi::AccessFlagEnum> write_access = ExtractWriteAccessFlags(state.m_access);
    if (write_access) {
      SubresourceState target_state;
      target_state.m_stageMask = state.m_stageMask;
      target_state.m_access = write_access;
      // Write - Write hazard
      for (auto iter = m_writeStates.begin();;) {
        if (iter == m_writeStates.end()) break;
        if (iter->range == range) {
          if (iter->state.m_access)
            barriers.emplace_back(
              to_barrier_descriptor(range, iter->state, target_state));
          break;
        }
        std::optional<SubresourceRange> isect =
          intersect(iter->range, range);
        if (isect.has_value()) {
          SubresourceRange const& isect_range = isect.value();
          if (iter->state.m_access)
            barriers.emplace_back(
              to_barrier_descriptor(isect_range, iter->state, target_state));
          iter++;
        }
        else {
          iter++;
        }
      }
      // Read - Write hazard
      for (auto iter = m_readStates.begin();;) {
        if (iter == m_readStates.end()) break;
        if (iter->range == range) {
          if (iter->state.m_access)
            barriers.emplace_back(
              to_barrier_descriptor(range, iter->state, target_state));
          break;
        }
        std::optional<SubresourceRange> isect =
          intersect(iter->range, range);
        if (isect.has_value()) {
          SubresourceRange const& isect_range = isect.value();
          if (iter->state.m_access)
            barriers.emplace_back(
              to_barrier_descriptor(isect_range, iter->state, target_state));
          iter++;
        }
        else {
          iter++;
        }
      }
    }

    // Then check read access
    Flags<rhi::AccessFlagEnum> read_access = ExtractReadAccessFlags(state.m_access);
    if (read_access) {
      SubresourceState target_state;
      target_state.m_stageMask = state.m_stageMask;
      target_state.m_access = read_access;
      // Write - Read hazard
      for (auto iter = m_writeStates.begin();;) {
        if (iter == m_writeStates.end()) break;
        if (iter->range == range) {
          if (iter->state.m_access)
            barriers.emplace_back(
              to_barrier_descriptor(range, iter->state, target_state));
          break;
        }
        std::optional<SubresourceRange> isect =
          intersect(iter->range, range);
        if (isect.has_value()) {
          SubresourceRange const& isect_range = isect.value();
          if (iter->state.m_access)
            barriers.emplace_back(
              to_barrier_descriptor(isect_range, iter->state, target_state));
          iter++;
        }
        else {
          iter++;
        }
      }
    }

    if (write_access) {
      // Update write state
      for (auto iter = m_writeStates.begin();;) {
        if (iter == m_writeStates.end()) break;
        if (iter->range == range) {
          iter->state.m_stageMask = state.m_stageMask;
          iter->state.m_access = write_access;
          break;
        }
        std::optional<SubresourceRange> isect =
          intersect(iter->range, range);
        if (isect.has_value()) {
          // isect ranges
          SubresourceRange const& isect_range = isect.value();
          SubresourceState isect_state;
          isect_state.m_stageMask = state.m_stageMask;
          isect_state.m_access = write_access;
          addedEntries.emplace_back(
            SubresourceEntry{ isect_range, isect_state });
          // diff ranges
          std::vector<SubresourceRange> diff_ranges =
            diff(iter->range, isect_range);
          for (auto const& drange : diff_ranges) {
            addedEntries.emplace_back(
              SubresourceEntry{ drange, iter->state });
          }
          iter = m_writeStates.erase(iter);
        }
        else {
          iter++;
        }
      }
      // Clear read state
      for (auto iter = m_readStates.begin();;) {
        if (iter == m_readStates.end()) break;
        if (iter->range == range) {
          iter->state.m_stageMask = 0;
          iter->state.m_access = 0;
          break;
        }
        std::optional<SubresourceRange> isect =
          intersect(iter->range, range);
        if (isect.has_value()) {
          // isect ranges
          SubresourceRange const& isect_range = isect.value();
          SubresourceState isect_state;
          isect_state.m_stageMask = 0;
          isect_state.m_access = 0;
          addedEntries.emplace_back(
            SubresourceEntry{ isect_range, isect_state });
          // diff ranges
          std::vector<SubresourceRange> diff_ranges =
            diff(iter->range, isect_range);
          for (auto const& drange : diff_ranges) {
            addedEntries.emplace_back(
              SubresourceEntry{ drange, iter->state });
          }
          iter = m_readStates.erase(iter);
        }
        else {
          iter++;
        }
      }
    }

    if (read_access) {
      // Update read state
      for (auto iter = m_readStates.begin();;) {
        if (iter == m_readStates.end()) break;
        if (iter->range == range) {
          iter->state.m_stageMask |= state.m_stageMask;
          iter->state.m_access |= read_access;
          break;
        }
        std::optional<SubresourceRange> isect =
          intersect(iter->range, range);
        if (isect.has_value()) {
          // isect ranges
          SubresourceRange const& isect_range = isect.value();
          SubresourceState isect_state = iter->state;
          isect_state.m_stageMask |= state.m_stageMask;
          isect_state.m_access |= read_access;
          addedEntries.emplace_back(
            SubresourceEntry{ isect_range, isect_state });
          // diff ranges
          std::vector<SubresourceRange> diff_ranges =
            diff(iter->range, isect_range);
          for (auto const& drange : diff_ranges) {
            addedEntries.emplace_back(
              SubresourceEntry{ drange, iter->state });
          }
          iter = m_readStates.erase(iter);
        }
        else {
          iter++;
        }
      }
    }
    //tryMerge();
    return barriers;
  }

  auto Buffer::host_to_device() noexcept -> void {
    // if nothing is on device, create both buffer and previous
    if (m_buffer == nullptr && m_previous == nullptr) {
      bool host_buffer_is_empty = false;
      if (m_host.size() == 0) {
        host_buffer_is_empty = true;
        m_host.resize(64);
      }

      // creation the rhi::buffers
      if (m_memoryCopyMode == MemoryCopyMode::COHERENT_MAPPING) {
        rhi::BufferDescriptor descriptor;
        descriptor.size = m_host.size();
        descriptor.usage = m_usages;
        descriptor.memoryProperties = rhi::MemoryPropertyEnum::HOST_VISIBLE_BIT
          | rhi::MemoryPropertyEnum::HOST_COHERENT_BIT;
        m_buffer = GFXContext::device()->create_buffer(descriptor);
        m_buffer->map_async(rhi::MapModeEnum::WRITE, 0, m_host.size()).wait();
        m_previous = GFXContext::device()->create_buffer(descriptor);
        m_previous->map_async(rhi::MapModeEnum::WRITE, 0, m_host.size()).wait();
        memcpy(m_buffer->m_mappedData, m_host.data(), m_host.size());
        memcpy(m_previous->m_mappedData, m_host.data(), m_host.size());
      }
      else if(m_memoryCopyMode == MemoryCopyMode::TEMPORARY_STAGING) {
        m_buffer = GFXContext::device()->create_device_local_buffer(
          static_cast<void const*>(m_host.data()), m_host.size(), m_usages);
        m_previous = GFXContext::device()->create_device_local_buffer(
          static_cast<void const*>(m_host.data()), m_host.size(), m_usages);
      }
    }
    // otherwise update the gpu buffer as long as prev is not equal to host
    else if (m_previousStamp != m_hostStamp) {
      if (m_memoryCopyMode == MemoryCopyMode::COHERENT_MAPPING) {
        std::swap(m_buffer, m_previous);
        memcpy(m_buffer->m_mappedData, m_host.data(), m_host.size());
      }
      else if (m_memoryCopyMode == MemoryCopyMode::TEMPORARY_STAGING) {
        m_previous = std::move(m_buffer);
        m_buffer = GFXContext::device()->create_device_local_buffer(
          static_cast<void const*>(m_host.data()), 
          m_host.size() * sizeof(unsigned char), m_usages);
      }
      m_bufferStamp = m_hostStamp;
      m_previousStamp = m_bufferStamp;
    }
  }

  auto Buffer::device_to_host() noexcept -> void {
    if (m_host.size() < m_buffer->size()) 
      m_host.resize(m_buffer->size());
    GFXContext::device()->readback_device_local_buffer(m_buffer.get(), m_host.data(), m_buffer->size());
  }

  auto Buffer::create_device() noexcept -> void {
    if (m_buffer == nullptr || (m_buffer->size() != m_host.size() * sizeof(unsigned char))) {
      rhi::BufferDescriptor descriptor;
      descriptor.size = m_host.size() * sizeof(unsigned char);
      descriptor.usage = m_usages;
      descriptor.memoryProperties = rhi::MemoryPropertyEnum::DEVICE_LOCAL_BIT;
      descriptor.mappedAtCreation = false;
      m_buffer = GFXContext::device()->create_buffer(descriptor);
    }
  }

  auto Buffer::memory_mapping() noexcept -> void* {
    m_host.resize(m_buffer->size());
    std::future<bool> finish = m_buffer->map_async(rhi::MapModeEnum::READ | rhi::MapModeEnum::WRITE, 0, m_buffer->size());
    finish.wait();
    return m_buffer->get_mapped_range();
  }

  auto Buffer::get_host() noexcept -> std::vector<std::byte>& {
    device_to_host();
    return m_host;
  }

  auto Buffer::get_device() noexcept -> rhi::Buffer* {
    return m_buffer.get();
  }

  auto Buffer::get_binding_resource() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ rhi::BufferBinding{ m_buffer.get(), 0, m_buffer->size() } };
  }
  
  auto Buffer::draw_gui(editor::IFragment* fragment) noexcept -> void {
    if (ImGui::BeginTable("Buffer", 2, ImGuiTableFlags_Borders)) {
      // Set column widths if needed
      ImGui::TableSetupColumn("Property");
      ImGui::TableSetupColumn("Value");
      ImGui::TableHeadersRow();  // Optional: shows a header row

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Name");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", m_name.c_str());

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Source");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", m_creator.c_str());

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Role");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", m_job.c_str());

      //// Example row 2
      //ImGui::TableNextRow();
      //ImGui::TableSetColumnIndex(0);
      //ImGui::Text("Height");
      //ImGui::TableSetColumnIndex(1);
      //ImGui::Text("%d", 720);

      //// Example row 3 with editable field
      //ImGui::TableNextRow();
      //ImGui::TableSetColumnIndex(0);
      //ImGui::Text("Name");
      //ImGui::TableSetColumnIndex(1);
      //static char name[64] = "Example";
      //ImGui::InputText("##NameInput", name, IM_ARRAYSIZE(name));

      ImGui::EndTable();
    }
  }

  BufferLoader::result_type BufferLoader::operator()(from_empty_tag) {
    BufferLoader::result_type result = std::make_shared<Buffer>();
    return result;
  }

  BufferLoader::result_type BufferLoader::operator()(from_desc_tag, rhi::BufferDescriptor desc) {
    BufferLoader::result_type result = std::make_shared<Buffer>();
    result->m_buffer = GFXContext::device()->create_buffer(desc);
    return result;
  }

  BufferLoader::result_type BufferLoader::operator()(from_host_tag, MiniBuffer const& input, Flags<rhi::BufferUsageEnum> usages) {
    BufferLoader::result_type result = std::make_shared<Buffer>();
    result->m_buffer = GFXContext::device()->create_device_local_buffer(
      static_cast<void const*>(input.m_data), input.m_size, usages);
    return result;
  }

  auto Texture::ConsumeEntry::add_stage(Flags<rhi::PipelineStageEnum> stage) noexcept -> ConsumeEntry& {
    stages |= stage;
    return *this;
  }

  auto Texture::ConsumeEntry::set_layout(rhi::TextureLayoutEnum _layout) noexcept -> ConsumeEntry& {
    layout = _layout;
    return *this;
  }

  auto Texture::ConsumeEntry::enable_depth_write(bool set) noexcept -> ConsumeEntry& {
    depthWrite = set;
    return *this;
  }

  auto Texture::ConsumeEntry::set_depth_compare_fn(rhi::CompareFunction fn) noexcept
    -> ConsumeEntry& {
    depthCmp = fn;
    return *this;
  }

  auto Texture::ConsumeEntry::set_subresource(uint32_t mip_beg, uint32_t mip_end,
    uint32_t level_beg, uint32_t level_end) noexcept
    -> ConsumeEntry& {
    this->mip_beg = mip_beg;
    this->mip_end = mip_end;
    this->level_beg = level_beg;
    this->level_end = level_end;
    return *this;
  }

  auto Texture::ConsumeEntry::set_attachment_loc(uint32_t loc) noexcept -> ConsumeEntry& {
    attachLoc = loc;
    return *this;
  }

  auto Texture::ConsumeEntry::set_access(Flags<rhi::AccessFlagEnum> acc) noexcept -> ConsumeEntry& {
    access = acc;
    return *this;
  }

  auto Texture::ConsumeEntry::set_blend_operation(rhi::BlendOperation operation) noexcept -> ConsumeEntry& {
    bldOperation = operation;
    return *this;
  }

  auto Texture::ConsumeEntry::set_source_blender_factor(rhi::BlendFactor factor) noexcept -> ConsumeEntry& {
    srcFactor = factor;
    return *this;
  }

  auto Texture::ConsumeEntry::set_target_blender_factor(rhi::BlendFactor factor) noexcept -> ConsumeEntry& {
    dstFactor = factor;
    return *this;
  }

  auto Texture::draw_gui(editor::IFragment* fragment) noexcept -> void {
    if (ImGui::BeginTable("Texture", 2, ImGuiTableFlags_Borders)) {
      // Set column widths if needed
      ImGui::TableSetupColumn("Property");
      ImGui::TableSetupColumn("Value");
      ImGui::TableHeadersRow();  // Optional: shows a header row

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Name");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", m_name.c_str());

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Source");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", m_creator.c_str());

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Role");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", m_job.c_str());

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Width");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%d", m_texture->width());
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("Height");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%d", m_texture->height());
      //// Example row 2
      //ImGui::TableNextRow();
      //ImGui::TableSetColumnIndex(0);
      //ImGui::Text("Height");
      //ImGui::TableSetColumnIndex(1);
      //ImGui::Text("%d", 720);

      //// Example row 3 with editable field
      //ImGui::TableNextRow();
      //ImGui::TableSetColumnIndex(0);
      //ImGui::Text("Name");
      //ImGui::TableSetColumnIndex(1);
      //static char name[64] = "Example";
      //ImGui::InputText("##NameInput", name, IM_ARRAYSIZE(name));

      ImGui::EndTable();
    }

    editor::ImageInspectorFragment* iif = static_cast<editor::ImageInspectorFragment*>(fragment);
    
    if (ImGui::Button("Save image")) {
      std::string filepath = se::Platform::save_file(
        "", Worldtime::get().to_string() + ".exr");
      Singleton<gfx::GFXContext>::instance()->m_jobsFrameEnd.push_back(
        [tex = this, filepath = filepath]() {tex->save_image(filepath); });
    }
    ImGui::SameLine();

    static std::vector<std::string> display_items = {
      "RGBA", "RGB", "R Channel", "G Channel", "B Channel"
    };
    if (ImGui::BeginCombo("##Display", display_items[iif->m_showChannel].c_str())) {
      for (int i = 0; i < display_items.size(); ++i) {
        bool isSelected = (i == iif->m_showChannel);
        if (ImGui::Selectable(display_items[i].c_str(), isSelected)) {
          iif->m_showChannel = i;
        }
      }
      ImGui::EndCombo();
    }

    gfx::Texture* out_img = iif->execute();

    std::vector<rhi::BarrierDescriptor> barriers = out_img->consume(ConsumeEntry()
      .add_stage(rhi::PipelineStageEnum::FRAGMENT_SHADER_BIT)
      .set_layout(rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL)
      .set_access(rhi::AccessFlagEnum::SHADER_READ_BIT));
    for (int i = 0; i < barriers.size(); ++i)
      editor::ImGuiContext::m_encoder->pipeline_barrier(barriers[i]);

    if (!iif->m_imguiTex) {
      iif->m_sampler = gfx::GFXContext::create_sampler_desc(
        rhi::AddressMode::CLAMP_TO_EDGE, rhi::FilterMode::NEAREST,
        rhi::MipmapFilterMode::NEAREST);

      iif->m_imguiTex = editor::ImGuiContext::create_imgui_texture(
        iif->m_sampler->m_sampler.get(),
        out_img->get_srv(0, 1, 0, 1),
        rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL);
    }

    ImGui::Image(iif->m_imguiTex->get_texture_id(),
      { (float)out_img->m_texture->width(), (float)out_img->m_texture->height() },
      { 0, 0 }, { 1, 1 });

    const ImGuiIO& IO = ImGui::GetIO();
    ivec2 output_pixel = iif->m_readbackInfo->outPixel;
    bool isInRange = output_pixel.x >= 0 && output_pixel.y >= 0 &&
      output_pixel.x < m_texture->width() && output_pixel.y < m_texture->height();
    if (ImGui::IsItemHovered() && IO.MouseDown[ImGuiMouseButton_Right] && isInRange) {
      // Show a tooltip for currently hovered texel
      ImVec2 texelTL;
      ImVec2 texelBR;
      //if (GetVisibleTexelRegionAndGetData(inspector, texelTL, texelBR))
      {
        //ImVec4 color = GetTexel(&inspector->Buffer, (int)mousePosTexel.x, (int)mousePosTexel.y);
        ImVec4 color = {
          iif->m_readbackInfo->color.x,
          iif->m_readbackInfo->color.y,
          iif->m_readbackInfo->color.z,
          iif->m_readbackInfo->color.w,
        };
        char buffer[128];
        sprintf(buffer, "Texel: (%d, %d)", (int)iif->m_readbackInfo->outPixel.x, (int)iif->m_readbackInfo->outPixel.y);

        ImGui::ColorTooltip(buffer, &color.x, 0);
      }
    }
  }

  auto Texture::width() noexcept -> size_t {
    return m_texture->width();
  }

  auto Texture::height() noexcept -> size_t {
    return m_texture->height();
  }

  auto Texture::get_imgui_texture() noexcept -> editor::ImguiTexture* {
    if (m_imguiTexture.get() == nullptr) {
      SamplerHandle sampler = GFXContext::create_sampler_desc({});
      m_imguiTexture = editor::ImGuiContext::create_imgui_texture(
        sampler->m_sampler.get(),
        get_srv(0, 1, 0, 1),
        rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL);
    }
    return m_imguiTexture.get();
  }

  auto ShaderModule::draw_gui(editor::IFragment* fragment) noexcept -> void {
    ImGui::Text("Shader Module");
    ImGui::Text("Name: ");
    ImGui::SameLine();
    ImGui::Text("%s", m_shaderModule->m_name.c_str());

    std::string stage_str = enum_flags_to_string(m_shaderModule->m_stages);
    ImGui::Text("Stages: %s", stage_str.c_str());

    m_reflection.on_draw_gui();
  }
  
  auto Mesh::draw_gui(editor::IFragment* fragment) noexcept -> void {
    if (m_primitives.empty()) {
      ImGui::Text("No mesh data available.");
      return;
    }

    // Compact styling
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));

    // If is customPRIMITIVE
    if(m_customPrimitives.size()>0) {
      for (size_t i = 0; i < m_customPrimitives.size(); ++i) {
        auto& mesh = m_customPrimitives[i];
        ImGui::PushID(static_cast<int>(i));

        // Collapsible header (primary info)
        bool open = ImGui::CollapsingHeader(
          ("Mesh " + std::to_string(i) +
            " | Type: " + std::to_string(mesh.primitiveType) +
            " | Material: " + std::to_string(1)).c_str(),
          ImGuiTreeNodeFlags_DefaultOpen
        );

        // Quick actions (buttons aligned to the right)
        ImGui::SameLine(ImGui::GetWindowWidth() - 100); // Right-align
        if (ImGui::SmallButton("Focus")) { /* Camera logic */ }
        ImGui::SameLine();
        if (ImGui::SmallButton("Delete")) { /* Mark for deletion */ }

        // Expanded details
        if (open) {
          ImGui::Indent();

          // Two-column layout for properties
          ImGui::Columns(2, "MeshDetails", false);
          ImGui::SetColumnWidth(0, 120); // Fixed width for labels
          
          ImGui::Text("Bounds Min"); ImGui::NextColumn();
          ImGui::Text("(%.2f, %.2f, %.2f)", mesh.min.x, mesh.min.y, mesh.min.z); ImGui::NextColumn();

          ImGui::Text("Bounds Max"); ImGui::NextColumn();
          ImGui::Text("(%.2f, %.2f, %.2f)", mesh.max.x, mesh.max.y, mesh.max.z); ImGui::NextColumn();

          ImGui::Text("Material"); ImGui::NextColumn();
          bool open_mat = ImGui::TreeNodeEx("##MyRightArrow", 
            ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_NoTreePushOnOpen | 
            ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen);
          // Content (only if open)
          if (ImGui::IsItemClicked()) {
            // Toggle open state via click
            ImGui::SetNextItemOpen(!open_mat, ImGuiCond_Always);
          }

          ImGui::Columns(1); // Reset

          if (open_mat) {
            ImGui::Indent();
            if (mesh.material.get())
              mesh.material->draw_gui(nullptr);
            ImGui::Unindent();
          }

          ImGui::Unindent();
        }


        ImGui::PopID();
        ImGui::Separator(); // Visual spacing between items
      }
    }
    else {
      for (size_t i = 0; i < m_primitives.size(); ++i) {
        auto& mesh = m_primitives[i];
        ImGui::PushID(static_cast<int>(i));

        // Collapsible header (primary info)
        bool open = ImGui::CollapsingHeader(
          ("Mesh " + std::to_string(i) +
            " | Vertices: " + std::to_string(mesh.numVertex) +
            " | Material: " + std::to_string(1)).c_str(),
          ImGuiTreeNodeFlags_DefaultOpen
        );

        // Quick actions (buttons aligned to the right)
        ImGui::SameLine(ImGui::GetWindowWidth() - 100); // Right-align
        if (ImGui::SmallButton("Focus")) { /* Camera logic */ }
        ImGui::SameLine();
        if (ImGui::SmallButton("Delete")) { /* Mark for deletion */ }

        // Expanded details
        if (open) {
          ImGui::Indent();

          // Two-column layout for properties
          ImGui::Columns(2, "MeshDetails", false);
          ImGui::SetColumnWidth(0, 120); // Fixed width for labels

          ImGui::Text("Offset"); ImGui::NextColumn();
          ImGui::Text("%zu", mesh.offset); ImGui::NextColumn();

          ImGui::Text("Size"); ImGui::NextColumn();
          ImGui::Text("%zu bytes", mesh.size); ImGui::NextColumn();

          ImGui::Text("Base Vertex"); ImGui::NextColumn();
          ImGui::Text("%zu", mesh.baseVertex); ImGui::NextColumn();

          ImGui::Text("Bounds Min"); ImGui::NextColumn();
          ImGui::Text("(%.2f, %.2f, %.2f)", mesh.min.x, mesh.min.y, mesh.min.z); ImGui::NextColumn();

          ImGui::Text("Bounds Max"); ImGui::NextColumn();
          ImGui::Text("(%.2f, %.2f, %.2f)", mesh.max.x, mesh.max.y, mesh.max.z); ImGui::NextColumn();

          ImGui::Text("Material"); ImGui::NextColumn();
          bool open_mat = ImGui::TreeNodeEx("##MyRightArrow", 
            ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_NoTreePushOnOpen | 
            ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen);
          // Content (only if open)
          if (ImGui::IsItemClicked()) {
            // Toggle open state via click
            ImGui::SetNextItemOpen(!open_mat, ImGuiCond_Always);
          }

          ImGui::Columns(1); // Reset

          if (open_mat) {
            ImGui::Indent();
            if (mesh.material.get())
              mesh.material->draw_gui(nullptr);
            ImGui::Unindent();
          }

          ImGui::Unindent();
        }


        ImGui::PopID();
        ImGui::Separator(); // Visual spacing between items
      }
    }

    ImGui::PopStyleVar(2);
  }

  auto draw_texture_inspector(TextureHandle handle, editor::IFragment* fragment) { handle.draw_gui(fragment); }
  auto draw_buffer_inspector(BufferHandle handle, editor::IFragment* fragment) { handle.draw_gui(fragment); }
  auto draw_shader_inspector(ShaderHandle handle) { handle.draw_gui(); }
  auto draw_mesh_inspector(MeshHandle handle) { handle.draw_gui(); }
  auto draw_material_inspector(MaterialHandle handle) { handle.draw_gui(); }

  namespace gfx_content_gui {

    struct AssetsBrowser {
      // Options
      bool            ShowTypeOverlay = true;
      bool            AllowSorting = true;
      bool            AllowDragUnselected = false;
      bool            AllowBoxSelect = true;
      float           IconSize = 96.0f;
      int             IconSpacing = 10;
      int             IconHitSpacing = 4;         // Increase hit-spacing if you want to make it possible to clear or box-select from gaps. Some spacing is required to able to amend with Shift+box-select. Value is small in Explorer.
      bool            StretchSpacing = true;

      //ExampleSelectionWithDeletion Selection;     // Our selection (ImGuiSelectionBasicStorage + helper funcs to handle deletion)
      ImGuiID         NextItemId = 0;             // Unique identifier when creating new items
      bool            RequestDelete = false;      // Deferred deletion request
      bool            RequestSort = false;        // Deferred sort request
      float           ZoomWheelAccum = 0.0f;      // Mouse wheel accumulator to handle smooth wheels better

      // Calculated sizes for layout, output of UpdateLayoutSizes(). Could be locals but our code is simpler this way.
      ImVec2          LayoutItemSize;
      ImVec2          LayoutItemStep;             // == LayoutItemSize + LayoutItemSpacing
      float           LayoutItemSpacing = 0.0f;
      float           LayoutSelectableSpacing = 0.0f;
      float           LayoutOuterPadding = 0.0f;
      int             LayoutColumnCount = 0;
      int             LayoutLineCount = 0;

      int32_t         SelectedID = -1;

      void UpdateLayoutSizes(float avail_width, size_t item_size) {
        // Layout: when not stretching: allow extending into right-most spacing.
        LayoutItemSpacing = (float)IconSpacing;
        if (StretchSpacing == false)
          avail_width += floorf(LayoutItemSpacing * 0.5f);

        // Layout: calculate number of icon per line and number of lines
        LayoutItemSize = ImVec2(floorf(IconSize), floorf(IconSize));
        LayoutColumnCount = std::max((int)(avail_width / (LayoutItemSize.x + LayoutItemSpacing)), 1);
        LayoutLineCount = (item_size + LayoutColumnCount - 1) / LayoutColumnCount;

        // Layout: when stretching: allocate remaining space to more spacing. Round before division, so item_spacing may be non-integer.
        if (StretchSpacing && LayoutColumnCount > 1)
          LayoutItemSpacing = floorf(avail_width - LayoutItemSize.x * LayoutColumnCount) / LayoutColumnCount;

        LayoutItemStep = ImVec2(LayoutItemSize.x + LayoutItemSpacing, LayoutItemSize.y + LayoutItemSpacing);
        LayoutSelectableSpacing = std::max(floorf(LayoutItemSpacing) - IconHitSpacing, 0.0f);
        LayoutOuterPadding = floorf(LayoutItemSpacing * 0.5f);
      }

      template<class TResource, class TLoader>
      void draw(ex::resource_cache<TResource, TLoader>& cache,
        std::function<void(ex::resource<TResource>)> callback_click = {},
        std::function<bool(TResource&)> callback_preview = {}) {
        //ImGui::PushItemWidth(ImGui::GetFontSize() * 10);
        //ImGui::SeparatorText("Contents");
        //ImGui::Checkbox("Show Type Overlay", &ShowTypeOverlay);
        //ImGui::Checkbox("Allow Sorting", &AllowSorting);
        //ImGui::SeparatorText("Selection Behavior");
        //ImGui::Checkbox("Allow dragging unselected item", &AllowDragUnselected);
        //ImGui::Checkbox("Allow box-selection", &AllowBoxSelect);
        //ImGui::SeparatorText("Layout");
        //ImGui::SliderFloat("Icon Size", &IconSize, 16.0f, 128.0f, "%.0f");
        ////ImGui::SameLine(); HelpMarker("Use CTRL+Wheel to zoom");
        //ImGui::SliderInt("Icon Spacing", &IconSpacing, 0, 32);
        //ImGui::SliderInt("Icon Hit Spacing", &IconHitSpacing, 0, 32);
        //ImGui::Checkbox("Stretch Spacing", &StretchSpacing);
        //ImGui::PopItemWidth();

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowContentSize(ImVec2(0.0f, LayoutOuterPadding + LayoutLineCount * (LayoutItemSize.y + LayoutItemSpacing)));
        if (ImGui::BeginChild("Assets", ImVec2(0.0f, -ImGui::GetTextLineHeightWithSpacing()), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoMove))
        {
          ImDrawList* draw_list = ImGui::GetWindowDrawList();

          const float avail_width = ImGui::GetContentRegionAvail().x;
          UpdateLayoutSizes(avail_width, cache.size());

          // Calculate and store start position.
          ImVec2 start_pos = ImGui::GetCursorScreenPos();
          start_pos = ImVec2(start_pos.x + LayoutOuterPadding, start_pos.y + LayoutOuterPadding);
          ImGui::SetCursorScreenPos(start_pos);

          // Multi-select
          ImGuiMultiSelectFlags ms_flags = ImGuiMultiSelectFlags_ClearOnEscape | ImGuiMultiSelectFlags_ClearOnClickVoid;

          // - Enable box-select (in 2D mode, so that changing box-select rectangle X1/X2 boundaries will affect clipped items)
          if (AllowBoxSelect)
            ms_flags |= ImGuiMultiSelectFlags_BoxSelect2d;

          // - This feature allows dragging an unselected item without selecting it (rarely used)
          if (AllowDragUnselected)
            ms_flags |= ImGuiMultiSelectFlags_SelectOnClickRelease;

          // - Enable keyboard wrapping on X axis
          // (FIXME-MULTISELECT: We haven't designed/exposed a general nav wrapping api yet, so this flag is provided as a courtesy to avoid doing:
          //    ImGui::NavMoveRequestTryWrapping(ImGui::GetCurrentWindow(), ImGuiNavMoveFlags_WrapX);
          // When we finish implementing a more general API for this, we will obsolete this flag in favor of the new system)
          ms_flags |= ImGuiMultiSelectFlags_NavWrapX;

          ImGuiMultiSelectIO* ms_io = ImGui::BeginMultiSelect(ms_flags);

          //// Use custom selection adapter: store ID in selection (recommended)
          //Selection.UserData = this;
          //Selection.AdapterIndexToStorageId = [](ImGuiSelectionBasicStorage* self_, int idx) { ExampleAssetsBrowser* self = (ExampleAssetsBrowser*)self_->UserData; return self->Items[idx].ID; };
          //Selection.ApplyRequests(ms_io);

          //const bool want_delete = (ImGui::Shortcut(ImGuiKey_Delete, ImGuiInputFlags_Repeat) && (Selection.Size > 0)) || RequestDelete;
          //const int item_curr_idx_to_focus = want_delete ? Selection.ApplyDeletionPreLoop(ms_io, Items.Size) : -1;
          //RequestDelete = false;

          // Push LayoutSelectableSpacing (which is LayoutItemSpacing minus hit-spacing, if we decide to have hit gaps between items)
          // Altering style ItemSpacing may seem unnecessary as we position every items using SetCursorScreenPos()...
          // But it is necessary for two reasons:
          // - Selectables uses it by default to visually fill the space between two items.
          // - The vertical spacing would be measured by Clipper to calculate line height if we didn't provide it explicitly (here we do).
          ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(LayoutSelectableSpacing, LayoutSelectableSpacing));

          // Rendering parameters
          const ImU32 icon_type_overlay_colors[3] = { 0, IM_COL32(200, 70, 70, 255), IM_COL32(70, 170, 70, 255) };
          const ImU32 icon_bg_color = ImGui::GetColorU32(IM_COL32(35, 35, 35, 220));
          const ImVec2 icon_type_overlay_size = ImVec2(4.0f, 4.0f);
          const bool display_label = (LayoutItemSize.x >= ImGui::CalcTextSize("999").x);

          const int column_count = LayoutColumnCount;

          int item_idx = 0;
          int line_idx = 0;
          for (auto [id, item] : cache) {
            if (item->m_countDown < 0) continue;
            ImGui::PushID((int)id);

            // Compute item position
            ImVec2 pos = ImVec2(start_pos.x + (item_idx % column_count) * LayoutItemStep.x,
              start_pos.y + line_idx * LayoutItemStep.y);
            ImGui::SetCursorScreenPos(pos);

            ImGui::SetNextItemSelectionUserData(item_idx);
            bool item_is_selected = (id == SelectedID);
            bool item_is_visible = ImGui::IsRectVisible(LayoutItemSize);

            // Optional: draw background rectangle manually
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 box_min = pos;
            ImVec2 box_max = ImVec2(pos.x + LayoutItemSize.x, pos.y + LayoutItemSize.y);

            // Draw background (or border) rectangle
            ImU32 rect_color = item_is_selected ? IM_COL32(100, 150, 250, 255) : IM_COL32(60, 60, 60, 255);
            draw_list->AddRectFilled(box_min, box_max, rect_color, 4.0f); // Rounded corners optional
            draw_list->AddRect(box_min, box_max, IM_COL32(200, 200, 200, 255)); // Border
            // Handle selection first (before drawing image)
            ImGui::SetCursorScreenPos(pos);
            bool selected = ImGui::Selectable("##item", item_is_selected, ImGuiSelectableFlags_None, LayoutItemSize);
            if (selected) {
              SelectedID = (SelectedID == id) ? -1 : id;
              if (SelectedID == id)
                callback_click(item);
            }

            // Now draw everything visually *after* interaction
            draw_list->AddRectFilled(box_min, box_max, rect_color, 4.0f);
            draw_list->AddRect(box_min, box_max, IM_COL32(200, 200, 200, 255));

            ImVec2 image_size = ImVec2(64, 64);
            ImVec2 image_pos = ImVec2(
              pos.x + (LayoutItemSize.x - image_size.x) * 0.5f,
              pos.y + (LayoutItemSize.y - image_size.y) * 0.3f
            );
            ImGui::SetCursorScreenPos(image_pos);
            if (!callback_preview(item)) {
              ImVec2 text_size = ImGui::CalcTextSize("No Img");
              ImVec2 text_pos = ImVec2(
                pos.x + (LayoutItemSize.x - text_size.x) * 0.5f,
                pos.y + (LayoutItemSize.y - text_size.y) * 0.5f
              );
              draw_list->AddText(text_pos, IM_COL32_WHITE, "No Img");
            }

            // 2. Begin drag source ONLY if the item is active or hovered
            if (ImGui::IsItemActive() && ImGui::BeginDragDropSource()) {
              if (ImGui::GetDragDropPayload() == NULL) {
                ImVector<ImGuiID> payload_items;
                payload_items.push_back((ImGuiID)id);
                ImGui::SetDragDropPayload("ASSETS_BROWSER_ITEMS", payload_items.Data, (size_t)payload_items.size_in_bytes());
              }

              const ImGuiPayload* payload = ImGui::GetDragDropPayload();
              int payload_count = (int)payload->DataSize / (int)sizeof(ImGuiID);
              ImGui::Text("%d assets", payload_count);

              ImGui::EndDragDropSource();
            }


            // Optional label below icon
            if (display_label) {
              ImU32 label_col = ImGui::GetColorU32(item_is_selected ? ImGuiCol_Text : ImGuiCol_TextDisabled);
              char label[32];
              sprintf(label, "%d", id);
              ImVec2 label_pos = ImVec2(pos.x, pos.y + LayoutItemSize.y - ImGui::GetFontSize());
              draw_list->AddText(label_pos, label_col, label);
            }

            item_idx++;
            if (item_idx % column_count == 0)
              line_idx++;

            ImGui::PopID();
          }

          ImGui::PopStyleVar(); // ImGuiStyleVar_ItemSpacing

          if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0)) {
            // Is the mouse NOT over any item? We can check if any item was hovered previously.
            //// Since InvisibleButton sets the hovered flag, if no item was hovered, clear selection.
            if (!ImGui::IsItemHovered()) {
              SelectedID = -1;  // clear selection
              ImGui::ClearActiveID();
              //ImGui::SetFocusID(ImGuiID(0), ImGui::GetCurrentWindow()); // optional but helpful

            }              //if (!ImGui::IsAnyItemHovered()) {
              //  SelectedID = -1;  // clear selection
              //}
          }

          //// Context menu
          //if (ImGui::BeginPopupContextWindow())
          //{
          //  ImGui::Text("Selection: %d items", Selection.Size);
          //  ImGui::Separator();
          //  if (ImGui::MenuItem("Delete", "Del", false, Selection.Size > 0))
          //    RequestDelete = true;
          //  ImGui::EndPopup();
          //}

          ms_io = ImGui::EndMultiSelect();
          //Selection.ApplyRequests(ms_io);
          //if (want_delete)
          //  Selection.ApplyDeletionPostLoop(ms_io, Items, item_curr_idx_to_focus);

          // Zooming with CTRL+Wheel
          if (ImGui::IsWindowAppearing())
            ZoomWheelAccum = 0.0f;
          if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f && ImGui::IsKeyDown(ImGuiMod_Ctrl) && ImGui::IsAnyItemActive() == false)
          {
            ZoomWheelAccum += io.MouseWheel;
            if (fabsf(ZoomWheelAccum) >= 1.0f)
            {
              // Calculate hovered item index from mouse location
              // FIXME: Locking aiming on 'hovered_item_idx' (with a cool-down timer) would ensure zoom keeps on it.
              const float hovered_item_nx = (io.MousePos.x - start_pos.x + LayoutItemSpacing * 0.5f) / LayoutItemStep.x;
              const float hovered_item_ny = (io.MousePos.y - start_pos.y + LayoutItemSpacing * 0.5f) / LayoutItemStep.y;
              const int hovered_item_idx = ((int)hovered_item_ny * LayoutColumnCount) + (int)hovered_item_nx;
              //ImGui::SetTooltip("%f,%f -> item %d", hovered_item_nx, hovered_item_ny, hovered_item_idx); // Move those 4 lines in block above for easy debugging

              // Zoom
              IconSize *= powf(1.1f, (float)(int)ZoomWheelAccum);
              IconSize = std::clamp(IconSize, 16.0f, 128.0f);
              ZoomWheelAccum -= (int)ZoomWheelAccum;
              UpdateLayoutSizes(avail_width, cache.size());

              // Manipulate scroll to that we will land at the same Y location of currently hovered item.
              // - Calculate next frame position of item under mouse
              // - Set new scroll position to be used in next ImGui::BeginChild() call.
              float hovered_item_rel_pos_y = ((float)(hovered_item_idx / LayoutColumnCount) + fmodf(hovered_item_ny, 1.0f)) * LayoutItemStep.y;
              hovered_item_rel_pos_y += ImGui::GetStyle().WindowPadding.y;
              float mouse_local_y = io.MousePos.y - ImGui::GetWindowPos().y;
              ImGui::SetScrollY(hovered_item_rel_pos_y - mouse_local_y);
            }
          }
        }
        ImGui::EndChild();
      }
    };
  };

  auto GFXContext::on_draw_gui_resources() noexcept -> void {
    if (ImGui::BeginTabBar("ResourcesList")) {
      if (ImGui::BeginTabItem("Buffer")) {
        if (ImGui::Button("Clean cache")) {
          se::gfx::GFXContext::clean_buffer_cache();
        }

        static gfx_content_gui::AssetsBrowser buffer_browser;
        auto& buffer_cache = Singleton<GFXContext>::instance()->m_buffers;
        std::function<void(ex::resource<se::gfx::Buffer>)> callback_click =
          [](ex::resource<se::gfx::Buffer> item) {
          BufferHandle handle = { item };
          std::function<void()> fn = std::bind(&draw_buffer_inspector, handle, nullptr);
          se::editor::EditorContext::set_inspector_callback(fn);
        };

        std::function<bool(se::gfx::Buffer&)> callback_preview =
          [](se::gfx::Buffer&) { return false; };

        buffer_browser.draw<Buffer, BufferLoader>(buffer_cache, callback_click, callback_preview);

        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Texture")) {
        if (ImGui::Button("Clean cache")) {
          se::gfx::GFXContext::clean_texture_cache();
        }

        static gfx_content_gui::AssetsBrowser texture_browser;
        auto& texture_cache = Singleton<GFXContext>::instance()->m_textures;
        std::function<void(ex::resource<se::gfx::Texture>)> callback_click =
          [](ex::resource<se::gfx::Texture> item) {
          TextureHandle handle = { item };
          editor::IFragment* frag =
            Singleton<editor::EditorContext>::instance()->m_fragmentPool.
            register_fragment<editor::ImageInspectorFragment>(handle);
          std::function<void()> fn = std::bind(&draw_texture_inspector, handle, frag);
          se::editor::EditorContext::set_inspector_callback(fn);
        };

        std::function<bool(se::gfx::Texture&)> callback_preview =
          [](se::gfx::Texture& texture) {
          std::vector<rhi::BarrierDescriptor> barriers = texture.consume(gfx::Texture::ConsumeEntry()
            .add_stage(rhi::PipelineStageEnum::FRAGMENT_SHADER_BIT)
            .set_layout(rhi::TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL)
            .set_access(rhi::AccessFlagEnum::SHADER_READ_BIT));
          for (int i = 0; i < barriers.size(); ++i)
            editor::ImGuiContext::m_encoder->pipeline_barrier(barriers[i]);

          ImVec2 image_size = ImVec2(64, 64); // Thumbnail size
          //ImVec2 image_pos = ImVec2(
          //  pos.x + (LayoutItemSize.x - image_size.x) * 0.5f,
          //  pos.y + (LayoutItemSize.y - image_size.y) * 0.5f
          //);

          //ImGui::SetCursorScreenPos(image_pos);
          //ImGui::Image(item.preview_texture, image_size);

          ImGui::Image(texture.get_imgui_texture()->get_texture_id(), image_size,
            { 0, 0 }, { 1, 1 });
          
          return true; 
        };

        texture_browser.draw<Texture, TextureLoader>(texture_cache, callback_click, callback_preview);

        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Shader")) {
        if (ImGui::Button("Clean cache")) {
          se::gfx::GFXContext::clean_shader_cache();
        }

        static gfx_content_gui::AssetsBrowser shader_browser;
        auto& shader_cache = Singleton<GFXContext>::instance()->m_shaders;

        std::function<void(ex::resource<se::gfx::ShaderModule>)> callback_click = 
          [](ex::resource<se::gfx::ShaderModule> item) {
          ShaderHandle handle = { item };
          std::function<void()> fn = std::bind(&draw_shader_inspector, handle);
          se::editor::EditorContext::set_inspector_callback(fn);
        };

        std::function<bool(se::gfx::ShaderModule&)> callback_preview =
          [](se::gfx::ShaderModule&) { return false; };

        shader_browser.draw<ShaderModule, ShaderLoader>(shader_cache, callback_click, callback_preview);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Mesh")) {
        if (ImGui::Button("Clean cache")) {
          //se::gfx::GFXContext::clean_shader_cache();
        }

        static gfx_content_gui::AssetsBrowser mesh_browser;
        auto& mesh_cache = Singleton<GFXContext>::instance()->m_meshs;

        std::function<void(ex::resource<se::gfx::Mesh>)> callback_click =
          [](ex::resource<se::gfx::Mesh> item) {
          MeshHandle handle = { item };
          std::function<void()> fn = std::bind(&draw_mesh_inspector, handle);
          se::editor::EditorContext::set_inspector_callback(fn);
        };

        std::function<bool(se::gfx::Mesh&)> callback_preview =
          [](se::gfx::Mesh&) { return false; };

        mesh_browser.draw<Mesh, MeshLoader>(mesh_cache, callback_click, callback_preview);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Material")) {
        if (ImGui::Button("Clean cache")) {
          //se::gfx::GFXContext::clean_shader_cache();
        }

        static gfx_content_gui::AssetsBrowser material_browser;
        auto& material_cache = Singleton<GFXContext>::instance()->m_materials;

        std::function<void(ex::resource<se::gfx::Material>)> callback_click =
          [](ex::resource<se::gfx::Material> item) {
          MaterialHandle handle = { item };
          std::function<void()> fn = std::bind(&draw_material_inspector, handle);
          se::editor::EditorContext::set_inspector_callback(fn);
        };

        std::function<bool(se::gfx::Material&)> callback_preview =
          [](se::gfx::Material&) { return false; };

        material_browser.draw<Material, MaterialLoader>(material_cache, callback_click, callback_preview);
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  }

  auto GFXContext::clean_cache() noexcept -> void {
    clean_texture_cache();
  }

  auto GFXContext::clean_buffer_cache() noexcept -> void {
    auto& buffers = Singleton<GFXContext>::instance()->m_buffers;
    for (auto it = buffers.begin(); it != buffers.end(); ) {
      if (it->second.handle().use_count() <= 2) {
        it->second->m_countDown--;
        if (it->second->m_countDown < -5)
          it = buffers.erase(it); // erase returns the next iterator
      }
      else {
        ++it;
      }
    }
  }
  
  auto GFXContext::clean_texture_cache() noexcept -> void {
    auto& textures = Singleton<GFXContext>::instance()->m_textures;
    for (auto it = textures.begin(); it != textures.end(); ) {
      if (it->second.handle().use_count() <= 2) {
        it->second->m_countDown--;
        if (it->second->m_countDown < -5)
          it = textures.erase(it); // erase returns the next iterator
        else
          ++it;
      }
      else {
        ++it;
      }
    }

  }

  auto GFXContext::clean_shader_cache() noexcept -> void {
    auto& shaders = Singleton<GFXContext>::instance()->m_shaders;
    for (auto it = shaders.begin(); it != shaders.end(); ) {
      if (it->second.handle().use_count() <= 2) {
        it = shaders.erase(it); // erase returns the next iterator
      }
      else {
        ++it;
      }
    }

  }

  inline uint64_t hash(rhi::SamplerDescriptor const& desc) {
    uint64_t hashed_value = 0;
    hashed_value |= (uint64_t)(desc.addressModeU) << 62;
    hashed_value |= (uint64_t)(desc.addressModeV) << 60;
    hashed_value |= (uint64_t)(desc.addressModeW) << 58;
    hashed_value |= (uint64_t)(desc.magFilter) << 57;
    hashed_value |= (uint64_t)(desc.minFilter) << 56;
    hashed_value |= (uint64_t)(desc.mipmapFilter) << 55;
    hashed_value |= (uint64_t)(desc.compare) << 50;
    return hashed_value;
  }

  auto GFXContext::create_sampler_desc(
    rhi::SamplerDescriptor const& desc
  ) noexcept -> SamplerHandle {
    const uint64_t id = hash(desc);
    auto ret = Singleton<GFXContext>::instance()->m_samplers.load(
      id, SamplerLoader::from_desc_tag{}, desc);
    return SamplerHandle{ ret.first->second };
  }

  auto GFXContext::create_sampler_desc(
    rhi::AddressMode address, rhi::FilterMode filter, rhi::MipmapFilterMode mipmap
  ) noexcept -> SamplerHandle {
    rhi::SamplerDescriptor desc;
    desc.addressModeU = address;
    desc.addressModeV = address;
    desc.addressModeW = address;
    desc.magFilter = filter;
    desc.minFilter = filter;
    desc.mipmapFilter = mipmap;
    const uint64_t id = hash(desc);
    auto ret = Singleton<GFXContext>::instance()->m_samplers.load(
      id, SamplerLoader::from_desc_tag{}, desc);
    return SamplerHandle{ ret.first->second };
  }

  auto GFXContext::load_shader_spirv(
    se::MiniBuffer* buffer,
    rhi::ShaderStageEnum stage
  ) noexcept -> ShaderHandle {
    std::string_view sv(static_cast<const char*>(buffer->m_data), buffer->m_size);
    UID const ruid = Resources::query_string_uid(sv);
    auto ret = Singleton<GFXContext>::instance()->m_shaders.load(
      ruid, ShaderLoader::from_spirv_tag{}, buffer, stage);
    // true only if the resource was not already present
    const bool loaded = ret.second;
    // takes the resource handle pointed to by the returned iterator
    entt::resource<ShaderModule> res = ret.first->second;
    return ShaderHandle{ ret.first->second };
  }

  auto GFXContext::load_shader_slang(
    std::string const& path,
    std::vector<std::pair<std::string, rhi::ShaderStageEnum>> const& entrypoints,
    std::vector<std::pair<char const*, char const*>> const& macros,
    bool glsl_intermediate
  ) noexcept -> std::vector<ShaderHandle> {
    slang_inline::SlangSession session(path, macros, glsl_intermediate);
    return session.load(entrypoints);
  }

  namespace shaders {

    inline auto combineResourceFlags(
      Flags<ShaderReflection::ResourceEnum> a,
      Flags<ShaderReflection::ResourceEnum> b) noexcept
      -> Flags<ShaderReflection::ResourceEnum> {
      Flags<ShaderReflection::ResourceEnum> r = 0;
      if ((a | ShaderReflection::ResourceEnum::NotReadable) &&
          (b | ShaderReflection::ResourceEnum::NotReadable))
        r |= ShaderReflection::ResourceEnum::NotReadable;
      if ((a | ShaderReflection::ResourceEnum::NotWritable) &&
          (b | ShaderReflection::ResourceEnum::NotWritable))
        r |= ShaderReflection::ResourceEnum::NotWritable;
      return r;
    }

    inline auto compare_pushconstant(
      ShaderReflection::PushConstantEntry& a,
      ShaderReflection::PushConstantEntry& b) noexcept -> bool {
      return a.offset < b.offset;
    }

    inline auto rearrange_pushconstant(
      ShaderReflection& reflection) noexcept
      -> void {
      if (reflection.pushConstant.size() == 0) return;
      std::sort(reflection.pushConstant.begin(), reflection.pushConstant.end(),
        compare_pushconstant);
      while (true) {
        bool should_break = false;
        for (auto iter = reflection.pushConstant.begin();
          iter != reflection.pushConstant.end();) {
          // break when reach the end of the push constants
          auto iter_next = iter + 1;
          if (iter_next == reflection.pushConstant.end()) {
            should_break = true;
            break;
          }
          else {
            if (iter->offset + iter->range <= iter_next->offset) {
              iter->range = iter_next->offset + iter_next->range - iter->offset;
              reflection.pushConstant.erase(iter_next);
            }
            else {
              iter++;
            }
          }
        }
        if (should_break) break;
      }

      reflection.pushConstant[0].range += reflection.pushConstant[0].offset;
      reflection.pushConstant[0].offset = 0;
    }

    auto SPIRV_TO_Reflection(MiniBuffer* code, rhi::ShaderStageEnum stage) noexcept
      -> ShaderReflection {
      ShaderReflection reflection = {};
      // add resource entry
      auto addResourceEntry = [&](ShaderReflection::ResourceEntry const& entry,
        int set, int binding) {
          if (reflection.bindings.size() <= set) reflection.bindings.resize(set + 1);
          if (reflection.bindings[set].size() <= binding)
            reflection.bindings[set].resize(binding + 1);
          reflection.bindings[set][binding] = entry;
      };

      // Generate reflection data for a shader
      SpvReflectShaderModule module;
      SpvReflectResult result =
        spvReflectCreateShaderModule(code->m_size, code->m_data, &module);
      assert(result == SPV_REFLECT_RESULT_SUCCESS);

      // Enumerate and extract shader's input variables
      uint32_t var_count = 0;
      result = spvReflectEnumerateInputVariables(&module, &var_count, NULL);
      assert(result == SPV_REFLECT_RESULT_SUCCESS);
      SpvReflectInterfaceVariable** input_vars =
        (SpvReflectInterfaceVariable**)malloc(
          var_count * sizeof(SpvReflectInterfaceVariable*));
      result = spvReflectEnumerateInputVariables(&module, &var_count, input_vars);
      assert(result == SPV_REFLECT_RESULT_SUCCESS);

      // Output variables, descriptor bindings, descriptor sets, and push
      // constants can be enumerated and extracted using a similar mechanism.
      for (int i = 0; i < module.descriptor_binding_count; ++i) {
        auto const& desc_set = module.descriptor_sets[i];
        for (int j = 0; j < desc_set.binding_count; ++j) {
          auto const& binding = desc_set.bindings[j];
          Flags<ShaderReflection::ResourceEnum> flag = 0;
          ShaderReflection::ResourceEntry entry;
          switch (binding->descriptor_type) {
          case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
            entry.type = ShaderReflection::ResourceType::Sampler; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            entry.type = ShaderReflection::ResourceType::SampledImages; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            entry.type = ShaderReflection::ResourceType::ReadonlyImage; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            entry.type = ShaderReflection::ResourceType::StorageImages; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            entry.type = ShaderReflection::ResourceType::UniformBuffer; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            entry.type = ShaderReflection::ResourceType::StorageBuffer; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            entry.type = ShaderReflection::ResourceType::UniformBuffer; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            entry.type = ShaderReflection::ResourceType::StorageBuffer; break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            entry.type = ShaderReflection::ResourceType::AccelerationStructure;
            entry.arraySize = binding->count;
            break;
          case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
          case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
          case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
          default:
            se::error("SPIRV-Reflect :: Unexpected resource type");
            break;
          }
          if (desc_set.bindings[j]->array.dims_count >= 1 
            && entry.arraySize == 1
          ) {
            entry.arraySize = 99;
          }
          entry.flags = flag;
          entry.stages = stage;
          addResourceEntry(entry, binding->set, binding->binding);
        }
      }
      // Push constants
      for (uint32_t i = 0; i < module.push_constant_block_count; ++i) {
        auto const& block = module.push_constant_blocks[i];
        reflection.pushConstant.emplace_back(
          ShaderReflection::PushConstantEntry{
              i, uint32_t(block.offset), uint32_t(block.size), stage });
      }
      // Destroy the reflection data when no longer required.
      spvReflectDestroyShaderModule(&module);
      // rearrange push constants
      rearrange_pushconstant(reflection);
      return reflection;
    }
  }

  bounds3 Medium::MajorantGrid::voxel_bounds(int x, int y, int z) const {
    vec3 p0(float(x) / res.x, float(y) / res.y, float(z) / res.z);
    vec3 p1(float(x + 1) / res.x, float(y + 1) / res.y, float(z + 1) / res.z);
    return bounds3(p0, p1);
  }

  void Medium::MajorantGrid::set(int x, int y, int z, float v) {
    voxels[x + res.x * (y + res.y * z)] = v;
  }

  float Medium::SampledGrid::max_value(const bounds3& bounds) const {
    vec3 ps[2] = { vec3(bounds.pMin.x * nx - .5f, bounds.pMin.y * ny - .5f,
                           bounds.pMin.z * nz - .5f),
                   vec3(bounds.pMax.x * nx - .5f, bounds.pMax.y * ny - .5f,
                           bounds.pMax.z * nz - .5f) };
    ivec3 pi[2] = { max(ivec3(floor(ps[0])), ivec3(0, 0, 0)),
                     min(ivec3(floor(ps[1])) + ivec3(1, 1, 1),
                         ivec3(nx - 1, ny - 1, nz - 1)) };

    float maxValue;
    if (gridChannel == 1) {
      maxValue = lookup(ivec3(pi[0]));
      for (int z = pi[0].z; z <= pi[1].z; ++z)
        for (int y = pi[0].y; y <= pi[1].y; ++y)
          for (int x = pi[0].x; x <= pi[1].x; ++x)
            maxValue = std::max(maxValue, lookup(ivec3(x, y, z)));
    }
    else if (gridChannel == 3) {
      maxValue = maxComponent(lookup3(ivec3(pi[0])));
      for (int z = pi[0].z; z <= pi[1].z; ++z)
        for (int y = pi[0].y; y <= pi[1].y; ++y)
          for (int x = pi[0].x; x <= pi[1].x; ++x)
            maxValue = std::max(maxValue, maxComponent(lookup3(ivec3(x, y, z))));

    }
    return maxValue;
  }

  float Medium::SampledGrid::lookup(const ivec3& p) const {
    ibounds3 sampleBounds(ivec3(0, 0, 0), ivec3(nx, ny, nz));
    //if (!InsideExclusive(p, sampleBounds))
    //  return convert(T{});
    return values[(p.z * ny + p.y) * nx + p.x];
  }

  vec3 Medium::SampledGrid::lookup3(const ivec3& p) const {
    ibounds3 sampleBounds(ivec3(0, 0, 0), ivec3(nx, ny, nz));
    return vec3{
    values[((p.z * ny + p.y) * nx + p.x) * 3 + 0],
    values[((p.z * ny + p.y) * nx + p.x) * 3 + 1],
    values[((p.z * ny + p.y) * nx + p.x) * 3 + 2] };
  }

  ShaderLoader::result_type ShaderLoader::operator()(
    ShaderLoader::from_spirv_tag,
    MiniBuffer* buffer,
    rhi::ShaderStageEnum stage
  ) {
    rhi::ShaderModuleDescriptor desc;
    desc.code = buffer;
    desc.name = "main";
    desc.stage = stage;
    auto ptr = std::make_shared<ShaderModule>();
    ptr->m_shaderModule = GFXContext::device()->create_shader_module(desc);
    ptr->m_reflection = shaders::SPIRV_TO_Reflection(desc.code, desc.stage);
    return ptr;
  }

  MeshLoader::result_type MeshLoader::operator()(MeshLoader::from_empty_tag) {
    auto ptr = std::make_shared<Mesh>();
    return ptr;
  }

  auto GFXContext::create_mesh_empty() noexcept -> MeshHandle {
    UID const ruid = se::Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_meshs.load(ruid, MeshLoader::from_empty_tag{});
    ret.first->second->m_uid = ruid;
    return MeshHandle{ ret.first->second };
  }
  
  auto Material::draw_gui(editor::IFragment* fragment) noexcept -> void {
    MaterialInterpreterManager::draw_gui(this, m_packet.bxdfType);
  }

  MaterialLoader::result_type MaterialLoader::operator()(MaterialLoader::from_empty_tag) {
    auto ptr = std::make_shared<Material>();
    return ptr;
  }

  MediumLoader::result_type MediumLoader::operator()(MediumLoader::from_empty_tag) {
    auto ptr = std::make_shared<Medium>();
    return ptr;
  }

  auto GFXContext::create_material_empty() noexcept -> MaterialHandle {
    UID const ruid = se::Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_materials.load(ruid, MaterialLoader::from_empty_tag{});
    ret.first->second->m_uid = ruid;
    return MaterialHandle{ ret.first->second };
  }
  
  auto GFXContext::create_medium_empty() noexcept -> MediumHandle {
    UID const ruid = se::Resources::query_runtime_uid();
    auto ret = Singleton<GFXContext>::instance()->m_mediums.load(ruid, MediumLoader::from_empty_tag{});
    ret.first->second->m_uid = ruid;
    return MediumHandle{ ret.first->second };
  }

}
}