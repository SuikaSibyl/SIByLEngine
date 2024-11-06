#define DLIB_EXPORT
#include <se.gfx.hpp>
#undef DLIB_EXPORT
#include "spirv_reflect/spirv_reflect.h"
#include <slang.h>
#include <slang-com-ptr.h>
#include <filesystem>
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>
#include <se.image.hpp>

namespace se::gfx {
// ===========================================================================
// Base class definition


// Base class definition
// ===========================================================================
// Shader Resource

inline auto combineResourceFlags(
  ShaderReflection::ResourceFlags a,
  ShaderReflection::ResourceFlags b) noexcept
  -> ShaderReflection::ResourceFlags {
  ShaderReflection::ResourceFlags r = 0;
  if ((a | uint32_t(ShaderReflection::ResourceFlag::NotReadable)) != 0 &&
      (b | uint32_t(ShaderReflection::ResourceFlag::NotReadable)) != 0)
    r |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
  if ((a | uint32_t(ShaderReflection::ResourceFlag::NotWritable)) != 0 &&
      (b | uint32_t(ShaderReflection::ResourceFlag::NotWritable)) != 0)
    r |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
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
        } else {
          if (iter->offset + iter->range <= iter_next->offset) {
            iter->range = iter_next->offset + iter_next->range - iter->offset;
            reflection.pushConstant.erase(iter_next);
          } else {
            iter++;
          }
        }
      }
      if (should_break) break;
    }

    reflection.pushConstant[0].range += reflection.pushConstant[0].offset;
    reflection.pushConstant[0].offset = 0;
}

auto SPIRV_TO_Reflection(se::buffer* code, rhi::ShaderStageBit stage) noexcept
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
      spvReflectCreateShaderModule(code->size, code->data, &module);
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
      ShaderReflection::ResourceFlags flag =
          uint32_t(ShaderReflection::ResourceFlag::None);
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
          entry.type = ShaderReflection::ResourceType::AccelerationStructure; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        default: 
          root::print::error("SPIRV-Reflect :: Unexpected resource type");  
          break;
      }
      if (desc_set.bindings[j]->array.dims_count >= 1) {
        entry.arraySize = 1000;
      }
      entry.flags = flag;
      entry.stages = uint32_t(stage);
      addResourceEntry(entry, binding->set, binding->binding);
    }
  }
  // Push constants
  for (uint32_t i = 0; i < module.push_constant_block_count; ++i) {
    auto const& block = module.push_constant_blocks[i];
    reflection.pushConstant.emplace_back(
        ShaderReflection::PushConstantEntry{
            i, uint32_t(block.offset), uint32_t(block.size), (uint32_t)stage});
  }
  // Destroy the reflection data when no longer required.
  spvReflectDestroyShaderModule(&module);
  // rearrange push constants
  rearrange_pushconstant(reflection);
  return reflection;
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
          rhi::BufferBindingLayout{rhi::BufferBindingType::UNIFORM}});
    else if (bind.type == ResourceType::StorageBuffer)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages,
          rhi::BufferBindingLayout{rhi::BufferBindingType::STORAGE}});
    else if (bind.type == ResourceType::StorageImages) {
      rhi::BindGroupLayoutEntry entry{i, bind.stages, rhi::StorageTextureBindingLayout{}};
      entry.array_size = bind.arraySize;
      descriptor.entries.emplace_back(entry);
    }
    else if (bind.type == ResourceType::AccelerationStructure)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages, rhi::AccelerationStructureBindingLayout{}});
    else if (bind.type == ResourceType::SampledImages)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages, rhi::BindlessTexturesBindingLayout{}});
    else if (bind.type == ResourceType::ReadonlyImage) {
      rhi::BindGroupLayoutEntry entry{i, bind.stages,
                                      rhi::TextureBindingLayout{}};
      entry.array_size = bind.arraySize;
      descriptor.entries.emplace_back(entry);
    }
    else if (bind.type == ResourceType::Sampler)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages, rhi::SamplerBindingLayout{}});
  }
  return descriptor;
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
    } else {
      for (int binding = 0; binding < reflection.bindings[set].size();
           ++binding) {
        if (added.bindings[set].size() <= binding) {
          added.bindings[set].resize(binding + 1);
          added.bindings[set][binding] = reflection.bindings[set][binding];
        } 
        else if (reflection.bindings[set][binding].type ==
                   ResourceType::Undefined) {
            // SKIP
        } else if (added.bindings[set][binding].type ==
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
    } else {
      added.pushConstant.push_back(reflection.pushConstant[i]);
    }
  }
  return added;
}

auto ShaderModule::getName() const noexcept -> char const* {
    return shaderModule->getName().c_str();
}

// loader
// -------------------------------------------------

ShaderModuleLoader::result_type ShaderModuleLoader::operator()(
  ShaderModuleLoader::from_spirv_tag, 
  se::buffer* buffer,
  rhi::ShaderStageBit stage) {
  rhi::ShaderModuleDescriptor desc;
  desc.code = buffer;
  desc.name = "main";
  desc.stage = stage;
  auto ptr = std::make_shared<ShaderModule>();
  ptr->shaderModule = GFXContext::device->createShaderModule(desc);
  ptr->reflection = SPIRV_TO_Reflection(desc.code, desc.stage);
  return ptr;
}

ShaderModuleLoader::result_type ShaderModuleLoader::operator()(
  ShaderModuleLoader::from_glsl_tag,
  std::string const& filepath, rhi::ShaderStageBit stage,
  std::vector<std::string> const& argv) {
  std::filesystem::path path(filepath);
  std::filesystem::path filename = path.filename();
  std::filesystem::path extension = path.extension();
  filename = std::filesystem::path(filename.string().substr(
      0, filename.string().size() - extension.string().size()));
  std::string extension_str =
      extension.string().substr(1, extension.string().length());
  std::filesystem::path parent_path = path.parent_path();
  std::string cmdLine("glslangValidator --target-env vulkan1.2");
  std::string spirv_path = (parent_path / filename).string() + "_" + extension_str + ".spv";
  cmdLine += " " + std::string(filepath);
  cmdLine += " -o " + spirv_path + " ";
  for (int i = 2; i < argv.size(); ++i) {
    cmdLine += " " + std::string(argv[i]);
  }

  auto const& engine_shader_path = RuntimeConfig::get()->string_array_property("shader_path");
  for (auto& shader_path : engine_shader_path) {
    cmdLine += " -I" + std::filesystem::absolute(std::filesystem::path(shader_path)).string();
  }
  se::root::print::debug(cmdLine);

  system(cmdLine.c_str());
  // load the compiled spir-v
  se::buffer spirv_code;
  se::syncReadFile(spirv_path.c_str(), spirv_code);
  return this->operator()(from_spirv_tag{}, &spirv_code, stage);
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

SamplerLoader::result_type SamplerLoader::operator()(
    SamplerLoader::from_desc_tag, rhi::SamplerDescriptor const& desc) {
  auto ptr = GFXContext::device->createSampler(desc);
  return ptr;
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

namespace slang_inline {
using Slang::ComPtr;

struct SlangSession {
  SlangSession(
    std::string const& filepath,
    std::vector<std::pair<char const*, char const*>> const& macros = {},
    bool use_glsl_intermediate = false);

  auto load(std::vector<std::pair<std::string, rhi::ShaderStageBit>> const&
    entrypoints) noexcept -> std::vector<ShaderHandle>;

  std::unordered_map<std::string, ShaderReflection::BindingInfo> bindingInfo;
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
      root::print::error("GFX::SLANG::Global session create failed.");
  });
  auto getGlobalSession() noexcept -> slang::IGlobalSession* {
    return slangGlobalSession.get();
  }
  ComPtr<slang::IGlobalSession> slangGlobalSession;
};

inline void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob) {
  if (diagnosticsBlob != nullptr) {
    root::print::error(std::string((const char*)diagnosticsBlob->getBufferPointer()));
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
  } else {
    // set target to spirv glsl460
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("glsl460");  
    //targetDesc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
  }
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;
  // set search path
  std::string parent_path = path.parent_path().string();
  auto const& engine_shader_path = RuntimeConfig::get()->string_array_property("shader_path");
  std::vector<std::string> search_paths_str;
  std::vector<const char*> search_paths;
  for (auto& shader_path : engine_shader_path) {
    search_paths_str.push_back(shader_path);
    search_paths_str.push_back(shader_path + parent_path);
  }
  search_paths_str.push_back(parent_path);
  for (auto& path_str : search_paths_str) {
    search_paths.push_back(path_str.c_str());
  }
  std::filesystem::path input_path = filepath;
  for (auto& path : search_paths_str) {
    auto concate_path = std::filesystem::path(path) / input_path.filename();
    if (std::filesystem::exists(concate_path)) {
      this->filepath = concate_path.string();
      break;
    }
  }
  sessionDesc.searchPaths = search_paths.data();
  sessionDesc.searchPathCount = search_paths.size();
  // push pre-defined macros
  std::vector<slang::PreprocessorMacroDesc> macro_list;
  for (auto const& macro : macros)
    macro_list.emplace_back(
        slang::PreprocessorMacroDesc{macro.first, macro.second});
  sessionDesc.preprocessorMacros = macro_list.data();
  sessionDesc.preprocessorMacroCount = macro_list.size();
  // create slang session
  SlangResult result =
      globalSession->createSession(sessionDesc, session.writeRef());
  if (result != 0) {
    root::print::error("GFX::SLANG::Session create failed.");
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
    ShaderReflection::BindingInfo bindinfo = {
        ShaderReflection::ResourceType::Undefined, 0, 0};
    slang::VariableLayoutReflection* parameter =
        shaderReflection->getParameterByIndex(pp);
    char const* parameterName = parameter->getName();
    slang::ParameterCategory category = parameter->getCategory();
    unsigned index = parameter->getBindingIndex();
    unsigned space = parameter->getBindingSpace();
    bindinfo.binding = index;
    bindinfo.set = space;
    slang::TypeLayoutReflection* typeLayout = parameter->getTypeLayout();
    slang::TypeReflection::Kind kind = typeLayout->getKind();
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
        bindinfo.type = ShaderReflection::ResourceType::Undefined;
        break;
      case slang::BindingType::Unknown:
        bindinfo.type = ShaderReflection::ResourceType::Undefined;
        break;
      case slang::BindingType::CombinedTextureSampler:
        bindinfo.type = ShaderReflection::ResourceType::SampledImages;
        break;
      case slang::BindingType::RayTracingAccelerationStructure:
        bindinfo.type = ShaderReflection::ResourceType::AccelerationStructure;
        break;
      case slang::BindingType::ConstantBuffer:
        bindinfo.type = ShaderReflection::ResourceType::UniformBuffer;
        break;
      case slang::BindingType::RawBuffer:
      case slang::BindingType::MutableRawBuffer:
        bindinfo.type = ShaderReflection::ResourceType::StorageBuffer;
        break;
      case slang::BindingType::MutableTexture:
        bindinfo.type = ShaderReflection::ResourceType::StorageImages;
        break;
      case slang::BindingType::Texture:
        bindinfo.type = ShaderReflection::ResourceType::ReadonlyImage;
        break;
      case slang::BindingType::Sampler:
        bindinfo.type = ShaderReflection::ResourceType::Sampler;
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
      default: root::print::error("SLANG :: Binding not valid");
      break;
    }
    bindingInfo[std::string(parameterName)] = bindinfo;
    float a = 1.f;
    // ...
  }
}

auto SlangSession::load(
    std::vector<std::pair<std::string, rhi::ShaderStageBit>> const&
    entrypoints) noexcept -> std::vector<ShaderHandle> {
  std::vector<ShaderHandle> sms(entrypoints.size());
  // add all entrypoints
  std::vector<ComPtr<slang::IEntryPoint>> entryPointsPtrs(entrypoints.size());
  std::vector<slang::IComponentType*> componentTypes;
  componentTypes.push_back(slangModule);
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    SlangInt32 w = slangModule->getDefinedEntryPointCount();
    SlangResult result = slangModule->findEntryPointByName(
        entrypoints[i].first.c_str(), entryPointsPtrs[i].writeRef());
    if (result != 0) {
      root::print::error("GFX::SLANG::cannot find entrypoint \"" + entrypoints[i].first + "\"");
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
    root::print::error("GFX::SLANG::createCompositeComponentType() failed.");
    return sms;
  }
  ComPtr<slang::IBlob> compiledCode;
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    ComPtr<slang::IBlob> diagnosticsBlob;

    SlangResult result = composedProgram->getEntryPointCode(
        i, 0, compiledCode.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (result != 0) {
      root::print::error("GFX::SLANG::getEntryPointCode() failed.");
      return sms;
    }
    se::buffer spirvcode;
    if (use_glsl_intermediate) {
      // compile SPIR-V from glsl
      se::buffer glslcode;
      glslcode.isReference = true;
      glslcode.data = (void*)(compiledCode->getBufferPointer());
      glslcode.size = compiledCode->getBufferSize();
      std::string suffix = "glsl";
      switch (entrypoints[i].second) {
        case rhi::ShaderStageBit::VERTEX:       suffix = "vert"; break;
        case rhi::ShaderStageBit::FRAGMENT:     suffix = "frag"; break;
        case rhi::ShaderStageBit::COMPUTE:      suffix = "comp"; break;
        case rhi::ShaderStageBit::GEOMETRY:     suffix = "geom"; break;
        case rhi::ShaderStageBit::RAYGEN:       suffix = "rgen"; break;
        case rhi::ShaderStageBit::MISS:         suffix = "rmiss"; break;
        case rhi::ShaderStageBit::CLOSEST_HIT:  suffix = "rchit"; break;
        case rhi::ShaderStageBit::INTERSECTION: suffix = "rint"; break;
        case rhi::ShaderStageBit::ANY_HIT:      suffix = "rahit"; break;
        case rhi::ShaderStageBit::CALLABLE:     suffix = "rcall"; break;
        case rhi::ShaderStageBit::TASK:         suffix = "task"; break;
        case rhi::ShaderStageBit::MESH:         suffix = "mesh"; break;
        default: break;
      }
      std::string glsl_path = filepath.substr(0, filepath.find_last_of('.')+1) + suffix;
      se::syncWriteFile(glsl_path.c_str(), glslcode);
      // create shader module
      sms[i] = GFXContext::load_shader_glsl(glsl_path.c_str(), entrypoints[i].second);
    } else {
      // directly use the compiled SPIR-V
      spirvcode.isReference = true;
      spirvcode.data = (void*)(compiledCode->getBufferPointer());
      spirvcode.size = compiledCode->getBufferSize();
      // create shader module
      sms[i] = GFXContext::load_shader_spirv(&spirvcode, entrypoints[i].second);
    }
  }
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    sms[i].get()->reflection.bindingInfo = bindingInfo;
  }
  return sms;
}
}

auto Buffer::hostToDevice() noexcept -> void {
  // if nothing is on device, create both buffer and previous
  if (buffer == nullptr && previous == nullptr) {
    bool host_buffer_is_empty = false;
    if (host.size() == 0) {
      host_buffer_is_empty = true;
      host.resize(64);
    }
    buffer = GFXContext::device->createDeviceLocalBuffer(
      static_cast<void const*>(host.data()), host.size() * sizeof(unsigned char), usages);
    previous = GFXContext::device->createDeviceLocalBuffer(
      static_cast<void const*>(host.data()), host.size() * sizeof(unsigned char), usages);
    buffer_stamp = host_stamp;
    previous_stamp = host_stamp;
    if (host_buffer_is_empty)host.resize(0);
  }
  // otherwise update the gpu buffer as long as prev is not equal to host
  else if (previous_stamp != host_stamp) {
    previous = std::move(buffer);
    buffer = GFXContext::device->createDeviceLocalBuffer(
      static_cast<void const*>(host.data()), host.size() * sizeof(unsigned char), usages);
    previous_stamp = buffer_stamp;
    buffer_stamp = host_stamp;
  }
}

auto Buffer::deviceToHost() noexcept -> void {
  if (host.size() < buffer->size()) host.resize(buffer->size());
  GFXContext::device->readbackDeviceLocalBuffer(buffer.get(), host.data(), buffer->size());
}

auto Buffer::createDevice() noexcept -> void {
  if (buffer == nullptr || (buffer->size() != host.size() * sizeof(unsigned char))) {
    rhi::BufferDescriptor descriptor;
    descriptor.size = host.size() * sizeof(unsigned char);
    descriptor.usage = usages;
    descriptor.memoryProperties = (uint32_t)rhi::MemoryPropertyBit::DEVICE_LOCAL_BIT;
    descriptor.mappedAtCreation = false;
    buffer = GFXContext::device->createBuffer(descriptor);
  }
}

auto Buffer::getHost() noexcept -> std::vector<unsigned char>& {
  return host;
}

auto Buffer::getDevice() noexcept -> rhi::Buffer* {
  return buffer.get();
}

auto Buffer::getBindingResource() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ rhi::BufferBinding{ buffer.get(), 0, buffer->size() } };
}

auto Buffer::getName() const noexcept -> char const* {
  return buffer->getName().c_str();
}

BufferLoader::result_type BufferLoader::operator()(from_empty_tag) {
  BufferLoader::result_type result = std::make_shared<Buffer>();
  return result;
}

BufferLoader::result_type BufferLoader::operator()(from_desc_tag, rhi::BufferDescriptor desc) {
  BufferLoader::result_type result = std::make_shared<Buffer>();
  result->buffer = GFXContext::device->createBuffer(desc);
  return result;
}

BufferLoader::result_type BufferLoader::operator()(from_gltf_tag, tinygltf::Buffer const& input) {
  auto const& usage_ext = input.extensions.find("usages");
  if (usage_ext == input.extensions.end()) {
    std::string const error = "BufferLoader::from_gltf_tag receive a buffer without \"usages\" extension.";
    root::print::error(error);
    throw error;
  }
  rhi::BufferUsages usages = usage_ext->second.GetNumberAsInt();
  BufferLoader::result_type result = std::make_shared<Buffer>();
  result->buffer = GFXContext::device->createDeviceLocalBuffer(
    static_cast<void const*>(input.data.data()), input.data.size() * sizeof(unsigned char), usages);
  return result;
}

BufferLoader::result_type BufferLoader::operator()(from_host_tag, se::buffer const& input, rhi::BufferUsages usages) {
  BufferLoader::result_type result = std::make_shared<Buffer>();
  result->buffer = GFXContext::device->createDeviceLocalBuffer(
    static_cast<void const*>(input.data), input.size, usages);
  return result;
}

bool ViewIndex::operator==(ViewIndex const& p) const {
  return type == p.type && mostDetailedMip == p.mostDetailedMip &&
    mipCount == p.mipCount && firstArraySlice == p.firstArraySlice &&
    arraySize == p.arraySize;
}

auto Texture::getName() const noexcept -> char const* {
    return texture->getName().c_str();
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

auto Texture::getUAV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> rhi::TextureView* {
  ViewIndex idx = {rhi::TextureViewType::UAV, mipLevel, 0, firstArraySlice,
                   arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)rhi::TextureAspectBit::COLOR_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getRTV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> rhi::TextureView* {
  ViewIndex idx = { rhi::TextureViewType::RTV, mipLevel, 0, firstArraySlice,
                   arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)rhi::TextureAspectBit::COLOR_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getDSV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> rhi::TextureView* {
  ViewIndex idx = { rhi::TextureViewType::DSV, mipLevel, 0, firstArraySlice,
                   arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)rhi::TextureAspectBit::DEPTH_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getSRV(uint32_t mostDetailedMip, uint32_t mipCount,
                     uint32_t firstArraySlice, uint32_t arraySize) noexcept
    -> rhi::TextureView* {
  ViewIndex idx = { rhi::TextureViewType::RTV, mostDetailedMip, mipCount,
                   firstArraySlice, arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  uint32_t aspect = (uint32_t)rhi::TextureAspectBit::COLOR_BIT;
  if (rhi::hasDepthBit(texture->format()))
    aspect = (uint32_t)rhi::TextureAspectBit::DEPTH_BIT;
  if (rhi::hasStencilBit(texture->format()))
    aspect |= (uint32_t)rhi::TextureAspectBit::STENCIL_BIT;

  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, aspect, mostDetailedMip, mipCount,
        firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getWidth() noexcept -> uint32_t {
  switch (type) {
  case se::gfx::Texture::TextureType::vkTexture:
    return texture->width(); break;
  case se::gfx::Texture::TextureType::bufTexture:
    return bufTex.value().packet.dim_0; break;
  default: return 0;
  }
}

auto Texture::getHeight() noexcept -> uint32_t {
  switch (type) {
  case se::gfx::Texture::TextureType::vkTexture:
    return texture->height(); break;
  case se::gfx::Texture::TextureType::bufTexture:
    return bufTex.value().packet.dim_1; break;
  default: return 0;
  }
}

TextureLoader::result_type TextureLoader::operator()(from_desc_tag, rhi::TextureDescriptor const& desc) {
  TextureLoader::result_type result = std::make_shared<Texture>();
  result->texture = GFXContext::device->createTexture(desc);
  return result;
}

TextureLoader::result_type TextureLoader::operator()(from_file_tag, std::filesystem::path const& path) {
  TextureLoader::result_type result = std::make_shared<Texture>();
  std::unique_ptr<image::Texture> host_tex = image::load_image(path);
  // create staging buffer
  rhi::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = host_tex->data_size;
  stagingBufferDescriptor.usage = (uint32_t)rhi::BufferUsageBit::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
    (uint32_t)rhi::MemoryPropertyBit::HOST_VISIBLE_BIT |
    (uint32_t)rhi::MemoryPropertyBit::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<rhi::Buffer> stagingBuffer = GFXContext::device->createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
    memcpy(mapdata, host_tex->getData(), (size_t)stagingBufferDescriptor.size);
    stagingBuffer->unmap();
  }
  std::unique_ptr<rhi::CommandEncoder> commandEncoder = GFXContext::device->createCommandEncoder({nullptr});
  // create texture image
  result->texture = GFXContext::device->createTexture(host_tex->getDescriptor());
  // copy to target
  commandEncoder->pipelineBarrier(rhi::BarrierDescriptor{
    (uint32_t)rhi::PipelineStageBit::TOP_OF_PIPE_BIT,
    (uint32_t)rhi::PipelineStageBit::TRANSFER_BIT,
    (uint32_t)rhi::DependencyType::NONE,
    {}, {}, { rhi::TextureMemoryBarrierDescriptor{
      result->texture.get(),
      rhi::ImageSubresourceRange{(uint32_t)rhi::TextureAspectBit::COLOR_BIT, 0,
                                     host_tex->mip_levels, 0, host_tex->array_layers},
      (uint32_t)rhi::AccessFlagBits::NONE,
      (uint32_t)rhi::AccessFlagBits::TRANSFER_WRITE_BIT,
      rhi::TextureLayout::UNDEFINED,
      rhi::TextureLayout::TRANSFER_DST_OPTIMAL}}});

  for (auto const& subresource : host_tex->subResources) {
    commandEncoder->copyBufferToTexture(
      {subresource.offset, 0, 0, stagingBuffer.get()},
      { result->texture.get(),
      subresource.mip,
      {},
      (uint32_t)rhi::TextureAspectBit::COLOR_BIT},
      {subresource.width, subresource.height, 1});
  }

  commandEncoder->pipelineBarrier(rhi::BarrierDescriptor{
      (uint32_t)rhi::PipelineStageBit::TRANSFER_BIT,
      (uint32_t)rhi::PipelineStageBit::FRAGMENT_SHADER_BIT,
      (uint32_t)rhi::DependencyType::NONE,
      {}, {},
      {rhi::TextureMemoryBarrierDescriptor{
          result->texture.get(),
          rhi::ImageSubresourceRange{(uint32_t)rhi::TextureAspectBit::COLOR_BIT, 0,
                                     host_tex->mip_levels, 0, host_tex->array_layers},
          (uint32_t)rhi::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT,
          rhi::TextureLayout::TRANSFER_DST_OPTIMAL,
          rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});

  GFXContext::device->getGraphicsQueue()->submit({commandEncoder->finish()});
  GFXContext::device->waitIdle();
  return result;
}

TextureLoader::result_type TextureLoader::operator()(from_desc_buf_tag, 
  rhi::TextureDescriptor const& desc, float default_value) {
  TextureLoader::result_type result = std::make_shared<Texture>();
  result->type = Texture::TextureType::bufTexture;
  result->bufTex = DifferentiableParameter{
    int(desc.size.width),
    int(desc.size.height),
    int(desc.arrayLayerCount),
    1, 0, 0,
    default_value
  };
  return result;
}

// Base class definition
// ===========================================================================
// Material Resource

Material::operator tinygltf::Material() const {
  tinygltf::Material material_gltf;
  material_gltf.pbrMetallicRoughness.baseColorFactor = {
    baseOrDiffuseColor.r, baseOrDiffuseColor.g, baseOrDiffuseColor.b, 1.0f };
  material_gltf.doubleSided = doubleSided;
  material_gltf.extensions.emplace("bxdf", bxdf);
  return material_gltf;
}

auto Material::getName() const noexcept -> char const* { return name.c_str(); }

auto Material::getDataPacket() const noexcept -> MaterialPacket {
  MaterialPacket packet;
  packet.bxdf_type = bxdf;
  packet.floatvec_0 = { baseOrDiffuseColor, roughnessFactor };
  packet.floatvec_1 = { emissiveColor, metallicFactor };
  packet.floatvec_2 = floatvec_2;
  return packet;
}

MaterialLoader::result_type MaterialLoader::operator()(from_empty_tag) {
  MaterialLoader::result_type mat = std::make_shared<Material>();
  return mat;
}

auto SerializeData::query_material(Material* mat) noexcept -> int32_t {
  auto iter = materials.find(mat);
  if (iter == materials.end()) {
    int32_t index = materials.size();;
    materials[mat] = index;
    return index;
  } else return iter->second;
}

auto Mesh::getName() const noexcept -> char const* { return name.c_str(); }

Mesh::operator tinygltf::Mesh() const {
  tinygltf::Mesh mesh_gltf;
  for (auto& prim : primitives) {
    //prim.material = 
    //mesh_gltf.primitives.emplace_back(tinygltf::Primitive(prim));
  }
  return mesh_gltf;
}

MeshLoader::result_type MeshLoader::operator()(MeshLoader::from_empty_tag) {
  auto ptr = std::make_shared<Mesh>();
  return ptr;
}

MediumLoader::result_type MediumLoader::operator()(MediumLoader::from_empty_tag) {
  auto ptr = std::make_shared<Medium>();
  return ptr;
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

auto Camera::getViewMat() noexcept -> se::mat4 {
  return se::mat4{};
}

auto Camera::getProjectionMat() const noexcept -> se::mat4 {
  se::mat4 projection;
  if (projectType == ProjectType::PERSPECTIVE) {
    projection = se::perspective(yfov, aspectRatio, znear, zfar).m;
  } else if (projectType == ProjectType::ORTHOGONAL) {
    projection = se::ortho(-aspectRatio * bottom_top, aspectRatio * bottom_top,
    -bottom_top, bottom_top, znear, zfar).m;
  }
  return projection;
}

// Material Resource
// ===========================================================================
// GFX Context definition

rhi::Device* GFXContext::device = nullptr;
std::unique_ptr<rhi::MultiFrameFlights> GFXContext::flights = nullptr;
ex::resource_cache<Mesh, MeshLoader> GFXContext::meshs = {};
ex::resource_cache<Material, MaterialLoader> GFXContext::materials = {};
ex::resource_cache<Buffer, BufferLoader> GFXContext::buffers = {};
ex::resource_cache<Texture, TextureLoader> GFXContext::textures = {};
ex::resource_cache<rhi::Sampler, SamplerLoader> GFXContext::samplers = {};
ex::resource_cache<ShaderModule, ShaderModuleLoader> GFXContext::shaders = {};
ex::resource_cache<Medium, MediumLoader> GFXContext::mediums = {};

auto GFXContext::initialize(rhi::Device* device) noexcept -> void {
  GFXContext::device = device;
}

auto GFXContext::createFlights(int maxFlightNum, rhi::SwapChain* swapchain) -> void {
  flights = device->createMultiFrameFlights({ MULTIFRAME_FLIGHTS_COUNT, nullptr });
}

auto GFXContext::getFlights() -> rhi::MultiFrameFlights* {
  return flights.get();
}

auto GFXContext::getDevice() noexcept -> rhi::Device* {
  return device;
}

auto GFXContext::finalize() noexcept -> void {
  buffers.clear();
  meshs.clear();
  shaders.clear();
  materials.clear();
  textures.clear();
  samplers.clear();
  scenes.clear();
  flights = nullptr;
}

auto GFXContext::load_buffer_empty() noexcept -> BufferHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = buffers.load(ruid, BufferLoader::from_empty_tag{});
  return BufferHandle{ ret.first->second };
}

auto GFXContext::load_buffer_gltf(tinygltf::Buffer const& buffer) noexcept -> BufferHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = buffers.load(ruid, BufferLoader::from_gltf_tag{}, buffer);
  return BufferHandle{ ret.first->second };
}

auto GFXContext::load_buffer_host(se::buffer const& buffer, rhi::BufferUsages usages) noexcept -> BufferHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = buffers.load(ruid, BufferLoader::from_host_tag{}, buffer, usages);
  return BufferHandle{ ret.first->second };
}

auto GFXContext::create_buffer_desc(rhi::BufferDescriptor const& desc) noexcept -> BufferHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = buffers.load(ruid, BufferLoader::from_desc_tag{}, desc);
  return BufferHandle{ ret.first->second };
}

auto GFXContext::create_texture_desc(rhi::TextureDescriptor const& desc) noexcept -> TextureHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = textures.load(ruid, TextureLoader::from_desc_tag{}, desc);
  return TextureHandle{ ret.first->second, ruid };
}

auto GFXContext::create_texture_file(std::string const& path) noexcept -> TextureHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = textures.load(ruid, TextureLoader::from_file_tag{}, path);
  gfx::GFXContext::device->trainsitionTextureLayout(ret.first->second->texture.get(),
    se::rhi::TextureLayout::UNDEFINED, se::rhi::TextureLayout::GENERAL);
  return TextureHandle{ ret.first->second, ruid };
}

auto GFXContext::create_buf_texture_desc(rhi::TextureDescriptor const& desc, float default_value) noexcept -> TextureHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = textures.load(ruid, TextureLoader::from_desc_buf_tag{}, desc, default_value);
  return TextureHandle{ ret.first->second, ruid };
}

auto GFXContext::create_sampler_desc(
    rhi::SamplerDescriptor const& desc
) noexcept -> SamplerHandle {
  const uint64_t id = hash(desc);
  auto ret = samplers.load(id, SamplerLoader::from_desc_tag{}, desc);
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
  auto ret = samplers.load(id, SamplerLoader::from_desc_tag{}, desc);
  return SamplerHandle{ ret.first->second };
}

auto GFXContext::load_mesh_empty() noexcept -> MeshHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = meshs.load(ruid, MeshLoader::from_empty_tag{});
  return MeshHandle{ ret.first->second, ruid };
}

auto GFXContext::load_material_empty() noexcept -> MaterialHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = materials.load(ruid, MaterialLoader::from_empty_tag{});
  return MaterialHandle{ ret.first->second, ruid };
}

auto GFXContext::load_medium_empty() noexcept -> MediumHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = mediums.load(ruid, MediumLoader::from_empty_tag{});
  return MediumHandle{ ret.first->second, ruid };
}

auto GFXContext::load_shader_spirv(se::buffer* buffer, rhi::ShaderStageBit stage) noexcept -> ShaderHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = shaders.load(ruid, ShaderModuleLoader::from_spirv_tag{}, buffer, stage);
  return ShaderHandle{ ret.first->second };
}

auto GFXContext::load_shader_glsl(const char* path, rhi::ShaderStageBit stage) noexcept -> ShaderHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = shaders.load(ruid, ShaderModuleLoader::from_glsl_tag{}, std::string(path), stage);
  return ShaderHandle{ ret.first->second };
}

auto GFXContext::load_shader_slang(
  std::string const& path,
  std::vector<std::pair<std::string, rhi::ShaderStageBit>> const& entrypoints,
  std::vector<std::pair<char const*, char const*>> const& macros,
  bool glsl_intermediate
) noexcept -> std::vector<ShaderHandle> {
  slang_inline::SlangSession session(path, macros, glsl_intermediate);
  return session.load(entrypoints);
}

auto captureImage(TextureHandle src) noexcept -> void {
  se::window* mainWindow = gfx::GFXContext::device->fromAdapter()
    ->fromContext()->getBindedWindow();
  std::string filepath = mainWindow->saveFile(
    "", se::worldtime::get().to_string() + ".exr");
  captureImage(src, filepath);
}

auto captureImage(TextureHandle src, std::string path) noexcept -> void {
  gfx::Texture* tex = src.get();
  size_t width = tex->texture->width();
  size_t height = tex->texture->height();

  rhi::TextureFormat format;
  size_t pixelSize;
  if (tex->texture->format() == rhi::TextureFormat::RGBA32_FLOAT) {
    format = rhi::TextureFormat::RGBA32_FLOAT;
    pixelSize = sizeof(se::vec4);
  }
  else if (tex->texture->format() == rhi::TextureFormat::RGBA8_UNORM) {
    format = rhi::TextureFormat::RGBA8_UNORM;
    pixelSize = sizeof(uint8_t) * 4;
  }
  else {
    root::print::error(
      "Editor :: ViewportWidget :: captureImage() :: Unsupported format to "
      "capture."); return;
  }
  
  std::unique_ptr<rhi::CommandEncoder> commandEncoder =
    gfx::GFXContext::device->createCommandEncoder({});
  
  TextureHandle copyDst{};
  if (copyDst.get() == nullptr) {
    rhi::TextureDescriptor desc{
      {uint32_t(width), uint32_t(height), 1}, 1, 1, 1,
      rhi::TextureDimension::TEX2D, format,
      (uint32_t)rhi::TextureUsageBit::COPY_DST |
      (uint32_t)rhi::TextureUsageBit::TEXTURE_BINDING,
      {format}, (uint32_t)rhi::TextureFlagBit::HOSTI_VISIBLE };
    copyDst = GFXContext::create_texture_desc(desc);
    commandEncoder->pipelineBarrier(rhi::BarrierDescriptor{
      (uint32_t)rhi::PipelineStageBit::ALL_GRAPHICS_BIT,
      (uint32_t)rhi::PipelineStageBit::TRANSFER_BIT,
      (uint32_t)rhi::DependencyType::NONE,
      {}, {}, {rhi::TextureMemoryBarrierDescriptor{
        copyDst->texture.get(),
        rhi::ImageSubresourceRange{(uint32_t)rhi::TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
        (uint32_t)rhi::AccessFlagBits::NONE,
        (uint32_t)rhi::AccessFlagBits::TRANSFER_WRITE_BIT,
        rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
        rhi::TextureLayout::TRANSFER_DST_OPTIMAL}} });
  }
  gfx::GFXContext::device->waitIdle();
  commandEncoder->pipelineBarrier(rhi::BarrierDescriptor{
    (uint32_t)rhi::PipelineStageBit::FRAGMENT_SHADER_BIT,
    (uint32_t)rhi::PipelineStageBit::TRANSFER_BIT,
    (uint32_t)rhi::DependencyType::NONE,
    {}, {}, { rhi::TextureMemoryBarrierDescriptor{
      src->texture.get(),
      rhi::ImageSubresourceRange{(uint32_t)rhi::TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
      (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT |
          (uint32_t)rhi::AccessFlagBits::SHADER_WRITE_BIT,
      (uint32_t)rhi::AccessFlagBits::TRANSFER_READ_BIT,
      rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
      rhi::TextureLayout::TRANSFER_SRC_OPTIMAL}} });
  commandEncoder->copyTextureToTexture(
    rhi::ImageCopyTexture{ src->texture.get() },
    rhi::ImageCopyTexture{ copyDst->texture.get() },
    rhi::Extend3D{ uint32_t(width), uint32_t(height), 1 });
  commandEncoder->pipelineBarrier(rhi::BarrierDescriptor{
    (uint32_t)rhi::PipelineStageBit::TRANSFER_BIT,
    (uint32_t)rhi::PipelineStageBit::FRAGMENT_SHADER_BIT,
    (uint32_t)rhi::DependencyType::NONE,
    {}, {}, { rhi::TextureMemoryBarrierDescriptor{
      src->texture.get(),
      rhi::ImageSubresourceRange{(uint32_t)rhi::TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
      (uint32_t)rhi::AccessFlagBits::TRANSFER_READ_BIT,
      (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT |
          (uint32_t)rhi::AccessFlagBits::SHADER_WRITE_BIT,
      rhi::TextureLayout::TRANSFER_SRC_OPTIMAL,
      rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL}} });
  commandEncoder->pipelineBarrier(rhi::BarrierDescriptor{
    (uint32_t)rhi::PipelineStageBit::TRANSFER_BIT,
    (uint32_t)rhi::PipelineStageBit::HOST_BIT,
    (uint32_t)rhi::DependencyType::NONE,
    {}, {}, { rhi::TextureMemoryBarrierDescriptor{
      copyDst->texture.get(),
      rhi::ImageSubresourceRange{(uint32_t)rhi::TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
      (uint32_t)rhi::AccessFlagBits::TRANSFER_WRITE_BIT,
      (uint32_t)rhi::AccessFlagBits::HOST_READ_BIT,
      rhi::TextureLayout::TRANSFER_DST_OPTIMAL,
      rhi::TextureLayout::TRANSFER_DST_OPTIMAL}} });

  gfx::GFXContext::device->getGraphicsQueue()->submit(
      { commandEncoder->finish() });
  gfx::GFXContext::device->waitIdle();
  std::future<bool> mapped = copyDst->texture->mapAsync((uint32_t)rhi::MapMode::READ, 0,
    width * height * pixelSize);
  if (mapped.get()) {
    void* data = copyDst->texture->getMappedRange(0, width * height * pixelSize);
    if (tex->texture->format() == rhi::TextureFormat::RGBA32_FLOAT) {
      image::EXR::writeEXR(path, width, height, 4,
        reinterpret_cast<float*>(data));
    }
    else if (tex->texture->format() == rhi::TextureFormat::RGBA8_UNORM) {
        //std::string filepath = mainWindow->saveFile(
        //    "", Core::WorldTimePoint::get().to_string() + ".bmp");
        //Image::BMP::writeBMP(filepath, width, height, 4,
        //    reinterpret_cast<float*>(data));
    }
    copyDst->texture->unmap();
  }
}
}