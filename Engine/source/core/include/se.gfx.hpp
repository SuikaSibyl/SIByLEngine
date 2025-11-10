#pragma once
#include "entt/entt.hpp"
#include "se.rhi.hpp"
#include "se.utils.hpp"
#include "imgui.h"
#include <tinygltf/tiny_gltf.h>
#include <typeindex>
#include <stack>
namespace ex = entt;

namespace se {
namespace editor {
  struct IFragment;
  struct ImguiTexture;
}
}

namespace se {
namespace gfx {
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource                                                                  ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ This is a submodule of gfx, that aims to provide further wrapping for all ┃  
  // ┃ kinds of resources. Unlike in rhi, all resources are naked, in gfx the    ┃
  // ┃ resource types are managed, and should be queried and used directly by    ┃
  // ┃ higher level applications.                                                ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  struct IResource {
    UID m_uid;           // a uniform identifier of the resource
    std::string m_name;  // name of the resource
    int32_t m_countDown = 0; // wait for be killed
    std::string m_creator = "UNKNOWN";
    std::string m_job = "UNKNOWN";
    bool m_dirtyToGPU = true;
    bool m_dirtyToFile = false;

    virtual ~IResource() = default;
    auto get_name() const noexcept -> std::string const& { return m_name; }
    virtual auto draw_gui(editor::IFragment* fragment) noexcept -> void {};
  };

  template<class T>
  struct ResourceHandle {
    ex::resource<T> m_handle;
    auto get() noexcept -> T* { return m_handle.handle().get(); }
    T* operator->() { return get(); }
    T const* operator->() const { return get(); }
    auto release() noexcept -> void { m_handle.reset(); }
    auto draw_gui(editor::IFragment* fragment = nullptr) noexcept -> void {
      std::string const str = "Reference count: " +
        std::to_string(m_handle.handle().use_count() - 2);
      ImGui::Text("%s", str.c_str());
      m_handle->draw_gui(fragment);
    }
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: buffer                                                        ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  // Buffer, but gfx resource.
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  struct Buffer : public IResource {
    // Definition to describe how a buffer is consumed by a pass
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    struct ConsumeEntry {
      Flags<rhi::AccessFlagEnum> m_access = 0;
      Flags<rhi::PipelineStageEnum> m_stages = 0;
      uint64_t m_offset = 0;
      uint64_t m_size = uint64_t(-1);

      auto add_stage(Flags<rhi::PipelineStageEnum> stage) noexcept -> ConsumeEntry&;
      auto set_access(Flags<rhi::AccessFlagEnum> acc) noexcept -> ConsumeEntry&;
      auto set_subresource(uint64_t offset, uint64_t size) noexcept -> ConsumeEntry&;
    };

    struct ConsumeState {
      std::vector<ConsumeEntry> m_entries;
    };

    struct ResourceStateMachine {
      struct SubresourceRange {
        size_t m_rangeBeg;
        size_t m_rangeEnd;
        auto operator==(SubresourceRange const& x) noexcept -> bool;
        auto valid() noexcept -> bool;
      };

      struct SubresourceState {
        Flags<rhi::PipelineStageEnum> m_stageMask = 0;
        Flags<rhi::AccessFlagEnum> m_access = 0;
        auto operator==(SubresourceState const& x) noexcept -> bool;
      };

      struct SubresourceEntry {
        SubresourceRange range;
        SubresourceState state;
      };

      rhi::Buffer* m_buffer;
      std::vector<SubresourceEntry> m_writeStates;
      std::vector<SubresourceEntry> m_readStates;

      ResourceStateMachine() = default;

      auto intersect(
        SubresourceRange const& x,
        SubresourceRange const& y) noexcept
        -> std::optional<SubresourceRange>;

      auto diff(
        SubresourceRange const& x,
        SubresourceRange const& y) noexcept
        -> std::vector<SubresourceRange>;

      auto to_barrier_descriptor(
        SubresourceRange const& range,
        SubresourceState const& prev,
        SubresourceState const& next) noexcept
        -> rhi::BarrierDescriptor;

      auto update_subresource(
        SubresourceRange const& range,
        SubresourceState const& state) noexcept
        -> std::vector<rhi::BarrierDescriptor>;

    };

    /** buffer resource */
    std::unique_ptr<rhi::Buffer> m_buffer = nullptr;
    /** previous buffer resource */
    std::unique_ptr<rhi::Buffer> m_previous = nullptr; 
    /** the host resource of buffer */
    std::vector<std::byte> m_host;
    /** the stamps of the buffer update */
    size_t m_bufferStamp = 0; 
    size_t m_previousStamp = 0;
    size_t m_hostStamp = 0;
    /** the usages of the buffer */
    Flags<rhi::BufferUsageEnum> m_usages = 0;
    /** the state machine of current texture */
    ResourceStateMachine m_stateMachine;
    /** whether we always map the buffer on host */
    enum struct MemoryCopyMode {
      TEMPORARY_STAGING,
      PERSISTENT_STAGING,
      COHERENT_MAPPING,
    } m_memoryCopyMode;

    /** lazy update host buffer to device */
    auto host_to_device() noexcept -> void;
    /** lazy update host device to host */
    auto device_to_host() noexcept -> void;
    /** create the device buffer if not exists */
    auto create_device() noexcept -> void;
    /** mapping local and device memory */
    auto memory_mapping() noexcept -> void*;
    /** get the host buffer, if not exists fetch from newest device */
    auto get_host() noexcept -> std::vector<std::byte>&;
    /** get the device buffer, if not exists fetch from newest host */
    auto get_device() noexcept -> rhi::Buffer*;
    /** get the resource binding of the entire buffer */
    auto get_binding_resource() noexcept -> rhi::BindingResource;
    /** emplace a data into host buffer */
    template<class T>
    auto emplace_host(T const& data) noexcept -> void {
      size_t host_size = m_host.size();
      m_host.resize(host_size + sizeof(T));
      memcpy(&m_host[host_size], &data, sizeof(T));
      m_hostStamp++;
    }
    /** copy a structure into the host buffer */
    template<class T>
    auto copy_to_host(int32_t index, T const& data) noexcept -> void {
      memcpy(&m_host[index * sizeof(T)], &data, sizeof(T));
      m_hostStamp++;
    }
    /** copy a structure into the host buffer */
    template<class T>
    auto read_from_host(int32_t index) noexcept -> T& {
      return *(T*)(&m_host[index * sizeof(T)]);
    }
    /** copy a structure into the host buffer */
    template<class T>
    auto read_from_host(int32_t index, size_t stride, size_t offset) noexcept -> T& {
      return *(T*)(&m_host[index * stride + offset]);
    }
    /** draw gui of the buffer */
    virtual auto draw_gui(editor::IFragment* fragment = nullptr) noexcept -> void override;
  };
  // The handle of buffer
  using BufferHandle = ResourceHandle<Buffer>;
  // The loader of buffer
  struct BufferLoader {
    using result_type = std::shared_ptr<Buffer>;

    struct from_empty_tag {};
    struct from_gltf_tag {};
    struct from_host_tag {};
    struct from_desc_tag {};

    result_type operator()(from_empty_tag);
    result_type operator()(from_desc_tag, rhi::BufferDescriptor desc);
    result_type operator()(from_host_tag, MiniBuffer const& buffer, Flags<rhi::BufferUsageEnum> usages);
  };

  template <class T>
  struct DynamicVectorBufferView {
    BufferHandle m_buffer;
    std::stack<int32_t> m_freeList;
    size_t m_size = 0;

    // Insert a new element, return its index in the buffer
    auto insert(T const& value) noexcept -> int32_t {
      int32_t idx;
      if (!m_freeList.empty()) {
        idx = m_freeList.top();
        m_freeList.pop();
      } else {
        idx = m_size++;
        if (idx * sizeof(T) >= m_buffer->m_host.size())
          m_buffer->m_host.resize(std::max(size_t(4) * sizeof(T), m_buffer->m_host.size() * 2));
      }
      std::memcpy(&m_buffer->m_host[idx * sizeof(T)], &value, sizeof(T));
      return idx;
    }

    // Insert a new element, return its index in the buffer
    auto insert_consecutive(std::vector<T> const& value) noexcept -> int32_t {
      int32_t idx = m_size;
      m_size += value.size();
      if (m_size * sizeof(T) >= m_buffer->m_host.size())
        m_buffer->m_host.resize(std::max(m_size * sizeof(T), m_buffer->m_host.size() * 2));
      std::memcpy(&m_buffer->m_host[idx * sizeof(T)], value.data(), sizeof(T) * value.size());
      return idx;
    }

    // Remove element at index
    auto remove(int32_t idx) noexcept -> void { m_freeList.push(idx); }

    // Update element at index
    auto update(int32_t idx, T const& value) noexcept -> void {
      std::memcpy(&m_buffer->m_host[idx * sizeof(T)], &value, sizeof(T));
      m_buffer->m_hostStamp++;
    }

    // Access element at index
    T& operator[](int32_t idx) {
      return *reinterpret_cast<T*>(&m_buffer->m_host[idx * sizeof(T)]);
    }

    const T& operator[](int32_t idx) const {
      return *reinterpret_cast<const T*>(&m_buffer->m_host[idx * sizeof(T)]);
    }
  };

  // Sampler, but gfx resource.
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  struct Sampler : public IResource {
    std::unique_ptr<rhi::Sampler> m_sampler = nullptr;
  };
  // The handle of sampler
  using SamplerHandle = ResourceHandle<Sampler>;
  // The loader of sampler
  struct SamplerLoader {
    using result_type = std::shared_ptr<Sampler>;

    struct from_desc_tag {};
    struct from_mode_tag {};

    result_type operator()(from_desc_tag, rhi::SamplerDescriptor const& desc);
    result_type operator()(from_mode_tag, rhi::AddressMode address, rhi::FilterMode filter, rhi::MipmapFilterMode mipmap);
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: texture                                                       ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  // Texture, but gfx resource.
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  struct Texture : public IResource {
    // Definition to describe how a texture is consumed by a pass
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    enum struct ConsumeType {
      ColorAttachment,
      DepthStencilAttachment,
      TextureBinding,
      StorageBinding,
    };

    struct ConsumeEntry {
      ConsumeType type;
      Flags<rhi::AccessFlagEnum> access = 0;
      Flags<rhi::PipelineStageEnum> stages = 0;
      uint32_t level_beg = 0;
      uint32_t level_end = 1;
      uint32_t mip_beg = 0;
      uint32_t mip_end = 1;
      rhi::TextureLayoutEnum layout = rhi::TextureLayoutEnum::UNDEFINED;
      bool depthWrite = false;
      rhi::CompareFunction depthCmp = rhi::CompareFunction::ALWAYS;
      uint32_t attachLoc = uint32_t(-1);
      rhi::BlendOperation bldOperation = rhi::BlendOperation::ADD;
      rhi::BlendFactor srcFactor = rhi::BlendFactor::ONE;
      rhi::BlendFactor dstFactor = rhi::BlendFactor::ZERO;

      ConsumeEntry() = default;
      ConsumeEntry(ConsumeType _type) : type(_type), access(0), stages(0),
        level_beg(0), level_end(1), mip_beg(0), mip_end(1),
        layout(rhi::TextureLayoutEnum::UNDEFINED), depthWrite(false), 
        depthCmp(rhi::CompareFunction::ALWAYS), attachLoc(uint32_t(-1)) {}
      ConsumeEntry(ConsumeType _type,
        Flags<rhi::AccessFlagEnum> _access, Flags<rhi::PipelineStageEnum> _stages = 0,
        uint32_t _level_beg = 0, uint32_t _level_end = 1,
        uint32_t _mip_beg = 0, uint32_t _mip_end = 1,
        rhi::TextureLayoutEnum _layout = rhi::TextureLayoutEnum::UNDEFINED,
        bool _depthWrite = false, rhi::CompareFunction _depthCmp = rhi::CompareFunction::ALWAYS,
        uint32_t _attachLoc = uint32_t(-1)) : type(_type), access(_access), stages(_stages),
        level_beg(_level_beg), level_end(_level_end), mip_beg(_mip_beg), mip_end(_mip_end),
        layout(_layout), depthWrite(_depthWrite), depthCmp(_depthCmp), attachLoc(_attachLoc) {}

      auto add_stage(Flags<rhi::PipelineStageEnum> stage) noexcept -> ConsumeEntry&;
      auto set_layout(rhi::TextureLayoutEnum _layout) noexcept -> ConsumeEntry&;
      auto enable_depth_write(bool set) noexcept -> ConsumeEntry&;
      auto set_depth_compare_fn(rhi::CompareFunction fn) noexcept -> ConsumeEntry&;
      auto set_subresource(uint32_t mip_beg, uint32_t mip_end,
        uint32_t level_beg, uint32_t level_end) noexcept -> ConsumeEntry&;
      auto set_attachment_loc(uint32_t loc) noexcept -> ConsumeEntry&;
      auto set_access(Flags<rhi::AccessFlagEnum> acc) noexcept -> ConsumeEntry&;
      auto set_blend_operation(rhi::BlendOperation operation) noexcept -> ConsumeEntry&;
      auto set_source_blender_factor(rhi::BlendFactor factor) noexcept -> ConsumeEntry&;
      auto set_target_blender_factor(rhi::BlendFactor factor) noexcept -> ConsumeEntry&;
    };

    struct ConsumeState {
      std::vector<ConsumeEntry> m_entries;
    };

    struct ResourceStateMachine {
      struct SubresourceRange {
        uint32_t m_levelBeg;
        uint32_t m_levelEnd;
        uint32_t m_mipBeg;
        uint32_t m_mipEnd;
        auto operator==(SubresourceRange const& x) noexcept -> bool;
        auto valid() noexcept -> bool;
      };

      struct SubresourceState {
        Flags<rhi::PipelineStageEnum> stageMask;
        Flags<rhi::AccessFlagEnum> access;
        rhi::TextureLayoutEnum layout;
        auto operator==(SubresourceState const& x) noexcept -> bool;
      };

      struct SubresourceEntry {
        SubresourceRange range;
        SubresourceState state;
      };

      rhi::Texture* m_texture;
      Flags<rhi::TextureAspectEnum> m_aspects = 0;
      std::vector<SubresourceEntry> m_states;

      ResourceStateMachine() = default;
      ResourceStateMachine(rhi::Texture* tex);

      /** Return the intersection of two subresource range
        * if exists, return nullopt if they do not overlap. */
      static auto intersect(
        SubresourceRange const& x,
        SubresourceRange const& y
      ) noexcept -> std::optional<SubresourceRange>;

      /** Try to merge two subresource ranges, if they are continuous
        * and thus can be combined into one, return nullopt otherwise. */
      static auto merge(
        SubresourceRange const& x,
        SubresourceRange const& y
      ) noexcept -> std::optional<SubresourceRange>;

      /** Try to get the diff of two subresource ranges, if there are something remains. */
      static auto diff(
        SubresourceRange const& x,
        SubresourceRange const& y
      ) noexcept -> std::vector<SubresourceRange>;

      auto to_barrier_descriptor(
        SubresourceRange const& range,
        SubresourceState const& prev,
        SubresourceState const& next
      ) noexcept -> rhi::BarrierDescriptor;

      auto try_merge() noexcept -> void;

      auto update_subresource(
        SubresourceRange const& range,
        SubresourceState const& state
      ) noexcept -> std::vector<rhi::BarrierDescriptor>;

      auto transition(ResourceStateMachine const& sm) noexcept ->
        std::vector<rhi::BarrierDescriptor>;
    };

    /** texture resource */
    std::unique_ptr<rhi::Texture> m_texture = nullptr;
    /** path string */
    std::optional<std::string> m_resourcePath;
    /** differentiable attributes */
    uint32_t m_differentiable_channels = 0u;
    /** the state machine of current texture */
    ResourceStateMachine m_stateMachine;
    /** texture type: default vkTex */
    enum struct TextureType {
      vkTexture,
      bufTexture,
    } type = TextureType::vkTexture;
    /** A pool that contains all potentail views */
    std::unordered_map<rhi::TextureViewIndex, std::unique_ptr<rhi::TextureView>,
      rhi::TextureViewIndexHash> m_viewPool;
    /** texture resource ImGui interfae */
    std::unique_ptr<editor::ImguiTexture> m_imguiTexture;

    auto init() noexcept -> void;

    auto consume(ConsumeEntry const& entry) noexcept -> std::vector<rhi::BarrierDescriptor>;
    /** Get the UAV of the texture */
    auto get_uav(uint32_t mipLevel, uint32_t firstArraySlice,
      uint32_t arraySize) noexcept -> rhi::TextureView*;
    /** Get the RTV of the texture */
    auto get_rtv(uint32_t mipLevel, uint32_t firstArraySlice,
      uint32_t arraySize) noexcept -> rhi::TextureView*;
    /** Get the RTV of the texture */
    auto get_dsv(uint32_t mipLevel, uint32_t firstArraySlice,
      uint32_t arraySize) noexcept -> rhi::TextureView*;
    /** Get the SRV of the texture */
    auto get_srv(uint32_t mostDetailedMip, uint32_t mipCount,
      uint32_t firstArraySlice, uint32_t arraySize) noexcept -> rhi::TextureView*;
    virtual auto draw_gui(editor::IFragment* fragment = nullptr) noexcept -> void override;

    auto width() noexcept -> size_t;
    auto height() noexcept -> size_t;

    auto get_imgui_texture() noexcept -> editor::ImguiTexture*;

    auto save_image(std::string const& path) noexcept -> void;
  };
  // The handle of texture
  using TextureHandle = ResourceHandle<Texture>;
  // The loader of texture
  struct TextureLoader {
    using result_type = std::shared_ptr<Texture>;

    struct from_desc_tag {};
    struct from_file_tag {};
    struct from_binary_tag {};
    struct from_desc_buf_tag {};

    result_type operator()(from_desc_tag, rhi::TextureDescriptor const& desc);
    result_type operator()(from_file_tag, std::string const& path);
    result_type operator()(from_binary_tag, int width, int height, int channel, int bits, const char* data);
    //result_type operator()(from_desc_buf_tag, rhi::TextureDescriptor const& desc, float default_value = 0.5,
    //  int aux_count = 0, int rep_count = 1);
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: shader                                                        ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  // we first define the shader reflection
  struct ShaderReflection {
    enum struct ResourceType {
      Undefined,
      UniformBuffer,
      StorageBuffer,
      StorageImages,
      SampledImages,
      ReadonlyImage,
      Sampler,
      AccelerationStructure,
    };

    enum struct ResourceEnum : uint32_t {
      None = 0,
      NotReadable = 1 << 0,
      NotWritable = 1 << 1,
    };

    struct ResourceEntry {
      ResourceType type = ResourceType::Undefined;
      Flags<ResourceEnum> flags = 0;
      Flags<rhi::ShaderStageEnum> stages = 0;
      uint32_t arraySize = 1;
    };

    struct PushConstantEntry {
      uint32_t index = -1;
      uint32_t offset = -1;
      uint32_t range = -1;
      Flags<rhi::ShaderStageEnum> stages = 0;
    };

    std::vector<PushConstantEntry> pushConstant;
    std::vector<std::vector<ResourceEntry>> bindings;

    struct BindingInfo {
      ResourceType type = ResourceType::Undefined;
      uint32_t set;
      uint32_t binding;
    };
    std::unordered_map<std::string, BindingInfo> bindingInfo;

    static auto toBindGroupLayoutDescriptor(
      std::vector<ResourceEntry> const& bindings) noexcept
      -> rhi::BindGroupLayoutDescriptor;
    auto operator+(ShaderReflection const& reflection) const->ShaderReflection;
    auto on_draw_gui() noexcept -> void;
  };

  struct ShaderModule : public IResource {
    /** rhi shader module */
    std::unique_ptr<rhi::ShaderModule> m_shaderModule = nullptr;
    /** reflection Information of the shader module */
    ShaderReflection m_reflection;
    /** get name */
    //auto getName() const noexcept -> char const*;
    virtual auto draw_gui(editor::IFragment* fragment) noexcept -> void override;
  };
  // The handle of texture
  using ShaderHandle = ResourceHandle<ShaderModule>;

  struct ShaderLoader {
    using result_type = std::shared_ptr<ShaderModule>;

    struct from_spirv_tag {};
    struct from_glsl_tag {};
    struct from_slang_tag {};

    result_type operator()(from_spirv_tag, MiniBuffer* buffer, rhi::ShaderStageEnum stage);
    result_type operator()(from_glsl_tag, std::string const& filepath,
      rhi::ShaderStageEnum stage, std::vector<std::string> const& argv = {});
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: material                                                      ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct Material : public IResource {
    struct MaterialPacket {
      int32_t bxdfType = 0;
      int32_t bitfield = 0;
      int16_t baseTex = -1;
      int16_t normTex = -1;
      int16_t ext1Tex = -1;
      int16_t ext2Tex = -1;
      vec4 vec4Data0 = vec4{ 0 };
      vec4 vec4Data1 = vec4{ 0 };
      vec4 vec4Data2 = vec4{ 0 };
    } m_packet;
    std::string   m_name;
    std::string   m_customString;
    TextureHandle m_basecolorTex;
    TextureHandle m_normalTex;
    TextureHandle m_additionalTex1;
    TextureHandle m_additionalTex2;
    BufferHandle  m_additionalBuffer1;
    BufferHandle  m_additionalBuffer2;

    virtual auto draw_gui(editor::IFragment* fragment) noexcept -> void override;
  };
  // The handle of material
  using MaterialHandle = ResourceHandle<Material>;
  // The loader of material
  struct MaterialLoader {
    using result_type = std::shared_ptr<Material>;
    struct from_empty_tag {};
    result_type operator()(from_empty_tag);
  };

  struct MaterialInterpreterManager {
    SINGLETON(MaterialInterpreterManager, {});

    using fn_material = std::function<void(Material*)>;

    struct MaterialInterpreterDictionary {
      std::string name;
      int32_t typeIdx;
      fn_material initMat;
      fn_material setDefault;
      fn_material drawGui;
    };

    std::map<int32_t, MaterialInterpreterDictionary> m_intepretors;
    std::unordered_map<size_t, int32_t> m_typeids;

    static auto init(Material* mat, int32_t type_id) noexcept -> void;
    static auto draw_gui(Material* mat, int32_t type_id) noexcept -> void;

    template<class T>
    static auto registrate(
      int32_t i,
      std::string const& display
    ) noexcept -> void {
      size_t index = typeid(T).hash_code();
      Singleton<MaterialInterpreterManager>::instance()->m_typeids[index] = i;
      fn_material fn_init = [](Material* mat) -> void { return T::init(mat); };
      fn_material fn_default = [](Material* mat) -> void { return T::set_default(mat); };
      fn_material fn_gui = [](Material* mat) -> void { return T::draw_gui(mat); };
      Singleton<MaterialInterpreterManager>::instance()->m_intepretors[i] =
        MaterialInterpreterDictionary{
        display, i, fn_init, fn_default, fn_gui
      };
    }
  };
  
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: medium                                                        ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct Medium : public IResource {
    enum struct MediumType : uint32_t {
      Homogeneous = 0,
      GridMedium = 1,
      RGBGridMedium = 2,
    };

    enum struct PhaseType :uint32_t {
      IsotropicPhase = 0,
      HenyeyGreenstein = 1,
    };

    struct SampledGrid {
      int nx, ny, nz;
      std::vector<float> values;
      bounds3 bounds;
      int gridChannel = 1;
      float max_value(const bounds3& bounds) const;
      float lookup(const ivec3& p) const;
      vec3 lookup3(const ivec3& p) const;
    };

    struct MajorantGrid {
      bounds3 bounds;
      std::vector<float> voxels;
      ivec3 res;

      bounds3 voxel_bounds(int x, int y, int z) const;
      void set(int x, int y, int z, float v);
    };

    struct MediumPacket {
      vec3 sigmaA;
      uint32_t bitfield;
      vec3 sigmaS;
      float scale;
      vec3 aniso;
      float temperatureScale;
      vec3 boundMin;
      float LeScale;
      vec3 boundMax;
      MediumType type = MediumType::Homogeneous;
      ivec3 densityNxyz;
      int32_t densityOffset = -1;
      ivec3 temperatureNxyz;
      int32_t temperatureOffset = -1;
      ivec3 leNxyz;
      int32_t leOffset = -1;
      ivec3 majorantNxyz;
      int32_t majorantOffset = -1;
      vec3 temperatureBoundMin;
      float temperatureStart;
      vec3 temperatureBoundMax;
      float pdadding;
      rhi::AffineTransformMatrix geometryTransform = {};
      rhi::AffineTransformMatrix geometryTransformInverse = {};
    } packet;

    std::optional<SampledGrid> density;
    std::optional<SampledGrid> LeScale;
    std::optional<SampledGrid> temperatureGrid;
    std::optional<MajorantGrid> majorantGrid;
    bool isDirty = false;
  };
  // The handle of medium
  using MediumHandle = ResourceHandle<Medium>;

  struct MediumLoader {
    using result_type = std::shared_ptr<Medium>;
    struct from_empty_tag {};

    result_type operator()(from_empty_tag);
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: mesh                                                          ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct Mesh : public IResource {
    struct MeshPrimitive {
      size_t offset;
      size_t size;
      size_t baseVertex;
      size_t numVertex;
      MaterialHandle material;
      MediumHandle exterior;
      MediumHandle interior;
      vec3 max, min;
      rhi::BLASDescriptor blasDesc;
      rhi::BLASDescriptor uvblasDesc;
      // blas for ray tracing the geometry
      std::unique_ptr<rhi::BLAS> primBlas, backBlas;
      // blas for sampling the texcoordinates
      std::unique_ptr<rhi::BLAS> primUVBlas, backUVBlas;
    };

    struct CustomPrimitive {
      uint32_t primitiveType;
      uint32_t bitfield;
      float scalarField0;
      float scalarField1;
      vec4 vecField0;
      vec4 vecField1;
      vec4 vecField2;
      MaterialHandle material;
      MediumHandle exterior;
      MediumHandle interior;
      vec3 max, min;
      rhi::BLASDescriptor blasDesc;
      rhi::BLASDescriptor uvblasDesc;
      // blas for ray tracing the geometry
      std::unique_ptr<rhi::BLAS> primBlas, backBlas;
      // blas for sampling the texcoordinates
      std::unique_ptr<rhi::BLAS> primUVBlas, backUVBlas;
    };

    BufferHandle m_positionBuffer;
    BufferHandle m_vertexBuffer;
    BufferHandle m_indexBuffer;
    std::vector<MeshPrimitive> m_primitives;
    std::vector<CustomPrimitive> m_customPrimitives;

    virtual auto draw_gui(editor::IFragment* fragment) noexcept -> void override;
  };
  // The handle of mesh
  using MeshHandle = ResourceHandle<Mesh>;
  // The loader of mesh
  struct MeshLoader {
    using result_type = std::shared_ptr<Mesh>;

    struct from_empty_tag {};

    result_type operator()(from_empty_tag);
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Scene and components                                                      ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ This is a submodule of gfx, that aims to provide the concept of scene.    ┃  
  // ┃ Components are attached to nodes, following the ECS concept.              ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  struct Node {
    ex::entity m_entity;
    ex::registry* m_registry;

    template<class T, class ...Args>
    auto add_component(Args&&... args) -> T& {
      return m_registry->emplace<T>(m_entity, std::forward<Args>(args)...);
    }
    template<class T>
    auto get_component() const -> T* { return m_registry->try_get<T>(m_entity); }
    template<class T>
    auto remove_component() -> void { m_registry->remove<T>(m_entity); }
  };

  struct Scene;
  struct DeserializeData {
    tinygltf::Model* model;
    std::vector<Node> nodes;
  };

  struct SerializeData {
    tinygltf::Model* model;
    gfx::Scene* gfx_scene;
    std::unordered_map<ex::entity, int32_t> nodes;
    std::unordered_map<ex::entity, int32_t> lights;
    std::unordered_map<Material*, int32_t> m_materials;
    auto add_buffer(
      std::vector<std::byte> const& data,
      std::string const& name
    ) noexcept -> int32_t;
    auto add_view_accessor(
      tinygltf::BufferView bufferView,
      tinygltf::Accessor accessor
    ) noexcept -> int32_t;
    auto add_accessor(tinygltf::Accessor accessor) noexcept -> int32_t;
    auto add_material(Material*) noexcept -> int32_t;
  };

  // Register all derivative custom scripts to manager,
  // so that we can later instantiate
  struct ComponentManager {
    SINGLETON(ComponentManager, {});

    using component_retrival = std::function<void*(gfx::Node const&)>;
    using component_callback = std::function<void(void*)>;
    using component_node = std::function<void(Node&)>;
    using component_dirty = std::function<bool(void*)>;
    using component_serialize = std::function<void(SerializeData&)>;
    using component_deserialize = std::function<void(DeserializeData&)>;

    struct ComponentDictionary {
      std::string name;
      component_retrival retrival;
      component_callback draw;
      component_node add;
      component_node remove;
      component_serialize serialize;
      component_deserialize deserialize;
      component_dirty dirtyToGPU;
      component_dirty dirtyToFile;
      bool couldRemove;
    };

    std::map<int32_t, ComponentDictionary> m_components;
    std::unordered_map<size_t, int32_t> m_typeids;

    template<class T>
    static auto registrate(
      int32_t i, 
      std::string const& display, 
      bool could_remove
    ) noexcept -> void {
      size_t index = typeid(T).hash_code();
      Singleton<ComponentManager>::instance()->m_typeids[index] = i;
      component_retrival retrival =
        [](gfx::Node const& node) -> void* { return static_cast<void*>(node.get_component<T>()); };
      component_callback draw = [](void* component) -> void { T::draw_component(component); };
      component_node add = [](Node& node) -> void { node.add_component<T>(); };
      component_node remove = [](Node& node) -> void { node.remove_component<T>(); };
      component_serialize serialize = [](SerializeData& data) -> void { T::serialize(data); };
      component_deserialize deserialize = [](DeserializeData& data) -> void { T::deserialize(data); };
      component_dirty dirtyToGPU = [](void* component) -> bool { return ((T*)component)->is_dirty_to_gpu(); };
      component_dirty dirtyToFile = [](void* component) -> bool { return ((T*)component)->is_dirty_to_file(); };
      Singleton<ComponentManager>::instance()->m_components[i] = ComponentDictionary{
        display, retrival, draw, add, remove, serialize, deserialize, dirtyToGPU, dirtyToFile, could_remove
      };
    }

    static auto draw_all_components(Node& node) noexcept -> void;
  };
  

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ component :: NodeProperty                                                 ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct NodeProperty {
    std::string name;
    std::vector<Node> children;
    bool m_dirtyToFile;

    static auto draw_component(void* component) noexcept -> void;
    static auto serialize(SerializeData& data) noexcept -> void;
    static auto deserialize(DeserializeData& data) noexcept -> void {};
    auto is_dirty_to_gpu() noexcept -> bool { return false; }
    auto is_dirty_to_file() noexcept -> bool { return m_dirtyToFile; }
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ component :: transform                                                    ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct Transform {
    vec3 translation = { 0.0f, 0.0f, 0.0f };
    vec3 scale = { 1.0f, 1.0f, 1.0f };
    float oddScaling = 1.f;
    Quaternion rotation = { 0.0f, 0.0f, 0.0f, 1.f };
    mat4 global = {};
    bool m_dirtyToFile; bool m_dirtyToGPU;

    auto local() const noexcept -> se::mat4;
    auto forward() const noexcept -> se::vec3;

    static auto draw_component(void* component) noexcept -> void;
    static auto serialize(SerializeData& data) noexcept -> void;
    static auto deserialize(DeserializeData& data) noexcept -> void {};
    auto is_dirty_to_gpu() noexcept -> bool { return m_dirtyToGPU; }
    auto is_dirty_to_file() noexcept -> bool { return m_dirtyToFile; }
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ component :: mesh renderer                                                ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct MeshRenderer {
    MeshHandle m_mesh;
    std::vector<rhi::BLASInstance> m_blasInstance;
    std::optional<std::vector<rhi::BLASInstance>> m_uvblasInstance;
    bool m_dirtyToFile; bool m_dirtyToGPU;

    static auto draw_component(void* component) noexcept -> void;
    static auto serialize(SerializeData& data) noexcept -> void;
    static auto deserialize(DeserializeData& data) noexcept -> void {};
    auto is_dirty_to_gpu() noexcept -> bool { return m_dirtyToGPU; }
    auto is_dirty_to_file() noexcept -> bool { return m_dirtyToFile; }
  };

  /** mesh / geometry draw call data */
  struct GeometryDrawData {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    int materialID = -1;
    uint32_t indexSize;
    int16_t mediumIDExterior = -1;
    int16_t mediumIDInterior = -1;
    int16_t primitiveType;
    int16_t meshID;
    int32_t lightID;
    float oddNegativeScaling;
    rhi::AffineTransformMatrix geometryTransform = {};
    rhi::AffineTransformMatrix geometryTransformInverse = {};
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ component :: camera                                                       ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  struct Camera {
    enum struct ProjectType {
      PERSPECTIVE,
      ORTHOGONAL,
    };
    float aspectRatio = 1.f;
    float yfov = 45.f;
    float znear = 0.1f, zfar = 10000.0f;
    float left_right = 0;
    float bottom_top = 0;
    ProjectType projectType = ProjectType::PERSPECTIVE;
    MediumHandle medium;
    bool m_dirtyToFile; bool m_dirtyToGPU;

    auto get_viewMat() noexcept -> se::mat4;
    auto get_projection_mat() const noexcept -> se::mat4;
    auto is_dirty_to_gpu() noexcept -> bool { return m_dirtyToGPU; }
    auto is_dirty_to_file() noexcept -> bool { return m_dirtyToFile; }
    static auto draw_component(void* component) noexcept -> void;
    static auto serialize(SerializeData& data) noexcept -> void;
    static auto deserialize(DeserializeData& data) noexcept -> void {};
  };

  // A data packet camera should provide to GPU
  struct CameraData {
    se::mat4 viewMat;
    se::mat4 invViewMat;
    se::mat4 projMat;
    se::mat4 invProjMat;
    se::mat4 viewProjMat;
    se::mat4 invViewProj;
    se::mat4 viewProjMatNoJitter;
    se::mat4 projMatNoJitter;
    se::vec3 posW;
    float focalLength;
    se::vec3 prevPosW;
    float rectArea;
    se::vec3 up;
    float aspectRatio;
    se::vec3 target;
    float nearZ;
    se::vec3 cameraU;
    float farZ;
    se::vec3 cameraV;
    float jitterX;
    se::vec3 cameraW;
    float jitterY;
    float frameHeight;
    float frameWidth;
    float focalDistance;
    float apertureRadius;
    float shutterSpeed;
    float ISOSpeed;
    int mediumID = -1;
    float _padding2;
    se::vec2 clipToWindowScale;
    se::vec2 clipToWindowBias;
    CameraData() = default;
    CameraData(gfx::Camera const& camera, gfx::Transform const& transform);
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ component :: light                                                        ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  enum struct LightTypeEnum :int32_t {
    DIRECTIONAL,
    POINT,
    SPOT,
    MESH_PRIMITIVE,
    SPHERE,
    RECTANGLE,
    ENVIRONMENT,
    VPL,
    MAX_ENUM,
  };

  struct LightData {
    LightTypeEnum light_type = LightTypeEnum::DIRECTIONAL;
    uint32_t bitfield = 0;
    uint32_t uintscalar_0;
    uint32_t uintscalar_1;
    vec4 floatvec_0;
    vec4 floatvec_1;
    vec4 floatvec_2;
  };

  struct Light {
    LightData light;
    bool m_dirtyToFile; bool m_dirtyToGPU;

    auto is_dirty_to_gpu() noexcept -> bool { return m_dirtyToGPU; }
    auto is_dirty_to_file() noexcept -> bool { return m_dirtyToFile; }
    static auto draw_component(void* component) noexcept -> void;
    static auto serialize(SerializeData& data) noexcept -> void;
    static auto deserialize(DeserializeData& data) noexcept -> void;
  };
  
  struct ILightSampler {

  };

  struct LightInterpreterManager {
    SINGLETON(LightInterpreterManager, {});

    using fn_light = std::function<void(Light*)>;

    struct LightInterpreterDictionary {
      std::string name;
      int32_t typeIdx;
      fn_light initMat;
      fn_light setDefault;
      fn_light drawGui;
    };

    std::map<int32_t, LightInterpreterDictionary> m_intepretors;
    std::unordered_map<size_t, int32_t> m_typeids;

    static auto init(Light* mat, int32_t type_id) noexcept -> void;
    static auto draw_gui(Light* mat, int32_t type_id) noexcept -> void;

    template<class T>
    static auto registrate(
      int32_t i,
      std::string const& display
    ) noexcept -> void {
      size_t index = typeid(T).hash_code();
      Singleton<LightInterpreterManager>::instance()->m_typeids[index] = i;
      fn_light fn_init = [](Light* light) -> void { return T::init(light); };
      fn_light fn_default = [](Light* light) -> void { return T::set_default(light); };
      fn_light fn_gui = [](Light* light) -> void { return T::draw_gui(light); };
      Singleton<LightInterpreterManager>::instance()->m_intepretors[i] =
        LightInterpreterDictionary{
        display, i, fn_init, fn_default, fn_gui
      };
    }
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ component :: script                                                       ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  // IScript is the base
  struct IScript {
    bool m_initialized = false;
    virtual auto on_init(Node& node) noexcept -> void {};
    virtual auto on_update(Node& node, double delta) noexcept -> void {};
    virtual auto on_end(Node& node) noexcept -> void {};
  };

  // Script is the actual component
  struct Script {
    std::list<std::pair<std::string, std::shared_ptr<IScript>>> m_scripts;
    bool m_dirtyToFile;

    auto update(Node& node, double delta) noexcept -> void;
    auto is_dirty_to_gpu() noexcept -> bool { return false; }
    auto is_dirty_to_file() noexcept -> bool { return m_dirtyToFile; }
    static auto draw_component(void* component) noexcept -> void;
    static auto serialize(SerializeData& data) noexcept -> void;
    static auto deserialize(DeserializeData& data) noexcept -> void;
  };

  // Register all derivative custom scripts to manager,
  // so that we can later instantiate
  struct ScriptManager {
    SINGLETON(ScriptManager, {});

    using instantiator = std::function<std::shared_ptr<IScript>()>;
    std::unordered_map<std::string, instantiator> m_instaniator;

    template<class T>
    static auto registrate(std::string const& name) noexcept -> void {
      Singleton<ScriptManager>::instance()->m_instaniator[name] = 
      []() ->std::shared_ptr<IScript> {
        return std::make_shared<T>();
      };
    }
    
    static auto instantiate(std::string const& name) noexcept -> std::shared_ptr<IScript> {
      std::unordered_map<std::string, instantiator>& map =
        Singleton<ScriptManager>::instance()->m_instaniator;
      auto find = map.find(name);
      if (find == map.end())
        se::error("Script component fail, script {} not found", name);
      return (find->second)();
    }
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ resource :: scene                                                         ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct Scene : public IResource {
    ex::registry m_registry;
    std::vector<Node> m_roots;
    std::string m_name;
    std::string m_filepath;
    se::timer m_timer;
    std::optional<ivec2> m_viewportSize;

    struct IndexInfo {
      int32_t assignedIndex;
      int32_t heartBeat = 0;
      int32_t length = 1;
    };

    struct GPUScene {
      DynamicVectorBufferView<uint64_t> positionBuffer;
      DynamicVectorBufferView<uint64_t> indexBuffer;
      DynamicVectorBufferView<uint64_t> vertexBuffer;
      std::unordered_map<gfx::Mesh*, IndexInfo> meshList;

      DynamicVectorBufferView<CameraData> cameraBuffer;
      std::unordered_map<ex::entity, IndexInfo> cameraList;

      DynamicVectorBufferView<GeometryDrawData> geometryBuffer;
      std::unordered_map<ex::entity, std::vector<IndexInfo>> geometryList;

      DynamicVectorBufferView<Material::MaterialPacket> materialBuffer;
      std::unordered_map<gfx::Material*, IndexInfo> materialList;

      DynamicVectorBufferView<LightData> lightBuffer;
      std::unordered_map<ex::entity, std::vector<IndexInfo>> lightList;

      struct TLAS {
        rhi::TLASDescriptor desc = {};
        std::unordered_map<ex::entity, std::vector<IndexInfo>> instanceList;

        std::unique_ptr<rhi::TLAS> prim = nullptr;
        std::unique_ptr<rhi::TLAS> back = nullptr;
      } tlas;

      struct LightSampler {
        std::unique_ptr<ILightSampler> sampler;
        BufferHandle treeBuffer;
        BufferHandle trailBuffer;
        bounds3 allLightBounds;
      } lightSampler;

      struct ImagePool {
        std::unordered_map<UID, std::pair<int, TextureHandle>> texture_loc_index;
        std::vector<rhi::TextureView*> prim_t;
        std::vector<rhi::TextureView*> back_t;
        std::vector<rhi::Sampler*> prim_s;
        std::vector<rhi::Sampler*> back_s;

        auto try_fetch_index(TextureHandle texture) noexcept -> int;
      } imagePool;

      struct MediumPool {
        std::unordered_map<UID, std::pair<int, MediumHandle>> medium_loc_index;
        DynamicVectorBufferView<Medium::MediumPacket> medium_buffer;
        BufferHandle grid_storage_buffer;

        auto try_fetch_index(MediumHandle media) noexcept -> int;
      } mediumPool;

      struct SceneData {
        vec3 light_bounds_min;
        int nondistant_light_count = 0;
        vec3 light_bounds_max;
        int distant_light_count = 0;
        int environment_map = -1;
        int padding = 0;
      };
      struct SceneInfo {
        BufferHandle sceneBuffer;
        SceneData* data;
      } sceneInfo;

      auto binding_resource_position() noexcept -> rhi::BindingResource;
      auto binding_resource_index() noexcept -> rhi::BindingResource;
      auto binding_resource_vertex() noexcept -> rhi::BindingResource;
      auto binding_resource_camera() noexcept -> rhi::BindingResource;
      auto binding_resource_geometry() noexcept -> rhi::BindingResource;
      auto binding_resource_material() noexcept -> rhi::BindingResource;
      auto binding_resource_textures() noexcept -> rhi::BindingResource;
      auto binding_resource_light() noexcept -> rhi::BindingResource;
      auto binding_resource_sceneinfo() noexcept -> rhi::BindingResource;
      auto binding_resource_lightbvh_tree() noexcept -> rhi::BindingResource;
      auto binding_resource_lightbvh_trail() noexcept -> rhi::BindingResource;
      auto binding_resource_tlas() noexcept -> rhi::BindingResource;
      auto binding_resource_medium() noexcept -> rhi::BindingResource;
      auto binding_resource_medium_grid() noexcept -> rhi::BindingResource;
    } m_gpuScene;
    
    Scene();
    
    auto update_scripts() noexcept -> void;
    auto update_transform() noexcept -> void;
    auto update_gpu_scene() noexcept -> void;
    auto update_gpu_meshes() noexcept -> void;
    auto update_gpu_camera() noexcept -> void;
    auto update_gpu_lights() noexcept -> void;
    auto update_gpu_medium() noexcept -> void;
    auto update_gpu_lightbvh() noexcept -> void;
    auto update_gpu_bvh() noexcept -> void;

    auto draw_meshes(rhi::RenderPassEncoder*, int32_t geometryIDOffset = 0) noexcept -> void;
    auto set_viewport_size(ivec2 size) noexcept -> void;

    auto gpu_scene() noexcept -> GPUScene* { return &m_gpuScene; }
    auto create_node(std::string const& name = "nameless") noexcept -> Node;
    auto create_node(Node parent, std::string const& name = "nameless") noexcept -> Node;

    auto load_gltf(std::string const& path) noexcept -> void;
    auto load_xml(std::string const& path) noexcept -> void;
    auto load_pbrt(std::string const& path) noexcept -> void;

    auto draw_gui(editor::IFragment* fragment = nullptr) noexcept -> void;
    auto open_node_with_geometry_index(int32_t index) noexcept -> void;
    auto reset() noexcept -> void;
    auto save(std::string const& path) noexcept -> void;
  };
  // The handle of texture
  using SceneHandle = ResourceHandle<Scene>;

  struct SceneLoader {
    using result_type = std::shared_ptr<Scene>;

    struct from_gltf_tag {};
    struct from_xml_tag {};
    struct from_pbrt_tag {};
    //struct from_scratch_tag {};

    result_type operator()(from_gltf_tag, std::string const& path);
    result_type operator()(from_xml_tag, std::string const& path);
    result_type operator()(from_pbrt_tag, std::string const& path);
    //result_type operator()(from_scratch_tag);
  };


  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ GFXContext                                                                ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ This is ...                                                               ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  struct GFXContext {
    SINGLETON(GFXContext, {});

    std::unique_ptr<se::rhi::Context> m_ctx;
    std::unique_ptr<se::rhi::Adapter> m_adapter;
    std::unique_ptr<se::rhi::Device> m_device;
    std::unique_ptr<rhi::FrameResources> m_flights;
    ex::resource_cache<Buffer, BufferLoader> m_buffers;
    ex::resource_cache<Sampler, SamplerLoader> m_samplers;
    ex::resource_cache<Texture, TextureLoader> m_textures;
    ex::resource_cache<ShaderModule, ShaderLoader> m_shaders;
    ex::resource_cache<Mesh, MeshLoader> m_meshs;
    ex::resource_cache<Material, MaterialLoader> m_materials;
    ex::resource_cache<Medium, MediumLoader> m_mediums;
    ex::resource_cache<Scene, SceneLoader> m_scenes;
    std::list<std::function<void()>> m_jobsFrameEnd;

    static auto initialize(Window* window, Flags<rhi::ContextExtensionEnum> ext) noexcept -> void;
    static auto device() noexcept -> rhi::Device*;
    static auto create_flights(int maxFlightNum, rhi::SwapChain* swapchain) -> void;
    static auto get_flights() -> rhi::FrameResources*;
    static auto finalize() noexcept -> void;

    static auto on_draw_gui_resources() noexcept -> void;

    static auto clean_cache() noexcept -> void;
    static auto clean_buffer_cache() noexcept -> void;
    static auto clean_texture_cache() noexcept -> void;
    static auto clean_shader_cache() noexcept -> void;

    // create buffer resource
    // -------------------------------------------
    static auto create_buffer_empty(
    ) noexcept -> BufferHandle;

    static auto create_buffer_desc(
      rhi::BufferDescriptor const& desc
    ) noexcept -> BufferHandle;

    static auto create_buffer_host(
      MiniBuffer const& buffer,
      Flags<rhi::BufferUsageEnum> usages
    ) noexcept -> BufferHandle;

    // create texture resource
    // -------------------------------------------
    static auto create_texture_desc(
      rhi::TextureDescriptor const& desc
    ) noexcept -> TextureHandle;

    static auto load_texture_file(
      std::string const& path
    ) noexcept -> TextureHandle;

    static auto load_texture_binary(
      int width, int height, int channel,
      int bits, const char* data
    ) noexcept -> TextureHandle;

    // create sampler resource
    // -------------------------------------------
    static auto create_sampler_desc(
      rhi::SamplerDescriptor const& desc
    ) noexcept -> SamplerHandle;
    static auto create_sampler_desc(
      rhi::AddressMode address, rhi::FilterMode filter, rhi::MipmapFilterMode mipmap
    ) noexcept -> SamplerHandle;

    // load shader module resource
    // -------------------------------------------
    // load one shader from spirv binary
    static auto load_shader_spirv(
      se::MiniBuffer* buffer,
      rhi::ShaderStageEnum stage
    ) noexcept -> ShaderHandle;
    // load one shader from slang file
    static auto load_shader_slang(
      std::string const& path,
      std::vector<std::pair<std::string, rhi::ShaderStageEnum>> const& entrypoints,
      std::vector<std::pair<char const*, char const*>> const& macros = {},
      bool glsl_intermediate = false
    ) noexcept -> std::vector<ShaderHandle>;
    // load N shaders from slang files
    template<size_t N>
    static auto load_shader_slang(
      std::string const& filepath,
      std::array<std::pair<std::string, rhi::ShaderStageEnum>, N> const& entrypoints,
      std::vector<std::pair<char const*, char const*>> const& macros = {},
      bool glsl_intermediate = false) noexcept -> std::array<ShaderHandle, N> {
      std::vector<std::pair<std::string, rhi::ShaderStageEnum>> vec;
      for (size_t i = 0; i < N; ++i)
        vec.push_back(std::make_pair(entrypoints[i].first, entrypoints[i].second));
      std::vector<ShaderHandle> handdles = GFXContext::load_shader_slang(filepath, vec, macros, glsl_intermediate);
      std::array<ShaderHandle, N> arr;
      for (size_t i = 0; i < N; ++i) arr[i] = handdles[i];
      return arr;
    }

    // load mesh resource
    // -------------------------------------------
    static auto create_mesh_empty() noexcept -> MeshHandle;

    // load material resource
    // -------------------------------------------
    static auto create_material_empty() noexcept -> MaterialHandle;

    // load mesh resource
    // -------------------------------------------
    static auto create_medium_empty() noexcept -> MediumHandle;

    // load scene resource
    // -------------------------------------------
    static auto load_scene_gltf(std::string const& path) noexcept -> SceneHandle;
    static auto load_scene_xml(std::string const& path) noexcept -> SceneHandle;
    static auto load_scene_pbrt(std::string const& path) noexcept -> SceneHandle;

    static auto frame_end() noexcept -> void;
  };
}

// ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
// ┃ image                                                                     ┃
// ┠───────────────────────────────────────────────────────────────────────────┨
// ┃ This is a submodule of gfx, that aims to support reading from / writing   ┃
// ┃ into all kinds of image formats. Also, it provides a CPU data structure   ┃
// ┃ Image, which is a unifying interface for interaction with other modules.  ┃
// ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
namespace image {
  // host texture
  struct Image {
    struct SubResource {
      uint32_t mip;
      uint32_t level;
      uint32_t offset;
      uint32_t size;
      uint32_t width;
      uint32_t height;
    };
    uvec3 m_extend;
    rhi::TextureFormat m_format;
    rhi::TextureDimension m_dimension;
    se::MiniBuffer m_buffer;
    uint32_t m_mipLevels = 0;
    uint32_t m_arrayLayers = 0;
    uint32_t m_dataOffset = 0;
    uint32_t m_dataSize = 0;
    std::vector<SubResource> m_subResources;

    Image() = default;
    Image(Image&&) = default;
    Image(Image const&) = delete;
    virtual ~Image() = default;
    auto operator=(Image&&)->Image & = default;
    auto operator=(Image const&)->Image & = delete;
    /** get texture descriptor for rhi */
    auto get_descriptor() noexcept -> rhi::TextureDescriptor;
    /** get the pointer to the binary image data */
    auto get_data() noexcept -> char const*;
  };

  /** A unifing interface to load an image file from path. 
    * The file format is inferenced from path extension. */
  auto load_image(std::string const& path) noexcept -> std::unique_ptr<Image>;

  struct PNG {
    static auto write_png(std::string const& path, uint32_t width,
      uint32_t height, uint32_t channel, float* data) noexcept -> void;
    static auto from_png(std::string const& path) noexcept -> std::unique_ptr<Image>;
  };

  struct JPEG {
    static auto write_jpeg(std::string const& path, uint32_t width,
      uint32_t height, uint32_t channel, float* data) noexcept -> void;
    static auto from_jpeg(std::string const& path) noexcept -> std::unique_ptr<Image>;
  };

  struct EXR {
    static auto write_exr(std::string const& path, uint32_t width,
      uint32_t height, uint32_t channel, float* data) noexcept -> void;
    static auto from_exr(std::string const& path) noexcept -> std::unique_ptr<Image>;
  };

  struct Binary {
    static auto from_binary(int width, int height, int channel, int bits,
      const char* path) noexcept -> std::unique_ptr<Image>;
  };
}
}

// ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
// ┃ Lights                                                                    ┃
// ┠───────────────────────────────────────────────────────────────────────────┨
// ┃ These are predefined materials intepretors.                               ┃
// ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
namespace se {
  namespace gfx {
    struct DirectionalLights {
      static auto init(Light* mat) noexcept -> void;
      static auto set_default(Light* mat) noexcept -> void;
      static auto draw_gui(Light* mat) noexcept -> void;
    };

    struct PointLights {
      static auto init(Light* mat) noexcept -> void;
      static auto set_default(Light* mat) noexcept -> void;
      static auto draw_gui(Light* mat) noexcept -> void;
    };

    struct SpotLights {
      static auto init(Light* mat) noexcept -> void;
      static auto set_default(Light* mat) noexcept -> void;
      static auto draw_gui(Light* mat) noexcept -> void;
    };
  }
}

// ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
// ┃ materials                                                                 ┃
// ┠───────────────────────────────────────────────────────────────────────────┨
// ┃ These are predefined materials intepretors.                               ┃
// ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
namespace se {
namespace gfx{
  struct LambertianMaterial {
    static auto init(Material* mat) noexcept -> void;
    static auto set_default(Material* mat) noexcept -> void;
    static auto draw_gui(Material* mat) noexcept -> void;
  };
}
}