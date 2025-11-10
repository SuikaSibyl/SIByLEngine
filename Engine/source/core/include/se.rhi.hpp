#pragma once
#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>
#include <memory>
#include <optional>
#include <future>
#include "se.utils.hpp"
#ifdef _WIN32
    #define VK_USE_PLATFORM_WIN32_KHR
    #include <vulkan/vulkan_win32.h>
#elif defined(__linux__)
#endif
#include <se.math.hpp>
#include <variant>

namespace se {
namespace rhi {
  struct Context;
  struct Adapter;
  struct Device;
  struct QueueFamilyIndices;
  struct CommandPool;
  struct CommandBuffer;
  struct BindGroupPool;
  struct TextureView;
  struct SwapChain;
  struct Semaphore;
  struct Fence;
  struct FrameResources;
  struct BLAS;
  struct TLAS;
  struct CommandEncoder;
  struct FrameBuffer;
  struct RenderPassEncoder;
  struct ComputePassEncoder;
  struct Barrier;             struct BarrierDescriptor;
  struct RenderPass;          struct RenderPassDescriptor;
  struct ComputePass;
  struct Buffer;              struct BufferDescriptor;
  struct Texture;             struct TextureDescriptor;
  struct Sampler;             struct SamplerDescriptor;
  struct ShaderModule;        struct ShaderModuleDescriptor;
  struct BindGroupLayout;     struct BindGroupLayoutDescriptor;
  struct ComputePipeline;     struct ComputePipelineDescriptor;
  struct RenderPipeline;      struct RenderPipelineDescriptor;
  struct PipelineLayout;      struct PipelineLayoutDescriptor;
  struct BindGroup;           struct BindGroupDescriptor;
  struct QuerySet;            struct QuerySetDescriptor;

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for context, adapter, device ...                          ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  /** Context Extensions for extending API capability */
  enum struct ContextExtensionEnum {
    NONE = 0 << 0,
    DEBUG_UTILS = 1 << 0,
    MESH_SHADER = 1 << 1,
    FRAGMENT_BARYCENTRIC = 1 << 2,
    SAMPLER_FILTER_MIN_MAX = 1 << 3,
    RAY_TRACING = 1 << 4,
    SHADER_NON_SEMANTIC_INFO = 1 << 5,
    BINDLESS_INDEXING = 1 << 6,
    ATOMIC_FLOAT = 1 << 7,
    CONSERVATIVE_RASTERIZATION = 1 << 8,
    COOPERATIVE_MATRIX = 1 << 9,
    CUDA_INTEROPERABILITY = 1 << 10,
    USE_AFTERMATH = 1 << 11,
  };
  ENABLE_BITMASK_OPERATORS(ContextExtensionEnum);

  /** Power preference */
  enum struct PowerPreferenceEnum { LOW_POWER, HIGH_PERFORMANCE, };


  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for buffers                                               ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  /** Current map state of the buffer */
  enum struct BufferMapState { UNMAPPED, PENDING, MAPPED };
  /** Shader mode of the buffer */
  enum struct BufferShareMode { CONCURRENT, EXCLUSIVE, };
  /** Buffer usage */
  enum struct BufferUsageEnum {
    MAP_READ = 1 << 0,
    MAP_WRITE = 1 << 1,
    COPY_SRC = 1 << 2,
    COPY_DST = 1 << 3,
    INDEX = 1 << 4,
    VERTEX = 1 << 5,
    UNIFORM = 1 << 6,
    STORAGE = 1 << 7,
    INDIRECT = 1 << 8,
    QUERY_RESOLVE = 1 << 9,
    SHADER_DEVICE_ADDRESS = 1 << 10,
    ACCELERATION_STRUCTURE_STORAGE = 1 << 11,
    ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY = 1 << 12,
    SHADER_BINDING_TABLE = 1 << 13,
    CUDA_ACCESS = 1 << 14,
  };
  ENABLE_BITMASK_OPERATORS(BufferUsageEnum);

  /** Determine the memory properties. */
  enum class MemoryPropertyEnum {
    DEVICE_LOCAL_BIT = 1 << 0,
    HOST_VISIBLE_BIT = 1 << 1,
    HOST_COHERENT_BIT = 1 << 2,
    HOST_CACHED_BIT = 1 << 3,
    LAZILY_ALLOCATED_BIT = 1 << 4,
    PROTECTED_BIT = 1 << 5,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
  };
  ENABLE_BITMASK_OPERATORS(MemoryPropertyEnum);

  /** Determine how a Buffer is mapped when calling map_async(). */
  enum struct MapModeEnum { READ = 1, WRITE = 2, ALL = 3 };
  ENABLE_BITMASK_OPERATORS(MapModeEnum);

  /** Descriptor to create buffer */
  struct BufferDescriptor {
    /** The size of the buffer in bytes. */
    size_t size;
    /** The allowed usages for the buffer. */
    Flags<BufferUsageEnum> usage = 0;
    /** The queue access share mode of the buffer. */
    BufferShareMode shareMode = BufferShareMode::EXCLUSIVE;
    /** The memory properties of the buffer. */
    Flags<MemoryPropertyEnum> memoryProperties = 0;
    /** If true creates the buffer in an already mapped state. */
    bool mappedAtCreation = false;
    /** buffer alignment (-1 means dont care) */
    int minimumAlignment = -1;

    BufferDescriptor() = default;
    BufferDescriptor(size_t size,
      Flags<BufferUsageEnum> _usage = 0,
      BufferShareMode _shareMode = BufferShareMode::EXCLUSIVE,
      Flags<MemoryPropertyEnum> _memoryProperties = 0,
      bool _mappedAtCreation = false,
      int _minimumAlignment = -1) :
      size(size), usage(_usage), shareMode(_shareMode),
      memoryProperties(_memoryProperties),
      mappedAtCreation(_mappedAtCreation),
      minimumAlignment(_minimumAlignment) {};
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for textures                                              ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  /** The dimension of texture */
  enum struct TextureDimension { TEX1D, TEX2D, TEX3D };
  /** The format of texture */
  enum struct TextureFormat {
    // Unknown
    UNKOWN,
    // 8-bit formats
    R8_UNORM,
    R8_SNORM,
    R8_UINT,
    R8_SINT,
    // 16-bit formats
    R16_UINT,
    R16_SINT,
    R16_FLOAT,
    RG8_UNORM,
    RG8_SNORM,
    RG8_UINT,
    RG8_SINT,
    // 32-bit formats
    R32_UINT,
    R32_SINT,
    R32_FLOAT,
    RG16_UINT,
    RG16_SINT,
    RG16_FLOAT,
    RGBA8_UNORM,
    RGBA8_UNORM_SRGB,
    RGBA8_SNORM,
    RGBA8_UINT,
    RGBA8_SINT,
    BGRA8_UNORM,
    BGRA8_UNORM_SRGB,
    // Packed 32-bit formats
    RGB9E5_UFLOAT,
    RG11B10_UFLOAT,
    // 64-bit formats
    RG32_UINT,
    RG32_SINT,
    RG32_FLOAT,
    RGBA16_UINT,
    RGBA16_SINT,
    RGBA16_FLOAT,
    // 128-bit formats
    RGBA32_UINT,
    RGBA32_SINT,
    RGBA32_FLOAT,
    // Depth/stencil formats
    STENCIL8,
    DEPTH16_UNORM,
    DEPTH24,
    DEPTH24STENCIL8,
    DEPTH32_FLOAT,
    // Compressed formats
    COMPRESSION,
    RGB10A2_UNORM,
    DEPTH32STENCIL8,
    BC1_RGB_UNORM_BLOCK,
    BC1_RGB_SRGB_BLOCK,
    BC1_RGBA_UNORM_BLOCK,
    BC1_RGBA_SRGB_BLOCK,
    BC2_UNORM_BLOCK,
    BC2_SRGB_BLOCK,
    BC3_UNORM_BLOCK,
    BC3_SRGB_BLOCK,
    BC4_UNORM_BLOCK,
    BC4_SNORM_BLOCK,
    BC5_UNORM_BLOCK,
    BC5_SNORM_BLOCK,
    BC6H_UFLOAT_BLOCK,
    BC6H_SFLOAT_BLOCK,
    BC7_UNORM_BLOCK,
    BC7_SRGB_BLOCK,
  };

  enum struct TextureAspectEnum {
    COLOR_BIT = 1 << 0,
    STENCIL_BIT = 1 << 1,
    DEPTH_BIT = 1 << 2,
  };
  ENABLE_BITMASK_OPERATORS(TextureAspectEnum);

  enum struct TextureUsageEnum {
    COPY_SRC = 1 << 0,
    COPY_DST = 1 << 1,
    TEXTURE_BINDING = 1 << 2,
    STORAGE_BINDING = 1 << 3,
    COLOR_ATTACHMENT = 1 << 4,
    DEPTH_ATTACHMENT = 1 << 5,
    TRANSIENT_ATTACHMENT = 1 << 6,
    INPUT_ATTACHMENT = 1 << 7,
  };
  ENABLE_BITMASK_OPERATORS(TextureUsageEnum);

  enum struct TextureFeatureEnum {
    NONE = 0,
    HOST_VISIBLE = 1 << 0,
    CUBE_COMPATIBLE = 1 << 1,
  };
  ENABLE_BITMASK_OPERATORS(TextureFeatureEnum);

  enum struct TextureLayoutEnum : uint32_t {
    UNDEFINED,
    GENERAL,
    COLOR_ATTACHMENT_OPTIMAL,
    DEPTH_STENCIL_ATTACHMENT_OPTIMA,
    DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    SHADER_READ_ONLY_OPTIMAL,
    TRANSFER_SRC_OPTIMAL,
    TRANSFER_DST_OPTIMAL,
    PREINITIALIZED,
    DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
    DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    DEPTH_ATTACHMENT_OPTIMAL,
    DEPTH_READ_ONLY_OPTIMAL,
    STENCIL_ATTACHMENT_OPTIMAL,
    STENCIL_READ_ONLY_OPTIMAL,
    PRESENT_SRC,
    SHARED_PRESENT,
    FRAGMENT_DENSITY_MAP_OPTIMAL,
    FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL,
    READ_ONLY_OPTIMAL,
    ATTACHMENT_OPTIMAL,
  };

  /* Define a region of buffer copy from in image_copy */
  struct ImageCopyBuffer {
    uint64_t offset = 0;
    uint32_t bytesPerRow;
    uint32_t rowsPerImage;
    Buffer* buffer;
  };

  /* Define a region of texture copy from/to in image_copy */
  struct ImageCopyTexture {
    Texture* texutre;
    uint32_t mipLevel = 0;
    uvec3 origin = {};
    Flags<TextureAspectEnum> aspect = TextureAspectEnum::COLOR_BIT;
  };

  /* The subresource range of texture */
  struct TextureRange {
    Flags<TextureAspectEnum> aspectMask = 0;
    uint32_t baseMipLevel;
    uint32_t levelCount;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
  };

  /* The behavior describing how to clean a part of texture. */
  struct TextureClearDescriptor {
    std::vector<TextureRange> subresources;
    se::vec4 clearColor;
  };

  /* The descriptor defining a texture resource. */
  struct TextureDescriptor {
    uvec3 size;
    uint32_t mipLevelCount = 1;
    uint32_t arrayLayerCount = 1;
    uint32_t sampleCount = 1;
    TextureDimension dimension = TextureDimension::TEX2D;
    TextureFormat format;
    /** The allowed usages for the texture. */
    Flags<TextureUsageEnum> usage = 0;
    /** Specifies what view format values will be allowed when calling createView()
     * on this texture (in addition to the texture’s actual format). */
    std::vector<TextureFormat> viewFormats;
    Flags<TextureFeatureEnum> flags = TextureFeatureEnum::NONE;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for textures view                                         ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  /** The dimension of textureview */
  enum struct TextureViewDimension {
    TEX1D, TEX1D_ARRAY,
    TEX2D, TEX2D_ARRAY,
    CUBE, CUBE_ARRAY,
    TEX3D, TEX3D_ARRAY,
  };

  enum struct TextureViewType : uint32_t {
    SRV,
    UAV,
    RTV,
    DSV,
  };

  struct TextureViewDescriptor {
    TextureFormat format;
    TextureViewDimension dimension = TextureViewDimension::TEX2D;
    Flags<TextureAspectEnum> aspect = TextureAspectEnum::COLOR_BIT;
    uint32_t baseMipLevel = 0;
    uint32_t mipLevelCount = 1;
    uint32_t baseArrayLayer = 0;
    uint32_t arrayLayerCount = 1;
  };

  struct TextureViewIndex {
    rhi::TextureViewType type;
    uint32_t mostDetailedMip;
    uint32_t mipCount;
    uint32_t firstArraySlice;
    uint32_t arraySize;
    bool operator==(TextureViewIndex const& p) const;
  };

  struct TextureViewIndexHash {
    std::size_t operator()(TextureViewIndex const& s) const {
      return std::hash<uint32_t>()(s.mostDetailedMip) + std::hash<uint32_t>()(s.mipCount) +
        std::hash<uint32_t>()(s.firstArraySlice) + std::hash<uint32_t>()(s.arraySize);
    }
  };

  auto has_depth_bit(TextureFormat format) noexcept -> bool;
  auto has_stencil_bit(TextureFormat format) noexcept -> bool;
  auto get_texture_aspect(TextureFormat format) noexcept -> Flags<TextureAspectEnum>;
  auto get_vk_image_layout(TextureLayoutEnum layout) noexcept -> VkImageLayout;

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for samplers                                              ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  /** The address mode of sampler */
  enum struct AddressMode {
    CLAMP_TO_EDGE,
    REPEAT,
    MIRROR_REPEAT,
  };

  enum struct FilterMode {
    NEAREST,
    LINEAR,
  };

  enum struct MipmapFilterMode {
    NEAREST,
    LINEAR,
  };

  enum struct CompareFunction {
    NEVER,
    LESS,
    EQUAL,
    LESS_EQUAL,
    GREATER,
    NOT_EQUAL,
    GREATER_EQUAL,
    ALWAYS,
  };

  struct SamplerDescriptor {
    AddressMode addressModeU = AddressMode::REPEAT;
    AddressMode addressModeV = AddressMode::REPEAT;
    AddressMode addressModeW = AddressMode::REPEAT;
    FilterMode magFilter = FilterMode::LINEAR;
    FilterMode minFilter = FilterMode::LINEAR;
    MipmapFilterMode mipmapFilter = MipmapFilterMode::LINEAR;
    float lodMinClamp = 0.f;
    float lodMapClamp = 32.f;
    CompareFunction compare = CompareFunction::ALWAYS;
    uint16_t maxAnisotropy = 1;
    float maxLod = 32.f;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for shaders                                               ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  enum struct ShaderStageEnum {
    VERTEX = 1 << 0,
    FRAGMENT = 1 << 1,
    COMPUTE = 1 << 2,
    GEOMETRY = 1 << 3,
    RAYGEN = 1 << 4,
    MISS = 1 << 5,
    CLOSEST_HIT = 1 << 6,
    INTERSECTION = 1 << 7,
    ANY_HIT = 1 << 8,
    CALLABLE = 1 << 9,
    TASK = 1 << 10,
    MESH = 1 << 11,
  };
  ENABLE_BITMASK_OPERATORS(ShaderStageEnum);

  struct ShaderModuleDescriptor {
    /** The shader source code for the shader module. */
    se::MiniBuffer* code;
    /** stage */
    ShaderStageEnum stage;
    /** name of entry point */
    std::string name = "main";
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for accleration structures                                ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  struct AffineTransformMatrix {
    AffineTransformMatrix() = default;
    AffineTransformMatrix(se::mat4 const& mat) {
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) matrix[i][j] = mat.data[i][j];
    }
    operator se::mat4() const {
      se::mat4 mat;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) mat.data[i][j] = matrix[i][j];
      mat.data[3][0] = 0;
      mat.data[3][1] = 0;
      mat.data[3][2] = 0;
      mat.data[3][3] = 1;
      return mat;
    }
    float matrix[3][4] = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0} };
  };

  enum struct BLASGeometryEnum : uint32_t {
    NONE = 0 << 0,
    OPAQUE_GEOMETRY = 1 << 0,
    NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1,
  };

  enum struct IndexFormat { UINT16_t, UINT32_T };

  struct BLASTriangleGeometry {
    Buffer* positionBuffer = nullptr;
    Buffer* indexBuffer = nullptr;
    IndexFormat indexFormat = IndexFormat::UINT16_t;
    uint32_t maxVertex = 0;
    uint32_t firstVertex = 0;
    uint32_t primitiveCount = 0;
    uint32_t primitiveOffset = 0;
    AffineTransformMatrix transform;
    Flags<BLASGeometryEnum> geometryFlags = 0;
    uint32_t materialID = 0;
    uint32_t vertexStride = 3 * sizeof(float);
    uint32_t vertexByteOffset = 0;
    enum struct VertexFormat {
      RGB32,
      RG32
    } vertexFormat = VertexFormat::RGB32;
  };

  struct BLASCustomGeometry {
    AffineTransformMatrix transform;
    std::vector<se::bounds3> aabbs;
    Flags<BLASGeometryEnum> geometryFlags = 0;
  };

  struct BLASDescriptor {
    std::vector<BLASTriangleGeometry> triangleGeometries;
    bool allowRefitting = false;
    bool allowCompaction = false;
    std::vector<BLASCustomGeometry> customGeometries;
  };

  struct BLASInstance {
    BLAS* blas = nullptr;
    se::mat4 transform = {};
    uint32_t instanceCustomIndex = 0;  // is used by system now
    uint32_t instanceShaderBindingTableRecordOffset = 0;
    uint32_t mask = 0xFF;
  };

  struct TLASDescriptor {
    std::vector<BLASInstance> instances;
    bool allowRefitting = false;
    bool allowCompaction = false;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for binding                                               ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  enum struct BindingResourceType {
    SAMPLER,
    TEXTURE_VIEW,
    BUFFER_BINDING,
    BINDLESS_TEXTURE,
    None,
  };

  enum struct BufferBindingType {
    UNIFORM, 
    STORAGE, 
    READ_ONLY_STORAGE 
  };

  enum struct SamplerBindingType { 
    FILTERING, 
    NON_FILTERING, 
    COMPARISON, 
  };

  struct BufferBindingLayout {
    BufferBindingType type = BufferBindingType::UNIFORM;
    bool hasDynamicOffset = false;
    size_t minBindingSize = 0;
  };

  struct SamplerBindingLayout {
    SamplerBindingType type = SamplerBindingType::FILTERING;
  };

  struct TextureBindingLayout {
    TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
    bool multisampled = false;
  };

  struct StorageTextureBindingLayout {
    TextureFormat format;
    TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
  };
  struct BindlessTexturesBindingLayout {};
  struct AccelerationStructureBindingLayout {};

  enum class AccessFlagEnum : uint32_t {
    INDIRECT_COMMAND_READ_BIT = 0x00000001,
    INDEX_READ_BIT = 0x00000002,
    VERTEX_ATTRIBUTE_READ_BIT = 0x00000004,
    UNIFORM_READ_BIT = 0x00000008,
    INPUT_ATTACHMENT_READ_BIT = 0x00000010,
    SHADER_READ_BIT = 0x00000020,
    SHADER_WRITE_BIT = 0x00000040,
    COLOR_ATTACHMENT_READ_BIT = 0x00000080,
    COLOR_ATTACHMENT_WRITE_BIT = 0x00000100,
    DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200,
    DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400,
    TRANSFER_READ_BIT = 0x00000800,
    TRANSFER_WRITE_BIT = 0x00001000,
    HOST_READ_BIT = 0x00002000,
    HOST_WRITE_BIT = 0x00004000,
    MEMORY_READ_BIT = 0x00008000,
    MEMORY_WRITE_BIT = 0x00010000,
    TRANSFORM_FEEDBACK_WRITE_BIT = 0x02000000,
    TRANSFORM_FEEDBACK_COUNTER_READ_BIT = 0x04000000,
    TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT = 0x08000000,
    CONDITIONAL_RENDERING_READ_BIT = 0x00100000,
    COLOR_ATTACHMENT_READ_NONCOHERENT_BIT = 0x00080000,
    ACCELERATION_STRUCTURE_READ_BIT = 0x00200000,
    ACCELERATION_STRUCTURE_WRITE_BIT = 0x00400000,
    FRAGMENT_DENSITY_MAP_READ_BIT = 0x01000000,
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT = 0x00800000,
    COMMAND_PREPROCESS_READ_BIT = 0x00020000,
    COMMAND_PREPROCESS_WRITE_BIT = 0x00040000,
    NONE = 0,
  };
  ENABLE_BITMASK_OPERATORS(AccessFlagEnum);

  enum class DependencyTypeEnum : uint32_t {
    NONE = 0 << 0,
    BY_REGION_BIT = 1 << 0,
    VIEW_LOCAL_BIT = 1 << 1,
    DEVICE_GROUP_BIT = 1 << 2,
  };
  ENABLE_BITMASK_OPERATORS(DependencyTypeEnum);

  enum class PipelineStageEnum : uint32_t {
    TOP_OF_PIPE_BIT = 0x00000001,
    DRAW_INDIRECT_BIT = 0x00000002,
    VERTEX_INPUT_BIT = 0x00000004,
    VERTEX_SHADER_BIT = 0x00000008,
    TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
    TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
    GEOMETRY_SHADER_BIT = 0x00000040,
    FRAGMENT_SHADER_BIT = 0x00000080,
    EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
    LATE_FRAGMENT_TESTS_BIT = 0x00000200,
    COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
    COMPUTE_SHADER_BIT = 0x00000800,
    TRANSFER_BIT = 0x00001000,
    BOTTOM_OF_PIPE_BIT = 0x00002000,
    HOST_BIT = 0x00004000,
    ALL_GRAPHICS_BIT = 0x00008000,
    ALL_COMMANDS_BIT = 0x00010000,
    TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
    CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
    ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
    RAY_TRACING_SHADER_BIT_KHR = 0x00200000,
    TASK_SHADER_BIT_NV = 0x00080000,
    MESH_SHADER_BIT_NV = 0x00100000,
    FRAGMENT_DENSITY_PROCESS_BIT = 0x00800000,
    FRAGMENT_SHADING_RATE_ATTACHMENT_BIT = 0x00400000,
    COMMAND_PREPROCESS_BIT = 0x00020000,
  };
  ENABLE_BITMASK_OPERATORS(PipelineStageEnum);

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for query                                                 ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  enum struct QueryType {
    OCCLUSION,
    PIPELINE_STATISTICS,
    TIMESTAMP,
  };

  enum struct QueryResultEnum {
    RESULT_64 = 0x1,
    RESULT_WAIT = 0x1 << 1,
    RESULT_WITH_AVAILABILITY = 0x1 << 2,
    RESULT_PARTIAL = 0x1 << 3,
  };
  ENABLE_BITMASK_OPERATORS(QueryResultEnum);

  struct QuerySetDescriptor {
    QueryType type;
    uint32_t count;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Pre-definitions for pipelines                                             ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  /** Could specify features in rasterization stage.
  * Like: Conservative rasterization. */
  struct RasterizeState {
    // The mode of conservative rasterization.
    // Read the article from NV for more information and understanding:
    // @url: https://developer.nvidia.com/content/dont-be-conservative-conservative-rasterization
    enum struct ConservativeMode {
      DISABLED,
      OVERESTIMATE,
      UNDERESTIMATE,
    } mode = ConservativeMode::DISABLED;
    // extraPrimitiveOverestimationSize is the extra size in pixels to increase
    // the generating primitive during conservative rasterization at each of its
    // edges in X and Y equally in screen space beyond the base overestimation
    // specified in
    // VkPhysicalDeviceConservativeRasterizationPropertiesEXT::primitiveOverestimationSize.
    // If conservativeRasterizationMode is not
    // VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT, this value is ignored.
    float extraPrimitiveOverestimationSize = 0.f;
  };

  enum struct PrimitiveTopology {
    POINT_LIST,
    LINE_LIST,
    LINE_STRIP,
    TRIANGLE_LIST,
    TRIANGLE_STRIP
  };

  /** Determine polygons with vertices whose framebuffer coordinates
   * are given in which order are considered front-facing. */
  enum struct FrontFace {
    CCW,  // counter-clockwise
    CW,   // clockwise
  };

  enum struct CullMode {
    NONE,
    FRONT,
    BACK,
    BOTH,
  };

  struct MultisampleState {
    uint32_t count = 1;
    uint32_t mask = 0xFFFFFFFF;
    bool alphaToCoverageEnabled = false;
  };

  enum struct BlendFactor {
    ZERO,
    ONE,
    SRC,
    ONE_MINUS_SRC,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA,
    DST,
    ONE_MINUS_DST,
    DST_ALPHA,
    ONE_MINUS_DST_ALPHA,
    SRC_ALPHA_SATURATED,
    CONSTANT,
    ONE_MINUS_CONSTANT,
  };

  enum struct BlendOperation {
    ADD,
    SUBTRACT,
    REVERSE_SUBTRACT,
    MIN,
    MAX,
  };

  struct BlendComponent {
    BlendOperation operation = BlendOperation::ADD;
    BlendFactor srcFactor = BlendFactor::ONE;
    BlendFactor dstFactor = BlendFactor::ZERO;
  };

  struct BlendState {
    BlendComponent color;
    BlendComponent alpha;
    /** check whether the blend should be enabled */
    inline auto blendEnable() const noexcept -> bool {
      return !((color.operation == BlendOperation::ADD) &&
        (color.srcFactor == BlendFactor::ONE) &&
        (color.dstFactor == BlendFactor::ZERO) &&
        (alpha.operation == BlendOperation::ADD) &&
        (alpha.srcFactor == BlendFactor::ONE) &&
        (alpha.dstFactor == BlendFactor::ZERO));
    }
  };

  struct ColorTargetState {
    TextureFormat format;
    BlendState blend;
    uint32_t writeMask = 0xF;
  };

  struct FragmentState {
    ShaderModule* module;
    std::string entryPoint;
    std::vector<ColorTargetState> targets;
  };

  enum struct RenderPassTimestampLocation {
    BEGINNING,
    END,
  };

  struct RenderPassTimestampWrite {
    QuerySet* querySet = nullptr;
    uint32_t queryIndex = 0;
    RenderPassTimestampLocation location = 
      RenderPassTimestampLocation::BEGINNING;
  };

  /** Describes the entry point in the user-provided ShaderModule that
   * controls one of the programmable stages of a pipeline. */
  struct ProgrammableStage {
    ShaderModule* module;
    std::string entryPoint;
  };

  enum struct LoadOp {
    DONT_CARE,
    LOAD,
    CLEAR,
  };

  enum struct StoreOp { DONT_CARE, STORE, DISCARD };

  struct RenderPassColorAttachment {
    TextureView* view;
    se::vec4 clearValue;
    LoadOp loadOp;
    StoreOp storeOp;
  };

  enum struct VertexStepMode {
    VERTEX,
    INSTANCE,
  };

  struct PrimitiveState {
    PrimitiveTopology topology = PrimitiveTopology::TRIANGLE_LIST;
    IndexFormat stripIndexFormat;
    FrontFace frontFace = FrontFace::CCW;
    CullMode cullMode = CullMode::NONE;
    bool unclippedDepth = false;
  };

  enum struct StencilOperation {
    KEEP,
    ZERO,
    REPLACE,
    INVERT,
    INCREMENT_CLAMP,
    DECREMENT_CLAMP,
    INCREMENT_WARP,
    DECREMENT_WARP,
  };

  struct StencilFaceState {
    CompareFunction compare = CompareFunction::ALWAYS;
    StencilOperation failOp = StencilOperation::KEEP;
    StencilOperation depthFailOp = StencilOperation::KEEP;
    StencilOperation passOp = StencilOperation::KEEP;
  };

  struct DepthStencilState {
    TextureFormat format;
    bool depthWriteEnabled = false;
    CompareFunction depthCompare = CompareFunction::ALWAYS;
    StencilFaceState stencilFront = {};
    StencilFaceState stencilBack = {};
    uint32_t stencilReadMask = 0xFFFFFFFF;
    uint32_t stencilWriteMask = 0xFFFFFFFF;
    int32_t depthBias = 0;
    float depthBiasSlopeScale = 0;
    float depthBiasClamp = 0;
  };

  enum struct VertexFormat {
    UINT8X2,
    UINT8X4,
    SINT8X2,
    SINT8X4,
    UNORM8X2,
    UNORM8X4,
    SNORM8X2,
    SNORM8X4,
    UINT16X2,
    UINT16X4,
    SINT16X2,
    SINT16X4,
    UNORM16X2,
    UNORM16X4,
    SNORM16X2,
    SNORM16X4,
    FLOAT16X2,
    FLOAT16X4,
    FLOAT32,
    FLOAT32X2,
    FLOAT32X3,
    FLOAT32X4,
    UINT32,
    UINT32X2,
    UINT32X3,
    UINT32X4,
    SINT32,
    SINT32X2,
    SINT32X3,
    SINT32X4
  };

  struct VertexAttribute {
    VertexFormat format;
    size_t offset;
    uint32_t shaderLocation;
  };

  struct VertexBufferLayout {
    size_t arrayStride;
    VertexStepMode stepMode = VertexStepMode::VERTEX;
    std::vector<VertexAttribute> attributes;
  };

  struct VertexState {
    ShaderModule* module;
    std::string entryPoint;
    std::vector<VertexBufferLayout> buffers;
  };

  struct RenderPassDepthStencilAttachment {
    TextureView* view = nullptr;
    float depthClearValue = 0;
    LoadOp depthLoadOp = LoadOp::DONT_CARE;
    StoreOp depthStoreOp = StoreOp::DONT_CARE;
    bool depthReadOnly = false;
    uint32_t stencilClearValue = 0;
    LoadOp stencilLoadOp = LoadOp::DONT_CARE;
    StoreOp stencilStoreOp = StoreOp::DONT_CARE;
    bool stencilReadOnly = false;
  };

  struct RenderPassDescriptor {
    std::vector<RenderPassColorAttachment> colorAttachments;
    RenderPassDepthStencilAttachment depthStencilAttachment = {};
    // std::unique_ptr<QuerySet> occlusionQuerySet = nullptr;
    std::vector<RenderPassTimestampWrite> timestampWrites = {};
    uint64_t maxDrawCount = 50000000;

    RenderPassDescriptor() {}
    RenderPassDescriptor(
      std::vector<RenderPassColorAttachment> const& color_attachment
    ) : colorAttachments(color_attachment) {}
    RenderPassDescriptor(
      std::vector<RenderPassColorAttachment> const& v1,
      RenderPassDepthStencilAttachment const& v2) :
      colorAttachments(v1), depthStencilAttachment(v2), maxDrawCount(50000000) {}
  };

  struct ComputePipelineDescriptor {
    /** The definition of the layout of resources which can be used with this */
    PipelineLayout* layout = nullptr;
    /* the compute shader */
    ProgrammableStage compute;
  };

  struct RenderPipelineDescriptor {
    /** The definition of the layout of resources which can be used with this */
    PipelineLayout* layout = nullptr;
    // required structures
    // --------------------------------------------------------------
    VertexState vertex;                   // vertex shader stage
    PrimitiveState primitive = {};        // vertex primitive config
    DepthStencilState depthStencil;       // OM pipeline config
    MultisampleState multisample = {};    // multisample config
    FragmentState fragment;               // fragment shader stage
    // optional structures
    // --------------------------------------------------------------
    ProgrammableStage geometry;           // geometry shader stage
    ProgrammableStage task;               // task shader stage
    ProgrammableStage mesh;               // mesh shader stage
    RasterizeState rasterize;             // rasterize pipeline stage
  };

  struct DebugLabelDescriptor {
    std::string name;
    se::vec4 color;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Definitions for context, adapter, queue, device.                          ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  struct Context {
    Window* m_bindedWindow = nullptr;
    VkInstance m_instance;
    VkSurfaceKHR m_surface;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    Flags<ContextExtensionEnum> m_extensions = 0;
    std::vector<VkPhysicalDevice> m_devices;
    std::vector<const char*> m_deviceExtensions = {};

    /** Initialize the context */
    Context(Window* window = nullptr, Flags<ContextExtensionEnum> ext = 0);
    /** virtual destructor */
    ~Context() { destroy(); }
    /** clean up context resources */
    auto destroy() noexcept -> void;
    /** Request an adapter */
    auto request_adapter(PowerPreferenceEnum pp = PowerPreferenceEnum::HIGH_PERFORMANCE
      ) noexcept -> std::unique_ptr<Adapter>;
    /** Get the binded window */
    auto get_binded_window() const noexcept -> Window* { return m_bindedWindow; }
    /** get Context Extensions Flags */
    auto get_context_extensions_flags() const noexcept -> Flags<ContextExtensionEnum> { return m_extensions; }
    /** get VkInstance */
    auto get_vk_instance() noexcept -> VkInstance& { return m_instance; }
    /** get VkSurfaceKHR */
    auto get_vk_surface_khr() noexcept -> VkSurfaceKHR& { return m_surface; }
    /** get DebugMessageFunc */
    auto get_vk_debug_messenger() noexcept -> VkDebugUtilsMessengerEXT& { return m_debugMessenger; }
    /** get All Device Extensions required */
    auto get_vk_device_extensions() noexcept -> std::vector<const char*>& { return m_deviceExtensions; }
    /** get All VkPhysicalDevices available */
    auto get_vk_physical_devices() noexcept -> std::vector<VkPhysicalDevice>& { return m_devices; }

    // ======================================================================
    // Ext Func Pointers
    typedef void(VKAPI_PTR* PFN_vkCmdBeginDebugUtilsLabelEXT)(
      VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT* pLabelInfo);
    PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT;
    typedef void(VKAPI_PTR* PFN_vkCmdEndDebugUtilsLabelEXT)(
      VkCommandBuffer commandBuffer);
    PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT;
    typedef VkResult(VKAPI_PTR* PFN_vkSetDebugUtilsObjectNameEXT)(
      VkDevice device, const VkDebugUtilsObjectNameInfoEXT* pNameInfo);
    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
    typedef VkResult(VKAPI_PTR* PFN_vkSetDebugUtilsObjectTagEXT)(
      VkDevice device, const VkDebugUtilsObjectTagInfoEXT* pTagInfo);
    PFN_vkSetDebugUtilsObjectTagEXT vkSetDebugUtilsObjectTagEXT;
    // Mesh Shader Ext Func Pointers
    typedef void(VKAPI_PTR* PFN_vkCmdDrawMeshTasksNV)(
      VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask);
    PFN_vkCmdDrawMeshTasksNV vkCmdDrawMeshTasksNV;
    // Ray Tracing Ext Func Pointers
    typedef void(VKAPI_PTR* PFN_vkCmdTraceRaysKHR)(
      VkCommandBuffer commandBuffer,
      const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
      const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
      const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
      const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
      uint32_t width, uint32_t height, uint32_t depth);
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
    typedef VkResult(VKAPI_PTR* PFN_vkCreateRayTracingPipelinesKHR)(
      VkDevice device, VkDeferredOperationKHR deferredOperation,
      VkPipelineCache pipelineCache, uint32_t createInfoCount,
      const VkRayTracingPipelineCreateInfoKHR* pCreateInfos,
      const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
    typedef VkResult(
      VKAPI_PTR* PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)(
        VkDevice device, VkPipeline pipeline, uint32_t firstGroup,
        uint32_t groupCount, size_t dataSize, void* pData);
    PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR
      vkGetRayTracingCaptureReplayShaderGroupHandlesKHR;
    typedef void(VKAPI_PTR* PFN_vkCmdTraceRaysIndirectKHR)(
      VkCommandBuffer commandBuffer,
      const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
      const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
      const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
      const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
      VkDeviceAddress indirectDeviceAddress);
    PFN_vkCmdTraceRaysIndirectKHR vkCmdTraceRaysIndirectKHR;
    typedef VkDeviceSize(VKAPI_PTR* PFN_vkGetRayTracingShaderGroupStackSizeKHR)(
      VkDevice device, VkPipeline pipeline, uint32_t group,
      VkShaderGroupShaderKHR groupShader);
    PFN_vkGetRayTracingShaderGroupStackSizeKHR
      vkGetRayTracingShaderGroupStackSizeKHR;
    typedef void(VKAPI_PTR* PFN_vkCmdSetRayTracingPipelineStackSizeKHR)(
      VkCommandBuffer commandBuffer, uint32_t pipelineStackSize);
    PFN_vkCmdSetRayTracingPipelineStackSizeKHR
      vkCmdSetRayTracingPipelineStackSizeKHR;
    typedef VkResult(VKAPI_PTR* PFN_vkCreateAccelerationStructureNV)(
      VkDevice device, const VkAccelerationStructureCreateInfoNV* pCreateInfo,
      const VkAllocationCallbacks* pAllocator,
      VkAccelerationStructureNV* pAccelerationStructure);
    PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructureNV;
    typedef void(VKAPI_PTR* PFN_vkDestroyAccelerationStructureNV)(
      VkDevice device, VkAccelerationStructureNV accelerationStructure,
      const VkAllocationCallbacks* pAllocator);
    PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructureNV;
    typedef void(VKAPI_PTR* PFN_vkGetAccelerationStructureMemoryRequirementsNV)(
      VkDevice device,
      const VkAccelerationStructureMemoryRequirementsInfoNV* pInfo,
      VkMemoryRequirements2KHR* pMemoryRequirements);
    PFN_vkGetAccelerationStructureMemoryRequirementsNV
      vkGetAccelerationStructureMemoryRequirementsNV;
    typedef VkResult(VKAPI_PTR* PFN_vkBindAccelerationStructureMemoryNV)(
      VkDevice device, uint32_t bindInfoCount,
      const VkBindAccelerationStructureMemoryInfoNV* pBindInfos);
    PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemoryNV;
    typedef void(VKAPI_PTR* PFN_vkCmdBuildAccelerationStructureNV)(
      VkCommandBuffer commandBuffer, const VkAccelerationStructureInfoNV* pInfo,
      VkBuffer instanceData, VkDeviceSize instanceOffset, VkBool32 update,
      VkAccelerationStructureNV dst, VkAccelerationStructureNV src,
      VkBuffer scratch, VkDeviceSize scratchOffset);
    PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructureNV;
    typedef void(VKAPI_PTR* PFN_vkCmdCopyAccelerationStructureNV)(
      VkCommandBuffer commandBuffer, VkAccelerationStructureNV dst,
      VkAccelerationStructureNV src, VkCopyAccelerationStructureModeKHR mode);
    PFN_vkCmdCopyAccelerationStructureNV vkCmdCopyAccelerationStructureNV;
    typedef void(VKAPI_PTR* PFN_vkCmdTraceRaysNV)(
      VkCommandBuffer commandBuffer, VkBuffer raygenShaderBindingTableBuffer,
      VkDeviceSize raygenShaderBindingOffset,
      VkBuffer missShaderBindingTableBuffer,
      VkDeviceSize missShaderBindingOffset,
      VkDeviceSize missShaderBindingStride,
      VkBuffer hitShaderBindingTableBuffer, VkDeviceSize hitShaderBindingOffset,
      VkDeviceSize hitShaderBindingStride,
      VkBuffer callableShaderBindingTableBuffer,
      VkDeviceSize callableShaderBindingOffset,
      VkDeviceSize callableShaderBindingStride, uint32_t width, uint32_t height,
      uint32_t depth);
    PFN_vkCmdTraceRaysNV vkCmdTraceRaysNV;
    typedef VkResult(VKAPI_PTR* PFN_vkCreateRayTracingPipelinesNV)(
      VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
      const VkRayTracingPipelineCreateInfoNV* pCreateInfos,
      const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);
    PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelinesNV;
    typedef VkResult(VKAPI_PTR* PFN_vkGetRayTracingShaderGroupHandlesKHR)(
      VkDevice device, VkPipeline pipeline, uint32_t firstGroup,
      uint32_t groupCount, size_t dataSize, void* pData);
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
    typedef VkResult(VKAPI_PTR* PFN_vkGetRayTracingShaderGroupHandlesNV)(
      VkDevice device, VkPipeline pipeline, uint32_t firstGroup,
      uint32_t groupCount, size_t dataSize, void* pData);
    PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandlesNV;
    typedef VkResult(VKAPI_PTR* PFN_vkGetAccelerationStructureHandleNV)(
      VkDevice device, VkAccelerationStructureNV accelerationStructure,
      size_t dataSize, void* pData);
    PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandleNV;
    typedef void(VKAPI_PTR* PFN_vkCmdWriteAccelerationStructuresPropertiesNV)(
      VkCommandBuffer commandBuffer, uint32_t accelerationStructureCount,
      const VkAccelerationStructureNV* pAccelerationStructures,
      VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery);
    PFN_vkCmdWriteAccelerationStructuresPropertiesNV
      vkCmdWriteAccelerationStructuresPropertiesNV;
    typedef VkResult(VKAPI_PTR* PFN_vkCompileDeferredNV)(VkDevice device,
      VkPipeline pipeline, uint32_t shader);
    PFN_vkCompileDeferredNV vkCompileDeferredNV;
    typedef void(VKAPI_PTR* PFN_vkGetAccelerationStructureBuildSizesKHR)(
      VkDevice device, VkAccelerationStructureBuildTypeKHR buildType,
      const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
      const uint32_t* pMaxPrimitiveCounts,
      VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo);
    PFN_vkGetAccelerationStructureBuildSizesKHR
      vkGetAccelerationStructureBuildSizesKHR;
    typedef void(VKAPI_PTR* PFN_vkCmdBuildAccelerationStructuresKHR)(
      VkCommandBuffer commandBuffer, uint32_t infoCount,
      const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
      const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos);
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    typedef VkResult(VKAPI_PTR* PFN_vkCreateAccelerationStructureKHR)(
      VkDevice device, const VkAccelerationStructureCreateInfoKHR* pCreateInfo,
      const VkAllocationCallbacks* pAllocator,
      VkAccelerationStructureKHR* pAccelerationStructure);
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    typedef void(VKAPI_PTR* PFN_vkDestroyAccelerationStructureKHR)(
      VkDevice device, VkAccelerationStructureKHR accelerationStructure,
      const VkAllocationCallbacks* pAllocator);
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    typedef VkDeviceAddress(
      VKAPI_PTR* PFN_vkGetAccelerationStructureDeviceAddressKHR)(
        VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR* pInfo);
    PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR;
    typedef void(VKAPI_PTR* PFN_vkCmdCopyAccelerationStructureKHR)(
      VkCommandBuffer commandBuffer,
      const VkCopyAccelerationStructureInfoKHR* pInfo);
    PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR;
    #ifdef _WIN32
        PFN_vkGetMemoryWin32HandleKHR vkCmdGetMemoryWin32HandleKHR;
    #elif defined(__linux__)
        PFN_vkGetMemoryFdKHR vkCmdGetMemoryFdKHR;
    #endif
  };

  struct Adapter {
    /** information of adapter */
    struct AdapterInfo {
      std::string vendor;
      std::string architecture;
      std::string device;
      std::string description;
      float timestampPeriod;
    };

    /** the context the adapter is requested from */
    Context* m_context = nullptr;
    /** adapter information */
    AdapterInfo const m_adapterInfo;
    /** the graphics card selected */
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    /** the timestamp period for timestemp query */
    float m_timestampPeriod = 0.0f;
    /** QueueFamilyIndices */
    struct QueueFamilyIndices {
      std::optional<uint32_t> m_graphicsFamily;
      std::optional<uint32_t> m_presentFamily;
      std::optional<uint32_t> m_computeFamily;
      /** check whether queue families are complete */
      auto is_complete() noexcept -> bool;
    } m_queueFamilyIndices;
    /** vulkan physical device properties */
    VkPhysicalDeviceProperties m_properties = {};

    /** constructor */
    Adapter(VkPhysicalDevice device, Context* context,
      VkPhysicalDeviceProperties const& properties);
    /** Requests a device from the adapter. */
    auto request_device() noexcept -> std::unique_ptr<Device>;
    /** Requests the AdapterInfo for this Adapter. */
    auto request_adapter_info() const noexcept -> AdapterInfo { return m_adapterInfo; }
    /** get the context the context created from */
    auto from_which_context() noexcept -> Context* { return m_context; }
    /** get VkPhysicalDevice */
    auto get_vk_physical_device() noexcept -> VkPhysicalDevice& { return m_physicalDevice; }
    /** get All Device Extensions required */
    auto get_vk_device_extensions() noexcept -> std::vector<const char*>&;
    /* get VkPhysicalDeviceProperties */
    auto get_vk_physical_device_properties() const noexcept -> VkPhysicalDeviceProperties const& { return m_properties; }
    /** get TimestampPeriod */
    auto get_timestamp_period() const noexcept -> float { return m_timestampPeriod; }
    /** get QueueFamilyIndices_VK */
    auto get_queue_family_indices() const noexcept -> QueueFamilyIndices const& { return m_queueFamilyIndices; }
    /** get All Device Extensions required */
    auto find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) noexcept -> uint32_t;
  };

  struct Queue {
    /** Vulkan queue handle */
    VkQueue m_queue;
    /* the device this buffer is created on */
    Device* m_device = nullptr;

    /** Schedules the execution of the command buffers by the GPU on this queue. */
    auto submit(std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void;
    /** Schedules the execution of the command buffers by the GPU on this queue.
     * With sync objects */
    auto submit(std::vector<CommandBuffer*> const& commandBuffers, Fence* fence) noexcept -> void;
    /** Schedules the execution of the command buffers by the GPU on this queue.
     * With sync objects */
    auto submit(std::vector<CommandBuffer*> const& commandBuffers,
      Semaphore* wait, Semaphore* signal, Fence* fence) noexcept -> void;
    /** Schedules the execution of the command buffers by the GPU on this queue.
     * With sync objects */
    auto submit(std::vector<CommandBuffer*> const& commandBuffers,
      std::vector<Semaphore*> const& wait_semaphores,
      std::vector<size_t>     const& wait_indices,
      std::vector<Flags<PipelineStageEnum>> const& wait_stages,
      std::vector<Semaphore*> const& signal_semaphores,
      std::vector<size_t>     const& signal_indices,
      Fence* fence
    ) noexcept -> void;

    /** Present swap chain. */
    auto present_swapchain(SwapChain* swapchain, uint32_t imageIndex,
      Semaphore* semaphore) noexcept -> void;
    /** wait until idle */
    auto wait_idle() noexcept -> void;
    /** set debug name */
    auto set_name(std::string const& name) -> void;
  };

  struct Device {
    /** vulkan logical device handle */
    VkDevice m_device;
    /** various queue handles */
    Queue m_graphicsQueue, m_computeQueue, m_presentQueue;
    /** various queue command pools */
    std::unique_ptr<CommandPool> m_graphicPool = nullptr;
    std::unique_ptr<CommandPool> m_computePool = nullptr;
    std::unique_ptr<CommandPool> m_presentPool = nullptr;
    /** the adapter from which this device was created */
    Adapter* m_adapter = nullptr;
    /** VMA allocator */
    VmaAllocator m_allocator = nullptr;
    /** bind group pool */
    std::unique_ptr<BindGroupPool> m_bindGroupPool = nullptr;
    /** whether the debug layer is enabled */
    bool m_debugLayerEnabled = false;

    /** device ray tracing properties */
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_vkRayTracingProperties;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR m_vASProperties;

    Device() = default; ~Device();
    /** initialzie */
    auto init() noexcept -> void;
    /** destroy */
    auto destroy() noexcept -> void;
    /** wait until idle */
    auto wait_idle() noexcept -> void;
    /** get the adapter the device created from */
    auto from_which_adapter() noexcept -> Adapter* { return m_adapter; }
    /** get graphics queue handle */
    auto get_graphics_queue() noexcept -> Queue& { return m_graphicsQueue; }
    /** get compute queue handle */
    auto get_compute_queue() noexcept -> Queue& { return m_computeQueue; }
    /** get present queue handle */
    auto get_present_queue() noexcept -> Queue& { return m_presentQueue; }
    /** get vulkan logical device handle */
    auto get_vk_device() noexcept -> VkDevice& { return m_device; }
    /** get the VMA allocator */
    auto get_vma_allocator() noexcept -> VmaAllocator& { return m_allocator; }
    // Create resources on device
    // ---------------------------
    /** create a buffer on the device */
    auto create_buffer(BufferDescriptor const& desc) noexcept -> std::unique_ptr<Buffer>;
    /** import an external buffer on the device */
    auto import_buffer(void* externalHandle, BufferDescriptor const& desc) noexcept -> std::unique_ptr<Buffer>;
    /** create a texture on the device */
    auto create_texture(TextureDescriptor const& desc) noexcept -> std::unique_ptr<Texture>;
    /** create a sampler on the device */
    auto create_sampler(SamplerDescriptor const& desc) noexcept -> std::unique_ptr<Sampler>;
    /* create a swapchain on the device */
    auto create_swapchain() noexcept -> std::unique_ptr<SwapChain>;
    /* create a blas on the device */
    auto create_blas(BLASDescriptor const& desc) -> std::unique_ptr<BLAS>;
    /* create a tlas on the device */
    auto create_tlas(TLASDescriptor const& desc) -> std::unique_ptr<TLAS>;
    // Create memory barrier objects
    // ---------------------------
    auto create_fence() noexcept -> std::unique_ptr<Fence>;
    // Create resources binding objects
    // ---------------------------
    /** create a bind group layout on the device */
    auto create_bindgroup_layout(
      BindGroupLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroupLayout>;
    /** create a pipeline layout on the device */
    auto create_pipeline_layout(
      PipelineLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<PipelineLayout>;
    /** create a bind group on the device */
    auto create_bindgroup(BindGroupDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroup>;
    // Create pipeline objects
    // ---------------------------
    /** create a shader module on the device */
    auto create_shader_module(ShaderModuleDescriptor const& desc) noexcept
      -> std::unique_ptr<ShaderModule>;
    /** create a compute pipeline on the device */
    auto create_compute_pipeline(
      ComputePipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePipeline>;
    /** create a render pipeline on the device */
    auto create_render_pipeline(
      RenderPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPipeline>;
    // Create command encoders
    // ---------------------------
    /** create a command encoder */
    auto create_command_encoder(CommandBuffer* external = nullptr) noexcept
      -> std::unique_ptr<CommandEncoder>;
    /** create a multi frame flights */
    auto create_frame_resources(int maxFlightNum = 1,
      SwapChain* swapchain = nullptr) noexcept
      -> std::unique_ptr<FrameResources>;
    /** create command pools */
    auto allocate_command_buffer() noexcept -> std::unique_ptr<CommandBuffer>;
    /** get bind group pool */
    auto get_bindgroup_pool() noexcept -> BindGroupPool* { return m_bindGroupPool.get(); }
    /** create a semaphore */
    auto create_semaphore(bool use_timeline = true, bool allow_export = true) noexcept -> std::unique_ptr<Semaphore>;
    // utilities: help create resources in a more handy way
    // ---------------------------------------------------------
    /** create a device local buffer with initialzie value */
    auto create_device_local_buffer(void const* data, uint32_t size,
      Flags<BufferUsageEnum> usage) noexcept -> std::unique_ptr<Buffer>;
    /** read back device local buffer */
    auto readback_device_local_buffer(Buffer* buffer, void* data, uint32_t size) noexcept -> void;
    /** create CUDA context extension */
    auto query_uuid() noexcept -> std::array<uint64_t, 2>;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Definitions for commands objects.                                         ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  struct CommandPool {
    /** vulkan command pool */
    VkCommandPool m_commandPool;
    /** the device this command pool is created on */
    Device* m_device = nullptr;

    CommandPool(Device*);
    ~CommandPool();
    /** allocate command buffer */
    auto allocate_command_buffer() noexcept -> std::unique_ptr<CommandBuffer>;
  };

  struct CommandBuffer {
    /** vulkan command buffer */
    VkCommandBuffer m_commandBuffer;
    /** command pool the buffer is on */
    CommandPool* m_commandPool = nullptr;
    /** the device this command buffer is created on */
    Device* m_device = nullptr;

    ~CommandBuffer();
  };

  struct CommandEncoder {
    /** underlying command buffer */
    std::unique_ptr<CommandBuffer> m_commandBufferOnce = nullptr;
    /** underlying command buffer */
    CommandBuffer* m_commandBuffer = nullptr;

    /** Begins encoding a render pass described by descriptor. */
    auto begin_render_pass(RenderPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPassEncoder>;
    /** Begins encoding a compute pass described by descriptor. */
    auto begin_compute_pass() noexcept -> std::unique_ptr<ComputePassEncoder>;
    ///** Begins encoding a ray tracing pass described by descriptor. */
    //auto begin_raytracing_pass(
    //  RayTracingPassDescriptor const& desc) noexcept
    //  -> std::unique_ptr<RayTracingPassEncoder>;
    /** Insert a barrier. */
    auto pipeline_barrier(BarrierDescriptor const& desc) noexcept -> void;
    /**  Encode a command into the CommandEncoder that copies data from
      * a sub-region of a GPUBuffer to a sub-region of another Buffer. */
    auto copy_buffer_to_buffer(Buffer* source, size_t sourceOffset,
      Buffer* destination, size_t destinationOffset, size_t size) noexcept -> void;
    /** Encode a command into the CommandEncoder that fills a sub-region of a
      * Buffer with zeros. */
    auto clear_buffer(Buffer* buffer, size_t offset, size_t size) noexcept -> void;
    /** Encode a command into the CommandEncoder that fills a sub-region of a
      * Buffer with a value. */
    auto fill_buffer(Buffer* buffer, size_t offset, size_t size,
      float fillValue) noexcept -> void;
    /** Encode a command into the CommandEncoder that fills a sub-region of a
      * texture with any color. */
    auto clear_texture(Texture* texture,
      TextureClearDescriptor const& desc) noexcept -> void;
    /** Encode a command into the CommandEncoder that copies data from a
      * sub-region of a Buffer to a sub-region of one or multiple continuous
      * texture subresources. */
    auto copy_buffer_to_texture(
      ImageCopyBuffer const& source,
      ImageCopyTexture const& destination,
      uvec3 const& copySize) noexcept -> void;
    /** Encode a command into the CommandEncoder that copies data from
      * a sub-region of one or multiple contiguous texture subresources to
      * another sub-region of one or multiple continuous texture subresources. */
    auto copy_texture_to_texture(
      ImageCopyTexture const& source,
      ImageCopyTexture const& destination,
      uvec3 const& copySize) noexcept -> void;
    /** Reset the queryset. */
    auto reset_query_set(QuerySet* querySet, uint32_t firstQuery,
      uint32_t queryCount) noexcept -> void;
    /** Writes a timestamp value into a querySet when all
     * previous commands have completed executing. */
    auto write_timestamp(QuerySet* querySet, PipelineStageEnum stageMask,
      uint32_t queryIndex) noexcept -> void;

    /** Completes recording of the commands sequence and returns a corresponding
      * GPUCommandBuffer. */
    auto finish() noexcept -> CommandBuffer*;
    /** begin Debug marker region */
    auto begin_debug_utils_label(std::string const& name, se::vec4 color) noexcept -> void;
    /** end Debug marker region */
    auto end_debug_utils_label() noexcept -> void;
  };

  struct RenderPassEncoder {
    /** render pass */
    std::unique_ptr<RenderPass> m_renderPass = nullptr;
    /** frame buffer */
    std::unique_ptr<FrameBuffer> m_frameBuffer = nullptr;
    /* current render pipeline */
    RenderPipeline* m_renderPipeline = nullptr;
    /** command buffer binded */
    CommandBuffer* m_commandBuffer = nullptr;

    /** Sets the current GPURenderPipeline. */
    auto set_pipeline(RenderPipeline* pipeline) noexcept -> void;
    /** Sets the current index buffer. */
    auto set_index_buffer(Buffer* buffer, IndexFormat indexFormat,
      uint64_t offset = 0, uint64_t size = 0) noexcept -> void;
    /** Sets the current vertex buffer for the given slot. */
    auto set_vertex_buffer(uint32_t slot, Buffer* buffer,
      uint64_t offset = 0, uint64_t size = 0) noexcept -> void;
    /** Draws primitives. */
    auto draw(uint32_t vertexCount, uint32_t instanceCount = 1,
      uint32_t firstVertex = 0,
      uint32_t firstInstance = 0) noexcept -> void;
    /** Draws indexed primitives. */
    auto draw_indexed(uint32_t indexCount, uint32_t instanceCount = 1,
      uint32_t firstIndex = 0, int32_t baseVertex = 0,
      uint32_t firstInstance = 0) noexcept -> void;
    /** Draws primitives using parameters read from a GPUBuffer. */
    auto draw_indirect(Buffer* indirectBuffer, uint64_t indirectOffset,
      uint32_t drawCount, uint32_t stride) noexcept -> void;

    /** Draws indexed primitives using parameters read from a GPUBuffer. */
    auto draw_indexed_indirect(Buffer* indirectBuffer, uint64_t offset,
      uint32_t drawCount, uint32_t stride) noexcept -> void;
    /** Sets the viewport used during the rasterization stage to linearly map
     * from normalized device coordinates to viewport coordinates. */
    auto set_viewport(float x, float y, float width, float height,
      float minDepth, float maxDepth) noexcept -> void;
    /** Sets the scissor rectangle used during the rasterization stage.
     * After transformation into viewport coordinates any fragments
     * which fall outside the scissor rectangle will be discarded. */
    auto set_scissor_rect(uint32_t x, uint32_t y,
      uint32_t width, uint32_t height) noexcept -> void;
    /** Completes recording of the render pass commands sequence. */
    auto end() noexcept -> void;
    /** Sets the current GPUBindGroup for the given index. */
    auto set_bindgroup( uint32_t index, BindGroup* bindgroup,
      std::vector<uint32_t> const& dynamicOffsets = {}) noexcept -> void;
    /** Sets the current GPUBindGroup for the given index. */
    auto set_bindgroup(uint32_t index, BindGroup* bindgroup,
      uint64_t dynamicOffsetDataStart,
      uint32_t dynamicOffsetDataLength) noexcept -> void;
    /** Push constants */
    auto push_constants(void* data, Flags<ShaderStageEnum> stages,
      uint32_t offset, uint32_t size) noexcept -> void;
    template<class T>
    auto push_constants(T const& data, Flags<ShaderStageEnum> stages) noexcept -> void {
      push_constants((void*)&data, stages, 0, sizeof(T));
    }
  };

  struct ComputePassEncoder {
    /** current compute pipeline */
    ComputePipeline* m_computePipeline = nullptr;
    /** command buffer binded */
    CommandBuffer* m_commandBuffer = nullptr;

    /** Sets the current GPUBindGroup for the given index. */
    auto set_bindgroup(uint32_t index, BindGroup* bindgroup,
      std::vector<uint32_t> const& dynamicOffsets = {}) noexcept -> void;
    /** Sets the current GPUBindGroup for the given index. */
    auto set_bindgroup(uint32_t index, BindGroup* bindgroup,
      uint64_t dynamicOffsetDataStart,
      uint32_t dynamicOffsetDataLength) noexcept -> void;
    /** Push constants */
    auto push_constants(void* data, Flags<ShaderStageEnum> stages,
      uint32_t offset, uint32_t size) noexcept -> void;
    /** Sets the current GPUComputePipeline. */
    auto set_pipeline(ComputePipeline* pipeline) noexcept -> void;
    /** Dispatch work to be performed with the current GPUComputePipeline.*/
    auto dispatch_workgroups(uint32_t workgroupCountX,
      uint32_t workgroupCountY = 1,
      uint32_t workgroupCountZ = 1) noexcept -> void;
    /** Dispatch work to be performed with the current GPUComputePipeline using
      * parameters read from a GPUBuffer. */
    auto dispatch_workgroups_indirect(Buffer* indirectBuffer,
      uint64_t indirectOffset) noexcept -> void;
    /** Completes recording of the compute pass commands sequence. */
    auto end() noexcept -> void;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Definitions for resource objects.                                         ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  struct Buffer {
    /** vulkan buffer */
    VkBuffer m_buffer = {};
    /** vulkan buffer device memory */
    VkDeviceMemory m_bufferMemory = {};
    /** buffer creation desc */
    BufferDescriptor m_descriptor;
    /** buffer creation desc */
    BufferMapState m_mapState = BufferMapState::UNMAPPED;
    /** mapped address of the buffer */
    void* m_mappedData = nullptr;
    /** size of the buffer */
    size_t m_size = 0;
    /** the device this buffer is created on */
    Device* m_device = nullptr;
    /** VMA allocation */
    VmaAllocation m_allocation;
    /** debug name */
    std::string m_name;
    /** external imported */
    bool m_external = false;

    /** constructor */
    Buffer(Device* device) : m_device(device) {}
    /** virtual destructor */
    virtual ~Buffer();
    /** copy functions */
    Buffer(Buffer const& buffer) = delete;
    Buffer(Buffer&& buffer) = default;
    auto operator=(Buffer const& buffer)->Buffer & = delete;
    auto operator=(Buffer&& buffer)->Buffer&;
    // Readonly Attributes
    // ---------------------------
    /** readonly get buffer size on GPU */
    auto size() const noexcept -> size_t { return m_size; }
    /** readonly get buffer usage flags on GPU */
    auto buffer_usage_flags() const noexcept -> Flags<BufferUsageEnum> { return m_descriptor.usage; }
    /** readonly get map state on GPU */
    auto buffer_map_state() const noexcept -> BufferMapState { return m_mapState; }
    /** readonly get device */
    auto get_device() const noexcept -> Device* { return m_device; }
    /** get the device memory address is applicable */
    auto get_device_address() const noexcept -> uint64_t;
    /** get the memory handle of the allocated buufer */
    struct ExternalHandle { void* handle; size_t offset; size_t size; };
    auto get_mem_handle() const noexcept -> ExternalHandle;
    // Map methods
    // ---------------------------
    /** Maps the given range of the GPUBuffer */
    auto map_async(Flags<MapModeEnum> mode, size_t offset = 0,
      size_t size = 0) noexcept -> std::future<bool>;
    /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
     * mapped range */
    auto get_mapped_range(size_t offset = 0) noexcept -> void*;
    /** Unmaps the mapped range of the GPUBuffer and makes its contents available
     * for use by the GPU again. */
    auto unmap() noexcept -> void;
    /** destroy the buffer */
    auto destroy() noexcept -> void;
    /** set debug name */
    auto set_name(std::string const& name) -> void;
    /** set debug name */
    auto get_name() const noexcept -> std::string const&;
    /** initialize the buffer */
    auto init(Device* device, size_t size,
      BufferDescriptor const& desc) noexcept -> void;
    /** get vulkan buffer */
    auto get_vk_buffer() noexcept -> VkBuffer& { return m_buffer; }
    /** get vulkan buffer device memory */
    auto get_vk_device_memory() noexcept -> VkDeviceMemory& { return m_bufferMemory; }
    /** get vulkan buffer device memory */
    auto get_vma_allocation() noexcept -> VmaAllocation& { return m_allocation; }
    /** set buffer state */
    auto set_buffer_map_state(BufferMapState const& state) noexcept -> void { m_mapState = state; }
  };

  struct Texture {
    /** vulkan image */
    VkImage m_image;
    /** vulkan image device memory */
    VkDeviceMemory m_deviceMemory = nullptr;
    /** Texture Descriptor */
    TextureDescriptor m_descriptor;
    /** VMA allocation */
    VmaAllocation m_allocation;
    /** buffer map state */
    BufferMapState m_mapState = BufferMapState::UNMAPPED;
    /** mapped address of the buffer */
    void* m_mappedData = nullptr;
    /** the device this texture is created on */
    Device* m_device = nullptr;
    /** name */
    std::string m_name = "Unnamed Texture";
    /** the VKImage is reference to an external resource */
    bool m_external = false;

    // Texture Behaviors
    // ---------------------------
    /** virtual descructor */
    Texture() noexcept = default;
    Texture(Texture&&) noexcept = default;
    Texture(Texture const&) noexcept = delete;
    Texture(Device* device, TextureDescriptor const& desc);
    Texture(Device* device, VkImage image, TextureDescriptor const& desc);
    auto operator=(Texture&& texture)->Texture & = default;
    auto operator=(Texture const& texture)->Texture & = delete;
    ~Texture();

    /** create texture view of this texture */
    auto create_view(TextureViewDescriptor const& desc) noexcept
      -> std::unique_ptr<TextureView>;
    /** destroy this texture */
    auto destroy() noexcept -> void;
    // Readonly Attributes
    // ---------------------------
    /** readonly width of the texture */
    auto width() const noexcept -> uint32_t { return m_descriptor.size.x; }
    /** readonly height of the texture */
    auto height() const noexcept -> uint32_t { return m_descriptor.size.y; }
    /** readonly depth or arrayLayers of the texture */
    auto depth_or_array_layers() const noexcept -> uint32_t { return m_descriptor.size.z; }
    /** readonly mip level count of the texture */
    auto mip_level_count() const noexcept -> uint32_t;
    /** readonly sample count of the texture */
    auto sample_count() const noexcept -> uint32_t;
    /** the dimension of the set of texel for each of this GPUTexture's
     * subresources. */
    auto dimension() const noexcept -> TextureDimension;
    /** readonly format of the texture */
    auto format() const noexcept -> TextureFormat;
    /** set debug name */
    auto set_name(std::string const& name) -> void;
    /** get name */
    auto get_name() -> std::string const&;
    /** get texture descriptor */
    auto get_descriptor() -> TextureDescriptor;
    // Map methods
    // ---------------------------
    /** Maps the given range of the GPUBuffer */
    auto map_async(Flags<MapModeEnum> mode, size_t offset = 0,
      size_t size = 0) noexcept -> std::future<bool>;
    /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
     * mapped range */
    auto get_mapped_range(size_t offset = 0, size_t size = 0) noexcept -> void*;
    /** Unmaps the mapped range of the GPUBuffer and makes it’s contents available
     * for use by the GPU again. */
    auto unmap() noexcept -> void;
    /** get the VkImage */
    auto get_vk_image() noexcept -> VkImage& { return m_image; }
    /** get the VkDeviceMemory */
    auto get_vk_device_memory() noexcept -> VkDeviceMemory& { return m_deviceMemory; }
    /* get VMA Allocation */
    auto get_vma_allocation() noexcept -> VmaAllocation& { return m_allocation; }
    /** set buffer state */
    auto set_buffer_map_state(BufferMapState const& state) noexcept -> void { m_mapState = state; }
  };

  struct TextureView {
    /** Vulkan texture view */
    VkImageView m_imageView;
    /** Texture view descriptor */
    TextureViewDescriptor m_descriptor;
    /** The texture this view is pointing to */
    Texture* m_texture = nullptr;
    /** The device that the pointed texture is created on */
    Device* m_device = nullptr;
    /** width of the view */
    uint32_t m_width;
    /** height of the view */
    uint32_t m_height;

    /* ctor */
    TextureView() noexcept = default;
    TextureView(TextureView&& view) noexcept;
    TextureView(TextureView const& view) = delete;
    TextureView(Device* device, Texture* texture,
      TextureViewDescriptor const& descriptor);
    /* copy functions */
    auto operator=(TextureView const& view)->TextureView & = delete;
    auto operator=(TextureView&& view)->TextureView&;
    /** virtual destructor */
    ~TextureView();
    /** set debug name */
    auto set_name(std::string const& name) -> void;
    /** get binded texture */
    auto get_texture() noexcept -> Texture* { return m_texture; }
    /** get width */
    auto get_width() noexcept -> uint32_t { return m_width; }
    /** get height */
    auto get_height() noexcept -> uint32_t { return m_height; }
  };

  struct Sampler {
    /** vulkan Texture Sampler */
    VkSampler m_textureSampler;
    /** the device this sampler is created on */
    Device* m_device = nullptr;
    /** debug name */
    std::string m_name;

    /** initialization */
    Sampler(SamplerDescriptor const& desc, Device* device);
    /** virtual destructor */
    ~Sampler();
    /** set debug name */
    auto set_name(std::string const& name) -> void;
    /** get debug name */
    auto get_name() const noexcept -> std::string const&;
  };

  struct SwapChain {
    /** vulkan SwapChain */
    VkSwapchainKHR m_swapChain;
    /** vulkan SwapChain Extent */
    VkExtent2D m_swapChainExtend;
    /** vulkan SwapChain format */
    VkFormat m_swapChainImageFormat;
    /** vulkan SwapChain fetched images */
    std::vector<Texture> m_swapChainTextures;
    /** vulkan SwapChain fetched images views */
    std::vector<TextureView> m_textureViews;
    /** the device this sampler is created on */
    Device* m_device = nullptr;
    
    /** virtual destructor */
    virtual ~SwapChain();
    SwapChain() = default;
    SwapChain(SwapChain&&) = default;
    SwapChain(SwapChain const&) = delete;
    SwapChain& operator= (SwapChain&&) = default;
    SwapChain& operator= (SwapChain const&) = delete;
    /** intialize the swapchin */
    auto init(Device* device) noexcept -> void;
    /** get texture view */
    auto get_texture(int i) noexcept -> Texture* { return &m_swapChainTextures[i]; }
    auto get_texture_view(int i) noexcept -> TextureView* { return &m_textureViews[i]; }
    /** invalid swapchain */
    auto recreate() noexcept -> void;
  };

  struct FrameBuffer {
    /** vulkan framebuffer */
    VkFramebuffer m_framebuffer = {};
    /** clear values */
    std::vector<VkClearValue> m_clearValues = {};
    /** vulkan device the framebuffer created on */
    Device* m_device = nullptr;
    /** width / height */
    uint32_t m_width = 0, m_height = 0;

    /** intializer */
    FrameBuffer(Device* device, RenderPassDescriptor const& desc, RenderPass* renderpass);
    /** destructor */
    ~FrameBuffer();
    /** get width of the framebuffer */
    auto width() -> uint32_t { return m_width; }
    /** get height of the framebuffer */
    auto height() -> uint32_t { return m_height; }
  };

  struct FrameResources {
    std::vector<std::unique_ptr<CommandBuffer>> m_commandBuffers;
    std::vector<Semaphore> m_imageAvailableSemaphores;
    std::vector<Semaphore> m_renderFinishedSemaphores;
    std::vector<Fence> m_inFlightFences;
    SwapChain* m_swapChain = nullptr;
    uint32_t m_currentFrame = 0;
    int m_maxFlightNum = 0;
    Device* m_device = nullptr;
    uint32_t m_imageIndex;

    FrameResources() = default;
    FrameResources(FrameResources&&) = default;
    FrameResources(FrameResources const&) = delete;
    FrameResources& operator= (FrameResources&&) = default;
    FrameResources& operator= (FrameResources const&) = delete;
    /** initialize */
    FrameResources(Device* device, int maxFlightNum = 2, SwapChain* swapchain = nullptr);
    /** virtual destructor */
    ~FrameResources() = default;
    /** start frame */
    auto frame_start() noexcept -> void;
    /** end frame */
    auto frame_end() noexcept -> void;
    /** get current flight id */
    auto get_flight_index() noexcept -> uint32_t { return m_currentFrame; }
    /** get current swapchain id */
    auto get_swapchain_index() noexcept -> uint32_t { return m_imageIndex; }
    /** get current command buffer */
    auto get_command_buffer() noexcept -> CommandBuffer*;
    /** get current Image Available Semaphore */
    auto get_image_available_semaphore() noexcept -> Semaphore*;
    /** get current Render Finished Semaphore */
    auto get_render_finished_semaphore() noexcept -> Semaphore*;
    /** get current fence */
    auto get_fence() noexcept -> Fence*;
    /** rest all */
    auto reset() noexcept -> void; 
  };

  struct ShaderModule {
    /** the shader stages included in this module */
    Flags<ShaderStageEnum> m_stages = 0;
    /** vulkan shader module */
    VkShaderModule m_shaderModule = {};
    /** name of entry point */
    std::string m_entryPoint = "main";
    /** vulkan shader stage create info */
    VkPipelineShaderStageCreateInfo m_shaderStageInfo{};
    /** the device this shader module is created on */
    Device* m_device = nullptr;
    /** name of shader module */
    std::string m_name;

    /** initalize shader module */
    ShaderModule(Device* device, ShaderModuleDescriptor const& desc);
    /** virtual descructor */
    ~ShaderModule();
    /** copy functions */
    ShaderModule(ShaderModule const& shader) = delete;
    ShaderModule(ShaderModule&& shader);
    auto operator=(ShaderModule const& shader)->ShaderModule & = delete;
    auto operator=(ShaderModule&& shader)->ShaderModule&;
    /** set debug name */
    auto set_name(std::string const& name) -> void;
    /** get debug name */
    auto get_name() -> std::string const&;
  };

  struct BLAS {
    /** VULKAN BLAS object*/
    VkAccelerationStructureKHR m_blas = {};
    /** VULKAN BLAS buffer */
    std::unique_ptr<Buffer> m_bufferBLAS = nullptr;
    /** descriptor is remained */
    BLASDescriptor m_descriptor;
    /** vulkan device the BLAS is created on */
    Device* m_device = nullptr;

    /** initialzie */
    BLAS(Device* device, BLASDescriptor const& descriptor);
    /** only allocate according to another BLAS */
    BLAS(Device* device, BLAS* src);
    /** virtual destructor */
    ~BLAS();
    /** get descriptor */
    auto get_descriptor() noexcept -> BLASDescriptor { return m_descriptor; }
  };

  struct TLAS {
    /** VULKAN TLAS object*/
    VkAccelerationStructureKHR m_tlas = {};
    /** VULKAN TLAS buffer */
    std::unique_ptr<Buffer> m_bufferTLAS = nullptr;
    /** vulkan device the TLAS is created on */
    Device* m_device = nullptr;

    /** initialzie */
    TLAS(Device* device, TLASDescriptor const& descriptor);
    /** virtual destructor */
    virtual ~TLAS();
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Definitions for binding objects.                                          ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  /**
   * Describes a single shader resource binding
   * to be included in a GPUBindGroupLayout.  */
  struct BindGroupLayoutEntry {
    /** create an entry with a buffer resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility, BufferBindingLayout const& buffer)
      : binding(binding), visibility(visibility), buffer(buffer) {}
    /** create an entry with a sampler resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility, SamplerBindingLayout const& sampler)
      : binding(binding), visibility(visibility), sampler(sampler) {}
    /** create an entry with a texture resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility, TextureBindingLayout const& texture)
      : binding(binding), visibility(visibility), texture(texture) {}
    /** create an entry with a texture-sampler combination resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility,
      TextureBindingLayout const& texture, SamplerBindingLayout const& sampler)
      : binding(binding), visibility(visibility), texture(texture), sampler(sampler) {}
    /** create an entry with a storage texture resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility, StorageTextureBindingLayout const& storageTexture)
      : binding(binding), visibility(visibility), storageTexture(storageTexture) {}
    /** create an entry with a acceleration structure resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility,
      AccelerationStructureBindingLayout const& accelerationStructure)
      : binding(binding), visibility(visibility), accelerationStructure(accelerationStructure) {}
    /** create an entry with a bindless texture array resource */
    BindGroupLayoutEntry(uint32_t binding, Flags<ShaderStageEnum> visibility, BindlessTexturesBindingLayout const& bindlessTextures)
      : binding(binding), visibility(visibility), bindlessTextures(bindlessTextures) {}
    /* A unique identifier for a resource binding within the BindGroupLayout */
    uint32_t binding;
    /* Array of elements binded on the binding position */
    uint32_t array_size = 1;
    /** indicates whether it will be accessible from the associated shader stage */
    Flags<ShaderStageEnum> visibility;
    // possible resources layout types
    // ---------------------------
    std::optional<BufferBindingLayout> buffer;
    std::optional<SamplerBindingLayout> sampler;
    std::optional<TextureBindingLayout> texture;
    std::optional<StorageTextureBindingLayout> storageTexture;
    std::optional<AccelerationStructureBindingLayout> accelerationStructure;
    std::optional<BindlessTexturesBindingLayout> bindlessTextures;
  };

  struct BindGroupLayoutDescriptor {
    std::vector<BindGroupLayoutEntry> entries;
  };

  struct BindGroupLayout {
    /** vulkan Descriptor Set Layout */
    VkDescriptorSetLayout m_layout;
    /** Bind Group Layout Descriptor */
    BindGroupLayoutDescriptor m_descriptor;
    /** the device this bind group layout is created on */
    Device* m_device = nullptr;

    /** contructor */
    BindGroupLayout(Device* device, BindGroupLayoutDescriptor const& desc);
    /** destructor */
    ~BindGroupLayout();
    /** get BindGroup Layout Descriptor */
    auto get_bindgroup_layout_descriptor() const noexcept
      -> BindGroupLayoutDescriptor const&;
  };

  struct BufferBinding { 
    Buffer* buffer; 
    size_t offset; 
    size_t size; 
  };

  struct BindingResource {
    BindingResource() = default;
    BindingResource(TextureView* view, Sampler* sampler)
      : type(BindingResourceType::SAMPLER), textureView(view), sampler(sampler) {}
    BindingResource(Sampler* sampler)
      : type(BindingResourceType::SAMPLER), sampler(sampler) {}
    BindingResource(TextureView* view)
      : type(BindingResourceType::TEXTURE_VIEW), textureView(view) {}
    BindingResource(BufferBinding const& buffer)
      : type(BindingResourceType::BUFFER_BINDING), bufferBinding(buffer) {}
    // Binding bindless texture-sampler-pair array,
    // where every texture share the same sampler
    BindingResource(std::vector<TextureView*> const& bindlessTextures, Sampler* sampler)
      : type(BindingResourceType::BINDLESS_TEXTURE), bindlessTextures(bindlessTextures), sampler(sampler) {}
    // Binding bindless texture-sampler-pair array,
    // where every texture bind its own sampler
    BindingResource(std::vector<TextureView*> const& bindlessTextures, std::vector<Sampler*> const& samplers)
      : type(BindingResourceType::BINDLESS_TEXTURE), bindlessTextures(bindlessTextures), samplers(samplers) {}
    // Binding bindless storage texture array
    BindingResource(std::vector<TextureView*> const& storageTextures)
      : type(BindingResourceType::TEXTURE_VIEW), storageArray(storageTextures) {}
    // Binding a tlas resource
    BindingResource(TLAS* tlas) : tlas(tlas) {}
    BindingResource(std::vector<TLAS*> const& tlas) : tlasList(tlas) {}
    BindingResourceType type = BindingResourceType::None;
    Sampler* sampler = nullptr;
    TextureView* textureView = nullptr;
    std::vector<Sampler*> samplers = {};
    std::vector<TextureView*> bindlessTextures = {};
    std::vector<TextureView*> storageArray = {};
    std::optional<BufferBinding> bufferBinding;
    TLAS* tlas = nullptr;
    std::vector<TLAS*> tlasList = {};
  };

  struct BindGroupEntry { 
    uint32_t binding; 
    BindingResource resource; 
  };

  struct BindGroupDescriptor {
    BindGroupLayout* layout;
    std::vector<BindGroupEntry> entries;
  };

  struct BindGroupPool {
    /** vulkan Bind Group Pool */
    VkDescriptorPool m_descriptorPool;
    /** the device this bind group pool is created on */
    Device* m_device = nullptr;

    /** initialzier */
    BindGroupPool(Device* device);
    /** destructor */
    ~BindGroupPool();
  };

  struct BindGroup {
    /** vulkan Descriptor Set */
    VkDescriptorSet m_set = {};
    /** the bind group set this bind group is created on */
    BindGroupPool* m_descriptorPool;
    /** layout */
    BindGroupLayout* m_layout;
    /** the device this bind group is created on */
    Device* m_device = nullptr;

    /** initialzie */
    BindGroup(Device* device, BindGroupDescriptor const& desc);
    /** update binding */
    auto update_binding(
      std::vector<BindGroupEntry> const& entries) noexcept -> void;
  };

  struct PushConstantEntry {
    Flags<ShaderStageEnum> shaderStages = 0;
    uint32_t offset;
    uint32_t size;
  };

  struct PipelineLayoutDescriptor {
    std::vector<PushConstantEntry> pushConstants;
    std::vector<BindGroupLayout*> bindGroupLayouts;
  };

  struct PipelineLayout {
    /** vulkan pipeline layout */
    VkPipelineLayout m_pipelineLayout;
    /** the push constans on pipeline layouts */
    std::vector<VkPushConstantRange> m_pushConstants;
    /** the device this pipeline layout is created on */
    Device* m_device = nullptr;

    /** intializer */
    PipelineLayout(Device* device, PipelineLayoutDescriptor const& desc);
    /** virtual destructor */
    virtual ~PipelineLayout();
    /** copy functions */
    PipelineLayout(PipelineLayout const& layout) = delete;
    PipelineLayout(PipelineLayout&& layout);
    auto operator=(PipelineLayout const& layout)
      ->PipelineLayout & = delete;
    auto operator=(PipelineLayout&& layout)->PipelineLayout&;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ Definitions for synchronize and query objects.                            ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  struct Fence {
    VkFence m_fence;
    Device* m_device = nullptr;

    Fence() = default;
    Fence(Device* device);
    ~Fence();
    /* wait the fence */
    auto wait() noexcept -> void;
    /* reset the fence */
    auto reset() noexcept -> void;
  };

  struct Semaphore {
    VkSemaphore m_semaphore;
    Device* m_device = nullptr;
    bool m_timelineSemaphore = false;
    size_t m_currentValue = 0;

    Semaphore() = default;
    Semaphore(Device* device);
    ~Semaphore();
    auto current_host() noexcept -> size_t;
    auto current_device() noexcept -> size_t;
    auto signal(size_t value) noexcept -> void;
    auto wait(size_t value) noexcept -> void;
    auto get_handle() noexcept -> void*;
  };

  struct MemoryBarrier {};

  struct BufferMemoryBarrierDescriptor {
    // buffer memory barrier mask
    Buffer* buffer;
    Flags<AccessFlagEnum> srcAccessMask;
    Flags<AccessFlagEnum> dstAccessMask;
    uint64_t offset = 0;
    uint64_t size = uint64_t(-1); // default: the whole buffer
    // only if queue transition is need
    Queue* srcQueue = nullptr;
    Queue* dstQueue = nullptr;
  };

  struct TextureMemoryBarrierDescriptor {
    // specify image object
    Texture* texture;
    TextureRange subresourceRange;
    // memory barrier mask
    Flags<AccessFlagEnum> srcAccessMask;
    Flags<AccessFlagEnum> dstAccessMask;
    // only if layout transition is need
    TextureLayoutEnum oldLayout;
    TextureLayoutEnum newLayout;
    // only if queue transition is need
    Queue* srcQueue = nullptr;
    Queue* dstQueue = nullptr;
  };

  struct BarrierDescriptor {
    // Necessary (Execution Barrier)
    Flags<PipelineStageEnum> srcStageMask = 0;
    Flags<PipelineStageEnum> dstStageMask = 0;
    Flags<DependencyTypeEnum> dependencyType = 0;
    // Optional (Memory Barriers)
    std::vector<MemoryBarrier*> memoryBarriers;
    std::vector<BufferMemoryBarrierDescriptor> bufferMemoryBarriers;
    std::vector<TextureMemoryBarrierDescriptor> textureMemoryBarriers;
  };

  struct QuerySet {
    QueryType m_type;
    uint32_t m_count;
    VkQueryPool m_queryPool;
    Device* m_device = nullptr;

    QuerySet() = default;
    QuerySet(Device* device, QuerySetDescriptor const& desc);
    ~QuerySet();
    /** fetch the query pooll result */
    auto resolve_query_result(uint32_t firstQuery, uint32_t queryCount,
      size_t dataSize, void* pData, uint64_t stride,
      Flags<QueryResultEnum> flag) noexcept -> void;
  };
  
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃                                                                           ┃
  // ┃ Definitions for pass and pipeline objects.                                ┃
  // ┃                                                                           ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  struct RenderPass {
    /** vulkan render pass */
    VkRenderPass m_renderPass;
    /** vulkan render pass clear value */
    std::vector<VkClearValue> m_clearValues;
    /** the device this render pass is created on */
    Device* m_device = nullptr;

    /** render pass initialize */
    RenderPass(Device* device, RenderPassDescriptor const& desc);
    ~RenderPass();
    RenderPass(RenderPass const& pass) = delete;
    RenderPass(RenderPass&& pass);
    auto operator=(RenderPass const& pass)->RenderPass & = delete;
    auto operator=(RenderPass&& pass)->RenderPass&;
  };

  struct RenderPipeline {
    /** vulkan render pipeline fixed function settings */
    struct RenderPipelineFixedFunctionSettings {
      // shader stages
      std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {};
      // dynamic state
      VkPipelineDynamicStateCreateInfo dynamicState = {};
      std::vector<VkDynamicState> dynamicStates = {};
      // vertex layout
      VkPipelineVertexInputStateCreateInfo vertexInputState = {};
      std::vector<VkVertexInputBindingDescription> vertexBindingDescriptor = {};
      std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions =
      {};
      // input assembly
      VkPipelineInputAssemblyStateCreateInfo assemblyState = {};
      // viewport settings
      VkViewport viewport = {};
      VkRect2D scissor = {};
      VkPipelineViewportStateCreateInfo viewportState = {};
      // multisample
      VkPipelineMultisampleStateCreateInfo multisampleState = {};
      VkPipelineRasterizationStateCreateInfo rasterizationState = {};
      VkPipelineRasterizationConservativeStateCreateInfoEXT
        conservativeRasterizationState = {};
      VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
      std::vector<VkPipelineColorBlendAttachmentState>
        colorBlendAttachmentStates = {};
      VkPipelineColorBlendStateCreateInfo colorBlendState = {};
      PipelineLayout* pipelineLayout = {};
    } m_fixedFunctionSetttings;
    /** vulkan render pipeline */
    VkPipeline m_pipeline = {};
    /** the reusable create information of the pipeline */
    VkGraphicsPipelineCreateInfo m_pipelineInfo{};
    /** the device this render pipeline is created on */
    Device* m_device = nullptr;

    /** constructor */
    RenderPipeline(Device* device, RenderPipelineDescriptor const& desc);
    /** virtual destructor */
    ~RenderPipeline();
    /** copy functions */
    RenderPipeline(RenderPipeline const& pipeline) = delete;
    RenderPipeline(RenderPipeline&& pipeline);
    auto operator=(RenderPipeline const& pipeline)
      ->RenderPipeline & = delete;
    auto operator=(RenderPipeline&& pipeline)->RenderPipeline&;
    /** set debug name */
    auto set_name(std::string const& name) -> void;
    /** combine the pipelien with a render pass and then re-valid it */
    auto combine_render_pass(RenderPass* renderpass) noexcept -> void;
  };

  struct ComputePipeline {
    /** vulkan compute pipeline */
    VkPipeline m_pipeline;
    /** compute pipeline layout */
    PipelineLayout* m_layout = nullptr;
    /** the device this compute pipeline is created on */
    Device* m_device = nullptr;

    ComputePipeline(Device* device, ComputePipelineDescriptor const& desc);
    /** virtual destructor */
    ~ComputePipeline();
    /** set debug name */
    auto set_name(std::string const& name) -> void;
  };

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ CUDA / Torch interoperation                                               ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  //struct CUDADevice {
  //  Device* m_vkDevice;

  //  auto torch_to_cuda(torch::Tensor& tensor) noexcept -> void;

  //};

  //struct CUDABuffer {
  //  cudaExternalMemory_t m_cudaMem;
  //  float* m_dataPtr;
  //  bool m_isExternal = false;

  //};
}
}