#pragma once

#include <se.core.hpp>
#include <se.math.hpp>
#include <future>
#include <optional>
#include <variant>

namespace se::rhi {
struct Context;
struct Adapter;
struct AdapterDescriptor;
struct Device;
struct Queue;
struct Buffer;
struct BufferDescriptor;
struct Texture;
struct TextureDescriptor;
struct TextureView;
struct TextureViewDescriptor;
struct ExternalTexture;
struct ExternalTextureDescriptor;
struct Sampler;
struct SamplerDescriptor;
struct SwapChain;
struct SwapChainDescriptor;
struct ShaderModule;
struct ShaderModuleDescriptor;
struct ComputePipeline;
struct ComputePipelineDescriptor;
struct RenderPipeline;
struct RenderPipelineDescriptor;
struct PipelineLayout;
struct PipelineLayoutDescriptor;
struct BindGroup;
struct BindGroupDescriptor;
struct BindGroupLayout;
struct BindGroupLayoutDescriptor;
struct CommandBuffer;
struct MultiFrameFlights;
struct MultiFrameFlightsDescriptor;
struct CommandEncoder;
struct CommandEncoderDescriptor;
struct RenderBundleEncoder;
struct RenderBundleEncoderDescriptor;
struct QuerySet;
struct QuerySetDescriptor;
struct Fence;
struct Barrier;
struct BarrierDescriptor;
struct MemoryBarrier;
struct MemoryBarrierDescriptor;
struct Semaphore;
struct BLAS;
struct BLASDescriptor;
struct TLAS;
struct TLASDescriptor;
struct RayTracingPipeline;
struct RayTracingPipelineDescriptor;
struct RayTracingPassEncoder;
struct RayTracingPassDescriptor;
struct RayTracingExtension;
struct AdapterInfo;
struct CUDAContext;

/** Context Extensions for extending API capability */
using ContextExtensions = uint32_t;
/** Context Extensions for extending API capability */
enum struct ContextExtensionBit {
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

/** pipeline stage enums */
using PipelineStages = uint32_t;
/** pipeline stage enums */
enum class PipelineStageBit : uint32_t {
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

/** Determine how a GPUBuffer may be used after its creation. */
using BufferUsages = uint32_t;
/** Determine how a GPUBuffer may be used after its creation. */
enum struct BufferUsageBit {
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

/** Determine how a Texture may be used after its creation. */
using ShaderStages = uint32_t;
/** Determine how a Texture may be used after its creation. */
enum struct ShaderStageBit {
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

enum struct TextureLayout : uint32_t {
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

// Forward definition
// ===========================================================================
// Initialization Interface

struct SIByL_API AdapterDescriptor {
  enum struct PowerPreference { LOW_POWER, HIGH_PERFORMANCE, };
  PowerPreference powerPerference = PowerPreference::HIGH_PERFORMANCE;
  bool forceFallbackAdapter = false;
};

/** Context Interface for multiple-Graphics-API */
struct SIByL_API Context {
  /** create the context by backend */
  enum struct Backend { Vulkan, };
  static auto create(Backend backend) noexcept -> std::unique_ptr<Context>;
  /** virtual destructor */
  virtual ~Context() = default;
  /** Initialize the context */
  virtual auto init(se::window* window = nullptr,
    ContextExtensions ext = 0) noexcept -> bool = 0;
  /** Request an adapter */
  virtual auto requestAdapter(
    AdapterDescriptor const& options = AdapterDescriptor{}) noexcept
    -> std::unique_ptr<Adapter> = 0;
  /** Get the binded window */
  virtual auto getBindedWindow() const noexcept -> se::window* = 0;
  /** clean up context resources */
  virtual auto destroy() noexcept -> void = 0;
};

/** Describes the physical properties of a given GPU. */
struct SIByL_API Adapter {
  /** information of adapter */
  struct SIByL_API AdapterInfo {
    std::string vendor;
    std::string architecture;
    std::string device;
    std::string description;
    float timestampPeriod;
  };
  /** virtual destructor */
  virtual ~Adapter() = default;
  /** Requests a device from the adapter. */
  virtual auto requestDevice() noexcept -> std::unique_ptr<Device> = 0;
  /** Requests the AdapterInfo for this Adapter. */
  virtual auto requestAdapterInfo() const noexcept -> AdapterInfo = 0;
  /** get the context the context created from */
  virtual auto fromContext() noexcept -> Context* = 0;
};

/** Device is a logical instantiation of an adapter, through which internal
 * objects are created. Is the exclusive owner of all internal objects created from it.  */
struct SIByL_API Device {
  /** virtual destructor */
  virtual ~Device() = default;
  /** destroy the device */
  virtual auto destroy() noexcept -> void = 0;
  /** wait until idle */
  virtual auto waitIdle() noexcept -> void = 0;
  /** get the adapter the device created from */
  virtual auto fromAdapter() noexcept -> Adapter* = 0;
  // Read-only fields
  // ---------------------------
  /** the graphics queue for this device */
  virtual auto getGraphicsQueue() noexcept -> Queue* = 0;
  /** the compute queue for this device */
  virtual auto getComputeQueue() noexcept -> Queue* = 0;
  /** the present queue for this device */
  virtual auto getPresentQueue() noexcept -> Queue* = 0;
  // Create resources on device
  // ---------------------------
  /** create a buffer on the device */
  virtual auto createBuffer(BufferDescriptor const& desc) noexcept
      -> std::unique_ptr<Buffer> = 0;
  /** create a texture on the device */
  virtual auto createTexture(TextureDescriptor const& desc) noexcept
      -> std::unique_ptr<Texture> = 0;
  /** create a sampler on the device */
  virtual auto createSampler(SamplerDescriptor const& desc) noexcept
      -> std::unique_ptr<Sampler> = 0;
  /* create a swapchain on the device */
  virtual auto createSwapChain(SwapChainDescriptor const& desc) noexcept
      -> std::unique_ptr<SwapChain> = 0;
  // Create resources binding objects
  // ---------------------------
  /** create a bind group layout on the device */
  virtual auto createBindGroupLayout(
      BindGroupLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroupLayout> = 0;
  /** create a pipeline layout on the device */
  virtual auto createPipelineLayout(
      PipelineLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<PipelineLayout> = 0;
  /** create a bind group on the device */
  virtual auto createBindGroup(BindGroupDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroup> = 0;
   //Create pipeline objects
   //---------------------------
  /** create a shader module on the device */
  virtual auto createShaderModule(ShaderModuleDescriptor const& desc) noexcept
      -> std::unique_ptr<ShaderModule> = 0;
  /** create a compute pipeline on the device */
  virtual auto createComputePipeline(
      ComputePipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePipeline> = 0;
  /** create a render pipeline on the device */
  virtual auto createRenderPipeline(
      RenderPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPipeline> = 0;
  // Create command encoders
  // ---------------------------
  /** create a multi frame flights */
  virtual auto createMultiFrameFlights(
      MultiFrameFlightsDescriptor const& desc) noexcept
      -> std::unique_ptr<MultiFrameFlights> = 0;
  /** create a command encoder */
  virtual auto createCommandEncoder(
      CommandEncoderDescriptor const& desc) noexcept
      -> std::unique_ptr<CommandEncoder> = 0;
  // Create query sets
  // ---------------------------
  virtual auto createQuerySet(QuerySetDescriptor const& desc) noexcept
      -> std::unique_ptr<QuerySet> = 0;
  // Create ray tracing objects
  // ---------------------------
  /** create a BLAS */
  virtual auto createBLAS(BLASDescriptor const& desc) noexcept
      -> std::unique_ptr<BLAS> = 0;
  /** create a TLAS */
  virtual auto createTLAS(TLASDescriptor const& desc) noexcept
      -> std::unique_ptr<TLAS> = 0;
  /** create a ray tracing pipeline on the device */
  virtual auto createRayTracingPipeline(
      RayTracingPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RayTracingPipeline> = 0;
  // Create memory barrier objects
  // ---------------------------
  virtual auto createFence() noexcept -> std::unique_ptr<Fence> = 0;
  // Get extensions
  // ---------------------------
  /** fetch a ray tracing extension is available */
  virtual auto getRayTracingExtension() noexcept -> RayTracingExtension* = 0;
  // Create utilities
  // ---------------------------
  /** create a device local buffer with initialzie value */
  auto createDeviceLocalBuffer(void const* data, uint32_t size,
    BufferUsages usage) noexcept -> std::unique_ptr<Buffer>;
  /** read back device local buffer */
  auto readbackDeviceLocalBuffer(Buffer* buffer, void* data,
                                 uint32_t size) noexcept -> void;
  /** write back device local buffer */
  auto writebackDeviceLocalBuffer(Buffer* buffer, void* data,
                                  uint32_t size) noexcept -> void;
  /** read back device local texture */
  auto readbackDeviceLocalTexture(Texture* texture, void* data,
                                  uint32_t size) noexcept -> void;
  /** copy a buffer to another buffer */
  auto copyBufferToBuffer(Buffer* src_buffer, size_t src_offset,
    Buffer* tgt_buffer, size_t tgt_offset, size_t size) noexcept -> void;
  /** transition the layout of an texture */
  auto trainsitionTextureLayout(Texture* texture, TextureLayout oldLayout,
    TextureLayout newLayout) noexcept -> void;
  // Other extensions methods
  // ---------------------------
  /** create CUDA context extension */
  virtual auto queryUUID() noexcept -> std::array<uint64_t, 2> = 0;
};

// Initialization Interface
// ===========================================================================
// Buffers Interface

/** An object that holds a pointer (which can be null) to a buffer of a fixed
 * number of bytes */
using ArrayBuffer = void*;

/** Determine the memory properties. */
using MemoryProperties = uint32_t;
/** Determine the memory properties. */
enum class MemoryPropertyBit {
  DEVICE_LOCAL_BIT = 1 << 0,
  HOST_VISIBLE_BIT = 1 << 1,
  HOST_COHERENT_BIT = 1 << 2,
  HOST_CACHED_BIT = 1 << 3,
  LAZILY_ALLOCATED_BIT = 1 << 4,
  PROTECTED_BIT = 1 << 5,
  FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};

/** Current map state of the buffer */
enum struct BufferMapState { UNMAPPED, PENDING, MAPPED };
/** Shader mode of the buffer */
enum struct BufferShareMode { CONCURRENT, EXCLUSIVE, };

/** Determine how a Buffer is mapped when calling mapAsync(). */
using MapModeFlags = uint32_t;
/** Determine how a Buffer is mapped when calling mapAsync(). */
enum struct MapMode { READ = 1, WRITE = 2, };

/** A block of memory that can be used in GPU operations.
 * Data is stored in linear layout, and some can be mapped. */
struct SIByL_API Buffer {
  /** virtual destructor */
  virtual ~Buffer() = default;
  // Readonly Attributes
  // ---------------------------
  /** readonly get buffer size on GPU */
  virtual auto size() const noexcept -> size_t = 0;
  /** readonly get buffer usage flags on GPU */
  virtual auto bufferUsageFlags() const noexcept -> BufferUsages = 0;
  /** readonly get map state on GPU */
  virtual auto bufferMapState() const noexcept -> BufferMapState = 0;
  /** readonly get device */
  virtual auto getDevice() const noexcept -> Device* = 0;
  /** get the device memory address is applicable */
  virtual auto getDeviceAddress() const noexcept -> uint64_t = 0;
  /** get the memory handle of the allocated buufer */
  struct ExternalHandle { void* handle; size_t offset; size_t size; };
  virtual auto getMemHandle() const noexcept -> ExternalHandle = 0;
  // Map methods
  // ---------------------------
  /** Maps the given range of the GPUBuffer */
  virtual auto mapAsync(MapModeFlags mode, size_t offset = 0,
                        size_t size = 0) noexcept -> std::future<bool> = 0;
  /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
   * mapped range */
  virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept
      -> ArrayBuffer = 0;
  /** Unmaps the mapped range of the GPUBuffer and makes it’s contents available
   * for use by the GPU again. */
  virtual auto unmap() noexcept -> void = 0;
  // Lifecycle methods
  // ---------------------------
  /** destroy the buffer */
  virtual auto destroy() noexcept -> void = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get debug name */
  virtual auto getName() const noexcept -> std::string const& = 0;
};

struct SIByL_API BufferDescriptor {
  BufferDescriptor() = default;
  BufferDescriptor(size_t size, BufferUsages usage, BufferShareMode shareMode = BufferShareMode::EXCLUSIVE,
    MemoryProperties memoryProperties = 0, bool mappedAtCreation = false, int minimumAlignment = -1)
    :size(size), usage(usage), shareMode(shareMode), memoryProperties(memoryProperties),
    mappedAtCreation(mappedAtCreation), minimumAlignment(minimumAlignment) {}
  /** The size of the buffer in bytes. */
  size_t size;
  /** The allowed usages for the buffer. */
  BufferUsages usage;
  /** The queue access share mode of the buffer. */
  BufferShareMode shareMode = BufferShareMode::EXCLUSIVE;
  /** The memory properties of the buffer. */
  MemoryProperties memoryProperties = 0;
  /** If true creates the buffer in an already mapped state. */
  bool mappedAtCreation = false;
  /** buffer alignment (-1 means dont care) */
  int minimumAlignment = -1;
};

// Buffers Interface
// ===========================================================================
// Textures/TextureViews Interface

enum struct TextureDimension { TEX1D, TEX2D, TEX3D };
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

auto SIByL_API hasDepthBit(TextureFormat format) noexcept -> bool;
auto SIByL_API hasStencilBit(TextureFormat format) noexcept -> bool;

/** Determine how a texture is used in command. */
using TextureAspects = uint32_t;
/** Determine how a texture is used in command. */
enum struct TextureAspectBit {
  COLOR_BIT = 1 << 0,
  STENCIL_BIT = 1 << 1,
  DEPTH_BIT = 1 << 2, };

/** Determine how a Texture may be used after its creation. */
using TextureUsages = uint32_t;
/** Determine how a Texture may be used after its creation. */
enum struct TextureUsageBit {
  COPY_SRC = 1 << 0,
  COPY_DST = 1 << 1,
  TEXTURE_BINDING = 1 << 2,
  STORAGE_BINDING = 1 << 3,
  COLOR_ATTACHMENT = 1 << 4,
  DEPTH_ATTACHMENT = 1 << 5,
  TRANSIENT_ATTACHMENT = 1 << 6,
  INPUT_ATTACHMENT = 1 << 7,
};

struct SIByL_API Extend3D {
  uint32_t width;
  uint32_t height;
  uint32_t depthOrArrayLayers;
};

struct SIByL_API Texture {
  // Texture Behaviors
  // ---------------------------
  /** virtual descructor */
  Texture() noexcept = default;
  Texture(Texture&&) noexcept = default;
  virtual ~Texture() = default;
  /** create texture view of this texture */
  virtual auto createView(TextureViewDescriptor const& desc) noexcept
      -> std::unique_ptr<TextureView> = 0;
  /** destroy this texture */
  virtual auto destroy() noexcept -> void = 0;
  // Readonly Attributes
  // ---------------------------
  /** readonly width of the texture */
  virtual auto width() const noexcept -> uint32_t = 0;
  /** readonly height of the texture */
  virtual auto height() const noexcept -> uint32_t = 0;
  /** readonly depth or arrayLayers of the texture */
  virtual auto depthOrArrayLayers() const noexcept -> uint32_t = 0;
  /** readonly mip level count of the texture */
  virtual auto mipLevelCount() const noexcept -> uint32_t = 0;
  /** readonly sample count of the texture */
  virtual auto sampleCount() const noexcept -> uint32_t = 0;
  /** the dimension of the set of texel for each of this GPUTexture's
   * subresources. */
  virtual auto dimension() const noexcept -> TextureDimension = 0;
  /** readonly format of the texture */
  virtual auto format() const noexcept -> TextureFormat = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get name */
  virtual auto getName() -> std::string const& = 0;
  /** get texture descriptor */
  virtual auto getDescriptor() -> TextureDescriptor = 0;
  // Map methods
  // ---------------------------
  /** Maps the given range of the GPUBuffer */
  virtual auto mapAsync(MapModeFlags mode, size_t offset = 0,
                        size_t size = 0) noexcept -> std::future<bool> = 0;
  /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
   * mapped range */
  virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept
      -> ArrayBuffer = 0;
  /** Unmaps the mapped range of the GPUBuffer and makes it’s contents available
   * for use by the GPU again. */
  virtual auto unmap() noexcept -> void = 0;
};

using TextureFlags = uint32_t;
enum struct TextureFlagBit {
  NONE = 0,
  HOSTI_VISIBLE = 1 << 0,
  CUBE_COMPATIBLE = 1 << 1,
};

struct SIByL_API TextureDescriptor {
  Extend3D size;
  uint32_t mipLevelCount = 1;
  uint32_t arrayLayerCount = 1;
  uint32_t sampleCount = 1;
  TextureDimension dimension = TextureDimension::TEX2D;
  TextureFormat format;
  /** The allowed usages for the texture. */
  TextureUsages usage;
  /** Specifies what view format values will be allowed when calling createView()
   * on this texture (in addition to the texture’s actual format). */
  std::vector<TextureFormat> viewFormats;
  TextureFlags flags = (TextureFlags)TextureFlagBit::NONE;
};

enum struct TextureViewDimension {
  TEX1D, TEX1D_ARRAY,
  TEX2D, TEX2D_ARRAY,
  CUBE,  CUBE_ARRAY,
  TEX3D, TEX3D_ARRAY,
};

struct SIByL_API ImageSubresourceRange {
  TextureAspects aspectMask;
  uint32_t baseMipLevel;
  uint32_t levelCount;
  uint32_t baseArrayLayer;
  uint32_t layerCount;
};

struct SIByL_API TextureClearDescriptor {
  std::vector<ImageSubresourceRange> subresources;
  se::vec4 clearColor;
};

/* get aspect from texture format */
auto SIByL_API getTextureAspect(TextureFormat format) noexcept -> TextureAspects;

struct SIByL_API TextureView {
  /** virtual destructor */
  TextureView() noexcept = default;
  TextureView(TextureView&&) noexcept = default;
  virtual ~TextureView() = default;
  /** get binded texture */
  virtual auto getTexture() noexcept -> Texture* = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get width */
  virtual auto getWidth() noexcept -> uint32_t = 0;
  /** get height */
  virtual auto getHeight() noexcept -> uint32_t = 0;
};

enum struct TextureViewType : uint32_t {
  SRV,
  UAV,
  RTV,
  DSV,
};

struct SIByL_API TextureViewDescriptor {
  TextureFormat format;
  TextureViewDimension dimension = TextureViewDimension::TEX2D;
  TextureAspects aspect = (TextureAspects)TextureAspectBit::COLOR_BIT;
  uint32_t baseMipLevel = 0;
  uint32_t mipLevelCount = 1;
  uint32_t baseArrayLayer = 0;
  uint32_t arrayLayerCount = 1;
};

// Textures/TextureViews Interface
// ===========================================================================
// Samplers Interface

struct SIByL_API Sampler {
  /** virtual destructor */
  virtual ~Sampler() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get debug name */
  virtual auto getName() const noexcept -> std::string const& = 0;
};

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

struct SIByL_API SamplerDescriptor {
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

// Samplers Interface
// ===========================================================================
// SwapChain Interface

struct SIByL_API SwapChain {
  /** virtual destructor */
  virtual ~SwapChain() = default;
  /** get texture */
  virtual auto getTexture(int i) noexcept -> Texture* = 0;
  /** get texture view */
  virtual auto getTextureView(int i) noexcept -> TextureView* = 0;
  /** invalid swapchain */
  virtual auto recreate() noexcept -> void = 0;
};

struct SIByL_API SwapChainDescriptor {};

struct SIByL_API MultiFrameFlights {
  /** virtual destructor */
  virtual ~MultiFrameFlights() = default;
  /** start frame */
  virtual auto frameStart() noexcept -> void = 0;
  /** end frame */
  virtual auto frameEnd() noexcept -> void = 0;
  /** get current flight id */
  virtual auto getFlightIndex() noexcept -> uint32_t = 0;
  /** get current swapchain id */
  virtual auto getSwapchainIndex() noexcept -> uint32_t = 0;
  /** get current command buffer */
  virtual auto getCommandBuffer() noexcept -> CommandBuffer* = 0;
  /** get current Image Available Semaphore */
  virtual auto getImageAvailableSeamaphore() noexcept -> Semaphore* = 0;
  /** get current Render Finished Semaphore */
  virtual auto getRenderFinishedSeamaphore() noexcept -> Semaphore* = 0;
  /** get current fence */
  virtual auto getFence() noexcept -> Fence* = 0;
};

struct SIByL_API MultiFrameFlightsDescriptor {
  int maxFlightNum = 1;
  SwapChain* swapchain = nullptr;
};

// SwapChain Interface
// ===========================================================================
// Shader Modules Interface

/** An internal shader module object */
struct SIByL_API ShaderModule {
  /** virtual destrucor */
  virtual ~ShaderModule() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get debug name */
  virtual auto getName() -> std::string const& = 0;
};

struct SIByL_API ShaderModuleDescriptor {
  /** The shader source code for the shader module. */
  se::buffer* code;
  /** stage */
  ShaderStageBit stage;
  /** name of entry point */
  std::string name = "main";
};

// Shader Modules Interface
// ===========================================================================
// Resource Binding Interface

/** Defines the interface between a set of resources bound
 * in a BindGroup and their accessibility in shader stages. */
struct SIByL_API BindGroupLayout {
  /** virtual destructor */
  virtual ~BindGroupLayout() = default;
  /** get BindGroup Layout Descriptor */
  virtual auto getBindGroupLayoutDescriptor() const noexcept
    -> BindGroupLayoutDescriptor const& = 0;
};

enum struct BindingResourceType {
  SAMPLER,
  TEXTURE_VIEW,
  BUFFER_BINDING,
  BINDLESS_TEXTURE,
};

// Buffer resource binding information
enum struct BufferBindingType { UNIFORM, STORAGE, READ_ONLY_STORAGE };
struct SIByL_API BufferBindingLayout {
  BufferBindingType type = BufferBindingType::UNIFORM;
  bool hasDynamicOffset = false;
  size_t minBindingSize = 0;
};

// Sampler resource binding information
enum struct SamplerBindingType { FILTERING, NON_FILTERING, COMPARISON, };
struct SIByL_API SamplerBindingLayout {
  SamplerBindingType type = SamplerBindingType::FILTERING; };
// Texture resource binding information
struct SIByL_API TextureBindingLayout {
  TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
  bool multisampled = false; };
struct SIByL_API StorageTextureBindingLayout {
  TextureFormat format;
  TextureViewDimension viewDimension = TextureViewDimension::TEX2D; };
struct SIByL_API BindlessTexturesBindingLayout {};
// Acceleration structure resource binding information
struct SIByL_API AccelerationStructureBindingLayout {};

/**
 * Describes a single shader resource binding
 * to be included in a GPUBindGroupLayout.  */
struct SIByL_API BindGroupLayoutEntry {
  /** create an entry with a buffer resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility, BufferBindingLayout const& buffer)
    : binding(binding), visibility(visibility), buffer(buffer) {}
  /** create an entry with a sampler resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility, SamplerBindingLayout const& sampler)
    : binding(binding), visibility(visibility), sampler(sampler) {}
  /** create an entry with a texture resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility, TextureBindingLayout const& texture)
      : binding(binding), visibility(visibility), texture(texture) {}
  /** create an entry with a texture-sampler combination resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility,
    TextureBindingLayout const& texture, SamplerBindingLayout const& sampler)
    : binding(binding), visibility(visibility), texture(texture), sampler(sampler) {}
  /** create an entry with a storage texture resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility, StorageTextureBindingLayout const& storageTexture)
    : binding(binding), visibility(visibility), storageTexture(storageTexture) {}
  /** create an entry with a acceleration structure resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility,
    AccelerationStructureBindingLayout const& accelerationStructure)
    : binding(binding), visibility(visibility), accelerationStructure(accelerationStructure) {}
  /** create an entry with a bindless texture array resource */
  BindGroupLayoutEntry(uint32_t binding, ShaderStages visibility, BindlessTexturesBindingLayout const& bindlessTextures)
    : binding(binding), visibility(visibility), bindlessTextures(bindlessTextures) {}
  /* A unique identifier for a resource binding within the BindGroupLayout */
  uint32_t binding;
  /* Array of elements binded on the binding position */
  uint32_t array_size = 1;
  /** indicates whether it will be accessible from the associated shader stage */
  ShaderStages visibility;
  // possible resources layout types
  // ---------------------------
  std::optional<BufferBindingLayout> buffer;
  std::optional<SamplerBindingLayout> sampler;
  std::optional<TextureBindingLayout> texture;
  std::optional<StorageTextureBindingLayout> storageTexture;
  std::optional<AccelerationStructureBindingLayout> accelerationStructure;
  std::optional<BindlessTexturesBindingLayout> bindlessTextures;
};

struct SIByL_API BindGroupLayoutDescriptor {
  std::vector<BindGroupLayoutEntry> entries;
};

/**
 * Defines a set of resources to be bound together in a group
 * and how the resources are used in shader stages.
 */
struct BindGroupEntry;
struct SIByL_API BindGroup {
  virtual ~BindGroup() = default;
  /** update binding */
  virtual auto updateBinding(std::vector<BindGroupEntry> const& entries) noexcept -> void = 0;
};
// binding of ta buffer resource
struct SIByL_API BufferBinding { Buffer* buffer; size_t offset; size_t size; };

struct SIByL_API BindingResource {
  BindingResource() = default;
  BindingResource(TextureView* view, Sampler* sampler)
    : type(BindingResourceType::SAMPLER), textureView(view), sampler(sampler) {}
  BindingResource(Sampler* sampler)
    : type(BindingResourceType::SAMPLER), sampler(sampler){}
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
  BindingResourceType type;
  Sampler* sampler = nullptr;
  TextureView* textureView = nullptr;
  std::vector<Sampler*> samplers = {};
  std::vector<TextureView*> bindlessTextures = {};
  std::vector<TextureView*> storageArray = {};
  ExternalTexture* externalTexture = nullptr;
  std::optional<BufferBinding> bufferBinding;
  TLAS* tlas = nullptr;
};

struct SIByL_API BindGroupEntry { uint32_t binding; BindingResource resource; };

struct SIByL_API BindGroupDescriptor {
  BindGroupLayout* layout;
  std::vector<BindGroupEntry> entries;
};

/**
 * Defines the mapping between resources of all BindGroup objects set up
 * during command encoding in setBindGroup(), and the shaders of the pipeline
 * set by RenderCommandsMixin.setPipeline or ComputePassEncoder.setPipeline.
 */
struct SIByL_API PipelineLayout {
  /** virtual destructor */
  virtual ~PipelineLayout() = default;
};

struct SIByL_API PushConstantEntry {
  ShaderStages shaderStages;
  uint32_t offset;
  uint32_t size;
};

struct SIByL_API PipelineLayoutDescriptor {
  std::vector<PushConstantEntry> pushConstants;
  std::vector<BindGroupLayout*> bindGroupLayouts;
};

// Resource Binding Interface
// ===========================================================================
// Pipelines Interface

struct SIByL_API PipelineBase {
  /** Gets a BindGroupLayout that is compatible with the PipelineBase's
   * BindGroupLayout at index.*/
  auto getBindGroupLayout(size_t index) noexcept -> BindGroupLayout&;
};

using PipelineConstantValue = double;

/** Describes the entry point in the user-provided ShaderModule that
 * controls one of the programmable stages of a pipeline. */
struct SIByL_API ProgrammableStage {
  ShaderModule* module;
  std::string entryPoint;
};

struct SIByL_API PipelineDescriptorBase {
  /** The definition of the layout of resources which can be used with this */
  PipelineLayout* layout = nullptr;
};

// Pipelines Interface
// ===========================================================================
// Rasterizer - Pipelines Interface

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

using SampleMask = uint32_t;

struct MultisampleState {
  uint32_t count = 1;
  SampleMask mask = 0xFFFFFFFF;
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

using ColorWriteFlags = uint32_t;

struct ColorTargetState {
  TextureFormat format;
  BlendState blend;
  ColorWriteFlags writeMask = 0xF;
};

struct FragmentState : public ProgrammableStage {
  std::vector<ColorTargetState> targets;
};

using StencilValue = uint32_t;
using DepthBias = int32_t;

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

enum struct DataFormat {
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

using VertexFormat = DataFormat;

auto SIByL_API getVertexFormatSize(VertexFormat format) noexcept -> size_t;

enum struct VertexStepMode {
  VERTEX,
  INSTANCE,
};

enum struct IndexFormat { UINT16_t, UINT32_T };

struct PrimitiveState {
  PrimitiveTopology topology = PrimitiveTopology::TRIANGLE_LIST;
  IndexFormat stripIndexFormat;
  FrontFace frontFace = FrontFace::CCW;
  CullMode cullMode = CullMode::NONE;
  bool unclippedDepth = false;
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
  StencilValue stencilReadMask = 0xFFFFFFFF;
  StencilValue stencilWriteMask = 0xFFFFFFFF;
  DepthBias depthBias = 0;
  float depthBiasSlopeScale = 0;
  float depthBiasClamp = 0;
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

struct VertexState : public ProgrammableStage {
  std::vector<VertexBufferLayout> buffers;
};

struct SIByL_API RenderPipeline {
  /** virtual destructor */
  virtual ~RenderPipeline() = default;
  /** virtual constructor */
  RenderPipeline() = default;
  RenderPipeline(RenderPipeline&&) = default;
  RenderPipeline(RenderPipeline const&) = delete;
  auto operator=(RenderPipeline&&) -> RenderPipeline& = default;
  auto operator=(RenderPipeline const&) -> RenderPipeline& = delete;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

/** Describes a render pipeline by configuring each of the render stages. */
struct SIByL_API RenderPipelineDescriptor : public PipelineDescriptorBase {
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

// Rasterizer - Pipelines Interface
// ===========================================================================
// Compute - Pipelines Interface

struct SIByL_API ComputePipeline {
  /** virtual destructor */
  virtual ~ComputePipeline() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

struct SIByL_API ComputePipelineDescriptor : public PipelineDescriptorBase {
  /* the compute shader */
  ProgrammableStage compute;
};

// Pipelines Interface
// ===========================================================================
// Command Buffers Interface

/**
 * Pre-recorded lists of GPU commands that can be submitted to a GPUQueue
 * for execution. Each GPU command represents a task to be performed on the
 * GPU, such as setting state, drawing, copying resources, etc.
 *
 * A CommandBuffer can only be submitted once, at which point it becomes
 * invalid. To reuse rendering commands across multiple submissions, use
 * RenderBundle.
 */
struct SIByL_API CommandBuffer {
  virtual ~CommandBuffer() = default;
};

struct SIByL_API CommandBufferDescriptor {};

//// Command Buffers Interface
//// ===========================================================================
//// Command Encoding Interface

struct CommandsMixin;
struct RenderPassEncoder;
struct RenderPassDescriptor;
struct ComputePassEncoder;
struct ComputePassDescriptor;
struct ImageCopyBuffer;
struct ImageCopyTexture;
struct RenderBundle;

/**
 * CommandsMixin defines state common to all interfaces
 * which encode commands. It has no methods.
 */
struct CommandsMixin {};

struct DebugUtilLabelDescriptor {
  std::string name;
  se::vec4 color;
};

struct SIByL_API CommandEncoder {
  /** virtual descructor */
  virtual ~CommandEncoder() = default;
  /** Begins encoding a render pass described by descriptor. */
  virtual auto beginRenderPass(RenderPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPassEncoder> = 0;
  /** Begins encoding a compute pass described by descriptor. */
  virtual auto beginComputePass(ComputePassDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePassEncoder> = 0;
  /** Begins encoding a ray tracing pass described by descriptor. */
  virtual auto beginRayTracingPass(
      RayTracingPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RayTracingPassEncoder> = 0;
  /** Insert a barrier. */
  virtual auto pipelineBarrier(BarrierDescriptor const& desc) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from
   * a sub-region of a GPUBuffer to a sub-region of another Buffer. */
  virtual auto copyBufferToBuffer(Buffer* source, size_t sourceOffset,
                                  Buffer* destination, size_t destinationOffset,
                                  size_t size) noexcept -> void = 0;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * Buffer with zeros. */
  virtual auto clearBuffer(Buffer* buffer, size_t offset, size_t size) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * texture with any color. */
  virtual auto clearTexture(Texture* texture,
                            TextureClearDescriptor const& desc) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * Buffer with a value. */
  virtual auto fillBuffer(Buffer* buffer, size_t offset, size_t size,
                          float fillValue) noexcept -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from a
   * sub-region of a Buffer to a sub-region of one or multiple continuous
   * texture subresources. */
  virtual auto copyBufferToTexture(ImageCopyBuffer const& source,
                                   ImageCopyTexture const& destination,
                                   Extend3D const& copySize) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from a
   * sub-region of one or multiple continuous texture subresourcesto a
   * sub-region of a Buffer. */
  virtual auto copyTextureToBuffer(ImageCopyTexture const& source,
                                   ImageCopyBuffer const& destination,
                                   Extend3D const& copySize) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from
   * a sub-region of one or multiple contiguous texture subresources to
   * another sub-region of one or multiple continuous texture subresources. */
  virtual auto copyTextureToTexture(ImageCopyTexture const& source,
                                    ImageCopyTexture const& destination,
                                    Extend3D const& copySize) noexcept
      -> void = 0;
  /** Reset the queryset. */
  virtual auto resetQuerySet(QuerySet* querySet, uint32_t firstQuery,
                             uint32_t queryCount) noexcept -> void = 0;
  /** Writes a timestamp value into a querySet when all
   * previous commands have completed executing. */
  virtual auto writeTimestamp(QuerySet* querySet, PipelineStages stageMask,
                              uint32_t queryIndex) noexcept -> void = 0;
  /** Resolves query results from a QuerySet out into a range of a Buffer. */
  virtual auto resolveQuerySet(QuerySet* querySet, uint32_t firstQuery,
                               uint32_t queryCount, Buffer& destination,
                               uint64_t destinationOffset) noexcept -> void = 0;
  /** copy accceleration structure to a new instance */
  virtual auto cloneBLAS(BLAS* src) noexcept -> std::unique_ptr<BLAS> = 0;
  /** update blas by refitting, only deformation is allowed */
  virtual auto updateBLAS(BLAS* src, Buffer* vertexBuffer,
                          Buffer* indexBuffer) noexcept -> void = 0;
  /** Completes recording of the commands sequence and returns a corresponding
   * GPUCommandBuffer. */
  virtual auto finish() noexcept -> CommandBuffer* = 0;
  /** begin Debug marker region */
  virtual auto beginDebugUtilsLabelEXT(
      DebugUtilLabelDescriptor const& desc) noexcept -> void = 0;
  /** end Debug marker region */
  virtual auto endDebugUtilsLabelEXT() noexcept -> void = 0;
};

struct SIByL_API CommandEncoderDescriptor {
  CommandBuffer* externalCommandBuffer = nullptr;
};

struct SIByL_API ImageDataLayout {
  uint64_t offset = 0;
  uint32_t bytesPerRow;
  uint32_t rowsPerImage;
};

struct SIByL_API ImageCopyBuffer : ImageDataLayout {
  Buffer* buffer;
};

using IntegerCoordinate = uint32_t;

struct Origin2DDict {
  IntegerCoordinate x = 0;
  IntegerCoordinate y = 0;
};

using Origin2D = std::variant<Origin2DDict, std::vector<IntegerCoordinate>>;

struct Origin3DDict {
  IntegerCoordinate x = 0;
  IntegerCoordinate y = 0;
  IntegerCoordinate z = 0;
};

using Origin3D = Origin3DDict;

struct ImageCopyTexture {
  Texture* texutre;
  uint32_t mipLevel = 0;
  Origin3D origin = {};
  TextureAspects aspect = (TextureAspects)TextureAspectBit::COLOR_BIT;
};

struct ImageCopyTextureTagged : public ImageCopyTexture {
  bool premultipliedAlpha = false;
};

struct ImageCopyExternalImage {
  Origin2D origin = {};
  bool flipY = false;
};

// Command Encoding Interface
// ===========================================================================
// Programmable Passes Interface

using BufferDynamicOffset = uint32_t;

struct SIByL_API BindingCommandMixin {
  /** virtual destructor */
  virtual ~BindingCommandMixin() = default;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
    std::vector<BufferDynamicOffset> const& dynamicOffsets = {}) noexcept -> void = 0;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
    uint64_t dynamicOffsetDataStart, uint32_t dynamicOffsetDataLength) noexcept -> void = 0;
  /** Push constants */
  virtual auto pushConstants(void* data, ShaderStages stages,
    uint32_t offset, uint32_t size) noexcept -> void = 0;
};

// Programmable Passes Interface
// ===========================================================================
// Debug Marks Interface

struct SIByL_API DebugCommandMixin {
  auto pushDebugGroup(std::string const& groupLabel) noexcept -> void;
  auto popDebugGroup() noexcept -> void;
  auto insertDebugMarker(std::string const& markerLabel) noexcept -> void;
};

// Debug Marks Interface
// ===========================================================================
// Compute Passes Interface

struct SIByL_API ComputePassEncoder : public BindingCommandMixin {
  /** virtual descructor */
  virtual ~ComputePassEncoder() = default;
  /** Sets the current GPUComputePipeline. */
  virtual auto setPipeline(ComputePipeline* pipeline) noexcept -> void = 0;
  /** Dispatch work to be performed with the current GPUComputePipeline.*/
  virtual auto dispatchWorkgroups(uint32_t workgroupCountX,
                                  uint32_t workgroupCountY = 1,
                                  uint32_t workgroupCountZ = 1) noexcept
      -> void = 0;
  /** Dispatch work to be performed with the current GPUComputePipeline using
   * parameters read from a GPUBuffer. */
  virtual auto dispatchWorkgroupsIndirect(Buffer* indirectBuffer,
                                          uint64_t indirectOffset) noexcept
      -> void = 0;
  /** Completes recording of the compute pass commands sequence. */
  virtual auto end() noexcept -> void = 0;
};

enum struct ComputePassTimestampLocation {
  BEGINNING,
  END,
};

struct ComputePassTimestampWrite {
  std::unique_ptr<QuerySet> querySet = nullptr;
  uint32_t queryIndex;
  ComputePassTimestampLocation location;
};

using ComputePassTimestampWrites =
    std::vector<ComputePassTimestampWrite>;

struct ComputePassDescriptor {
  ComputePassTimestampWrites timestampWrites = {};
};

// Compute Passes Interface
// ===========================================================================
// Render Passes Interface

struct RenderPassColorAttachment;
struct RenderPassDepthStencilAttachment;

/**
 * RenderCommandsMixin defines rendering commands common to
 * RenderPassEncoder and RenderBundleEncoder.
 */
struct SIByL_API RenderCommandsMixin {
  /** virtual descructor */
  virtual ~RenderCommandsMixin() = default;
  /** Sets the current GPURenderPipeline. */
  virtual auto setPipeline(RenderPipeline* pipeline) noexcept -> void = 0;
  /** Sets the current index buffer. */
  virtual auto setIndexBuffer(Buffer* buffer, IndexFormat indexFormat,
                              uint64_t offset = 0, uint64_t size = 0) noexcept
      -> void = 0;
  /** Sets the current vertex buffer for the given slot. */
  virtual auto setVertexBuffer(uint32_t slot, Buffer* buffer,
                               uint64_t offset = 0, uint64_t size = 0) noexcept
      -> void = 0;
  /** Draws primitives. */
  virtual auto draw(uint32_t vertexCount, uint32_t instanceCount = 1,
                    uint32_t firstVertex = 0,
                    uint32_t firstInstance = 0) noexcept -> void = 0;
  /** Draws indexed primitives. */
  virtual auto drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
                           uint32_t firstIndex = 0, int32_t baseVertex = 0,
                           uint32_t firstInstance = 0) noexcept -> void = 0;
  /** Draws primitives using parameters read from a GPUBuffer. */
  virtual auto drawIndirect(Buffer* indirectBuffer, uint64_t indirectOffset,
                            uint32_t drawCount, uint32_t stride) noexcept
      -> void = 0;
  /** Draws indexed primitives using parameters read from a GPUBuffer. */
  virtual auto drawIndexedIndirect(Buffer* indirectBuffer, uint64_t offset,
                                   uint32_t drawCount, uint32_t stride) noexcept
      -> void = 0;
};

struct SIByL_API RenderPassEncoder : public RenderCommandsMixin,
                                  public BindingCommandMixin {
  /** virtual descructor */
  virtual ~RenderPassEncoder() = default;
  /** Sets the viewport used during the rasterization stage to linearly map
   * from normalized device coordinates to viewport coordinates. */
  virtual auto setViewport(float x, float y, float width, float height,
                           float minDepth, float maxDepth) noexcept -> void = 0;
  /** Sets the scissor rectangle used during the rasterization stage.
   * After transformation into viewport coordinates any fragments
   * which fall outside the scissor rectangle will be discarded. */
  virtual auto setScissorRect(IntegerCoordinate x, IntegerCoordinate y,
                              IntegerCoordinate width,
                              IntegerCoordinate height) noexcept -> void = 0;
  /** Sets the constant blend color and alpha values used with
   * "constant" and "one-minus-constant" GPUBlendFactors. */
  virtual auto setBlendConstant(se::vec4 color) noexcept -> void = 0;
  /** Sets the [[stencil_reference]] value used during
   * stencil tests with the "replace" GPUStencilOperation. */
  virtual auto setStencilReference(StencilValue reference) noexcept -> void = 0;
  /** begin occlusion query */
  virtual auto beginOcclusionQuery(uint32_t queryIndex) noexcept -> void = 0;
  /** end occlusion query */
  virtual auto endOcclusionQuery() noexcept -> void = 0;
  /** Executes the commands previously recorded into the given GPURenderBundles
   * as part of this render pass. */
  virtual auto executeBundles(std::vector<RenderBundle> const& bundles) noexcept
      -> void = 0;
  /** Completes recording of the render pass commands sequence. */
  virtual auto end() noexcept -> void = 0;
};

enum struct RenderPassTimestampLocation {
  BEGINNING,
  END,
};

struct RenderPassTimestampWrite {
  QuerySet* querySet;
  uint32_t queryIndex;
  RenderPassTimestampLocation location;
};

using RenderPassTimestampWrites = std::vector<RenderPassTimestampWrite>;

enum struct LoadOp {
  DONT_CARE,
  LOAD,
  CLEAR,
};

enum struct StoreOp { DONT_CARE, STORE, DISCARD };

struct SIByL_API RenderPassColorAttachment {
  TextureView* view;
  TextureView* resolveTarget = nullptr;
  se::vec4 clearValue;
  LoadOp loadOp;
  StoreOp storeOp;
};

struct SIByL_API RenderPassDepthStencilAttachment {
  TextureView* view = nullptr;
  float depthClearValue = 0;
  LoadOp depthLoadOp;
  StoreOp depthStoreOp;
  bool depthReadOnly = false;
  StencilValue stencilClearValue = 0;
  LoadOp stencilLoadOp;
  StoreOp stencilStoreOp;
  bool stencilReadOnly = false;
};

struct SIByL_API RenderPassDescriptor {
  std::vector<RenderPassColorAttachment> colorAttachments;
  RenderPassDepthStencilAttachment depthStencilAttachment;
  // std::unique_ptr<QuerySet> occlusionQuerySet = nullptr;
  RenderPassTimestampWrites timestampWrites = {};
  uint64_t maxDrawCount = 50000000;
};

struct SIByL_API RenderPassLayout {
  std::vector<TextureFormat> colorFormats;
  TextureFormat depthStencilFormat;
  uint32_t sampleCount = 1;
};

/** Describes the queue object of a given Device. */
struct SIByL_API Queue {
  /** virtual destructor */
  virtual ~Queue() = default;
  /** Schedules the execution of the command buffers by the GPU on this queue. */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void = 0;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   * With sync objects */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers, Fence* fence) noexcept -> void = 0;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   * With sync objects */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers,
    Semaphore* wait, Semaphore* signal, Fence* fence) noexcept -> void = 0;
  /** Returns a Promise that resolves once this queue finishes
   * processing all the work submitted up to this moment. */
  virtual auto onSubmittedWorkDone() noexcept -> std::future<bool> = 0;
  /** Issues a write operation of the provided data into a Buffer. */
  virtual auto writeBuffer(Buffer* buffer, uint64_t bufferOffset,
    ArrayBuffer* data, uint64_t dataOffset, Extend3D const& size) noexcept -> void = 0;
  /** Issues a write operation of the provided data into a Texture. */
  virtual auto writeTexture(ImageCopyTexture const& destination,
    ArrayBuffer* data, ImageDataLayout const& layout, Extend3D const& size) noexcept -> void = 0;
  /** Present swap chain. */
  virtual auto presentSwapChain(SwapChain* swapchain, 
    uint32_t imageIndex, Semaphore* semaphore) noexcept -> void = 0;
  /** wait until idle */
  virtual auto waitIdle() noexcept -> void = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

// Render Passes Interface
// ===========================================================================
// Queries Interface

enum struct QueryType { 
    OCCLUSION, 
    PIPELINE_STATISTICS,
    TIMESTAMP,
};

enum struct QueryResultBits {
  RESULT_64 = 0x1,
  RESULT_WAIT = 0x1 << 1,
  RESULT_WITH_AVAILABILITY = 0x1 << 2,
  RESULT_PARTIAL = 0x1 << 3,
};
using QueryResultFlag = uint32_t;

struct SIByL_API QuerySet {
  virtual ~QuerySet() = default;
  /** fetch the query pooll result */
  virtual auto resolveQueryResult(uint32_t firstQuery, uint32_t queryCount,
    size_t dataSize, void* pData, uint64_t stride, QueryResultFlag flag) noexcept -> void = 0;
  QueryType type;
  uint32_t count;
};

struct QuerySetDescriptor {
  QueryType type;
  uint32_t count;
};

// Queries Interface
// ===========================================================================
// Synchronization Interface

struct SIByL_API Fence {
  /** virtual desctructor */
  virtual ~Fence() = default;
  /* wait the fence */
  virtual auto wait() noexcept -> void = 0;
  /* reset the fence */
  virtual auto reset() noexcept -> void = 0;
};

struct SIByL_API Barrier {
  /** virtual desctructor */
  virtual ~Barrier() = default;
};

struct SIByL_API MemoryBarrier {
  /** virtual desctructor */
  virtual ~MemoryBarrier() = default;
};

struct SIByL_API BufferMemoryBarrier {
  /** virtual desctructor */
  virtual ~BufferMemoryBarrier() = default;
};

struct SIByL_API ImageMemoryBarrier {
  /** virtual desctructor */
  virtual ~ImageMemoryBarrier() = default;
};

using AccessFlags = uint32_t;
enum class AccessFlagBits : uint32_t {
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

struct SIByL_API MemoryBarrierDesc {
  // memory barrier mask
  AccessFlags srcAccessMask;
  AccessFlags dstAccessMask;
};

struct SIByL_API BufferMemoryBarrierDescriptor {
  // buffer memory barrier mask
  Buffer* buffer;
  AccessFlags srcAccessMask;
  AccessFlags dstAccessMask;
  uint64_t offset = 0;
  uint64_t size = uint64_t(-1); // default: the whole buffer
  // only if queue transition is need
  Queue* srcQueue = nullptr;
  Queue* dstQueue = nullptr;
};

struct SIByL_API TextureMemoryBarrierDescriptor {
  // specify image object
  Texture* texture;
  ImageSubresourceRange subresourceRange;
  // memory barrier mask
  AccessFlags srcAccessMask;
  AccessFlags dstAccessMask;
  // only if layout transition is need
  TextureLayout oldLayout;
  TextureLayout newLayout;
  // only if queue transition is need
  Queue* srcQueue = nullptr;
  Queue* dstQueue = nullptr;
};

/** dependency of barriers */
using DependencyTypeFlags = uint32_t;
/** dependency of barriers */
enum class DependencyType : uint32_t {
  NONE = 0 << 0,
  BY_REGION_BIT = 1 << 0,
  VIEW_LOCAL_BIT = 1 << 1,
  DEVICE_GROUP_BIT = 1 << 2,
};

struct SIByL_API BarrierDescriptor {
  // Necessary (Execution Barrier)
  PipelineStages srcStageMask;
  PipelineStages dstStageMask;
  DependencyTypeFlags dependencyType;
  // Optional (Memory Barriers)
  std::vector<MemoryBarrier*> memoryBarriers;
  std::vector<BufferMemoryBarrierDescriptor> bufferMemoryBarriers;
  std::vector<TextureMemoryBarrierDescriptor> textureMemoryBarriers;
};

struct SIByL_API Semaphore {
  virtual ~Semaphore() = default;
};

// Synchronization Interface
// ===========================================================================
// Ray Tracing Interface

struct BLAS {
  /** virtual destructor */
  virtual ~BLAS() = default;
  /** get descriptor */
  virtual auto getDescriptor() noexcept -> BLASDescriptor = 0;
};

enum struct BLASGeometryFlagBits : uint32_t {
  NONE = 0 << 0,
  OPAQUE_GEOMETRY = 1 << 0,
  NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1,
};
using BLASGeometryFlags = uint32_t;

/** Affline transform matrix */
struct SIByL_API AffineTransformMatrix {
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
  float matrix[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
};

struct SIByL_API BLASTriangleGeometry {
  Buffer* positionBuffer = nullptr;
  Buffer* indexBuffer = nullptr;
  IndexFormat indexFormat = IndexFormat::UINT16_t;
  uint32_t maxVertex = 0;
  uint32_t firstVertex = 0;
  uint32_t primitiveCount = 0;
  uint32_t primitiveOffset = 0;
  AffineTransformMatrix transform;
  BLASGeometryFlags geometryFlags = 0;
  uint32_t materialID = 0;
  uint32_t vertexStride = 3 * sizeof(float);
  uint32_t vertexByteOffset = 0;
  enum struct VertexFormat {
    RGB32,
    RG32
  } vertexFormat = VertexFormat::RGB32;
};

struct SIByL_API BLASCustomGeometry {
  AffineTransformMatrix transform;
  std::vector<se::bounds3> aabbs;
  BLASGeometryFlags geometryFlags = 0;
};

struct SIByL_API BLASDescriptor {
  std::vector<BLASTriangleGeometry> triangleGeometries;
  bool allowRefitting = false;
  bool allowCompaction = false;
  std::vector<BLASCustomGeometry> customGeometries;
};

struct SIByL_API TLAS {
  /** virtual destructor */
  virtual ~TLAS() = default;
};

struct SIByL_API BLASInstance {
  BLAS* blas = nullptr;
  se::mat4 transform = {};
  uint32_t instanceCustomIndex = 0;  // is used by system now
  uint32_t instanceShaderBindingTableRecordOffset = 0;
  uint32_t mask = 0xFF;
};

struct SIByL_API TLASDescriptor {
  std::vector<BLASInstance> instances;
  bool allowRefitting = false;
  bool allowCompaction = false;
};

struct SIByL_API RayTracingPipeline {
  /** virtual destructor */
  virtual ~RayTracingPipeline() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

struct SIByL_API RayGenerationShaderBindingTableDescriptor {
  ShaderModule* rayGenShader = nullptr;
};
struct SIByL_API RayMissShaderBindingTableDescriptor {
  ShaderModule* rayMissShader = nullptr;
};
struct SIByL_API RayHitGroupShaderBindingTableDescriptor {
  struct HitGroupDescriptor {
    ShaderModule* closetHitShader = nullptr;
    ShaderModule* anyHitShader = nullptr;
    ShaderModule* intersectionShader = nullptr;
  };
};
struct SIByL_API CallableShaderBindingTableDescriptor {
  ShaderModule* callableShader = nullptr;
};

/**
 * Describe a set of SBTs that are stored consecutively.
 * When looking up a shader, different factors will tell the traversal engine
 * the index of the shader to call. These factors are:
 * - 1. missIndex from the GLSL traceRayEXT call;
 * - 2. The instance's instanceShaderBindingTableRecordOffset from TLAS
 * creation;
 * - 3. sbtRecordOffset from the GLSL traceRayEXT call;
 * - 4. sbtRecordStride from the GLSL traceRayEXT call;
 * - 5. The geometry index geometryIndex of each geometry inside a BLAS.
 */
struct SIByL_API SBTsDescriptor {
  /** @indexing: By default, traceRayEXT always uses the ray generation shader
   * at index 0. Therefore we currently support single record slot for a ray
   * generation SBT. */
  struct RayGenerationSBT {
    /** A ray generation record only has a ray generation shader. */
    struct RayGenerationRecord {
      ShaderModule* rayGenShader = nullptr;
    };
    /** As defaultly 0 is chosen, we only provide one record slot*/
    RayGenerationRecord rgenRecord = {};
  } rgenSBT;
  /** @indexing: When a ray didn't intersect anything, traversal calls
   * the index missIndex miss shader, specified in traceRayEXT call. */
  struct MissSBT {
    /** A ray miss record only has a miss shader. */
    struct MissRecord {
      ShaderModule* missShader = nullptr;
    };
    /** There could be multiple miss shader to be selected from */
    std::vector<MissRecord> rmissRecords = {};
  } missSBT;
  /** @indexing: Traversal calls the corresponding shader from the hit record
   * with index. instanceShaderBindingTableRecordOffset (from TLAS) +
   * sbtRecordOffset (from traceRayEXT call)
   * + sbtRecordStride (from traceRayEXT call)* geometryIndex (from BLAS) */
  struct HitGroupSBT {
    /** A hit group record includes a closest hit shader, an optional any hit
     * shader, and an optional intersection shader (only for procedural hit
     * groups). */
    struct HitGroupRecord {
      ShaderModule* closetHitShader = nullptr;
      ShaderModule* anyHitShader = nullptr;
      ShaderModule* intersectionShader = nullptr;
    };
    /** There could be hit group shader to be selected from */
    std::vector<HitGroupRecord> hitGroupRecords = {};
  } hitGroupSBT;
  struct CallableSBT {
    /** A callable record includes only a callable shader. */
    struct CallableRecord {
      ShaderModule* callableShader = nullptr;
    };
    /** There could be hit group shader to be selected from */
    std::vector<CallableRecord> callableRecords = {};
  } callableSBT;
};

struct SIByL_API RayTracingPipelineDescriptor : public PipelineDescriptorBase {
  uint32_t maxPipelineRayRecursionDepth = 1;
  SBTsDescriptor sbtsDescriptor = {};
};

struct SIByL_API RayTracingExtension {
  virtual ~RayTracingExtension() = default;
};

struct SIByL_API RayTracingPassEncoder : public BindingCommandMixin {
  /** virtual destructor */
  virtual ~RayTracingPassEncoder() = default;
  /** set a ray tracing pipeline as the current pipeline */
  virtual auto setPipeline(RayTracingPipeline* pipeline) noexcept -> void = 0;
  /** trace rays using current ray tracing pipeline */
  virtual auto traceRays(uint32_t width, uint32_t height, uint32_t depth) noexcept -> void = 0;
  /** trace rays using current ray tracing pipeline by an indirect buffer */
  virtual auto traceRaysIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void = 0;
  /**  end the ray tracing pass */
  virtual auto end() noexcept -> void = 0;
};

struct SIByL_API RayTracingPassDescriptor {};

// Ray Tracing Interface
// ===========================================================================
// CUDA Extension Interface

struct CUDABuffer;

struct SIByL_API CUDAContext {
  static auto initialize(std::array<uint64_t, 2> const& uuid) noexcept -> void;
  static auto initialize(Device* device) noexcept -> void;
  static auto synchronize() noexcept -> void;
  static auto toCUDABuffer(Buffer* buffer) noexcept -> std::unique_ptr<CUDABuffer>;
  static auto toCUDABufferInterval(Buffer* buffer, int offset, int size) noexcept -> std::unique_ptr<CUDABuffer>;
  static auto allocCUDABuffer(size_t size)noexcept -> std::unique_ptr<CUDABuffer>;
};

struct SIByL_API CUDABuffer {
  virtual ~CUDABuffer() = default;
  virtual auto ptr() noexcept -> void* = 0;
};

struct SIByL_API CUDASemaphore {
  virtual ~CUDASemaphore() = default;
};

struct SIByL_API CUDAStream {
  virtual ~CUDAStream() = default;
  virtual auto waitSemaphoreAsync(CUDASemaphore* semaphore, size_t waitValue = 0) noexcept -> void = 0;
  virtual auto signalSemaphoreAsync(CUDASemaphore* semaphore, size_t waitValue = 0) noexcept -> void = 0;
};
}