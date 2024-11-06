#pragma once
#include <se.core.hpp>
#include <se.rhi.hpp>
#include <vulkan/vulkan.h>
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>
#include <vulkan/vulkan_win32.h>
#define USE_VMA
#include <vma/vk_mem_alloc.h>
#include <optional>

namespace se::rhi {
struct Queue_VK;
struct Device_VK;
struct CommandPool_VK;
struct CommandBuffer_VK;
struct BindGroupPool_VK;
struct RayTracingExtension_VK;

struct SIByL_API Queue_VK : public Queue {
  /** virtual destructor */
  virtual ~Queue_VK() = default;
  /** Schedules the execution of the command buffers by the GPU on this queue. */
  virtual auto submit(
      std::vector<CommandBuffer*> const& commandBuffers) noexcept
      -> void override;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   * With sync objects */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers,
                      Fence* fence) noexcept -> void override;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   * With sync objects */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers,
                      Semaphore* wait, Semaphore* signal, Fence* fence) noexcept
      -> void override;
  /** Returns a Promise that resolves once this queue finishes
   * processing all the work submitted up to this moment. */
  virtual auto onSubmittedWorkDone() noexcept -> std::future<bool> override;
  /** Issues a write operation of the provided data into a Buffer. */
  virtual auto writeBuffer(Buffer* buffer, uint64_t bufferOffset,
                           ArrayBuffer* data, uint64_t dataOffset,
                           Extend3D const& size) noexcept -> void override;
  /** Issues a write operation of the provided data into a Texture. */
  virtual auto writeTexture(ImageCopyTexture const& destination,
                            ArrayBuffer* data, ImageDataLayout const& layout,
                            Extend3D const& size) noexcept -> void override;
  /** Present swap chain. */
  virtual auto presentSwapChain(SwapChain* swapchain, uint32_t imageIndex,
                                Semaphore* semaphore) noexcept -> void override;
  /** wait until idle */
  virtual auto waitIdle() noexcept -> void override;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** Vulkan queue handle */
  VkQueue queue;
  /* the device this buffer is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API Context_VK final : public Context{
  Context_VK() = default;
  /** virtual destructor */
  virtual ~Context_VK() { destroy(); }
  /** initialize the context */
  virtual auto init(se::window* window = nullptr,
                    ContextExtensions ext = 0) noexcept -> bool override;
  /** Request an adapter */
  virtual auto requestAdapter(
    AdapterDescriptor const& options = {}) noexcept
    -> std::unique_ptr<Adapter> override;
  /** Get the binded window */
  virtual auto getBindedWindow() const noexcept -> se::window* override;
  /** clean up context resources */
  virtual auto destroy() noexcept -> void override;
 public:
  /** get VkInstance */
  auto getVkInstance() noexcept -> VkInstance& { return instance; }
  /** get VkSurfaceKHR */
  auto getVkSurfaceKHR() noexcept -> VkSurfaceKHR& { return surface; }
  /** get DebugMessageFunc */
  auto getDebugMessenger() noexcept -> VkDebugUtilsMessengerEXT& { return debugMessenger; }
  /** get All VkPhysicalDevices available */
  auto getVkPhysicalDevices() noexcept -> std::vector<VkPhysicalDevice>& { return devices; }
  /** get All Device Extensions required */
  auto getDeviceExtensions() noexcept -> std::vector<const char*>& { return deviceExtensions; }
  /** get Context Extensions Flags */
  auto getContextExtensionsFlags() const noexcept -> ContextExtensions { return extensions; }
  // Debug Ext Func Pointers
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
  PFN_vkGetMemoryWin32HandleKHR vkCmdGetMemoryWin32HandleKHR;
 private:
  VkInstance instance;
  VkSurfaceKHR surface;
  VkDebugUtilsMessengerEXT debugMessenger;
  se::window* bindedWindow = nullptr;
  ContextExtensions extensions = 0;
  std::vector<VkPhysicalDevice> devices;
  std::vector<const char*> deviceExtensions = {};
};

struct SIByL_API QueueFamilyIndices_VK {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  std::optional<uint32_t> computeFamily;
  /** check whether queue families are complete */
  auto isComplete() noexcept -> bool {
    return graphicsFamily.has_value() && presentFamily.has_value() &&
           computeFamily.has_value(); }
};

struct SIByL_API Adapter_VK final : public Adapter {
  /** constructor */
  Adapter_VK(VkPhysicalDevice device, Context_VK* context,
             VkPhysicalDeviceProperties const& properties);
  /** Requests a device from the adapter. */
  virtual auto requestDevice() noexcept -> std::unique_ptr<Device> override;
  /** Requests the AdapterInfo for this Adapter. */
  virtual auto requestAdapterInfo() const noexcept -> AdapterInfo override;
  /** get the context the context created from */
  virtual auto fromContext() noexcept -> Context* override;
 public:
  /** get context the adapter is on */
  auto getContext() noexcept -> Context_VK* { return context; }
  /** get VkPhysicalDevice */
  auto getVkPhysicalDevice() noexcept -> VkPhysicalDevice& { return physicalDevice; }
  /** get TimestampPeriod */
  auto getTimestampPeriod() const noexcept -> float { return timestampPeriod; }
  /** get QueueFamilyIndices_VK */
  auto getQueueFamilyIndices() const noexcept -> QueueFamilyIndices_VK const& {
    return queueFamilyIndices; }
  /** get All Device Extensions required */
  auto getDeviceExtensions() noexcept -> std::vector<const char*>& {
    return context->getDeviceExtensions(); }
  /** get All Device Extensions required */
  auto findMemoryType(uint32_t typeFilter,
    VkMemoryPropertyFlags properties) noexcept -> uint32_t;
  /* get VkPhysicalDeviceProperties */
  auto getVkPhysicalDeviceProperties() const noexcept
      -> VkPhysicalDeviceProperties const& { return properties; }
 private:
  /** the context the adapter is requested from */
  Context_VK* context = nullptr;
  /** adapter information */
  AdapterInfo const adapterInfo;
  /** the graphics card selected */
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  /** the timestamp period for timestemp query */
  float timestampPeriod = 0.0f;
  /** QueueFamilyIndices_VK */
  QueueFamilyIndices_VK queueFamilyIndices;
  /** vulkan physical device properties */
  VkPhysicalDeviceProperties properties = {};
};

struct SIByL_API Device_VK final : public Device {
  /** virtual destructor */
  virtual ~Device_VK();
  /** destroy the device */
  virtual auto destroy() noexcept -> void override;
  /** wait until idle */
  virtual auto waitIdle() noexcept -> void override;
  /** get the adapter the device created from */
  virtual auto fromAdapter() noexcept -> Adapter* override;
  // Read-only fields
  // ---------------------------
  /** the graphics queue for this device */
  virtual auto getGraphicsQueue() noexcept -> Queue* override { return &graphicsQueue; }
  /** the compute queue for this device */
  virtual auto getComputeQueue() noexcept -> Queue* override { return &computeQueue; }
  /** the present queue for this device */
  virtual auto getPresentQueue() noexcept -> Queue* override { return &presentQueue; }
  // Create resources on device
  // ---------------------------
  /** create a buffer on the device */
  virtual auto createBuffer(BufferDescriptor const& desc) noexcept
      -> std::unique_ptr<Buffer> override;
  /** create a texture on the device */
  virtual auto createTexture(TextureDescriptor const& desc) noexcept
      -> std::unique_ptr<Texture> override;
  /** create a sampler on the device */
  virtual auto createSampler(SamplerDescriptor const& desc) noexcept
      -> std::unique_ptr<Sampler> override;
  /* create a swapchain on the device */
  virtual auto createSwapChain(SwapChainDescriptor const& desc) noexcept
      -> std::unique_ptr<SwapChain> override;
  // Create resources binding objects
  // ---------------------------
  /** create a bind group layout on the device */
  virtual auto createBindGroupLayout(
      BindGroupLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroupLayout> override;
  /** create a pipeline layout on the device */
  virtual auto createPipelineLayout(
      PipelineLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<PipelineLayout> override;
  /** create a bind group on the device */
  virtual auto createBindGroup(BindGroupDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroup> override;
  // Create pipeline objects
  // ---------------------------
  /** create a shader module on the device */
  virtual auto createShaderModule(ShaderModuleDescriptor const& desc) noexcept
      -> std::unique_ptr<ShaderModule> override;
  /** create a compute pipeline on the device */
  virtual auto createComputePipeline(
      ComputePipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePipeline> override;
  /** create a render pipeline on the device */
  virtual auto createRenderPipeline(
      RenderPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPipeline> override;
  // Create command encoders
  // ---------------------------
  /** create a multi frame flights */
  virtual auto createMultiFrameFlights(
      MultiFrameFlightsDescriptor const& desc) noexcept
      -> std::unique_ptr<MultiFrameFlights> override;
  /** create a command encoder */
  virtual auto createCommandEncoder(
      CommandEncoderDescriptor const& desc) noexcept
      -> std::unique_ptr<CommandEncoder> override;
  // Create query sets
  // ---------------------------
  virtual auto createQuerySet(QuerySetDescriptor const& desc) noexcept
      -> std::unique_ptr<QuerySet> override;
  // Create ray tracing objects
  // ---------------------------
  /** create a BLAS */
  virtual auto createBLAS(BLASDescriptor const& desc) noexcept
      -> std::unique_ptr<BLAS> override;
  /** create a TLAS */
  virtual auto createTLAS(TLASDescriptor const& desc) noexcept
      -> std::unique_ptr<TLAS> override;
  /** create a ray tracing pipeline on the device */
  virtual auto createRayTracingPipeline(
      RayTracingPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RayTracingPipeline> override;
  // Create memory barrier objects
  // ---------------------------
  virtual auto createFence() noexcept -> std::unique_ptr<Fence> override;
  // Get extensions
  // ---------------------------
  /** fetch a ray tracing extension is available */
  virtual auto getRayTracingExtension() noexcept
      -> RayTracingExtension* override;

 public:
  /** initialzie */
  auto init() noexcept -> void;
  /** get vulkan logical device handle */
  auto getVkDevice() noexcept -> VkDevice& { return device; }
  /** get graphics queue handle */
  auto getVkGraphicsQueue() noexcept -> Queue_VK& { return graphicsQueue; }
  /** get compute queue handle */
  auto getVkComputeQueue() noexcept -> Queue_VK& { return computeQueue; }
  /** get present queue handle */
  auto getVkPresentQueue() noexcept -> Queue_VK& { return presentQueue; }
  /** get the adapter from which this device was created */
  auto getAdapterVk() noexcept -> Adapter_VK*& { return adapter; }
  /** get bind group pool */
  auto getBindGroupPool() noexcept -> BindGroupPool_VK* { return bindGroupPool.get(); }
  /** get bind group pool */
  auto getVMAAllocator() noexcept -> VmaAllocator& { return allocator; }
  /** create command pools */
  auto createCommandPools() noexcept -> void;
  /** create bind group pool */
  auto createBindGroupPool() noexcept -> void;
  /** create command pools */
  auto allocateCommandBuffer() noexcept -> std::unique_ptr<CommandBuffer_VK>;
  // Other extensions methods
  // ---------------------------
  /** create device UUID extension */
  virtual auto queryUUID() noexcept -> std::array<uint64_t, 2> override;
 private:
  /** vulkan logical device handle */
  VkDevice device;
  /** various queue handles */
  Queue_VK graphicsQueue, computeQueue, presentQueue;
  /** various queue command pools */
  std::unique_ptr<CommandPool_VK> graphicPool = nullptr;
  std::unique_ptr<CommandPool_VK> computePool = nullptr;
  std::unique_ptr<CommandPool_VK> presentPool = nullptr;
  /** the adapter from which this device was created */
  Adapter_VK* adapter = nullptr;
  /** VMA allocator */
  VmaAllocator allocator = nullptr;
  /** bind group pool */
  std::unique_ptr<BindGroupPool_VK> bindGroupPool = nullptr;
  /** vulkan ray tracing extension initialziation */
  auto initRayTracingExt() noexcept -> void;
  /** vulkan ray tracing extension properties */
  std::unique_ptr<RayTracingExtension_VK> raytracingExt = nullptr;
};

// Initialization Interface
// ===========================================================================
// Buffers Interface

struct SIByL_API Buffer_VK : public Buffer {
  /** constructor */
  Buffer_VK(Device_VK* device) : device(device) {}
  /** virtual destructor */
  virtual ~Buffer_VK();
  /** copy functions */
  Buffer_VK(Buffer_VK const& buffer) = delete;
  Buffer_VK(Buffer_VK&& buffer);
  auto operator=(Buffer_VK const& buffer) -> Buffer_VK& = delete;
  auto operator=(Buffer_VK&& buffer) -> Buffer_VK&;
  // Readonly Attributes
  // ---------------------------
  /** readonly get buffer size on GPU */
  virtual auto size() const noexcept -> size_t override { return _size; }
  /** readonly get buffer usage flags on GPU */
  virtual auto bufferUsageFlags() const noexcept -> BufferUsages override {
    return descriptor.usage;
  }
  /** readonly get map state on GPU */
  virtual auto bufferMapState() const noexcept -> BufferMapState override {
    return mapState;
  }
  /** readonly get device */
  virtual auto getDevice() const noexcept -> Device* override { return device; }
  /** get the device memory address is applicable */
  virtual auto getDeviceAddress() const noexcept -> uint64_t override;
  /** get the memory handle of the allocated buufer */
  virtual auto getMemHandle() const noexcept -> ExternalHandle override;
  // Map methods
  // ---------------------------
  /** Maps the given range of the GPUBuffer */
  virtual auto mapAsync(MapModeFlags mode, size_t offset = 0,
                        size_t size = 0) noexcept -> std::future<bool> override;
  /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
   * mapped range */
  virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept
      -> ArrayBuffer override;
  /** Unmaps the mapped range of the GPUBuffer and makes it��s contents available
   * for use by the GPU again. */
  virtual auto unmap() noexcept -> void override;
  // Lifecycle methods
  // ---------------------------
  /** destroy the buffer */
  virtual auto destroy() noexcept -> void override;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** set debug name */
  virtual auto getName() const noexcept -> std::string const& override;

 public:
  /** initialize the buffer */
  auto init(Device_VK* device, size_t size,
            BufferDescriptor const& desc) noexcept -> void;
  /** get vulkan buffer */
  auto getVkBuffer() noexcept -> VkBuffer& { return buffer; }
  /** get vulkan buffer device memory */
  auto getVkDeviceMemory() noexcept -> VkDeviceMemory& { return bufferMemory; }
  /** get vulkan buffer device memory */
  auto getVMAAllocation() noexcept -> VmaAllocation& { return allocation; }
  /** set buffer state */
  auto setBufferMapState(BufferMapState const& state) noexcept -> void {
    mapState = state;
  }
 protected:
  /** vulkan buffer */
  VkBuffer buffer = {};
  /** vulkan buffer device memory */
  VkDeviceMemory bufferMemory = {};
  /** buffer creation desc */
  BufferDescriptor descriptor = {};
  /** buffer creation desc */
  BufferMapState mapState = BufferMapState::UNMAPPED;
  /** mapped address of the buffer */
  void* mappedData = nullptr;
  /** size of the buffer */
  size_t _size = 0;
  /** the device this buffer is created on */
  Device_VK* device = nullptr;
  /** VMA allocation */
  VmaAllocation allocation;
  /** debug name */
  std::string name;
};

// Buffers Interface
// ===========================================================================
// Textures/TextureViews Interface

struct SIByL_API Texture_VK : public Texture {
  // Texture Behaviors
  // ---------------------------
  /** constructor */
  Texture_VK() noexcept = default;
  Texture_VK(Texture_VK&&) noexcept = default;
  Texture_VK(Texture_VK const&) noexcept = delete;
  Texture_VK(Device_VK* device, TextureDescriptor const& desc);
  Texture_VK(Device_VK* device, VkImage image, TextureDescriptor const& desc);
  auto operator=(Texture_VK&& texture)->Texture_VK & = default;
  auto operator=(Texture_VK const& texture)->Texture_VK & = delete;
  /** virtual descructor */
  virtual ~Texture_VK();
  /** create texture view of this texture */
  virtual auto createView(TextureViewDescriptor const& desc) noexcept
      -> std::unique_ptr<TextureView> override;
  /** destroy this texture */
  virtual auto destroy() noexcept -> void override;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** get name */
  virtual auto getName() -> std::string const& override;
  /** get texture descriptor */
  virtual auto getDescriptor() -> TextureDescriptor override;
  // Readonly Attributes
  // ---------------------------
  /** readonly width of the texture */
  virtual auto width() const noexcept -> uint32_t override;
  /** readonly height of the texture */
  virtual auto height() const noexcept -> uint32_t override;
  /** readonly depth or arrayLayers of the texture */
  virtual auto depthOrArrayLayers() const noexcept -> uint32_t override;
  /** readonly mip level count of the texture */
  virtual auto mipLevelCount() const noexcept -> uint32_t override;
  /** readonly sample count of the texture */
  virtual auto sampleCount() const noexcept -> uint32_t override;
  /** the dimension of the set of texel for each of this GPUTexture's
   * subresources. */
  virtual auto dimension() const noexcept -> TextureDimension override;
  /** readonly format of the texture */
  virtual auto format() const noexcept -> TextureFormat override;
  // Map methods
  // ---------------------------
  /** Maps the given range of the GPUBuffer */
  virtual auto mapAsync(MapModeFlags mode, size_t offset = 0,
                        size_t size = 0) noexcept -> std::future<bool> override;
  /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
   * mapped range */
  virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept
      -> ArrayBuffer override;
  /** Unmaps the mapped range of the GPUBuffer and makes it��s contents available
   * for use by the GPU again. */
  virtual auto unmap() noexcept -> void override;

 public:
  /** get the VkImage */
  auto getVkImage() noexcept -> VkImage& { return image; }
  /** get the VkDeviceMemory */
  auto getVkDeviceMemory() noexcept -> VkDeviceMemory& { return deviceMemory; }
  /* get VMA Allocation */
  auto getVMAAllocation() noexcept -> VmaAllocation& { return allocation; }
  /** set buffer state */
  auto setBufferMapState(BufferMapState const& state) noexcept -> void { mapState = state; }
 private:
  /** vulkan image */
  VkImage image;
  /** vulkan image device memory */
  VkDeviceMemory deviceMemory = nullptr;
  /** Texture Descriptor */
  TextureDescriptor descriptor;
  /** VMA allocation */
  VmaAllocation allocation;
  /** buffer map state */
  BufferMapState mapState = BufferMapState::UNMAPPED;
  /** mapped address of the buffer */
  void* mappedData = nullptr;
  /** the device this texture is created on */
  Device_VK* device = nullptr;
  /** name */
  std::string name = "Unnamed Texture";
  /** the VKImage is reference to an external resource */
  bool external = false;
};

struct SIByL_API TextureView_VK : public TextureView {
  /** create textureviw */
  TextureView_VK() noexcept = default;
  TextureView_VK(TextureView_VK&& view) noexcept;
  TextureView_VK(TextureView_VK const& view) = delete;
  TextureView_VK(Device_VK* device, Texture_VK* texture,
                 TextureViewDescriptor const& descriptor);
  /* copy functions */
  auto operator=(TextureView_VK const& view) -> TextureView_VK& = delete;
  auto operator=(TextureView_VK && view) -> TextureView_VK&;
  /** virtual destructor */
  virtual ~TextureView_VK();
  /** get binded texture */
  virtual auto getTexture() noexcept -> Texture* override { return texture; }
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** get width */
  virtual auto getWidth() noexcept -> uint32_t override;
  /** get height */
  virtual auto getHeight() noexcept -> uint32_t override;
  /** Vulkan texture view */
  VkImageView imageView;
  /** Texture view descriptor */
  TextureViewDescriptor descriptor;
  /** The texture this view is pointing to */
  Texture_VK* texture = nullptr;
  /** The device that the pointed texture is created on */
  Device_VK* device = nullptr;
  /** width of the view */
  uint32_t width;
  /** height of the view */
  uint32_t height;
};

// Textures/TextureViews Interface
// ===========================================================================
// Samplers Interface

struct SIByL_API Sampler_VK : public Sampler {
  /** initialization */
  Sampler_VK(SamplerDescriptor const& desc, Device_VK* device);
  /** virtual destructor */
  virtual ~Sampler_VK();
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** get debug name */
  virtual auto getName() const noexcept -> std::string const& override;
  /** vulkan Texture Sampler */
  VkSampler textureSampler;
  /** the device this sampler is created on */
  Device_VK* device = nullptr;
  /** debug name */
  std::string name;
};

// Samplers Interface
// ===========================================================================
// SwapChain Interface

struct SIByL_API SwapChain_VK : public SwapChain {
  /** virtual destructor */
  virtual ~SwapChain_VK();
  SwapChain_VK() = default;
  SwapChain_VK(SwapChain_VK&&) = default;
  SwapChain_VK(SwapChain_VK const&) = delete;
  SwapChain_VK& operator= (SwapChain_VK &&) = default;
  SwapChain_VK& operator= (SwapChain_VK const&) = delete;
  /** intialize the swapchin */
  auto init(Device_VK* device, SwapChainDescriptor const& desc) noexcept -> void;
  /** get texture view */
  virtual auto getTexture(int i) noexcept -> Texture* override {
    return &swapChainTextures[i];
  }
  virtual auto getTextureView(int i) noexcept -> TextureView* override {
    return &textureViews[i];
  }
  /** invalid swapchain */
  virtual auto recreate() noexcept -> void override;
  /** vulkan SwapChain */
  VkSwapchainKHR swapChain;
  /** vulkan SwapChain Extent */
  VkExtent2D swapChainExtend;
  /** vulkan SwapChain format */
  VkFormat swapChainImageFormat;
  /** vulkan SwapChain fetched images */
  std::vector<Texture_VK> swapChainTextures;
  /** vulkan SwapChain fetched images views */
  std::vector<TextureView_VK> textureViews;
  /** the device this sampler is created on */
  Device_VK* device = nullptr;
};

// SwapChain Interface
// ===========================================================================
// Shader Modules Interface

struct SIByL_API ShaderModule_VK : public ShaderModule {
  /** initalize shader module */
  ShaderModule_VK(Device_VK* device, ShaderModuleDescriptor const& desc);
  /** virtual descructor */
  ~ShaderModule_VK();
  /** copy functions */
  ShaderModule_VK(ShaderModule_VK const& shader) = delete;
  ShaderModule_VK(ShaderModule_VK&& shader);
  auto operator=(ShaderModule_VK const& shader) -> ShaderModule_VK& = delete;
  auto operator=(ShaderModule_VK&& shader) -> ShaderModule_VK&;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** get debug name */
  virtual auto getName() -> std::string const& override;
  /** the shader stages included in this module */
  ShaderStages stages;
  /** vulkan shader module */
  VkShaderModule shaderModule = {};
  /** name of entry point */
  std::string entryPoint = "main";
  /** vulkan shader stage create info */
  VkPipelineShaderStageCreateInfo shaderStageInfo{};
  /** the device this shader module is created on */
  Device_VK* device = nullptr;
 private:
  std::string name;
};

// Shader Modules Interface
// ===========================================================================
// Resource Binding Interface

struct SIByL_API BindGroupLayout_VK : public BindGroupLayout {
  /** contructor */
  BindGroupLayout_VK(Device_VK* device, BindGroupLayoutDescriptor const& desc);
  /** destructor */
  ~BindGroupLayout_VK();
  /** get BindGroup Layout Descriptor */
  virtual auto getBindGroupLayoutDescriptor() const noexcept
      -> BindGroupLayoutDescriptor const& override;
  /** vulkan Descriptor Set Layout */
  VkDescriptorSetLayout layout;
  /** Bind Group Layout Descriptor */
  BindGroupLayoutDescriptor descriptor;
  /** the device this bind group layout is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API BindGroupPool_VK {
  /** initialzier */
  BindGroupPool_VK(Device_VK* device);
  /** destructor */
  ~BindGroupPool_VK();
  /** vulkan Bind Group Pool */
  VkDescriptorPool descriptorPool;
  /** the device this bind group pool is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API BindGroup_VK : public BindGroup {
  /** initialzie */
  BindGroup_VK(Device_VK* device, BindGroupDescriptor const& desc);
  /** update binding */
  virtual auto updateBinding(
      std::vector<BindGroupEntry> const& entries) noexcept -> void override;
  /** vulkan Descriptor Set */
  VkDescriptorSet set = {};
  /** the bind group set this bind group is created on */
  BindGroupPool_VK* descriptorPool;
  /** layout */
  BindGroupLayout* layout;
  /** the device this bind group is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API PipelineLayout_VK : public PipelineLayout {
  /** intializer */
  PipelineLayout_VK(Device_VK* device, PipelineLayoutDescriptor const& desc);
  /** virtual destructor */
  virtual ~PipelineLayout_VK();
  /** copy functions */
  PipelineLayout_VK(PipelineLayout_VK const& layout) = delete;
  PipelineLayout_VK(PipelineLayout_VK&& layout);
  auto operator=(PipelineLayout_VK const& layout)
      -> PipelineLayout_VK& = delete;
  auto operator=(PipelineLayout_VK&& layout) -> PipelineLayout_VK&;
  /** vulkan pipeline layout */
  VkPipelineLayout pipelineLayout;
  /** the push constans on pipeline layouts */
  std::vector<VkPushConstantRange> pushConstants;
  /** the device this pipeline layout is created on */
  Device_VK* device = nullptr;
};

// Resource Binding Interface
// ===========================================================================
// Pipelines Interface

struct SIByL_API ComputePipeline_VK : public ComputePipeline {
  /** constructor */
  ComputePipeline_VK(Device_VK* device, ComputePipelineDescriptor const& desc);
  /** virtual destructor */
  virtual ~ComputePipeline_VK();
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** vulkan compute pipeline */
  VkPipeline pipeline;
  /** compute pipeline layout */
  PipelineLayout_VK* layout = nullptr;
  /** the device this compute pipeline is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API RenderPass_VK {
  /** render pass initialize */
  RenderPass_VK(Device_VK* device, RenderPassDescriptor const& desc);
  /** virtual destructor */
  virtual ~RenderPass_VK();
  /** copy functions */
  RenderPass_VK(RenderPass_VK const& pass) = delete;
  RenderPass_VK(RenderPass_VK&& pass);
  auto operator=(RenderPass_VK const& pass) -> RenderPass_VK& = delete;
  auto operator=(RenderPass_VK&& pass) -> RenderPass_VK&;
  /** vulkan render pass */
  VkRenderPass renderPass;
  /** vulkan render pass clear value */
  std::vector<VkClearValue> clearValues;
  /** the device this render pass is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API RenderPipeline_VK : public RenderPipeline {
  /** constructor */
  RenderPipeline_VK(Device_VK* device, RenderPipelineDescriptor const& desc);
  /** virtual destructor */
  virtual ~RenderPipeline_VK();
  /** copy functions */
  RenderPipeline_VK(RenderPipeline_VK const& pipeline) = delete;
  RenderPipeline_VK(RenderPipeline_VK&& pipeline);
  auto operator=(RenderPipeline_VK const& pipeline)
      -> RenderPipeline_VK& = delete;
  auto operator=(RenderPipeline_VK&& pipeline) -> RenderPipeline_VK&;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** vulkan render pipeline */
  VkPipeline pipeline = {};
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
  } fixedFunctionSetttings;
  /** the reusable create information of the pipeline */
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  /** combine the pipelien with a render pass and then re-valid it */
  auto combineRenderPass(RenderPass_VK* renderpass) noexcept -> void;
  /** the device this render pipeline is created on */
  Device_VK* device = nullptr;
};

// Pipelines Interface
// ===========================================================================
// Command objects Interface

struct SIByL_API CommandPool_VK {
  /** initialize */
  CommandPool_VK(Device_VK* device);
  /** destructor */
  ~CommandPool_VK();
  /** allocate command buffer */
  auto allocateCommandBuffer() noexcept -> std::unique_ptr<CommandBuffer_VK>;
  /** vulkan command pool */
  VkCommandPool commandPool;
  /** the device this command pool is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API CommandBuffer_VK : public CommandBuffer {
  /** destructor */
  virtual ~CommandBuffer_VK();
  /** vulkan command buffer */
  VkCommandBuffer commandBuffer;
  /** command pool the buffer is on */
  CommandPool_VK* commandPool = nullptr;
  /** the device this command buffer is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API CommandEncoder_VK : public CommandEncoder {
  /** virtual descructor */
  virtual ~CommandEncoder_VK();
  /** Begins encoding a render pass described by descriptor. */
  virtual auto beginRenderPass(RenderPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPassEncoder> override;
  /** Begins encoding a compute pass described by descriptor. */
  virtual auto beginComputePass(ComputePassDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePassEncoder> override;
  /** Begins encoding a ray tracing pass described by descriptor. */
  virtual auto beginRayTracingPass(
      RayTracingPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RayTracingPassEncoder> override;
  /** Insert a barrier. */
  virtual auto pipelineBarrier(BarrierDescriptor const& desc) noexcept
      -> void override;
  /**  Encode a command into the CommandEncoder that copies data from
   * a sub-region of a GPUBuffer to a sub-region of another Buffer. */
  virtual auto copyBufferToBuffer(Buffer* source, size_t sourceOffset,
                                  Buffer* destination, size_t destinationOffset,
                                  size_t size) noexcept -> void override;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * Buffer with zeros. */
  virtual auto clearBuffer(Buffer* buffer, size_t offset, size_t size) noexcept
      -> void override;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * texture with any color. */
  virtual auto clearTexture(Texture* texture,
                            TextureClearDescriptor const& desc) noexcept
      -> void override;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * Buffer with a value. */
  virtual auto fillBuffer(Buffer* buffer, size_t offset, size_t size,
                          float fillValue) noexcept -> void override;
  /** Encode a command into the CommandEncoder that copies data from a
   * sub-region of a Buffer to a sub-region of one or multiple continuous
   * texture subresources. */
  virtual auto copyBufferToTexture(ImageCopyBuffer const& source,
                                   ImageCopyTexture const& destination,
                                   Extend3D const& copySize) noexcept
      -> void override;
  /** Encode a command into the CommandEncoder that copies data from a
   * sub-region of one or multiple continuous texture subresourcesto a
   * sub-region of a Buffer. */
  virtual auto copyTextureToBuffer(ImageCopyTexture const& source,
                                   ImageCopyBuffer const& destination,
                                   Extend3D const& copySize) noexcept
      -> void override;
  /** Encode a command into the CommandEncoder that copies data from
   * a sub-region of one or multiple contiguous texture subresources to
   * another sub-region of one or multiple continuous texture subresources. */
  virtual auto copyTextureToTexture(ImageCopyTexture const& source,
                                    ImageCopyTexture const& destination,
                                    Extend3D const& copySize) noexcept
      -> void override;
  /** Reset the queryset. */
  virtual auto resetQuerySet(QuerySet* querySet, uint32_t firstQuery,
                             uint32_t queryCount) noexcept -> void override;
  /** Writes a timestamp value into a querySet when all
   * previous commands have completed executing. */
  virtual auto writeTimestamp(QuerySet* querySet, PipelineStages stageMask,
                              uint32_t queryIndex) noexcept -> void override;
  /** Resolves query results from a QuerySet out into a range of a Buffer. */
  virtual auto resolveQuerySet(QuerySet* querySet, uint32_t firstQuery,
                               uint32_t queryCount, Buffer& destination,
                               uint64_t destinationOffset) noexcept
      -> void override;
  /** copy accceleration structure to a new instance */
  virtual auto cloneBLAS(BLAS* src) noexcept -> std::unique_ptr<BLAS> override;
  /** update blas by refitting, only deformation is allowed */
  virtual auto updateBLAS(BLAS* src, Buffer* vertexBuffer,
                          Buffer* indexBuffer) noexcept -> void override;
  /** Completes recording of the commands sequence and returns a corresponding
   * GPUCommandBuffer. */
  virtual auto finish() noexcept -> CommandBuffer* override;
  /** begin Debug marker region */
  virtual auto beginDebugUtilsLabelEXT(
      DebugUtilLabelDescriptor const& desc) noexcept -> void override;
  /** end Debug marker region */
  virtual auto endDebugUtilsLabelEXT() noexcept -> void override;
  /** underlying command buffer */
  std::unique_ptr<CommandBuffer_VK> commandBufferOnce = nullptr;
  /** underlying command buffer */
  CommandBuffer_VK* commandBuffer = nullptr;
};

// Pipelines Interface
// ===========================================================================
// Compute - Pipelines Interface

struct SIByL_API ComputePassEncoder_VK : public ComputePassEncoder {
  /** virtual descructor */
  virtual ~ComputePassEncoder_VK();
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(
      uint32_t index, BindGroup* bindgroup,
      std::vector<BufferDynamicOffset> const& dynamicOffsets = {}) noexcept
      -> void override;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
                            uint64_t dynamicOffsetDataStart,
                            uint32_t dynamicOffsetDataLength) noexcept
      -> void override;
  /** Push constants */
  virtual auto pushConstants(void* data, ShaderStages stages,
                             uint32_t offset, uint32_t size) noexcept
      -> void override;
  /** Sets the current GPUComputePipeline. */
  virtual auto setPipeline(ComputePipeline* pipeline) noexcept -> void override;
  /** Dispatch work to be performed with the current GPUComputePipeline.*/
  virtual auto dispatchWorkgroups(uint32_t workgroupCountX,
                                  uint32_t workgroupCountY = 1,
                                  uint32_t workgroupCountZ = 1) noexcept
      -> void override;
  /** Dispatch work to be performed with the current GPUComputePipeline using
   * parameters read from a GPUBuffer. */
  virtual auto dispatchWorkgroupsIndirect(Buffer* indirectBuffer,
                                          uint64_t indirectOffset) noexcept
      -> void override;
  /** Completes recording of the compute pass commands sequence. */
  virtual auto end() noexcept -> void override;
  /** current compute pipeline */
  ComputePipeline_VK* computePipeline = nullptr;
  /** command buffer binded */
  CommandBuffer_VK* commandBuffer = nullptr;
};

// Compute - Pipelines Interface
// ===========================================================================
// Rasterizer - Pipelines Interface

struct SIByL_API FrameBuffer_VK {
  /** intializer */
  FrameBuffer_VK(Device_VK* device, RenderPassDescriptor const& desc,
                 RenderPass_VK* renderpass);
  /** destructor */
  ~FrameBuffer_VK();
  /** get width of the framebuffer */
  auto width() -> uint32_t { return _width; }
  /** get height of the framebuffer */
  auto height() -> uint32_t { return _height; }
  /** vulkan framebuffer */
  VkFramebuffer framebuffer = {};
  /** clear values */
  std::vector<VkClearValue> clearValues = {};
  /** vulkan device the framebuffer created on */
  Device_VK* device = nullptr;
  /** width / height */
  uint32_t _width = 0, _height = 0;
};

struct SIByL_API RenderPassEncoder_VK : public RenderPassEncoder {
  /** virtual descructor */
  virtual ~RenderPassEncoder_VK();
  /** Sets the current GPURenderPipeline. */
  virtual auto setPipeline(RenderPipeline* pipeline) noexcept -> void override;
  /** Sets the current index buffer. */
  virtual auto setIndexBuffer(Buffer* buffer, IndexFormat indexFormat,
                              uint64_t offset = 0, uint64_t size = 0) noexcept
      -> void override;
  /** Sets the current vertex buffer for the given slot. */
  virtual auto setVertexBuffer(uint32_t slot, Buffer* buffer,
                               uint64_t offset = 0, uint64_t size = 0) noexcept
      -> void override;
  /** Draws primitives. */
  virtual auto draw(uint32_t vertexCount, uint32_t instanceCount = 1,
                    uint32_t firstVertex = 0,
                    uint32_t firstInstance = 0) noexcept -> void override;
  /** Draws indexed primitives. */
  virtual auto drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
                           uint32_t firstIndex = 0, int32_t baseVertex = 0,
                           uint32_t firstInstance = 0) noexcept
      -> void override;
  /** Draws primitives using parameters read from a GPUBuffer. */
  virtual auto drawIndirect(Buffer* indirectBuffer, uint64_t indirectOffset,
                            uint32_t drawCount, uint32_t stride) noexcept
      -> void override;
  /** Draws indexed primitives using parameters read from a GPUBuffer. */
  virtual auto drawIndexedIndirect(Buffer* indirectBuffer, uint64_t offset,
                                   uint32_t drawCount, uint32_t stride) noexcept
      -> void override;
  /** Sets the viewport used during the rasterization stage to linearly map
   * from normalized device coordinates to viewport coordinates. */
  virtual auto setViewport(float x, float y, float width, float height,
                           float minDepth, float maxDepth) noexcept
      -> void override;
  /** Sets the scissor rectangle used during the rasterization stage.
   * After transformation into viewport coordinates any fragments
   * which fall outside the scissor rectangle will be discarded. */
  virtual auto setScissorRect(IntegerCoordinate x, IntegerCoordinate y,
                              IntegerCoordinate width,
                              IntegerCoordinate height) noexcept
      -> void override;
  /** Sets the constant blend color and alpha values used with
   * "constant" and "one-minus-constant" GPUBlendFactors. */
  virtual auto setBlendConstant(se::vec4 color) noexcept -> void override;
  /** Sets the [[stencil_reference]] value used during
   * stencil tests with the "replace" GPUStencilOperation. */
  virtual auto setStencilReference(StencilValue reference) noexcept
      -> void override;
  /** begin occlusion query */
  virtual auto beginOcclusionQuery(uint32_t queryIndex) noexcept
      -> void override;
  /** end occlusion query */
  virtual auto endOcclusionQuery() noexcept -> void override;
  /** Executes the commands previously recorded into the given GPURenderBundles
   * as part of this render pass. */
  virtual auto executeBundles(std::vector<RenderBundle> const& bundles) noexcept
      -> void override;
  /** Completes recording of the render pass commands sequence. */
  virtual auto end() noexcept -> void override;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(
      uint32_t index, BindGroup* bindgroup,
      std::vector<BufferDynamicOffset> const& dynamicOffsets = {}) noexcept
      -> void override;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
                            uint64_t dynamicOffsetDataStart,
                            uint32_t dynamicOffsetDataLength) noexcept
      -> void override;
  /** Push constants */
  virtual auto pushConstants(void* data, ShaderStages stages,
                             uint32_t offset, uint32_t size) noexcept
      -> void override;
  /** render pass */
  std::unique_ptr<RenderPass_VK> renderPass = nullptr;
  /** frame buffer */
  std::unique_ptr<FrameBuffer_VK> frameBuffer = nullptr;
  /* current render pipeline */
  RenderPipeline_VK* renderPipeline = nullptr;
  /** command buffer binded */
  CommandBuffer_VK* commandBuffer = nullptr;
};

// ===========================================================================
// Queue Interface

struct SIByL_API QuerySet_VK : public QuerySet {
  /** initialize */
  QuerySet_VK(Device_VK* device, QuerySetDescriptor const& desc);
  /** virtual destructor */
  ~QuerySet_VK();
  /** fetch the query pooll result */
  virtual auto resolveQueryResult(uint32_t firstQuery, uint32_t queryCount,
    size_t dataSize, void* pData, uint64_t stride, QueryResultFlag flag) noexcept
      -> void override;
  VkQueryPool queryPool;
  /** vulkan device the QuerySet is created on */
  Device_VK* device = nullptr;
};

// Queue Interface
// ===========================================================================
// Synchronization Interface

struct SIByL_API Fence_VK : public Fence {
  /** initialize */
  Fence_VK() = default;
  Fence_VK(Device_VK* device);
  /** virtual destructor */
  virtual ~Fence_VK();
  /* wait the fence */
  virtual auto wait() noexcept -> void override;
  /* reset the fence */
  virtual auto reset() noexcept -> void override;
  /** vulkan fence */
  VkFence fence;
  /** the device this fence is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API Semaphore_VK : public Semaphore {
  /** initialize */
  Semaphore_VK() = default;
  Semaphore_VK(Device_VK* device);
  /** virtual destructor */
  virtual ~Semaphore_VK();
  /** vulkan semaphore */
  VkSemaphore semaphore;
  /** the device this semaphore is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API MultiFrameFlights_VK : public MultiFrameFlights {
  MultiFrameFlights_VK() = default;
  MultiFrameFlights_VK(MultiFrameFlights_VK&&) = default;
  MultiFrameFlights_VK(MultiFrameFlights_VK const&) = delete;
  MultiFrameFlights_VK& operator= (MultiFrameFlights_VK&&) = default;
  MultiFrameFlights_VK& operator= (MultiFrameFlights_VK const&) = delete;
  /** initialize */
  MultiFrameFlights_VK(Device_VK* device, int maxFlightNum = 2, SwapChain* swapchain = nullptr);
  /** virtual destructor */
  ~MultiFrameFlights_VK() = default;
  /** start frame */
  virtual auto frameStart() noexcept -> void override;
  /** end frame */
  virtual auto frameEnd() noexcept -> void override;
  /** get current flight id */
  virtual auto getFlightIndex() noexcept -> uint32_t { return currentFrame; }
  /** get current swapchain id */
  virtual auto getSwapchainIndex() noexcept -> uint32_t { return imageIndex; }
  /** get current command buffer */
  virtual auto getCommandBuffer() noexcept -> CommandBuffer* override;
  /** get current Image Available Semaphore */
  virtual auto getImageAvailableSeamaphore() noexcept -> Semaphore* override;
  /** get current Render Finished Semaphore */
  virtual auto getRenderFinishedSeamaphore() noexcept -> Semaphore* override;
  /** get current fence */
  virtual auto getFence() noexcept -> Fence* override;
  std::vector<std::unique_ptr<CommandBuffer_VK>> commandBuffers;
  std::vector<Semaphore_VK> imageAvailableSemaphores;
  std::vector<Semaphore_VK> renderFinishedSemaphores;
  std::vector<Fence_VK> inFlightFences;
  SwapChain_VK* swapChain = nullptr;
  uint32_t currentFrame = 0;
  int maxFlightNum = 0;
  Device_VK* device = nullptr;
  uint32_t imageIndex;
};

// Synchronization Interface
// ===========================================================================
// Ray Tracing Interface

struct SIByL_API RayTracingPipeline_VK : public RayTracingPipeline {
  /** initialzie */
  RayTracingPipeline_VK(Device_VK* device, RayTracingPipelineDescriptor const& desc);
  /** destructor */
  ~RayTracingPipeline_VK();
  /** set debug name */
  virtual auto setName(std::string const& name) -> void override;
  /** vulkan ray tracing pipeline */
  VkPipeline pipeline = {};
  /** vulkan pipeline layout */
  PipelineLayout_VK* pipelineLayout = nullptr;
  /** vulkan SBT buffer */
  std::unique_ptr<Buffer> SBTBuffer = nullptr;
  /** vulkan strided device address region */
  VkStridedDeviceAddressRegionKHR rayGenRegion = {};
  VkStridedDeviceAddressRegionKHR missRegion = {};
  VkStridedDeviceAddressRegionKHR hitRegion = {};
  VkStridedDeviceAddressRegionKHR callableRegion = {};
  /** The device where the pipeline is created on */
  Device_VK* device;
};

struct SIByL_API BLAS_VK : public BLAS {
  /** initialzie */
  BLAS_VK(Device_VK* device, BLASDescriptor const& descriptor);
  /** only allocate according to another BLAS */
  BLAS_VK(Device_VK* device, BLAS_VK* descriptor);
  /** virtual destructor */
  virtual ~BLAS_VK();
  /** get descriptor */
  virtual auto getDescriptor() noexcept -> BLASDescriptor override { return descriptor; }
  /** VULKAN BLAS object*/
  VkAccelerationStructureKHR blas = {};
  /** VULKAN BLAS buffer */
  std::unique_ptr<Buffer> bufferBLAS = nullptr;
  /** descriptor is remained */
  BLASDescriptor descriptor;
  /** vulkan device the BLAS is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API TLAS_VK : public TLAS {
  /** initialzie */
  TLAS_VK(Device_VK* device, TLASDescriptor const& descriptor);
  /** virtual destructor */
  virtual ~TLAS_VK();
  /** VULKAN TLAS object*/
  VkAccelerationStructureKHR tlas = {};
  /** VULKAN TLAS buffer */
  std::unique_ptr<Buffer> bufferTLAS = nullptr;
  /** vulkan device the TLAS is created on */
  Device_VK* device = nullptr;
};

struct SIByL_API RayTracingExtension_VK : public RayTracingExtension {
  /** device ray tracing properties */
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR vkRayTracingProperties;
  VkPhysicalDeviceAccelerationStructurePropertiesKHR vASProperties;
};

struct SIByL_API RayTracingPassEncoder_VK : public RayTracingPassEncoder {
  /** initialize */
  RayTracingPassEncoder_VK(CommandEncoder_VK* commandEncoder,
                           RayTracingPassDescriptor const& desc);
  /** virtual destructor */
  ~RayTracingPassEncoder_VK();
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(
      uint32_t index, BindGroup* bindgroup,
      std::vector<BufferDynamicOffset> const& dynamicOffsets = {}) noexcept
      -> void override;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
                            uint64_t dynamicOffsetDataStart,
                            uint32_t dynamicOffsetDataLength) noexcept
      -> void override;
  /** Push constants */
  virtual auto pushConstants(void* data, ShaderStages stages,
                             uint32_t offset, uint32_t size) noexcept
      -> void override;
  /** Set a ray tracing pipeline as the current pipeline */
  virtual auto setPipeline(RayTracingPipeline* pipeline) noexcept
      -> void override;
  /** Trace rays using current ray tracing pipeline */
  virtual auto traceRays(uint32_t width, uint32_t height,
                         uint32_t depth) noexcept -> void override;
  /** Trace rays using current ray tracing pipeline by an indirect buffer */
  virtual auto traceRaysIndirect(Buffer* indirectBuffer,
                                 uint64_t indirectOffset) noexcept
      -> void override;
  /** End the ray tracing pass */
  virtual auto end() noexcept -> void override;
  /** Vulkan ray tracing pipeline which is current binded */
  RayTracingPipeline_VK* raytracingPipeline = nullptr;
  /** Vulkan command encoder the pass encoder is working on */
  CommandEncoder_VK* commandEncoder;
};

auto SIByL_API getVkGeometryFlagsKHR(BLASGeometryFlags input) noexcept -> VkGeometryFlagsKHR;
auto SIByL_API getBufferVkDeviceAddress(Device_VK* device, Buffer* buffer) noexcept -> VkDeviceAddress;
auto SIByL_API getVkImageLayout(TextureLayout layout) noexcept -> VkImageLayout;

// Ray Tracing Interface
// ===========================================================================
}