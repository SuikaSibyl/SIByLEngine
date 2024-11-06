#define DLIB_EXPORT
#include <se.rhi.hpp>
#undef DLIB_EXPORT
#include <cuda_runtime.h>
#include <Windows.h>
#include <se.core.hpp>
#include <format>
#include <optional>

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::string message = std::format("CUDA error at {0}:{1} code={2}({3}) \"{4}\"", file, line,
        static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    se::root::print::error(message);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

namespace se::rhi {
auto CUDAContext::initialize(std::array<uint64_t, 2> const& uuid) noexcept -> void {
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    root::print::error("CUDA error: no devices supporting CUDA.");
    return;
  }

  // Find the GPU which is selected by Vulkan
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
      // Compare the cuda device UUID with vulkan UUID
      int ret = memcmp((void *)&deviceProp.uuid, uuid.data(), 16);
      if (ret == 0) {
        checkCudaErrors(cudaSetDevice(current_device));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
        std::string message = std::format("CUDA :: GPU Device {0}: \"{1}\" with compute capability {2}.{3}",
            current_device, deviceProp.name, deviceProp.major, deviceProp.minor);
        se::root::print::log(message);
        break;
      }
    } else {
      devices_prohibited++;
    }
    current_device++;
  }

  if (devices_prohibited == device_count) {
    se::root::print::error("CUDA error: No Vulkan-CUDA Interop capable GPU found.");
    return;
  }
  int device = current_device;
  cudaDeviceProp prop = {};
  checkCudaErrors(cudaSetDevice(device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  se::root::print::debug("CUDA :: Device set successfully.");
}

auto CUDAContext::initialize(Device* device) noexcept -> void {
  std::array<uint64_t, 2> const uuid = device->queryUUID();
  se::rhi::CUDAContext::initialize(uuid);
}

auto CUDAContext::synchronize() noexcept -> void {
  cudaDeviceSynchronize();
}

struct CUDABufferImpl : public CUDABuffer {
  virtual ~CUDABufferImpl();
  virtual auto ptr() noexcept -> void* override;
  cudaExternalMemory_t cudaMem;
  float* dataPtr;
  bool isExternal = false;
};

CUDABufferImpl::~CUDABufferImpl() {
  cudaFree(dataPtr);
  if (isExternal) {
    checkCudaErrors(cudaDestroyExternalMemory(cudaMem));
  }
}

auto CUDABufferImpl::ptr() noexcept -> void* { return (void*)dataPtr; }

auto CUDAContext::toCUDABuffer(Buffer* buffer) noexcept -> std::unique_ptr<CUDABuffer> {
  std::unique_ptr<CUDABufferImpl> cubuffer = std::make_unique<CUDABufferImpl>();
  cubuffer->isExternal = true;
  se::rhi::Buffer::ExternalHandle handle = buffer->getMemHandle();
  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  externalMemoryHandleDesc.size = handle.size + handle.offset;
  externalMemoryHandleDesc.handle.win32.handle = (HANDLE)handle.handle;
  checkCudaErrors(cudaImportExternalMemory(&cubuffer->cudaMem, &externalMemoryHandleDesc));
  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = handle.offset;
  externalMemBufferDesc.size = handle.size;
  externalMemBufferDesc.flags = 0;
  void** ptr = (void**)&cubuffer->dataPtr;
  checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
    ptr, cubuffer->cudaMem, &externalMemBufferDesc));
  return cubuffer;
}

auto CUDAContext::toCUDABufferInterval(Buffer* buffer, int offset, int size) noexcept -> std::unique_ptr<CUDABuffer> {
  std::unique_ptr<CUDABufferImpl> cubuffer = std::make_unique<CUDABufferImpl>();
  cubuffer->isExternal = true;
  se::rhi::Buffer::ExternalHandle handle = buffer->getMemHandle();
  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  externalMemoryHandleDesc.size = handle.size + handle.offset;
  externalMemoryHandleDesc.handle.win32.handle = (HANDLE)handle.handle;
  checkCudaErrors(cudaImportExternalMemory(&cubuffer->cudaMem, &externalMemoryHandleDesc));
  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = handle.offset + offset;
  externalMemBufferDesc.size = size;
  externalMemBufferDesc.flags = 0;
  void** ptr = (void**)&cubuffer->dataPtr;
  checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
    ptr, cubuffer->cudaMem, &externalMemBufferDesc));
  return cubuffer;
}

auto CUDAContext::allocCUDABuffer(size_t size)noexcept -> std::unique_ptr<CUDABuffer> {
  std::unique_ptr<CUDABufferImpl> cubuffer = std::make_unique<CUDABufferImpl>();
  cudaMalloc(&cubuffer->dataPtr, size);
  cudaMemset(cubuffer->dataPtr, 0, size);
  return cubuffer;
}

struct CUDASemaphoreImpl : public CUDASemaphore {
  virtual ~CUDASemaphoreImpl() {
  }
  cudaExternalSemaphore_t semaphore;
};

struct CUDAStreamImpl : public CUDAStream {
  virtual ~CUDAStreamImpl() {}

  virtual auto waitSemaphoreAsync(CUDASemaphore* semaphore, size_t waitValue) noexcept -> void override {
    // Wait for vulkan to complete it's work
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = waitValue;
    CUDASemaphoreImpl* semaimpl = reinterpret_cast<CUDASemaphoreImpl*>(semaphore);
    checkCudaErrors(cudaWaitExternalSemaphoresAsync(
      &semaimpl->semaphore, &waitParams, 1, stream));
  }
  
  virtual auto signalSemaphoreAsync(CUDASemaphore* semaphore, size_t signalValue) noexcept -> void override {
    // Signal vulkan to continue with the updated buffers
    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = signalValue;
    CUDASemaphoreImpl* semaimpl = reinterpret_cast<CUDASemaphoreImpl*>(semaphore);
    checkCudaErrors(cudaSignalExternalSemaphoresAsync(
      &semaimpl->semaphore, &signalParams, 1, stream));
  }

  cudaStream_t stream;
};
}