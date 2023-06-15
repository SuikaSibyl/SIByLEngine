﻿#include "../Public/SE.RHI-VK.hpp"
#define USE_VMA
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace SIByL::RHI {

#pragma region VK_QUEUE_IMPL

auto Queue_VK::onSubmittedWorkDone() noexcept -> std::future<bool> {
  return std::future<bool>{};
}

auto Queue_VK::writeBuffer(Buffer* buffer, uint64_t bufferOffset,
                           ArrayBuffer* data, uint64_t dataOffset,
                           Extend3D const& size) noexcept -> void {}

auto Queue_VK::writeTexture(ImageCopyTexture const& destination,
                            ArrayBuffer* data, ImageDataLayout const& layout,
                            Extend3D const& size) noexcept -> void {}

auto Queue_VK::copyExternalImageToTexture(
    ImageCopyExternalImage const& source,
    ImageCopyExternalImage const& destination,
    Extend3D const& copySize) noexcept -> void {}

auto Queue_VK::waitIdle() noexcept -> void { vkQueueWaitIdle(queue); }

#pragma endregion

#pragma region VK_CONTEXT_IMPL

/** Whether enable validation layer */
constexpr bool const enableValidationLayers = true;
/** Whether enable validation layer verbose output */
#ifdef VK_VERBOSE
constexpr bool const enableValidationLayerVerboseOutput = true;
#else
constexpr bool const enableValidationLayerVerboseOutput = false;
#endif  // VK_VERBOSE

/** Possible names of validation layer */
std::vector<const char*> const validationLayerNames = {
    "VK_LAYER_KHRONOS_validation",
};

/** Debug callback of vulkan validation layer */
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {
  switch (messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      if (enableValidationLayerVerboseOutput)
        Core::LogManager::Log(
            std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      Core::LogManager::Log(
          std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      Core::LogManager::Warning(
          std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      Core::LogManager::Error(
          std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
      Core::LogManager::Error(
          std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
      break;
    default:
      break;
  }
  return VK_FALSE;
}

inline auto checkValidationLayerSupport() noexcept -> bool {
  // get extension count
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  // get extension details
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  // for each possible name
  for (char const* layerName : validationLayerNames) {
    bool layerFound = false;
    // compare with every abailable layer name
    for (auto const& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }
    // layer not found
    if (!layerFound) {
      return false;
    }
  }
  // find validation layer
  return true;
}

#define VK_KHR_WIN32_SURFACE_EXTENSION_NAME "VK_KHR_win32_surface"

auto getRequiredExtensions(Context_VK* context,
                           ContextExtensionsFlags& ext) noexcept
    -> std::vector<const char*> {
  // extensions needed
  std::vector<const char*> extensions;
  // add glfw extension
  if (context->getBindedWindow()->getVendor() == Platform::WindowVendor::GLFW) {
    uint32_t glfwExtensionCount = 0;
    char const** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> glfwExtensionNames(
        glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.insert(extensions.end(), glfwExtensionNames.begin(),
                      glfwExtensionNames.end());
  }
  // add extensions that glfw needs
  else if (context->getBindedWindow()->getVendor() ==
           Platform::WindowVendor::WIN_64) {
    extensions.emplace_back(VK_KHR_SURFACE_EXTENSION_NAME);
    extensions.emplace_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
  }
  // add other extensions according to ext bits
  if (ext & (ContextExtensionsFlags)ContextExtension::MESH_SHADER) {
    extensions.emplace_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  }
  // add extensions that validation layer needs
  if (enableValidationLayers) {
    ext = ContextExtensionsFlags((uint32_t)ext |
                                 (uint32_t)ContextExtension::DEBUG_UTILS);
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  // finialize collection
  return extensions;
}

void populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
}

auto createInstance(Context_VK* context, ContextExtensionsFlags& ext) noexcept
    -> void {
  // Check we could enable validation layers
  if (enableValidationLayers && !checkValidationLayerSupport())
    Core::LogManager::Error(
        "Vulkan :: validation layers requested, but not available!");
  // Optional, but it may provide some useful information to
  // the driver in order to optimize our specific application
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "SIByLEngine";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 2, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 2, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;
  // Not optional, specify the desired global extensions
  auto extensions = getRequiredExtensions(context, ext);
  // Not optional,
  // Tells the Vulkan driver which global extensions and validation layers we
  // want to use. Global here means that they apply to the entire program and
  // not a specific device,
  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();
  // determine the global validation layers to enable
  void const** tail = &createInfo.pNext;
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayerNames.size());
    createInfo.ppEnabledLayerNames = validationLayerNames.data();
    // add debug messenger for init
    populateDebugMessengerCreateInfo(debugCreateInfo);
    *tail = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    tail = &((VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo)->pNext;
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.pNext = nullptr;
  }
  VkValidationFeaturesEXT validationInfo = {};
  VkValidationFeatureEnableEXT validationFeatureToEnable =
      VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;
  if (ext & (uint32_t)ContextExtension::SHADER_NON_SEMANTIC_INFO) {
    validationInfo.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validationInfo.enabledValidationFeatureCount = 1;
    validationInfo.pEnabledValidationFeatures = &validationFeatureToEnable;
    //_putenv_s("DEBUG_PRINTF_TO_STDOUT", "1");
    *tail = &validationInfo;
    tail = &(validationInfo.pNext);
  }
  // Create Vk Instance
  if (vkCreateInstance(&createInfo, nullptr, &(context->getVkInstance())) !=
      VK_SUCCESS) {
    Core::LogManager::Error("Vulkan :: Failed to create instance!");
  }
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, VkDebugUtilsMessengerCreateInfoEXT const* pCreateInfo,
    VkAllocationCallbacks const* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

auto setupDebugMessenger(Context_VK* context) noexcept -> void {
  if (!enableValidationLayers) return;
  VkDebugUtilsMessengerCreateInfoEXT createInfo;
  populateDebugMessengerCreateInfo(createInfo);
  // load function from extern
  if (CreateDebugUtilsMessengerEXT(context->getVkInstance(), &createInfo,
                                   nullptr,
                                   &context->getDebugMessenger()) != VK_SUCCESS)
    Core::LogManager::Error("Vulkan :: failed to set up debug messenger!");
}

PFN_vkVoidFunction vkGetInstanceProcAddrStub(void* context, const char* name) {
  return vkGetInstanceProcAddr((VkInstance)context, name);
}

auto setupExtensions(Context_VK* context, ContextExtensionsFlags& ext) -> void {
  if (ext & (ContextExtensionsFlags)ContextExtension::DEBUG_UTILS) {
    context->vkCmdBeginDebugUtilsLabelEXT =
        (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdBeginDebugUtilsLabelEXT");
    context->vkCmdEndDebugUtilsLabelEXT =
        (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdEndDebugUtilsLabelEXT");
    context->vkSetDebugUtilsObjectNameEXT =
        (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkSetDebugUtilsObjectNameEXT");
    context->vkSetDebugUtilsObjectTagEXT =
        (PFN_vkSetDebugUtilsObjectTagEXT)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkSetDebugUtilsObjectTagEXT");
  }
  if (ext & (ContextExtensionsFlags)ContextExtension::MESH_SHADER) {
    context->vkCmdDrawMeshTasksNV =
        (PFN_vkCmdDrawMeshTasksNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdDrawMeshTasksNV");
    context->getDeviceExtensions().emplace_back(
        VK_NV_MESH_SHADER_EXTENSION_NAME);
  }
  if (ext & (ContextExtensionsFlags)ContextExtension::RAY_TRACING) {
    context->vkCmdTraceRaysKHR =
        (PFN_vkCmdTraceRaysKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdTraceRaysKHR");
    context->vkCreateRayTracingPipelinesKHR =
        (PFN_vkCreateRayTracingPipelinesKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCreateRayTracingPipelinesKHR");
    context->vkGetRayTracingCaptureReplayShaderGroupHandlesKHR =
        (PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)
            vkGetInstanceProcAddrStub(
                context->getVkInstance(),
                "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR");
    context->vkCmdTraceRaysIndirectKHR =
        (PFN_vkCmdTraceRaysIndirectKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdTraceRaysIndirectKHR");
    context->vkGetRayTracingShaderGroupStackSizeKHR =
        (PFN_vkGetRayTracingShaderGroupStackSizeKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkGetRayTracingShaderGroupStackSizeKHR");
    context->vkCmdSetRayTracingPipelineStackSizeKHR =
        (PFN_vkCmdSetRayTracingPipelineStackSizeKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdSetRayTracingPipelineStackSizeKHR");
    context->vkCreateAccelerationStructureNV =
        (PFN_vkCreateAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCreateAccelerationStructureNV");
    context->vkDestroyAccelerationStructureNV =
        (PFN_vkDestroyAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkDestroyAccelerationStructureNV");
    context->vkGetAccelerationStructureMemoryRequirementsNV =
        (PFN_vkGetAccelerationStructureMemoryRequirementsNV)
            vkGetInstanceProcAddrStub(
                context->getVkInstance(),
                "vkGetAccelerationStructureMemoryRequirementsNV");
    context->vkBindAccelerationStructureMemoryNV =
        (PFN_vkBindAccelerationStructureMemoryNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkBindAccelerationStructureMemoryNV");
    context->vkCmdBuildAccelerationStructureNV =
        (PFN_vkCmdBuildAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdBuildAccelerationStructureNV");
    context->vkCmdCopyAccelerationStructureNV =
        (PFN_vkCmdCopyAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdCopyAccelerationStructureNV");
    context->vkCmdTraceRaysNV = (PFN_vkCmdTraceRaysNV)vkGetInstanceProcAddrStub(
        context->getVkInstance(), "vkCmdTraceRaysNV");
    context->vkCreateRayTracingPipelinesNV =
        (PFN_vkCreateRayTracingPipelinesNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCreateRayTracingPipelinesNV");
    context->vkGetRayTracingShaderGroupHandlesKHR =
        (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkGetRayTracingShaderGroupHandlesKHR");
    context->vkGetRayTracingShaderGroupHandlesNV =
        (PFN_vkGetRayTracingShaderGroupHandlesNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkGetRayTracingShaderGroupHandlesNV");
    context->vkGetAccelerationStructureHandleNV =
        (PFN_vkGetAccelerationStructureHandleNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkGetAccelerationStructureHandleNV");
    context->vkCmdWriteAccelerationStructuresPropertiesNV =
        (PFN_vkCmdWriteAccelerationStructuresPropertiesNV)
            vkGetInstanceProcAddrStub(
                context->getVkInstance(),
                "vkCmdWriteAccelerationStructuresPropertiesNV");
    context->vkCompileDeferredNV =
        (PFN_vkCompileDeferredNV)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCompileDeferredNV");
    context->vkGetAccelerationStructureBuildSizesKHR =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(),
            "vkGetAccelerationStructureBuildSizesKHR");
    context->vkCmdBuildAccelerationStructuresKHR =
        (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdBuildAccelerationStructuresKHR");
    context->vkCreateAccelerationStructureKHR =
        (PFN_vkCreateAccelerationStructureKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCreateAccelerationStructureKHR");
    context->vkDestroyAccelerationStructureKHR =
        (PFN_vkDestroyAccelerationStructureKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkDestroyAccelerationStructureKHR");
    context->vkGetAccelerationStructureDeviceAddressKHR =
        (PFN_vkGetAccelerationStructureDeviceAddressKHR)
            vkGetInstanceProcAddrStub(
                context->getVkInstance(),
                "vkGetAccelerationStructureDeviceAddressKHR");
    context->vkCmdCopyAccelerationStructureKHR =
        (PFN_vkCmdCopyAccelerationStructureKHR)vkGetInstanceProcAddrStub(
            context->getVkInstance(), "vkCmdCopyAccelerationStructureKHR");

    // emplace back device extensions
    context->getDeviceExtensions().emplace_back(
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    context->getDeviceExtensions().emplace_back(
        VK_KHR_RAY_QUERY_EXTENSION_NAME);
  }
  if (ext &
      (ContextExtensionsFlags)ContextExtension::SHADER_NON_SEMANTIC_INFO) {
    context->getDeviceExtensions().emplace_back(
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  }
  if (ext & (ContextExtensionsFlags)ContextExtension::ATOMIC_FLOAT) {
    context->getDeviceExtensions().emplace_back(
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
  }
  if (ext & (ContextExtensionsFlags)ContextExtension::FRAGMENT_BARYCENTRIC) {
    context->getDeviceExtensions().emplace_back(
        VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
  }
}

auto attachWindow(Context_VK* contexVk) noexcept -> void {
  if (contexVk->getBindedWindow()->getVendor() ==
      Platform::WindowVendor::GLFW) {
    if (glfwCreateWindowSurface(
            contexVk->getVkInstance(),
            (GLFWwindow*)contexVk->getBindedWindow()->getHandle(), nullptr,
            &contexVk->getVkSurfaceKHR()) != VK_SUCCESS) {
      Core::LogManager::Error("Vulkan :: glfwCreateWindowSurface failed!");
    }
  } else if (contexVk->getBindedWindow()->getVendor() ==
             Platform::WindowVendor::WIN_64) {
    VkWin32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hwnd = (HWND)contexVk->getBindedWindow()->getHandle();
    createInfo.hinstance = GetModuleHandle(nullptr);
    if (vkCreateWin32SurfaceKHR(contexVk->getVkInstance(), &createInfo, nullptr,
                                &contexVk->getVkSurfaceKHR()) != VK_SUCCESS) {
      Core::LogManager::Error(
          "Vulkan :: failed to create WIN_64 window surface!");
    }
  }
}

auto Context_VK::init(Platform::Window* window,
                      ContextExtensionsFlags ext) noexcept -> bool {
  bindedWindow = window;
  // create VkInstance
  createInstance(this, ext);
  setupDebugMessenger(this);
  setupExtensions(this, ext);
  attachWindow(this);
  // set extensions
  extensions = ext;
  return true;
}

auto findQueueFamilies(Context_VK* contextVk, VkPhysicalDevice& device) noexcept
    -> QueueFamilyIndices_VK {
  QueueFamilyIndices_VK indices;
  // Logic to find queue family indices to populate struct with
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());
  // find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
  int i = 0;
  for (auto const& queueFamily : queueFamilies) {
    // check graphic support
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
      if (queueFamily.timestampValidBits <= 0)
        Core::LogManager::Error(
            "VULKAN :: Graphics Family not support timestamp ValidBits");
    }
    // check queue support
    if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      indices.computeFamily = i;
    }
    // check present support
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(
        device, i, contextVk->getVkSurfaceKHR(), &presentSupport);
    if (presentSupport) indices.presentFamily = i;
    // check support completeness
    if (indices.isComplete()) break;
    i++;
  }
  return indices;
}

auto checkDeviceExtensionSupport(Context_VK* contextVk,
                                 VkPhysicalDevice& device,
                                 std::string& device_diagnosis) noexcept
    -> bool {
  // find all available extensions
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());
  // find all required extensions
  std::set<std::string> requiredExtensions(
      contextVk->getDeviceExtensions().begin(),
      contextVk->getDeviceExtensions().end());
  // find all required-but-not-available extensions
  for (const auto& extension : availableExtensions)
    requiredExtensions.erase(extension.extensionName);
  // create device diagnosis error codes if there are invalid requirement
  if (!requiredExtensions.empty()) {
    device_diagnosis.append("Required Extension not supported: ");
    for (const auto& extension : requiredExtensions) {
      device_diagnosis.append(extension);
      device_diagnosis.append(" | ");
    }
  }
  return requiredExtensions.empty();
}

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

auto querySwapChainSupport(Context_VK* contextVk, VkPhysicalDevice& device)
    -> SwapChainSupportDetails {
  SwapChainSupportDetails details;
  // query basic surface capabilities
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      device, contextVk->getVkSurfaceKHR(), &details.capabilities);
  // query the supported surface formats
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, contextVk->getVkSurfaceKHR(),
                                       &formatCount, nullptr);
  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, contextVk->getVkSurfaceKHR(),
                                         &formatCount, details.formats.data());
  }
  // query the supported presentation modes
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device, contextVk->getVkSurfaceKHR(), &presentModeCount, nullptr);
  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, contextVk->getVkSurfaceKHR(), &presentModeCount,
        details.presentModes.data());
  }
  return details;
}

auto isDeviceSuitable(Context_VK* contextVk, VkPhysicalDevice& device,
                      std::string& device_diagnosis) noexcept -> bool {
  // check queue family supports
  QueueFamilyIndices_VK indices = findQueueFamilies(contextVk, device);
  // check extension supports
  bool const extensionsSupported =
      checkDeviceExtensionSupport(contextVk, device, device_diagnosis);
  // check swapchain support
  bool swapChainAdequate = false;
  if (extensionsSupported) {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(contextVk, device);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }
  // check physical device feature supported
  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
  bool physicalDeviceFeatureSupported = supportedFeatures.samplerAnisotropy;
  return indices.isComplete() && extensionsSupported && swapChainAdequate &&
         physicalDeviceFeatureSupported;
}

auto rateDeviceSuitability(VkPhysicalDevice& device) noexcept -> int {
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
  int score = 0;
  // Discrete GPUs have a significant performance advantage
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000;
  }
  // Maximum possible size of textures affects graphics quality
  score += deviceProperties.limits.maxImageDimension2D;
  // Application can't function without geometry shaders
  if (!deviceFeatures.geometryShader) return 0;
  return score;
}

auto queryAllPhysicalDevice(Context_VK* contextVk) noexcept -> void {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(contextVk->getVkInstance(), &deviceCount, nullptr);
  // If there are 0 devices with Vulkan support
  if (deviceCount == 0)
    Core::LogManager::Error(
        "VULKAN :: Failed to find GPUs with Vulkan support!");
  // get all of the VkPhysicalDevice handles
  std::vector<VkPhysicalDevice>& devices = contextVk->getVkPhysicalDevices();
  devices.resize(deviceCount);
  vkEnumeratePhysicalDevices(contextVk->getVkInstance(), &deviceCount,
                             devices.data());
  // check if any of the physical devices meet the requirements
  int i = 0;
  for (const auto& device : devices) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    Core::LogManager::Log(
        std::format("VULKAN :: Physical Device [{}] Found, {}", i,
                    deviceProperties.deviceName));
    ++i;
  }
  // Find the best
  std::vector<std::string> diagnosis;
  std::vector<int> scores;
  for (auto& device : devices) {
    std::string device_diagnosis;
    if (isDeviceSuitable(contextVk, device, device_diagnosis)) {
      int rate = rateDeviceSuitability(device);
      scores.emplace_back(rate);
    } else {
      diagnosis.emplace_back(std::move(device_diagnosis));
      scores.emplace_back(0);
    }
  }
  for (int i = 0; i < devices.size(); ++i)
    for (int j = i + 1; j < devices.size(); ++j) {
      if (scores[i] < scores[j]) {
        std::swap(scores[i], scores[j]);
        std::swap(devices[i], devices[j]);
      }
    }
}

auto Context_VK::requestAdapter(RequestAdapterOptions const& options) noexcept
    -> std::unique_ptr<Adapter> {
  if (devices.size() == 0) queryAllPhysicalDevice(this);

  if (devices.size() == 0)
    return nullptr;
  else {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(devices[0], &deviceProperties);
    Core::LogManager::Debug(std::format("VULKAN :: Adapter selected, Name: {}",
                                        deviceProperties.deviceName));
    return std::make_unique<Adapter_VK>(devices[0], this, deviceProperties);
  }
}

auto Context_VK::getBindedWindow() const noexcept -> Platform::Window* {
  return bindedWindow;
}

auto inline destroyDebugUtilsMessengerEXT(
    VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
    VkAllocationCallbacks const* pAllocator) noexcept -> void {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) func(instance, debugMessenger, pAllocator);
}

auto Context_VK::destroy() noexcept -> void {
  if (enableValidationLayers)
    destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
}

#pragma endregion

#pragma region VK_ADAPTER_IMPL

Adapter_VK::Adapter_VK(VkPhysicalDevice device, Context_VK* context,
                       VkPhysicalDeviceProperties const& properties)
    : physicalDevice(device),
      context(context),
      adapterInfo([&]() -> AdapterInfo {
        AdapterInfo info;
        info.device = properties.deviceName;
        ;
        info.vendor = properties.vendorID;
        info.architecture = properties.deviceType;
        info.description = properties.deviceID;
        return info;
      }()),
      timestampPeriod(properties.limits.timestampPeriod),
      queueFamilyIndices(findQueueFamilies(context, physicalDevice)),
      properties(properties) {}

auto Adapter_VK::requestDevice() noexcept -> std::unique_ptr<Device> {
  // get queues
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {
      queueFamilyIndices.graphicsFamily.value(),
      queueFamilyIndices.presentFamily.value(),
      queueFamilyIndices.computeFamily.value()};
  // Desc VkDeviceQueueCreateInfo
  VkDeviceQueueCreateInfo queueCreateInfo{};
  // the number of queues we want for a single queue family
  float queuePriority = 1.0f;  // a queue with graphics capabilities
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }
  // Desc Vk Device Create Info
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pEnabledFeatures = nullptr;
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  // enable extesions
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(getDeviceExtensions().size());
  createInfo.ppEnabledExtensionNames = getDeviceExtensions().data();
  createInfo.pNext = nullptr;
  // get all physical device features chain
  void const** pNextChainHead = &(createInfo.pNext);
  void** pNextChainTail = nullptr;
  VkPhysicalDeviceHostQueryResetFeatures resetFeatures;
  // Add various features
  VkPhysicalDeviceMeshShaderFeaturesNV mesh_shader_feature{};
  if (context->getContextExtensionsFlags() &
      (ContextExtensionsFlags)ContextExtension::MESH_SHADER) {
    mesh_shader_feature.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV;
    mesh_shader_feature.pNext = nullptr;
    mesh_shader_feature.taskShader = VK_TRUE;
    mesh_shader_feature.meshShader = VK_TRUE;
    if (pNextChainTail == nullptr)
      *pNextChainHead = &mesh_shader_feature;
    else
      *pNextChainTail = &mesh_shader_feature;
    pNextChainTail = &(mesh_shader_feature.pNext);
  }
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV
      shader_fragment_barycentric{};
  if (context->getContextExtensionsFlags() &
      (ContextExtensionsFlags)ContextExtension::FRAGMENT_BARYCENTRIC) {
    shader_fragment_barycentric.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV;
    shader_fragment_barycentric.pNext = nullptr;
    shader_fragment_barycentric.fragmentShaderBarycentric = VK_TRUE;
    if (pNextChainTail == nullptr)
      *pNextChainHead = &shader_fragment_barycentric;
    else
      *pNextChainTail = &shader_fragment_barycentric;
    pNextChainTail = &(shader_fragment_barycentric.pNext);
  }
  VkPhysicalDeviceVulkan12Features sampler_filter_min_max_properties{};
  if (context->getContextExtensionsFlags() &
      (ContextExtensionsFlags)ContextExtension::SAMPLER_FILTER_MIN_MAX) {
    sampler_filter_min_max_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    sampler_filter_min_max_properties.pNext = nullptr;
    sampler_filter_min_max_properties.samplerFilterMinmax = VK_TRUE;
    sampler_filter_min_max_properties.shaderInt8 = VK_TRUE;
    sampler_filter_min_max_properties.hostQueryReset = VK_TRUE;
    if (pNextChainTail == nullptr)
      *pNextChainHead = &sampler_filter_min_max_properties;
    else
      *pNextChainTail = &sampler_filter_min_max_properties;
    pNextChainTail = &(sampler_filter_min_max_properties.pNext);
  }
  VkPhysicalDeviceFeatures2 features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  void** pFeature2Tail = &(features2.pNext);
  // sub : bindless
  VkPhysicalDeviceDescriptorIndexingFeatures indexing_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
  // sub : ray tracing
  VkPhysicalDeviceVulkan12Features features12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan11Features features11{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shader_atomic_float{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  if (context->getContextExtensionsFlags() &
      (ContextExtensionsFlags)ContextExtension::BINDLESS_INDEXING) {
    if (context->getContextExtensionsFlags() &
        (ContextExtensionsFlags)ContextExtension::RAY_TRACING) {
      features12.descriptorIndexing = VK_TRUE;
      features12.descriptorBindingPartiallyBound = VK_TRUE;
      features12.runtimeDescriptorArray = VK_TRUE;
    } else {
      *pFeature2Tail = &indexing_features;
      pFeature2Tail = &(indexing_features.pNext);
      ;
      indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
      indexing_features.runtimeDescriptorArray = VK_TRUE;
    }
  }
  if (context->getContextExtensionsFlags() &
      (ContextExtensionsFlags)ContextExtension::RAY_TRACING) {
    *pFeature2Tail = &rayQueryFeatures;
    rayQueryFeatures.pNext = &features12;
    features12.pNext = &features11;
    features11.pNext = &asFeatures;
    asFeatures.pNext = &rtPipelineFeatures;
    pFeature2Tail = &(rtPipelineFeatures.pNext);
    ;
  }
  if (context->getContextExtensionsFlags() &
      (ContextExtensionsFlags)ContextExtension::ATOMIC_FLOAT) {
    shader_atomic_float.shaderBufferFloat32AtomicAdd =
        true;  // this allows to perform atomic operations on storage buffers
    shader_atomic_float.shaderBufferFloat32Atomics = true;
    shader_atomic_float.pNext = nullptr;

    *pFeature2Tail = &shader_atomic_float;
    pFeature2Tail = &(shader_atomic_float.pNext);
    ;
  }
  vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
  if (pNextChainTail == nullptr)
    *pNextChainHead = &features2;
  else
    *pNextChainTail = &features2;
  pNextChainTail = pFeature2Tail;

  // create logical device
  std::unique_ptr<Device_VK> device = std::make_unique<Device_VK>();
  device->getAdapterVk() = this;
  if (vkCreateDevice(getVkPhysicalDevice(), &createInfo, nullptr,
                     &device->getVkDevice()) != VK_SUCCESS) {
    Core::LogManager::Log("VULKAN :: failed to create logical device!");
  }
  // get queue handlevul
  vkGetDeviceQueue(device->getVkDevice(),
                   queueFamilyIndices.graphicsFamily.value(), 0,
                   &device->getVkGraphicsQueue().queue);
  vkGetDeviceQueue(device->getVkDevice(),
                   queueFamilyIndices.presentFamily.value(), 0,
                   &device->getVkPresentQueue().queue);
  vkGetDeviceQueue(device->getVkDevice(),
                   queueFamilyIndices.computeFamily.value(), 0,
                   &device->getVkComputeQueue().queue);
  device->getVkGraphicsQueue().device = device.get();
  device->getVkPresentQueue().device = device.get();
  device->getVkComputeQueue().device = device.get();
  device->createCommandPools();
  device->createBindGroupPool();
  device->init();
  return std::move(device);
}

auto Adapter_VK::requestAdapterInfo() const noexcept -> AdapterInfo {
  return adapterInfo;
}

auto Adapter_VK::findMemoryType(uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) noexcept
    -> uint32_t {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  Core::LogManager::Log("VULKAN :: failed to find suitable memory type!");
}

#pragma endregion

#pragma region VK_DEVICE_IMPL

Device_VK::~Device_VK() { destroy(); }

auto Device_VK::destroy() noexcept -> void {
  graphicPool = nullptr, computePool = nullptr, presentPool = nullptr;
  bindGroupPool = nullptr;
#ifdef USE_VMA
  vmaDestroyAllocator(allocator);
#endif
  if (device) vkDestroyDevice(device, nullptr);
}

auto Device_VK::waitIdle() noexcept -> void {
  VkResult result = vkDeviceWaitIdle(device);
  if (result != VK_SUCCESS)
    Core::LogManager::Error("VULKAN :: Device WaitIdle not Success!");
}

inline auto getVkBufferUsageFlags(BufferUsagesFlags usage) noexcept
    -> VkBufferUsageFlags {
  VkBufferUsageFlags flags = 0;
  if (usage & (uint32_t)BufferUsage::COPY_SRC)
    flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (usage & (uint32_t)BufferUsage::COPY_DST)
    flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (usage & (uint32_t)BufferUsage::INDEX)
    flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if (usage & (uint32_t)BufferUsage::VERTEX)
    flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (usage & (uint32_t)BufferUsage::UNIFORM)
    flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (usage & (uint32_t)BufferUsage::STORAGE)
    flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (usage & (uint32_t)BufferUsage::INDIRECT)
    flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (usage & (uint32_t)BufferUsage::QUERY_RESOLVE)
    flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (usage & (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS)
    flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if (usage & (uint32_t)BufferUsage::ACCELERATION_STRUCTURE_STORAGE)
    flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
  if (usage &
      (uint32_t)BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
    flags |=
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  if (usage & (uint32_t)BufferUsage::SHADER_BINDING_TABLE)
    flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  return flags;
}

auto Device_VK::importExternalTexture(
    ExternalTextureDescriptor const& desc) noexcept
    -> std::unique_ptr<ExternalTexture> {
  return nullptr;
}

auto Device_VK::createComputePipelineAsync(
    ComputePipelineDescriptor const& desc) noexcept
    -> std::future<std::unique_ptr<ComputePipeline>> {
  std::future<std::unique_ptr<ComputePipeline>> r;
  return r;
}

auto Device_VK::createRenderPipelineAsync(
    RenderPipelineDescriptor const& desc) noexcept
    -> std::future<std::unique_ptr<RenderPipeline>> {
  std::future<std::unique_ptr<RenderPipeline>> r;
  return r;
}

auto Device_VK::createRenderBundleEncoder(
    CommandEncoderDescriptor const& desc) noexcept
    -> std::unique_ptr<RenderBundleEncoder> {
  return nullptr;
}

auto Device_VK::createQuerySet(QuerySetDescriptor const& desc) noexcept
    -> std::unique_ptr<QuerySet> {
  return nullptr;
}

auto Device_VK::init() noexcept -> void {
  // initialize allocator
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
  allocatorInfo.physicalDevice = adapter->getVkPhysicalDevice();
  allocatorInfo.device = device;
  allocatorInfo.instance = adapter->getContext()->getVkInstance();
  allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&allocatorInfo, &allocator);
  // initialize extensions
  if (static_cast<Context_VK*>(adapter->getContext())
          ->getContextExtensionsFlags() &
      (uint32_t)RHI::ContextExtension::RAY_TRACING)
    initRayTracingExt();
}

auto Device_VK::createCommandPools() noexcept -> void {
  graphicPool = std::make_unique<CommandPool_VK>(this);
}

auto Queue_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_QUEUE;
  objectNameInfo.objectHandle = uint64_t(queue);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
}

#pragma endregion

#pragma region VK_BUFFER_IMPL

#define MACRO_BUFFER_USAGE_BITMAP(USAGE)                                    \
  if ((uint32_t)usage & (uint32_t)SIByL::RHI::BufferUsageFlagBits::USAGE) { \
    flags |= VK_BUFFER_USAGE_##USAGE;                                       \
  }

inline auto getVkBufferShareMode(BufferShareMode shareMode) noexcept
    -> VkSharingMode {
  if (shareMode == SIByL::RHI::BufferShareMode::CONCURRENT)
    return VK_SHARING_MODE_CONCURRENT;
  else if (shareMode == SIByL::RHI::BufferShareMode::EXCLUSIVE)
    return VK_SHARING_MODE_EXCLUSIVE;
  else
    return VK_SHARING_MODE_MAX_ENUM;
}

inline auto getVkMemoryProperty(MemoryPropertiesFlags memoryProperties) noexcept
    -> VkMemoryPropertyFlags {
  uint32_t flags{};
  if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT)
    flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::HOST_VISIBLE_BIT)
    flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::HOST_COHERENT_BIT)
    flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::HOST_CACHED_BIT)
    flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  if ((uint32_t)memoryProperties &
      (uint32_t)MemoryProperty::LAZILY_ALLOCATED_BIT)
    flags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
  if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::PROTECTED_BIT)
    flags |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
  return (VkMemoryPropertyFlags)flags;
}

Buffer_VK::~Buffer_VK() { destroy(); }

Buffer_VK::Buffer_VK(Buffer_VK&& x)
    : buffer(x.buffer), bufferMemory(x.bufferMemory), _size(x._size) {
  x.buffer = nullptr;
  x.bufferMemory = nullptr;
}

auto Buffer_VK::operator=(Buffer_VK&& x) -> Buffer_VK& {
  buffer = x.buffer;
  bufferMemory = x.bufferMemory;
  _size = x._size;
  x.buffer = nullptr;
  x.bufferMemory = nullptr;
  return *this;
}

auto Buffer_VK::init(Device_VK* device, size_t size,
                     BufferDescriptor const& desc) noexcept -> void {
  this->_size = size;
  this->device = device;
  // createBuffer(
  //	_size,
  //	getVkBufferUsage(desc.usage),
  //	getVkBufferShareMode(desc.shareMode),
  //	getVkMemoryProperty(desc.memoryProperties),
  //	buffer,
  //	bufferMemory,
  //	device
  //);
}

inline auto mapMemory(Device_VK* device, Buffer_VK* buffer, size_t offset,
                      size_t size, void*& mappedData) noexcept -> bool {
#ifdef USE_VMA
  VkResult result = vmaMapMemory(device->getVMAAllocator(),
                                 buffer->getVMAAllocation(), &mappedData);
#else
  VkResult result =
      vkMapMemory(device->getVkDevice(), buffer->getVkDeviceMemory(), offset,
                  size, 0, &mappedData);
#endif
  if (result) buffer->setBufferMapState(BufferMapState::MAPPED);
  return result == VkResult::VK_SUCCESS ? true : false;
}

auto Buffer_VK::mapAsync(MapModeFlags mode, size_t offset, size_t size) noexcept
    -> std::future<bool> {
  mapState = BufferMapState::PENDING;
  return std::async(mapMemory, device, this, offset, size,
                    std::ref(mappedData));
}

auto Buffer_VK::getMappedRange(size_t offset, size_t size) noexcept
    -> ArrayBuffer {
  return (void*)&(((char*)mappedData)[offset]);
}

auto Buffer_VK::unmap() noexcept -> void {
#ifdef USE_VMA
  vmaUnmapMemory(device->getVMAAllocator(), getVMAAllocation());
#else
  vkUnmapMemory(device->getVkDevice(), bufferMemory);
#endif
  mappedData = nullptr;
  BufferMapState mapState = BufferMapState::UNMAPPED;
}

auto Buffer_VK::destroy() noexcept -> void {
#ifdef USE_VMA
  if (buffer)
    vmaDestroyBuffer(device->getVMAAllocator(), buffer, getVMAAllocation());
#else
  if (buffer) vkDestroyBuffer(device->getVkDevice(), buffer, nullptr);
  if (bufferMemory) vkFreeMemory(device->getVkDevice(), bufferMemory, nullptr);
#endif
}

auto Buffer_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
  objectNameInfo.objectHandle = uint64_t(buffer);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
  this->name = name;
}

auto Buffer_VK::getName() const noexcept -> std::string const& { return name; }

inline auto findMemoryType(Device_VK* device, uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) noexcept
    -> uint32_t {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(
      device->getAdapterVk()->getVkPhysicalDevice(), &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  Core::LogManager::Error("VULKAN :: failed to find suitable memory type!");
  return 0;
}

inline auto getVkMemoryPropertyFlags(
    MemoryPropertiesFlags memoryProperties) noexcept -> VkMemoryPropertyFlags {
  VkMemoryPropertyFlags flags = 0;
  if (memoryProperties & (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT)
    flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (memoryProperties & (uint32_t)MemoryProperty::HOST_VISIBLE_BIT)
    flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  if (memoryProperties & (uint32_t)MemoryProperty::HOST_COHERENT_BIT)
    flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (memoryProperties & (uint32_t)MemoryProperty::HOST_CACHED_BIT)
    flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  if (memoryProperties & (uint32_t)MemoryProperty::LAZILY_ALLOCATED_BIT)
    flags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
  if (memoryProperties & (uint32_t)MemoryProperty::PROTECTED_BIT)
    flags |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
  if (memoryProperties == (uint32_t)MemoryProperty::FLAG_BITS_MAX_ENUM)
    flags |= VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM;
  return flags;
}

auto Device_VK::createBuffer(BufferDescriptor const& desc) noexcept
    -> std::unique_ptr<Buffer> {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = desc.size;
  bufferInfo.usage = getVkBufferUsageFlags(desc.usage);
  bufferInfo.sharingMode = desc.shareMode == BufferShareMode::EXCLUSIVE
                               ? VK_SHARING_MODE_EXCLUSIVE
                               : VK_SHARING_MODE_CONCURRENT;
  std::unique_ptr<Buffer_VK> buffer = std::make_unique<Buffer_VK>(this);
  buffer->init(this, desc.size, desc);
#ifdef USE_VMA
  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  if (desc.memoryProperties & (uint32_t)MemoryProperty::HOST_VISIBLE_BIT ||
      desc.memoryProperties & (uint32_t)MemoryProperty::HOST_COHERENT_BIT) {
    allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                      &buffer->getVkBuffer(), &buffer->getVMAAllocation(),
                      nullptr) != VK_SUCCESS) {
    Core::LogManager::Log("VULKAN :: failed to create vertex buffer!");
  }
#else
  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer->getVkBuffer()) !=
      VK_SUCCESS) {
    Core::LogManager::Log("VULKAN :: failed to create vertex buffer!");
  }
  // assign memory to buffer
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer->getVkBuffer(),
                                &memRequirements);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(this, memRequirements.memoryTypeBits,
                     getVkMemoryPropertyFlags(desc.memoryProperties));
  VkMemoryAllocateFlagsInfo allocFlagsInfo = {};
  allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  if (desc.usage & (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS) {
    allocFlagsInfo.flags =
        VkMemoryAllocateFlagBits::VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &allocFlagsInfo;
  }
  if (vkAllocateMemory(device, &allocInfo, nullptr,
                       &buffer->getVkDeviceMemory()) != VK_SUCCESS) {
    Core::LogManager::Log("VULKAN :: failed to allocate vertex buffer memory!");
  }
  vkBindBufferMemory(device, buffer->getVkBuffer(), buffer->getVkDeviceMemory(),
                     0);
#endif
  return buffer;
}

#pragma endregion
  
#pragma region VK_TEXTURE_IMPL

inline auto getVkImageType(TextureDimension const& dim) noexcept
    -> VkImageType {
  switch (dim) {
    case TextureDimension::TEX1D:
      return VkImageType::VK_IMAGE_TYPE_1D;
    case TextureDimension::TEX2D:
      return VkImageType::VK_IMAGE_TYPE_2D;
    case TextureDimension::TEX3D:
      return VkImageType::VK_IMAGE_TYPE_3D;
    default:
      return VkImageType::VK_IMAGE_TYPE_MAX_ENUM;
  }
}

inline auto getVkFormat(TextureFormat format) noexcept -> VkFormat {
  switch (format) {
    case SIByL::RHI::TextureFormat::DEPTH32STENCIL8:
      return VK_FORMAT_D32_SFLOAT_S8_UINT;
      break;
    case SIByL::RHI::TextureFormat::DEPTH32_FLOAT:
      return VK_FORMAT_D32_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::DEPTH24STENCIL8:
      return VK_FORMAT_D24_UNORM_S8_UINT;
      break;
    case SIByL::RHI::TextureFormat::DEPTH24:
      return VK_FORMAT_X8_D24_UNORM_PACK32;
      break;
    case SIByL::RHI::TextureFormat::DEPTH16_UNORM:
      return VK_FORMAT_D16_UNORM;
      break;
    case SIByL::RHI::TextureFormat::STENCIL8:
      return VK_FORMAT_S8_UINT;
      break;
    case SIByL::RHI::TextureFormat::RGBA32_FLOAT:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::RGBA32_SINT:
      return VK_FORMAT_R32G32B32A32_SINT;
      break;
    case SIByL::RHI::TextureFormat::RGBA32_UINT:
      return VK_FORMAT_R32G32B32A32_UINT;
      break;
    case SIByL::RHI::TextureFormat::RGBA16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::RGBA16_SINT:
      return VK_FORMAT_R16G16B16A16_SINT;
      break;
    case SIByL::RHI::TextureFormat::RGBA16_UINT:
      return VK_FORMAT_R16G16B16A16_UINT;
      break;
    case SIByL::RHI::TextureFormat::RG32_FLOAT:
      return VK_FORMAT_R32G32_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::RG32_SINT:
      return VK_FORMAT_R32G32_SINT;
      break;
    case SIByL::RHI::TextureFormat::RG32_UINT:
      return VK_FORMAT_R32G32_UINT;
      break;
    case SIByL::RHI::TextureFormat::RG11B10_UFLOAT:
      return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
      break;
    case SIByL::RHI::TextureFormat::RGB10A2_UNORM:
      return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
      break;
    case SIByL::RHI::TextureFormat::RGB9E5_UFLOAT:
      return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
      break;
    case SIByL::RHI::TextureFormat::BGRA8_UNORM_SRGB:
      return VK_FORMAT_B8G8R8A8_SRGB;
      break;
    case SIByL::RHI::TextureFormat::BGRA8_UNORM:
      return VK_FORMAT_B8G8R8A8_UNORM;
      break;
    case SIByL::RHI::TextureFormat::RGBA8_SINT:
      return VK_FORMAT_R8G8B8A8_SINT;
      break;
    case SIByL::RHI::TextureFormat::RGBA8_UINT:
      return VK_FORMAT_R8G8B8A8_UINT;
      break;
    case SIByL::RHI::TextureFormat::RGBA8_SNORM:
      return VK_FORMAT_R8G8B8A8_SNORM;
      break;
    case SIByL::RHI::TextureFormat::RGBA8_UNORM_SRGB:
      return VK_FORMAT_R8G8B8A8_SRGB;
      break;
    case SIByL::RHI::TextureFormat::RGBA8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
      break;
    case SIByL::RHI::TextureFormat::RG16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::RG16_SINT:
      return VK_FORMAT_R16G16_SINT;
      break;
    case SIByL::RHI::TextureFormat::RG16_UINT:
      return VK_FORMAT_R16G16_UINT;
      break;
    case SIByL::RHI::TextureFormat::R32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::R32_SINT:
      return VK_FORMAT_R32_SINT;
      break;
    case SIByL::RHI::TextureFormat::R32_UINT:
      return VK_FORMAT_R32_UINT;
      break;
    case SIByL::RHI::TextureFormat::RG8_SINT:
      return VK_FORMAT_R8G8_SINT;
      break;
    case SIByL::RHI::TextureFormat::RG8_UINT:
      return VK_FORMAT_R8G8_UINT;
      break;
    case SIByL::RHI::TextureFormat::RG8_SNORM:
      return VK_FORMAT_R8G8_SNORM;
      break;
    case SIByL::RHI::TextureFormat::RG8_UNORM:
      return VK_FORMAT_R8G8_UNORM;
      break;
    case SIByL::RHI::TextureFormat::R16_FLOAT:
      return VK_FORMAT_R16_SFLOAT;
      break;
    case SIByL::RHI::TextureFormat::R16_SINT:
      return VK_FORMAT_R16_SINT;
      break;
    case SIByL::RHI::TextureFormat::R16_UINT:
      return VK_FORMAT_R16_UINT;
      break;
    case SIByL::RHI::TextureFormat::R8_SINT:
      return VK_FORMAT_R8_SINT;
      break;
    case SIByL::RHI::TextureFormat::R8_UINT:
      return VK_FORMAT_R8_UINT;
      break;
    case SIByL::RHI::TextureFormat::R8_SNORM:
      return VK_FORMAT_R8_SNORM;
      break;
    case SIByL::RHI::TextureFormat::R8_UNORM:
      return VK_FORMAT_R8_UNORM;
      break;
    case SIByL::RHI::TextureFormat::BC1_RGB_UNORM_BLOCK:
      return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC1_RGB_SRGB_BLOCK:
      return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC1_RGBA_UNORM_BLOCK:
      return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC1_RGBA_SRGB_BLOCK:
      return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC2_UNORM_BLOCK:
      return VK_FORMAT_BC2_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC2_SRGB_BLOCK:
      return VK_FORMAT_BC2_SRGB_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC3_UNORM_BLOCK:
      return VK_FORMAT_BC3_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC3_SRGB_BLOCK:
      return VK_FORMAT_BC3_SRGB_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC4_UNORM_BLOCK:
      return VK_FORMAT_BC4_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC4_SNORM_BLOCK:
      return VK_FORMAT_BC4_SNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC5_UNORM_BLOCK:
      return VK_FORMAT_BC5_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC5_SNORM_BLOCK:
      return VK_FORMAT_BC5_SNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC6H_UFLOAT_BLOCK:
      return VK_FORMAT_BC6H_UFLOAT_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC6H_SFLOAT_BLOCK:
      return VK_FORMAT_BC6H_SFLOAT_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC7_UNORM_BLOCK:
      return VK_FORMAT_BC7_UNORM_BLOCK;
      break;
    case SIByL::RHI::TextureFormat::BC7_SRGB_BLOCK:
      return VK_FORMAT_BC7_SRGB_BLOCK;
      break;
    default:
      return VK_FORMAT_UNDEFINED;
      break;
  }
}

inline auto getVkImageUsageFlagBits(TextureUsagesFlags flags) noexcept
    -> VkImageUsageFlags {
  VkImageUsageFlags usageFlags = 0;
  if (flags & (uint32_t)TextureUsage::COPY_SRC)
    usageFlags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (flags & (uint32_t)TextureUsage::COPY_DST)
    usageFlags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (flags & (uint32_t)TextureUsage::TEXTURE_BINDING)
    usageFlags |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (flags & (uint32_t)TextureUsage::STORAGE_BINDING)
    usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (flags & (uint32_t)TextureUsage::COLOR_ATTACHMENT)
    usageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (flags & (uint32_t)TextureUsage::DEPTH_ATTACHMENT)
    usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (flags & (uint32_t)TextureUsage::TRANSIENT_ATTACHMENT)
    usageFlags |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  if (flags & (uint32_t)TextureUsage::INPUT_ATTACHMENT)
    usageFlags |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
  return usageFlags;
}

inline auto getVkImageCreateFlags(TextureFlags descflags) noexcept
    -> VkImageCreateFlags {
  VkImageCreateFlags flags = 0;
  if (uint32_t(descflags) & (uint32_t)TextureFlags::CUBE_COMPATIBLE)
    flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  return flags;
}

Texture_VK::Texture_VK(Device_VK* device, TextureDescriptor const& desc)
    : device(device), descriptor{desc} {
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = getVkImageType(desc.dimension);
  imageInfo.extent.width = static_cast<uint32_t>(desc.size.width);
  imageInfo.extent.height = static_cast<uint32_t>(desc.size.height);
  imageInfo.extent.depth = (desc.dimension == TextureDimension::TEX2D)
                               ? 1
                               : desc.size.depthOrArrayLayers;
  imageInfo.mipLevels = desc.mipLevelCount;
  imageInfo.arrayLayers = (desc.dimension == TextureDimension::TEX2D)
                              ? desc.size.depthOrArrayLayers
                              : 1;
  imageInfo.format = getVkFormat(desc.format);
  imageInfo.tiling = hasBit(desc.flags, RHI::TextureFlags::HOSTI_VISIBLE)
                         ? VK_IMAGE_TILING_LINEAR
                         : VK_IMAGE_TILING_OPTIMAL;
  if (desc.format >= TextureFormat::COMPRESSION)
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = getVkImageUsageFlagBits(desc.usage);
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.flags = getVkImageCreateFlags(desc.flags);  // Optional

#ifdef USE_VMA
  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  if (hasBit(desc.flags, RHI::TextureFlags::HOSTI_VISIBLE)) {
    allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  }
  if (vmaCreateImage(device->getVMAAllocator(), &imageInfo, &allocInfo, &image,
                     &allocation, nullptr) != VK_SUCCESS) {
    Core::LogManager::Log("VULKAN :: failed to create vertex buffer!");
  }
#else
  if (vkCreateImage(device->getVkDevice(), &imageInfo, nullptr, &image) !=
      VK_SUCCESS) {
    Core::LogManager::Log("Vulkan :: failed to create image!");
  }
  // Allocating memory for an image
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device->getVkDevice(), image, &memRequirements);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (desc.hostVisible)
    memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  allocInfo.memoryTypeIndex =
      findMemoryType(device, memRequirements.memoryTypeBits, memoryProperties);
  if (vkAllocateMemory(device->getVkDevice(), &allocInfo, nullptr,
                       &deviceMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }
  vkBindImageMemory(device->getVkDevice(), image, deviceMemory, 0);
#endif
}

Texture_VK::Texture_VK(Device_VK* device, VkImage image,
                       TextureDescriptor const& desc)
    : device(device), image(image), descriptor{desc} {}

Texture_VK::~Texture_VK() { destroy(); }

auto Texture_VK::createView(TextureViewDescriptor const& desc) noexcept
    -> std::unique_ptr<TextureView> {
  return std::make_unique<TextureView_VK>(device, this, desc);
}

auto Texture_VK::destroy() noexcept -> void {
#ifdef USE_VMA
  if (image) vmaDestroyImage(device->getVMAAllocator(), image, allocation);
#else
  if (image && deviceMemory)
    vkDestroyImage(device->getVkDevice(), image, nullptr);
  if (deviceMemory) vkFreeMemory(device->getVkDevice(), deviceMemory, nullptr);
#endif
}

auto Texture_VK::width() const noexcept -> uint32_t {
  return descriptor.size.width;
}

auto Texture_VK::height() const noexcept -> uint32_t {
  return descriptor.size.height;
}

auto Texture_VK::depthOrArrayLayers() const noexcept -> uint32_t {
  return descriptor.size.depthOrArrayLayers;
}

auto Texture_VK::mipLevelCount() const noexcept -> uint32_t {
  return descriptor.mipLevelCount;
}

auto Texture_VK::sampleCount() const noexcept -> uint32_t {
  return descriptor.sampleCount;
}

auto Texture_VK::dimension() const noexcept -> TextureDimension {
  return descriptor.dimension;
}

auto Texture_VK::format() const noexcept -> TextureFormat {
  return descriptor.format;
}

inline auto mapMemoryTexture(Device_VK* device, Texture_VK* texture,
                             size_t offset, size_t size,
                             void*& mappedData) noexcept -> bool {
#ifdef USE_VMA
  VkResult result = vmaMapMemory(device->getVMAAllocator(),
                                 texture->getVMAAllocation(), &mappedData);
#else
  VkResult result =
      vkMapMemory(device->getVkDevice(), texture->getVkDeviceMemory(), offset,
                  size, 0, &mappedData);
#endif
  if (result) texture->setBufferMapState(BufferMapState::MAPPED);
  return result == VkResult::VK_SUCCESS ? true : false;
}

auto Texture_VK::mapAsync(MapModeFlags mode, size_t offset,
                          size_t size) noexcept -> std::future<bool> {
  mapState = BufferMapState::PENDING;
  return std::async(mapMemoryTexture, device, this, offset, size,
                    std::ref(mappedData));
}

auto Texture_VK::getMappedRange(size_t offset, size_t size) noexcept
    -> ArrayBuffer {
  return (void*)&(((char*)mappedData)[offset]);
}

auto Texture_VK::unmap() noexcept -> void {
#ifdef USE_VMA
  vmaUnmapMemory(device->getVMAAllocator(), getVMAAllocation());
#else
  vkUnmapMemory(device->getVkDevice(), getVkDeviceMemory());
#endif
  mappedData = nullptr;
  BufferMapState mapState = BufferMapState::UNMAPPED;
}

auto Texture_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_IMAGE;
  objectNameInfo.objectHandle = uint64_t(image);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
  this->name = name;
}

auto Texture_VK::getName() -> std::string const& { return name; }

auto Texture_VK::getDescriptor() -> TextureDescriptor { return descriptor; }

auto TextureView_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
  objectNameInfo.objectHandle = uint64_t(imageView);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
}

auto TextureView_VK::getWidth() noexcept -> uint32_t { return width; }

auto TextureView_VK::getHeight() noexcept -> uint32_t { return height; }

auto Device_VK::createTexture(TextureDescriptor const& desc) noexcept
    -> std::unique_ptr<Texture> {
  return std::make_unique<Texture_VK>(this, desc);
}

#pragma endregion

#pragma region VK_TEXTUREVIEW_IMPL

inline auto getTextureFormat(VkFormat format) noexcept -> TextureFormat {
  switch (format) {
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return SIByL::RHI::TextureFormat::DEPTH32STENCIL8;
    case VK_FORMAT_D32_SFLOAT:
      return SIByL::RHI::TextureFormat::DEPTH32_FLOAT;
    case VK_FORMAT_D24_UNORM_S8_UINT:
      return SIByL::RHI::TextureFormat::DEPTH24STENCIL8;
    case VK_FORMAT_X8_D24_UNORM_PACK32:
      return SIByL::RHI::TextureFormat::DEPTH24;
    case VK_FORMAT_D16_UNORM:
      return SIByL::RHI::TextureFormat::DEPTH16_UNORM;
    case VK_FORMAT_S8_UINT:
      return SIByL::RHI::TextureFormat::STENCIL8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return SIByL::RHI::TextureFormat::RGBA32_FLOAT;
    case VK_FORMAT_R32G32B32A32_SINT:
      return SIByL::RHI::TextureFormat::RGBA32_SINT;
    case VK_FORMAT_R32G32B32A32_UINT:
      return SIByL::RHI::TextureFormat::RGBA32_UINT;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return SIByL::RHI::TextureFormat::RGBA16_FLOAT;
    case VK_FORMAT_R16G16B16A16_SINT:
      return SIByL::RHI::TextureFormat::RGBA16_SINT;
    case VK_FORMAT_R16G16B16A16_UINT:
      return SIByL::RHI::TextureFormat::RGBA16_UINT;
    case VK_FORMAT_R32G32_SFLOAT:
      return SIByL::RHI::TextureFormat::RG32_FLOAT;
    case VK_FORMAT_R32G32_SINT:
      return SIByL::RHI::TextureFormat::RG32_SINT;
    case VK_FORMAT_R32G32_UINT:
      return SIByL::RHI::TextureFormat::RG32_UINT;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return SIByL::RHI::TextureFormat::RG11B10_UFLOAT;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return SIByL::RHI::TextureFormat::RGB10A2_UNORM;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
      return SIByL::RHI::TextureFormat::RGB9E5_UFLOAT;
    case VK_FORMAT_B8G8R8A8_SRGB:
      return SIByL::RHI::TextureFormat::BGRA8_UNORM_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM:
      return SIByL::RHI::TextureFormat::BGRA8_UNORM;
    case VK_FORMAT_B8G8R8A8_SINT:
      return SIByL::RHI::TextureFormat::RGBA8_SINT;
    case VK_FORMAT_B8G8R8A8_UINT:
      return SIByL::RHI::TextureFormat::RGBA8_UINT;
    case VK_FORMAT_R8G8B8A8_SNORM:
      return SIByL::RHI::TextureFormat::RGBA8_SNORM;
    case VK_FORMAT_R8G8B8A8_SRGB:
      return SIByL::RHI::TextureFormat::RGBA8_UNORM_SRGB;
    case VK_FORMAT_R8G8B8A8_UNORM:
      return SIByL::RHI::TextureFormat::RGBA8_UNORM;
    case VK_FORMAT_R16G16_SFLOAT:
      return SIByL::RHI::TextureFormat::RG16_FLOAT;
    case VK_FORMAT_R16G16_SINT:
      return SIByL::RHI::TextureFormat::RG16_SINT;
    case VK_FORMAT_R16G16_UINT:
      return SIByL::RHI::TextureFormat::RG16_UINT;
    case VK_FORMAT_R32_SFLOAT:
      return SIByL::RHI::TextureFormat::R32_FLOAT;
    case VK_FORMAT_R32_SINT:
      return SIByL::RHI::TextureFormat::R32_SINT;
    case VK_FORMAT_R32_UINT:
      return SIByL::RHI::TextureFormat::R32_UINT;
    case VK_FORMAT_R8G8_SINT:
      return SIByL::RHI::TextureFormat::RG8_SINT;
    case VK_FORMAT_R8G8_UINT:
      return SIByL::RHI::TextureFormat::RG8_UINT;
    case VK_FORMAT_R8G8_SNORM:
      return SIByL::RHI::TextureFormat::RG8_SNORM;
    case VK_FORMAT_R8G8_UNORM:
      return SIByL::RHI::TextureFormat::RG8_UNORM;
    case VK_FORMAT_R16_SFLOAT:
      return SIByL::RHI::TextureFormat::R16_FLOAT;
    case VK_FORMAT_R16_SINT:
      return SIByL::RHI::TextureFormat::R16_SINT;
    case VK_FORMAT_R16_UINT:
      return SIByL::RHI::TextureFormat::R16_UINT;
    case VK_FORMAT_R8_SINT:
      return SIByL::RHI::TextureFormat::R8_SINT;
    case VK_FORMAT_R8_UINT:
      return SIByL::RHI::TextureFormat::R8_UINT;
    case VK_FORMAT_R8_SNORM:
      return SIByL::RHI::TextureFormat::R8_SNORM;
    case VK_FORMAT_R8_UNORM:
      return SIByL::RHI::TextureFormat::R8_UNORM;
    default:
      return SIByL::RHI::TextureFormat(0);
      break;
  }
}

inline auto getVkImageViewType(TextureViewDimension const& dim) noexcept
    -> VkImageViewType {
  switch (dim) {
    case TextureViewDimension::TEX1D:
      return VkImageViewType::VK_IMAGE_VIEW_TYPE_1D;
      break;
    case TextureViewDimension::TEX2D:
      return VkImageViewType::VK_IMAGE_VIEW_TYPE_2D;
      break;
    case TextureViewDimension::TEX2D_ARRAY:
      return VkImageViewType::VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      break;
    case TextureViewDimension::CUBE:
      return VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE;
      break;
    case TextureViewDimension::CUBE_ARRAY:
      return VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      break;
    case TextureViewDimension::TEX3D:
      return VkImageViewType::VK_IMAGE_VIEW_TYPE_3D;
      break;
    default:
      break;
  }
}

inline auto getVkImageAspectFlags(TextureAspectFlags aspect) noexcept
    -> VkImageAspectFlags {
  VkImageAspectFlags ret = 0;
  if (aspect & (uint32_t)TextureAspect::COLOR_BIT)
    ret |= VK_IMAGE_ASPECT_COLOR_BIT;
  if (aspect & (uint32_t)TextureAspect::DEPTH_BIT)
    ret |= VK_IMAGE_ASPECT_DEPTH_BIT;
  if (aspect & (uint32_t)TextureAspect::STENCIL_BIT)
    ret |= VK_IMAGE_ASPECT_STENCIL_BIT;
  return ret;
}

TextureView_VK::TextureView_VK(Device_VK* device, Texture_VK* texture,
                               TextureViewDescriptor const& descriptor)
    : device(device), texture(texture), descriptor(descriptor) {
  VkImageViewCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.image = texture->getVkImage();
  createInfo.viewType = getVkImageViewType(descriptor.dimension);
  createInfo.format = getVkFormat(descriptor.format);
  createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.subresourceRange.aspectMask =
      getVkImageAspectFlags(descriptor.aspect);
  createInfo.subresourceRange.baseMipLevel = descriptor.baseMipLevel;
  createInfo.subresourceRange.levelCount = descriptor.mipLevelCount;
  createInfo.subresourceRange.baseArrayLayer = descriptor.baseArrayLayer;
  createInfo.subresourceRange.layerCount = descriptor.arrayLayerCount;

  width = texture->width();
  height = texture->height();
  for (int i = 0; i < descriptor.baseMipLevel; ++i) {
    width >>= 1;
    height >>= 1;
  }
  width = max(width, 1);
  height = max(height, 1);

  if (vkCreateImageView(device->getVkDevice(), &createInfo, nullptr,
                        &imageView) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create image views!");
  }
}

TextureView_VK::TextureView_VK(TextureView_VK&& view)
    : imageView(view.imageView),
      descriptor(view.descriptor),
      texture(view.texture),
      device(view.device),
      width(view.width),
      height(width) {
  view.imageView = nullptr;
}

auto TextureView_VK::operator=(TextureView_VK&& view) -> TextureView_VK& {
  imageView = view.imageView;
  descriptor = view.descriptor;
  texture = view.texture;
  device = view.device;
  view.imageView = nullptr;
  return *this;
}

TextureView_VK::~TextureView_VK() {
  if (imageView) vkDestroyImageView(device->getVkDevice(), imageView, nullptr);
}

#pragma endregion

#pragma region VK_SAMPLER_IMPL

inline auto getVkFilter(FilterMode mode) noexcept -> VkFilter {
  switch (mode) {
    case SIByL::RHI::FilterMode::LINEAR:
      return VkFilter::VK_FILTER_LINEAR;
    case SIByL::RHI::FilterMode::NEAREST:
      return VkFilter::VK_FILTER_NEAREST;
    default:
      return VkFilter::VK_FILTER_MAX_ENUM;
  }
}

inline auto getVkSamplerAddressMode(AddressMode address) noexcept
    -> VkSamplerAddressMode {
  switch (address) {
    case SIByL::RHI::AddressMode::MIRROR_REPEAT:
      return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case SIByL::RHI::AddressMode::REPEAT:
      return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case SIByL::RHI::AddressMode::CLAMP_TO_EDGE:
      return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    default:
      return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_MAX_ENUM;
  }
}

Sampler_VK::Sampler_VK(SamplerDescriptor const& desc, Device_VK* device)
    : device(device) {
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = getVkFilter(desc.magFilter);
  samplerInfo.minFilter = getVkFilter(desc.minFilter);
  samplerInfo.addressModeU = getVkSamplerAddressMode(desc.addressModeU);
  samplerInfo.addressModeV = getVkSamplerAddressMode(desc.addressModeV);
  samplerInfo.addressModeW = getVkSamplerAddressMode(desc.addressModeW);
  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = device->getAdapterVk()
                                  ->getVkPhysicalDeviceProperties()
                                  .limits.maxSamplerAnisotropy;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = (desc.mipmapFilter == MipmapFilterMode::LINEAR)
                               ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                               : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = desc.maxLod;
  if (vkCreateSampler(device->getVkDevice(), &samplerInfo, nullptr,
                      &textureSampler) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create texture sampler!");
  }
}

Sampler_VK::~Sampler_VK() {
  vkDestroySampler(device->getVkDevice(), textureSampler, nullptr);
}

auto Sampler_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_SAMPLER;
  objectNameInfo.objectHandle = uint64_t(textureSampler);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
  this->name = name;
}

auto Sampler_VK::getName() const noexcept -> std::string const& { return name; }

auto Device_VK::createSampler(SamplerDescriptor const& desc) noexcept
    -> std::unique_ptr<Sampler> {
  return std::make_unique<Sampler_VK>(desc, this);
}

#pragma endregion

#pragma region VK_SWAPCHAIN_IMPL

inline auto chooseSwapSurfaceFormat(
    std::vector<VkSurfaceFormatKHR> const& availableFormats) noexcept
    -> VkSurfaceFormatKHR {
  for (auto const& availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
        availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

inline auto chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& availablePresentModes) noexcept
    -> VkPresentModeKHR {
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

inline auto chooseSwapExtent(VkSurfaceCapabilitiesKHR const& capabilities,
                             Platform::Window* bindedWindow) noexcept
    -> VkExtent2D {
  if (capabilities.currentExtent.width != Math::uint32_max)
    return capabilities.currentExtent;
  else {
    int width, height;
    bindedWindow->getFramebufferSize(&width, &height);
    VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                               static_cast<uint32_t>(height)};
    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);
    return actualExtent;
  }
}

void createSwapChain(Device_VK* device, SwapChain_VK* swapchain) {
  Adapter_VK* adapater = device->getAdapterVk();
  SwapChainSupportDetails swapChainSupport = querySwapChainSupport(
      adapater->getContext(), adapater->getVkPhysicalDevice());
  VkSurfaceFormatKHR surfaceFormat =
      chooseSwapSurfaceFormat(swapChainSupport.formats);
  VkPresentModeKHR presentMode =
      chooseSwapPresentMode(swapChainSupport.presentModes);
  VkExtent2D extent = chooseSwapExtent(
      swapChainSupport.capabilities, adapater->getContext()->getBindedWindow());
  swapchain->swapChainExtend = extent;
  swapchain->swapChainImageFormat = surfaceFormat.format;
  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = adapater->getContext()->getVkSurfaceKHR();
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  QueueFamilyIndices_VK const& indices = adapater->getQueueFamilyIndices();
  uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                   indices.presentFamily.value()};
  if (indices.graphicsFamily != indices.presentFamily) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;      // Optional
    createInfo.pQueueFamilyIndices = nullptr;  // Optional
  }
  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;
  if (vkCreateSwapchainKHR(device->getVkDevice(), &createInfo, nullptr,
                           &swapchain->swapChain) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create swap chain!");
  }
}

SwapChain_VK::~SwapChain_VK() {
  if (swapChain)
    vkDestroySwapchainKHR(device->getVkDevice(), swapChain, nullptr);
}

auto SwapChain_VK::init(Device_VK* device,
                        SwapChainDescriptor const& desc) noexcept -> void {
  this->device = device;
  recreate();
}

auto SwapChain_VK::recreate() noexcept -> void {
  device->waitIdle();
  // clean up swap chain
  swapChainTextures.clear();
  textureViews.clear();
  if (swapChain)
    vkDestroySwapchainKHR(device->getVkDevice(), swapChain, nullptr);
  // recreate swap chain
  createSwapChain(device, this);
  // retrieving the swap chian image
  uint32_t imageCount = 0;
  vkGetSwapchainImagesKHR(device->getVkDevice(), swapChain, &imageCount,
                          nullptr);
  std::vector<VkImage> swapChainImages;
  swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(device->getVkDevice(), swapChain, &imageCount,
                          swapChainImages.data());
  // create Image views for images
  TextureDescriptor textureDesc;
  textureDesc.dimension = TextureDimension::TEX2D;
  textureDesc.format = getTextureFormat(swapChainImageFormat);
  textureDesc.size = {swapChainExtend.width, swapChainExtend.height};
  textureDesc.usage = 0;
  TextureViewDescriptor viewDesc;
  viewDesc.format = getTextureFormat(swapChainImageFormat);
  viewDesc.aspect = (uint32_t)TextureAspect::COLOR_BIT;
  for (size_t i = 0; i < swapChainImages.size(); i++)
    swapChainTextures.push_back(
        Texture_VK{device, swapChainImages[i], textureDesc});
  for (size_t i = 0; i < swapChainImages.size(); i++)
    textureViews.push_back(
        TextureView_VK{device, &swapChainTextures[i], viewDesc});
}

auto Device_VK::createSwapChain(SwapChainDescriptor const& desc) noexcept
    -> std::unique_ptr<SwapChain> {
  std::unique_ptr<SwapChain_VK> swapChain = std::make_unique<SwapChain_VK>();
  swapChain->init(this, desc);
  return std::move(swapChain);
}

#pragma endregion

#pragma region VK_BINDGROUPLAYOUT_IMPL

inline auto getVkDecriptorType(BindGroupLayoutEntry const& entry)
    -> VkDescriptorType {
  if (entry.buffer.has_value()) {
    switch (entry.buffer.value().type) {
      case BufferBindingType::UNIFORM:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      case BufferBindingType::STORAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      case BufferBindingType::READ_ONLY_STORAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      default:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
  }
  if (entry.sampler.has_value() && entry.texture.has_value())
    return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  if (entry.sampler.has_value()) return VK_DESCRIPTOR_TYPE_SAMPLER;
  if (entry.texture.has_value()) return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  if (entry.storageTexture.has_value()) return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  if (entry.externalTexture.has_value())
    return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  if (entry.accelerationStructure.has_value())
    return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  if (entry.bindlessTextures.has_value())
    return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
}

inline auto getVkShaderStageFlags(ShaderStagesFlags flags) noexcept
    -> VkShaderStageFlags {
  VkShaderStageFlags ret = 0;
  if (flags & (uint32_t)ShaderStages::VERTEX)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT;
  if (flags & (uint32_t)ShaderStages::FRAGMENT)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT;
  if (flags & (uint32_t)ShaderStages::COMPUTE)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT;
  if (flags & (uint32_t)ShaderStages::RAYGEN)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_NV;
  if (flags & (uint32_t)ShaderStages::MISS)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV;
  if (flags & (uint32_t)ShaderStages::CLOSEST_HIT)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
  if (flags & (uint32_t)ShaderStages::INTERSECTION)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_INTERSECTION_BIT_NV;
  if (flags & (uint32_t)ShaderStages::ANY_HIT)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_ANY_HIT_BIT_NV;
  if (flags & (uint32_t)ShaderStages::CALLABLE)
    ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_CALLABLE_BIT_NV;
  return ret;
}

BindGroupLayout_VK::BindGroupLayout_VK(Device_VK* device,
                                       BindGroupLayoutDescriptor const& desc)
    : device(device), descriptor(desc) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(desc.entries.size());
  std::vector<VkDescriptorBindingFlags> bindingFlags(desc.entries.size());
  for (int i = 0; i < desc.entries.size(); i++) {
    bindings[i].binding = desc.entries[i].binding;
    bindings[i].descriptorType = getVkDecriptorType(desc.entries[i]);
    bindings[i].descriptorCount =
        bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
            ? 200
            : 1;
    bindings[i].stageFlags = getVkShaderStageFlags(desc.entries[i].visibility);
    bindings[i].pImmutableSamplers = nullptr;
    bindingFlags[i] = 0;
    if (desc.entries[i].bindlessTextures.has_value())
      bindingFlags[i] |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = bindings.size();
  layoutInfo.pBindings = bindings.data();
  VkDescriptorSetLayoutBindingFlagsCreateInfo flagsExt = {};
  flagsExt.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  flagsExt.bindingCount = bindingFlags.size();
  flagsExt.pBindingFlags = bindingFlags.data();
  layoutInfo.pNext = &flagsExt;
  if (vkCreateDescriptorSetLayout(device->getVkDevice(), &layoutInfo, nullptr,
                                  &layout) != VK_SUCCESS) {
    Core::LogManager::Error(
        "VULKAN :: failed to create descriptor set layout!");
  }
}

BindGroupLayout_VK::~BindGroupLayout_VK() {
  if (layout)
    vkDestroyDescriptorSetLayout(device->getVkDevice(), layout, nullptr);
}

auto BindGroupLayout_VK::getBindGroupLayoutDescriptor() const noexcept
    -> BindGroupLayoutDescriptor const& {
  return descriptor;
}

auto Device_VK::createBindGroupLayout(
    BindGroupLayoutDescriptor const& desc) noexcept
    -> std::unique_ptr<BindGroupLayout> {
  return std::make_unique<BindGroupLayout_VK>(this, desc);
}

#pragma endregion

#pragma region VK_BINDGROUPPOOL_IMPL

BindGroupPool_VK::BindGroupPool_VK(Device_VK* device) : device(device) {
  std::vector<VkDescriptorPoolSize> poolSizes(7);
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(99);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(99);
  poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;
  poolSizes[2].descriptorCount = static_cast<uint32_t>(99);
  poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  poolSizes[3].descriptorCount = static_cast<uint32_t>(99);
  poolSizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSizes[4].descriptorCount = static_cast<uint32_t>(99);
  poolSizes[5].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  poolSizes[5].descriptorCount = static_cast<uint32_t>(99);
  poolSizes[6].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[6].descriptorCount = static_cast<uint32_t>(99);
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = poolSizes.size();
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(999);
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;
  if (vkCreateDescriptorPool(device->getVkDevice(), &poolInfo, nullptr,
                             &descriptorPool) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create descriptor pool!");
  }
}

BindGroupPool_VK::~BindGroupPool_VK() {
  if (descriptorPool)
    vkDestroyDescriptorPool(device->getVkDevice(), descriptorPool, nullptr);
}

auto Device_VK::createBindGroupPool() noexcept -> void {
  bindGroupPool = std::make_unique<BindGroupPool_VK>(this);
}

#pragma endregion

#pragma region VK_BINDGROUP_IMPL

auto Device_VK::createBindGroup(BindGroupDescriptor const& desc) noexcept
    -> std::unique_ptr<BindGroup> {
  return std::make_unique<BindGroup_VK>(this, desc);
}

#pragma endregion

#pragma region VK_PIPELINELAYOUT_IMPL

PipelineLayout_VK::PipelineLayout_VK(Device_VK* device,
                                     PipelineLayoutDescriptor const& desc)
    : device(device) {
  // push constants
  for (auto& ps : desc.pushConstants) {
    pushConstants.push_back(VkPushConstantRange{
        getVkShaderStageFlags(ps.shaderStages), ps.offset, ps.size});
  }
  // descriptor set layouts
  std::vector<VkDescriptorSetLayout> descriptorSets;
  for (auto& bindgroupLayout : desc.bindGroupLayouts) {
    descriptorSets.push_back(
        static_cast<BindGroupLayout_VK*>(bindgroupLayout)->layout);
  }
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = descriptorSets.size();
  pipelineLayoutInfo.pSetLayouts = descriptorSets.data();
  pipelineLayoutInfo.pushConstantRangeCount = pushConstants.size();
  pipelineLayoutInfo.pPushConstantRanges = pushConstants.data();
  if (vkCreatePipelineLayout(device->getVkDevice(), &pipelineLayoutInfo,
                             nullptr, &pipelineLayout) != VK_SUCCESS) {
    Core::LogManager::Error("failed to create pipeline layout!");
  }
}

PipelineLayout_VK::~PipelineLayout_VK() {
  if (pipelineLayout)
    vkDestroyPipelineLayout(device->getVkDevice(), pipelineLayout, nullptr);
}

PipelineLayout_VK::PipelineLayout_VK(PipelineLayout_VK&& layout)
    : pipelineLayout(layout.pipelineLayout),
      pushConstants(layout.pushConstants),
      device(layout.device) {
  layout.pipelineLayout = nullptr;
}

auto PipelineLayout_VK::operator=(PipelineLayout_VK&& layout)
    -> PipelineLayout_VK& {
  pipelineLayout = layout.pipelineLayout;
  pushConstants = layout.pushConstants;
  device = layout.device;
  layout.pipelineLayout = nullptr;
  return *this;
}

auto Device_VK::createPipelineLayout(
    PipelineLayoutDescriptor const& desc) noexcept
    -> std::unique_ptr<PipelineLayout> {
  return std::make_unique<PipelineLayout_VK>(this, desc);
}

#pragma endregion

#pragma region VK_SHADERMODULE_IMPL

inline auto getVkShaderStageFlagBits(ShaderStages flag) noexcept
    -> VkShaderStageFlagBits {
  switch (flag) {
    case SIByL::RHI::ShaderStages::COMPUTE:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT;
      break;
    case SIByL::RHI::ShaderStages::FRAGMENT:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT;
      break;
    case SIByL::RHI::ShaderStages::VERTEX:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT;
      break;
    case SIByL::RHI::ShaderStages::RAYGEN:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_KHR;
      break;
    case SIByL::RHI::ShaderStages::MISS:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_KHR;
      break;
    case SIByL::RHI::ShaderStages::INTERSECTION:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
      break;
    case SIByL::RHI::ShaderStages::CLOSEST_HIT:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      break;
    case SIByL::RHI::ShaderStages::CALLABLE:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_CALLABLE_BIT_KHR;
      break;
    case SIByL::RHI::ShaderStages::ANY_HIT:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
      break;
    default:
      return VkShaderStageFlagBits::VK_SHADER_STAGE_ALL;
      break;
      break;
  }
}

ShaderModule_VK::ShaderModule_VK(Device_VK* device,
                                 ShaderModuleDescriptor const& desc)
    : device(device), stages((uint32_t)desc.stage), entryPoint(desc.name) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = desc.code->size;
  createInfo.pCode = reinterpret_cast<const uint32_t*>(desc.code->data);
  if (vkCreateShaderModule(device->getVkDevice(), &createInfo, nullptr,
                           &shaderModule) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create shader module!");
  }
  // create info
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = getVkShaderStageFlagBits(desc.stage);
  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName = entryPoint.c_str();
}

ShaderModule_VK::ShaderModule_VK(ShaderModule_VK&& shader)
    : device(shader.device),
      stages(shader.stages),
      shaderModule(shader.shaderModule),
      shaderStageInfo(shader.shaderStageInfo) {
  shader.shaderModule = nullptr;
}

auto ShaderModule_VK::operator=(ShaderModule_VK&& shader) -> ShaderModule_VK& {
  device = shader.device;
  stages = shader.stages;
  shaderModule = shader.shaderModule;
  shaderStageInfo = shader.shaderStageInfo;
  shader.shaderModule = nullptr;
  return *this;
}

ShaderModule_VK::~ShaderModule_VK() {
  if (shaderModule)
    vkDestroyShaderModule(device->getVkDevice(), shaderModule, nullptr);
}

auto ShaderModule_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
  objectNameInfo.objectHandle = uint64_t(shaderModule);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
  this->name = name;
}

auto ShaderModule_VK::getName() -> std::string const& { return name; }

auto Device_VK::createShaderModule(ShaderModuleDescriptor const& desc) noexcept
    -> std::unique_ptr<ShaderModule> {
  std::unique_ptr<ShaderModule_VK> shadermodule =
      std::make_unique<ShaderModule_VK>(this, desc);
  return shadermodule;
}

#pragma endregion

#pragma region VK_COMPUTE_PIPELINE_IMPL

ComputePipeline_VK::ComputePipeline_VK(Device_VK* device,
                                       ComputePipelineDescriptor const& desc)
    : device(device), layout(static_cast<PipelineLayout_VK*>(desc.layout)) {
  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module =
      (static_cast<ShaderModule_VK*>(desc.compute.module))->shaderModule;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout =
      static_cast<PipelineLayout_VK*>(desc.layout)->pipelineLayout;
  if (vkCreateComputePipelines(device->getVkDevice(), VK_NULL_HANDLE, 1,
                               &pipelineInfo, nullptr,
                               &pipeline) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create graphics pipeline!");
  }
}

ComputePipeline_VK::~ComputePipeline_VK() {
  if (pipeline) vkDestroyPipeline(device->getVkDevice(), pipeline, nullptr);
}

auto ComputePipeline_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
  objectNameInfo.objectHandle = uint64_t(pipeline);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
}

#pragma endregion

#pragma region VK_RENDERPASS_IMPL

inline auto getVkAttachmentLoadOp(LoadOp op) noexcept -> VkAttachmentLoadOp {
  switch (op) {
    case SIByL::RHI::LoadOp::DONT_CARE:
      return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    case SIByL::RHI::LoadOp::CLEAR:
      return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case SIByL::RHI::LoadOp::LOAD:
      return VK_ATTACHMENT_LOAD_OP_LOAD;
    default:
      return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
}

inline auto getVkAttachmentStoreOp(StoreOp op) noexcept -> VkAttachmentStoreOp {
  switch (op) {
    case SIByL::RHI::StoreOp::DONT_CARE:
      return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    case SIByL::RHI::StoreOp::DISCARD:
      return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    case SIByL::RHI::StoreOp::STORE:
      return VK_ATTACHMENT_STORE_OP_STORE;
    default:
      return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  }
}

RenderPass_VK::RenderPass_VK(Device_VK* device,
                             RenderPassDescriptor const& desc)
    : device(device) {
  // color attachments
  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> attachmentRefs;
  for (auto const& colorAttach : desc.colorAttachments) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = getVkFormat(
        static_cast<TextureView_VK*>(colorAttach.view)->descriptor.format);
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = getVkAttachmentLoadOp(colorAttach.loadOp);
    colorAttachment.storeOp = getVkAttachmentStoreOp(colorAttach.storeOp);
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments.emplace_back(colorAttachment);

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = attachmentRefs.size();
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentRefs.emplace_back(colorAttachmentRef);
  }

  // depth attachment
  if (desc.depthStencilAttachment.view != nullptr) {
    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = getVkFormat(
        static_cast<TextureView_VK*>(desc.depthStencilAttachment.view)
            ->descriptor.format);
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp =
        getVkAttachmentLoadOp(desc.depthStencilAttachment.depthLoadOp);
    depthAttachment.storeOp =
        getVkAttachmentStoreOp(desc.depthStencilAttachment.depthStoreOp);
    depthAttachment.stencilLoadOp =
        getVkAttachmentLoadOp(desc.depthStencilAttachment.stencilLoadOp);
    depthAttachment.stencilStoreOp =
        getVkAttachmentStoreOp(desc.depthStencilAttachment.stencilStoreOp);
    depthAttachment.initialLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments.emplace_back(depthAttachment);
  }
  VkAttachmentReference depthAttachmentRef = {};
  depthAttachmentRef.attachment = desc.colorAttachments.size();
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  // subpass
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = desc.colorAttachments.size();
  subpass.pColorAttachments = attachmentRefs.data();
  subpass.pDepthStencilAttachment =
      (desc.depthStencilAttachment.view != nullptr) ? &depthAttachmentRef
                                                    : nullptr;
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = attachments.size();
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  if (vkCreateRenderPass(device->getVkDevice(), &renderPassInfo, nullptr,
                         &renderPass) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create render pass!");
  }
}

RenderPass_VK::~RenderPass_VK() {
  if (renderPass)
    vkDestroyRenderPass(device->getVkDevice(), renderPass, nullptr);
}

RenderPass_VK::RenderPass_VK(RenderPass_VK&& pass)
    : device(pass.device),
      renderPass(pass.renderPass),
      clearValues(pass.clearValues) {
  pass.renderPass = nullptr;
}

auto RenderPass_VK::operator=(RenderPass_VK&& pass) -> RenderPass_VK& {
  device = pass.device;
  renderPass = pass.renderPass;
  clearValues = pass.clearValues;
  pass.renderPass = nullptr;
  return *this;
}

auto Device_VK::createComputePipeline(
    ComputePipelineDescriptor const& desc) noexcept
    -> std::unique_ptr<ComputePipeline> {
  return std::make_unique<ComputePipeline_VK>(this, desc);
}

#pragma endregion

#pragma region VK_RENDERPIPELINE_IMPL

inline auto getVkPrimitiveTopology(PrimitiveTopology topology) noexcept
    -> VkPrimitiveTopology {
  switch (topology) {
    case SIByL::RHI::PrimitiveTopology::TRIANGLE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      break;
    case SIByL::RHI::PrimitiveTopology::TRIANGLE_LIST:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      break;
    case SIByL::RHI::PrimitiveTopology::LINE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
      break;
    case SIByL::RHI::PrimitiveTopology::LINE_LIST:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      break;
    case SIByL::RHI::PrimitiveTopology::POINT_LIST:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      break;
    default:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      break;
  }
}

inline auto getVkPipelineInputAssemblyStateCreateInfo(
    PrimitiveTopology topology) noexcept
    -> VkPipelineInputAssemblyStateCreateInfo {
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = getVkPrimitiveTopology(topology);
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

inline auto getVkCullModeFlagBits(CullMode cullmode) noexcept
    -> VkCullModeFlagBits {
  switch (cullmode) {
    case SIByL::RHI::CullMode::BACK:
      return VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT;
    case SIByL::RHI::CullMode::FRONT:
      return VkCullModeFlagBits::VK_CULL_MODE_FRONT_BIT;
    case SIByL::RHI::CullMode::NONE:
      return VkCullModeFlagBits::VK_CULL_MODE_NONE;
    case SIByL::RHI::CullMode::BOTH:
      return VkCullModeFlagBits::VK_CULL_MODE_FRONT_AND_BACK;
    default:
      return VkCullModeFlagBits::VK_CULL_MODE_NONE;
  }
}

inline auto getVkFrontFace(FrontFace ff) noexcept -> VkFrontFace {
  switch (ff) {
    case SIByL::RHI::FrontFace::CW:
      return VkFrontFace::VK_FRONT_FACE_CLOCKWISE;
    case SIByL::RHI::FrontFace::CCW:
      return VkFrontFace::VK_FRONT_FACE_COUNTER_CLOCKWISE;
    default:
      return VkFrontFace::VK_FRONT_FACE_CLOCKWISE;
  }
}

inline auto getVkPipelineRasterizationStateCreateInfo(
    DepthStencilState const& dsstate, FragmentState const& fstate,
    PrimitiveState const& pstate) noexcept
    -> VkPipelineRasterizationStateCreateInfo {
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = getVkCullModeFlagBits(pstate.cullMode);
  rasterizer.frontFace = getVkFrontFace(pstate.frontFace);
  rasterizer.depthBiasEnable = (dsstate.depthBias == 0) ? VK_FALSE : VK_TRUE;
  rasterizer.depthBiasConstantFactor = dsstate.depthBias;
  rasterizer.depthBiasClamp = dsstate.depthBiasClamp;
  rasterizer.depthBiasSlopeFactor = dsstate.depthBiasSlopeScale;
  return rasterizer;
}

inline auto getVkPipelineViewportStateCreateInfo(VkViewport& viewport,
                                                 VkRect2D& scissor) noexcept
    -> VkPipelineViewportStateCreateInfo {
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)720;
  viewport.height = (float)480;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  scissor.offset = {0, 0};
  scissor.extent = {720, 480};
  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;
  return viewportState;
}

inline auto getVkPipelineMultisampleStateCreateInfo(
    MultisampleState const& state) noexcept
    -> VkPipelineMultisampleStateCreateInfo {
  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;  // Optional
  multisampling.pSampleMask = nullptr;    // Optional
  multisampling.alphaToCoverageEnable =
      state.alphaToCoverageEnabled ? VK_TRUE : VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;  // Optional
  return multisampling;
}

inline auto getVkCompareOp(CompareFunction compare) noexcept -> VkCompareOp {
  switch (compare) {
    case SIByL::RHI::CompareFunction::ALWAYS:
      return VK_COMPARE_OP_ALWAYS;
      break;
    case SIByL::RHI::CompareFunction::GREATER_EQUAL:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
      break;
    case SIByL::RHI::CompareFunction::NOT_EQUAL:
      return VK_COMPARE_OP_NOT_EQUAL;
      break;
    case SIByL::RHI::CompareFunction::GREATER:
      return VK_COMPARE_OP_GREATER;
      break;
    case SIByL::RHI::CompareFunction::LESS_EQUAL:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
      break;
    case SIByL::RHI::CompareFunction::EQUAL:
      return VK_COMPARE_OP_EQUAL;
      break;
    case SIByL::RHI::CompareFunction::LESS:
      return VK_COMPARE_OP_LESS;
      break;
    case SIByL::RHI::CompareFunction::NEVER:
      return VK_COMPARE_OP_NEVER;
      break;
    default:
      return VK_COMPARE_OP_ALWAYS;
      break;
  }
}

inline auto getVkPipelineDepthStencilStateCreateInfo(
    DepthStencilState const& state) noexcept
    -> VkPipelineDepthStencilStateCreateInfo {
  VkPipelineDepthStencilStateCreateInfo depthStencil = {};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable =
      state.depthCompare != CompareFunction::ALWAYS ? VK_TRUE : VK_FALSE;
  depthStencil.depthWriteEnable = state.depthWriteEnabled ? VK_TRUE : VK_FALSE;
  depthStencil.depthCompareOp = getVkCompareOp(state.depthCompare);
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.minDepthBounds = 0.0f;
  depthStencil.maxDepthBounds = 1.0f;
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {};
  depthStencil.back = {};
  return depthStencil;
}

inline auto getVkBlendFactor(BlendFactor factor) noexcept -> VkBlendFactor {
  switch (factor) {
    case SIByL::RHI::BlendFactor::ONE_MINUS_CONSTANT:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
      break;
    case SIByL::RHI::BlendFactor::CONSTANT:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
      break;
    case SIByL::RHI::BlendFactor::SRC_ALPHA_SATURATED:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
      break;
    case SIByL::RHI::BlendFactor::ONE_MINUS_DST_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
      break;
    case SIByL::RHI::BlendFactor::DST_ALPHA:
      return VK_BLEND_FACTOR_DST_ALPHA;
      break;
    case SIByL::RHI::BlendFactor::ONE_MINUS_DST:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
      break;
    case SIByL::RHI::BlendFactor::DST:
      return VK_BLEND_FACTOR_DST_COLOR;
      break;
    case SIByL::RHI::BlendFactor::ONE_MINUS_SRC_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      break;
    case SIByL::RHI::BlendFactor::SRC_ALPHA:
      return VK_BLEND_FACTOR_SRC_ALPHA;
      break;
    case SIByL::RHI::BlendFactor::ONE_MINUS_SRC:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
      break;
    case SIByL::RHI::BlendFactor::SRC:
      return VK_BLEND_FACTOR_SRC_COLOR;
      break;
    case SIByL::RHI::BlendFactor::ONE:
      return VK_BLEND_FACTOR_ONE;
      break;
    case SIByL::RHI::BlendFactor::ZERO:
      return VK_BLEND_FACTOR_ZERO;
      break;
    default:
      return VK_BLEND_FACTOR_MAX_ENUM;
      break;
  }
}

inline auto getVkBlendOp(BlendOperation const& op) noexcept -> VkBlendOp {
  switch (op) {
    case BlendOperation::ADD:
      return VkBlendOp::VK_BLEND_OP_ADD;
    case BlendOperation::SUBTRACT:
      return VkBlendOp::VK_BLEND_OP_SUBTRACT;
    case BlendOperation::REVERSE_SUBTRACT:
      return VkBlendOp::VK_BLEND_OP_REVERSE_SUBTRACT;
    case BlendOperation::MIN:
      return VkBlendOp::VK_BLEND_OP_MIN;
    case BlendOperation::MAX:
      return VkBlendOp::VK_BLEND_OP_MAX;
    default:
      return VkBlendOp::VK_BLEND_OP_MAX_ENUM;
  }
}

inline auto getVkPipelineColorBlendAttachmentState(
    FragmentState const& state) noexcept
    -> std::vector<VkPipelineColorBlendAttachmentState> {
  std::vector<VkPipelineColorBlendAttachmentState> attachmentStates;
  for (ColorTargetState const& attchment : state.targets) {
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable =
        attchment.blend.blendEnable() ? VK_TRUE : VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor =
        getVkBlendFactor(attchment.blend.color.srcFactor);
    colorBlendAttachment.dstColorBlendFactor =
        getVkBlendFactor(attchment.blend.color.dstFactor);
    colorBlendAttachment.colorBlendOp =
        getVkBlendOp(attchment.blend.color.operation);
    colorBlendAttachment.srcAlphaBlendFactor =
        getVkBlendFactor(attchment.blend.alpha.srcFactor);
    colorBlendAttachment.dstAlphaBlendFactor =
        getVkBlendFactor(attchment.blend.alpha.dstFactor);
    colorBlendAttachment.alphaBlendOp =
        getVkBlendOp(attchment.blend.color.operation);
    attachmentStates.emplace_back(colorBlendAttachment);
  }
  return attachmentStates;
}

inline auto getVkPipelineColorBlendStateCreateInfo(
    std::vector<VkPipelineColorBlendAttachmentState>&
        colorBlendAttachments) noexcept -> VkPipelineColorBlendStateCreateInfo {
  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = colorBlendAttachments.size();
  colorBlending.pAttachments = colorBlendAttachments.data();
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;
  return colorBlending;
}

auto getVkVertexInputBindingDescription(VertexState const& state) noexcept
    -> std::vector<VkVertexInputBindingDescription> {
  std::vector<VkVertexInputBindingDescription> descriptions;
  for (auto& buffer : state.buffers) {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = buffer.arrayStride;
    bindingDescription.inputRate = buffer.stepMode == VertexStepMode::VERTEX
                                       ? VK_VERTEX_INPUT_RATE_VERTEX
                                       : VK_VERTEX_INPUT_RATE_INSTANCE;
    descriptions.push_back(bindingDescription);
  }
  return descriptions;
}

inline auto getVkFormat(VertexFormat format) noexcept -> VkFormat {
  switch (format) {
    case SIByL::RHI::VertexFormat::SINT32X4:
      return VK_FORMAT_R32G32B32A32_SINT;
    case SIByL::RHI::VertexFormat::SINT32X3:
      return VK_FORMAT_R32G32B32_SINT;
    case SIByL::RHI::VertexFormat::SINT32X2:
      return VK_FORMAT_R32G32_SINT;
    case SIByL::RHI::VertexFormat::SINT32:
      return VK_FORMAT_R32_SINT;
    case SIByL::RHI::VertexFormat::UINT32X4:
      return VK_FORMAT_R32G32B32A32_UINT;
    case SIByL::RHI::VertexFormat::UINT32X3:
      return VK_FORMAT_R32G32B32_UINT;
    case SIByL::RHI::VertexFormat::UINT32X2:
      return VK_FORMAT_R32G32_UINT;
    case SIByL::RHI::VertexFormat::UINT32:
      return VK_FORMAT_R32_UINT;
    case SIByL::RHI::VertexFormat::FLOAT32X4:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case SIByL::RHI::VertexFormat::FLOAT32X3:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case SIByL::RHI::VertexFormat::FLOAT32X2:
      return VK_FORMAT_R32G32_SFLOAT;
    case SIByL::RHI::VertexFormat::FLOAT32:
      return VK_FORMAT_R32_SFLOAT;
    case SIByL::RHI::VertexFormat::FLOAT16X4:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case SIByL::RHI::VertexFormat::FLOAT16X2:
      return VK_FORMAT_R16G16_SFLOAT;
    case SIByL::RHI::VertexFormat::SNORM16X4:
      return VK_FORMAT_R16G16B16A16_SNORM;
    case SIByL::RHI::VertexFormat::SNORM16X2:
      return VK_FORMAT_R16G16_SNORM;
    case SIByL::RHI::VertexFormat::UNORM16X4:
      return VK_FORMAT_R16G16B16A16_UNORM;
    case SIByL::RHI::VertexFormat::UNORM16X2:
      return VK_FORMAT_R16G16_UNORM;
    case SIByL::RHI::VertexFormat::SINT16X4:
      return VK_FORMAT_R16G16B16A16_SINT;
    case SIByL::RHI::VertexFormat::SINT16X2:
      return VK_FORMAT_R16G16_SINT;
    case SIByL::RHI::VertexFormat::UINT16X4:
      return VK_FORMAT_R16G16B16A16_UINT;
    case SIByL::RHI::VertexFormat::UINT16X2:
      return VK_FORMAT_R16G16_UINT;
    case SIByL::RHI::VertexFormat::SNORM8X4:
      return VK_FORMAT_R8G8B8A8_SNORM;
    case SIByL::RHI::VertexFormat::SNORM8X2:
      return VK_FORMAT_R8G8_SNORM;
    case SIByL::RHI::VertexFormat::UNORM8X4:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case SIByL::RHI::VertexFormat::UNORM8X2:
      return VK_FORMAT_R8G8_UNORM;
    case SIByL::RHI::VertexFormat::SINT8X4:
      return VK_FORMAT_R8G8B8A8_SINT;
    case SIByL::RHI::VertexFormat::SINT8X2:
      return VK_FORMAT_R8G8_SINT;
    case SIByL::RHI::VertexFormat::UINT8X4:
      return VK_FORMAT_R8G8B8A8_UINT;
    case SIByL::RHI::VertexFormat::UINT8X2:
      return VK_FORMAT_R8G8_UINT;
    default:
      return VK_FORMAT_MAX_ENUM;
  }
}

inline auto getAttributeDescriptions(VertexState const& state) noexcept
    -> std::vector<VkVertexInputAttributeDescription> {
  std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
  for (int i = 0; i < state.buffers.size(); ++i) {
    auto& buffer = state.buffers[i];
    for (int j = 0; j < buffer.attributes.size(); ++j) {
      auto& attribute = buffer.attributes[j];
      VkVertexInputAttributeDescription description = {};
      description.binding = i;
      description.location = attribute.shaderLocation;
      description.format = getVkFormat(attribute.format);
      description.offset = attribute.offset;
      attributeDescriptions.push_back(description);
    }
  }
  return attributeDescriptions;
}

inline auto fillFixedFunctionSettingDynamicInfo(
    RenderPipeline_VK::RenderPipelineFixedFunctionSettings& settings) noexcept
    -> void {
  // fill in 2 structure in the settings:
  // 1. std::vector<VkDynamicState> dynamicStates
  settings.dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                            VK_DYNAMIC_STATE_SCISSOR};
  // 2. VkPipelineDynamicStateCreateInfo dynamicState
  settings.dynamicState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  settings.dynamicState.dynamicStateCount =
      static_cast<uint32_t>(settings.dynamicStates.size());
  settings.dynamicState.pDynamicStates = settings.dynamicStates.data();
}

inline auto fillFixedFunctionSettingVertexInfo(
    VertexState const& state,
    RenderPipeline_VK::RenderPipelineFixedFunctionSettings& settings) noexcept
    -> void {
  // fill in 3 structure in the settings:
  // 1. std::vector<VkVertexInputBindingDescription>   vertexBindingDescriptor
  settings.vertexBindingDescriptor = getVkVertexInputBindingDescription(state);
  // 2. std::vector<VkVertexInputAttributeDescription>
  // vertexAttributeDescriptions{};
  settings.vertexAttributeDescriptions = getAttributeDescriptions(state);
  // 3. VkPipelineVertexInputStateCreateInfo		   vertexInputState =
  // {};
  VkPipelineVertexInputStateCreateInfo& vertexInput = settings.vertexInputState;
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount =
      settings.vertexBindingDescriptor.size();
  vertexInput.pVertexBindingDescriptions =
      settings.vertexBindingDescriptor.data();
  vertexInput.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(settings.vertexAttributeDescriptions.size());
  vertexInput.pVertexAttributeDescriptions =
      settings.vertexAttributeDescriptions.data();
}

inline auto fillFixedFunctionSettingViewportInfo(
    RenderPipeline_VK::RenderPipelineFixedFunctionSettings& settings) noexcept
    -> void {
  // fill in 1 structure in the settings, whose viewport & scisor could be set
  // later
  VkPipelineViewportStateCreateInfo& viewportState = settings.viewportState;
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
}

RenderPipeline_VK::RenderPipeline_VK(Device_VK* device,
                                     RenderPipelineDescriptor const& desc)
    : device(device) {
  if (desc.vertex.module)
    fixedFunctionSetttings.shaderStages.push_back(
        static_cast<ShaderModule_VK*>(desc.vertex.module)->shaderStageInfo);
  if (desc.fragment.module)
    fixedFunctionSetttings.shaderStages.push_back(
        static_cast<ShaderModule_VK*>(desc.fragment.module)->shaderStageInfo);

  fillFixedFunctionSettingDynamicInfo(fixedFunctionSetttings);
  fillFixedFunctionSettingVertexInfo(desc.vertex, fixedFunctionSetttings);
  fixedFunctionSetttings.assemblyState =
      getVkPipelineInputAssemblyStateCreateInfo(desc.primitive.topology);
  fillFixedFunctionSettingViewportInfo(fixedFunctionSetttings);
  fixedFunctionSetttings.rasterizationState =
      getVkPipelineRasterizationStateCreateInfo(desc.depthStencil,
                                                desc.fragment, desc.primitive);

  fixedFunctionSetttings.multisampleState =
      getVkPipelineMultisampleStateCreateInfo(desc.multisample);
  fixedFunctionSetttings.depthStencilState =
      getVkPipelineDepthStencilStateCreateInfo(desc.depthStencil);
  fixedFunctionSetttings.colorBlendAttachmentStates =
      getVkPipelineColorBlendAttachmentState(desc.fragment);
  fixedFunctionSetttings.colorBlendState =
      getVkPipelineColorBlendStateCreateInfo(
          fixedFunctionSetttings.colorBlendAttachmentStates);
  fixedFunctionSetttings.pipelineLayout = desc.layout;

  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = fixedFunctionSetttings.shaderStages.size();
  pipelineInfo.pStages = fixedFunctionSetttings.shaderStages.data();
  pipelineInfo.pVertexInputState = &fixedFunctionSetttings.vertexInputState;
  pipelineInfo.pInputAssemblyState = &fixedFunctionSetttings.assemblyState;
  pipelineInfo.pViewportState = &fixedFunctionSetttings.viewportState;
  pipelineInfo.pRasterizationState = &fixedFunctionSetttings.rasterizationState;
  pipelineInfo.pMultisampleState = &fixedFunctionSetttings.multisampleState;
  pipelineInfo.pDepthStencilState = &fixedFunctionSetttings.depthStencilState;
  pipelineInfo.pColorBlendState = &fixedFunctionSetttings.colorBlendState;
  pipelineInfo.pDynamicState = &fixedFunctionSetttings.dynamicState;
  pipelineInfo.layout =
      static_cast<PipelineLayout_VK*>(fixedFunctionSetttings.pipelineLayout)
          ->pipelineLayout;
}

auto RenderPipeline_VK::combineRenderPass(RenderPass_VK* renderpass) noexcept
    -> void {
  // destroy current pipeline
  if (pipeline) {
    vkDestroyPipeline(device->getVkDevice(), pipeline, nullptr);
    pipeline = {};
  }
  pipelineInfo.renderPass = renderpass->renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineInfo.basePipelineIndex = -1;

  if (vkCreateGraphicsPipelines(device->getVkDevice(), VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr,
                                &pipeline) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create graphics pipeline!");
  }
}

RenderPipeline_VK::~RenderPipeline_VK() {
  if (pipeline) vkDestroyPipeline(device->getVkDevice(), pipeline, nullptr);
}

RenderPipeline_VK::RenderPipeline_VK(RenderPipeline_VK&& pipeline)
    : device(pipeline.device), pipeline(pipeline.pipeline) {
  pipeline.pipeline = nullptr;
}

auto RenderPipeline_VK::operator=(RenderPipeline_VK&& pipeline)
    -> RenderPipeline_VK& {
  device = pipeline.device;
  this->pipeline = pipeline.pipeline;
  pipeline.pipeline = nullptr;
  return *this;
}

auto RenderPipeline_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
  objectNameInfo.objectHandle = uint64_t(pipeline);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
}

auto Device_VK::createRenderPipeline(
    RenderPipelineDescriptor const& desc) noexcept
    -> std::unique_ptr<RenderPipeline> {
  return std::make_unique<RenderPipeline_VK>(this, desc);
}

#pragma endregion

#pragma region VK_COMMANDPOOL_IMPL

CommandPool_VK::CommandPool_VK(Device_VK* device) : device(device) {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex =
      device->getAdapterVk()->getQueueFamilyIndices().graphicsFamily.value();
  if (vkCreateCommandPool(device->getVkDevice(), &poolInfo, nullptr,
                          &commandPool) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create command pool!");
  }
}

CommandPool_VK::~CommandPool_VK() {
  if (commandPool)
    vkDestroyCommandPool(device->getVkDevice(), commandPool, nullptr);
}

auto Device_VK::allocateCommandBuffer() noexcept
    -> std::unique_ptr<CommandBuffer_VK> {
  return graphicPool->allocateCommandBuffer();
}

#pragma endregion

#pragma region VK_COMMANDBUFFER_IMPL

CommandBuffer_VK::~CommandBuffer_VK() {
  vkFreeCommandBuffers(device->getVkDevice(), commandPool->commandPool, 1,
                       &commandBuffer);
}

#pragma endregion

#pragma region VK_SEMAPHORE_IMPL

Semaphore_VK::Semaphore_VK(Device_VK* device) : device(device) {
  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  if (vkCreateSemaphore(device->getVkDevice(), &semaphoreInfo, nullptr,
                        &semaphore) != VK_SUCCESS)
    Core::LogManager::Error("VULKAN :: failed to create semaphores!");
}

Semaphore_VK::~Semaphore_VK() {
  if (semaphore) vkDestroySemaphore(device->getVkDevice(), semaphore, nullptr);
}

auto Queue_VK::presentSwapChain(SwapChain* swapchain, uint32_t imageIndex,
                                Semaphore* semaphore) noexcept -> void {
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores =
      &static_cast<Semaphore_VK*>(semaphore)->semaphore;
  VkSwapchainKHR swapChains[] = {
      static_cast<SwapChain_VK*>(swapchain)->swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.pResults = nullptr;
  vkQueuePresentKHR(queue, &presentInfo);
}

#pragma endregion

#pragma region VK_FENCE_IMPL

Fence_VK::Fence_VK(Device_VK* device) : device(device) {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  if (vkCreateFence(device->getVkDevice(), &fenceInfo, nullptr, &fence) !=
      VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create fence");
  }
}

Fence_VK::~Fence_VK() {
  if (fence) vkDestroyFence(device->getVkDevice(), fence, nullptr);
}

auto Fence_VK::wait() noexcept -> void {
  vkWaitForFences(device->getVkDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
}

auto Fence_VK::reset() noexcept -> void {
  vkResetFences(device->getVkDevice(), 1, &fence);
}

#pragma endregion

#pragma region VK_COMMANDBUFFER_IMPL

auto CommandPool_VK::allocateCommandBuffer() noexcept
    -> std::unique_ptr<CommandBuffer_VK> {
  std::unique_ptr<CommandBuffer_VK> command =
      std::make_unique<CommandBuffer_VK>();
  command->device = device;
  command->commandPool = this;
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device->getVkDevice(), &allocInfo,
                               &command->commandBuffer) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to allocate command buffers!");
  }
  return command;
}

auto Queue_VK::submit(
    std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  std::vector<VkCommandBuffer> vkCommandBuffers;
  for (auto buffer : commandBuffers)
    vkCommandBuffers.push_back(
        (static_cast<CommandBuffer_VK*>(buffer))->commandBuffer);
  submitInfo.commandBufferCount = vkCommandBuffers.size();
  submitInfo.pCommandBuffers = vkCommandBuffers.data();
  vkQueueSubmit(device->getVkGraphicsQueue().queue, 1, &submitInfo,
                VK_NULL_HANDLE);
}

auto Queue_VK::submit(std::vector<CommandBuffer*> const& commandBuffers,
    Fence* fence) noexcept -> void {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  std::vector<VkCommandBuffer> vkCommandBuffers;
  for (auto buffer : commandBuffers)
    vkCommandBuffers.push_back(
        (static_cast<CommandBuffer_VK*>(buffer))->commandBuffer);
  submitInfo.commandBufferCount = vkCommandBuffers.size();
  submitInfo.pCommandBuffers = vkCommandBuffers.data();
  VkResult result = vkQueueSubmit(device->getVkGraphicsQueue().queue, 1, &submitInfo,
                static_cast<Fence_VK*>(fence)->fence);
  if (result != VK_SUCCESS) {
    Core::LogManager::Error("Vulkan :: Queue Submit Failed!");
  }
}

auto Queue_VK::submit(std::vector<CommandBuffer*> const& commandBuffers,
                      Semaphore* wait, Semaphore* signal, Fence* fence) noexcept
    -> void {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  std::vector<VkSemaphore> waitSemaphores;
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  if (wait) {
    waitSemaphores.push_back(static_cast<Semaphore_VK*>(wait)->semaphore);
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages;
  }
  std::vector<VkCommandBuffer> vkCommandBuffers;
  for (auto buffer : commandBuffers)
    vkCommandBuffers.push_back(
        (static_cast<CommandBuffer_VK*>(buffer))->commandBuffer);
  submitInfo.commandBufferCount = vkCommandBuffers.size();
  submitInfo.pCommandBuffers = vkCommandBuffers.data();
  std::vector<VkSemaphore> signalSemaphores = {};
  if (signal) {
    signalSemaphores.push_back(static_cast<Semaphore_VK*>(signal)->semaphore);
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores.data();
  }
  if (vkQueueSubmit(device->getVkGraphicsQueue().queue, 1, &submitInfo,
                    static_cast<Fence_VK*>(fence)->fence) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to submit draw command buffer!");
  }
}

#pragma endregion

#pragma region VK_MULTIFRAMEFLIGHTS_IMPL

MultiFrameFlights_VK::MultiFrameFlights_VK(Device_VK* device, int maxFlightNum,
                                           SwapChain* swapchain)
    : device(device),
      maxFlightNum(maxFlightNum),
      swapChain(static_cast<SwapChain_VK*>(swapchain)) {
  commandBuffers.resize(maxFlightNum);
  for (size_t i = 0; i < maxFlightNum; ++i) {
    commandBuffers[i] = device->allocateCommandBuffer();
  }
  // createSyncObjects
  imageAvailableSemaphores.resize(maxFlightNum);
  renderFinishedSemaphores.resize(maxFlightNum);
  inFlightFences.resize(maxFlightNum);
  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < maxFlightNum; ++i) {
    if (vkCreateSemaphore(device->getVkDevice(), &semaphoreInfo, nullptr,
                          &imageAvailableSemaphores[i].semaphore) !=
            VK_SUCCESS ||
        vkCreateSemaphore(device->getVkDevice(), &semaphoreInfo, nullptr,
                          &renderFinishedSemaphores[i].semaphore) !=
            VK_SUCCESS ||
        vkCreateFence(device->getVkDevice(), &fenceInfo, nullptr,
                      &inFlightFences[i].fence) != VK_SUCCESS) {
      Core::LogManager::Error(
          "VULKAN :: failed to create synchronization objects for a frame!");
    } else {
      imageAvailableSemaphores[i].device = device;
      renderFinishedSemaphores[i].device = device;
      inFlightFences[i].device = device;
    }
  }
}

auto MultiFrameFlights_VK::frameStart() noexcept -> void {
  VkResult result = vkWaitForFences(device->getVkDevice(), 1,
                          &inFlightFences[currentFrame].fence,
                  VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    Core::LogManager::Error(
        "Vulkan::MultiFrameFlight::frameStart()::WaitForFenceFailed!");
  }
  vkResetFences(device->getVkDevice(), 1, &inFlightFences[currentFrame].fence);
  if (swapChain)
    vkAcquireNextImageKHR(device->getVkDevice(), swapChain->swapChain,
                          UINT64_MAX,
                          imageAvailableSemaphores[currentFrame].semaphore,
                          VK_NULL_HANDLE, &imageIndex);
  vkResetCommandBuffer(commandBuffers[currentFrame]->commandBuffer, 0);
}

auto MultiFrameFlights_VK::frameEnd() noexcept -> void {
  if (swapChain)
    device->getPresentQueue()->presentSwapChain(
        swapChain, imageIndex, &renderFinishedSemaphores[currentFrame]);
  currentFrame = (currentFrame + 1) % maxFlightNum;
}

auto MultiFrameFlights_VK::getCommandBuffer() noexcept -> CommandBuffer* {
  return commandBuffers[currentFrame].get();
}

auto Device_VK::createMultiFrameFlights(
    MultiFrameFlightsDescriptor const& desc) noexcept
    -> std::unique_ptr<MultiFrameFlights> {
  return std::make_unique<MultiFrameFlights_VK>(this, desc.maxFlightNum,
                                                desc.swapchain);
}

#pragma endregion

#pragma region VK_COMMANDENCODER_IMPL

CommandEncoder_VK::~CommandEncoder_VK() {}

inline auto getVkPipelineStageFlags(PipelineStageFlags stages) noexcept
    -> VkPipelineStageFlags {
  uint32_t flags = 0;
  if (stages & (uint32_t)PipelineStages::TOP_OF_PIPE_BIT)
    flags |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  if (stages & (uint32_t)PipelineStages::DRAW_INDIRECT_BIT)
    flags |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  if (stages & (uint32_t)PipelineStages::VERTEX_INPUT_BIT)
    flags |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
  if (stages & (uint32_t)PipelineStages::VERTEX_SHADER_BIT)
    flags |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
  if (stages & (uint32_t)PipelineStages::TESSELLATION_CONTROL_SHADER_BIT)
    flags |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
  if (stages & (uint32_t)PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT)
    flags |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
  if (stages & (uint32_t)PipelineStages::GEOMETRY_SHADER_BIT)
    flags |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
  if (stages & (uint32_t)PipelineStages::FRAGMENT_SHADER_BIT)
    flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  if (stages & (uint32_t)PipelineStages::EARLY_FRAGMENT_TESTS_BIT)
    flags |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  if (stages & (uint32_t)PipelineStages::LATE_FRAGMENT_TESTS_BIT)
    flags |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  if (stages & (uint32_t)PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT)
    flags |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (stages & (uint32_t)PipelineStages::COMPUTE_SHADER_BIT)
    flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (stages & (uint32_t)PipelineStages::TRANSFER_BIT)
    flags |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (stages & (uint32_t)PipelineStages::BOTTOM_OF_PIPE_BIT)
    flags |= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  if (stages & (uint32_t)PipelineStages::HOST_BIT)
    flags |= VK_PIPELINE_STAGE_HOST_BIT;
  if (stages & (uint32_t)PipelineStages::ALL_GRAPHICS_BIT)
    flags |= VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
  if (stages & (uint32_t)PipelineStages::ALL_COMMANDS_BIT)
    flags |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  if (stages & (uint32_t)PipelineStages::TRANSFORM_FEEDBACK_BIT_EXT)
    flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
  if (stages & (uint32_t)PipelineStages::CONDITIONAL_RENDERING_BIT_EXT)
    flags |= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
  if (stages & (uint32_t)PipelineStages::ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
    flags |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
  if (stages & (uint32_t)PipelineStages::RAY_TRACING_SHADER_BIT_KHR)
    flags |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
  if (stages & (uint32_t)PipelineStages::TASK_SHADER_BIT_NV)
    flags |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV;
  if (stages & (uint32_t)PipelineStages::MESH_SHADER_BIT_NV)
    flags |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV;
  if (stages & (uint32_t)PipelineStages::FRAGMENT_DENSITY_PROCESS_BIT)
    flags |= VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT;
  if (stages & (uint32_t)PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT_BIT)
    flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
  if (stages & (uint32_t)PipelineStages::COMMAND_PREPROCESS_BIT)
    flags |= VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV;
  return (VkPipelineStageFlags)flags;
}

inline auto getVkAccessFlags(AccessFlags accessFlags) noexcept
    -> VkAccessFlags {
  VkAccessFlags flags = 0;
  if (accessFlags & (uint32_t)AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
    flags |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::INDEX_READ_BIT)
    flags |= VK_ACCESS_INDEX_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::VERTEX_ATTRIBUTE_READ_BIT)
    flags |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::UNIFORM_READ_BIT)
    flags |= VK_ACCESS_UNIFORM_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::INPUT_ATTACHMENT_READ_BIT)
    flags |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::SHADER_READ_BIT)
    flags |= VK_ACCESS_SHADER_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::SHADER_WRITE_BIT)
    flags |= VK_ACCESS_SHADER_WRITE_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::COLOR_ATTACHMENT_READ_BIT)
    flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT)
    flags |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
  if (accessFlags &
      (uint32_t)AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
    flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::TRANSFER_READ_BIT)
    flags |= VK_ACCESS_TRANSFER_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::TRANSFER_WRITE_BIT)
    flags |= VK_ACCESS_TRANSFER_WRITE_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::HOST_READ_BIT)
    flags |= VK_ACCESS_HOST_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::HOST_WRITE_BIT)
    flags |= VK_ACCESS_HOST_WRITE_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::MEMORY_READ_BIT)
    flags |= VK_ACCESS_MEMORY_READ_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::MEMORY_WRITE_BIT)
    flags |= VK_ACCESS_MEMORY_WRITE_BIT;
  if (accessFlags & (uint32_t)AccessFlagBits::TRANSFORM_FEEDBACK_WRITE_BIT)
    flags |= VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
  if (accessFlags &
      (uint32_t)AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_READ_BIT)
    flags |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT;
  if (accessFlags &
      (uint32_t)AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT)
    flags |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
  if (accessFlags & (uint32_t)AccessFlagBits::CONDITIONAL_RENDERING_READ_BIT)
    flags |= VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
  if (accessFlags &
      (uint32_t)AccessFlagBits::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT)
    flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT;
  if (accessFlags & (uint32_t)AccessFlagBits::ACCELERATION_STRUCTURE_READ_BIT)
    flags |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  if (accessFlags & (uint32_t)AccessFlagBits::ACCELERATION_STRUCTURE_WRITE_BIT)
    flags |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  if (accessFlags & (uint32_t)AccessFlagBits::FRAGMENT_DENSITY_MAP_READ_BIT)
    flags |= VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT;
  if (accessFlags &
      (uint32_t)AccessFlagBits::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT)
    flags |= VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
  if (accessFlags & (uint32_t)AccessFlagBits::COMMAND_PREPROCESS_READ_BIT)
    flags |= VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV;
  if (accessFlags & (uint32_t)AccessFlagBits::COMMAND_PREPROCESS_WRITE_BIT)
    flags |= VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV;
  return flags;
}

inline auto getVkDependencyTypeFlags(DependencyTypeFlags type) noexcept
    -> VkDependencyFlags {
  uint32_t flags = 0;
  if (type & (uint32_t)DependencyType::BY_REGION_BIT)
    flags |= VK_DEPENDENCY_BY_REGION_BIT;
  if (type & (uint32_t)DependencyType::VIEW_LOCAL_BIT)
    flags |= VK_DEPENDENCY_VIEW_LOCAL_BIT;
  if (type & (uint32_t)DependencyType::DEVICE_GROUP_BIT)
    flags |= VK_DEPENDENCY_DEVICE_GROUP_BIT;
  return (VkDependencyFlags)flags;
}

auto CommandEncoder_VK::pipelineBarrier(BarrierDescriptor const& desc) noexcept
    -> void {
  // memory barriers
  std::vector<VkMemoryBarrier> memoryBarriers(desc.memoryBarriers.size());
  // buffer memory barriers
  std::vector<VkBufferMemoryBarrier> bufferBemoryBarriers(
      desc.bufferMemoryBarriers.size());
  for (int i = 0; i < bufferBemoryBarriers.size(); ++i) {
    VkBufferMemoryBarrier& bmb = bufferBemoryBarriers[i];
    BufferMemoryBarrierDescriptor const& descriptor =
        desc.bufferMemoryBarriers[i];
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.buffer = static_cast<Buffer_VK*>(descriptor.buffer)->getVkBuffer();
    bmb.offset = 0;
    bmb.size = static_cast<Buffer_VK*>(descriptor.buffer)->size();
    bmb.srcAccessMask = getVkAccessFlags(descriptor.srcAccessMask);
    bmb.dstAccessMask = getVkAccessFlags(descriptor.dstAccessMask);
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  }
  // image memory
  std::vector<VkImageMemoryBarrier> imageMemoryBarriers(
      desc.textureMemoryBarriers.size());
  for (int i = 0; i < imageMemoryBarriers.size(); ++i) {
    VkImageMemoryBarrier& imb = imageMemoryBarriers[i];
    TextureMemoryBarrierDescriptor const& descriptor =
        desc.textureMemoryBarriers[i];
    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imb.oldLayout = getVkImageLayout(descriptor.oldLayout);
    imb.newLayout = getVkImageLayout(descriptor.newLayout);
    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imb.srcAccessMask = getVkAccessFlags(descriptor.srcAccessMask);
    imb.dstAccessMask = getVkAccessFlags(descriptor.dstAccessMask);
    imb.image = static_cast<Texture_VK*>(descriptor.texture)->getVkImage();
    imb.subresourceRange.aspectMask =
        getVkImageAspectFlags(descriptor.subresourceRange.aspectMask);
    imb.subresourceRange.baseMipLevel =
        descriptor.subresourceRange.baseMipLevel;
    imb.subresourceRange.levelCount = descriptor.subresourceRange.levelCount;
    imb.subresourceRange.baseArrayLayer =
        descriptor.subresourceRange.baseArrayLayer;
    imb.subresourceRange.layerCount = descriptor.subresourceRange.layerCount;
  }
  vkCmdPipelineBarrier(commandBuffer->commandBuffer,
                       getVkPipelineStageFlags(desc.srcStageMask),
                       getVkPipelineStageFlags(desc.dstStageMask),
                       getVkDependencyTypeFlags(desc.dependencyType),
                       memoryBarriers.size(), memoryBarriers.data(),
                       bufferBemoryBarriers.size(), bufferBemoryBarriers.data(),
                       imageMemoryBarriers.size(), imageMemoryBarriers.data());
}

auto CommandEncoder_VK::copyBufferToBuffer(Buffer* source, size_t sourceOffset,
                                           Buffer* destination,
                                           size_t destinationOffset,
                                           size_t size) noexcept -> void {
  VkBufferCopy copyRegion{};
  copyRegion.srcOffset = sourceOffset;
  copyRegion.dstOffset = destinationOffset;
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer->commandBuffer,
                  static_cast<Buffer_VK*>(source)->getVkBuffer(),
                  static_cast<Buffer_VK*>(destination)->getVkBuffer(), 1,
                  &copyRegion);
}

auto CommandEncoder_VK::clearBuffer(Buffer* buffer, size_t offset,
                                    size_t size) noexcept -> void {
  float const fillValueConst = 0;
  uint32_t const& fillValueU32 =
      reinterpret_cast<const uint32_t&>(fillValueConst);
  vkCmdFillBuffer(commandBuffer->commandBuffer,
                  static_cast<Buffer_VK*>(buffer)->getVkBuffer(), offset, size,
                  fillValueU32);
}

auto CommandEncoder_VK::fillBuffer(Buffer* buffer, size_t offset, size_t size,
                                   float fillValue) noexcept -> void {
  float const fillValueConst = fillValue;
  uint32_t const& fillValueU32 =
      reinterpret_cast<const uint32_t&>(fillValueConst);
  vkCmdFillBuffer(commandBuffer->commandBuffer,
                  static_cast<Buffer_VK*>(buffer)->getVkBuffer(), offset, size,
                  fillValueU32);
}

auto CommandEncoder_VK::copyBufferToTexture(ImageCopyBuffer const& source,
                                            ImageCopyTexture const& destination,
                                            Extend3D const& copySize) noexcept
    -> void {
  VkBufferImageCopy region{};
  region.bufferOffset = source.offset;
  region.bufferRowLength = source.bytesPerRow;
  region.bufferImageHeight = source.rowsPerImage;
  region.imageSubresource.aspectMask =
      getVkImageAspectFlags(destination.aspect);
  region.imageSubresource.mipLevel = destination.mipLevel;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = copySize.depthOrArrayLayers;
  region.imageOffset = {static_cast<int>(destination.origin.x),
                        static_cast<int>(destination.origin.y),
                        static_cast<int>(destination.origin.z)};
  region.imageExtent = {copySize.width, copySize.height, 1};
  vkCmdCopyBufferToImage(
      commandBuffer->commandBuffer,
      static_cast<Buffer_VK*>(source.buffer)->getVkBuffer(),
      static_cast<Texture_VK*>(destination.texutre)->getVkImage(),
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

auto CommandEncoder_VK::copyTextureToBuffer(ImageCopyTexture const& source,
                                            ImageCopyBuffer const& destination,
                                            Extend3D const& copySize) noexcept
    -> void {}

auto CommandEncoder_VK::copyTextureToTexture(
    ImageCopyTexture const& source, ImageCopyTexture const& destination,
    Extend3D const& copySize) noexcept -> void {
  VkImageCopy region;
  // We copy the image aspect, layer 0, mip 0:
  region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.srcSubresource.baseArrayLayer = 0;
  region.srcSubresource.layerCount = 1;
  region.srcSubresource.mipLevel = source.mipLevel;
  // (0, 0, 0) in the first image corresponds to (0, 0, 0) in the second image:
  region.srcOffset = {int(source.origin.x), int(source.origin.y),
                      int(source.origin.z)};
  region.dstSubresource = region.srcSubresource;
  region.dstSubresource.mipLevel = destination.mipLevel;
  region.dstOffset = {int(destination.origin.x), int(destination.origin.y),
                      int(destination.origin.z)};
  // Copy the entire image:
  region.extent = {copySize.width, copySize.height,
                   copySize.depthOrArrayLayers};
  vkCmdCopyImage(
      commandBuffer->commandBuffer,                            // Command buffer
      static_cast<Texture_VK*>(source.texutre)->getVkImage(),  // Source image
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  // Source image layout
      static_cast<Texture_VK*>(destination.texutre)
          ->getVkImage(),                    // Destination image
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  // Destination image layout
      1, &region);                           // Regions
}

auto CommandEncoder_VK::writeTimestamp(QuerySet* querySet,
                                       uint32_t queryIndex) noexcept -> void {}

auto CommandEncoder_VK::resolveQuerySet(QuerySet* querySet, uint32_t firstQuery,
                                        uint32_t queryCount,
                                        Buffer& destination,
                                        uint64_t destinationOffset) noexcept
    -> void {}

auto CommandEncoder_VK::finish(
    std::optional<CommandBufferDescriptor> const& descriptor) noexcept
    -> CommandBuffer* {
  if (vkEndCommandBuffer(commandBuffer->commandBuffer) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to record command buffer!");
  }
  return commandBuffer;
}

auto CommandEncoder_VK::beginDebugUtilsLabelEXT(
    DebugUtilLabelDescriptor const& desc) noexcept -> void {
  VkDebugUtilsLabelEXT debugUtilLabel = {};
  debugUtilLabel.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  debugUtilLabel.pNext = nullptr;
  debugUtilLabel.pLabelName = desc.name.c_str();
  memcpy(debugUtilLabel.color, &(desc.color[0]), sizeof(float) * 4);
  commandBuffer->device->getAdapterVk()
      ->getContext()
      ->vkCmdBeginDebugUtilsLabelEXT(commandBuffer->commandBuffer,
                                     &debugUtilLabel);
}

auto CommandEncoder_VK::endDebugUtilsLabelEXT() noexcept -> void {
  commandBuffer->device->getAdapterVk()
      ->getContext()
      ->vkCmdEndDebugUtilsLabelEXT(commandBuffer->commandBuffer);
}

auto Device_VK::createCommandEncoder(
    CommandEncoderDescriptor const& desc) noexcept
    -> std::unique_ptr<CommandEncoder> {
  std::unique_ptr<CommandEncoder_VK> encoder =
      std::make_unique<CommandEncoder_VK>();
  if (desc.externalCommandBuffer) {
    encoder->commandBuffer =
        static_cast<CommandBuffer_VK*>(desc.externalCommandBuffer);
  } else {
    encoder->commandBufferOnce = graphicPool->allocateCommandBuffer();
    encoder->commandBuffer = encoder->commandBufferOnce.get();
  }
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;
  beginInfo.pInheritanceInfo = nullptr;
  if (vkBeginCommandBuffer(encoder->commandBuffer->commandBuffer, &beginInfo) !=
      VK_SUCCESS) {
    Core::LogManager::Error(
        "VULKAN :: failed to begin recording command buffer!");
  }
  return encoder;
}

#pragma endregion

#pragma region VK_COMPUTE_PASS_ENCODER_IMPL

ComputePassEncoder_VK::~ComputePassEncoder_VK() {}

auto ComputePassEncoder_VK::setBindGroup(
    uint32_t index, BindGroup* bindgroup,
    std::vector<BufferDynamicOffset> const& dynamicOffsets) noexcept -> void {
  vkCmdBindDescriptorSets(
      commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      static_cast<PipelineLayout_VK*>(computePipeline->layout)->pipelineLayout,
      index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set, 0, nullptr);
}

auto ComputePassEncoder_VK::setBindGroup(
    uint32_t index, BindGroup* bindgroup, uint64_t dynamicOffsetDataStart,
    uint32_t dynamicOffsetDataLength) noexcept -> void {
  vkCmdBindDescriptorSets(
      commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      static_cast<PipelineLayout_VK*>(computePipeline->layout)->pipelineLayout,
      index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set, 0, nullptr);
}

auto ComputePassEncoder_VK::pushConstants(void* data, ShaderStagesFlags stages,
                                          uint32_t offset,
                                          uint32_t size) noexcept -> void {
  vkCmdPushConstants(
      commandBuffer->commandBuffer,
      static_cast<PipelineLayout_VK*>(computePipeline->layout)->pipelineLayout,
      getVkShaderStageFlags(stages), offset, size, data);
}

auto ComputePassEncoder_VK::setPipeline(ComputePipeline* pipeline) noexcept
    -> void {
  ComputePipeline_VK* vkpipeline = static_cast<ComputePipeline_VK*>(pipeline);
  computePipeline = vkpipeline;
  vkCmdBindPipeline(commandBuffer->commandBuffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE, vkpipeline->pipeline);
}

auto ComputePassEncoder_VK::dispatchWorkgroups(
    uint32_t workgroupCountX, uint32_t workgroupCountY,
    uint32_t workgroupCountZ) noexcept -> void {
  vkCmdDispatch(commandBuffer->commandBuffer, workgroupCountX, workgroupCountY,
                workgroupCountZ);
}

auto ComputePassEncoder_VK::dispatchWorkgroupsIndirect(
    Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void {}

auto ComputePassEncoder_VK::end() noexcept -> void {
  computePipeline = nullptr;
}

auto CommandEncoder_VK::beginComputePass(
    ComputePassDescriptor const& desc) noexcept
    -> std::unique_ptr<ComputePassEncoder> {
  std::unique_ptr<ComputePassEncoder_VK> computePassEncoder =
      std::make_unique<ComputePassEncoder_VK>();
  computePassEncoder->commandBuffer = this->commandBuffer;
  return computePassEncoder;
}

#pragma endregion

#pragma region VK_FRAMEBUFFER_IMPL

FrameBuffer_VK::FrameBuffer_VK(Device_VK* device,
                               RHI::RenderPassDescriptor const& desc,
                               RenderPass_VK* renderpass)
    : device(device) {
  std::vector<VkImageView> attachments;
  for (int i = 0; i < desc.colorAttachments.size(); ++i) {
    attachments.push_back(
        static_cast<TextureView_VK*>(desc.colorAttachments[i].view)->imageView);
    clearValues.push_back(VkClearValue{VkClearColorValue{
        (float)desc.colorAttachments[i].clearValue.r,
        (float)desc.colorAttachments[i].clearValue.g,
        (float)desc.colorAttachments[i].clearValue.b,
        (float)desc.colorAttachments[i].clearValue.a,
    }});
  }
  if (desc.depthStencilAttachment.view != nullptr) {
    attachments.push_back(
        static_cast<TextureView_VK*>(desc.depthStencilAttachment.view)
            ->imageView);
    VkClearValue clearValue = {};
    clearValue.color = {0.f, 0.f, 0.f, 0.f};
    clearValue.depthStencil = {
        (float)desc.depthStencilAttachment.depthClearValue, (uint32_t)0};
    clearValues.push_back(clearValue);
  }
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = renderpass->renderPass;
  framebufferInfo.attachmentCount = attachments.size();
  framebufferInfo.pAttachments = attachments.data();
  framebufferInfo.width = (desc.colorAttachments.size() > 0)
                              ? desc.colorAttachments[0].view->getWidth()
                              : desc.depthStencilAttachment.view->getWidth();
  framebufferInfo.height = (desc.colorAttachments.size() > 0)
                               ? desc.colorAttachments[0].view->getHeight()
                               : desc.depthStencilAttachment.view->getHeight();
  _width = framebufferInfo.width;
  _height = framebufferInfo.height;
  framebufferInfo.layers = 1;
  if (vkCreateFramebuffer(device->getVkDevice(), &framebufferInfo, nullptr,
                          &framebuffer) != VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to create framebuffer!");
  }
}

FrameBuffer_VK::~FrameBuffer_VK() {
  if (framebuffer)
    vkDestroyFramebuffer(device->getVkDevice(), framebuffer, nullptr);
}

#pragma endregion

#pragma region VK_RenderPipeline_IMPL

RenderPassEncoder_VK::~RenderPassEncoder_VK() {}

auto RenderPassEncoder_VK::setPipeline(RenderPipeline* pipeline) noexcept
    -> void {
  RenderPipeline_VK* vkpipeline = static_cast<RenderPipeline_VK*>(pipeline);
  renderPipeline = vkpipeline;
  vkpipeline->combineRenderPass(renderPass.get());
  vkCmdBindPipeline(commandBuffer->commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS, vkpipeline->pipeline);
}

auto RenderPassEncoder_VK::setIndexBuffer(Buffer* buffer,
                                          IndexFormat indexFormat,
                                          uint64_t offset,
                                          uint64_t size) noexcept -> void {
  vkCmdBindIndexBuffer(commandBuffer->commandBuffer,
                       static_cast<Buffer_VK*>(buffer)->getVkBuffer(), offset,
                       indexFormat == IndexFormat::UINT16_t
                           ? VK_INDEX_TYPE_UINT16
                           : VK_INDEX_TYPE_UINT32);
}

auto RenderPassEncoder_VK::setVertexBuffer(uint32_t slot, Buffer* buffer,
                                           uint64_t offset,
                                           uint64_t size) noexcept -> void {
  VkBuffer vertexBuffers[] = {static_cast<Buffer_VK*>(buffer)->getVkBuffer()};
  VkDeviceSize offsets[] = {offset};
  vkCmdBindVertexBuffers(commandBuffer->commandBuffer, 0, 1, vertexBuffers,
                         offsets);
}

auto RenderPassEncoder_VK::draw(uint32_t vertexCount, uint32_t instanceCount,
                                uint32_t firstVertex,
                                uint32_t firstInstance) noexcept -> void {
  vkCmdDraw(commandBuffer->commandBuffer, vertexCount, instanceCount,
            firstVertex, firstInstance);
}

auto RenderPassEncoder_VK::drawIndexed(uint32_t indexCount,
                                       uint32_t instanceCount,
                                       uint32_t firstIndex, int32_t baseVertex,
                                       uint32_t firstInstance) noexcept
    -> void {
  vkCmdDrawIndexed(commandBuffer->commandBuffer, indexCount, instanceCount,
                   firstIndex, baseVertex, firstInstance);
}

auto RenderPassEncoder_VK::drawIndirect(Buffer* indirectBuffer,
                                        uint64_t indirectOffset) noexcept
    -> void {}

auto RenderPassEncoder_VK::drawIndexedIndirect(Buffer* indirectBuffer,
                                               uint64_t offset,
                                               uint32_t drawCount,
                                               uint32_t stride) noexcept
    -> void {
  vkCmdDrawIndexedIndirect(
      commandBuffer->commandBuffer,
      static_cast<Buffer_VK*>(indirectBuffer)->getVkBuffer(), offset, drawCount,
      stride);
}

auto RenderPassEncoder_VK::setViewport(float x, float y, float width,
                                       float height, float minDepth,
                                       float maxDepth) noexcept -> void {
  VkViewport viewport = {};
  viewport.x = x;
  viewport.y = y;
  viewport.width = width;
  viewport.height = height;
  viewport.minDepth = minDepth;
  viewport.maxDepth = maxDepth;
  vkCmdSetViewport(commandBuffer->commandBuffer, 0, 1, &viewport);
}

auto RenderPassEncoder_VK::setScissorRect(IntegerCoordinate x,
                                          IntegerCoordinate y,
                                          IntegerCoordinate width,
                                          IntegerCoordinate height) noexcept
    -> void {
  VkRect2D scissor;
  scissor.offset.x = x;
  scissor.offset.y = y;
  scissor.extent.width = width;
  scissor.extent.height = height;
  vkCmdSetScissor(commandBuffer->commandBuffer, 0, 1, &scissor);
}

auto RenderPassEncoder_VK::setBlendConstant(Color color) noexcept -> void {}

auto RenderPassEncoder_VK::setStencilReference(StencilValue reference) noexcept
    -> void {}

auto RenderPassEncoder_VK::beginOcclusionQuery(uint32_t queryIndex) noexcept
    -> void {}

auto RenderPassEncoder_VK::endOcclusionQuery() noexcept -> void {}

auto RenderPassEncoder_VK::executeBundles(
    std::vector<RenderBundle> const& bundles) noexcept -> void {}

auto RenderPassEncoder_VK::end() noexcept -> void {
  vkCmdEndRenderPass(commandBuffer->commandBuffer);
}

auto RenderPassEncoder_VK::setBindGroup(
    uint32_t index, BindGroup* bindgroup,
    std::vector<BufferDynamicOffset> const& dynamicOffsets) noexcept -> void {
  vkCmdBindDescriptorSets(
      commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
      static_cast<PipelineLayout_VK*>(
          renderPipeline->fixedFunctionSetttings.pipelineLayout)
          ->pipelineLayout,
      index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set, 0, nullptr);
}

auto RenderPassEncoder_VK::setBindGroup(
    uint32_t index, BindGroup* bindgroup, uint64_t dynamicOffsetDataStart,
    uint32_t dynamicOffsetDataLength) noexcept -> void {
  vkCmdBindDescriptorSets(
      commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
      static_cast<PipelineLayout_VK*>(
          renderPipeline->fixedFunctionSetttings.pipelineLayout)
          ->pipelineLayout,
      index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set, 0, nullptr);
}

auto RenderPassEncoder_VK::pushConstants(void* data, ShaderStagesFlags stages,
                                         uint32_t offset,
                                         uint32_t size) noexcept -> void {
  vkCmdPushConstants(commandBuffer->commandBuffer,
                     static_cast<PipelineLayout_VK*>(
                         renderPipeline->fixedFunctionSetttings.pipelineLayout)
                         ->pipelineLayout,
                     getVkShaderStageFlags(stages), offset, size, data);
}

auto CommandEncoder_VK::beginRenderPass(
    RenderPassDescriptor const& desc) noexcept
    -> std::unique_ptr<RenderPassEncoder> {
  std::unique_ptr<RenderPassEncoder_VK> renderpassEncoder =
      std::make_unique<RenderPassEncoder_VK>();
  renderpassEncoder->renderPass =
      std::make_unique<RenderPass_VK>(commandBuffer->device, desc);
  renderpassEncoder->commandBuffer = commandBuffer;
  renderpassEncoder->frameBuffer = std::make_unique<FrameBuffer_VK>(
      commandBuffer->device, desc, renderpassEncoder->renderPass.get());
  // render pass
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = renderpassEncoder->renderPass->renderPass;
  renderPassInfo.framebuffer = renderpassEncoder->frameBuffer->framebuffer;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent =
      VkExtent2D{renderpassEncoder->frameBuffer->width(),
                 renderpassEncoder->frameBuffer->height()};
  renderPassInfo.pClearValues =
      renderpassEncoder->frameBuffer->clearValues.data();
  renderPassInfo.clearValueCount =
      renderpassEncoder->frameBuffer->clearValues.size();
  vkCmdBeginRenderPass(commandBuffer->commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);
  return renderpassEncoder;
}

#pragma endregion

#pragma region VK_BLAS_IMPL

BLAS_VK::BLAS_VK(Device_VK* device, BLASDescriptor const& descriptor)
    : device(device), descriptor(descriptor) {
  // 1. Declare BLAS geometry infos
  std::vector<VkAccelerationStructureGeometryKHR> geometries;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
  std::vector<uint32_t> primitiveCountArray;
  // transform buffer
  std::vector<RHI::AffineTransformMatrix> affine_transforms;
  for (BLASTriangleGeometry const& triangleDesc : descriptor.triangleGeometries)
    affine_transforms.push_back(
        RHI::AffineTransformMatrix(triangleDesc.transform));
  for (BLASCustomGeometry const& customDesc : descriptor.customGeometries)
    affine_transforms.push_back(
        RHI::AffineTransformMatrix(customDesc.transform));
  std::unique_ptr<RHI::Buffer> transformBuffer =
      device->createDeviceLocalBuffer(
          affine_transforms.data(),
          affine_transforms.size() * sizeof(RHI::AffineTransformMatrix),
          (uint32_t)RHI::BufferUsage::STORAGE |
              (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
              (uint32_t)RHI::BufferUsage::
                  ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);

  uint32_t transformOffset = 0;
  for (BLASTriangleGeometry const& triangleDesc :
       descriptor.triangleGeometries) {
    // 1.1. Get the host / device addresses of the geometry��s buffers
    VkDeviceAddress vertexBufferAddress =
        getBufferVkDeviceAddress(device, triangleDesc.positionBuffer);
    VkDeviceAddress indexBufferAddress =
        getBufferVkDeviceAddress(device, triangleDesc.indexBuffer);
    // 1.2. Describe the instance��s geometry
    VkAccelerationStructureGeometryTrianglesDataKHR triangles = {};
    triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBufferAddress;
    triangles.vertexStride = 3 * sizeof(float);
    triangles.indexType = triangleDesc.indexFormat == IndexFormat::UINT16_t
                              ? VK_INDEX_TYPE_UINT16
                              : VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBufferAddress;
    triangles.maxVertex = triangleDesc.maxVertex;
    triangles.transformData.deviceAddress =
        getBufferVkDeviceAddress(device, transformBuffer.get());
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    geometry.flags = getVkGeometryFlagsKHR(triangleDesc.geometryFlags);
    geometries.push_back(geometry);
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.firstVertex = triangleDesc.firstVertex;
    rangeInfo.primitiveCount = triangleDesc.primitiveCount;
    rangeInfo.primitiveOffset = triangleDesc.primitiveOffset;
    rangeInfo.transformOffset = transformOffset;
    rangeInfos.push_back(rangeInfo);
    primitiveCountArray.push_back(triangleDesc.primitiveCount);
    transformOffset += sizeof(RHI::AffineTransformMatrix);
  }
  std::vector<std::unique_ptr<RHI::Buffer>> aabbBuffers;
  for (BLASCustomGeometry const& customDesc : descriptor.customGeometries) {
    std::unique_ptr<RHI::Buffer> aabbBuffer = device->createDeviceLocalBuffer(
        (void*)customDesc.aabbs.data(),
        customDesc.aabbs.size() * sizeof(Math::bounds3),
        (uint32_t)RHI::BufferUsage::STORAGE |
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)
                RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);
    aabbBuffers.emplace_back(std::move(aabbBuffer));
    VkDeviceAddress dataAddress =
        getBufferVkDeviceAddress(device, aabbBuffers.back().get());
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
    aabbs.data.deviceAddress = dataAddress;
    aabbs.stride = sizeof(Math::bounds3);
    // Setting up the build info of the acceleration (C version, c++ gives wrong
    // type)
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags = getVkGeometryFlagsKHR(customDesc.geometryFlags);
    geometry.geometry.aabbs = aabbs;
    geometries.push_back(geometry);
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.firstVertex = 0;
    rangeInfo.primitiveCount = (uint32_t)customDesc.aabbs.size();  // Nb aabb
    rangeInfo.primitiveOffset = 0;
    rangeInfo.transformOffset = transformOffset;
    rangeInfos.push_back(rangeInfo);
    primitiveCountArray.push_back(customDesc.aabbs.size());
    transformOffset += sizeof(RHI::AffineTransformMatrix);
  }
  // 2. Partially specifying VkAccelerationStructureBuildGeometryInfoKHR and
  // querying
  //	  worst-case memory usage for scratch storage required when building.
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
  buildInfo.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.geometryCount = geometries.size();
  buildInfo.pGeometries = geometries.data();
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  buildInfo.flags = descriptor.allowRefitting
                        ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                        : 0;
  // We will set dstAccelerationStructure and scratchData once
  // we have created those objects.
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
  sizeInfo.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  device->getAdapterVk()->getContext()->vkGetAccelerationStructureBuildSizesKHR(
      device->getVkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo,                  // Pointer to build info
      primitiveCountArray.data(),  // Array of number of primitives per geometry
      &sizeInfo);                  // Output pointer to store sizes
  // 3. Create an empty acceleration structure and its underlying VkBuffer.
  bufferBLAS = device->createBuffer(BufferDescriptor{
      sizeInfo.accelerationStructureSize,
      (uint32_t)BufferUsage::ACCELERATION_STRUCTURE_STORAGE |
          (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
          (uint32_t)BufferUsage::STORAGE,
      BufferShareMode::EXCLUSIVE, (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT});
  VkAccelerationStructureCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  createInfo.type = buildInfo.type;
  createInfo.size = sizeInfo.accelerationStructureSize;
  createInfo.buffer = static_cast<Buffer_VK*>(bufferBLAS.get())->getVkBuffer();
  createInfo.offset = 0;
  /*VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR*/
  device->getAdapterVk()->getContext()->vkCreateAccelerationStructureKHR(
      device->getVkDevice(), &createInfo, nullptr, &blas);
  buildInfo.dstAccelerationStructure = blas;
  // 5. Allocate scratch space.
  std::unique_ptr<Buffer> scratchBuffer = device->createBuffer(BufferDescriptor{
      sizeInfo.buildScratchSize,
      (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
          (uint32_t)BufferUsage::STORAGE,
      BufferShareMode::EXCLUSIVE, (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT});
  buildInfo.scratchData.deviceAddress =
      getBufferVkDeviceAddress(device, scratchBuffer.get());
  // 6. Call vkCmdBuildAccelerationStructuresKHR() with a populated
  //	  VkAccelerationStructureBuildSizesInfoKHR structand range info to
  //    build the geometry into an acceleration structure.
  VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = rangeInfos.data();
  std::unique_ptr<CommandEncoder> commandEncoder =
      device->createCommandEncoder({nullptr});
  CommandEncoder_VK* commandEncoderVK =
      static_cast<CommandEncoder_VK*>(commandEncoder.get());
  device->getAdapterVk()->getContext()->vkCmdBuildAccelerationStructuresKHR(
      commandEncoderVK->commandBuffer
          ->commandBuffer,  // The command buffer to record the command
      1,                    // Number of acceleration structures to build
      &buildInfo,           // Array of ...BuildGeometryInfoKHR objects
      &pRangeInfo);         // Array of ...RangeInfoKHR objects
  device->getGraphicsQueue()->submit({commandEncoder->finish({})});
  device->waitIdle();
}

BLAS_VK::BLAS_VK(Device_VK* device, BLAS_VK* src)
    : device(device), descriptor(src->descriptor) {
  //// 1. Get the host or device addresses of the geometry��s buffers
  // VkBufferDeviceAddressInfo deviceAddressInfo = {};
  // deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  // deviceAddressInfo.buffer =
  // static_cast<Buffer_VK*>(descriptor.vertexBuffer)->getVkBuffer();
  // VkDeviceAddress vertexBufferAddress =
  // vkGetBufferDeviceAddress(device->getVkDevice(), &deviceAddressInfo);
  // deviceAddressInfo.buffer =
  // static_cast<Buffer_VK*>(descriptor.indexBuffer)->getVkBuffer();
  // VkDeviceAddress indexBufferAddress =
  // vkGetBufferDeviceAddress(device->getVkDevice(), &deviceAddressInfo);
  //// 2. Describe the instance��s geometry
  // VkAccelerationStructureGeometryTrianglesDataKHR triangles = {};
  // triangles.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  // triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  // triangles.vertexData.deviceAddress = vertexBufferAddress;
  // triangles.vertexStride = 3 * sizeof(float);
  // triangles.indexType = descriptor.indexFormat == IndexFormat::UINT16_t ?
  // VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
  // triangles.indexData.deviceAddress = indexBufferAddress;
  // triangles.maxVertex = descriptor.maxVertex;
  // triangles.transformData.deviceAddress = 0; // No transform
  // VkAccelerationStructureGeometryKHR geometry = {};
  // geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  // geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  // geometry.geometry.triangles = triangles;
  // geometry.flags = getVkGeometryFlagsKHR(descriptor.geometryFlags);
  //// 3. Determine the worst-case memory requirements for the AS and
  ////    for scratch storage required when building.
  // VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
  // rangeInfo.firstVertex = 0;
  // rangeInfo.primitiveCount = descriptor.primitiveCount;
  // rangeInfo.primitiveOffset = 0;
  // rangeInfo.transformOffset = 0;
  // VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
  // buildInfo.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  // buildInfo.flags =
  // VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  // buildInfo.geometryCount = 1;
  // buildInfo.pGeometries = &geometry;
  // buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  // buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  // buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  // buildInfo.flags = 0;
  // if (descriptor.allowRefitting) buildInfo.flags |=
  // VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR; if
  // (descriptor.allowCompaction) buildInfo.flags |=
  // VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  //// We will set dstAccelerationStructure and scratchData once
  //// we have created those objects.
  // VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
  // sizeInfo.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  // device->getAdapterVk()->getContext()->vkGetAccelerationStructureBuildSizesKHR(
  //	device->getVkDevice(),
  //	VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
  //	&buildInfo,
  //	&rangeInfo.primitiveCount,
  //	&sizeInfo);
  //// create blas
  // bufferBLAS = device->createBuffer(BufferDescriptor{
  //	sizeInfo.accelerationStructureSize,
  //	(uint32_t)BufferUsage::ACCELERATION_STRUCTURE_STORAGE |
  //(uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
  //(uint32_t)BufferUsage::STORAGE, 	BufferShareMode::EXCLUSIVE,
  //	(uint32_t)MemoryProperty::DEVICE_LOCAL_BIT });
  // VkAccelerationStructureCreateInfoKHR createInfo = {};
  // createInfo.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR; createInfo.type =
  // buildInfo.type; createInfo.size = sizeInfo.accelerationStructureSize;
  // createInfo.buffer =
  // static_cast<Buffer_VK*>(bufferBLAS.get())->getVkBuffer(); createInfo.offset
  // = 0;
  // device->getAdapterVk()->getContext()->vkCreateAccelerationStructureKHR(device->getVkDevice(),
  // &createInfo, nullptr, &blas); buildInfo.dstAccelerationStructure = blas;
}

BLAS_VK::~BLAS_VK() {
  if (blas)
    device->getAdapterVk()->getContext()->vkDestroyAccelerationStructureKHR(
        device->getVkDevice(), blas, nullptr);
}

auto Device_VK::createBLAS(BLASDescriptor const& desc) noexcept
    -> std::unique_ptr<BLAS> {
  if (desc.customGeometries.size() == 0 &&
      desc.triangleGeometries.size() == 0) {
    Core::LogManager::get()->Error(
        "RHI :: Vulkan :: Create BLAS with no input geometry!");
    return nullptr;
  }
  return std::make_unique<BLAS_VK>(this, desc);
}

auto CommandEncoder_VK::cloneBLAS(BLAS* src) noexcept -> std::unique_ptr<BLAS> {
  BLAS_VK* src_blas = static_cast<BLAS_VK*>(src);
  std::unique_ptr<BLAS_VK> dst_blas =
      std::make_unique<BLAS_VK>(src_blas->device, src_blas);
  VkCopyAccelerationStructureInfoKHR copyInfo = {};
  copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
  copyInfo.src = src_blas->blas;
  copyInfo.dst = dst_blas->blas;
  copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
  dst_blas->device->getAdapterVk()
      ->getContext()
      ->vkCmdCopyAccelerationStructureKHR(commandBuffer->commandBuffer,
                                          &copyInfo);
  return dst_blas;
}

auto CommandEncoder_VK::updateBLAS(BLAS* src, Buffer* vertexBuffer,
                                   Buffer* indexBuffer) noexcept -> void {
  // BLAS_VK* blas = static_cast<BLAS_VK*>(src);
  // blas->descriptor.vertexBuffer = vertexBuffer;
  // blas->descriptor.indexBuffer = indexBuffer;
  //// 1. Get the host or device addresses of the geometry��s buffers
  // VkBufferDeviceAddressInfo deviceAddressInfo = {};
  // deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  // deviceAddressInfo.buffer =
  // static_cast<Buffer_VK*>(blas->descriptor.vertexBuffer)->getVkBuffer();
  // VkDeviceAddress vertexBufferAddress =
  // vkGetBufferDeviceAddress(blas->device->getVkDevice(), &deviceAddressInfo);
  // deviceAddressInfo.buffer =
  // static_cast<Buffer_VK*>(blas->descriptor.indexBuffer)->getVkBuffer();
  // VkDeviceAddress indexBufferAddress =
  // vkGetBufferDeviceAddress(blas->device->getVkDevice(), &deviceAddressInfo);
  //// 2. Describe the instance��s geometry
  // VkAccelerationStructureGeometryTrianglesDataKHR triangles = {};
  // triangles.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  // triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  // triangles.vertexData.deviceAddress = vertexBufferAddress;
  // triangles.vertexStride = 3 * sizeof(float);
  // triangles.indexType = blas->descriptor.indexFormat == IndexFormat::UINT16_t
  // ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
  // triangles.indexData.deviceAddress = indexBufferAddress;
  // triangles.maxVertex = blas->descriptor.maxVertex;
  // triangles.transformData.deviceAddress = 0; // No transform
  // VkAccelerationStructureGeometryKHR geometry = {};
  // geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  // geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  // geometry.geometry.triangles = triangles;
  // geometry.flags = getVkGeometryFlagsKHR(blas->descriptor.geometryFlags);
  //// 3. Determine the worst-case memory requirements for the AS and
  ////    for scratch storage required when building.
  // VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
  // rangeInfo.firstVertex = 0;
  // rangeInfo.primitiveCount = blas->descriptor.primitiveCount;
  // rangeInfo.primitiveOffset = 0;
  // rangeInfo.transformOffset = 0;
  // VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
  // buildInfo.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  // buildInfo.flags =
  // VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  // buildInfo.geometryCount = 1;
  // buildInfo.pGeometries = &geometry;
  // buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
  // buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  // buildInfo.srcAccelerationStructure = blas->blas;
  // buildInfo.flags = blas->descriptor.allowRefitting ?
  // VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0;
  //// We will set dstAccelerationStructure and scratchData once
  //// we have created those objects.
  // VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
  // sizeInfo.sType =
  // VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  // blas->device->getAdapterVk()->getContext()->vkGetAccelerationStructureBuildSizesKHR(
  //	blas->device->getVkDevice(),
  //	VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
  //	&buildInfo,
  //	&rangeInfo.primitiveCount,
  //	&sizeInfo);
  // buildInfo.dstAccelerationStructure = blas->blas;
  //// 5. Allocate scratch space.
  // std::unique_ptr<Buffer> scratchBuffer =
  // blas->device->createBuffer(BufferDescriptor{
  // sizeInfo.buildScratchSize,
  //	(uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
  //(uint32_t)BufferUsage::STORAGE, 	BufferShareMode::EXCLUSIVE,
  //	(uint32_t)MemoryProperty::DEVICE_LOCAL_BIT });
  // deviceAddressInfo.buffer =
  // static_cast<Buffer_VK*>(scratchBuffer.get())->getVkBuffer();
  // buildInfo.scratchData.deviceAddress =
  // vkGetBufferDeviceAddress(blas->device->getVkDevice(), &deviceAddressInfo);
  //// 6. Call vkCmdBuildAccelerationStructuresKHR() with a populated
  ////	  VkAccelerationStructureBuildSizesInfoKHR structand range info to
  ////    build the geometry into an acceleration structure.
  // VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;
  // blas->device->getAdapterVk()->getContext()->vkCmdBuildAccelerationStructuresKHR(
  //	commandBuffer->commandBuffer, // The command buffer to record the
  // command 	1, // Number of acceleration structures to build 	&buildInfo,
  // // Array of ...BuildGeometryInfoKHR objects 	&pRangeInfo); // Array
  // of
  //...RangeInfoKHR objects
}

#pragma endregion

#pragma region VK_TLAS_IMPL

TLAS_VK::TLAS_VK(Device_VK* device, TLASDescriptor const& descriptor)
    : device(device) {
  // specify an instance
  //  Zero-initialize.
  std::vector<VkAccelerationStructureInstanceKHR> instances(
      descriptor.instances.size());
  for (int i = 0; i < instances.size(); ++i) {
    // frst get the device address of one or more BLASs
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
    addressInfo.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure =
        static_cast<BLAS_VK*>(descriptor.instances[i].blas)->blas;
    VkDeviceAddress blasAddress =
        device->getAdapterVk()
            ->getContext()
            ->vkGetAccelerationStructureDeviceAddressKHR(device->getVkDevice(),
                                                         &addressInfo);
    VkAccelerationStructureInstanceKHR& instance = instances[i];
    //  Set the instance transform to given transform.
    for (int m = 0; m < 3; ++m)
      for (int n = 0; n < 4; ++n)
        instance.transform.matrix[m][n] =
            descriptor.instances[i].transform.data[m][n];
    instance.instanceCustomIndex = descriptor.instances[i].instanceCustomIndex;
    instance.mask = descriptor.instances[i].mask;
    instance.instanceShaderBindingTableRecordOffset =
        descriptor.instances[i].instanceShaderBindingTableRecordOffset;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blasAddress;
  }
  // 2. Uploading an instance buffer of one instance to the VkDevice and waiting
  // for it to complete.
  std::unique_ptr<Buffer> bufferInstances = nullptr;
  if (instances.size() > 0) {
    bufferInstances = device->createDeviceLocalBuffer(
        (void*)instances.data(),
        sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
        (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)
                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);
  }
  //	  Add a memory barrier to ensure that createBuffer's upload command
  //	  finishes before starting the TLAS build.
  //	  *: here createDeviceLocalBuffer already implicit use a synchronized
  // fence.
  // 3. Specifying range information for the TLAS build.
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.primitiveCount = instances.size();  // Number of instances
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  // 4. Constructing a VkAccelerationStructureGeometryKHR struct of instances
  VkAccelerationStructureGeometryInstancesDataKHR instancesVk = {};
  instancesVk.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  instancesVk.arrayOfPointers = VK_FALSE;
  instancesVk.data.deviceAddress =
      (bufferInstances != nullptr)
          ? getBufferVkDeviceAddress(device, bufferInstances.get())
          : 0;
  // Like creating the BLAS, point to the geometry (in this case, the
  // instances) in a polymorphic object.
  VkAccelerationStructureGeometryKHR geometry = {};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances = instancesVk;
  // 5. Allocating and building a top-level acceleration structure.
  // Create the build info: in this case, pointing to only one
  // geometry object.
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
  buildInfo.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &geometry;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  if (descriptor.allowRefitting) {
    buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  }
  // Query the worst-case AS size and scratch space size based on
  // the number of instances (in this case, 1).
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
  sizeInfo.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  device->getAdapterVk()->getContext()->vkGetAccelerationStructureBuildSizesKHR(
      device->getVkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo, &rangeInfo.primitiveCount, &sizeInfo);
  // Allocate a buffer for the acceleration structure.
  bufferTLAS = device->createBuffer(BufferDescriptor{
      sizeInfo.accelerationStructureSize,
      (uint32_t)BufferUsage::ACCELERATION_STRUCTURE_STORAGE |
          (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
          (uint32_t)BufferUsage::STORAGE,
      BufferShareMode::EXCLUSIVE, (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT});
  // Create the acceleration structure object.
  // (Data has not yet been set.)
  VkAccelerationStructureCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  createInfo.type = buildInfo.type;
  createInfo.size = sizeInfo.accelerationStructureSize;
  createInfo.buffer = static_cast<Buffer_VK*>(bufferTLAS.get())->getVkBuffer();
  createInfo.offset = 0;
  device->getAdapterVk()->getContext()->vkCreateAccelerationStructureKHR(
      device->getVkDevice(), &createInfo, nullptr, &tlas);
  buildInfo.dstAccelerationStructure = tlas;
  // Allocate the scratch buffer holding temporary build data.
  std::unique_ptr<Buffer> scratchBuffer = device->createBuffer(BufferDescriptor{
      sizeInfo.buildScratchSize,
      (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
          (uint32_t)BufferUsage::STORAGE,
      BufferShareMode::EXCLUSIVE, (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT});
  buildInfo.scratchData.deviceAddress =
      getBufferVkDeviceAddress(device, scratchBuffer.get());
  // Create a one-element array of pointers to range info objects.
  VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;
  // Build the TLAS.
  std::unique_ptr<CommandEncoder> commandEncoder =
      device->createCommandEncoder({nullptr});
  CommandEncoder_VK* commandEncoderVK =
      static_cast<CommandEncoder_VK*>(commandEncoder.get());
  device->getAdapterVk()->getContext()->vkCmdBuildAccelerationStructuresKHR(
      commandEncoderVK->commandBuffer
          ->commandBuffer,  // The command buffer to record the command
      1,                    // Number of acceleration structures to build
      &buildInfo,           // Array of ...BuildGeometryInfoKHR objects
      &pRangeInfo);         // Array of ...RangeInfoKHR objects
  device->getGraphicsQueue()->submit({commandEncoder->finish({})});
  device->getGraphicsQueue()->waitIdle();
}

TLAS_VK::~TLAS_VK() {
  if (tlas)
    device->getAdapterVk()->getContext()->vkDestroyAccelerationStructureKHR(
        device->getVkDevice(), tlas, nullptr);
}

auto Device_VK::createTLAS(TLASDescriptor const& desc) noexcept
    -> std::unique_ptr<TLAS> {
  return std::make_unique<TLAS_VK>(this, desc);
}

#pragma endregion

#pragma region VK_RAY_TRACING_PIPELINE_IMPL

RayTracingPipeline_VK::RayTracingPipeline_VK(
    Device_VK* device, RayTracingPipelineDescriptor const& desc)
    : device(device),
      pipelineLayout(static_cast<PipelineLayout_VK*>(desc.layout)) {
  int rayMissBegin = 0;
  int rayMissCount = 0;
  int closetHitBegin = 0;
  int closetHitCount = 0;
  int hitGroupCount = 0;
  int anyHitBegin = 0;
  int anyHitCount = 0;
  int intersectionBegin = 0;
  int intersectionCount = 0;
  int callableBegin = 0;
  int callableCount = 0;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> rtsgci{};
  // First create RT Pipeline
  {
    // Creating a table of VkPipelineShaderStageCreateInfo objects, containing
    // all shader stages.
    std::vector<VkPipelineShaderStageCreateInfo> pssci = {};
    // push ray generation SBT shader stages
    if (desc.sbtsDescriptor.rgenSBT.rgenRecord.rayGenShader) {
      pssci.push_back(static_cast<ShaderModule_VK*>(
                          desc.sbtsDescriptor.rgenSBT.rgenRecord.rayGenShader)
                          ->shaderStageInfo);
    }
    // push ray miss SBT shader stages
    if (desc.sbtsDescriptor.missSBT.rmissRecords.size() > 0) {
      rayMissBegin = pssci.size();
      rayMissCount = desc.sbtsDescriptor.missSBT.rmissRecords.size();
      for (auto& rmissRecord : desc.sbtsDescriptor.missSBT.rmissRecords) {
        pssci.push_back(static_cast<ShaderModule_VK*>(rmissRecord.missShader)
                            ->shaderStageInfo);
      }
    }
    // push hit groups SBT shader stages
    if (desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords.size() > 0) {
      // closet hit shaders
      closetHitBegin = pssci.size();
      for (auto& hitGroupRecord :
           desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords) {
        ++hitGroupCount;
        if (hitGroupRecord.closetHitShader) {
          pssci.push_back(
              static_cast<ShaderModule_VK*>(hitGroupRecord.closetHitShader)
                  ->shaderStageInfo);
          ++closetHitCount;
        }
      }
      // any hit shaders
      anyHitBegin = pssci.size();
      for (auto& hitGroupRecord :
           desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords) {
        if (hitGroupRecord.anyHitShader) {
          pssci.push_back(
              static_cast<ShaderModule_VK*>(hitGroupRecord.anyHitShader)
                  ->shaderStageInfo);
          ++anyHitCount;
        }
      }
      // intersection shader
      intersectionBegin = pssci.size();
      for (auto& hitGroupRecord :
           desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords) {
        if (hitGroupRecord.intersectionShader) {
          pssci.push_back(
              static_cast<ShaderModule_VK*>(hitGroupRecord.intersectionShader)
                  ->shaderStageInfo);
          ++intersectionCount;
        }
      }
    }
    // push callable SBT shader stages
    callableBegin = pssci.size();
    callableCount = desc.sbtsDescriptor.callableSBT.callableRecords.size();
    if (desc.sbtsDescriptor.callableSBT.callableRecords.size() > 0) {
      for (auto& callableRecord :
           desc.sbtsDescriptor.callableSBT.callableRecords) {
        pssci.push_back(
            static_cast<ShaderModule_VK*>(callableRecord.callableShader)
                ->shaderStageInfo);
      }
    }
    // 1.2. Then we make groups point to the shader stages. Each group can point
    // to one or two shader stages depending on the type, by specifying the
    // index in the stages array. These groups of handles then become the most
    // important part of the entries in the shader binding table. Stores the
    // indices of stages in each group. Enumerating the elements of an array of
    // shader groups, which contains one ray generation group, one miss group,
    // and one hit group.
    VkRayTracingShaderGroupCreateInfoKHR rtsgTemplate = {};
    // inita a template rtsgci
    rtsgTemplate.sType =
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rtsgTemplate.generalShader = VK_SHADER_UNUSED_KHR;
    rtsgTemplate.anyHitShader = VK_SHADER_UNUSED_KHR;
    rtsgTemplate.closestHitShader = VK_SHADER_UNUSED_KHR;
    rtsgTemplate.intersectionShader = VK_SHADER_UNUSED_KHR;
    if (desc.sbtsDescriptor.rgenSBT.rgenRecord.rayGenShader) {
      VkRayTracingShaderGroupCreateInfoKHR rtsg = rtsgTemplate;
      rtsg.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      rtsg.generalShader = 0;
      rtsgci.push_back(rtsg);
    }
    if (desc.sbtsDescriptor.missSBT.rmissRecords.size() > 0) {
      for (int i = 0; i < desc.sbtsDescriptor.missSBT.rmissRecords.size();
           ++i) {
        VkRayTracingShaderGroupCreateInfoKHR rtsg = rtsgTemplate;
        rtsg.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        rtsg.generalShader = rayMissBegin + i;
        rtsgci.push_back(rtsg);
      }
    }
    if (desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords.size() > 0) {
      int closetHitIdx = 0, anyHitIdx = 0, intersectionIdx = 0;
      for (int i = 0;
           i < desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords.size(); ++i) {
        VkRayTracingShaderGroupCreateInfoKHR rtsg = rtsgTemplate;
        rtsg.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        auto& hitGroupRecord =
            desc.sbtsDescriptor.hitGroupSBT.hitGroupRecords[i];
        if (hitGroupRecord.closetHitShader)
          rtsg.closestHitShader = closetHitBegin + closetHitIdx++;
        if (hitGroupRecord.anyHitShader)
          rtsg.anyHitShader = anyHitBegin + anyHitIdx++;
        if (hitGroupRecord.intersectionShader)
          rtsg.intersectionShader = intersectionBegin + intersectionIdx++;
        if (hitGroupRecord.intersectionShader)
          rtsg.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        rtsgci.push_back(rtsg);
      }
    }
    if (desc.sbtsDescriptor.callableSBT.callableRecords.size() > 0) {
      for (int i = 0;
           i < desc.sbtsDescriptor.callableSBT.callableRecords.size(); ++i) {
        VkRayTracingShaderGroupCreateInfoKHR rtsg = rtsgTemplate;
        rtsg.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        rtsg.generalShader = callableBegin + i;
        rtsgci.push_back(rtsg);
      }
    }
    // Now, describe the ray tracing pipeline.
    VkRayTracingPipelineCreateInfoKHR rtpci = {};
    rtpci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtpci.stageCount = uint32_t(pssci.size());
    rtpci.pStages = pssci.data();
    rtpci.groupCount = uint32_t(rtsgci.size());
    rtpci.pGroups = rtsgci.data();
    rtpci.maxPipelineRayRecursionDepth =
        desc.maxPipelineRayRecursionDepth;  // Depth of call tree
    rtpci.pLibraryInfo = nullptr;
    rtpci.layout = static_cast<PipelineLayout_VK*>(desc.layout)->pipelineLayout;
    // create the ray tracing pipeline
    device->getAdapterVk()->getContext()->vkCreateRayTracingPipelinesKHR(
        device->getVkDevice(),  // The VkDevice
        VK_NULL_HANDLE,         // Don't request deferral
        VK_NULL_HANDLE,         // No Pipeline Cahce (?)
        1, &rtpci,              // Array of structures
        nullptr,                // Default host allocator
        &pipeline);             // Output VkPipelines
  }
  // Second create SBTs
  {
    // 1. Get the properties of ray tracing pipelines on this device
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties =
        static_cast<RayTracingExtension_VK*>(device->getRayTracingExtension())
            ->vkRayTracingProperties;
    // Computing a valid size and stride for the SBT, first, get the number of
    // groups
    auto groupCount = uint32_t(rtsgci.size());
    // The size of a program identifier
    uint32_t groupHandleSize = rtPipelineProperties.shaderGroupHandleSize;
    uint32_t baseAlignment = rtPipelineProperties.shaderGroupBaseAlignment;
    uint32_t handleAlignment = rtPipelineProperties.shaderGroupHandleAlignment;
    // Compute the stride between shader binding table (SBT) records / actual
    // size needed per SBT entry. This must be:
    // - Greater than rtPipelineProperties.shaderGroupHandleSize (since a record
    // contains a shader group handle)
    // - A multiple of rtPipelineProperties.shaderGroupHandleAlignment
    // - Less than or equal to rtPipelineProperties.maxShaderGroupStride
    // In addition, each SBT must start at a multiple of
    // rtPipelineProperties.shaderGroupBaseAlignment.
    // Since we store all records contiguously in a single SBT, we assert that
    // sbtBaseAlignment is a multiple of sbtHandleAlignment, round sbtHeaderSize
    // up to a multiple of sbtBaseAlignment, and then assert that the result is
    // less than or equal to maxShaderGroupStride.
    uint32_t groupSizeAligned = Math::alignUp(groupHandleSize, baseAlignment);
    // ray gen region
    rayGenRegion.stride = Math::alignUp(
        groupHandleSize, rtPipelineProperties.shaderGroupBaseAlignment);
    rayGenRegion.size =
        rayGenRegion.stride;  // The size member of pRayGenShaderBindingTable
                              // must be equal to its stride member
    // miss region
    missRegion.stride = groupSizeAligned;
    missRegion.size =
        Math::alignUp(rayMissCount * groupSizeAligned,
                      rtPipelineProperties.shaderGroupBaseAlignment);
    // hit region
    hitRegion.stride = groupSizeAligned;
    hitRegion.size =
        Math::alignUp(hitGroupCount * groupSizeAligned,
                      rtPipelineProperties.shaderGroupBaseAlignment);
    // callable region
    callableRegion.stride = groupSizeAligned;
    callableRegion.size =
        Math::alignUp(callableCount * groupSizeAligned,
                      rtPipelineProperties.shaderGroupBaseAlignment);
    // Allocating and writing shader handles from the ray tracing pipeline into
    // the SBT Fetch all the shader handles used in the pipeline. This is opaque
    // data, so we store it in a vector of bytes. Bytes needed for the SBT.
    uint32_t dataSize = groupCount * groupHandleSize;
    std::vector<uint8_t> shaderHandleStorage(dataSize);
    device->getAdapterVk()->getContext()->vkGetRayTracingShaderGroupHandlesKHR(
        device->getVkDevice(),        // The device
        pipeline,                     // The ray tracing pipeline
        0,                            // Index of the group to start from
        groupCount,                   // The number of groups
        dataSize,                     // Size of the output buffer in bytes
        shaderHandleStorage.data());  // The output buffer
    // Allocate a buffer for storing the SBT.
    VkDeviceSize sbtSize = rayGenRegion.size + missRegion.size +
                           hitRegion.size + callableRegion.size;
    SBTBuffer =
        device->createBuffer({sbtSize,
                              (uint32_t)BufferUsage::COPY_SRC |
                                  (uint32_t)BufferUsage::SHADER_DEVICE_ADDRESS |
                                  (uint32_t)BufferUsage::SHADER_BINDING_TABLE,
                              BufferShareMode::EXCLUSIVE,
                              (uint32_t)MemoryProperty::HOST_VISIBLE_BIT |
                                  (uint32_t)MemoryProperty::HOST_COHERENT_BIT});
    // Copy the shader group handles to the SBT.
    std::future<bool> sync =
        SBTBuffer->mapAsync((uint32_t)MapMode::WRITE, 0, sbtSize);
    if (sync.get()) {
      void* mapped = SBTBuffer->getMappedRange(0, sbtSize);
      auto* pData = reinterpret_cast<uint8_t*>(mapped);
      for (uint32_t g = 0; g < groupCount; g++) {
        memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize,
               groupHandleSize);
        pData += groupSizeAligned;
      }
    }
    SBTBuffer->unmap();
    // VkCmdTraceRaysKHR uses VkStridedDeviceAddressregionKHR objects to say
    // where each block of shaders is held in memory. These could change per
    // draw call, but let's create them up front since they're the same every
    // time here. �� first fetch the device address of the SBT buffer
    VkBufferDeviceAddressInfo deviceAddressInfo = {};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    deviceAddressInfo.buffer =
        static_cast<Buffer_VK*>(SBTBuffer.get())->getVkBuffer();
    VkDeviceAddress SBTBufferAddress =
        vkGetBufferDeviceAddress(device->getVkDevice(), &deviceAddressInfo);
    // �� then fill the address infos into all kinds of regions
    rayGenRegion.deviceAddress = SBTBufferAddress;
    missRegion.deviceAddress = SBTBufferAddress + rayGenRegion.size;
    hitRegion.deviceAddress =
        SBTBufferAddress + rayGenRegion.size + missRegion.size;
    callableRegion.deviceAddress = (callableCount == 0) ? 0 :
        SBTBufferAddress + rayGenRegion.size + missRegion.size + hitRegion.size;
  }
}

RayTracingPipeline_VK::~RayTracingPipeline_VK() {
  if (pipeline) vkDestroyPipeline(device->getVkDevice(), pipeline, nullptr);
}

auto RayTracingPipeline_VK::setName(std::string const& name) -> void {
  VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
  objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  objectNameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
  objectNameInfo.objectHandle = uint64_t(pipeline);
  objectNameInfo.pObjectName = name.c_str();
  device->getAdapterVk()->getContext()->vkSetDebugUtilsObjectNameEXT(
      device->getVkDevice(), &objectNameInfo);
}

#pragma endregion

#pragma region VK_RAY_TRACING_PASS_ENCODER_IMPL

RayTracingPassEncoder_VK::RayTracingPassEncoder_VK(
    CommandEncoder_VK* commandEncoder, RayTracingPassDescriptor const& desc)
    : commandEncoder(commandEncoder) {}

RayTracingPassEncoder_VK::~RayTracingPassEncoder_VK() {}

auto RayTracingPassEncoder_VK::setPipeline(
    RayTracingPipeline* pipeline) noexcept -> void {
  raytracingPipeline = static_cast<RayTracingPipeline_VK*>(pipeline);
  vkCmdBindPipeline(commandEncoder->commandBuffer->commandBuffer,
                    VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                    raytracingPipeline->pipeline);
}

auto RayTracingPassEncoder_VK::setBindGroup(
    uint32_t index, BindGroup* bindgroup,
    std::vector<BufferDynamicOffset> const& dynamicOffsets) noexcept -> void {
  vkCmdBindDescriptorSets(commandEncoder->commandBuffer->commandBuffer,
                          VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                          raytracingPipeline->pipelineLayout->pipelineLayout,
                          index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set,
                          0, nullptr);
}

auto RayTracingPassEncoder_VK::setBindGroup(
    uint32_t index, BindGroup* bindgroup, uint64_t dynamicOffsetDataStart,
    uint32_t dynamicOffsetDataLength) noexcept -> void {
  vkCmdBindDescriptorSets(commandEncoder->commandBuffer->commandBuffer,
                          VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                          raytracingPipeline->pipelineLayout->pipelineLayout,
                          index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set,
                          0, nullptr);
}

auto RayTracingPassEncoder_VK::pushConstants(void* data,
                                             ShaderStagesFlags stages,
                                             uint32_t offset,
                                             uint32_t size) noexcept -> void {
  vkCmdPushConstants(commandEncoder->commandBuffer->commandBuffer,
                     raytracingPipeline->pipelineLayout->pipelineLayout,
                     getVkShaderStageFlags(stages), offset, size, data);
}

auto RayTracingPassEncoder_VK::traceRays(uint32_t width, uint32_t height,
                                         uint32_t depth) noexcept -> void {
  commandEncoder->commandBuffer->device->getAdapterVk()
      ->getContext()
      ->vkCmdTraceRaysKHR(
          commandEncoder->commandBuffer->commandBuffer,
          &raytracingPipeline->rayGenRegion,    // Raygen Shader Binding Table
          &raytracingPipeline->missRegion,      // Miss Shader Binding Table
          &raytracingPipeline->hitRegion,       // Hit Shader Binding Table
          &raytracingPipeline->callableRegion,  // Callable Shader Binding Table
          width, height, depth);
}

auto RayTracingPassEncoder_VK::traceRaysIndirect(
    Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void {
  VkBufferDeviceAddressInfo deviceAddressInfo = {};
  deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  deviceAddressInfo.buffer =
      static_cast<Buffer_VK*>(indirectBuffer)->getVkBuffer();
  VkDeviceAddress indirectBufferAddress = vkGetBufferDeviceAddress(
      commandEncoder->commandBuffer->device->getVkDevice(), &deviceAddressInfo);
  commandEncoder->commandBuffer->device->getAdapterVk()
      ->getContext()
      ->vkCmdTraceRaysIndirectKHR(
          commandEncoder->commandBuffer->commandBuffer,
          &raytracingPipeline->rayGenRegion,    // Raygen Shader Binding Table
          &raytracingPipeline->missRegion,      // Miss Shader Binding Table
          &raytracingPipeline->hitRegion,       // Hit Shader Binding Table
          &raytracingPipeline->callableRegion,  // Callable Shader Binding Table
          indirectBufferAddress);
}

auto RayTracingPassEncoder_VK::end() noexcept -> void {}

auto CommandEncoder_VK::beginRayTracingPass(
    RayTracingPassDescriptor const& desc) noexcept
    -> std::unique_ptr<RayTracingPassEncoder> {
  return std::make_unique<RayTracingPassEncoder_VK>(this, desc);
}

#pragma endregion

#pragma region VK_RAY_TRACING_EXT_IMPL

auto Device_VK::initRayTracingExt() noexcept -> void {
  raytracingExt = std::make_unique<RayTracingExtension_VK>();
  raytracingExt->vkRayTracingProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  // fetch properties
  VkPhysicalDeviceProperties2 prop2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &raytracingExt->vkRayTracingProperties;
  vkGetPhysicalDeviceProperties2(getAdapterVk()->getVkPhysicalDevice(), &prop2);
}

auto Device_VK::getRayTracingExtension() noexcept -> RayTracingExtension* {
  return raytracingExt.get();
}

auto Device_VK::createRayTracingPipeline(
    RayTracingPipelineDescriptor const& desc) noexcept
    -> std::unique_ptr<RayTracingPipeline> {
  return std::make_unique<RayTracingPipeline_VK>(this, desc);
}

auto Device_VK::createFence() noexcept -> std::unique_ptr<Fence> {
  return std::make_unique<Fence_VK>(this);
}

#pragma endregion

#pragma region DELAYED_IMPL

BindGroup_VK::BindGroup_VK(Device_VK* device, BindGroupDescriptor const& desc) {
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = device->getBindGroupPool()->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts =
      &static_cast<BindGroupLayout_VK*>(desc.layout)->layout;

  bool hasBindless = false;
  for (auto& entry : desc.entries) {
    if (entry.resource.bindlessTextures.size() > 0) hasBindless = true;
  }
  VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
  uint32_t max_binding = 200 - 1;
  if (hasBindless) {
    count_info.descriptorSetCount = 1;
    count_info.pDescriptorCounts = &max_binding;
    allocInfo.pNext = &count_info;
  }

  if (vkAllocateDescriptorSets(device->getVkDevice(), &allocInfo, &set) !=
      VK_SUCCESS) {
    Core::LogManager::Error("VULKAN :: failed to allocate descriptor sets!");
  }

  layout = desc.layout;
  this->device = device;
  updateBinding(desc.entries);
}

auto BindGroup_VK::updateBinding(
    std::vector<BindGroupEntry> const& entries) noexcept -> void {
  // configure the descriptors
  uint32_t bufferCounts = 0;
  uint32_t imageCounts = 0;
  uint32_t accStructCounts = 0;
  for (auto& entry : entries) {
    if (entry.resource.bufferBinding.has_value())
      ++bufferCounts;
    else if (entry.resource.textureView)
      ++imageCounts;
    else if (entry.resource.tlas)
      ++accStructCounts;
  }
  std::vector<VkWriteDescriptorSet> descriptorWrites = {};
  std::vector<VkDescriptorBufferInfo> bufferInfos(bufferCounts);
  std::vector<VkDescriptorImageInfo> imageInfos(imageCounts);
  std::vector<std::vector<VkDescriptorImageInfo>> bindlessImageInfos = {};
  std::vector<VkWriteDescriptorSetAccelerationStructureKHR>
      accelerationStructureInfos(accStructCounts);
  uint32_t bufferIndex = 0;
  uint32_t imageIndex = 0;
  uint32_t accStructIndex = 0;
  for (auto& entry : entries) {
    if (entry.resource.bufferBinding.has_value()) {
      VkDescriptorBufferInfo& bufferInfo = bufferInfos[bufferIndex++];
      bufferInfo.buffer =
          static_cast<Buffer_VK*>(entry.resource.bufferBinding.value().buffer)
              ->getVkBuffer();
      bufferInfo.offset = entry.resource.bufferBinding.value().offset;
      bufferInfo.range = entry.resource.bufferBinding.value().size;
      descriptorWrites.push_back(VkWriteDescriptorSet{});
      VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = set;
      descriptorWrite.dstBinding = entry.binding;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = getVkDecriptorType(
          layout->getBindGroupLayoutDescriptor().entries[entry.binding]);
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = &bufferInfo;
      descriptorWrite.pImageInfo = nullptr;
      descriptorWrite.pTexelBufferView = nullptr;
    } else if (entry.resource.sampler && entry.resource.textureView) {
      VkDescriptorImageInfo& imageInfo = imageInfos[imageIndex++];
      imageInfo.imageView =
          static_cast<TextureView_VK*>(entry.resource.textureView)->imageView;
      imageInfo.sampler =
          static_cast<Sampler_VK*>(entry.resource.sampler)->textureSampler;
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      descriptorWrites.push_back(VkWriteDescriptorSet{});
      VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = set;
      descriptorWrite.dstBinding = entry.binding;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = getVkDecriptorType(
          layout->getBindGroupLayoutDescriptor().entries[entry.binding]);
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = nullptr;
      descriptorWrite.pImageInfo = &imageInfo;
      descriptorWrite.pTexelBufferView = nullptr;
    } else if (entry.resource.textureView) {
      VkDescriptorImageInfo& imageInfo = imageInfos[imageIndex++];
      imageInfo.sampler = {};
      imageInfo.imageView =
          static_cast<TextureView_VK*>(entry.resource.textureView)->imageView;
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      descriptorWrites.push_back(VkWriteDescriptorSet{});
      VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = set;
      descriptorWrite.dstBinding = entry.binding;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = getVkDecriptorType(
          layout->getBindGroupLayoutDescriptor().entries[entry.binding]);
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = nullptr;
      descriptorWrite.pImageInfo = &imageInfo;
      descriptorWrite.pTexelBufferView = nullptr;
    } else if (entry.resource.tlas) {
      VkWriteDescriptorSetAccelerationStructureKHR& descASInfo =
          accelerationStructureInfos[accStructIndex++];
      descASInfo.sType =
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
      descASInfo.accelerationStructureCount = 1;
      descASInfo.pAccelerationStructures =
          &static_cast<TLAS_VK*>(entry.resource.tlas)->tlas;
      descriptorWrites.push_back(VkWriteDescriptorSet{});
      VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = set;
      descriptorWrite.dstBinding = entry.binding;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = getVkDecriptorType(
          layout->getBindGroupLayoutDescriptor().entries[entry.binding]);
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = nullptr;
      descriptorWrite.pImageInfo = nullptr;
      descriptorWrite.pTexelBufferView = nullptr;
      descriptorWrite.pNext = &descASInfo;
    } else if (entry.resource.bindlessTextures.size() != 0) {
      bindlessImageInfos.push_back(std::vector<VkDescriptorImageInfo>(
          entry.resource.bindlessTextures.size()));
      std::vector<VkDescriptorImageInfo>& bindelessImageInfo =
          bindlessImageInfos.back();
      for (int i = 0; i < entry.resource.bindlessTextures.size(); ++i) {
        auto bindlessTexture = entry.resource.bindlessTextures[i];
        VkDescriptorImageInfo& imageInfo = bindelessImageInfo[i];
        imageInfo.sampler =
            static_cast<Sampler_VK*>(entry.resource.sampler)->textureSampler;
        imageInfo.imageView =
            static_cast<TextureView_VK*>(bindlessTexture)->imageView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        descriptorWrites.push_back(VkWriteDescriptorSet{});
        VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = set;
        descriptorWrite.dstBinding = entry.binding;
        descriptorWrite.dstArrayElement = i;
        descriptorWrite.descriptorType = getVkDecriptorType(
            layout->getBindGroupLayoutDescriptor().entries[entry.binding]);
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = nullptr;
        descriptorWrite.pImageInfo = &imageInfo;
        descriptorWrite.pTexelBufferView = nullptr;
      }
    }
  }
  vkUpdateDescriptorSets(device->getVkDevice(), descriptorWrites.size(),
                         descriptorWrites.data(), 0, nullptr);
}
#pragma endregion
}