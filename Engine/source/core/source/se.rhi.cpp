#include "se.rhi.hpp"
#include <set>
#define VMA_IMPLEMENTATION 
#define USE_VMA
#include <vma/vk_mem_alloc.h>
#include <aftermath/include/GFSDK_Aftermath.h>
#include <aftermath/include/GFSDK_Aftermath_GpuCrashDump.h>

namespace se {
namespace rhi {

  namespace impl {
    /** Whether enable validation layer */
    constexpr bool const enableValidationLayerVerboseOutput = false;
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
          se::info("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage); break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        se::info("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage); break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        se::warn("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage); break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        se::error("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage); break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
        se::error("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage); break;
      default: break;
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
            layerFound = true; break;
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

    auto getRequiredExtensions(Context* context,
      Flags<ContextExtensionEnum> ext) noexcept -> std::vector<const char*> {
      // extensions needed
      std::vector<const char*> extensions;
      // no extension needed for non-window application
      if (context->get_binded_window() == nullptr) {}
      // add glfw extension
      else {
        uint32_t glfwExtensionCount = 0;
        char const** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        int code = glfwGetError(NULL);
        if (code != GLFW_NO_ERROR) {
          if (code == GLFW_NOT_INITIALIZED)
            se::error("GLFW :: glfwGetRequiredInstanceExtensions :: GLFW_NOT_INITIALIZED!");
          else if (code == GLFW_API_UNAVAILABLE)
            se::error("GLFW :: glfwGetRequiredInstanceExtensions :: GLFW_API_UNAVAILABLE!");
        }
        std::vector<const char*> glfwExtensionNames(
          glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.insert(extensions.end(), glfwExtensionNames.begin(),
          glfwExtensionNames.end());
      }

      // add other extensions according to ext bits
      if (ext & ContextExtensionEnum::MESH_SHADER) {
        extensions.emplace_back(
          VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
      }
      // add extensions that validation layer needs
      if (ext & ContextExtensionEnum::DEBUG_UTILS) {
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      }
      // add other extensions according to ext bits
      if (ext & ContextExtensionEnum::CUDA_INTEROPERABILITY) {
        extensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        extensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
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

    auto createInstance(Context* context, Flags<ContextExtensionEnum> ext) noexcept -> void {
      // Check we could enable validation layers
      if ((ext & ContextExtensionEnum::DEBUG_UTILS)
        && !checkValidationLayerSupport())
        se::error("Vulkan :: validation layers requested, but not available!");
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
      if ((ext & ContextExtensionEnum::DEBUG_UTILS)) {
        createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayerNames.size());
        createInfo.ppEnabledLayerNames = validationLayerNames.data();
        // add debug messenger for init
        populateDebugMessengerCreateInfo(debugCreateInfo);
        *tail = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        tail = &((VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo)->pNext;
      }
      else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
      }
      VkValidationFeaturesEXT validationInfo = {};
      VkValidationFeatureEnableEXT validationFeatureToEnable =
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;
      if ((ext & ContextExtensionEnum::SHADER_NON_SEMANTIC_INFO)) {
        validationInfo.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validationInfo.enabledValidationFeatureCount = 1;
        validationInfo.pEnabledValidationFeatures = &validationFeatureToEnable;
        //_putenv_s("DEBUG_PRINTF_TO_STDOUT", "1");
        *tail = &validationInfo;
        tail = &(validationInfo.pNext);
      }
      // Create Vk Instance
      if (vkCreateInstance(&createInfo, nullptr, &(context->get_vk_instance())) != VK_SUCCESS) {
        se::error("Vulkan :: Failed to create instance!");
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

    auto setupDebugMessenger(Context* context) noexcept -> void {
      VkDebugUtilsMessengerCreateInfoEXT createInfo;
      populateDebugMessengerCreateInfo(createInfo);
      // load function from extern
      if (CreateDebugUtilsMessengerEXT(context->get_vk_instance(), &createInfo,
        nullptr, &context->get_vk_debug_messenger()) != VK_SUCCESS)
        se::error("Vulkan :: failed to set up debug messenger!");
    }

    PFN_vkVoidFunction vkGetInstanceProcAddrStub(void* context, const char* name) {
      return vkGetInstanceProcAddr((VkInstance)context, name);
    }

    auto setupExtensions(Context* context, Flags<ContextExtensionEnum> ext) -> void {
      if (context->get_binded_window()) {
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_SWAPCHAIN_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::DEBUG_UTILS)) {
        context->vkCmdBeginDebugUtilsLabelEXT =
          (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdBeginDebugUtilsLabelEXT");
        context->vkCmdEndDebugUtilsLabelEXT =
          (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdEndDebugUtilsLabelEXT");
        context->vkSetDebugUtilsObjectNameEXT =
          (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkSetDebugUtilsObjectNameEXT");
        context->vkSetDebugUtilsObjectTagEXT =
          (PFN_vkSetDebugUtilsObjectTagEXT)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkSetDebugUtilsObjectTagEXT");
      }
      if ((ext & ContextExtensionEnum::MESH_SHADER)) {
        context->vkCmdDrawMeshTasksNV =
          (PFN_vkCmdDrawMeshTasksNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdDrawMeshTasksNV");
        context->get_vk_device_extensions().emplace_back(
          VK_NV_MESH_SHADER_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::RAY_TRACING)) {
        context->vkCmdTraceRaysKHR =
          (PFN_vkCmdTraceRaysKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdTraceRaysKHR");
        context->vkCreateRayTracingPipelinesKHR =
          (PFN_vkCreateRayTracingPipelinesKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCreateRayTracingPipelinesKHR");
        context->vkGetRayTracingCaptureReplayShaderGroupHandlesKHR =
          (PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)
          vkGetInstanceProcAddrStub(
            context->get_vk_instance(),
            "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR");
        context->vkCmdTraceRaysIndirectKHR =
          (PFN_vkCmdTraceRaysIndirectKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdTraceRaysIndirectKHR");
        context->vkGetRayTracingShaderGroupStackSizeKHR =
          (PFN_vkGetRayTracingShaderGroupStackSizeKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkGetRayTracingShaderGroupStackSizeKHR");
        context->vkCmdSetRayTracingPipelineStackSizeKHR =
          (PFN_vkCmdSetRayTracingPipelineStackSizeKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdSetRayTracingPipelineStackSizeKHR");
        context->vkCreateAccelerationStructureNV =
          (PFN_vkCreateAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCreateAccelerationStructureNV");
        context->vkDestroyAccelerationStructureNV =
          (PFN_vkDestroyAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkDestroyAccelerationStructureNV");
        context->vkGetAccelerationStructureMemoryRequirementsNV =
          (PFN_vkGetAccelerationStructureMemoryRequirementsNV)
          vkGetInstanceProcAddrStub(
            context->get_vk_instance(),
            "vkGetAccelerationStructureMemoryRequirementsNV");
        context->vkBindAccelerationStructureMemoryNV =
          (PFN_vkBindAccelerationStructureMemoryNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkBindAccelerationStructureMemoryNV");
        context->vkCmdBuildAccelerationStructureNV =
          (PFN_vkCmdBuildAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdBuildAccelerationStructureNV");
        context->vkCmdCopyAccelerationStructureNV =
          (PFN_vkCmdCopyAccelerationStructureNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdCopyAccelerationStructureNV");
        context->vkCmdTraceRaysNV = (PFN_vkCmdTraceRaysNV)vkGetInstanceProcAddrStub(
          context->get_vk_instance(), "vkCmdTraceRaysNV");
        context->vkCreateRayTracingPipelinesNV =
          (PFN_vkCreateRayTracingPipelinesNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCreateRayTracingPipelinesNV");
        context->vkGetRayTracingShaderGroupHandlesKHR =
          (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkGetRayTracingShaderGroupHandlesKHR");
        context->vkGetRayTracingShaderGroupHandlesNV =
          (PFN_vkGetRayTracingShaderGroupHandlesNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkGetRayTracingShaderGroupHandlesNV");
        context->vkGetAccelerationStructureHandleNV =
          (PFN_vkGetAccelerationStructureHandleNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkGetAccelerationStructureHandleNV");
        context->vkCmdWriteAccelerationStructuresPropertiesNV =
          (PFN_vkCmdWriteAccelerationStructuresPropertiesNV)
          vkGetInstanceProcAddrStub(
            context->get_vk_instance(),
            "vkCmdWriteAccelerationStructuresPropertiesNV");
        context->vkCompileDeferredNV =
          (PFN_vkCompileDeferredNV)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCompileDeferredNV");
        context->vkGetAccelerationStructureBuildSizesKHR =
          (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(),
            "vkGetAccelerationStructureBuildSizesKHR");
        context->vkCmdBuildAccelerationStructuresKHR =
          (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdBuildAccelerationStructuresKHR");
        context->vkCreateAccelerationStructureKHR =
          (PFN_vkCreateAccelerationStructureKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCreateAccelerationStructureKHR");
        context->vkDestroyAccelerationStructureKHR =
          (PFN_vkDestroyAccelerationStructureKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkDestroyAccelerationStructureKHR");
        context->vkGetAccelerationStructureDeviceAddressKHR =
          (PFN_vkGetAccelerationStructureDeviceAddressKHR)
          vkGetInstanceProcAddrStub(
            context->get_vk_instance(),
            "vkGetAccelerationStructureDeviceAddressKHR");
        context->vkCmdCopyAccelerationStructureKHR =
          (PFN_vkCmdCopyAccelerationStructureKHR)vkGetInstanceProcAddrStub(
            context->get_vk_instance(), "vkCmdCopyAccelerationStructureKHR");
        
        // emplace back device extensions
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_RAY_QUERY_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::SHADER_NON_SEMANTIC_INFO)) {
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::CONSERVATIVE_RASTERIZATION)) {
        context->get_vk_device_extensions().emplace_back(
          VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::COOPERATIVE_MATRIX)) {
        context->get_vk_device_extensions().emplace_back(
          VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::ATOMIC_FLOAT)) {
        context->get_vk_device_extensions().emplace_back(
          VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::FRAGMENT_BARYCENTRIC)) {
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
      }
      if ((ext & ContextExtensionEnum::CUDA_INTEROPERABILITY)) {
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        context->get_vk_device_extensions().emplace_back(
          VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        #ifdef _WIN32
            context->get_vk_device_extensions().emplace_back(
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            context->get_vk_device_extensions().emplace_back(
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
        #elif defined(__linux__)
            context->get_vk_device_extensions().emplace_back(
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            context->get_vk_device_extensions().emplace_back(
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
        #endif
      }
      #ifdef _WIN32
          context->vkCmdGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddrStub(
          context->get_vk_instance(), "vkGetMemoryWin32HandleKHR");
      #elif defined(__linux__)
          context->vkCmdGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddrStub(
          context->get_vk_instance(), "vkGetMemoryFdKHR");
      #endif
      context->get_vk_device_extensions().emplace_back(
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
      context->get_vk_device_extensions().emplace_back(
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
      context->get_vk_device_extensions().emplace_back(
        VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);

      context->get_vk_device_extensions().emplace_back(
        VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    }

    auto attachWindow(Context* contexVk) noexcept -> void {
      if (contexVk->get_binded_window() == nullptr) {
        // do not create surface if no window is attached
      }
      else {
        if (!glfwVulkanSupported()) {
          se::error("Vulkan :: glfw cannot support Vulkan!");
        }

        VkResult result = glfwCreateWindowSurface(
          contexVk->get_vk_instance(),
          (GLFWwindow*)contexVk->get_binded_window()->get_handle(), nullptr,
          &contexVk->get_vk_surface_khr());
        if (result != VK_SUCCESS) {
          if (result == GLFW_NOT_INITIALIZED)
            se::error("Vulkan :: glfwCreateWindowSurface failed :: GLFW not initialized!");
          else if (result == GLFW_API_UNAVAILABLE)
            se::error("Vulkan :: glfwCreateWindowSurface failed :: GLFW API unabailable!");
          else if (result == GLFW_PLATFORM_ERROR)
            se::error("Vulkan :: glfwCreateWindowSurface failed :: GLFW platform error!");
          else if (result == GLFW_INVALID_VALUE)
            se::error("Vulkan :: glfwCreateWindowSurface failed :: GLFW invalid value!");
          else
            se::error("Vulkan :: glfwCreateWindowSurface failed :: GLFW not initialized!");
        }
      }
    }

    auto inline destroyDebugUtilsMessengerEXT(
      VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
      VkAllocationCallbacks const* pAllocator) noexcept -> void {
      auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
      if (func != nullptr) func(instance, debugMessenger, pAllocator);
    }


    auto findQueueFamilies(Context* contextVk, VkPhysicalDevice& device) noexcept
      -> Adapter::QueueFamilyIndices {
      Adapter::QueueFamilyIndices indices;
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
          indices.m_graphicsFamily = i;
          if (queueFamily.timestampValidBits <= 0)
            se::error("VULKAN :: Graphics Family not support timestamp ValidBits");
        }
        // check queue support
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
          indices.m_computeFamily = i;
        }
        // check present support
        if (contextVk->get_binded_window() != nullptr) {
          VkBool32 presentSupport = false;
          vkGetPhysicalDeviceSurfaceSupportKHR(
            device, i, contextVk->get_vk_surface_khr(), &presentSupport);
          if (presentSupport) indices.m_presentFamily = i;
        }
        // check support completeness
        if (indices.is_complete()) break;
        i++;
      }
      return indices;
    }

    auto checkDeviceExtensionSupport(Context* contextVk,
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
        contextVk->get_vk_device_extensions().begin(),
        contextVk->get_vk_device_extensions().end());
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

    auto querySwapChainSupport(Context* contextVk, VkPhysicalDevice& device)
      -> SwapChainSupportDetails {
      SwapChainSupportDetails details;
      // query basic surface capabilities
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        device, contextVk->get_vk_surface_khr(), &details.capabilities);
      // query the supported surface formats
      uint32_t formatCount;
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, contextVk->get_vk_surface_khr(),
        &formatCount, nullptr);
      if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, contextVk->get_vk_surface_khr(),
          &formatCount, details.formats.data());
      }
      // query the supported presentation modes
      uint32_t presentModeCount;
      vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, contextVk->get_vk_surface_khr(), &presentModeCount, nullptr);
      if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, contextVk->get_vk_surface_khr(), &presentModeCount,
          details.presentModes.data());
      }
      return details;
    }

    auto isDeviceSuitable(Context* contextVk, VkPhysicalDevice& device,
      std::string& device_diagnosis) noexcept -> bool {
      // check queue family supports
      Adapter::QueueFamilyIndices indices = findQueueFamilies(contextVk, device);
      // check extension supports
      bool const extensionsSupported =
        checkDeviceExtensionSupport(contextVk, device, device_diagnosis);
      // check swapchain support
      bool swapChainAdequate = false;
      if (extensionsSupported) {
        if (contextVk->get_binded_window() == nullptr) {
          swapChainAdequate = true;
        }
        else {
          SwapChainSupportDetails swapChainSupport =
            querySwapChainSupport(contextVk, device);
          swapChainAdequate = !swapChainSupport.formats.empty() &&
            !swapChainSupport.presentModes.empty();
        }
      }
      // check physical device feature supported
      VkPhysicalDeviceFeatures supportedFeatures;
      vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
      bool physicalDeviceFeatureSupported = supportedFeatures.samplerAnisotropy;
      return indices.is_complete() && extensionsSupported && swapChainAdequate &&
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

    auto queryAllPhysicalDevice(Context* contextVk) noexcept -> void {
      uint32_t deviceCount = 0;
      vkEnumeratePhysicalDevices(contextVk->get_vk_instance(), &deviceCount, nullptr);
      // If there are 0 devices with Vulkan support
      if (deviceCount == 0)
        se::error("VULKAN :: Failed to find GPUs with Vulkan support!");
      // get all of the VkPhysicalDevice handles
      std::vector<VkPhysicalDevice>& devices = contextVk->get_vk_physical_devices();
      devices.resize(deviceCount);
      vkEnumeratePhysicalDevices(contextVk->get_vk_instance(), &deviceCount,
        devices.data());
      // check if any of the physical devices meet the requirements
      int i = 0;
      for (const auto& device : devices) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        se::info("VULKAN :: Physical Device [{}] Found, {}", i, deviceProperties.deviceName);
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
        }
        else {
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

    inline auto getTextureFormat(VkFormat format) noexcept -> TextureFormat {
      switch (format) {
      case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return TextureFormat::DEPTH32STENCIL8;
      case VK_FORMAT_D32_SFLOAT:
        return TextureFormat::DEPTH32_FLOAT;
      case VK_FORMAT_D24_UNORM_S8_UINT:
        return TextureFormat::DEPTH24STENCIL8;
      case VK_FORMAT_X8_D24_UNORM_PACK32:
        return TextureFormat::DEPTH24;
      case VK_FORMAT_D16_UNORM:
        return TextureFormat::DEPTH16_UNORM;
      case VK_FORMAT_S8_UINT:
        return TextureFormat::STENCIL8;
      case VK_FORMAT_R32G32B32A32_SFLOAT:
        return TextureFormat::RGBA32_FLOAT;
      case VK_FORMAT_R32G32B32A32_SINT:
        return TextureFormat::RGBA32_SINT;
      case VK_FORMAT_R32G32B32A32_UINT:
        return TextureFormat::RGBA32_UINT;
      case VK_FORMAT_R16G16B16A16_SFLOAT:
        return TextureFormat::RGBA16_FLOAT;
      case VK_FORMAT_R16G16B16A16_SINT:
        return TextureFormat::RGBA16_SINT;
      case VK_FORMAT_R16G16B16A16_UINT:
        return TextureFormat::RGBA16_UINT;
      case VK_FORMAT_R32G32_SFLOAT:
        return TextureFormat::RG32_FLOAT;
      case VK_FORMAT_R32G32_SINT:
        return TextureFormat::RG32_SINT;
      case VK_FORMAT_R32G32_UINT:
        return TextureFormat::RG32_UINT;
      case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return TextureFormat::RG11B10_UFLOAT;
      case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return TextureFormat::RGB10A2_UNORM;
      case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return TextureFormat::RGB9E5_UFLOAT;
      case VK_FORMAT_B8G8R8A8_SRGB:
        return TextureFormat::BGRA8_UNORM_SRGB;
      case VK_FORMAT_B8G8R8A8_UNORM:
        return TextureFormat::BGRA8_UNORM;
      case VK_FORMAT_B8G8R8A8_SINT:
        return TextureFormat::RGBA8_SINT;
      case VK_FORMAT_B8G8R8A8_UINT:
        return TextureFormat::RGBA8_UINT;
      case VK_FORMAT_R8G8B8A8_SNORM:
        return TextureFormat::RGBA8_SNORM;
      case VK_FORMAT_R8G8B8A8_SRGB:
        return TextureFormat::RGBA8_UNORM_SRGB;
      case VK_FORMAT_R8G8B8A8_UNORM:
        return TextureFormat::RGBA8_UNORM;
      case VK_FORMAT_R16G16_SFLOAT:
        return TextureFormat::RG16_FLOAT;
      case VK_FORMAT_R16G16_SINT:
        return TextureFormat::RG16_SINT;
      case VK_FORMAT_R16G16_UINT:
        return TextureFormat::RG16_UINT;
      case VK_FORMAT_R32_SFLOAT:
        return TextureFormat::R32_FLOAT;
      case VK_FORMAT_R32_SINT:
        return TextureFormat::R32_SINT;
      case VK_FORMAT_R32_UINT:
        return TextureFormat::R32_UINT;
      case VK_FORMAT_R8G8_SINT:
        return TextureFormat::RG8_SINT;
      case VK_FORMAT_R8G8_UINT:
        return TextureFormat::RG8_UINT;
      case VK_FORMAT_R8G8_SNORM:
        return TextureFormat::RG8_SNORM;
      case VK_FORMAT_R8G8_UNORM:
        return TextureFormat::RG8_UNORM;
      case VK_FORMAT_R16_SFLOAT:
        return TextureFormat::R16_FLOAT;
      case VK_FORMAT_R16_SINT:
        return TextureFormat::R16_SINT;
      case VK_FORMAT_R16_UINT:
        return TextureFormat::R16_UINT;
      case VK_FORMAT_R8_SINT:
        return TextureFormat::R8_SINT;
      case VK_FORMAT_R8_UINT:
        return TextureFormat::R8_UINT;
      case VK_FORMAT_R8_SNORM:
        return TextureFormat::R8_SNORM;
      case VK_FORMAT_R8_UNORM:
        return TextureFormat::R8_UNORM;
      default:
        return TextureFormat(0);
        break;
      }
    }

    inline auto mapMemoryTexture(Device* device, Texture* texture,
      size_t offset, size_t size,
      void*& mappedData) noexcept -> bool {
#ifdef USE_VMA
      VkResult result = vmaMapMemory(device->get_vma_allocator(),
        texture->get_vma_allocation(), &mappedData);
#else
      VkResult result =
        vkMapMemory(device->getVkDevice(), texture->getVkDeviceMemory(), offset,
          size, 0, &mappedData);
#endif
      if (result) texture->set_buffer_map_state(BufferMapState::MAPPED);
      return result == VkResult::VK_SUCCESS ? true : false;
    }

    inline auto getVkFilter(FilterMode mode) noexcept -> VkFilter {
      switch (mode) {
      case FilterMode::LINEAR:
        return VkFilter::VK_FILTER_LINEAR;
      case FilterMode::NEAREST:
        return VkFilter::VK_FILTER_NEAREST;
      default:
        return VkFilter::VK_FILTER_MAX_ENUM;
      }
    }

    inline auto getVkSamplerAddressMode(AddressMode address) noexcept
      -> VkSamplerAddressMode {
      switch (address) {
      case AddressMode::MIRROR_REPEAT:
        return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      case AddressMode::REPEAT:
        return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT;
      case AddressMode::CLAMP_TO_EDGE:
        return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      default:
        return VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_MAX_ENUM;
      }
    }

    inline auto getVkQueryType(QueryType const type) noexcept -> VkQueryType {
      switch (type) {
      case QueryType::OCCLUSION: return VK_QUERY_TYPE_OCCLUSION;
      case QueryType::PIPELINE_STATISTICS: return VK_QUERY_TYPE_PIPELINE_STATISTICS;
      case QueryType::TIMESTAMP: return VK_QUERY_TYPE_TIMESTAMP;
      default: return VK_QUERY_TYPE_MAX_ENUM;
      }
    }

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
        if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
          return availablePresentMode;
        }
      }
      return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    inline auto chooseSwapExtent(VkSurfaceCapabilitiesKHR const& capabilities,
      se::Window* bindedWindow) noexcept
      -> VkExtent2D {
      if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
        return capabilities.currentExtent;
      else {
        int width, height;
        bindedWindow->get_framebuffer_size(&width, &height);
        VkExtent2D actualExtent = { static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height) };
        actualExtent.width =
          std::clamp(actualExtent.width, capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        actualExtent.height =
          std::clamp(actualExtent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);
        return actualExtent;
      }
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

    inline auto fillFixedFunctionSettingDynamicInfo(
      RenderPipeline::RenderPipelineFixedFunctionSettings& settings) noexcept
      -> void {
      // fill in 2 structure in the settings:
      // 1. std::vector<VkDynamicState> dynamicStates
      settings.dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT,
                                VK_DYNAMIC_STATE_SCISSOR };
      // 2. VkPipelineDynamicStateCreateInfo dynamicState
      settings.dynamicState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      settings.dynamicState.dynamicStateCount =
        static_cast<uint32_t>(settings.dynamicStates.size());
      settings.dynamicState.pDynamicStates = settings.dynamicStates.data();
    }

    inline auto getVkFormat(VertexFormat format) noexcept -> VkFormat {
      switch (format) {
      case VertexFormat::SINT32X4:
        return VK_FORMAT_R32G32B32A32_SINT;
      case VertexFormat::SINT32X3:
        return VK_FORMAT_R32G32B32_SINT;
      case VertexFormat::SINT32X2:
        return VK_FORMAT_R32G32_SINT;
      case VertexFormat::SINT32:
        return VK_FORMAT_R32_SINT;
      case VertexFormat::UINT32X4:
        return VK_FORMAT_R32G32B32A32_UINT;
      case VertexFormat::UINT32X3:
        return VK_FORMAT_R32G32B32_UINT;
      case VertexFormat::UINT32X2:
        return VK_FORMAT_R32G32_UINT;
      case VertexFormat::UINT32:
        return VK_FORMAT_R32_UINT;
      case VertexFormat::FLOAT32X4:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
      case VertexFormat::FLOAT32X3:
        return VK_FORMAT_R32G32B32_SFLOAT;
      case VertexFormat::FLOAT32X2:
        return VK_FORMAT_R32G32_SFLOAT;
      case VertexFormat::FLOAT32:
        return VK_FORMAT_R32_SFLOAT;
      case VertexFormat::FLOAT16X4:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
      case VertexFormat::FLOAT16X2:
        return VK_FORMAT_R16G16_SFLOAT;
      case VertexFormat::SNORM16X4:
        return VK_FORMAT_R16G16B16A16_SNORM;
      case VertexFormat::SNORM16X2:
        return VK_FORMAT_R16G16_SNORM;
      case VertexFormat::UNORM16X4:
        return VK_FORMAT_R16G16B16A16_UNORM;
      case VertexFormat::UNORM16X2:
        return VK_FORMAT_R16G16_UNORM;
      case VertexFormat::SINT16X4:
        return VK_FORMAT_R16G16B16A16_SINT;
      case VertexFormat::SINT16X2:
        return VK_FORMAT_R16G16_SINT;
      case VertexFormat::UINT16X4:
        return VK_FORMAT_R16G16B16A16_UINT;
      case VertexFormat::UINT16X2:
        return VK_FORMAT_R16G16_UINT;
      case VertexFormat::SNORM8X4:
        return VK_FORMAT_R8G8B8A8_SNORM;
      case VertexFormat::SNORM8X2:
        return VK_FORMAT_R8G8_SNORM;
      case VertexFormat::UNORM8X4:
        return VK_FORMAT_R8G8B8A8_UNORM;
      case VertexFormat::UNORM8X2:
        return VK_FORMAT_R8G8_UNORM;
      case VertexFormat::SINT8X4:
        return VK_FORMAT_R8G8B8A8_SINT;
      case VertexFormat::SINT8X2:
        return VK_FORMAT_R8G8_SINT;
      case VertexFormat::UINT8X4:
        return VK_FORMAT_R8G8B8A8_UINT;
      case VertexFormat::UINT8X2:
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

    auto getVkImageLayout(TextureLayoutEnum layout) noexcept -> VkImageLayout {
      switch (layout) {
      case TextureLayoutEnum::UNDEFINED:
        return VK_IMAGE_LAYOUT_UNDEFINED;
      case TextureLayoutEnum::GENERAL:
        return VK_IMAGE_LAYOUT_GENERAL;
      case TextureLayoutEnum::COLOR_ATTACHMENT_OPTIMAL:
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      case TextureLayoutEnum::DEPTH_STENCIL_ATTACHMENT_OPTIMA:
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      case TextureLayoutEnum::DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
      case TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL:
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      case TextureLayoutEnum::TRANSFER_SRC_OPTIMAL:
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      case TextureLayoutEnum::TRANSFER_DST_OPTIMAL:
        return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      case TextureLayoutEnum::PREINITIALIZED:
        return VK_IMAGE_LAYOUT_PREINITIALIZED;
      case TextureLayoutEnum::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
        return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
      case TextureLayoutEnum::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
        return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
      case TextureLayoutEnum::DEPTH_ATTACHMENT_OPTIMAL:
        return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      case TextureLayoutEnum::DEPTH_READ_ONLY_OPTIMAL:
        return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      case TextureLayoutEnum::STENCIL_ATTACHMENT_OPTIMAL:
        return VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
      case TextureLayoutEnum::STENCIL_READ_ONLY_OPTIMAL:
        return VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
      case TextureLayoutEnum::PRESENT_SRC:
        return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      case TextureLayoutEnum::SHARED_PRESENT:
        return VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
      case TextureLayoutEnum::FRAGMENT_DENSITY_MAP_OPTIMAL:
        return VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
      case TextureLayoutEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL:
        return VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
      case TextureLayoutEnum::READ_ONLY_OPTIMAL:
        return VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
      case TextureLayoutEnum::ATTACHMENT_OPTIMAL:
        return VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
      default:
        return VK_IMAGE_LAYOUT_MAX_ENUM;
      }
    }

    inline auto getVkAccessFlags(Flags<AccessFlagEnum> accessFlags) noexcept
      -> VkAccessFlags {
      VkAccessFlags flags = 0;
      if (accessFlags & AccessFlagEnum::INDIRECT_COMMAND_READ_BIT)
        flags |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      if (accessFlags & AccessFlagEnum::INDEX_READ_BIT)
        flags |= VK_ACCESS_INDEX_READ_BIT;
      if (accessFlags & AccessFlagEnum::VERTEX_ATTRIBUTE_READ_BIT)
        flags |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
      if (accessFlags & AccessFlagEnum::UNIFORM_READ_BIT)
        flags |= VK_ACCESS_UNIFORM_READ_BIT;
      if (accessFlags & AccessFlagEnum::INPUT_ATTACHMENT_READ_BIT)
        flags |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
      if (accessFlags & AccessFlagEnum::SHADER_READ_BIT)
        flags |= VK_ACCESS_SHADER_READ_BIT;
      if (accessFlags & AccessFlagEnum::SHADER_WRITE_BIT)
        flags |= VK_ACCESS_SHADER_WRITE_BIT;
      if (accessFlags & AccessFlagEnum::COLOR_ATTACHMENT_READ_BIT)
        flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
      if (accessFlags & AccessFlagEnum::COLOR_ATTACHMENT_WRITE_BIT)
        flags |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      if (accessFlags & AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_READ_BIT)
        flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      if (accessFlags & AccessFlagEnum::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      if (accessFlags & AccessFlagEnum::TRANSFER_READ_BIT)
        flags |= VK_ACCESS_TRANSFER_READ_BIT;
      if (accessFlags & AccessFlagEnum::TRANSFER_WRITE_BIT)
        flags |= VK_ACCESS_TRANSFER_WRITE_BIT;
      if (accessFlags & AccessFlagEnum::HOST_READ_BIT)
        flags |= VK_ACCESS_HOST_READ_BIT;
      if (accessFlags & AccessFlagEnum::HOST_WRITE_BIT)
        flags |= VK_ACCESS_HOST_WRITE_BIT;
      if (accessFlags & AccessFlagEnum::MEMORY_READ_BIT)
        flags |= VK_ACCESS_MEMORY_READ_BIT;
      if (accessFlags & AccessFlagEnum::MEMORY_WRITE_BIT)
        flags |= VK_ACCESS_MEMORY_WRITE_BIT;
      if (accessFlags & AccessFlagEnum::TRANSFORM_FEEDBACK_WRITE_BIT)
        flags |= VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
      if (accessFlags & AccessFlagEnum::TRANSFORM_FEEDBACK_COUNTER_READ_BIT)
        flags |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT;
      if (accessFlags & AccessFlagEnum::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT)
        flags |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
      if (accessFlags & AccessFlagEnum::CONDITIONAL_RENDERING_READ_BIT)
        flags |= VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
      if (accessFlags & AccessFlagEnum::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT)
        flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT;
      if (accessFlags & AccessFlagEnum::ACCELERATION_STRUCTURE_READ_BIT)
        flags |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      if (accessFlags & AccessFlagEnum::ACCELERATION_STRUCTURE_WRITE_BIT)
        flags |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      if (accessFlags & AccessFlagEnum::FRAGMENT_DENSITY_MAP_READ_BIT)
        flags |= VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT;
      if (accessFlags & AccessFlagEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT)
        flags |= VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
      if (accessFlags & AccessFlagEnum::COMMAND_PREPROCESS_READ_BIT)
        flags |= VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV;
      if (accessFlags & AccessFlagEnum::COMMAND_PREPROCESS_WRITE_BIT)
        flags |= VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV;
      return flags;
    }

    inline auto getVkPipelineStageFlagBits(PipelineStageEnum stage) noexcept
      -> VkPipelineStageFlagBits {
      switch (stage) {
      case PipelineStageEnum::TOP_OF_PIPE_BIT: return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      case PipelineStageEnum::DRAW_INDIRECT_BIT: return VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      case PipelineStageEnum::VERTEX_INPUT_BIT: return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      case PipelineStageEnum::VERTEX_SHADER_BIT: return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
      case PipelineStageEnum::TESSELLATION_CONTROL_SHADER_BIT: return VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
      case PipelineStageEnum::TESSELLATION_EVALUATION_SHADER_BIT: return VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
      case PipelineStageEnum::GEOMETRY_SHADER_BIT: return VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
      case PipelineStageEnum::FRAGMENT_SHADER_BIT: return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      case PipelineStageEnum::EARLY_FRAGMENT_TESTS_BIT: return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
      case PipelineStageEnum::LATE_FRAGMENT_TESTS_BIT: return VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      case PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT: return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      case PipelineStageEnum::COMPUTE_SHADER_BIT: return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      case PipelineStageEnum::TRANSFER_BIT: return VK_PIPELINE_STAGE_TRANSFER_BIT;
      case PipelineStageEnum::BOTTOM_OF_PIPE_BIT: return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      case PipelineStageEnum::HOST_BIT: return VK_PIPELINE_STAGE_HOST_BIT;
      case PipelineStageEnum::ALL_GRAPHICS_BIT: return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
      case PipelineStageEnum::ALL_COMMANDS_BIT: return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      case PipelineStageEnum::TRANSFORM_FEEDBACK_BIT_EXT: return VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
      case PipelineStageEnum::CONDITIONAL_RENDERING_BIT_EXT: return VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
      case PipelineStageEnum::ACCELERATION_STRUCTURE_BUILD_BIT_KHR: return VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      case PipelineStageEnum::RAY_TRACING_SHADER_BIT_KHR: return VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      case PipelineStageEnum::TASK_SHADER_BIT_NV: return VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV;
      case PipelineStageEnum::MESH_SHADER_BIT_NV: return VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV;
      case PipelineStageEnum::FRAGMENT_DENSITY_PROCESS_BIT: return VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT;
      case PipelineStageEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_BIT: return VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
      case PipelineStageEnum::COMMAND_PREPROCESS_BIT: return VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV;
      default: return VK_PIPELINE_STAGE_NONE_KHR;
      }
    }

    inline auto getVkPrimitiveTopology(PrimitiveTopology topology) noexcept
      -> VkPrimitiveTopology {
      switch (topology) {
      case PrimitiveTopology::TRIANGLE_STRIP:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        break;
      case PrimitiveTopology::TRIANGLE_LIST:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        break;
      case PrimitiveTopology::LINE_STRIP:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        break;
      case PrimitiveTopology::LINE_LIST:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        break;
      case PrimitiveTopology::POINT_LIST:
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
      case CullMode::BACK:
        return VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT;
      case CullMode::FRONT:
        return VkCullModeFlagBits::VK_CULL_MODE_FRONT_BIT;
      case CullMode::NONE:
        return VkCullModeFlagBits::VK_CULL_MODE_NONE;
      case CullMode::BOTH:
        return VkCullModeFlagBits::VK_CULL_MODE_FRONT_AND_BACK;
      default:
        return VkCullModeFlagBits::VK_CULL_MODE_NONE;
      }
    }

    inline auto getVkFrontFace(FrontFace ff) noexcept -> VkFrontFace {
      switch (ff) {
      case FrontFace::CW:
        return VkFrontFace::VK_FRONT_FACE_CLOCKWISE;
      case FrontFace::CCW:
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
      case CompareFunction::ALWAYS:
        return VK_COMPARE_OP_ALWAYS;
        break;
      case CompareFunction::GREATER_EQUAL:
        return VK_COMPARE_OP_GREATER_OR_EQUAL;
        break;
      case CompareFunction::NOT_EQUAL:
        return VK_COMPARE_OP_NOT_EQUAL;
        break;
      case CompareFunction::GREATER:
        return VK_COMPARE_OP_GREATER;
        break;
      case CompareFunction::LESS_EQUAL:
        return VK_COMPARE_OP_LESS_OR_EQUAL;
        break;
      case CompareFunction::EQUAL:
        return VK_COMPARE_OP_EQUAL;
        break;
      case CompareFunction::LESS:
        return VK_COMPARE_OP_LESS;
        break;
      case CompareFunction::NEVER:
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
      case BlendFactor::ONE_MINUS_CONSTANT:
        return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        break;
      case BlendFactor::CONSTANT:
        return VK_BLEND_FACTOR_CONSTANT_COLOR;
        break;
      case BlendFactor::SRC_ALPHA_SATURATED:
        return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        break;
      case BlendFactor::ONE_MINUS_DST_ALPHA:
        return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        break;
      case BlendFactor::DST_ALPHA:
        return VK_BLEND_FACTOR_DST_ALPHA;
        break;
      case BlendFactor::ONE_MINUS_DST:
        return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        break;
      case BlendFactor::DST:
        return VK_BLEND_FACTOR_DST_COLOR;
        break;
      case BlendFactor::ONE_MINUS_SRC_ALPHA:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        break;
      case BlendFactor::SRC_ALPHA:
        return VK_BLEND_FACTOR_SRC_ALPHA;
        break;
      case BlendFactor::ONE_MINUS_SRC:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        break;
      case BlendFactor::SRC:
        return VK_BLEND_FACTOR_SRC_COLOR;
        break;
      case BlendFactor::ONE:
        return VK_BLEND_FACTOR_ONE;
        break;
      case BlendFactor::ZERO:
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

    inline auto fillFixedFunctionSettingVertexInfo(
      VertexState const& state,
      RenderPipeline::RenderPipelineFixedFunctionSettings& settings) noexcept
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
      RenderPipeline::RenderPipelineFixedFunctionSettings& settings) noexcept
      -> void {
      // fill in 1 structure in the settings, whose viewport & scisor could be set
      // later
      VkPipelineViewportStateCreateInfo& viewportState = settings.viewportState;
      viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewportState.viewportCount = 1;
      viewportState.scissorCount = 1;
    }

    void createSwapChain(Device* device, SwapChain* swapchain) {
      Adapter* adapater = device->from_which_adapter();
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(
        adapater->from_which_context(), adapater->get_vk_physical_device());
      VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
      VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
      VkExtent2D extent = chooseSwapExtent(
        swapChainSupport.capabilities, adapater->from_which_context()->get_binded_window());
      swapchain->m_swapChainExtend = extent;
      swapchain->m_swapChainImageFormat = surfaceFormat.format;
      uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
      if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
      }
      VkSwapchainCreateInfoKHR createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      createInfo.surface = adapater->from_which_context()->get_vk_surface_khr();
      createInfo.minImageCount = imageCount;
      createInfo.imageFormat = surfaceFormat.format;
      createInfo.imageColorSpace = surfaceFormat.colorSpace;
      createInfo.imageExtent = extent;
      createInfo.imageArrayLayers = 1;
      createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      Adapter::QueueFamilyIndices const& indices = adapater->get_queue_family_indices();
      uint32_t queueFamilyIndices[] = { indices.m_graphicsFamily.value(),
                                       indices.m_presentFamily.value() };
      if (indices.m_graphicsFamily != indices.m_presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
      }
      else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;      // Optional
        createInfo.pQueueFamilyIndices = nullptr;  // Optional
      }
      createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
      createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
      createInfo.presentMode = presentMode;
      createInfo.clipped = VK_TRUE;
      createInfo.oldSwapchain = VK_NULL_HANDLE;
      if (vkCreateSwapchainKHR(device->get_vk_device(), &createInfo, nullptr,
        &swapchain->m_swapChain) != VK_SUCCESS) {
        se::error("VULKAN :: failed to create swap chain!");
      }
    }

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

    inline auto getVkImageViewType(TextureViewDimension const& dim) noexcept
      -> VkImageViewType {
      switch (dim) {
      case TextureViewDimension::TEX1D:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_1D;
      case TextureViewDimension::TEX2D:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_2D;
      case TextureViewDimension::TEX2D_ARRAY:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      case TextureViewDimension::CUBE:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE;
      case TextureViewDimension::CUBE_ARRAY:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      case TextureViewDimension::TEX3D:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_3D;
      default:
        return VkImageViewType::VK_IMAGE_VIEW_TYPE_MAX_ENUM;
      }
    }

    inline auto getVkImageAspectFlags(Flags<TextureAspectEnum> aspect) noexcept
      -> VkImageAspectFlags {
      VkImageAspectFlags ret = 0;
      if (aspect & TextureAspectEnum::COLOR_BIT)
        ret |= VK_IMAGE_ASPECT_COLOR_BIT;
      if (aspect & TextureAspectEnum::DEPTH_BIT)
        ret |= VK_IMAGE_ASPECT_DEPTH_BIT;
      if (aspect & TextureAspectEnum::STENCIL_BIT)
        ret |= VK_IMAGE_ASPECT_STENCIL_BIT;
      return ret;
    }

    inline auto getVkDependencyTypeFlags(Flags<DependencyTypeEnum> type) noexcept
      -> VkDependencyFlags {
      uint32_t flags = 0;
      if (type & DependencyTypeEnum::BY_REGION_BIT)
        flags |= VK_DEPENDENCY_BY_REGION_BIT;
      if (type & DependencyTypeEnum::VIEW_LOCAL_BIT)
        flags |= VK_DEPENDENCY_VIEW_LOCAL_BIT;
      if (type & DependencyTypeEnum::DEVICE_GROUP_BIT)
        flags |= VK_DEPENDENCY_DEVICE_GROUP_BIT;
      return (VkDependencyFlags)flags;
    }

    inline auto getVkPipelineStageFlags(Flags<PipelineStageEnum> stages) noexcept
      -> VkPipelineStageFlags {
      uint32_t flags = 0;
      if (stages & PipelineStageEnum::TOP_OF_PIPE_BIT)
        flags |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      if (stages & PipelineStageEnum::DRAW_INDIRECT_BIT)
        flags |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      if (stages & PipelineStageEnum::VERTEX_INPUT_BIT)
        flags |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      if (stages & PipelineStageEnum::VERTEX_SHADER_BIT)
        flags |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
      if (stages & PipelineStageEnum::TESSELLATION_CONTROL_SHADER_BIT)
        flags |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
      if (stages & PipelineStageEnum::TESSELLATION_EVALUATION_SHADER_BIT)
        flags |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
      if (stages & PipelineStageEnum::GEOMETRY_SHADER_BIT)
        flags |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
      if (stages & PipelineStageEnum::FRAGMENT_SHADER_BIT)
        flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      if (stages & PipelineStageEnum::EARLY_FRAGMENT_TESTS_BIT)
        flags |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
      if (stages & PipelineStageEnum::LATE_FRAGMENT_TESTS_BIT)
        flags |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      if (stages & PipelineStageEnum::COLOR_ATTACHMENT_OUTPUT_BIT)
        flags |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (stages & PipelineStageEnum::COMPUTE_SHADER_BIT)
        flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      if (stages & PipelineStageEnum::TRANSFER_BIT)
        flags |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      if (stages & PipelineStageEnum::BOTTOM_OF_PIPE_BIT)
        flags |= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      if (stages & PipelineStageEnum::HOST_BIT)
        flags |= VK_PIPELINE_STAGE_HOST_BIT;
      if (stages & PipelineStageEnum::ALL_GRAPHICS_BIT)
        flags |= VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
      if (stages & PipelineStageEnum::ALL_COMMANDS_BIT)
        flags |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      if (stages & PipelineStageEnum::TRANSFORM_FEEDBACK_BIT_EXT)
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
      if (stages & PipelineStageEnum::CONDITIONAL_RENDERING_BIT_EXT)
        flags |= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
      if (stages & PipelineStageEnum::ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
        flags |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      if (stages & PipelineStageEnum::RAY_TRACING_SHADER_BIT_KHR)
        flags |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      if (stages & PipelineStageEnum::TASK_SHADER_BIT_NV)
        flags |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV;
      if (stages & PipelineStageEnum::MESH_SHADER_BIT_NV)
        flags |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV;
      if (stages & PipelineStageEnum::FRAGMENT_DENSITY_PROCESS_BIT)
        flags |= VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT;
      if (stages & PipelineStageEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_BIT)
        flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
      if (stages & PipelineStageEnum::COMMAND_PREPROCESS_BIT)
        flags |= VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV;
      return (VkPipelineStageFlags)flags;
    }

    inline auto getVkFormat(TextureFormat format) noexcept -> VkFormat {
      switch (format) {
      case TextureFormat::DEPTH32STENCIL8:
        return VK_FORMAT_D32_SFLOAT_S8_UINT;
        break;
      case TextureFormat::DEPTH32_FLOAT:
        return VK_FORMAT_D32_SFLOAT;
        break;
      case TextureFormat::DEPTH24STENCIL8:
        return VK_FORMAT_D24_UNORM_S8_UINT;
        break;
      case TextureFormat::DEPTH24:
        return VK_FORMAT_X8_D24_UNORM_PACK32;
        break;
      case TextureFormat::DEPTH16_UNORM:
        return VK_FORMAT_D16_UNORM;
        break;
      case TextureFormat::STENCIL8:
        return VK_FORMAT_S8_UINT;
        break;
      case TextureFormat::RGBA32_FLOAT:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
        break;
      case TextureFormat::RGBA32_SINT:
        return VK_FORMAT_R32G32B32A32_SINT;
        break;
      case TextureFormat::RGBA32_UINT:
        return VK_FORMAT_R32G32B32A32_UINT;
        break;
      case TextureFormat::RGBA16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
        break;
      case TextureFormat::RGBA16_SINT:
        return VK_FORMAT_R16G16B16A16_SINT;
        break;
      case TextureFormat::RGBA16_UINT:
        return VK_FORMAT_R16G16B16A16_UINT;
        break;
      case TextureFormat::RG32_FLOAT:
        return VK_FORMAT_R32G32_SFLOAT;
        break;
      case TextureFormat::RG32_SINT:
        return VK_FORMAT_R32G32_SINT;
        break;
      case TextureFormat::RG32_UINT:
        return VK_FORMAT_R32G32_UINT;
        break;
      case TextureFormat::RG11B10_UFLOAT:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        break;
      case TextureFormat::RGB10A2_UNORM:
        return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
        break;
      case TextureFormat::RGB9E5_UFLOAT:
        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
        break;
      case TextureFormat::BGRA8_UNORM_SRGB:
        return VK_FORMAT_B8G8R8A8_SRGB;
        break;
      case TextureFormat::BGRA8_UNORM:
        return VK_FORMAT_B8G8R8A8_UNORM;
        break;
      case TextureFormat::RGBA8_SINT:
        return VK_FORMAT_R8G8B8A8_SINT;
        break;
      case TextureFormat::RGBA8_UINT:
        return VK_FORMAT_R8G8B8A8_UINT;
        break;
      case TextureFormat::RGBA8_SNORM:
        return VK_FORMAT_R8G8B8A8_SNORM;
        break;
      case TextureFormat::RGBA8_UNORM_SRGB:
        return VK_FORMAT_R8G8B8A8_SRGB;
        break;
      case TextureFormat::RGBA8_UNORM:
        return VK_FORMAT_R8G8B8A8_UNORM;
        break;
      case TextureFormat::RG16_FLOAT:
        return VK_FORMAT_R16G16_SFLOAT;
        break;
      case TextureFormat::RG16_SINT:
        return VK_FORMAT_R16G16_SINT;
        break;
      case TextureFormat::RG16_UINT:
        return VK_FORMAT_R16G16_UINT;
        break;
      case TextureFormat::R32_FLOAT:
        return VK_FORMAT_R32_SFLOAT;
        break;
      case TextureFormat::R32_SINT:
        return VK_FORMAT_R32_SINT;
        break;
      case TextureFormat::R32_UINT:
        return VK_FORMAT_R32_UINT;
        break;
      case TextureFormat::RG8_SINT:
        return VK_FORMAT_R8G8_SINT;
        break;
      case TextureFormat::RG8_UINT:
        return VK_FORMAT_R8G8_UINT;
        break;
      case TextureFormat::RG8_SNORM:
        return VK_FORMAT_R8G8_SNORM;
        break;
      case TextureFormat::RG8_UNORM:
        return VK_FORMAT_R8G8_UNORM;
        break;
      case TextureFormat::R16_FLOAT:
        return VK_FORMAT_R16_SFLOAT;
        break;
      case TextureFormat::R16_SINT:
        return VK_FORMAT_R16_SINT;
        break;
      case TextureFormat::R16_UINT:
        return VK_FORMAT_R16_UINT;
        break;
      case TextureFormat::R8_SINT:
        return VK_FORMAT_R8_SINT;
        break;
      case TextureFormat::R8_UINT:
        return VK_FORMAT_R8_UINT;
        break;
      case TextureFormat::R8_SNORM:
        return VK_FORMAT_R8_SNORM;
        break;
      case TextureFormat::R8_UNORM:
        return VK_FORMAT_R8_UNORM;
        break;
      case TextureFormat::BC1_RGB_UNORM_BLOCK:
        return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        break;
      case TextureFormat::BC1_RGB_SRGB_BLOCK:
        return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        break;
      case TextureFormat::BC1_RGBA_UNORM_BLOCK:
        return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        break;
      case TextureFormat::BC1_RGBA_SRGB_BLOCK:
        return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        break;
      case TextureFormat::BC2_UNORM_BLOCK:
        return VK_FORMAT_BC2_UNORM_BLOCK;
        break;
      case TextureFormat::BC2_SRGB_BLOCK:
        return VK_FORMAT_BC2_SRGB_BLOCK;
        break;
      case TextureFormat::BC3_UNORM_BLOCK:
        return VK_FORMAT_BC3_UNORM_BLOCK;
        break;
      case TextureFormat::BC3_SRGB_BLOCK:
        return VK_FORMAT_BC3_SRGB_BLOCK;
        break;
      case TextureFormat::BC4_UNORM_BLOCK:
        return VK_FORMAT_BC4_UNORM_BLOCK;
        break;
      case TextureFormat::BC4_SNORM_BLOCK:
        return VK_FORMAT_BC4_SNORM_BLOCK;
        break;
      case TextureFormat::BC5_UNORM_BLOCK:
        return VK_FORMAT_BC5_UNORM_BLOCK;
        break;
      case TextureFormat::BC5_SNORM_BLOCK:
        return VK_FORMAT_BC5_SNORM_BLOCK;
        break;
      case TextureFormat::BC6H_UFLOAT_BLOCK:
        return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        break;
      case TextureFormat::BC6H_SFLOAT_BLOCK:
        return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        break;
      case TextureFormat::BC7_UNORM_BLOCK:
        return VK_FORMAT_BC7_UNORM_BLOCK;
        break;
      case TextureFormat::BC7_SRGB_BLOCK:
        return VK_FORMAT_BC7_SRGB_BLOCK;
        break;
      default:
        return VK_FORMAT_UNDEFINED;
        break;
      }
    }

    inline auto getVkAttachmentLoadOp(LoadOp op) noexcept -> VkAttachmentLoadOp {
      switch (op) {
      case LoadOp::DONT_CARE:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      case LoadOp::CLEAR:
        return VK_ATTACHMENT_LOAD_OP_CLEAR;
      case LoadOp::LOAD:
        return VK_ATTACHMENT_LOAD_OP_LOAD;
      default:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      }
    }

    inline auto getVkAttachmentStoreOp(StoreOp op) noexcept -> VkAttachmentStoreOp {
      switch (op) {
      case StoreOp::DONT_CARE:
        return VK_ATTACHMENT_STORE_OP_DONT_CARE;
      case StoreOp::DISCARD:
        return VK_ATTACHMENT_STORE_OP_DONT_CARE;
      case StoreOp::STORE:
        return VK_ATTACHMENT_STORE_OP_STORE;
      default:
        return VK_ATTACHMENT_STORE_OP_DONT_CARE;
      }
    }

    inline auto getVkImageUsageFlagBits(Flags<TextureUsageEnum> flags) noexcept
      -> VkImageUsageFlags {
      VkImageUsageFlags usageFlags = 0;
      if (flags & TextureUsageEnum::COPY_SRC)
        usageFlags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      if (flags & TextureUsageEnum::COPY_DST)
        usageFlags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      if (flags & TextureUsageEnum::TEXTURE_BINDING)
        usageFlags |= VK_IMAGE_USAGE_SAMPLED_BIT;
      if (flags & TextureUsageEnum::STORAGE_BINDING)
        usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
      if (flags & TextureUsageEnum::COLOR_ATTACHMENT)
        usageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      if (flags & TextureUsageEnum::DEPTH_ATTACHMENT)
        usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      if (flags & TextureUsageEnum::TRANSIENT_ATTACHMENT)
        usageFlags |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
      if (flags & TextureUsageEnum::INPUT_ATTACHMENT)
        usageFlags |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
      return usageFlags;
    }

    inline auto getVkImageCreateFlags(Flags<TextureFeatureEnum> descflags) noexcept
      -> VkImageCreateFlags {
      VkImageCreateFlags flags = 0;
      if (descflags & TextureFeatureEnum::CUBE_COMPATIBLE)
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      return flags;
    }

    inline auto getVkShaderStageFlagBits(ShaderStageEnum flag) noexcept
      -> VkShaderStageFlagBits {
      switch (flag) {
      case ShaderStageEnum::COMPUTE:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT;
        break;
      case ShaderStageEnum::FRAGMENT:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT;
        break;
      case ShaderStageEnum::VERTEX:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT;
        break;
      case ShaderStageEnum::GEOMETRY:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_GEOMETRY_BIT;
        break;
      case ShaderStageEnum::RAYGEN:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        break;
      case ShaderStageEnum::MISS:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_KHR;
        break;
      case ShaderStageEnum::INTERSECTION:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        break;
      case ShaderStageEnum::CLOSEST_HIT:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        break;
      case ShaderStageEnum::CALLABLE:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_CALLABLE_BIT_KHR;
        break;
      case ShaderStageEnum::ANY_HIT:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        break;
      case ShaderStageEnum::TASK:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_TASK_BIT_EXT;
        break;
      case ShaderStageEnum::MESH:
        return VkShaderStageFlagBits::VK_SHADER_STAGE_MESH_BIT_EXT;
        break;
      default: {
        se::error("RHI :: Vulkan :: Unkown shader stage while creating shader module");
        return VkShaderStageFlagBits::VK_SHADER_STAGE_ALL;
        break;
      }
             break;
      }
    }
  
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
      if (entry.accelerationStructure.has_value())
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      if (entry.bindlessTextures.has_value())
        return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      se::error("rhi::getVkDecriptorType:: get wrong type");
      return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }

    inline auto getVkShaderStageFlags(Flags<ShaderStageEnum> flags) noexcept
      -> VkShaderStageFlags {
      VkShaderStageFlags ret = 0;
      if (flags & (uint32_t)ShaderStageEnum::VERTEX)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT;
      if (flags & (uint32_t)ShaderStageEnum::FRAGMENT)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT;
      if (flags & (uint32_t)ShaderStageEnum::GEOMETRY)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_GEOMETRY_BIT;
      if (flags & (uint32_t)ShaderStageEnum::TASK)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_TASK_BIT_EXT;
      if (flags & (uint32_t)ShaderStageEnum::MESH)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_MESH_BIT_EXT;
      if (flags & (uint32_t)ShaderStageEnum::COMPUTE)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT;
      if (flags & (uint32_t)ShaderStageEnum::RAYGEN)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_NV;
      if (flags & (uint32_t)ShaderStageEnum::MISS)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV;
      if (flags & (uint32_t)ShaderStageEnum::CLOSEST_HIT)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
      if (flags & (uint32_t)ShaderStageEnum::INTERSECTION)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_INTERSECTION_BIT_NV;
      if (flags & (uint32_t)ShaderStageEnum::ANY_HIT)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_ANY_HIT_BIT_NV;
      if (flags & (uint32_t)ShaderStageEnum::CALLABLE)
        ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_CALLABLE_BIT_NV;
      return ret;
    }

    inline auto getVkQueryResultFlags(Flags<QueryResultEnum> flags) noexcept
      -> VkQueryResultFlags {
      uint32_t flag = 0;
      if (flags & QueryResultEnum::RESULT_64) flag |= VK_QUERY_RESULT_64_BIT;
      if (flags & QueryResultEnum::RESULT_WAIT) flag |= VK_QUERY_RESULT_WAIT_BIT;
      if (flags & QueryResultEnum::RESULT_WITH_AVAILABILITY) flag |= VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
      if (flags & QueryResultEnum::RESULT_PARTIAL) flag |= VK_QUERY_RESULT_PARTIAL_BIT;
      return flag;
    }

    inline auto getVkMemoryPropertyFlags(
      Flags<MemoryPropertyEnum> memoryProperties) noexcept -> VkMemoryPropertyFlags {
      VkMemoryPropertyFlags flags = 0;
      if (memoryProperties & MemoryPropertyEnum::DEVICE_LOCAL_BIT)
        flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      if (memoryProperties & MemoryPropertyEnum::HOST_VISIBLE_BIT)
        flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      if (memoryProperties & MemoryPropertyEnum::HOST_COHERENT_BIT)
        flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if (memoryProperties & MemoryPropertyEnum::HOST_CACHED_BIT)
        flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      if (memoryProperties & MemoryPropertyEnum::LAZILY_ALLOCATED_BIT)
        flags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
      if (memoryProperties & MemoryPropertyEnum::PROTECTED_BIT)
        flags |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
      return flags;
    }

    inline auto getVkBufferUsageFlags(Flags<BufferUsageEnum> usage) noexcept
      -> VkBufferUsageFlags {
      VkBufferUsageFlags flags = 0;
      if (usage & BufferUsageEnum::COPY_SRC)
        flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      if (usage & BufferUsageEnum::COPY_DST)
        flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      if (usage & BufferUsageEnum::INDEX)
        flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      if (usage & BufferUsageEnum::VERTEX)
        flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      if (usage & BufferUsageEnum::UNIFORM)
        flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      if (usage & BufferUsageEnum::STORAGE)
        flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      if (usage & BufferUsageEnum::INDIRECT)
        flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      if (usage & BufferUsageEnum::QUERY_RESOLVE)
        flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      if (usage & BufferUsageEnum::SHADER_DEVICE_ADDRESS)
        flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      if (usage & BufferUsageEnum::ACCELERATION_STRUCTURE_STORAGE)
        flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
      if (usage & BufferUsageEnum::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
        flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      if (usage & BufferUsageEnum::SHADER_BINDING_TABLE)
        flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
      return flags;
    }
}

  namespace Aftermath {
    std::mutex m_mutex;
    // Static wrapper for the GPU crash dump handler. See the 'Handling GPU crash
    // dump Callbacks' section for details.
    void GpuCrashDumpCallback(const void* pGpuCrashDump,
      const uint32_t gpuCrashDumpSize, void* pUserData) {
      //// Make sure only one thread at a time...
      std::lock_guard<std::mutex> lock(m_mutex);
      se::error("CrashDumpDescriptionCallback");
      ////  Write to file for later in-depth analysis.
      se::MiniBuffer buffer;
      buffer.m_isReference = true;
      buffer.m_data = (void*)pGpuCrashDump;
      buffer.m_size = gpuCrashDumpSize;
      std::string timestamp = "1";
      std::string project_path = se::Configuration::string_property("project_path");
      se::Filesys::sync_write_file(std::string(project_path + "/" + timestamp + ".nv-gpudmp").c_str(), buffer);
    }

    // Static wrapper for the shader debug information handler. See the 'Handling
    // Shader Debug Information callbacks' section for details.
    void ShaderDebugInfoCallback(const void* pShaderDebugInfo,
      const uint32_t shaderDebugInfoSize,
      void* pUserData) {
      se::error("ShaderDebugInfoCallback");
    }

    // Static wrapper for the GPU crash dump description handler. See the
    // 'Handling GPU Crash Dump Description Callbacks' section for details.
    void CrashDumpDescriptionCallback(
      PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription,
      void* pUserData) {
      // GpuCrashTracker* pGpuCrashTracker =
      //     reinterpret_cast<GpuCrashTracker*>(pUserData);
      // pGpuCrashTracker->OnDescription(addDescription);

      se::error("CrashDumpDescriptionCallback");
    }

    // Static wrapper for the resolve marker handler. See the 'Handling Marker
    // Resolve Callbacks' section for details.
    void ResolveMarkerCallback(const void* pMarkerData,
      const uint32_t markerDataSize, void* pUserData,
      void** ppResolvedMarkerData,
      uint32_t* pResolvedMarkerDataSize) {
      se::error("ResolveMarkerCallback");
      // GpuCrashTracker* pGpuCrashTracker =
      // reinterpret_cast<GpuCrashTracker*>(pUserData);
      // pGpuCrashTracker->OnResolveMarker(pMarkerData, markerDataSize,
      // ppResolvedMarkerData, pResolvedMarkerDataSize);
    }
  }

  bool TextureViewIndex::operator==(TextureViewIndex const& p) const {
    return type == p.type && mostDetailedMip == p.mostDetailedMip &&
      mipCount == p.mipCount && firstArraySlice == p.firstArraySlice &&
      arraySize == p.arraySize;
  }

  auto has_depth_bit(TextureFormat format) noexcept -> bool {
    return format == TextureFormat::DEPTH16_UNORM ||
      format == TextureFormat::DEPTH24 ||
      format == TextureFormat::DEPTH24STENCIL8 ||
      format == TextureFormat::DEPTH32_FLOAT ||
      format == TextureFormat::DEPTH32STENCIL8;
  }

  auto has_stencil_bit(TextureFormat format) noexcept -> bool {
    return format == TextureFormat::STENCIL8 ||
      format == TextureFormat::DEPTH24STENCIL8 ||
      format == TextureFormat::DEPTH32STENCIL8;
  }

  auto get_vk_image_layout(TextureLayoutEnum layout) noexcept -> VkImageLayout {
    switch (layout) {
    case TextureLayoutEnum::UNDEFINED:
      return VK_IMAGE_LAYOUT_UNDEFINED;
    case TextureLayoutEnum::GENERAL:
      return VK_IMAGE_LAYOUT_GENERAL;
    case TextureLayoutEnum::COLOR_ATTACHMENT_OPTIMAL:
      return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case TextureLayoutEnum::DEPTH_STENCIL_ATTACHMENT_OPTIMA:
      return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case TextureLayoutEnum::DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    case TextureLayoutEnum::SHADER_READ_ONLY_OPTIMAL:
      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case TextureLayoutEnum::TRANSFER_SRC_OPTIMAL:
      return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case TextureLayoutEnum::TRANSFER_DST_OPTIMAL:
      return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case TextureLayoutEnum::PREINITIALIZED:
      return VK_IMAGE_LAYOUT_PREINITIALIZED;
    case TextureLayoutEnum::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
    case TextureLayoutEnum::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
      return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
    case TextureLayoutEnum::DEPTH_ATTACHMENT_OPTIMAL:
      return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    case TextureLayoutEnum::DEPTH_READ_ONLY_OPTIMAL:
      return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    case TextureLayoutEnum::STENCIL_ATTACHMENT_OPTIMAL:
      return VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
    case TextureLayoutEnum::STENCIL_READ_ONLY_OPTIMAL:
      return VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
    case TextureLayoutEnum::PRESENT_SRC:
      return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case TextureLayoutEnum::SHARED_PRESENT:
      return VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
    case TextureLayoutEnum::FRAGMENT_DENSITY_MAP_OPTIMAL:
      return VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
    case TextureLayoutEnum::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL:
      return VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
    case TextureLayoutEnum::READ_ONLY_OPTIMAL:
      return VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
    case TextureLayoutEnum::ATTACHMENT_OPTIMAL:
      return VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
    default:
      return VK_IMAGE_LAYOUT_MAX_ENUM;
    }
  }

  auto get_texture_aspect(TextureFormat format) noexcept -> Flags<TextureAspectEnum> {
    switch (format) {
    case TextureFormat::STENCIL8:
      return TextureAspectEnum::STENCIL_BIT;
    case TextureFormat::DEPTH16_UNORM:
    case TextureFormat::DEPTH24:
    case TextureFormat::DEPTH32_FLOAT:
      return TextureAspectEnum::DEPTH_BIT;
    case TextureFormat::DEPTH24STENCIL8:
    case TextureFormat::DEPTH32STENCIL8:
      return TextureAspectEnum::DEPTH_BIT |
        TextureAspectEnum::STENCIL_BIT;
    default:
      return TextureAspectEnum::COLOR_BIT;
    }
  }

  Context::Context(Window* window, Flags<ContextExtensionEnum> ext) {
    m_bindedWindow = window;
    if ((ext & ContextExtensionEnum::USE_AFTERMATH)) {
      // Enable GPU crash dumps and register callbacks.
      GFSDK_Aftermath_EnableGpuCrashDumps(
        GFSDK_Aftermath_Version_API,
        GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
        GFSDK_Aftermath_GpuCrashDumpFeatureFlags_Default,  // Default behavior.
        Aftermath::GpuCrashDumpCallback,  // Register callback for GPU crash dumps.
        Aftermath::ShaderDebugInfoCallback,  // Register callback for shader debug information.
        Aftermath::CrashDumpDescriptionCallback,  // Register callback for GPU crash dump description.
        Aftermath::ResolveMarkerCallback,  // Register callback for marker resolution (R495 or later NVIDIA graphics driver).
        nullptr);               // Set the GpuCrashTracker object as user data passed back by the above callbacks.
    }

    // create VkInstance
    impl::createInstance(this, ext);
    if ((ext & ContextExtensionEnum::DEBUG_UTILS)) {
      impl::setupDebugMessenger(this);
    }
    impl::setupExtensions(this, ext);
    impl::attachWindow(this);
    // set extensions
    m_extensions = ext;
  }

  auto Context::destroy() noexcept -> void {
    if ((m_extensions & ContextExtensionEnum::USE_AFTERMATH)) {
      // Disable GPU crash dump creation.
      GFSDK_Aftermath_DisableGpuCrashDumps();
    }
    if ((m_extensions & ContextExtensionEnum::DEBUG_UTILS))
      impl::destroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);
  }

  auto Context::request_adapter(PowerPreferenceEnum pp) noexcept -> std::unique_ptr<Adapter> {
    if (m_devices.size() == 0) impl::queryAllPhysicalDevice(this);
    if (m_devices.size() == 0) return nullptr;
    else {
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(m_devices[0], &deviceProperties);
      se::debug("VULKAN :: Adapter selected, Name: {}", deviceProperties.deviceName);
      return std::make_unique<Adapter>(m_devices[0], this, deviceProperties);
    }
  }

  auto Adapter::QueueFamilyIndices::is_complete() noexcept -> bool {
    return m_graphicsFamily.has_value() && m_presentFamily.has_value() &&
      m_computeFamily.has_value();
  }

  Adapter::Adapter(VkPhysicalDevice device, Context* context,
    VkPhysicalDeviceProperties const& properties)
    : m_physicalDevice(device),
    m_context(context),
    m_adapterInfo([&]() -> AdapterInfo {
      AdapterInfo info;
      info.device = properties.deviceName;
      info.vendor = properties.vendorID;
      info.architecture = properties.deviceType;
      info.description = properties.deviceID;
      info.timestampPeriod = properties.limits.timestampPeriod;
      return info;
    }()),
    m_timestampPeriod(properties.limits.timestampPeriod),
    m_queueFamilyIndices(impl::findQueueFamilies(context, m_physicalDevice)),
    m_properties(properties) {}

  auto Adapter::get_vk_device_extensions() noexcept -> std::vector<const char*>& {
    return m_context->get_vk_device_extensions();
  }

  auto Adapter::find_memory_type(uint32_t typeFilter,
    VkMemoryPropertyFlags properties) noexcept -> uint32_t {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
      if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        return i;
    se::error("VULKAN :: failed to find suitable memory type!");
    return 0;
  }

  auto Adapter::request_device() noexcept -> std::unique_ptr<Device> {
    // get queues
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        m_queueFamilyIndices.m_graphicsFamily.value(),
        m_queueFamilyIndices.m_computeFamily.value() };
    if (m_queueFamilyIndices.m_presentFamily.has_value()) {
      uniqueQueueFamilies.emplace(m_queueFamilyIndices.m_presentFamily.value());
    }
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
      static_cast<uint32_t>(get_vk_device_extensions().size());
    createInfo.ppEnabledExtensionNames = get_vk_device_extensions().data();
    createInfo.pNext = nullptr;
    // get all physical device features chain
    void const** pNextChainHead = &(createInfo.pNext);
    void** pNextChainTail = nullptr;
    VkPhysicalDeviceHostQueryResetFeatures resetFeatures;
    // Add various features
    VkPhysicalDeviceMeshShaderFeaturesNV mesh_shader_feature{};
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::MESH_SHADER) {
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
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::FRAGMENT_BARYCENTRIC) {
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

    // 
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::SAMPLER_FILTER_MIN_MAX) {
      VkPhysicalDeviceVulkan12Features device12features{};
      device12features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
      device12features.pNext = nullptr;
      device12features.shaderInt8 = VK_TRUE;
      device12features.hostQueryReset = VK_TRUE;
      device12features.timelineSemaphore = VK_TRUE;
      device12features.samplerFilterMinmax = VK_TRUE;
      if (pNextChainTail == nullptr) *pNextChainHead = &device12features;
      else *pNextChainTail = &device12features;
      pNextChainTail = &(device12features.pNext);
    }
    // 
    VkPhysicalDeviceCooperativeMatrixFeaturesNV cooperativeMatrixFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV };
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::COOPERATIVE_MATRIX) {
      cooperativeMatrixFeatures.cooperativeMatrix = VK_TRUE;
      //cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess = VK_TRUE;
      if (pNextChainTail == nullptr)
        *pNextChainHead = &cooperativeMatrixFeatures;
      else
        *pNextChainTail = &cooperativeMatrixFeatures;
      pNextChainTail = &(cooperativeMatrixFeatures.pNext);
    }
    VkPhysicalDeviceFeatures2 features2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    void** pFeature2Tail = &(features2.pNext);
    // sub: compute derivative
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR compute_derivative_physics_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR };
    compute_derivative_physics_features.computeDerivativeGroupLinear = VK_TRUE;
    compute_derivative_physics_features.computeDerivativeGroupQuads = VK_TRUE;
    *pFeature2Tail = &compute_derivative_physics_features;
    pFeature2Tail = &(compute_derivative_physics_features.pNext);

    // sub : bindless
    VkPhysicalDeviceDescriptorIndexingFeatures indexing_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT };
    // sub : ray tracing
    VkPhysicalDeviceVulkan12Features features12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceVulkan11Features features11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shader_atomic_float{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT };
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::BINDLESS_INDEXING) {
      if (m_context->get_context_extensions_flags() &
        ContextExtensionEnum::RAY_TRACING) {
        features12.descriptorIndexing = VK_TRUE;
        features12.descriptorBindingPartiallyBound = VK_TRUE;
        features12.runtimeDescriptorArray = VK_TRUE;
      }
      else {
        *pFeature2Tail = &indexing_features;
        pFeature2Tail = &(indexing_features.pNext);
        ;
        indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
        indexing_features.runtimeDescriptorArray = VK_TRUE;
      }
    }
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::RAY_TRACING) {
      *pFeature2Tail = &rayQueryFeatures;
      rayQueryFeatures.pNext = &features12;
      features12.pNext = &features11;
      features11.pNext = &asFeatures;
      asFeatures.pNext = &rtPipelineFeatures;
      pFeature2Tail = &(rtPipelineFeatures.pNext);
      ;
    }
    else {
      *pFeature2Tail = &features12;
      pFeature2Tail = &(features12.pNext);
    }
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::ATOMIC_FLOAT) {
      shader_atomic_float.shaderBufferFloat32AtomicAdd =
        true;  // this allows to perform atomic operations on storage buffers
      shader_atomic_float.shaderBufferFloat32Atomics = true;
      shader_atomic_float.pNext = nullptr;

      *pFeature2Tail = &shader_atomic_float;
      pFeature2Tail = &(shader_atomic_float.pNext);
      ;
    }
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);
    if (pNextChainTail == nullptr)
      *pNextChainHead = &features2;
    else
      *pNextChainTail = &features2;
    pNextChainTail = pFeature2Tail;

    // create logical device
    std::unique_ptr<Device> ptr_device = std::make_unique<Device>();
    Device* device = ptr_device.get();
    if (m_context->get_context_extensions_flags() &
      ContextExtensionEnum::DEBUG_UTILS) {
      device->m_debugLayerEnabled = true;
    }
    else device->m_debugLayerEnabled = false;

    device->m_adapter = this;
    if (vkCreateDevice(get_vk_physical_device(), &createInfo, nullptr,
      &device->m_device) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create logical device!");
    }
    // get queue handlevul
    vkGetDeviceQueue(device->get_vk_device(),
      m_queueFamilyIndices.m_graphicsFamily.value(), 0,
      &device->get_graphics_queue().m_queue);
    if (m_queueFamilyIndices.m_presentFamily.has_value()) {
      vkGetDeviceQueue(device->get_vk_device(),
        m_queueFamilyIndices.m_presentFamily.value(), 0,
        &device->get_present_queue().m_queue);
    }
    vkGetDeviceQueue(device->get_vk_device(),
      m_queueFamilyIndices.m_computeFamily.value(), 0,
      &device->get_compute_queue().m_queue);
    device->get_graphics_queue().m_device = device;
    device->get_present_queue().m_device = device;
    device->get_compute_queue().m_device = device;
    
    // createCommandPools
    device->m_graphicPool = std::make_unique<CommandPool>(device);
    device->m_bindGroupPool = std::make_unique<BindGroupPool>(device);
    device->init();
    return ptr_device;
  }

  auto Device::init() noexcept -> void {
    Flags<ContextExtensionEnum> ext = m_adapter->from_which_context()
      ->get_context_extensions_flags();
    std::vector<VkExternalMemoryHandleTypeFlagsKHR> external_flags;
    // initialize allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorInfo.physicalDevice = m_adapter->get_vk_physical_device();
    allocatorInfo.device = m_device;
    allocatorInfo.instance = m_adapter->from_which_context()->get_vk_instance();
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (ext & ContextExtensionEnum::CUDA_INTEROPERABILITY) {
      VkPhysicalDeviceMemoryProperties memProperties;
      vkGetPhysicalDeviceMemoryProperties(
        from_which_adapter()->get_vk_physical_device(), &memProperties);
      for (auto iter : memProperties.memoryTypes) {
        if (iter.propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        #ifdef _WIN32
          external_flags.push_back(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
        #elif defined(__linux__)
          external_flags.push_back(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);
        #endif
        }
        else {
          external_flags.push_back(0);
        }
      }
      allocatorInfo.pTypeExternalMemoryHandleTypes = external_flags.data();
    }
    vmaCreateAllocator(&allocatorInfo, &m_allocator);
    // initialize extensions
    if (ext & ContextExtensionEnum::RAY_TRACING) {

      m_vkRayTracingProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
      m_vASProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
      // fetch properties
      VkPhysicalDeviceProperties2 prop2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
      prop2.pNext = &m_vkRayTracingProperties;
      m_vkRayTracingProperties.pNext = &m_vASProperties;
      vkGetPhysicalDeviceProperties2(from_which_adapter()->get_vk_physical_device(), &prop2);
    }
  }

  auto Queue::submit(
    std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto buffer : commandBuffers)
      vkCommandBuffers.push_back(buffer->m_commandBuffer);
    submitInfo.commandBufferCount = vkCommandBuffers.size();
    submitInfo.pCommandBuffers = vkCommandBuffers.data();
    vkQueueSubmit(m_device->get_graphics_queue().m_queue, 1, &submitInfo,
      VK_NULL_HANDLE);
  }

  auto Queue::submit(std::vector<CommandBuffer*> const& commandBuffers,
    Fence* fence) noexcept -> void {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto buffer : commandBuffers)
      vkCommandBuffers.push_back(buffer->m_commandBuffer);
    submitInfo.commandBufferCount = vkCommandBuffers.size();
    submitInfo.pCommandBuffers = vkCommandBuffers.data();

    VkResult result;
    if (fence == nullptr) {
      result = vkQueueSubmit(m_device->get_graphics_queue().m_queue,
        1, &submitInfo, VK_NULL_HANDLE);
    }
    else {
      result = vkQueueSubmit(m_device->get_graphics_queue().m_queue,
        1, &submitInfo, fence->m_fence);
    }
    if (result != VK_SUCCESS) {
      se::error("Vulkan :: Queue Submit Failed!");
    }
  }

  auto Queue::submit(std::vector<CommandBuffer*> const& commandBuffers,
    Semaphore* wait, Semaphore* signal, Fence* fence) noexcept
    -> void {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    std::vector<VkSemaphore> waitSemaphores;
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    if (wait) {
      waitSemaphores.push_back(static_cast<Semaphore*>(wait)->m_semaphore);
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pWaitSemaphores = waitSemaphores.data();
      submitInfo.pWaitDstStageMask = waitStages;
    }
    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto buffer : commandBuffers)
      vkCommandBuffers.push_back(buffer->m_commandBuffer);
    submitInfo.commandBufferCount = vkCommandBuffers.size();
    submitInfo.pCommandBuffers = vkCommandBuffers.data();
    std::vector<VkSemaphore> signalSemaphores = {};
    if (signal) {
      signalSemaphores.push_back(signal->m_semaphore);
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores = signalSemaphores.data();
    }
    if (fence) {
      if (vkQueueSubmit(m_device->get_graphics_queue().m_queue, 1, &submitInfo,
        fence->m_fence) != VK_SUCCESS)
        se::error("VULKAN :: failed to submit draw command buffer!");
    }
    else {
      if (vkQueueSubmit(m_device->get_graphics_queue().m_queue, 1, &submitInfo,
        VK_NULL_HANDLE) != VK_SUCCESS)
        se::error("VULKAN :: failed to submit draw command buffer!");
    }
  }

  auto Queue::submit(std::vector<CommandBuffer*> const& commandBuffers,
    std::vector<Semaphore*> const& wait_semaphores,
    std::vector<size_t>     const& wait_indices,
    std::vector<Flags<PipelineStageEnum>> const& wait_stages,
    std::vector<Semaphore*> const& signal_semaphores,
    std::vector<size_t>     const& signal_indices,
    Fence* fence) noexcept
    -> void {

    VkTimelineSemaphoreSubmitInfo timelineInfo;
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.pNext = NULL;
    timelineInfo.waitSemaphoreValueCount = wait_indices.size();
    timelineInfo.pWaitSemaphoreValues = wait_indices.data();
    timelineInfo.signalSemaphoreValueCount = signal_indices.size();
    timelineInfo.pSignalSemaphoreValues = signal_indices.data();

    std::vector<VkSemaphore> wait_semaphores_vk;
    std::vector<VkPipelineStageFlags> wait_pipeline_stages;
    wait_semaphores_vk.reserve(wait_semaphores.size());
    wait_pipeline_stages.reserve(wait_semaphores.size());
    for (auto& iter : wait_semaphores)wait_semaphores_vk.emplace_back(iter->m_semaphore);
    for (auto& iter : wait_stages) wait_pipeline_stages.emplace_back(impl::getVkPipelineStageFlags(iter));

    std::vector<VkSemaphore> signal_semaphores_vk;
    signal_semaphores_vk.reserve(signal_semaphores.size());
    for (auto& iter : signal_semaphores)signal_semaphores_vk.emplace_back(iter->m_semaphore);

    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto buffer : commandBuffers) vkCommandBuffers.push_back(buffer->m_commandBuffer);

    VkSubmitInfo info;
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = &timelineInfo;
    info.waitSemaphoreCount = wait_semaphores_vk.size();
    info.pWaitSemaphores = wait_semaphores_vk.data();
    info.signalSemaphoreCount = signal_semaphores_vk.size();
    info.pSignalSemaphores = signal_semaphores_vk.data();
    info.commandBufferCount = vkCommandBuffers.size();
    info.pCommandBuffers = vkCommandBuffers.data();
    info.pWaitDstStageMask = wait_pipeline_stages.data();
    
    if (fence) {
      if (vkQueueSubmit(m_device->get_graphics_queue().m_queue, 1, &info,
        fence->m_fence) != VK_SUCCESS)
        se::error("VULKAN :: failed to submit draw command buffer!");
    }
    else {
      if (vkQueueSubmit(m_device->get_graphics_queue().m_queue, 1, &info,
        VK_NULL_HANDLE) != VK_SUCCESS)
        se::error("VULKAN :: failed to submit draw command buffer!");
    }
  }

  auto Queue::present_swapchain(SwapChain* swapchain, uint32_t imageIndex,
    Semaphore* semaphore) noexcept -> void {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores =
      &static_cast<Semaphore*>(semaphore)->m_semaphore;
    VkSwapchainKHR swapChains[] = {
        static_cast<SwapChain*>(swapchain)->m_swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(m_queue, &presentInfo);
  }
  
  auto Queue::wait_idle() noexcept -> void { vkQueueWaitIdle(m_queue); }

  auto Queue::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_QUEUE;
    objectNameInfo.objectHandle = uint64_t(m_queue);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()
      ->from_which_context()->vkSetDebugUtilsObjectNameEXT(
        m_device->get_vk_device(), &objectNameInfo);
  }


  Device::~Device() { destroy(); }

  auto Device::destroy() noexcept -> void {
    m_graphicPool = nullptr; 
    m_computePool = nullptr; 
    m_presentPool = nullptr;
    m_bindGroupPool = nullptr;
#ifdef USE_VMA
    vmaDestroyAllocator(m_allocator);
#endif
    if (m_device) vkDestroyDevice(m_device, nullptr);
  }

  auto Device::wait_idle() noexcept -> void {
    VkResult result = vkDeviceWaitIdle(m_device);
    if (result != VK_SUCCESS) {
      std::string error_str = "UNKOWN";
      switch (result) {
        case VK_SUCCESS: error_str="Success"; break;
        case VK_NOT_READY: error_str="Not Ready"; break;
        case VK_TIMEOUT: error_str="Timeout"; break;
        case VK_EVENT_SET: error_str="Event Set"; break;
        case VK_EVENT_RESET: error_str="Event Reset"; break;
        case VK_INCOMPLETE: error_str="Incomplete"; break;
        case VK_ERROR_OUT_OF_HOST_MEMORY: error_str="Out of host memory"; break;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: error_str="Out of device memory"; break;
        case VK_ERROR_INITIALIZATION_FAILED: error_str="Initialization failed"; break;
        case VK_ERROR_DEVICE_LOST: error_str="Device lost"; break;
        case VK_ERROR_MEMORY_MAP_FAILED: error_str="Memory map failed"; break;
        case VK_ERROR_LAYER_NOT_PRESENT: error_str="Layer not present"; break;
        case VK_ERROR_EXTENSION_NOT_PRESENT: error_str="Extension not present"; break;
        case VK_ERROR_FEATURE_NOT_PRESENT: error_str="Feature not present"; break;
        case VK_ERROR_INCOMPATIBLE_DRIVER: error_str="Incompatible driver"; break;
        case VK_ERROR_TOO_MANY_OBJECTS: error_str="Too many objects"; break;
        case VK_ERROR_FORMAT_NOT_SUPPORTED: error_str="Format not supported"; break;
        case VK_ERROR_FRAGMENTED_POOL: error_str="Fragmented pool"; break;
        case VK_ERROR_UNKNOWN: error_str="Unkown"; break;
        case VK_ERROR_OUT_OF_POOL_MEMORY: error_str="Out of pool memory"; break;
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: error_str="Invalid external handle"; break;
        case VK_ERROR_FRAGMENTATION: error_str="Fragmentation"; break;
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: error_str="Invalid opaque capture address"; break;
        case VK_PIPELINE_COMPILE_REQUIRED: error_str="Compile required"; break;
        case VK_ERROR_SURFACE_LOST_KHR: error_str="Surface lost"; break;
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: error_str="Native window in use"; break;
        case VK_SUBOPTIMAL_KHR: error_str="Suboptimal"; break;
        case VK_ERROR_OUT_OF_DATE_KHR: error_str="Out of date"; break;
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: error_str="Incompatible display"; break;
        case VK_ERROR_VALIDATION_FAILED_EXT: error_str="Invalidation failed"; break;
        case VK_ERROR_INVALID_SHADER_NV: error_str="Invalid shader"; break;
        case VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR:  error_str="Image usage not supported"; break;
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:error_str="Invalid DRM format modifier plane layout"; break;
        case VK_ERROR_NOT_PERMITTED_KHR: error_str="Not permitted"; break;
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:error_str="Full screen exclusive model lost"; break;
        case VK_THREAD_IDLE_KHR: error_str="Thread Idle"; break;
        case VK_THREAD_DONE_KHR: error_str="Thread Done"; break;
        case VK_OPERATION_DEFERRED_KHR: error_str="Operation deferred"; break;
        case VK_OPERATION_NOT_DEFERRED_KHR: error_str="Operation not deferred"; break;
        case VK_ERROR_COMPRESSION_EXHAUSTED_EXT: error_str="Error compression exhausted"; break;
        default: break;
      }
    
      if (result == VK_ERROR_DEVICE_LOST) {
        GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
        GFSDK_Aftermath_GetCrashDumpStatus(&status);
        auto tStart = std::chrono::steady_clock::now();
        auto tElapsed = std::chrono::milliseconds::zero();
        // Loop while Aftermath crash dump data collection has not finished or
        // the application is still processing the crash dump data.
        while (status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed &&
               status != GFSDK_Aftermath_CrashDump_Status_Finished &&
               tElapsed.count() < 1500000)
        {
          // Sleep a couple of milliseconds and poll the status again.
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          GFSDK_Aftermath_GetCrashDumpStatus(&status);
          tElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tStart);
        }

        if (status == GFSDK_Aftermath_CrashDump_Status_Finished) {
          se::error("Aftermath finished processing the crash dump.\n");
        } else {
          se::error("Unexpected crash dump status after timeout: " + std::to_string(status));
        }
      }
      se::error("VULKAN :: Device WaitIdle not Success! Error code: "+ error_str);
    }
  }
  
  auto Device::create_command_encoder(CommandBuffer* external) noexcept
    -> std::unique_ptr<CommandEncoder> {

    std::unique_ptr<CommandEncoder> encoder =
      std::make_unique<CommandEncoder>();
    if (external) {
      encoder->m_commandBuffer = external;
    }
    else {
      encoder->m_commandBufferOnce = m_graphicPool->allocate_command_buffer();
      encoder->m_commandBuffer = encoder->m_commandBufferOnce.get();
    }
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;
    if (vkBeginCommandBuffer(encoder->m_commandBuffer->m_commandBuffer, &beginInfo) !=
      VK_SUCCESS) {
      se::error("VULKAN :: failed to begin recording command buffer!");
    }
    return encoder;
  }

  auto Device::create_buffer(BufferDescriptor const& _desc) noexcept
    -> std::unique_ptr<Buffer> {
    BufferDescriptor desc = _desc;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = desc.size;
    bufferInfo.usage = impl::getVkBufferUsageFlags(desc.usage);
    bufferInfo.sharingMode = desc.shareMode == BufferShareMode::EXCLUSIVE
      ? VK_SHARING_MODE_EXCLUSIVE
      : VK_SHARING_MODE_CONCURRENT;

    VkExternalMemoryBufferCreateInfo external_info;
    if (static_cast<Context*>(m_adapter->from_which_context())->get_context_extensions_flags()
      & ContextExtensionEnum::CUDA_INTEROPERABILITY) {
      external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
      #ifdef _WIN32
      external_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
      #elif defined(__linux__)
      external_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      #endif
      external_info.pNext = nullptr;
      bufferInfo.pNext = &external_info;
    }

    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(this);
    buffer->init(this, desc.size, desc);

    if ((bufferInfo.usage & (uint32_t)BufferUsageEnum::MAP_READ) ||
      bufferInfo.usage & (uint32_t)BufferUsageEnum::MAP_WRITE) {
      desc.memoryProperties |= MemoryPropertyEnum::HOST_VISIBLE_BIT;
      desc.memoryProperties |= MemoryPropertyEnum::HOST_COHERENT_BIT;
    }

#ifdef USE_VMA
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (desc.memoryProperties & MemoryPropertyEnum::HOST_VISIBLE_BIT ||
      desc.memoryProperties & MemoryPropertyEnum::HOST_COHERENT_BIT) {
      allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
      allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    if (desc.minimumAlignment == -1) {
      // do not use alignment
      if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo,
        &buffer->get_vk_buffer(), &buffer->get_vma_allocation(),
        nullptr) != VK_SUCCESS) {
        se::error("VULKAN :: failed to create a device buffer!");
      }
    }
    else {
      // use alignment
      if (vmaCreateBufferWithAlignment(m_allocator, &bufferInfo, &allocInfo, desc.minimumAlignment,
        &buffer->get_vk_buffer(), &buffer->get_vma_allocation(),
        nullptr) != VK_SUCCESS) {
        se::error("VULKAN :: failed to create a device buffer with alignment!");
      }
    }
#else
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer->getVkBuffer()) !=
      VK_SUCCESS) {
      se::error("VULKAN :: failed to create vulkan buffer!");
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
    if (desc.usage & (uint32_t)BufferUsageBit::SHADER_DEVICE_ADDRESS) {
      allocFlagsInfo.flags =
        VkMemoryAllocateFlagBits::VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
      allocInfo.pNext = &allocFlagsInfo;
    }
    if (vkAllocateMemory(device, &allocInfo, nullptr,
      &buffer->getVkDeviceMemory()) != VK_SUCCESS) {
      se::error("VULKAN :: failed to allocate vulkan buffer memory!");
    }
    vkBindBufferMemory(device, buffer->getVkBuffer(), buffer->getVkDeviceMemory(),
      0);
#endif
    return buffer;
  }

  VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() {
#ifdef _WIN64
    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
  }

  //void importExternalBuffer(
  //  void* handle,
  //  VkExternalMemoryHandleTypeFlagBits handleType,
  //  size_t                             size,
  //  VkBufferUsageFlags                 usage,
  //  VkMemoryPropertyFlags              properties,
  //  VkBuffer& buffer,
  //  VkDeviceMemory& memory)
  //{

  static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, 
    uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    return ~0;
  }

  auto Device::import_buffer(void* externalHandle, BufferDescriptor const& _desc) noexcept
    -> std::unique_ptr<Buffer> {
    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(this);
    buffer->init(this, _desc.size, _desc);
    buffer->m_external = true;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = _desc.size;
    bufferInfo.usage = impl::getVkBufferUsageFlags(_desc.usage);
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
    externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    bufferInfo.pNext = &externalMemoryBufferInfo;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer->m_buffer) != VK_SUCCESS) {
      se::error("Device::create_buffer:: failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer->m_buffer, &memRequirements);

#ifdef _WIN64
    VkImportMemoryWin32HandleInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    handleInfo.pNext = NULL;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    handleInfo.handle = externalHandle;
    handleInfo.name = NULL;
#else
    VkImportMemoryFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    handleInfo.pNext = NULL;
    handleInfo.fd = (int)(uintptr_t)externalHandle;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */

    VkMemoryAllocateInfo memAllocation = {};
    memAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocation.pNext = (void*)&handleInfo;
    memAllocation.allocationSize = memRequirements.size;
    memAllocation.memoryTypeIndex = findMemoryType(from_which_adapter()->m_physicalDevice, 
      memRequirements.memoryTypeBits, impl::getVkMemoryPropertyFlags(_desc.memoryProperties));

    if (vkAllocateMemory(m_device, &memAllocation, nullptr, &buffer->m_bufferMemory) != VK_SUCCESS) {
      se::error("Failed to import allocation!");
    }

    vkBindBufferMemory(m_device, buffer->m_buffer, buffer->m_bufferMemory, 0);
    return buffer;
  }

  auto Device::create_texture(TextureDescriptor const& desc) noexcept
    -> std::unique_ptr<Texture> {
    return std::make_unique<Texture>(this, desc);
  }

  auto Device::create_sampler(SamplerDescriptor const& desc) noexcept
    -> std::unique_ptr<Sampler> {
    return std::make_unique<Sampler>(desc, this);
  }

  auto Device::create_swapchain() noexcept -> std::unique_ptr<SwapChain> {
    std::unique_ptr<SwapChain> swapChain = std::make_unique<SwapChain>();
    swapChain->init(this);
    return swapChain;
  }
  
  auto Device::create_blas(BLASDescriptor const& desc) -> std::unique_ptr<BLAS> {
    if (desc.customGeometries.size() == 0 &&
      desc.triangleGeometries.size() == 0) {
      se::error("RHI :: Vulkan :: Create BLAS with no input geometry!");
      return nullptr;
    }
    return std::make_unique<BLAS>(this, desc);
  }
  
  auto Device::create_tlas(TLASDescriptor const& desc) -> std::unique_ptr<TLAS> {
    return std::make_unique<TLAS>(this, desc);
  }

  auto Device::create_bindgroup_layout(
    BindGroupLayoutDescriptor const& desc) noexcept
    -> std::unique_ptr<BindGroupLayout> {
    return std::make_unique<BindGroupLayout>(this, desc);
  }

  auto Device::create_pipeline_layout(
    PipelineLayoutDescriptor const& desc) noexcept
    -> std::unique_ptr<PipelineLayout> {
    return std::make_unique<PipelineLayout>(this, desc);
  }

  auto Device::create_bindgroup(BindGroupDescriptor const& desc) noexcept
    -> std::unique_ptr<BindGroup> {
    return std::make_unique<BindGroup>(this, desc);
  }

  auto Device::create_shader_module(ShaderModuleDescriptor const& desc) noexcept
    -> std::unique_ptr<ShaderModule> {
    std::unique_ptr<ShaderModule> shadermodule =
      std::make_unique<ShaderModule>(this, desc);
    return shadermodule;
  }

  auto Device::create_compute_pipeline(
    ComputePipelineDescriptor const& desc) noexcept
    -> std::unique_ptr<ComputePipeline> {
    return std::make_unique<ComputePipeline>(this, desc);
  }

  auto Device::create_render_pipeline(
    RenderPipelineDescriptor const& desc) noexcept
    -> std::unique_ptr<RenderPipeline> {
    return std::make_unique<RenderPipeline>(this, desc);
  }

  auto Device::create_frame_resources(int maxFlightNum,
    SwapChain* swapchain) noexcept
    -> std::unique_ptr<FrameResources> {
    return std::make_unique<FrameResources>(this, maxFlightNum,
      swapchain);
  }

  auto Device::create_semaphore(bool use_timeline, bool allow_export) noexcept -> std::unique_ptr<Semaphore> {
    VkSemaphoreTypeCreateInfo timelineCreateInfo;
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.pNext = NULL;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = use_timeline ? &timelineCreateInfo : NULL;
    createInfo.flags = 0;

    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    #ifdef _WIN32
    exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #elif defined(__linux__)
    exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif

    if (allow_export) {
      if (use_timeline) timelineCreateInfo.pNext = &exportSemaphoreCreateInfo;
      else createInfo.pNext = &exportSemaphoreCreateInfo;;
    }

    VkSemaphore createdSemaphore;
    vkCreateSemaphore(get_vk_device(), &createInfo, NULL, &createdSemaphore);

    std::unique_ptr<Semaphore> semaphore = std::make_unique<Semaphore>();
    semaphore->m_semaphore = createdSemaphore;
    semaphore->m_device = this;
    semaphore->m_timelineSemaphore = use_timeline;
    return semaphore;
  }

  auto Device::allocate_command_buffer() noexcept
    -> std::unique_ptr<CommandBuffer> {
    return m_graphicPool->allocate_command_buffer();
  }

  auto Device::create_fence() noexcept -> std::unique_ptr<Fence> {
    return std::make_unique<Fence>(this);
  }

  auto Device::create_device_local_buffer(void const* data, uint32_t size,
    Flags<BufferUsageEnum> usage) noexcept -> std::unique_ptr<Buffer> {
    std::unique_ptr<Buffer> buffer = nullptr;
    // create vertex buffer
    BufferDescriptor descriptor;
    descriptor.size = size;
    descriptor.usage = usage | BufferUsageEnum::COPY_DST | BufferUsageEnum::COPY_SRC;
    descriptor.memoryProperties = MemoryPropertyEnum::DEVICE_LOCAL_BIT;
    descriptor.mappedAtCreation = true;
    buffer = create_buffer(descriptor);
    // create staging buffer
    BufferDescriptor stagingBufferDescriptor;
    stagingBufferDescriptor.size = size;
    stagingBufferDescriptor.usage = BufferUsageEnum::COPY_SRC;
    stagingBufferDescriptor.memoryProperties =
      MemoryPropertyEnum::HOST_VISIBLE_BIT |
      MemoryPropertyEnum::HOST_COHERENT_BIT;
    stagingBufferDescriptor.mappedAtCreation = true;
    std::unique_ptr<Buffer> stagingBuffer =
      create_buffer(stagingBufferDescriptor);
    std::future<bool> mapped = stagingBuffer->map_async(0, 0, descriptor.size);
    if (mapped.get()) {
      void* mapdata = stagingBuffer->get_mapped_range(0);
      memcpy(mapdata, data, (size_t)descriptor.size);
      stagingBuffer->unmap();
    }
    std::unique_ptr<CommandEncoder> commandEncoder =
      create_command_encoder({ nullptr });
    commandEncoder->pipeline_barrier(BarrierDescriptor{
        PipelineStageEnum::HOST_BIT,
        PipelineStageEnum::TRANSFER_BIT,
        0,
        // Optional (Memory Barriers)
        {},
        {BufferMemoryBarrierDescriptor{
            stagingBuffer.get(),
            AccessFlagEnum::HOST_WRITE_BIT,
            AccessFlagEnum::TRANSFER_READ_BIT,
        }},
        {} });
    commandEncoder->copy_buffer_to_buffer(stagingBuffer.get(), 0, buffer.get(), 0, descriptor.size);
    std::unique_ptr<Fence> fence = create_fence();
    fence->reset();
    get_graphics_queue().submit({ commandEncoder->finish() }, fence.get());
    get_graphics_queue().wait_idle();
    fence->wait();
    return buffer;
  }

  auto Device::readback_device_local_buffer(Buffer* buffer, 
    void* data, uint32_t size) noexcept -> void {
    // create staging buffer
    BufferDescriptor stagingBufferDescriptor;
    stagingBufferDescriptor.size = size;
    stagingBufferDescriptor.usage = BufferUsageEnum::COPY_DST;
    stagingBufferDescriptor.memoryProperties =
      MemoryPropertyEnum::HOST_VISIBLE_BIT |
      MemoryPropertyEnum::HOST_COHERENT_BIT;
    stagingBufferDescriptor.mappedAtCreation = true;
    std::unique_ptr<Buffer> stagingBuffer = create_buffer(stagingBufferDescriptor);
    // copy buffer
    std::unique_ptr<CommandEncoder> commandEncoder = create_command_encoder({ nullptr });
    commandEncoder->pipeline_barrier(BarrierDescriptor{
        PipelineStageEnum::ALL_COMMANDS_BIT,
        PipelineStageEnum::TRANSFER_BIT,
        DependencyTypeEnum::NONE,
        {},
        {BufferMemoryBarrierDescriptor{
            buffer,
            AccessFlagEnum::SHADER_WRITE_BIT,
            AccessFlagEnum::TRANSFER_READ_BIT,
        }} });
    commandEncoder->copy_buffer_to_buffer(buffer, 0, stagingBuffer.get(), 0,
      buffer->size());
    get_graphics_queue().submit({ commandEncoder->finish() });
    get_graphics_queue().wait_idle();
    // buffer readback
    std::future<bool> mapped = stagingBuffer->map_async(0, 0, buffer->size());
    if (mapped.get()) {
      void* mapdata = stagingBuffer->get_mapped_range(0);
      memcpy(data, mapdata, (size_t)buffer->size());
      stagingBuffer->unmap();
    }
  }

  auto Device::query_uuid() noexcept -> std::array<uint64_t, 2> {
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;
    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;
    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
    fpGetPhysicalDeviceProperties2 =
      (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
        from_which_adapter()->from_which_context()->get_vk_instance(), "vkGetPhysicalDeviceProperties2");
    if (fpGetPhysicalDeviceProperties2 == NULL) {
      se::error("Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not found.\n");
    }
    fpGetPhysicalDeviceProperties2(from_which_adapter()->get_vk_physical_device(),
      &vkPhysicalDeviceProperties2);
    std::array<uint64_t, 2> deviceUUID;
    memcpy(deviceUUID.data(), vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE);
    return deviceUUID;
  }

  auto Buffer::get_device_address() const noexcept -> uint64_t {
    VkBufferDeviceAddressInfo deviceAddressInfo = {};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    deviceAddressInfo.buffer = m_buffer;
    return vkGetBufferDeviceAddress(m_device->get_vk_device(), &deviceAddressInfo);
  }

  inline auto mapMemory(Device* device, Buffer* buffer, size_t offset,
    size_t size, void*& mappedData) noexcept -> bool {
#ifdef USE_VMA
    VkResult result = vmaMapMemory(device->get_vma_allocator(),
      buffer->get_vma_allocation(), &mappedData);
#else
    VkResult result =
      vkMapMemory(device->getVkDevice(), buffer->getVkDeviceMemory(), offset,
        size, 0, &mappedData);
#endif
    if (result) buffer->set_buffer_map_state(BufferMapState::MAPPED);
    return result == VkResult::VK_SUCCESS ? true : false;
  }

  auto Buffer::map_async(Flags<MapModeEnum> mode, size_t offset, size_t size) noexcept
    -> std::future<bool> {
    m_mapState = BufferMapState::PENDING;
    return std::async(mapMemory, m_device, this, offset, size,
      std::ref(m_mappedData));
  }

  auto Buffer::get_mapped_range(size_t offset) noexcept
    -> void* {
    return (void*)&(((char*)m_mappedData)[offset]);
  }

  auto Buffer::unmap() noexcept -> void {
#ifdef USE_VMA
    vmaUnmapMemory(m_device->get_vma_allocator(), get_vma_allocation());
#else
    vkUnmapMemory(device->getVkDevice(), bufferMemory);
#endif
    m_mappedData = nullptr;
    BufferMapState mapState = BufferMapState::UNMAPPED;
  }

  Buffer::~Buffer() { destroy(); }

  auto Buffer::destroy() noexcept -> void {
    if (m_mappedData) {
      unmap();
    }
    if (m_external) {
      if (m_buffer) vkDestroyBuffer(m_device->get_vk_device(), m_buffer, nullptr);
      if (m_bufferMemory) vkFreeMemory(m_device->get_vk_device(), m_bufferMemory, nullptr);
    }
    else {
      #ifdef USE_VMA
      if (m_buffer)
        vmaDestroyBuffer(m_device->get_vma_allocator(),
          m_buffer, get_vma_allocation());
      #else
      if (buffer) vkDestroyBuffer(device->getVkDevice(), buffer, nullptr);
      if (bufferMemory) vkFreeMemory(device->getVkDevice(), bufferMemory, nullptr);
      #endif
    }
  }

  auto Buffer::get_mem_handle() const noexcept -> ExternalHandle {
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(m_device->get_vma_allocator(), m_allocation, &allocInfo);

    // create extern handle object
    ExternalHandle extern_handle;
    #ifdef _WIN32
        HANDLE handle = 0;
        VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
        vkMemoryGetWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
        vkMemoryGetWin32HandleInfoKHR.memory = allocInfo.deviceMemory;
        vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        if (m_device->from_which_adapter()->from_which_context()->vkCmdGetMemoryWin32HandleKHR(
        m_device->get_vk_device(),
        &vkMemoryGetWin32HandleInfoKHR, &handle) != VK_SUCCESS) {
        se::error("Vulkan :: Failed to retrieve handle for buffer!");
        }

        extern_handle.handle = (void*)handle;
    #elif defined(__linux__)
        int fd = -1;  // File descriptor for external memory

        VkMemoryGetFdInfoKHR getFdInfo{};
        getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        getFdInfo.pNext = nullptr;
        getFdInfo.memory = allocInfo.deviceMemory;
        getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        // Call vkGetMemoryFdKHR
        if (m_device->from_which_adapter()->from_which_context()->vkCmdGetMemoryFdKHR(
            m_device->get_vk_device(), &getFdInfo, &fd) != VK_SUCCESS) {
            se::error("Vulkan :: Failed to retrieve FD handle for buffer!");
        }
        
        // Wrap the file descriptor in your external handle
        extern_handle.handle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));  // Cast to void*
    #endif

    extern_handle.offset = allocInfo.offset;
    extern_handle.size = allocInfo.size;
    return extern_handle;
  }

  auto Buffer::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
    objectNameInfo.objectHandle = uint64_t(m_buffer);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
    this->m_name = name;
  }

  auto Buffer::init(Device* device, size_t size,
    BufferDescriptor const& desc) noexcept -> void {
    this->m_size = size;
    this->m_device = device;
  }

  auto Buffer::get_name() const noexcept -> std::string const& { return m_name; }

  Texture::Texture(Device* device, TextureDescriptor const& desc)
    : m_device(device), m_descriptor{ desc } {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = impl::getVkImageType(desc.dimension);
    imageInfo.extent.width = static_cast<uint32_t>(desc.size.x);
    imageInfo.extent.height = static_cast<uint32_t>(desc.size.y);
    imageInfo.extent.depth = desc.size.z;
    imageInfo.mipLevels = desc.mipLevelCount;
    imageInfo.arrayLayers = desc.arrayLayerCount;
    imageInfo.format = impl::getVkFormat(desc.format);
    imageInfo.tiling = (desc.flags & TextureFeatureEnum::HOST_VISIBLE)
      ? VK_IMAGE_TILING_LINEAR
      : VK_IMAGE_TILING_OPTIMAL;
    if (desc.format >= TextureFormat::COMPRESSION)
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = impl::getVkImageUsageFlagBits(desc.usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = impl::getVkImageCreateFlags(desc.flags);  // Optional

    VkExternalMemoryImageCreateInfo external_info;
    if (static_cast<Context*>(device->from_which_adapter()->from_which_context())->get_context_extensions_flags()
      & ContextExtensionEnum::CUDA_INTEROPERABILITY) {
      external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
      #ifdef _WIN32
      external_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
      #elif defined(__linux__)
      external_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      #endif
      external_info.pNext = nullptr;
      imageInfo.pNext = &external_info;
    }

#ifdef USE_VMA
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (desc.flags & TextureFeatureEnum::HOST_VISIBLE) {
      allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }
    if (vmaCreateImage(device->get_vma_allocator(), &imageInfo, &allocInfo, &m_image,
      &m_allocation, nullptr) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create vertex buffer!");
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

  Texture::Texture(Device* device, VkImage image,
    TextureDescriptor const& desc)
    : m_device(device), m_image(image), m_descriptor{ desc }, m_external(true) {}

  Texture::~Texture() { if (!m_external) destroy(); }

  auto Texture::create_view(TextureViewDescriptor const& desc) noexcept
    -> std::unique_ptr<TextureView> {
    return std::make_unique<TextureView>(m_device, this, desc);
  }

  auto Texture::destroy() noexcept -> void {
#ifdef USE_VMA
    if (m_image) vmaDestroyImage(m_device->get_vma_allocator(), m_image, m_allocation);
#else
    if (image && deviceMemory)
      vkDestroyImage(device->getVkDevice(), image, nullptr);
    if (deviceMemory) vkFreeMemory(device->getVkDevice(), deviceMemory, nullptr);
#endif
  }

  auto Texture::mip_level_count() const noexcept -> uint32_t {
    return m_descriptor.mipLevelCount;
  }

  auto Texture::sample_count() const noexcept -> uint32_t {
    return m_descriptor.sampleCount;
  }

  auto Texture::dimension() const noexcept -> TextureDimension {
    return m_descriptor.dimension;
  }

  auto Texture::format() const noexcept -> TextureFormat {
    return m_descriptor.format;
  }

  auto Texture::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_IMAGE;
    objectNameInfo.objectHandle = uint64_t(m_image);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
    this->m_name = name;
  }

  auto Texture::get_name() -> std::string const& {
    return m_name;
  }

  auto Texture::get_descriptor() -> TextureDescriptor {
    return m_descriptor;
  }

  auto Texture::map_async(Flags<MapModeEnum> mode, size_t offset,
    size_t size) noexcept -> std::future<bool> {
    m_mapState = BufferMapState::PENDING;
    return std::async(impl::mapMemoryTexture, m_device, this, offset, size,
      std::ref(m_mappedData));
  }

  auto Texture::get_mapped_range(size_t offset, size_t size) noexcept -> void* {
    return (void*)&(((char*)m_mappedData)[offset]);
  }

  auto Texture::unmap() noexcept -> void {
#ifdef USE_VMA
    vmaUnmapMemory(m_device->get_vma_allocator(), get_vma_allocation());
#else
    vkUnmapMemory(device->getVkDevice(), getVkDeviceMemory());
#endif
    m_mappedData = nullptr;
    BufferMapState mapState = BufferMapState::UNMAPPED;
  }

  TextureView::TextureView(TextureView&& view) noexcept
    : m_imageView(view.m_imageView),
    m_descriptor(view.m_descriptor),
    m_texture(view.m_texture),
    m_device(view.m_device),
    m_width(view.m_width),
    m_height(m_width) {
    view.m_imageView = nullptr;
  }

  TextureView::TextureView(Device* device, Texture* texture,
    TextureViewDescriptor const& descriptor)
    : m_device(device), m_texture(texture), m_descriptor(descriptor) {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = texture->get_vk_image();
    createInfo.viewType = impl::getVkImageViewType(descriptor.dimension);
    createInfo.format = impl::getVkFormat(descriptor.format);
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask =
      impl::getVkImageAspectFlags(descriptor.aspect);
    createInfo.subresourceRange.baseMipLevel = descriptor.baseMipLevel;
    createInfo.subresourceRange.levelCount = descriptor.mipLevelCount;
    createInfo.subresourceRange.baseArrayLayer = descriptor.baseArrayLayer;
    createInfo.subresourceRange.layerCount = descriptor.arrayLayerCount;

    m_width = texture->width();
    m_height = texture->height();
    for (int i = 0; i < descriptor.baseMipLevel; ++i) {
      m_width >>= 1;
      m_height >>= 1;
    }
    m_width = std::max(m_width, 1u);
    m_height = std::max(m_height, 1u);

    if (vkCreateImageView(device->get_vk_device(), &createInfo, nullptr,
      &m_imageView) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create image views!");
    }
  }

  TextureView::~TextureView() {
    if (m_imageView) vkDestroyImageView(m_device->get_vk_device(), 
      m_imageView, nullptr);
  }

  auto TextureView::operator=(TextureView&& view) -> TextureView& {
    m_imageView = view.m_imageView;
    m_descriptor = view.m_descriptor;
    m_texture = view.m_texture;
    m_device = view.m_device;
    view.m_imageView = nullptr;
    return *this;
  }

  auto TextureView::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
    objectNameInfo.objectHandle = uint64_t(m_imageView);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
  }


  Sampler::Sampler(SamplerDescriptor const& desc, Device* device)
    : m_device(device) {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = impl::getVkFilter(desc.magFilter);
    samplerInfo.minFilter = impl::getVkFilter(desc.minFilter);
    samplerInfo.addressModeU = impl::getVkSamplerAddressMode(desc.addressModeU);
    samplerInfo.addressModeV = impl::getVkSamplerAddressMode(desc.addressModeV);
    samplerInfo.addressModeW = impl::getVkSamplerAddressMode(desc.addressModeW);
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = device->from_which_adapter()
      ->get_vk_physical_device_properties()
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
    if (vkCreateSampler(device->get_vk_device(), &samplerInfo, nullptr,
      &m_textureSampler) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create texture sampler!");
    }
  }

  Sampler::~Sampler() {
    vkDestroySampler(m_device->get_vk_device(), m_textureSampler, nullptr);
  }

  auto Sampler::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_SAMPLER;
    objectNameInfo.objectHandle = uint64_t(m_textureSampler);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
    this->m_name = name;
  }

  SwapChain::~SwapChain() {
    if (m_swapChain)
      vkDestroySwapchainKHR(m_device->get_vk_device(), m_swapChain, nullptr);
  }

  auto SwapChain::init(Device* device) noexcept -> void {
    this->m_device = device;
    recreate();
  }

  auto SwapChain::recreate() noexcept -> void {
    m_device->wait_idle();
    // clean up swap chain
    m_swapChainTextures.clear();
    m_textureViews.clear();
    if (m_swapChain)
      vkDestroySwapchainKHR(m_device->get_vk_device(), m_swapChain, nullptr);
    // recreate swap chain
    impl::createSwapChain(m_device, this);
    // retrieving the swap chian image
    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(m_device->get_vk_device(), m_swapChain, &imageCount,
      nullptr);
    std::vector<VkImage> swapChainImages;
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device->get_vk_device(), m_swapChain, &imageCount,
      swapChainImages.data());
    // create Image views for images
    TextureDescriptor textureDesc;
    textureDesc.dimension = TextureDimension::TEX2D;
    textureDesc.format = impl::getTextureFormat(m_swapChainImageFormat);
    textureDesc.size = { m_swapChainExtend.width, m_swapChainExtend.height, 1 };
    textureDesc.usage = 0;
    TextureViewDescriptor viewDesc;
    viewDesc.format = impl::getTextureFormat(m_swapChainImageFormat);
    viewDesc.aspect = TextureAspectEnum::COLOR_BIT;
    for (size_t i = 0; i < swapChainImages.size(); i++)
      m_swapChainTextures.emplace_back(m_device, swapChainImages[i], textureDesc);
    for (size_t i = 0; i < swapChainImages.size(); i++)
      m_textureViews.emplace_back(m_device, &m_swapChainTextures[i], viewDesc);
  }

  FrameBuffer::FrameBuffer(Device* device,
    RenderPassDescriptor const& desc, RenderPass* renderpass)
    : m_device(device) {
    std::vector<VkImageView> attachments;
    for (int i = 0; i < desc.colorAttachments.size(); ++i) {
      attachments.push_back(
        static_cast<TextureView*>(desc.colorAttachments[i].view)->m_imageView);
      m_clearValues.push_back(VkClearValue{ VkClearColorValue{
          (float)desc.colorAttachments[i].clearValue.r,
          (float)desc.colorAttachments[i].clearValue.g,
          (float)desc.colorAttachments[i].clearValue.b,
          (float)desc.colorAttachments[i].clearValue.a,
      } });
    }
    if (desc.depthStencilAttachment.view != nullptr) {
      attachments.push_back(
        static_cast<TextureView*>(desc.depthStencilAttachment.view)
        ->m_imageView);
      VkClearValue clearValue = {};
      clearValue.color = { 0.f, 0.f, 0.f, 0.f };
      clearValue.depthStencil = {
          (float)desc.depthStencilAttachment.depthClearValue, (uint32_t)0 };
      m_clearValues.push_back(clearValue);
    }
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderpass->m_renderPass;
    framebufferInfo.attachmentCount = attachments.size();
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = (desc.colorAttachments.size() > 0)
      ? desc.colorAttachments[0].view->get_width()
      : desc.depthStencilAttachment.view->get_width();
    framebufferInfo.height = (desc.colorAttachments.size() > 0)
      ? desc.colorAttachments[0].view->get_height()
      : desc.depthStencilAttachment.view->get_height();
    m_width = framebufferInfo.width;
    m_height = framebufferInfo.height;
    framebufferInfo.layers = 1;
    if (vkCreateFramebuffer(device->get_vk_device(), &framebufferInfo, nullptr,
      &m_framebuffer) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create framebuffer!");
    }
  }

  FrameBuffer::~FrameBuffer() {
    if (m_framebuffer)
      vkDestroyFramebuffer(m_device->get_vk_device(), m_framebuffer, nullptr);
  }


  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃                                                                           ┃
  // ┃ Definitions for commands objects.                                         ┃
  // ┃                                                                           ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  CommandPool::CommandPool(Device* device): m_device(device) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex =
      m_device->from_which_adapter()->get_queue_family_indices().m_graphicsFamily.value();
    if (vkCreateCommandPool(m_device->get_vk_device(), &poolInfo, nullptr,
      &m_commandPool) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create command pool!");
    }
  }

  CommandPool::~CommandPool() {
    if (m_commandPool)
      vkDestroyCommandPool(m_device->get_vk_device(), m_commandPool, nullptr);
  }

  auto CommandPool::allocate_command_buffer() noexcept -> std::unique_ptr<CommandBuffer> {
    std::unique_ptr<CommandBuffer> command =
      std::make_unique<CommandBuffer>();
    command->m_device = m_device;
    command->m_commandPool = this;
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(m_device->get_vk_device(), &allocInfo,
      &command->m_commandBuffer) != VK_SUCCESS) {
      se::error("VULKAN :: failed to allocate command buffers!");
    }
    return command;
  }

  CommandBuffer::~CommandBuffer() {
    vkFreeCommandBuffers(m_device->get_vk_device(),
      m_commandPool->m_commandPool, 1, &m_commandBuffer);
  }

  auto CommandEncoder::begin_render_pass(RenderPassDescriptor const& desc) noexcept
    -> std::unique_ptr<RenderPassEncoder> {
    std::unique_ptr<RenderPassEncoder> renderpassEncoder =
      std::make_unique<RenderPassEncoder>();
    renderpassEncoder->m_renderPass =
      std::make_unique<RenderPass>(m_commandBuffer->m_device, desc);
    renderpassEncoder->m_commandBuffer = m_commandBuffer;
    renderpassEncoder->m_frameBuffer = std::make_unique<FrameBuffer>(
      m_commandBuffer->m_device, desc, renderpassEncoder->m_renderPass.get());
    // render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderpassEncoder->m_renderPass->m_renderPass;
    renderPassInfo.framebuffer = renderpassEncoder->m_frameBuffer->m_framebuffer;
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent =
      VkExtent2D{ renderpassEncoder->m_frameBuffer->width(),
                 renderpassEncoder->m_frameBuffer->height() };
    renderPassInfo.pClearValues =
      renderpassEncoder->m_frameBuffer->m_clearValues.data();
    renderPassInfo.clearValueCount =
      renderpassEncoder->m_frameBuffer->m_clearValues.size();
    vkCmdBeginRenderPass(m_commandBuffer->m_commandBuffer, &renderPassInfo,
      VK_SUBPASS_CONTENTS_INLINE);
    return renderpassEncoder;
  }

  auto CommandEncoder::begin_compute_pass() noexcept
    -> std::unique_ptr<ComputePassEncoder> {
    std::unique_ptr<ComputePassEncoder> computePassEncoder =
      std::make_unique<ComputePassEncoder>();
    computePassEncoder->m_commandBuffer = this->m_commandBuffer;
    return computePassEncoder;
  }

  auto CommandEncoder::pipeline_barrier(BarrierDescriptor const& desc) noexcept -> void {
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
      bmb.buffer = descriptor.buffer->get_vk_buffer();
      bmb.offset = descriptor.offset;
      bmb.size = (descriptor.size == uint64_t(-1))
        ? static_cast<Buffer*>(descriptor.buffer)->size()
        : descriptor.size;
      bmb.srcAccessMask = impl::getVkAccessFlags(descriptor.srcAccessMask);
      bmb.dstAccessMask = impl::getVkAccessFlags(descriptor.dstAccessMask);
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
      imb.oldLayout = impl::getVkImageLayout(descriptor.oldLayout);
      imb.newLayout = impl::getVkImageLayout(descriptor.newLayout);
      imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imb.srcAccessMask = impl::getVkAccessFlags(descriptor.srcAccessMask);
      imb.dstAccessMask = impl::getVkAccessFlags(descriptor.dstAccessMask);
      imb.image = descriptor.texture->get_vk_image();
      imb.subresourceRange.aspectMask =
        impl::getVkImageAspectFlags(descriptor.subresourceRange.aspectMask);
      imb.subresourceRange.baseMipLevel =
        descriptor.subresourceRange.baseMipLevel;
      imb.subresourceRange.levelCount = descriptor.subresourceRange.levelCount;
      imb.subresourceRange.baseArrayLayer =
        descriptor.subresourceRange.baseArrayLayer;
      imb.subresourceRange.layerCount = descriptor.subresourceRange.layerCount;
    }
    vkCmdPipelineBarrier(m_commandBuffer->m_commandBuffer,
      impl::getVkPipelineStageFlags(desc.srcStageMask),
      impl::getVkPipelineStageFlags(desc.dstStageMask),
      impl::getVkDependencyTypeFlags(desc.dependencyType),
      memoryBarriers.size(), memoryBarriers.data(),
      bufferBemoryBarriers.size(), bufferBemoryBarriers.data(),
      imageMemoryBarriers.size(), imageMemoryBarriers.data());
  }

  auto CommandEncoder::copy_buffer_to_buffer(Buffer* source, size_t sourceOffset,
    Buffer* destination, size_t destinationOffset, size_t size) noexcept -> void {
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = sourceOffset;
    copyRegion.dstOffset = destinationOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(m_commandBuffer->m_commandBuffer,
      source->get_vk_buffer(),
      destination->get_vk_buffer(), 1,
      &copyRegion);
  }

  auto CommandEncoder::clear_buffer(Buffer* buffer, size_t offset,
    size_t size) noexcept -> void {
    float const fillValueConst = 0;
    uint32_t const& fillValueU32 =
      reinterpret_cast<const uint32_t&>(fillValueConst);
    vkCmdFillBuffer(m_commandBuffer->m_commandBuffer,
      buffer->get_vk_buffer(), offset, size, fillValueU32);
  }

  auto CommandEncoder::clear_texture(Texture* texture,
    TextureClearDescriptor const& desc) noexcept -> void {
    std::vector<VkImageSubresourceRange> subresourceRanges;
    for (auto const& subresource : desc.subresources) {
      VkImageSubresourceRange subresourceRange = {};
      subresourceRange.aspectMask =
        impl::getVkImageAspectFlags(subresource.aspectMask);  // Assuming a color texture
      subresourceRange.baseMipLevel =
        subresource.baseMipLevel;       // Assuming mip level 0
      subresourceRange.levelCount =
        subresource.levelCount;         // Only clearing one level
      subresourceRange.baseArrayLayer =
        subresource.baseArrayLayer;     // Assuming layer 0
      subresourceRange.layerCount =
        subresource.layerCount;         // Only clearing one layer
      subresourceRanges.emplace_back(subresourceRange);
    }

    VkClearColorValue clearColor = {};  // Set the clear color value
    clearColor.float32[0] = desc.clearColor.r;       // Red component
    clearColor.float32[1] = desc.clearColor.g;       // Green component
    clearColor.float32[2] = desc.clearColor.b;       // Blue component
    clearColor.float32[3] = desc.clearColor.a;       // Alpha component

    vkCmdClearColorImage(m_commandBuffer->m_commandBuffer,
      texture->get_vk_image(),
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor,
      subresourceRanges.size(),
      subresourceRanges.data());
  }

  auto CommandEncoder::copy_buffer_to_texture(
    ImageCopyBuffer const& source,
    ImageCopyTexture const& destination,
    uvec3 const& copySize
  ) noexcept -> void {
    VkBufferImageCopy region{};
    region.bufferOffset = source.offset;
    region.bufferRowLength = source.bytesPerRow;
    region.bufferImageHeight = source.rowsPerImage;
    region.imageSubresource.aspectMask =
      impl::getVkImageAspectFlags(destination.aspect);
    region.imageSubresource.mipLevel = destination.mipLevel;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = copySize.z;
    region.imageOffset = { static_cast<int>(destination.origin.x),
                          static_cast<int>(destination.origin.y),
                          static_cast<int>(destination.origin.z) };
    region.imageExtent = { copySize.x, copySize.y, 1 };
    vkCmdCopyBufferToImage(
      m_commandBuffer->m_commandBuffer,
      source.buffer->get_vk_buffer(),
      destination.texutre->get_vk_image(),
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }

  auto CommandEncoder::copy_texture_to_texture(ImageCopyTexture const& source,
    ImageCopyTexture const& destination,
    uvec3 const& copySize) noexcept -> void {
    VkImageCopy region;
    // We copy the image aspect, layer 0, mip 0:
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount = 1;
    region.srcSubresource.mipLevel = source.mipLevel;
    // (0, 0, 0) in the first image corresponds to (0, 0, 0) in the second image:
    region.srcOffset = { int(source.origin.x), int(source.origin.y),
                        int(source.origin.z) };
    region.dstSubresource = region.srcSubresource;
    region.dstSubresource.mipLevel = destination.mipLevel;
    region.dstOffset = { int(destination.origin.x), int(destination.origin.y),
                        int(destination.origin.z) };
    // Copy the entire image:
    region.extent = { copySize.x, copySize.y, copySize.z };
    vkCmdCopyImage(
      m_commandBuffer->m_commandBuffer,       // Command buffer
      source.texutre->get_vk_image(),         // Source image
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,   // Source image layout
      destination.texutre->get_vk_image(),    // Destination image
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,   // Destination image layout
      1, &region);                            // Regions
  }

  auto CommandEncoder::reset_query_set(
    QuerySet* queryset_vk, uint32_t firstQuery,
    uint32_t queryCount) noexcept -> void {
    vkCmdResetQueryPool(m_commandBuffer->m_commandBuffer, queryset_vk->m_queryPool,
      firstQuery, queryCount);
  }

  auto CommandEncoder::write_timestamp(
    QuerySet* queryset_vk,
    PipelineStageEnum stageMask,
    uint32_t queryIndex) noexcept -> void {
    vkCmdWriteTimestamp(m_commandBuffer->m_commandBuffer,
      impl::getVkPipelineStageFlagBits(stageMask),
      queryset_vk->m_queryPool, queryIndex);
  }

  auto CommandEncoder::fill_buffer(Buffer* buffer, size_t offset, size_t size,
    float fillValue) noexcept -> void {
    float const fillValueConst = fillValue;
    uint32_t const& fillValueU32 =
      reinterpret_cast<const uint32_t&>(fillValueConst);
    vkCmdFillBuffer(m_commandBuffer->m_commandBuffer,
      buffer->get_vk_buffer(), offset, size,
      fillValueU32);
  }

  auto CommandEncoder::finish() noexcept -> CommandBuffer* {
    if (vkEndCommandBuffer(m_commandBuffer->m_commandBuffer) != VK_SUCCESS) {
      se::error("VULKAN :: failed to record command buffer!");
    }
    return m_commandBuffer;
  }

  auto CommandEncoder::begin_debug_utils_label(std::string const& name, se::vec4 color) noexcept -> void {
    if (!m_commandBuffer->m_device->m_debugLayerEnabled) return;
    VkDebugUtilsLabelEXT debugUtilLabel = {};
    debugUtilLabel.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    debugUtilLabel.pNext = nullptr;
    debugUtilLabel.pLabelName = name.c_str();
    memcpy(debugUtilLabel.color, &(color[0]), sizeof(float) * 4);
    m_commandBuffer->m_device->from_which_adapter()->from_which_context()
      ->vkCmdBeginDebugUtilsLabelEXT(m_commandBuffer->m_commandBuffer, &debugUtilLabel);
  }

  auto CommandEncoder::end_debug_utils_label() noexcept -> void {
    if (!m_commandBuffer->m_device->m_debugLayerEnabled) return;
    m_commandBuffer->m_device->from_which_adapter()->from_which_context()
      ->vkCmdEndDebugUtilsLabelEXT(m_commandBuffer->m_commandBuffer);
  }

  auto RenderPassEncoder::set_pipeline(RenderPipeline* vkpipeline) noexcept
    -> void {
    m_renderPipeline = vkpipeline;
    vkpipeline->combine_render_pass(m_renderPass.get());
    vkCmdBindPipeline(m_commandBuffer->m_commandBuffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS, vkpipeline->m_pipeline);
  }

  auto RenderPassEncoder::set_index_buffer(Buffer* buffer,
    IndexFormat indexFormat,
    uint64_t offset,
    uint64_t size) noexcept -> void {
    vkCmdBindIndexBuffer(m_commandBuffer->m_commandBuffer,
      buffer->get_vk_buffer(), offset,
      indexFormat == IndexFormat::UINT16_t
      ? VK_INDEX_TYPE_UINT16
      : VK_INDEX_TYPE_UINT32);
  }

  auto RenderPassEncoder::set_vertex_buffer(uint32_t slot, Buffer* buffer,
    uint64_t offset,
    uint64_t size) noexcept -> void {
    VkBuffer vertexBuffers[] = { buffer->get_vk_buffer() };
    VkDeviceSize offsets[] = { offset };
    vkCmdBindVertexBuffers(m_commandBuffer->m_commandBuffer, 0, 1, vertexBuffers,
      offsets);
  }

  auto RenderPassEncoder::draw(uint32_t vertexCount, uint32_t instanceCount,
    uint32_t firstVertex, uint32_t firstInstance) noexcept -> void {
    vkCmdDraw(m_commandBuffer->m_commandBuffer, vertexCount, instanceCount,
      firstVertex, firstInstance);
  }

  auto RenderPassEncoder::draw_indexed(uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance) noexcept -> void {
    vkCmdDrawIndexed(m_commandBuffer->m_commandBuffer, indexCount, instanceCount,
      firstIndex, baseVertex, firstInstance);
  }

  auto RenderPassEncoder::draw_indirect(Buffer* indirectBuffer, uint64_t indirectOffset,
    uint32_t drawCount, uint32_t stride) noexcept -> void {
    vkCmdDrawIndirect(m_commandBuffer->m_commandBuffer,
      indirectBuffer->get_vk_buffer(),
      indirectOffset, drawCount, stride);
  }

  auto RenderPassEncoder::draw_indexed_indirect(Buffer* indirectBuffer, uint64_t offset,
    uint32_t drawCount, uint32_t stride) noexcept -> void {
    vkCmdDrawIndexedIndirect(
      m_commandBuffer->m_commandBuffer,
      indirectBuffer->get_vk_buffer(), offset, drawCount,
      stride);
  }

  auto RenderPassEncoder::set_viewport(float x, float y, float width,
    float height, float minDepth,
    float maxDepth) noexcept -> void {
    VkViewport viewport = {};
    viewport.x = x;
    viewport.y = y;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = minDepth;
    viewport.maxDepth = maxDepth;
    vkCmdSetViewport(m_commandBuffer->m_commandBuffer, 0, 1, &viewport);
  }

  auto RenderPassEncoder::set_scissor_rect(uint32_t x, uint32_t y,
    uint32_t width, uint32_t height) noexcept -> void {
    VkRect2D scissor;
    scissor.offset.x = x;
    scissor.offset.y = y;
    scissor.extent.width = width;
    scissor.extent.height = height;
    vkCmdSetScissor(m_commandBuffer->m_commandBuffer, 0, 1, &scissor);
  }

  auto RenderPassEncoder::end() noexcept -> void {
    vkCmdEndRenderPass(m_commandBuffer->m_commandBuffer);
  }

  auto RenderPassEncoder::set_bindgroup(uint32_t index, BindGroup* bindgroup,
    std::vector<uint32_t> const& dynamicOffsets) noexcept -> void {
    vkCmdBindDescriptorSets(
      m_commandBuffer->m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_renderPipeline->m_fixedFunctionSetttings.pipelineLayout->m_pipelineLayout,
      index, 1, &bindgroup->m_set, 0, nullptr);
  }

  auto RenderPassEncoder::set_bindgroup(uint32_t index, BindGroup* bindgroup,
    uint64_t dynamicOffsetDataStart,
    uint32_t dynamicOffsetDataLength) noexcept -> void {
    vkCmdBindDescriptorSets(
      m_commandBuffer->m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_renderPipeline->m_fixedFunctionSetttings.pipelineLayout->m_pipelineLayout,
      index, 1, &bindgroup->m_set, 0, nullptr);
  }

  auto RenderPassEncoder::push_constants(void* data, Flags<ShaderStageEnum> stages,
    uint32_t offset, uint32_t size) noexcept -> void {

    vkCmdPushConstants(m_commandBuffer->m_commandBuffer,
      m_renderPipeline->m_fixedFunctionSetttings.pipelineLayout->m_pipelineLayout,
      impl::getVkShaderStageFlags(stages), offset, size, data);
  }

  auto ComputePassEncoder::set_bindgroup(uint32_t index, BindGroup* bindgroup,
    std::vector<uint32_t> const& dynamicOffsets) noexcept -> void {
    vkCmdBindDescriptorSets(
      m_commandBuffer->m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      m_computePipeline->m_layout->m_pipelineLayout,
      index, 1, &bindgroup->m_set, 0, nullptr);
  }

  auto ComputePassEncoder::set_bindgroup(uint32_t index, BindGroup* bindgroup,
    uint64_t dynamicOffsetDataStart,
    uint32_t dynamicOffsetDataLength) noexcept -> void {
    vkCmdBindDescriptorSets(
      m_commandBuffer->m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      m_computePipeline->m_layout->m_pipelineLayout,
      index, 1, &bindgroup->m_set, 0, nullptr);
  }

  auto ComputePassEncoder::push_constants(void* data, Flags<ShaderStageEnum> stages,
    uint32_t offset, uint32_t size) noexcept -> void {
    vkCmdPushConstants(
      m_commandBuffer->m_commandBuffer,
      m_computePipeline->m_layout->m_pipelineLayout,
      impl::getVkShaderStageFlags(stages), offset, size, data);
  }

  auto ComputePassEncoder::set_pipeline(ComputePipeline* vkpipeline) noexcept -> void {
    m_computePipeline = vkpipeline;
    vkCmdBindPipeline(m_commandBuffer->m_commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE, vkpipeline->m_pipeline);
  }

  auto ComputePassEncoder::dispatch_workgroups(uint32_t workgroupCountX,
    uint32_t workgroupCountY,
    uint32_t workgroupCountZ) noexcept -> void {
    vkCmdDispatch(m_commandBuffer->m_commandBuffer, workgroupCountX, workgroupCountY,
      workgroupCountZ);
  }

  auto ComputePassEncoder::dispatch_workgroups_indirect(Buffer* indirectBuffer,
    uint64_t indirectOffset) noexcept -> void {
    vkCmdDispatchIndirect(m_commandBuffer->m_commandBuffer,
      indirectBuffer->get_vk_buffer(),
      indirectOffset);
  }

  auto ComputePassEncoder::end() noexcept -> void {
    m_computePipeline = nullptr;
  }

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃                                                                           ┃
  // ┃ Definitions for commands objects.                                         ┃
  // ┃                                                                           ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //
  FrameResources::FrameResources(Device* device, int maxFlightNum,
    SwapChain* swapchain)
    : m_device(device),
    m_maxFlightNum(maxFlightNum),
    m_swapChain(static_cast<SwapChain*>(swapchain)) {
    m_commandBuffers.resize(maxFlightNum);
    for (size_t i = 0; i < maxFlightNum; ++i) {
      m_commandBuffers[i] = device->allocate_command_buffer();
    }
    // createSyncObjects
    m_imageAvailableSemaphores.resize(maxFlightNum);
    m_renderFinishedSemaphores.resize(maxFlightNum);
    m_inFlightFences.resize(maxFlightNum);
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (size_t i = 0; i < maxFlightNum; ++i) {
      if (vkCreateSemaphore(device->get_vk_device(), &semaphoreInfo, nullptr,
        &m_imageAvailableSemaphores[i].m_semaphore) !=
        VK_SUCCESS ||
        vkCreateSemaphore(device->get_vk_device(), &semaphoreInfo, nullptr,
          &m_renderFinishedSemaphores[i].m_semaphore) !=
        VK_SUCCESS ||
        vkCreateFence(device->get_vk_device(), &fenceInfo, nullptr,
          &m_inFlightFences[i].m_fence) != VK_SUCCESS) {
        se::error("VULKAN :: failed to create synchronization objects for a frame!");
      }
      else {
        m_imageAvailableSemaphores[i].m_device = device;
        m_renderFinishedSemaphores[i].m_device = device;
        m_inFlightFences[i].m_device = device;
      }
    }
  }

  auto FrameResources::frame_start() noexcept -> void {
    VkResult result = vkWaitForFences(m_device->get_vk_device(), 1,
      &m_inFlightFences[m_currentFrame].m_fence,
      VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
      se::error("Vulkan::MultiFrameFlight::frameStart()::WaitForFenceFailed!");
    }
    vkResetFences(m_device->get_vk_device(), 1, &m_inFlightFences[m_currentFrame].m_fence);
    if (m_swapChain)
      vkAcquireNextImageKHR(m_device->get_vk_device(), m_swapChain->m_swapChain,
        UINT64_MAX,
        m_imageAvailableSemaphores[m_currentFrame].m_semaphore,
        VK_NULL_HANDLE, &m_imageIndex);
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame]->m_commandBuffer, 0);
  }

  auto FrameResources::frame_end() noexcept -> void {
    if (m_swapChain)
      m_device->get_present_queue().present_swapchain(
        m_swapChain, m_imageIndex, &m_renderFinishedSemaphores[m_currentFrame]);
    m_currentFrame = (m_currentFrame + 1) % m_maxFlightNum;
  }

  auto FrameResources::get_command_buffer() noexcept -> CommandBuffer* {
    return m_commandBuffers[m_currentFrame].get();
  }

  auto FrameResources::get_image_available_semaphore() noexcept -> Semaphore* {
    return m_swapChain ? &m_imageAvailableSemaphores[m_currentFrame] : nullptr;
  }
  /** get current Render Finished Semaphore */
  auto FrameResources::get_render_finished_semaphore() noexcept -> Semaphore* {
    return &m_renderFinishedSemaphores[m_currentFrame];
  }

  auto FrameResources::get_fence() noexcept -> Fence* {
    return &m_inFlightFences[m_currentFrame];
  }

  auto FrameResources::reset() noexcept -> void {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < m_maxFlightNum; ++i) {
    //   vkResetCommandBuffer(m_commandBuffers[i]->m_commandBuffer, 0);
    //   vkResetFences(m_device->get_vk_device(), 1,
    //     &m_inFlightFences[i].m_fence);

    //   // reset semaphores
    //   vkDestroySemaphore(m_device->get_vk_device(), m_imageAvailableSemaphores[i].m_semaphore, nullptr);
    //   vkDestroySemaphore(m_device->get_vk_device(), m_renderFinishedSemaphores[i].m_semaphore, nullptr);


    //   if (vkCreateSemaphore(m_device->get_vk_device(), &semaphoreInfo, nullptr,
    //     &m_imageAvailableSemaphores[i].m_semaphore) !=
    //     VK_SUCCESS ||
    //     vkCreateSemaphore(m_device->get_vk_device(), &semaphoreInfo, nullptr,
    //       &m_renderFinishedSemaphores[i].m_semaphore) !=
    //     VK_SUCCESS) {
    //     se::error("VULKAN :: failed to create synchronization objects for a frame!");
    //   }
    //   else {
    //     m_imageAvailableSemaphores[i].m_device = m_device;
    //     m_renderFinishedSemaphores[i].m_device = m_device;
    //   }
    }
  }

  ShaderModule::ShaderModule(Device* device,
    ShaderModuleDescriptor const& desc)
    : m_device(device), m_stages((uint32_t)desc.stage), m_entryPoint(desc.name) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = desc.code->m_size;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(desc.code->m_data);
    if (vkCreateShaderModule(device->get_vk_device(), &createInfo, nullptr,
      &m_shaderModule) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create shader module!");
    }
    // create info
    m_shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    m_shaderStageInfo.stage = impl::getVkShaderStageFlagBits(desc.stage);
    m_shaderStageInfo.module = m_shaderModule;
    m_shaderStageInfo.pName = m_entryPoint.c_str();
  }

  ShaderModule::ShaderModule(ShaderModule&& shader)
    : m_device(shader.m_device),
    m_stages(shader.m_stages),
    m_shaderModule(shader.m_shaderModule),
    m_shaderStageInfo(shader.m_shaderStageInfo) {
    shader.m_shaderModule = nullptr;
  }

  auto ShaderModule::operator=(ShaderModule&& shader) -> ShaderModule& {
    m_device = shader.m_device;
    m_stages = shader.m_stages;
    m_shaderModule = shader.m_shaderModule;
    m_shaderStageInfo = shader.m_shaderStageInfo;
    shader.m_shaderModule = nullptr;
    return *this;
  }

  ShaderModule::~ShaderModule() {
    if (m_shaderModule)
      vkDestroyShaderModule(m_device->get_vk_device(), m_shaderModule, nullptr);
  }

  auto ShaderModule::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
    objectNameInfo.objectHandle = uint64_t(m_shaderModule);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
    this->m_name = name;
  }

  auto ShaderModule::get_name() -> std::string const& { return m_name; }

  auto getVertexFormat(BLASTriangleGeometry::VertexFormat format) -> VkFormat {
    switch (format) {
    case se::rhi::BLASTriangleGeometry::VertexFormat::RGB32: return VK_FORMAT_R32G32B32_SFLOAT;
    case se::rhi::BLASTriangleGeometry::VertexFormat::RG32: return VK_FORMAT_R32G32_SFLOAT;
    default: return VK_FORMAT_R32G32B32_SFLOAT;
    }
  }

  auto getBufferVkDeviceAddress(Device* device, Buffer* buffer) noexcept -> VkDeviceAddress {
    VkBufferDeviceAddressInfo deviceAddressInfo = {};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    deviceAddressInfo.buffer = buffer->get_vk_buffer();
    return vkGetBufferDeviceAddress(device->get_vk_device(), &deviceAddressInfo);
  }

  auto getVkGeometryFlagsKHR(Flags<BLASGeometryEnum> input) noexcept -> VkGeometryFlagsKHR {
    VkGeometryFlagsKHR flag = 0;
    if (input & BLASGeometryEnum::OPAQUE_GEOMETRY)
      flag |= VK_GEOMETRY_OPAQUE_BIT_KHR;
    if (input & BLASGeometryEnum::NO_DUPLICATE_ANY_HIT_INVOCATION)
      flag |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    return flag;
  }

  BLAS::BLAS(Device* device, BLASDescriptor const& descriptor)
    : m_device(device), m_descriptor(descriptor) {
    // 1. Declare BLAS geometry infos
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
    std::vector<uint32_t> primitiveCountArray;
    // transform buffer
    std::vector<AffineTransformMatrix> affine_transforms;
    for (BLASTriangleGeometry const& triangleDesc : descriptor.triangleGeometries)
      affine_transforms.push_back(
        AffineTransformMatrix(triangleDesc.transform));
    for (BLASCustomGeometry const& customDesc : descriptor.customGeometries)
      affine_transforms.push_back(
        AffineTransformMatrix(customDesc.transform));
    std::unique_ptr<Buffer> transformBuffer =
      device->create_device_local_buffer(
        affine_transforms.data(),
        affine_transforms.size() * sizeof(AffineTransformMatrix),
        BufferUsageEnum::STORAGE |
        BufferUsageEnum::SHADER_DEVICE_ADDRESS |
        BufferUsageEnum::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);

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
      triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      triangles.vertexFormat = getVertexFormat(triangleDesc.vertexFormat);
      triangles.vertexData.deviceAddress = vertexBufferAddress + triangleDesc.vertexByteOffset;
      triangles.vertexStride = triangleDesc.vertexStride;
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
      transformOffset += sizeof(AffineTransformMatrix);
    }
    std::vector<std::unique_ptr<Buffer>> aabbBuffers;
    for (BLASCustomGeometry const& customDesc : descriptor.customGeometries) {
      std::unique_ptr<Buffer> aabbBuffer = device->create_device_local_buffer(
        (void*)customDesc.aabbs.data(),
        customDesc.aabbs.size() * sizeof(se::bounds3),
        BufferUsageEnum::STORAGE |
        BufferUsageEnum::SHADER_DEVICE_ADDRESS |
        BufferUsageEnum::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);
      aabbBuffers.emplace_back(std::move(aabbBuffer));
      VkDeviceAddress dataAddress =
        getBufferVkDeviceAddress(device, aabbBuffers.back().get());
      VkAccelerationStructureGeometryAabbsDataKHR aabbs{
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR };
      aabbs.data.deviceAddress = dataAddress;
      aabbs.stride = sizeof(se::bounds3);
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
      transformOffset += sizeof(AffineTransformMatrix);
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
    device->from_which_adapter()->from_which_context()->vkGetAccelerationStructureBuildSizesKHR(
      device->get_vk_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo,                  // Pointer to build info
      primitiveCountArray.data(),  // Array of number of primitives per geometry
      &sizeInfo);                  // Output pointer to store sizes
    // 3. Create an empty acceleration structure and its underlying VkBuffer.
    m_bufferBLAS = device->create_buffer(BufferDescriptor{
      sizeInfo.accelerationStructureSize,
      BufferUsageEnum::ACCELERATION_STRUCTURE_STORAGE |
      BufferUsageEnum::SHADER_DEVICE_ADDRESS |
      BufferUsageEnum::STORAGE,
      BufferShareMode::EXCLUSIVE, MemoryPropertyEnum::DEVICE_LOCAL_BIT });
    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.type = buildInfo.type;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.buffer = m_bufferBLAS->get_vk_buffer();
    createInfo.offset = 0;
    /*VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR*/
    device->from_which_adapter()->from_which_context()->vkCreateAccelerationStructureKHR(
      device->get_vk_device(), &createInfo, nullptr, &m_blas);
    buildInfo.dstAccelerationStructure = m_blas;
    // 5. Allocate scratch space.
    uint32_t minOffsetAlignment =
      device->m_vASProperties.minAccelerationStructureScratchOffsetAlignment;

    std::unique_ptr<Buffer> scratchBuffer = device->create_buffer(BufferDescriptor{
      sizeInfo.buildScratchSize,
      BufferUsageEnum::SHADER_DEVICE_ADDRESS |
      BufferUsageEnum::STORAGE,
      BufferShareMode::EXCLUSIVE, MemoryPropertyEnum::DEVICE_LOCAL_BIT,
      false, int(minOffsetAlignment) });
    buildInfo.scratchData.deviceAddress =
      getBufferVkDeviceAddress(device, scratchBuffer.get());
    // 6. Call vkCmdBuildAccelerationStructuresKHR() with a populated
    //	  VkAccelerationStructureBuildSizesInfoKHR structand range info to
    //    build the geometry into an acceleration structure.
    VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = rangeInfos.data();
    std::unique_ptr<CommandEncoder> commandEncoder =
      device->create_command_encoder({ nullptr });
    CommandEncoder* commandEncoderVK = commandEncoder.get();
    device->from_which_adapter()->from_which_context()->vkCmdBuildAccelerationStructuresKHR(
      commandEncoderVK->m_commandBuffer
      ->m_commandBuffer,  // The command buffer to record the command
      1,                    // Number of acceleration structures to build
      &buildInfo,           // Array of ...BuildGeometryInfoKHR objects
      &pRangeInfo);         // Array of ...RangeInfoKHR objects
    device->get_graphics_queue().submit({ commandEncoder->finish() });
    device->wait_idle();
  }

  BLAS::BLAS(Device* device, BLAS* src)
    : m_device(device), m_descriptor(src->m_descriptor) {}

  BLAS::~BLAS() {
    if (m_blas)
      m_device->from_which_adapter()->from_which_context()->vkDestroyAccelerationStructureKHR(
        m_device->get_vk_device(), m_blas, nullptr);
  }

  TLAS::TLAS(Device* device, TLASDescriptor const& descriptor)
    : m_device(device) {
    // specify an instance
    //  Zero-initialize.
    std::vector<VkAccelerationStructureInstanceKHR> instances(
      descriptor.instances.size());
    for (int i = 0; i < instances.size(); ++i) {
      // frst get the device address of one or more BLASs
      VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
      addressInfo.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addressInfo.accelerationStructure = descriptor.instances[i].blas->m_blas;
      VkDeviceAddress blasAddress =
        device->from_which_adapter()
        ->from_which_context()
        ->vkGetAccelerationStructureDeviceAddressKHR(device->get_vk_device(),
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
      bufferInstances = device->create_device_local_buffer(
        (void*)instances.data(),
        sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
        BufferUsageEnum::SHADER_DEVICE_ADDRESS |
        BufferUsageEnum::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);
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
    device->from_which_adapter()->from_which_context()->vkGetAccelerationStructureBuildSizesKHR(
      device->get_vk_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo, &rangeInfo.primitiveCount, &sizeInfo);
    // Allocate a buffer for the acceleration structure.
    m_bufferTLAS = device->create_buffer(BufferDescriptor{
      sizeInfo.accelerationStructureSize,
      BufferUsageEnum::ACCELERATION_STRUCTURE_STORAGE |
      BufferUsageEnum::SHADER_DEVICE_ADDRESS |
      BufferUsageEnum::STORAGE,
      BufferShareMode::EXCLUSIVE, MemoryPropertyEnum::DEVICE_LOCAL_BIT });
    // Create the acceleration structure object.
    // (Data has not yet been set.)
    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.type = buildInfo.type;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.buffer = m_bufferTLAS.get()->get_vk_buffer();
    createInfo.offset = 0;
    device->from_which_adapter()->from_which_context()->vkCreateAccelerationStructureKHR(
      device->get_vk_device(), &createInfo, nullptr, &m_tlas);
    buildInfo.dstAccelerationStructure = m_tlas;
    // Allocate the scratch buffer holding temporary build data.
    uint32_t minOffsetAlignment =
      device->m_vASProperties.minAccelerationStructureScratchOffsetAlignment;

    std::unique_ptr<Buffer> scratchBuffer = device->create_buffer(BufferDescriptor{
      sizeInfo.buildScratchSize,
      BufferUsageEnum::SHADER_DEVICE_ADDRESS |
      BufferUsageEnum::STORAGE,
      BufferShareMode::EXCLUSIVE, MemoryPropertyEnum::DEVICE_LOCAL_BIT,
      false, int(minOffsetAlignment) });
    buildInfo.scratchData.deviceAddress =
      getBufferVkDeviceAddress(device, scratchBuffer.get());

    // Create a one-element array of pointers to range info objects.
    VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;
    // Build the TLAS.
    std::unique_ptr<CommandEncoder> commandEncoder =
      device->create_command_encoder({ nullptr });
    CommandEncoder* commandEncoderVK = commandEncoder.get();
    device->from_which_adapter()->from_which_context()->vkCmdBuildAccelerationStructuresKHR(
      commandEncoderVK->m_commandBuffer
      ->m_commandBuffer,  // The command buffer to record the command
      1,                    // Number of acceleration structures to build
      &buildInfo,           // Array of ...BuildGeometryInfoKHR objects
      &pRangeInfo);         // Array of ...RangeInfoKHR objects
    device->get_graphics_queue().submit({ commandEncoder->finish() });
    device->get_graphics_queue().wait_idle();
  }

  TLAS::~TLAS() {
    if (m_tlas) m_device->from_which_adapter()->from_which_context()->vkDestroyAccelerationStructureKHR(
      m_device->get_vk_device(), m_tlas, nullptr);
  }

  BindGroupLayout::BindGroupLayout(Device* device,
    BindGroupLayoutDescriptor const& desc)
    : m_device(device), m_descriptor(desc) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(desc.entries.size());
    std::vector<VkDescriptorBindingFlags> bindingFlags(desc.entries.size());
    for (int i = 0; i < desc.entries.size(); i++) {
      bindings[i].binding = desc.entries[i].binding;
      bindings[i].descriptorType = impl::getVkDecriptorType(desc.entries[i]);
      bindings[i].descriptorCount =
        bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        ? 200
        : desc.entries[i].array_size;
      bindings[i].stageFlags = impl::getVkShaderStageFlags(desc.entries[i].visibility);
      bindings[i].pImmutableSamplers = nullptr;
      bindingFlags[i] = 0;
      if (bindings[i].descriptorCount > 10)
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
    if (vkCreateDescriptorSetLayout(device->get_vk_device(), &layoutInfo, nullptr,
      &m_layout) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create descriptor set layout!");
    }
  }

  BindGroupLayout::~BindGroupLayout() {
    if (m_layout)
      vkDestroyDescriptorSetLayout(m_device->get_vk_device(), m_layout, nullptr);
  }

  auto BindGroupLayout::get_bindgroup_layout_descriptor() const noexcept
    -> BindGroupLayoutDescriptor const& {
    return m_descriptor;
  }

  BindGroupPool::BindGroupPool(Device* device) : m_device(device) {
    std::vector<VkDescriptorPoolSize> poolSizes(6);
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
    poolSizes[5].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[5].descriptorCount = static_cast<uint32_t>(99);
    if (device->from_which_adapter()->from_which_context()->get_context_extensions_flags()
      & ContextExtensionEnum::RAY_TRACING) {
      poolSizes.push_back({});
      poolSizes[6].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      poolSizes[6].descriptorCount = static_cast<uint32_t>(99);
    }
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(999);
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;
    if (vkCreateDescriptorPool(device->get_vk_device(), &poolInfo, nullptr,
      &m_descriptorPool) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create descriptor pool!");
    }
  }

  BindGroupPool::~BindGroupPool() {
    if (m_descriptorPool)
      vkDestroyDescriptorPool(m_device->get_vk_device(), m_descriptorPool, nullptr);
  }

  BindGroup::BindGroup(Device* device, BindGroupDescriptor const& desc) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = device->get_bindgroup_pool()->m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &desc.layout->m_layout;

    bool hasBindless = false;
    for (auto& entry : desc.entries) {
      if (entry.resource.bindlessTextures.size() > 0) hasBindless = true;
    }
    VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT };
    uint32_t max_binding = 200 - 1;
    if (hasBindless) {
      count_info.descriptorSetCount = 1;
      count_info.pDescriptorCounts = &max_binding;
      allocInfo.pNext = &count_info;
    }

    if (vkAllocateDescriptorSets(device->get_vk_device(), &allocInfo, &m_set) !=
      VK_SUCCESS) {
      se::error("VULKAN :: failed to allocate descriptor sets!");
    }

    m_layout = desc.layout;
    this->m_device = device;
    update_binding(desc.entries);
  }

  auto BindGroup::update_binding(
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
      else if (entry.resource.storageArray.size() > 0)
        imageCounts += entry.resource.storageArray.size();
      else if (entry.resource.tlas)
        ++accStructCounts;
      else if (entry.resource.sampler)
        ++imageCounts;
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
    std::vector<BindGroupLayoutEntry> const& layout_entries =
      m_layout->get_bindgroup_layout_descriptor().entries;
    std::function<std::optional<VkDescriptorType>(uint32_t)> getType =
      [&layout_entries](uint32_t binding) -> std::optional<VkDescriptorType> {
      for (auto& iter : layout_entries) {
        if (iter.binding == binding) return impl::getVkDecriptorType(iter);
      }
      return std::nullopt;
    };
    for (auto& entry : entries) {
      if (entry.resource.bufferBinding.has_value()) {
        VkDescriptorBufferInfo& bufferInfo = bufferInfos[bufferIndex++];
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        bufferInfo.buffer =
          static_cast<Buffer*>(entry.resource.bufferBinding.value().buffer)
          ->get_vk_buffer();
        bufferInfo.offset = entry.resource.bufferBinding.value().offset;
        bufferInfo.range = entry.resource.bufferBinding.value().size;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = m_set;
        descriptorWrite.dstBinding = entry.binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = type.value();
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;
      }
      else if (entry.resource.sampler && entry.resource.textureView) {
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        VkDescriptorImageInfo& imageInfo = imageInfos[imageIndex++];
        imageInfo.imageView =
          static_cast<TextureView*>(entry.resource.textureView)->m_imageView;
        imageInfo.sampler =
          static_cast<Sampler*>(entry.resource.sampler)->m_textureSampler;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = m_set;
        descriptorWrite.dstBinding = entry.binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = type.value();
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = nullptr;
        descriptorWrite.pImageInfo = &imageInfo;
        descriptorWrite.pTexelBufferView = nullptr;
      }
      else if (entry.resource.textureView) {
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        VkDescriptorImageInfo& imageInfo = imageInfos[imageIndex++];
        imageInfo.sampler = {};
        imageInfo.imageView =
          static_cast<TextureView*>(entry.resource.textureView)->m_imageView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = m_set;
        descriptorWrite.dstBinding = entry.binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = type.value();
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = nullptr;
        descriptorWrite.pImageInfo = &imageInfo;
        descriptorWrite.pTexelBufferView = nullptr;
      }
      else if (entry.resource.storageArray.size() > 0) {
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        bindlessImageInfos.push_back(std::vector<VkDescriptorImageInfo>(
          entry.resource.storageArray.size()));
        std::vector<VkDescriptorImageInfo>& bindelessImageInfo =
          bindlessImageInfos.back();
        for (int i = 0; i < entry.resource.storageArray.size(); ++i) {
          auto bindlessTexture = entry.resource.storageArray[i];
          VkDescriptorImageInfo& imageInfo = bindelessImageInfo[i];
          imageInfo.sampler = {};
          imageInfo.imageView = static_cast<TextureView*>(bindlessTexture)->m_imageView;
          imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

          descriptorWrites.push_back(VkWriteDescriptorSet{});
          VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
          descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          descriptorWrite.dstSet = m_set;
          descriptorWrite.dstBinding = entry.binding;
          descriptorWrite.dstArrayElement = i;
          descriptorWrite.descriptorType = type.value();
          descriptorWrite.descriptorCount = 1;
          descriptorWrite.pBufferInfo = nullptr;
          descriptorWrite.pImageInfo = &imageInfo;
          descriptorWrite.pTexelBufferView = nullptr;
        }
      }
      else if (entry.resource.tlas) {
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        VkWriteDescriptorSetAccelerationStructureKHR& descASInfo =
          accelerationStructureInfos[accStructIndex++];
        descASInfo.sType =
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        descASInfo.accelerationStructureCount = 1;
        descASInfo.pAccelerationStructures =
          &static_cast<TLAS*>(entry.resource.tlas)->m_tlas;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = m_set;
        descriptorWrite.dstBinding = entry.binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = type.value();
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = nullptr;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;
        descriptorWrite.pNext = &descASInfo;
      }
      else if (entry.resource.bindlessTextures.size() != 0) {
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        bindlessImageInfos.push_back(std::vector<VkDescriptorImageInfo>(
          entry.resource.bindlessTextures.size()));
        std::vector<VkDescriptorImageInfo>& bindelessImageInfo =
          bindlessImageInfos.back();

        auto get_sampler = [&](int index) {
          if (entry.resource.samplers.size() > 0) {
            return static_cast<Sampler*>(entry.resource.samplers[index])->m_textureSampler;
          }
          else {
            return static_cast<Sampler*>(entry.resource.sampler)->m_textureSampler;
          }
        };

        for (int i = 0; i < entry.resource.bindlessTextures.size(); ++i) {
          auto bindlessTexture = entry.resource.bindlessTextures[i];
          VkDescriptorImageInfo& imageInfo = bindelessImageInfo[i];
          imageInfo.sampler = get_sampler(i);
          imageInfo.imageView = static_cast<TextureView*>(bindlessTexture)->m_imageView;
          imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

          descriptorWrites.push_back(VkWriteDescriptorSet{});
          VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
          descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          descriptorWrite.dstSet = m_set;
          descriptorWrite.dstBinding = entry.binding;
          descriptorWrite.dstArrayElement = i;
          descriptorWrite.descriptorType = type.value();
          descriptorWrite.descriptorCount = 1;
          descriptorWrite.pBufferInfo = nullptr;
          descriptorWrite.pImageInfo = &imageInfo;
          descriptorWrite.pTexelBufferView = nullptr;
        }
      }
      else if (entry.resource.sampler) {
        // bind a sampler state
        std::optional<VkDescriptorType> type = getType(entry.binding);
        if (!type.has_value()) continue;
        VkDescriptorImageInfo& imageInfo = imageInfos[imageIndex++];
        imageInfo.sampler =
          static_cast<Sampler*>(entry.resource.sampler)->m_textureSampler;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        VkWriteDescriptorSet& descriptorWrite = descriptorWrites.back();
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = m_set;
        descriptorWrite.dstBinding = entry.binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = type.value();
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = nullptr;
        descriptorWrite.pImageInfo = &imageInfo;
        descriptorWrite.pTexelBufferView = nullptr;
      }
    }
    vkUpdateDescriptorSets(m_device->get_vk_device(), descriptorWrites.size(),
      descriptorWrites.data(), 0, nullptr);
  }

  PipelineLayout::PipelineLayout(Device* device,
    PipelineLayoutDescriptor const& desc)
    : m_device(device) {
    // push constants
    for (auto& ps : desc.pushConstants) {
      m_pushConstants.push_back(VkPushConstantRange{
          impl::getVkShaderStageFlags(ps.shaderStages), ps.offset, ps.size });
    }
    // descriptor set layouts
    std::vector<VkDescriptorSetLayout> descriptorSets;
    for (auto& bindgroupLayout : desc.bindGroupLayouts) {
      descriptorSets.push_back(
        static_cast<BindGroupLayout*>(bindgroupLayout)->m_layout);
    }
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = descriptorSets.size();
    pipelineLayoutInfo.pSetLayouts = descriptorSets.data();
    pipelineLayoutInfo.pushConstantRangeCount = m_pushConstants.size();
    pipelineLayoutInfo.pPushConstantRanges = m_pushConstants.data();
    if (vkCreatePipelineLayout(device->get_vk_device(), &pipelineLayoutInfo,
      nullptr, &m_pipelineLayout) != VK_SUCCESS) {
      se::error("failed to create pipeline layout!");
    }
  }

  PipelineLayout::~PipelineLayout() {
    if (m_pipelineLayout)
      vkDestroyPipelineLayout(m_device->get_vk_device(), m_pipelineLayout, nullptr);
  }

  PipelineLayout::PipelineLayout(PipelineLayout&& layout)
    : m_pipelineLayout(layout.m_pipelineLayout),
    m_pushConstants(layout.m_pushConstants),
    m_device(layout.m_device) {
    layout.m_pipelineLayout = nullptr;
  }

  auto PipelineLayout::operator=(PipelineLayout&& layout)
    -> PipelineLayout& {
    m_pipelineLayout = layout.m_pipelineLayout;
    m_pushConstants = layout.m_pushConstants;
    m_device = layout.m_device;
    layout.m_pipelineLayout = nullptr;
    return *this;
  }

  QuerySet::QuerySet(Device* device, QuerySetDescriptor const& desc)
    : m_device(device) {
    m_type = desc.type;
    m_count = desc.count;
    VkQueryPoolCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = impl::getVkQueryType(m_type);
    info.queryCount = m_count;
    info.flags = 0;
    info.pNext = nullptr;
    if (vkCreateQueryPool(device->get_vk_device(), &info,
      nullptr, &m_queryPool) != VK_SUCCESS) {
      se::error("RHI :: Vulkan :: Create query set failed!");
    }
  }

  QuerySet::~QuerySet() {
    vkDestroyQueryPool(m_device->get_vk_device(), m_queryPool, nullptr);
  }

  auto QuerySet::resolve_query_result(uint32_t firstQuery, uint32_t queryCount,
    size_t dataSize, void* pData, uint64_t stride,
    Flags<QueryResultEnum> flag) noexcept -> void {
    vkGetQueryPoolResults(m_device->get_vk_device(), m_queryPool, firstQuery,
      queryCount, dataSize, pData, stride,
      impl::getVkQueryResultFlags(flag));
  }

  Fence::Fence(Device* device) : m_device(device) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device->get_vk_device(), &fenceInfo, nullptr, &m_fence) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create fence");
    }
  }

  Fence::~Fence() {
    if (m_fence) vkDestroyFence(m_device->get_vk_device(), m_fence, nullptr);
  }

  auto Fence::wait() noexcept -> void {
    vkWaitForFences(m_device->get_vk_device(), 1, &m_fence, VK_TRUE, UINT64_MAX);
  }

  auto Fence::reset() noexcept -> void {
    vkResetFences(m_device->get_vk_device(), 1, &m_fence);
  }

  Semaphore::Semaphore(Device* device) : m_device(device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(device->get_vk_device(), &semaphoreInfo, nullptr,
      &m_semaphore) != VK_SUCCESS)
      se::error("VULKAN :: failed to create semaphores!");
  }

  Semaphore::~Semaphore() {
    if (m_semaphore) vkDestroySemaphore(m_device->get_vk_device(), m_semaphore, nullptr);
  }

  auto Semaphore::current_host() noexcept -> size_t {
    return m_currentValue;
  }

  auto Semaphore::current_device() noexcept -> size_t {
    if (!m_timelineSemaphore) {
      se::error("rhi :: Current semaphore is not a timeline one, query from host is not allowed.");
      return 0;
    }
    uint64_t value;
    vkGetSemaphoreCounterValue(m_device->get_vk_device(), m_semaphore, &value);
    return value;
  }

  auto Semaphore::signal(size_t value) noexcept -> void {
    if (!m_timelineSemaphore) {
      se::error("rhi :: Current semaphore is not a timeline one, signal from host is not allowed.");
      return;
    }
    VkSemaphoreSignalInfo signalInfo;
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.pNext = NULL;
    signalInfo.semaphore = m_semaphore;
    signalInfo.value = value;
    m_currentValue = value;
    vkSignalSemaphore(m_device->get_vk_device(), &signalInfo);
  }

  auto Semaphore::wait(size_t value) noexcept -> void {
    if (!m_timelineSemaphore) {
      se::error("rhi :: Current semaphore is not a timeline one, wait from host is not allowed.");
      return;
    }
    VkSemaphoreWaitInfo waitInfo;
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.pNext = NULL;
    waitInfo.flags = 0;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_semaphore;
    waitInfo.pValues = &value;
    vkWaitSemaphores(m_device->get_vk_device(), &waitInfo, UINT64_MAX);
  }

  auto Semaphore::get_handle() noexcept -> void* {
    #ifdef _WIN32
    HANDLE handle;
    VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
    semaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    semaphoreGetWin32HandleInfoKHR.pNext = NULL;
    semaphoreGetWin32HandleInfoKHR.semaphore = m_semaphore;
    semaphoreGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
    fpGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(m_device->get_vk_device(), "vkGetSemaphoreWin32HandleKHR");
    if (!fpGetSemaphoreWin32HandleKHR) {
      se::error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    if (fpGetSemaphoreWin32HandleKHR(m_device->get_vk_device(), &semaphoreGetWin32HandleInfoKHR, &handle) != VK_SUCCESS) {
      se::error("Failed to retrieve handle for buffer!");
    }
    return (void*)handle;
    #elif defined(__linux__)
    int fd;
    VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
    semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    semaphoreGetFdInfoKHR.pNext = NULL;
    semaphoreGetFdInfoKHR.semaphore = m_semaphore;
    semaphoreGetFdInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
    fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(m_device->get_vk_device(), "vkGetSemaphoreFdKHR");
    if (!fpGetSemaphoreFdKHR) {
      se::error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    if (fpGetSemaphoreFdKHR(m_device->get_vk_device(), &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
      se::error("Failed to retrieve handle for buffer!");
    }
    return (void*)(uintptr_t)fd;
    #endif
  }

  ComputePipeline::ComputePipeline(Device* device,
    ComputePipelineDescriptor const& desc)
    : m_device(device), m_layout(desc.layout) {
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = desc.compute.module->m_shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout =
      static_cast<PipelineLayout*>(desc.layout)->m_pipelineLayout;
    if (vkCreateComputePipelines(device->get_vk_device(), VK_NULL_HANDLE, 1,
      &pipelineInfo, nullptr,
      &m_pipeline) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create graphics pipeline!");
    }
  }

  ComputePipeline::~ComputePipeline() {
    if (m_pipeline) vkDestroyPipeline(m_device->get_vk_device(), m_pipeline, nullptr);
  }

  auto ComputePipeline::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
    objectNameInfo.objectHandle = uint64_t(m_pipeline);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
  }

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃                                                                           ┃
  // ┃ Definitions for pass and pipeline objects.                                ┃
  // ┃                                                                           ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  //

  RenderPass::RenderPass(Device* device, RenderPassDescriptor const& desc)
    : m_device(device) {
    // color attachments
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> attachmentRefs;
    for (auto const& colorAttach : desc.colorAttachments) {
      VkAttachmentDescription colorAttachment{};
      colorAttachment.format = impl::getVkFormat(
        static_cast<TextureView*>(colorAttach.view)->m_descriptor.format);
      colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
      colorAttachment.loadOp = impl::getVkAttachmentLoadOp(colorAttach.loadOp);
      colorAttachment.storeOp = impl::getVkAttachmentStoreOp(colorAttach.storeOp);
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
      depthAttachment.format = impl::getVkFormat(
        static_cast<TextureView*>(desc.depthStencilAttachment.view)
        ->m_descriptor.format);
      depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
      depthAttachment.loadOp =
        impl::getVkAttachmentLoadOp(desc.depthStencilAttachment.depthLoadOp);
      depthAttachment.storeOp =
        impl::getVkAttachmentStoreOp(desc.depthStencilAttachment.depthStoreOp);
      depthAttachment.stencilLoadOp =
        impl::getVkAttachmentLoadOp(desc.depthStencilAttachment.stencilLoadOp);
      depthAttachment.stencilStoreOp =
        impl::getVkAttachmentStoreOp(desc.depthStencilAttachment.stencilStoreOp);
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
    if (vkCreateRenderPass(device->get_vk_device(), &renderPassInfo, nullptr,
      &m_renderPass) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create render pass!");
    }
  }

  RenderPass::~RenderPass() {
    if (m_renderPass)
      vkDestroyRenderPass(m_device->get_vk_device(), m_renderPass, nullptr);
  }

  RenderPass::RenderPass(RenderPass&& pass)
    : m_device(pass.m_device),
    m_renderPass(pass.m_renderPass),
    m_clearValues(pass.m_clearValues) {
    pass.m_renderPass = nullptr;
  }

  auto RenderPass::operator=(RenderPass&& pass) -> RenderPass& {
    m_device = pass.m_device;
    m_renderPass = pass.m_renderPass;
    m_clearValues = pass.m_clearValues;
    pass.m_renderPass = nullptr;
    return *this;
  }

  RenderPipeline::RenderPipeline(Device* device,
    RenderPipelineDescriptor const& desc)
    : m_device(device) {
    if (desc.vertex.module)
      m_fixedFunctionSetttings.shaderStages.push_back(desc.vertex.module->m_shaderStageInfo);
    if (desc.fragment.module)
      m_fixedFunctionSetttings.shaderStages.push_back(desc.fragment.module->m_shaderStageInfo);
    if (desc.geometry.module)
      m_fixedFunctionSetttings.shaderStages.push_back(desc.geometry.module->m_shaderStageInfo);


    impl::fillFixedFunctionSettingDynamicInfo(m_fixedFunctionSetttings);
    impl::fillFixedFunctionSettingVertexInfo(desc.vertex, m_fixedFunctionSetttings);
    m_fixedFunctionSetttings.assemblyState =
      impl::getVkPipelineInputAssemblyStateCreateInfo(desc.primitive.topology);
    impl::fillFixedFunctionSettingViewportInfo(m_fixedFunctionSetttings);
    m_fixedFunctionSetttings.rasterizationState =
      impl::getVkPipelineRasterizationStateCreateInfo(desc.depthStencil,
        desc.fragment, desc.primitive);

    m_fixedFunctionSetttings.multisampleState =
      impl::getVkPipelineMultisampleStateCreateInfo(desc.multisample);
    m_fixedFunctionSetttings.depthStencilState =
      impl::getVkPipelineDepthStencilStateCreateInfo(desc.depthStencil);
    m_fixedFunctionSetttings.colorBlendAttachmentStates =
      impl::getVkPipelineColorBlendAttachmentState(desc.fragment);
    m_fixedFunctionSetttings.colorBlendState =
      impl::getVkPipelineColorBlendStateCreateInfo(
        m_fixedFunctionSetttings.colorBlendAttachmentStates);
    m_fixedFunctionSetttings.pipelineLayout = desc.layout;

    m_pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    m_pipelineInfo.stageCount = m_fixedFunctionSetttings.shaderStages.size();
    m_pipelineInfo.pStages = m_fixedFunctionSetttings.shaderStages.data();
    m_pipelineInfo.pVertexInputState = &m_fixedFunctionSetttings.vertexInputState;
    m_pipelineInfo.pInputAssemblyState = &m_fixedFunctionSetttings.assemblyState;
    m_pipelineInfo.pViewportState = &m_fixedFunctionSetttings.viewportState;
    m_pipelineInfo.pRasterizationState = &m_fixedFunctionSetttings.rasterizationState;
    m_pipelineInfo.pMultisampleState = &m_fixedFunctionSetttings.multisampleState;
    m_pipelineInfo.pDepthStencilState = &m_fixedFunctionSetttings.depthStencilState;
    m_pipelineInfo.pColorBlendState = &m_fixedFunctionSetttings.colorBlendState;
    m_pipelineInfo.pDynamicState = &m_fixedFunctionSetttings.dynamicState;
    m_pipelineInfo.layout = m_fixedFunctionSetttings.pipelineLayout->m_pipelineLayout;

    // conservative rasterization
    if (desc.rasterize.mode != RasterizeState::ConservativeMode::DISABLED) {
      m_fixedFunctionSetttings.conservativeRasterizationState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
      m_fixedFunctionSetttings.conservativeRasterizationState
        .extraPrimitiveOverestimationSize =
        desc.rasterize.extraPrimitiveOverestimationSize;
      m_fixedFunctionSetttings.conservativeRasterizationState
        .conservativeRasterizationMode =
        desc.rasterize.mode == RasterizeState::ConservativeMode::UNDERESTIMATE
        ? VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT
        : VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
      m_fixedFunctionSetttings.rasterizationState.pNext =
        &m_fixedFunctionSetttings.conservativeRasterizationState;
    }
  }

  RenderPipeline::~RenderPipeline() {
    if (m_pipeline) vkDestroyPipeline(m_device->get_vk_device(), m_pipeline, nullptr);
  }

  RenderPipeline::RenderPipeline(RenderPipeline&& pipeline)
    : m_device(pipeline.m_device), m_pipeline(pipeline.m_pipeline) {
    pipeline.m_pipeline = nullptr;
  }

  auto RenderPipeline::operator=(RenderPipeline&& pipeline)
    -> RenderPipeline& {
    m_device = pipeline.m_device;
    this->m_pipeline = pipeline.m_pipeline;
    pipeline.m_pipeline = nullptr;
    return *this;
  }

  auto RenderPipeline::set_name(std::string const& name) -> void {
    if (!m_device->m_debugLayerEnabled) return;
    VkDebugUtilsObjectNameInfoEXT objectNameInfo = {};
    objectNameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objectNameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
    objectNameInfo.objectHandle = uint64_t(m_pipeline);
    objectNameInfo.pObjectName = name.c_str();
    m_device->from_which_adapter()->from_which_context()->vkSetDebugUtilsObjectNameEXT(
      m_device->get_vk_device(), &objectNameInfo);
  }

  auto RenderPipeline::combine_render_pass(RenderPass* renderpass) noexcept
    -> void {
    // destroy current pipeline
    if (m_pipeline) {
      vkDestroyPipeline(m_device->get_vk_device(), m_pipeline, nullptr);
      m_pipeline = {};
    }
    m_pipelineInfo.renderPass = renderpass->m_renderPass;
    m_pipelineInfo.subpass = 0;
    m_pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    m_pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(m_device->get_vk_device(), VK_NULL_HANDLE, 1,
      &m_pipelineInfo, nullptr,
      &m_pipeline) != VK_SUCCESS) {
      se::error("VULKAN :: failed to create graphics pipeline!");
    }
  }

//  void importCudaExternalMemory(void** cudaPtr,
//    cudaExternalMemory_t& cudaMem,
//    VkDeviceMemory& vkMem,
//    VkDeviceSize                       size,
//    VkExternalMemoryHandleTypeFlagBits handleType)
//  {
//    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
//
//    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
//      externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
//    }
//    else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
//      externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
//    }
//    else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
//      externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
//    }
//    else {
//      throw std::runtime_error("Unknown handle type requested!");
//    }
//
//    externalMemoryHandleDesc.size = size;
////
////#ifdef _WIN64
////    externalMemoryHandleDesc.handle.win32.handle = (HANDLE)getMemHandle(vkMem, handleType);
////#else
////    externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(vkMem, handleType);
////#endif
//
//    checkCudaErrors(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));
//
//    cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
//    externalMemBufferDesc.offset = 0;
//    externalMemBufferDesc.size = size;
//    externalMemBufferDesc.flags = 0;
//
//    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
//  }
//
//  void importExternalBuffer(
//    void* handle,
//    VkExternalMemoryHandleTypeFlagBits handleType,
//    size_t                             size,
//    VkBufferUsageFlags                 usage,
//    VkMemoryPropertyFlags              properties,
//    VkBuffer& buffer,
//    VkDeviceMemory& memory)
//  {
//    VkBufferCreateInfo bufferInfo = {};
//    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//    bufferInfo.size = size;
//    bufferInfo.usage = usage;
//    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//
//    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
//      throw std::runtime_error("failed to create buffer!");
//    }
//
//    VkMemoryRequirements memRequirements;
//    vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);
//
//#ifdef _WIN64
//    VkImportMemoryWin32HandleInfoKHR handleInfo = {};
//    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
//    handleInfo.pNext = NULL;
//    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
//    handleInfo.handle = handle;
//    handleInfo.name = NULL;
//#else
//    VkImportMemoryFdInfoKHR handleInfo = {};
//    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
//    handleInfo.pNext = NULL;
//    handleInfo.fd = (int)(uintptr_t)handle;
//    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
//#endif /* _WIN64 */
//
//    VkMemoryAllocateInfo memAllocation = {};
//    memAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//    memAllocation.pNext = (void*)&handleInfo;
//    memAllocation.allocationSize = size;
//    memAllocation.memoryTypeIndex = findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, properties);
//
//    if (vkAllocateMemory(m_device, &memAllocation, nullptr, &memory) != VK_SUCCESS) {
//      throw std::runtime_error("Failed to import allocation!");
//    }
//
//    vkBindBufferMemory(m_device, buffer, memory, 0);
//  }

}
}