# The external SDK deliberately exports only the header-only C ABI.
# It has no CUDA, OpenSSL, Python, or monolithic-runtime dependency.
include(CMakePackageConfigHelpers)
set_target_properties(pih_plugin_sdk PROPERTIES EXPORT_NAME plugin_sdk)
install(TARGETS pih_plugin_sdk EXPORT PIHPluginSDKTargets
        COMPONENT pih-plugin-sdk)
install(EXPORT PIHPluginSDKTargets NAMESPACE PIH::
        DESTINATION lib/cmake/PIHPluginSDK COMPONENT pih-plugin-sdk)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/pih/plugin_sdk/"
        DESTINATION include/pih/plugin_sdk COMPONENT pih-plugin-sdk
        FILES_MATCHING PATTERN "*.h" PATTERN "status_bridge.h" EXCLUDE)
install(FILES
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/engine_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/token_generation_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/text_inference_v2.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/qwen_kernels_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/deepseek_kernels_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/deepseek_v41_sm103_kernels_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/execution_default_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/execution_controller_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/transport_collective_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/health_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/memory_host_spill_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/nvidia_cuda_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/nvidia_cuda_memory_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/nvidia_cuda_resources_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/nvidia_cuda_async_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/openai_http_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/platform_linux_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/verified_artifact_v1.h"
    "${PROJECT_SOURCE_DIR}/include/pih/contracts/artifact_snapshot_v1.h"
    DESTINATION include/pih/contracts COMPONENT pih-plugin-sdk)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/PIHPluginSDKConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/PIHPluginSDKConfig.cmake"
    INSTALL_DESTINATION lib/cmake/PIHPluginSDK)
# Preview consumers must request an exact SDK package version.
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/PIHPluginSDKConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY ExactVersion)
install(FILES
    "${PROJECT_BINARY_DIR}/PIHPluginSDKConfig.cmake"
    "${PROJECT_BINARY_DIR}/PIHPluginSDKConfigVersion.cmake"
    DESTINATION lib/cmake/PIHPluginSDK COMPONENT pih-plugin-sdk)
