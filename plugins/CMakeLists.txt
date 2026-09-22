function(pih_install_plugin_manifest source_directory plugin_id component)
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/${source_directory}/manifest.json"
        DESTINATION "share/pih/plugins/${plugin_id}"
        COMPONENT "${component}"
    )
endfunction()

function(pih_plugin_enabled plugin_id result)
    if(PIH_ENABLED_PLUGINS STREQUAL "all")
        set(${result} TRUE PARENT_SCOPE)
        return()
    endif()
    list(FIND PIH_ENABLED_PLUGINS "${plugin_id}" plugin_index)
    if(plugin_index EQUAL -1)
        set(${result} FALSE PARENT_SCOPE)
    else()
        set(${result} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(pih_require_exact_plugin_set profile_id)
    set(expected_plugins ${ARGN})
    set(actual_plugins ${PIH_ENABLED_PLUGINS})
    list(SORT expected_plugins)
    list(SORT actual_plugins)
    if(NOT "${actual_plugins}" STREQUAL "${expected_plugins}")
        message(
            FATAL_ERROR
            "PIH_DEPLOYMENT_PROFILE=${profile_id} requires exactly: ${expected_plugins}; selected: ${actual_plugins}"
        )
    endif()
endfunction()

set(
    PIH_KNOWN_PLUGIN_IDS
    pih.platform.linux
    pih.execution.default
    pih.memory.host-spill
    pih.storage.verified-artifact
    pih.backend.nvidia-cuda
    pih.transport.nccl
    pih.kernels.deepseek-v4.sm89
    pih.kernels.deepseek-v4.sm90
    pih.model.deepseek-v4-flash
    pih.model.qwen3
    pih.model.deepseek-v41
    pih.kernels.deepseek-v41.sm103
    pih.surface.text-http
    pih.surface.health
    pih.surface.openai-http
    pih.testing.minimal
)
if(NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    foreach(plugin_id IN LISTS PIH_ENABLED_PLUGINS)
        list(FIND PIH_KNOWN_PLUGIN_IDS "${plugin_id}" plugin_index)
        if(plugin_index EQUAL -1)
            message(FATAL_ERROR "Unknown PIH plugin ID: ${plugin_id}")
        endif()
    endforeach()
endif()

# Deployment profiles are closed products; custom selections remain available
# for independently building or packaging one plugin.
if(PIH_DEPLOYMENT_PROFILE STREQUAL "phase1-native")
    pih_require_exact_plugin_set(phase1-native pih.testing.minimal)
elseif(PIH_DEPLOYMENT_PROFILE STREQUAL "deepseek-4090d-pp1")
    pih_require_exact_plugin_set(
        deepseek-4090d-pp1
        pih.platform.linux
        pih.storage.verified-artifact
        pih.backend.nvidia-cuda
        pih.execution.default
        pih.memory.host-spill
        pih.model.deepseek-v4-flash
        pih.surface.openai-http
        pih.surface.health
        pih.kernels.deepseek-v4.sm89
    )
endif()

pih_plugin_enabled(pih.platform.linux PIH_ENABLE_PLUGIN_PLATFORM_LINUX)
if(PIH_ENABLE_PLUGIN_PLATFORM_LINUX AND
   NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR "pih.platform.linux requires a Linux target")
endif()
if(PIH_ENABLE_PLUGIN_PLATFORM_LINUX AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(platform-linux)
    pih_install_plugin_manifest(platform-linux pih.platform.linux
                                pih-plugin-platform-linux)
endif()

pih_plugin_enabled(pih.execution.default PIH_ENABLE_PLUGIN_EXECUTION_DEFAULT)
if(PIH_ENABLE_PLUGIN_EXECUTION_DEFAULT)
    add_subdirectory(execution-default)
    pih_install_plugin_manifest(execution-default pih.execution.default
                                pih-plugin-execution-default)
endif()

pih_plugin_enabled(pih.memory.host-spill PIH_ENABLE_PLUGIN_MEMORY_HOST_SPILL)
if(PIH_ENABLE_PLUGIN_MEMORY_HOST_SPILL)
    add_subdirectory(memory-host-spill)
    pih_install_plugin_manifest(memory-host-spill pih.memory.host-spill
                                pih-plugin-memory-host-spill)
endif()

pih_plugin_enabled(pih.storage.verified-artifact PIH_ENABLE_PLUGIN_STORAGE)
if(PIH_ENABLE_PLUGIN_STORAGE)
    add_subdirectory(storage-verified-artifact)
    pih_install_plugin_manifest(storage-verified-artifact
                                pih.storage.verified-artifact
                                pih-plugin-storage-verified-artifact)
endif()

pih_plugin_enabled(pih.backend.nvidia-cuda PIH_ENABLE_PLUGIN_CUDA)
if(PIH_ENABLE_PLUGIN_CUDA AND NOT PIH_ENABLE_CUDA AND
   NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR "pih.backend.nvidia-cuda requires PIH_ENABLE_CUDA=ON")
endif()
if(PIH_ENABLE_PLUGIN_CUDA AND PIH_ENABLE_CUDA)
    add_subdirectory(backend-nvidia-cuda)
    pih_install_plugin_manifest(backend-nvidia-cuda pih.backend.nvidia-cuda
                                pih-plugin-backend-nvidia-cuda)
endif()

pih_plugin_enabled(pih.transport.nccl PIH_ENABLE_PLUGIN_NCCL)
if(PIH_ENABLE_PLUGIN_NCCL AND PIH_BUILD_NATIVE_ENGRAM_NCCL AND
   CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(transport-nccl)
    pih_install_plugin_manifest(transport-nccl pih.transport.nccl
                                pih-plugin-transport-nccl)
elseif(PIH_ENABLE_PLUGIN_NCCL AND NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR "pih.transport.nccl requires Linux and pinned NCCL")
endif()

pih_plugin_enabled(pih.kernels.deepseek-v4.sm89 PIH_ENABLE_PLUGIN_DS_KERNELS)
if(PIH_ENABLE_PLUGIN_DS_KERNELS AND NOT PIH_ENABLE_CUDA AND
   NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR
            "pih.kernels.deepseek-v4.sm89 requires PIH_ENABLE_CUDA=ON")
endif()
if(PIH_ENABLE_PLUGIN_DS_KERNELS AND PIH_ENABLE_CUDA)
    add_subdirectory(kernels/deepseek-v4-sm89)
    pih_install_plugin_manifest(kernels/deepseek-v4-sm89
                                pih.kernels.deepseek-v4.sm89
                                pih-kernel-deepseek-v4-sm89)
endif()

pih_plugin_enabled(pih.kernels.deepseek-v4.sm90 PIH_ENABLE_PLUGIN_DS_SM90_KERNELS)
if(PIH_ENABLE_PLUGIN_DS_SM90_KERNELS AND NOT PIH_ENABLE_CUDA AND
   NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR
            "pih.kernels.deepseek-v4.sm90 requires PIH_ENABLE_CUDA=ON")
endif()
if(PIH_ENABLE_PLUGIN_DS_SM90_KERNELS AND PIH_ENABLE_CUDA)
    add_subdirectory(kernels/deepseek-v4-sm90)
    pih_install_plugin_manifest(kernels/deepseek-v4-sm90
                                pih.kernels.deepseek-v4.sm90
                                pih-kernel-deepseek-v4-sm90)
endif()

pih_plugin_enabled(pih.model.deepseek-v4-flash PIH_ENABLE_PLUGIN_DS_MODEL)
if(PIH_ENABLE_PLUGIN_DS_MODEL AND NOT PIH_ENABLE_CUDA AND
   NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR
            "pih.model.deepseek-v4-flash requires PIH_ENABLE_CUDA=ON")
endif()
if(PIH_ENABLE_PLUGIN_DS_MODEL AND PIH_ENABLE_CUDA)
    add_subdirectory(model-deepseek-v4-flash)
    pih_install_plugin_manifest(model-deepseek-v4-flash
                                pih.model.deepseek-v4-flash
                                pih-plugin-model-deepseek-v4-flash)
endif()

pih_plugin_enabled(pih.surface.health PIH_ENABLE_PLUGIN_SURFACE_HEALTH)
pih_plugin_enabled(pih.model.deepseek-v41 PIH_ENABLE_PLUGIN_V41)
pih_plugin_enabled(pih.kernels.deepseek-v41.sm103 PIH_ENABLE_PLUGIN_V41_KERNELS)
# 'all' is not an architecture selection. V4.1 needs an explicit SM103 product.
if(NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    if(PIH_ENABLE_PLUGIN_V41_KERNELS)
        if(NOT PIH_ENABLE_CUDA OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
            message(FATAL_ERROR "V4.1 SM103 Pack requires Linux and CUDA")
        endif()
        add_subdirectory(kernels/deepseek-v41-sm103)
        pih_install_plugin_manifest(kernels/deepseek-v41-sm103 pih.kernels.deepseek-v41.sm103 pih-kernel-deepseek-v41-sm103)
    endif()
    if(PIH_ENABLE_PLUGIN_V41)
        include(model-deepseek-v41/plugin.cmake)
        pih_install_plugin_manifest(model-deepseek-v41 pih.model.deepseek-v41 pih-plugin-model-deepseek-v41)
    endif()
endif()
pih_plugin_enabled(pih.model.qwen3 PIH_ENABLE_PLUGIN_QWEN)
if(PIH_BUILD_QWEN_QUALIFICATION_TOOLS AND NOT PIH_ENABLE_PLUGIN_QWEN)
    message(FATAL_ERROR "Qwen CUDA qualification tools require pih.model.qwen3")
endif()
if(PIH_ENABLE_PLUGIN_QWEN AND PIH_ENABLE_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(kernels/qwen3)
    add_subdirectory(model-qwen3)
    pih_install_plugin_manifest(model-qwen3 pih.model.qwen3 pih-plugin-model-qwen3)
elseif(PIH_ENABLE_PLUGIN_QWEN AND NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR "pih.model.qwen3 requires Linux and CUDA")
endif()
pih_plugin_enabled(pih.surface.text-http PIH_ENABLE_PLUGIN_TEXT_HTTP)
if(PIH_ENABLE_PLUGIN_TEXT_HTTP AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(surface-text-http)
    pih_install_plugin_manifest(surface-text-http pih.surface.text-http pih-plugin-surface-text-http)
elseif(PIH_ENABLE_PLUGIN_TEXT_HTTP AND NOT PIH_ENABLED_PLUGINS STREQUAL "all")
    message(FATAL_ERROR "pih.surface.text-http requires Linux")
endif()
if(PIH_ENABLE_PLUGIN_SURFACE_HEALTH)
    add_subdirectory(surface-health)
    pih_install_plugin_manifest(surface-health pih.surface.health
                                pih-plugin-surface-health)
endif()

pih_plugin_enabled(pih.surface.openai-http PIH_ENABLE_PLUGIN_SURFACE_OPENAI)
if(PIH_ENABLE_PLUGIN_SURFACE_OPENAI)
    add_subdirectory(surface-openai-http)
    pih_install_plugin_manifest(surface-openai-http pih.surface.openai-http
                                pih-plugin-surface-openai-http)
endif()

pih_plugin_enabled(pih.testing.minimal PIH_ENABLE_PLUGIN_TEST_MINIMAL)
if(PIH_ENABLE_PLUGIN_TEST_MINIMAL)
    add_subdirectory(testing/minimal-plugin)
    pih_install_plugin_manifest(testing/minimal-plugin pih.testing.minimal
                                pih-plugin-testing-minimal)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(
        PIH_NATIVE_BUNDLE_TARGET
        IN ITEMS
            pih_plugin_platform_linux
            pih_plugin_backend_nvidia_cuda
            pih_plugin_execution_default
            pih_plugin_memory_host_spill
            pih_plugin_storage_verified_artifact
            pih_kernels_deepseek_v4_sm89
            pih_kernels_deepseek_v4_sm90
            pih_plugin_model_deepseek_v4_flash
            pih_plugin_surface_health
            pih_plugin_surface_openai_http
            pih_test_minimal_plugin
    )
        if(TARGET ${PIH_NATIVE_BUNDLE_TARGET})
            target_link_options(
                ${PIH_NATIVE_BUNDLE_TARGET} PRIVATE "LINKER:-z,defs"
            )
        endif()
    endforeach()
endif()
