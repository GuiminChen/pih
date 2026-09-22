cmake_minimum_required(VERSION 3.26)

project(PIH VERSION 0.1.0 LANGUAGES CXX)

find_package(OpenSSL 3.0 REQUIRED COMPONENTS Crypto)

# CMake 3.31 can misdecode localized MSVC /showIncludes output on Chinese
# Windows, leaving Ninja object files with zero header dependencies. Override
# the detected mojibake so public-header ABI changes always trigger rebuilds.
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
    set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
endif()

include(cmake/PIHOptions.cmake)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options("LINKER:-z,defs" "LINKER:--no-allow-shlib-undefined")
endif()

if(PIH_ENABLED_PLUGINS STREQUAL "all" AND PIH_ENABLE_CUDA)
    message(FATAL_ERROR
        "CUDA Kernel Packs have mutually exclusive architectures; select an explicit plugin set and one CMAKE_CUDA_ARCHITECTURES value")
endif()

if(PIH_BUILD_MONOLITH)
    message(FATAL_ERROR "The monolithic inference build has been removed; select native plugins")
endif()
if(PIH_BUILD_TESTS)
    message(FATAL_ERROR "The legacy aggregate test build has been removed; use PIH_BUILD_NATIVE_CONTRACT_TESTS")
endif()

if(PIH_BUILD_NATIVE_ENGRAM_NCCL AND
   (NOT PIH_ENABLE_CUDA OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "Native Engram NCCL primitives require Linux and PIH_ENABLE_CUDA=ON")
endif()

if(PIH_BUILD_PYTHON)
    message(
        FATAL_ERROR
        "The _pih Python inference extension has been removed; use native plugin workers and pih_client"
    )
endif()
if(PIH_ENABLE_NCCL)
    message(
        FATAL_ERROR
        "The legacy NCCL aggregate has been removed; select pih.transport.nccl with PIH_BUILD_NATIVE_ENGRAM_NCCL=ON"
    )
endif()
if(NOT PIH_DEPLOYMENT_PROFILE STREQUAL "custom" AND
   (NOT PIH_BUILD_WORKER OR NOT PIH_BUILD_PLUGINS))
    message(
        FATAL_ERROR
        "PIH_DEPLOYMENT_PROFILE=${PIH_DEPLOYMENT_PROFILE} requires PIH_BUILD_WORKER=ON, PIH_BUILD_PLUGINS=ON"
    )
endif()
if(PIH_DEPLOYMENT_PROFILE STREQUAL "deepseek-4090d-pp1" AND
   (NOT PIH_ENABLE_CUDA OR PIH_ENABLE_NCCL OR PIH_BUILD_PYTHON OR
    NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(
        FATAL_ERROR
        "deepseek-4090d-pp1 requires Linux, CUDA, native worker/plugins, and excludes Python, NCCL, and the monolith"
    )
endif()
if(PIH_DEPLOYMENT_PROFILE STREQUAL "deepseek-4090d-pp1" AND
   PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256 STREQUAL
       "0000000000000000000000000000000000000000000000000000000000000000")
    message(
        FATAL_ERROR
        "deepseek-4090d-pp1 requires the exact verified artifact root; set PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256"
    )
endif()

# Native plugins deliberately use a small set of support archives instead of
# the monolithic core.  Keep direct target dependencies closed at configure
# time so a convenient new link dependency cannot silently turn a minimal
# bundle back into an all-features binary.
function(pih_assert_direct_link_allowlist PIH_TARGET)
    set(PIH_ALLOWED_LINKS ${ARGN})
    get_target_property(PIH_DIRECT_LINKS ${PIH_TARGET} LINK_LIBRARIES)
    if(NOT PIH_DIRECT_LINKS OR PIH_DIRECT_LINKS MATCHES "-NOTFOUND$")
        set(PIH_DIRECT_LINKS)
    endif()
    foreach(PIH_DIRECT_LINK IN LISTS PIH_DIRECT_LINKS)
        if(NOT PIH_DIRECT_LINK IN_LIST PIH_ALLOWED_LINKS)
            message(
                FATAL_ERROR
                "${PIH_TARGET} has forbidden direct link dependency: ${PIH_DIRECT_LINK}"
            )
        endif()
    endforeach()
endfunction()

add_library(pih_plugin_sdk INTERFACE)
add_library(PIH::plugin_sdk ALIAS pih_plugin_sdk)
target_compile_features(pih_plugin_sdk INTERFACE cxx_std_20)
target_include_directories(
    pih_plugin_sdk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

configure_file(
    include/pih/version.h.in
    generated/pih/version.h
    @ONLY
)

add_library(
    pih_core_primitives STATIC
    src/core/bounded_json.cpp
    src/core/sha256.cpp
    src/core/canonical_extent_verifier.cpp
    src/core/status.cpp
)
target_compile_features(pih_core_primitives PUBLIC cxx_std_20)
set_target_properties(
    pih_core_primitives PROPERTIES POSITION_INDEPENDENT_CODE ON
)
    set_target_properties(
        pih_core_primitives
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
target_include_directories(
    pih_core_primitives
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(PIH_BUILD_TOKENIZER_TOOLS)
    add_subdirectory(plugins/common)
endif()
if(PIH_BUILD_ARTIFACT_TOOLS)
    add_subdirectory(plugins/offline-deepseek)
endif()
if(PIH_BUILD_QWEN_ARTIFACT_TOOLS)
    add_subdirectory(plugins/offline-qwen)
endif()

include(plugins/storage-verified-artifact/implementation.cmake)

# Native model plugins may use these model-agnostic value types and parsers,
# but must not link the monolithic PIH::core archive.  Keeping the support
# closure explicit prevents a new model reference from silently pulling an
# unrelated scheduler, backend, transport, or model implementation into the
# plugin shared object.
add_library(
    pih_native_model_support STATIC EXCLUDE_FROM_ALL
    src/core/allocator.cpp
    src/core/canonical_json.cpp
    src/core/buffer.cpp
    src/core/float16.cpp
    src/core/memory_copier.cpp
    src/core/canonical_hash.cpp
    src/core/tensor_view.cpp
    src/model/safetensors_header.cpp
    src/model/safetensors_shard_index.cpp
    src/model/safetensors_file.cpp
)
target_compile_features(pih_native_model_support PUBLIC cxx_std_20)
set_target_properties(
    pih_native_model_support PROPERTIES POSITION_INDEPENDENT_CODE ON
)
    set_target_properties(
        pih_native_model_support
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
target_include_directories(
    pih_native_model_support
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(
    pih_native_model_support
    PUBLIC pih_core_primitives
           pih_storage_verified_artifact_impl
)
pih_assert_direct_link_allowlist(
    pih_native_model_support
    pih_core_primitives
    pih_storage_verified_artifact_impl
)

include(plugins/execution-default/implementation.cmake)
include(plugins/memory-host-spill/implementation.cmake)
include(plugins/platform-linux/implementation.cmake)

# Model-owned CPU/offline/process supervision targets; no Python inference.
add_subdirectory(plugins/model-deepseek-v41)

include(plugins/model-deepseek-v4-flash/host.cmake)
include(plugins/model-deepseek-v4-flash/cuda.cmake)

if(PIH_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "PIH_ENABLE_CUDA=ON requires a CUDA compiler")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 13.2 REQUIRED)
    include(plugins/model-deepseek-v41/gpu.cmake)
    if("pih.kernels.deepseek-v41.sm103" IN_LIST PIH_ENABLED_PLUGINS)
        include(plugins/kernels/deepseek-v41-sm103/native.cmake)
    endif()
    if("pih.kernels.deepseek-v4.sm89" IN_LIST PIH_ENABLED_PLUGINS OR
       "pih.kernels.deepseek-v4.sm90" IN_LIST PIH_ENABLED_PLUGINS)
        include(plugins/kernels/deepseek-v4-sm89/native.cmake)
    endif()
    if(PIH_ENABLED_PLUGINS STREQUAL "all" OR
       "pih.kernels.deepseek-v4.sm90" IN_LIST PIH_ENABLED_PLUGINS)
        include(plugins/kernels/deepseek-v4-sm90/native.cmake)
    endif()
    if("pih.model.qwen3" IN_LIST PIH_ENABLED_PLUGINS OR PIH_ENABLED_PLUGINS STREQUAL "all")
        include(cmake/PIHCubins.cmake)
        foreach(qwen_architecture IN LISTS PIH_QWEN_KERNEL_ARCHITECTURES)
            pih_add_qwen_cubin(pih_qwen_sm${qwen_architecture}_cubin ${qwen_architecture})
            list(APPEND pih_qwen_cubin_outputs
                "${CMAKE_BINARY_DIR}/cubin/sm_${qwen_architecture}/qwen_bf16_primitives.cubin")
        endforeach()
        add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/generated/pih/qwen_cubin_digests.cpp"
            COMMAND "${CMAKE_COMMAND}"
                "-DARCHITECTURES=${PIH_QWEN_KERNEL_ARCHITECTURES}"
                "-DCUBIN_DIRECTORY=${CMAKE_BINARY_DIR}/cubin"
                "-DOUTPUT=${CMAKE_BINARY_DIR}/generated/pih/qwen_cubin_digests.cpp"
                -P "${CMAKE_SOURCE_DIR}/cmake/WriteQwenCubinDigests.cmake"
            DEPENDS ${pih_qwen_cubin_outputs} cmake/WriteQwenCubinDigests.cmake
            VERBATIM)
        add_custom_target(pih_qwen_cubin_identity
            DEPENDS "${CMAKE_BINARY_DIR}/generated/pih/qwen_cubin_digests.cpp")
    endif()
endif()

if(PIH_BUILD_WORKER)
    add_library(
        pih_microkernel STATIC
        src/microkernel/activation_runtime.cpp
        src/microkernel/capability_binding.cpp
        src/microkernel/plugin_loader.cpp
        src/microkernel/kernel_pack_loader.cpp
        src/microkernel/verified_native_library.cpp
        src/microkernel/plugin_registry.cpp
    )
    add_library(PIH::microkernel ALIAS pih_microkernel)
    target_link_libraries(
        pih_microkernel PUBLIC PIH::plugin_sdk pih_core_primitives
    )
    target_compile_features(pih_microkernel PUBLIC cxx_std_20)
    target_include_directories(
        pih_microkernel
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    if(CMAKE_DL_LIBS)
        target_link_libraries(pih_microkernel PRIVATE ${CMAKE_DL_LIBS})
    endif()
    pih_assert_direct_link_allowlist(
        pih_microkernel
        PIH::plugin_sdk
        pih_core_primitives
        ${CMAKE_DL_LIBS}
    )

    add_library(
        pih_development_lock STATIC
        src/worker/deployment_lock.cpp
    )
    target_compile_features(pih_development_lock PUBLIC cxx_std_20)
    target_link_libraries(
        pih_development_lock PUBLIC PIH::plugin_sdk pih_core_primitives
    )
    pih_assert_direct_link_allowlist(
        pih_development_lock
        PIH::plugin_sdk
        pih_core_primitives
    )
    target_include_directories(
        pih_development_lock
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )

    add_executable(
        pih-worker
        src/worker/main.cpp
        src/worker/worker_bootstrap.cpp
    )
    target_link_libraries(
        pih-worker PRIVATE PIH::microkernel pih_development_lock
    )
    pih_assert_direct_link_allowlist(
        pih-worker
        PIH::microkernel
        pih_development_lock
    )
    target_compile_features(pih-worker PRIVATE cxx_std_20)

    include(plugins/model-deepseek-v41/worker_commands.cmake)

    add_executable(pih-cli src/cli/main.cpp)
    set_target_properties(pih-cli PROPERTIES OUTPUT_NAME pih)
    target_link_libraries(pih-cli PRIVATE pih_development_lock)
    pih_assert_direct_link_allowlist(
        pih-cli
        pih_development_lock
    )
    target_compile_features(pih-cli PRIVATE cxx_std_20)
    install(
        TARGETS pih-cli
        RUNTIME DESTINATION bin COMPONENT pih-cli
    )
endif()

if(PIH_BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()

set(PIH_PHASE2_SHARED_STACK_AVAILABLE FALSE)
if(TARGET pih_plugin_platform_linux AND
   TARGET pih_plugin_storage_verified_artifact AND
   TARGET pih_plugin_backend_nvidia_cuda AND
   TARGET pih_plugin_execution_default AND
   TARGET pih_plugin_memory_host_spill)
    set(PIH_PHASE2_SHARED_STACK_AVAILABLE TRUE)
endif()

set(PIH_DEEPSEEK_4090D_PP1_STACK_AVAILABLE FALSE)
if(PIH_PHASE2_SHARED_STACK_AVAILABLE AND
   TARGET pih_kernels_deepseek_v4_sm89 AND
   TARGET pih_plugin_model_deepseek_v4_flash AND
   TARGET pih_plugin_surface_openai_http AND
   TARGET pih_plugin_surface_health)
    set(PIH_DEEPSEEK_4090D_PP1_STACK_AVAILABLE TRUE)
endif()

if(PIH_BUILD_WORKER AND PIH_BUILD_PLUGINS AND PIH_ENABLE_CUDA AND
   CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   PIH_PHASE2_SHARED_STACK_AVAILABLE)
    file(
        GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/pih-phase2-shared-stack.lock"
        CONTENT "{\"schema\":\"pih.development-lock.v1\",\"support_status\":\"unsupported\",\"kernel_packs\":[],\"plugins\":[{\"plugin_id\":\"pih.platform.linux\",\"plugin_version\":\"1.0.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:pih_plugin_platform_linux>\"},{\"plugin_id\":\"pih.storage.verified-artifact\",\"plugin_version\":\"1.0.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:pih_plugin_storage_verified_artifact>\"},{\"plugin_id\":\"pih.backend.nvidia-cuda\",\"plugin_version\":\"1.0.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:pih_plugin_backend_nvidia_cuda>\"},{\"plugin_id\":\"pih.execution.default\",\"plugin_version\":\"1.0.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:pih_plugin_execution_default>\"},{\"plugin_id\":\"pih.memory.host-spill\",\"plugin_version\":\"1.0.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:pih_plugin_memory_host_spill>\"}],\"capabilities\":[{\"capability_id\":\"artifact.verified-reader.v1\",\"provider_id\":\"pih.storage.verified-artifact\",\"contract_id\":\"pih.artifact.verified-reader.v1\",\"threading_model\":2,\"scope\":2,\"cardinality\":1},{\"capability_id\":\"device.cuda-async.v1\",\"provider_id\":\"pih.backend.nvidia-cuda\",\"contract_id\":\"pih.device.cuda-async.v1\",\"threading_model\":2,\"scope\":1,\"cardinality\":1},{\"capability_id\":\"device.cuda-memory.v1\",\"provider_id\":\"pih.backend.nvidia-cuda\",\"contract_id\":\"pih.device.cuda-memory.v1\",\"threading_model\":2,\"scope\":1,\"cardinality\":1},{\"capability_id\":\"device.cuda-resources.v1\",\"provider_id\":\"pih.backend.nvidia-cuda\",\"contract_id\":\"pih.device.cuda-resources.v1\",\"threading_model\":2,\"scope\":1,\"cardinality\":1},{\"capability_id\":\"device.cuda-runtime.v1\",\"provider_id\":\"pih.backend.nvidia-cuda\",\"contract_id\":\"pih.device.cuda-runtime.v1\",\"threading_model\":2,\"scope\":1,\"cardinality\":1},{\"capability_id\":\"execution.default.v1\",\"provider_id\":\"pih.execution.default\",\"contract_id\":\"pih.execution.default.v1\",\"threading_model\":3,\"scope\":2,\"cardinality\":1},{\"capability_id\":\"memory.host-spill.v1\",\"provider_id\":\"pih.memory.host-spill\",\"contract_id\":\"pih.memory.host-spill.v1\",\"threading_model\":2,\"scope\":2,\"cardinality\":1},{\"capability_id\":\"platform.linux.v1\",\"provider_id\":\"pih.platform.linux\",\"contract_id\":\"pih.platform.linux.v1\",\"threading_model\":2,\"scope\":1,\"cardinality\":1}]}"
    )
    add_custom_target(
        pih_phase2_shared_stack_smoke
        COMMAND
            "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/lib"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
            $<TARGET_FILE:pih_plugin_platform_linux>
            $<TARGET_FILE:pih_plugin_storage_verified_artifact>
            $<TARGET_FILE:pih_plugin_backend_nvidia_cuda>
            $<TARGET_FILE:pih_plugin_execution_default>
            $<TARGET_FILE:pih_plugin_memory_host_spill>
            "${CMAKE_BINARY_DIR}/lib"
        COMMAND
            $<TARGET_FILE:pih-worker>
            --lock "${CMAKE_BINARY_DIR}/pih-phase2-shared-stack.lock"
        DEPENDS
            pih-worker
            pih_plugin_platform_linux
            pih_plugin_storage_verified_artifact
            pih_plugin_backend_nvidia_cuda
            pih_plugin_execution_default
            pih_plugin_memory_host_spill
        USES_TERMINAL
        VERBATIM
    )
endif()

if(PIH_BUILD_WORKER AND PIH_BUILD_PLUGINS AND PIH_ENABLE_CUDA AND
   CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   PIH_DEEPSEEK_4090D_PP1_STACK_AVAILABLE)
    add_custom_target(
        pih_deepseek_4090d_pp1_bundle
        COMMAND
            "${CMAKE_COMMAND}"
            "-DPIH_ARTIFACT_ROOT_SHA256=${PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256}"
            "-DPIH_LOCK_PATH=${CMAKE_CURRENT_BINARY_DIR}/deepseek-v4-flash-rtx4090d-pp1.lock"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PIHValidateDeepSeekBundle.cmake"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDESTINATION=${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PIHRequireNewDirectory.cmake"
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-worker
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-cli
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-platform-linux
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-storage-verified-artifact
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-backend-nvidia-cuda
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-execution-default
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-memory-host-spill
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-kernel-deepseek-v4-sm89
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-model-deepseek-v4-flash
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-surface-openai-http
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-plugin-surface-health
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            --component pih-lock-deepseek-4090d-pp1
            --config $<CONFIG>
        COMMAND
            "${CMAKE_COMMAND}"
            "-DPIH_BUNDLE_ROOT=${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PIHSealDevelopmentLock.cmake"
        COMMAND
            "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1/bin/pih"
            inspect --lock
            "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1/deepseek-v4-flash-rtx4090d-pp1.lock"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DPIH_BUNDLE_ROOT=${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PIHVerifyDeepSeekPp1Bundle.cmake"
        DEPENDS
            pih-worker
            pih-cli
            pih_plugin_platform_linux
            pih_plugin_storage_verified_artifact
            pih_plugin_backend_nvidia_cuda
            pih_plugin_execution_default
            pih_plugin_memory_host_spill
            pih_kernels_deepseek_v4_sm89
            pih_plugin_model_deepseek_v4_flash
            pih_plugin_surface_openai_http
            pih_plugin_surface_health
        USES_TERMINAL
        VERBATIM
    )
    add_custom_target(
        pih_deepseek_4090d_pp1_smoke
        COMMAND
            "${CMAKE_COMMAND}"
            "-DPIH_BUNDLE_ROOT=${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
            "-DPIH_ARTIFACT_DIRECTORY=${PIH_DEEPSEEK_4090D_ARTIFACT_DIRECTORY}"
            "-DPIH_ARTIFACT_ROOT_SHA256=${PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PIHStageDeepSeekPp1Artifact.cmake"
        COMMAND
            "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1/bin/pih-worker"
            --lock
            "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1/deepseek-v4-flash-rtx4090d-pp1.lock"
        DEPENDS pih_deepseek_4090d_pp1_bundle
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bundle/deepseek-4090d-pp1"
        USES_TERMINAL
        VERBATIM
    )
endif()

include(CTest)
include(cmake/PIHRelease.cmake)
if(PIH_BUILD_NATIVE_CONTRACT_TESTS)
    add_subdirectory(tests/native)
endif()

if(NOT SKBUILD)
    include(cmake/PIHPluginSDK.cmake)
    if(PIH_BUILD_WORKER)
        install(
            TARGETS pih-worker
            RUNTIME DESTINATION bin
            COMPONENT pih-worker
        )
    endif()
    if(PIH_BUILD_WORKER AND PIH_BUILD_PLUGINS AND PIH_ENABLE_CUDA AND
       CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
       PIH_DEEPSEEK_4090D_PP1_STACK_AVAILABLE)
        file(
            READ
                "${CMAKE_CURRENT_SOURCE_DIR}/deploy/development/deepseek-v4-flash-rtx4090d-pp1.lock"
            PIH_DEEPSEEK_4090D_LOCK
        )
        set(
            PIH_DEEPSEEK_ZERO_ARTIFACT_ROOT
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        string(
            REPLACE
                "\"artifact_root_sha256_hex\":\"${PIH_DEEPSEEK_ZERO_ARTIFACT_ROOT}\""
                "\"artifact_root_sha256_hex\":\"${PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256}\""
                PIH_DEEPSEEK_4090D_LOCK
                "${PIH_DEEPSEEK_4090D_LOCK}"
        )
        file(
            GENERATE
            OUTPUT
                "${CMAKE_CURRENT_BINARY_DIR}/deepseek-v4-flash-rtx4090d-pp1.lock"
            CONTENT "${PIH_DEEPSEEK_4090D_LOCK}"
        )
        install(
            FILES
                "${CMAKE_CURRENT_BINARY_DIR}/deepseek-v4-flash-rtx4090d-pp1.lock"
            DESTINATION .
            COMPONENT pih-lock-deepseek-4090d-pp1
        )
    endif()
    # Native bundles expose only the explicit C ABI SDK installed above.
    # Publishing the full internal tree would silently restore C++ engine and
    # status-bridge dependencies for external plugins.
endif()
