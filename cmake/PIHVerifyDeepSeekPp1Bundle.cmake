if(NOT DEFINED PIH_BUNDLE_ROOT OR
   NOT IS_DIRECTORY "${PIH_BUNDLE_ROOT}" OR
   IS_SYMLINK "${PIH_BUNDLE_ROOT}")
    message(FATAL_ERROR "DeepSeek PP1 bundle root is missing")
endif()

set(
    PIH_REQUIRED_BUNDLE_FILES
    bin/pih-worker
    bin/pih
    lib/pih_plugin_platform_linux.so
    lib/pih_plugin_storage_verified_artifact.so
    lib/pih_plugin_backend_nvidia_cuda.so
    lib/pih_plugin_execution_default.so
    lib/pih_plugin_memory_host_spill.so
    lib/pih_kernels_deepseek_v4_sm89.so
    lib/pih_plugin_model_deepseek_v4_flash.so
    lib/pih_plugin_surface_openai_http.so
    lib/pih_plugin_surface_health.so
    share/pih/plugins/pih.platform.linux/manifest.json
    share/pih/plugins/pih.storage.verified-artifact/manifest.json
    share/pih/plugins/pih.backend.nvidia-cuda/manifest.json
    share/pih/plugins/pih.execution.default/manifest.json
    share/pih/plugins/pih.memory.host-spill/manifest.json
    share/pih/plugins/pih.kernels.deepseek-v4.sm89/manifest.json
    share/pih/plugins/pih.model.deepseek-v4-flash/manifest.json
    share/pih/plugins/pih.surface.openai-http/manifest.json
    share/pih/plugins/pih.surface.health/manifest.json
    deepseek-v4-flash-rtx4090d-pp1.lock
)
foreach(PIH_REQUIRED_FILE IN LISTS PIH_REQUIRED_BUNDLE_FILES)
    get_filename_component(
        PIH_REQUIRED_PARENT
        "${PIH_REQUIRED_FILE}"
        DIRECTORY
    )
    while(NOT PIH_REQUIRED_PARENT STREQUAL "" AND
          NOT PIH_REQUIRED_PARENT STREQUAL ".")
        set(
            PIH_REQUIRED_PARENT_PATH
            "${PIH_BUNDLE_ROOT}/${PIH_REQUIRED_PARENT}"
        )
        if(NOT IS_DIRECTORY "${PIH_REQUIRED_PARENT_PATH}" OR
           IS_SYMLINK "${PIH_REQUIRED_PARENT_PATH}")
            message(
                FATAL_ERROR
                "DeepSeek PP1 bundle parent must be a real directory: ${PIH_REQUIRED_PARENT}"
            )
        endif()
        get_filename_component(
            PIH_REQUIRED_PARENT
            "${PIH_REQUIRED_PARENT}"
            DIRECTORY
        )
    endwhile()
    set(PIH_REQUIRED_PATH "${PIH_BUNDLE_ROOT}/${PIH_REQUIRED_FILE}")
    if(NOT EXISTS "${PIH_REQUIRED_PATH}" OR
       IS_DIRECTORY "${PIH_REQUIRED_PATH}" OR
       IS_SYMLINK "${PIH_REQUIRED_PATH}")
        message(
            FATAL_ERROR
            "DeepSeek PP1 bundle artifact is missing, a directory, or a symbolic link: ${PIH_REQUIRED_FILE}"
        )
    endif()
endforeach()

function(pih_reject_runtime_dependencies PIH_RELATIVE_PATH
         PIH_ALLOW_CUDA_RUNTIME)
    set(PIH_RUNTIME_DEPENDENCY_PATH
        "${PIH_BUNDLE_ROOT}/${PIH_RELATIVE_PATH}")
    if(PIH_RELATIVE_PATH MATCHES "^bin/")
        file(
            GET_RUNTIME_DEPENDENCIES
            EXECUTABLES "${PIH_RUNTIME_DEPENDENCY_PATH}"
            DIRECTORIES "${PIH_BUNDLE_ROOT}/lib"
            RESOLVED_DEPENDENCIES_VAR PIH_RESOLVED_DEPENDENCIES
            UNRESOLVED_DEPENDENCIES_VAR PIH_UNRESOLVED_DEPENDENCIES
        )
    else()
        file(
            GET_RUNTIME_DEPENDENCIES
            LIBRARIES "${PIH_RUNTIME_DEPENDENCY_PATH}"
            DIRECTORIES "${PIH_BUNDLE_ROOT}/lib"
            RESOLVED_DEPENDENCIES_VAR PIH_RESOLVED_DEPENDENCIES
            UNRESOLVED_DEPENDENCIES_VAR PIH_UNRESOLVED_DEPENDENCIES
        )
    endif()
    foreach(
        PIH_RUNTIME_DEPENDENCY
        IN LISTS
            PIH_RESOLVED_DEPENDENCIES
            PIH_UNRESOLVED_DEPENDENCIES
    )
        get_filename_component(
            PIH_RUNTIME_DEPENDENCY_NAME
            "${PIH_RUNTIME_DEPENDENCY}"
            NAME
        )
        string(
            TOLOWER
            "${PIH_RUNTIME_DEPENDENCY_NAME}"
            PIH_RUNTIME_DEPENDENCY_NAME
        )
        if(PIH_RUNTIME_DEPENDENCY_NAME MATCHES
               "^(lib)?(cublas|cublaslt|nccl)(\\.|$)" OR
           (NOT PIH_ALLOW_CUDA_RUNTIME AND
            PIH_RUNTIME_DEPENDENCY_NAME MATCHES
                "^(lib)?(cuda|cudart)(\\.|$)") OR
           PIH_RUNTIME_DEPENDENCY_NAME MATCHES
               "^(_pih(\\.|$)|libpython)" OR
           PIH_RUNTIME_DEPENDENCY_NAME MATCHES "^(lib)?pih_(plugin|kernels)_")
            message(
                FATAL_ERROR
                "${PIH_RELATIVE_PATH} has forbidden runtime dependency: ${PIH_RUNTIME_DEPENDENCY_NAME}"
            )
        endif()
    endforeach()
endfunction()

foreach(
    PIH_HOST_ONLY_ARTIFACT
    IN ITEMS
        bin/pih-worker
        bin/pih
        lib/pih_plugin_platform_linux.so
        lib/pih_plugin_storage_verified_artifact.so
        lib/pih_plugin_execution_default.so
        lib/pih_plugin_memory_host_spill.so
        lib/pih_plugin_surface_openai_http.so
        lib/pih_plugin_surface_health.so
)
    pih_reject_runtime_dependencies("${PIH_HOST_ONLY_ARTIFACT}" FALSE)
endforeach()
pih_reject_runtime_dependencies(
    lib/pih_plugin_backend_nvidia_cuda.so TRUE
)
pih_reject_runtime_dependencies(
    lib/pih_kernels_deepseek_v4_sm89.so TRUE
)
pih_reject_runtime_dependencies(
    lib/pih_plugin_model_deepseek_v4_flash.so FALSE
)

function(pih_verify_manifest PIH_RELATIVE_PATH PIH_EXPECTED_ID PIH_EXPECTED_KIND)
    file(READ "${PIH_BUNDLE_ROOT}/${PIH_RELATIVE_PATH}" PIH_MANIFEST_JSON)
    foreach(PIH_FIELD IN ITEMS schema plugin_id plugin_version artifact_kind)
        string(
            JSON PIH_MANIFEST_${PIH_FIELD}
            ERROR_VARIABLE PIH_MANIFEST_ERROR
            GET "${PIH_MANIFEST_JSON}" "${PIH_FIELD}"
        )
        if(PIH_MANIFEST_ERROR)
            message(FATAL_ERROR "${PIH_RELATIVE_PATH} has no valid ${PIH_FIELD}")
        endif()
    endforeach()
    if(NOT PIH_MANIFEST_schema STREQUAL "pih.plugin-manifest.v1" OR
       NOT PIH_MANIFEST_plugin_id STREQUAL "${PIH_EXPECTED_ID}" OR
       NOT PIH_MANIFEST_plugin_version STREQUAL "1.0.0" OR
       NOT PIH_MANIFEST_artifact_kind STREQUAL "${PIH_EXPECTED_KIND}")
        message(FATAL_ERROR "${PIH_RELATIVE_PATH} identity does not match the PP1 Lock")
    endif()
    if(PIH_EXPECTED_KIND STREQUAL "kernel_pack")
        string(JSON PIH_PACK_ABI GET "${PIH_MANIFEST_JSON}" pack_abi)
        string(JSON PIH_PACK_ARCHITECTURE GET "${PIH_MANIFEST_JSON}"
               sm_architecture)
        string(JSON PIH_PACK_BINARY GET "${PIH_MANIFEST_JSON}" binary)
        string(JSON PIH_SUPPORTED_MODEL_COUNT LENGTH "${PIH_MANIFEST_JSON}"
               supports_model_abi)
        string(JSON PIH_SUPPORTED_MODEL GET "${PIH_MANIFEST_JSON}"
               supports_model_abi 0)
        if(NOT PIH_PACK_ABI STREQUAL "pih.deepseek-sm89-kernel-pack.v1" OR
           NOT PIH_PACK_ARCHITECTURE STREQUAL "sm89" OR
           NOT PIH_PACK_BINARY STREQUAL
               "lib/pih_kernels_deepseek_v4_sm89.so" OR
           NOT PIH_SUPPORTED_MODEL_COUNT EQUAL 1 OR
           NOT PIH_SUPPORTED_MODEL STREQUAL
               "pih.model.deepseek-v4-flash.v1")
            message(FATAL_ERROR "${PIH_RELATIVE_PATH} Kernel Pack mismatch")
        endif()
    else()
        string(JSON PIH_PLUGIN_ABI GET "${PIH_MANIFEST_JSON}" plugin_abi)
        string(JSON PIH_ENTRY_SYMBOL GET "${PIH_MANIFEST_JSON}" entry_symbol)
        if(NOT PIH_PLUGIN_ABI STREQUAL "pih.native-plugin.v1" OR
           NOT PIH_ENTRY_SYMBOL STREQUAL "pih_plugin_entry_v1")
            message(FATAL_ERROR "${PIH_RELATIVE_PATH} native ABI mismatch")
        endif()
    endif()
endfunction()

function(pih_require_manifest_section_count PIH_RELATIVE_PATH PIH_SECTION
         PIH_EXPECTED_COUNT)
    file(READ "${PIH_BUNDLE_ROOT}/${PIH_RELATIVE_PATH}" PIH_MANIFEST_JSON)
    string(
        JSON PIH_ACTUAL_COUNT
        ERROR_VARIABLE PIH_MANIFEST_ERROR
        LENGTH "${PIH_MANIFEST_JSON}" "${PIH_SECTION}"
    )
    if(PIH_MANIFEST_ERROR OR NOT PIH_ACTUAL_COUNT EQUAL PIH_EXPECTED_COUNT)
        message(
            FATAL_ERROR
            "${PIH_RELATIVE_PATH} ${PIH_SECTION} closure is not exact"
        )
    endif()
endfunction()

function(pih_require_manifest_contract PIH_RELATIVE_PATH PIH_SECTION
         PIH_CAPABILITY_ID PIH_CONTRACT_ID PIH_EXPECTED_SCOPE
         PIH_EXPECTED_CARDINALITY PIH_EXPECTED_THREADING)
    file(READ "${PIH_BUNDLE_ROOT}/${PIH_RELATIVE_PATH}" PIH_MANIFEST_JSON)
    string(
        JSON PIH_CONTRACT_COUNT
        ERROR_VARIABLE PIH_MANIFEST_ERROR
        LENGTH "${PIH_MANIFEST_JSON}" "${PIH_SECTION}"
    )
    if(PIH_MANIFEST_ERROR OR PIH_CONTRACT_COUNT EQUAL 0)
        message(FATAL_ERROR "${PIH_RELATIVE_PATH} has no ${PIH_SECTION} contracts")
    endif()
    math(EXPR PIH_CONTRACT_LAST "${PIH_CONTRACT_COUNT} - 1")
    foreach(PIH_CONTRACT_INDEX RANGE 0 ${PIH_CONTRACT_LAST})
        string(JSON PIH_ACTUAL_CAPABILITY GET "${PIH_MANIFEST_JSON}"
               "${PIH_SECTION}" ${PIH_CONTRACT_INDEX} capability_id)
        if(PIH_ACTUAL_CAPABILITY STREQUAL "${PIH_CAPABILITY_ID}")
            string(JSON PIH_ACTUAL_CONTRACT GET "${PIH_MANIFEST_JSON}"
                   "${PIH_SECTION}" ${PIH_CONTRACT_INDEX} contract_id)
            if(NOT PIH_ACTUAL_CONTRACT STREQUAL "${PIH_CONTRACT_ID}")
                message(FATAL_ERROR "${PIH_RELATIVE_PATH} contract mismatch")
            endif()
            string(JSON PIH_CAPABILITY_SCOPE ERROR_VARIABLE PIH_SCOPE_ERROR
                   GET "${PIH_MANIFEST_JSON}" "${PIH_SECTION}"
                   ${PIH_CONTRACT_INDEX} scope)
            string(JSON PIH_CAPABILITY_CARDINALITY
                   ERROR_VARIABLE PIH_CARDINALITY_ERROR
                   GET "${PIH_MANIFEST_JSON}" "${PIH_SECTION}"
                   ${PIH_CONTRACT_INDEX} cardinality)
            if(PIH_SCOPE_ERROR OR PIH_CARDINALITY_ERROR OR
               NOT PIH_CAPABILITY_SCOPE STREQUAL "${PIH_EXPECTED_SCOPE}" OR
               NOT PIH_CAPABILITY_CARDINALITY STREQUAL
                   "${PIH_EXPECTED_CARDINALITY}")
                message(FATAL_ERROR
                        "${PIH_RELATIVE_PATH} has invalid capability shape")
            endif()
            if(PIH_SECTION STREQUAL "provides")
                string(JSON PIH_THREADING_MODEL ERROR_VARIABLE PIH_THREADING_ERROR
                       GET "${PIH_MANIFEST_JSON}" "${PIH_SECTION}"
                       ${PIH_CONTRACT_INDEX} threading_model)
                if(PIH_THREADING_ERROR OR
                   NOT PIH_THREADING_MODEL STREQUAL
                       "${PIH_EXPECTED_THREADING}")
                    message(FATAL_ERROR
                            "${PIH_RELATIVE_PATH} has no valid threading model")
                endif()
            endif()
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "${PIH_RELATIVE_PATH} capability is missing")
endfunction()

pih_verify_manifest(
    share/pih/plugins/pih.platform.linux/manifest.json
    pih.platform.linux native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.storage.verified-artifact/manifest.json
    pih.storage.verified-artifact native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.backend.nvidia-cuda/manifest.json
    pih.backend.nvidia-cuda native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.execution.default/manifest.json
    pih.execution.default native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.memory.host-spill/manifest.json
    pih.memory.host-spill native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.kernels.deepseek-v4.sm89/manifest.json
    pih.kernels.deepseek-v4.sm89 kernel_pack
)
pih_verify_manifest(
    share/pih/plugins/pih.model.deepseek-v4-flash/manifest.json
    pih.model.deepseek-v4-flash native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.surface.openai-http/manifest.json
    pih.surface.openai-http native_plugin
)
pih_verify_manifest(
    share/pih/plugins/pih.surface.health/manifest.json
    pih.surface.health native_plugin
)

set(PIH_MANIFEST_PREFIX share/pih/plugins)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.platform.linux/manifest.json provides 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.platform.linux/manifest.json requires 0)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.storage.verified-artifact/manifest.json provides 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.storage.verified-artifact/manifest.json requires 0)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.backend.nvidia-cuda/manifest.json provides 4)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.backend.nvidia-cuda/manifest.json requires 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.execution.default/manifest.json provides 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.execution.default/manifest.json requires 0)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.memory.host-spill/manifest.json provides 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.memory.host-spill/manifest.json requires 3)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.model.deepseek-v4-flash/manifest.json provides 3)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.model.deepseek-v4-flash/manifest.json requires 7)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.model.deepseek-v4-flash/manifest.json kernel_packs 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.surface.openai-http/manifest.json provides 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.surface.openai-http/manifest.json requires 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.surface.health/manifest.json provides 1)
pih_require_manifest_section_count(
    ${PIH_MANIFEST_PREFIX}/pih.surface.health/manifest.json requires 0)

file(READ
    "${PIH_BUNDLE_ROOT}/${PIH_MANIFEST_PREFIX}/pih.model.deepseek-v4-flash/manifest.json"
    PIH_MODEL_MANIFEST_JSON)
string(JSON PIH_MODEL_PACK_ID GET "${PIH_MODEL_MANIFEST_JSON}"
       kernel_packs 0 pack_id)
string(JSON PIH_MODEL_PACK_ABI GET "${PIH_MODEL_MANIFEST_JSON}"
       kernel_packs 0 pack_abi)
string(JSON PIH_MODEL_PACK_CARDINALITY GET "${PIH_MODEL_MANIFEST_JSON}"
       kernel_packs 0 cardinality)
if(NOT PIH_MODEL_PACK_ID STREQUAL "pih.kernels.deepseek-v4.sm89" OR
   NOT PIH_MODEL_PACK_ABI STREQUAL "pih.deepseek-sm89-kernel-pack.v1" OR
   NOT PIH_MODEL_PACK_CARDINALITY STREQUAL "exactly_one")
    message(FATAL_ERROR "DeepSeek model Kernel Pack requirement drifted")
endif()

pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.platform.linux/manifest.json provides
    platform.linux.v1 pih.platform.linux.v1 process exactly_one serialized
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.storage.verified-artifact/manifest.json provides
    artifact.verified-reader.v1 pih.artifact.verified-reader.v1
    activation exactly_one serialized
)
foreach(PIH_CUDA_CONTRACT IN ITEMS runtime memory resources async)
    pih_require_manifest_contract(
        ${PIH_MANIFEST_PREFIX}/pih.backend.nvidia-cuda/manifest.json provides
        device.cuda-${PIH_CUDA_CONTRACT}.v1
        pih.device.cuda-${PIH_CUDA_CONTRACT}.v1 process exactly_one serialized
    )
endforeach()
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.backend.nvidia-cuda/manifest.json requires
    platform.linux.v1 pih.platform.linux.v1 process exactly_one ""
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.execution.default/manifest.json provides
    execution.default.v1 pih.execution.default.v1 activation exactly_one concurrent
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.execution.default/manifest.json provides
    execution.controller.v1 pih.execution.controller.v1 activation exactly_one concurrent
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.memory.host-spill/manifest.json provides
    memory.host-spill.v1 pih.memory.host-spill.v1 activation exactly_one serialized
)
foreach(PIH_CUDA_CONTRACT IN ITEMS memory resources async)
    pih_require_manifest_contract(
        ${PIH_MANIFEST_PREFIX}/pih.memory.host-spill/manifest.json requires
        device.cuda-${PIH_CUDA_CONTRACT}.v1
        pih.device.cuda-${PIH_CUDA_CONTRACT}.v1 process exactly_one ""
    )
endforeach()
  set(PIH_MODEL_MANIFEST
      ${PIH_MANIFEST_PREFIX}/pih.model.deepseek-v4-flash/manifest.json)
  pih_require_manifest_contract(
    ${PIH_MODEL_MANIFEST} provides inference.text.v2 pih.inference.text.v2
    activation exactly_one serialized
  )
pih_require_manifest_contract(
    ${PIH_MODEL_MANIFEST} provides engine.primary.v1 pih.engine.v1
    activation exactly_one serialized
)
pih_require_manifest_contract(
    ${PIH_MODEL_MANIFEST} provides inference.tokens.v1 pih.inference.tokens.v1
    activation exactly_one serialized
)
pih_require_manifest_contract(
    ${PIH_MODEL_MANIFEST} requires artifact.verified-reader.v1
    pih.artifact.verified-reader.v1 activation exactly_one ""
)
foreach(PIH_CUDA_CONTRACT IN ITEMS runtime memory resources async)
    pih_require_manifest_contract(
        ${PIH_MODEL_MANIFEST} requires device.cuda-${PIH_CUDA_CONTRACT}.v1
        pih.device.cuda-${PIH_CUDA_CONTRACT}.v1 process exactly_one ""
    )
endforeach()
pih_require_manifest_contract(
    ${PIH_MODEL_MANIFEST} requires execution.default.v1 pih.execution.default.v1
    activation exactly_one ""
)
pih_require_manifest_contract(
    ${PIH_MODEL_MANIFEST} requires memory.host-spill.v1 pih.memory.host-spill.v1
    activation zero_or_one ""
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.surface.openai-http/manifest.json provides
    surface.openai-http.v1 pih.surface.openai-http.v1
    activation exactly_one concurrent
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.surface.openai-http/manifest.json requires
    engine.primary.v1 pih.engine.v1 activation exactly_one ""
)
pih_require_manifest_contract(
    ${PIH_MANIFEST_PREFIX}/pih.surface.health/manifest.json provides
    surface.health.v1 pih.surface.health.v1
    activation exactly_one concurrent
)

file(
    READ
    "${PIH_BUNDLE_ROOT}/deepseek-v4-flash-rtx4090d-pp1.lock"
    PIH_LOCK_JSON
)
string(JSON PIH_LOCK_SCHEMA GET "${PIH_LOCK_JSON}" schema)
string(JSON PIH_LOCK_SUPPORT_STATUS GET "${PIH_LOCK_JSON}" support_status)
string(JSON PIH_LOCK_PLUGIN_COUNT LENGTH "${PIH_LOCK_JSON}" plugins)
string(JSON PIH_LOCK_KERNEL_COUNT LENGTH "${PIH_LOCK_JSON}" kernel_packs)
string(JSON PIH_LOCK_CAPABILITY_COUNT LENGTH "${PIH_LOCK_JSON}" capabilities)
if(NOT PIH_LOCK_SCHEMA STREQUAL "pih.development-lock.v1" OR
   NOT PIH_LOCK_SUPPORT_STATUS STREQUAL "unsupported" OR
   NOT PIH_LOCK_KERNEL_COUNT EQUAL 1)
    message(FATAL_ERROR "DeepSeek PP1 Lock closure shape is invalid")
endif()

set(
    PIH_LOCKED_PLUGIN_IDS
    pih.platform.linux
    pih.storage.verified-artifact
    pih.backend.nvidia-cuda
    pih.execution.default
    pih.memory.host-spill
    pih.model.deepseek-v4-flash
    pih.surface.openai-http
    pih.surface.health
)
set(
    PIH_LOCKED_PLUGIN_BINARIES
    pih_plugin_platform_linux.so
    pih_plugin_storage_verified_artifact.so
    pih_plugin_backend_nvidia_cuda.so
    pih_plugin_execution_default.so
    pih_plugin_memory_host_spill.so
    pih_plugin_model_deepseek_v4_flash.so
    pih_plugin_surface_openai_http.so
    pih_plugin_surface_health.so
)
list(LENGTH PIH_LOCKED_PLUGIN_IDS PIH_EXPECTED_PLUGIN_COUNT)
if(NOT PIH_LOCK_PLUGIN_COUNT EQUAL PIH_EXPECTED_PLUGIN_COUNT)
    message(FATAL_ERROR "DeepSeek PP1 Lock plugin count drifted")
endif()
math(EXPR PIH_LOCK_PLUGIN_LAST "${PIH_LOCK_PLUGIN_COUNT} - 1")
foreach(PIH_PLUGIN_INDEX RANGE 0 ${PIH_LOCK_PLUGIN_LAST})
    list(GET PIH_LOCKED_PLUGIN_IDS ${PIH_PLUGIN_INDEX} PIH_EXPECTED_PLUGIN_ID)
    list(GET PIH_LOCKED_PLUGIN_BINARIES ${PIH_PLUGIN_INDEX}
         PIH_EXPECTED_PLUGIN_BINARY)
    string(JSON PIH_LOCKED_PLUGIN_ID GET "${PIH_LOCK_JSON}" plugins
           ${PIH_PLUGIN_INDEX} plugin_id)
    string(JSON PIH_LOCKED_PLUGIN_VERSION GET "${PIH_LOCK_JSON}" plugins
           ${PIH_PLUGIN_INDEX} plugin_version)
    string(JSON PIH_LOCKED_PLUGIN_ENTRYPOINT GET "${PIH_LOCK_JSON}" plugins
           ${PIH_PLUGIN_INDEX} entrypoint)
    string(JSON PIH_LOCKED_PLUGIN_SHA256 GET "${PIH_LOCK_JSON}" plugins
           ${PIH_PLUGIN_INDEX} entrypoint_sha256_hex)
    file(SHA256
         "${PIH_BUNDLE_ROOT}/lib/${PIH_EXPECTED_PLUGIN_BINARY}"
         PIH_OBSERVED_PLUGIN_SHA256)
    if(NOT PIH_LOCKED_PLUGIN_ID STREQUAL "${PIH_EXPECTED_PLUGIN_ID}" OR
       NOT PIH_LOCKED_PLUGIN_VERSION STREQUAL "1.0.0" OR
       NOT PIH_LOCKED_PLUGIN_ENTRYPOINT STREQUAL
           "lib/${PIH_EXPECTED_PLUGIN_BINARY}" OR
       NOT PIH_LOCKED_PLUGIN_SHA256 STREQUAL
           "${PIH_OBSERVED_PLUGIN_SHA256}")
        message(FATAL_ERROR "DeepSeek PP1 Lock plugin topology drifted")
    endif()
endforeach()

set(
    PIH_LOCKED_CAPABILITIES
    "artifact.verified-reader.v1|pih.storage.verified-artifact|pih.artifact.verified-reader.v1|2|2|1"
    "device.cuda-async.v1|pih.backend.nvidia-cuda|pih.device.cuda-async.v1|2|1|1"
    "device.cuda-memory.v1|pih.backend.nvidia-cuda|pih.device.cuda-memory.v1|2|1|1"
    "device.cuda-resources.v1|pih.backend.nvidia-cuda|pih.device.cuda-resources.v1|2|1|1"
    "device.cuda-runtime.v1|pih.backend.nvidia-cuda|pih.device.cuda-runtime.v1|2|1|1"
    "engine.primary.v1|pih.model.deepseek-v4-flash|pih.engine.v1|2|2|1"
    "execution.controller.v1|pih.execution.default|pih.execution.controller.v1|3|2|1"
    "execution.default.v1|pih.execution.default|pih.execution.default.v1|3|2|1"
    "inference.text.v2|pih.model.deepseek-v4-flash|pih.inference.text.v2|2|2|1"
    "inference.tokens.v1|pih.model.deepseek-v4-flash|pih.inference.tokens.v1|2|2|1"
    "memory.host-spill.v1|pih.memory.host-spill|pih.memory.host-spill.v1|2|2|1"
    "pih.kernels.deepseek-v4.sm89|pih.kernels.deepseek-v4.sm89|pih.deepseek-sm89-kernel-pack.v1|3|2|1"
    "platform.linux.v1|pih.platform.linux|pih.platform.linux.v1|2|1|1"
    "surface.health.v1|pih.surface.health|pih.surface.health.v1|3|2|1"
    "surface.openai-http.v1|pih.surface.openai-http|pih.surface.openai-http.v1|3|2|1"
)
list(LENGTH PIH_LOCKED_CAPABILITIES PIH_EXPECTED_CAPABILITY_COUNT)
if(NOT PIH_LOCK_CAPABILITY_COUNT EQUAL PIH_EXPECTED_CAPABILITY_COUNT)
    message(FATAL_ERROR "DeepSeek PP1 Lock capability count drifted")
endif()
math(EXPR PIH_LOCK_CAPABILITY_LAST "${PIH_LOCK_CAPABILITY_COUNT} - 1")
foreach(PIH_CAPABILITY_INDEX RANGE 0 ${PIH_LOCK_CAPABILITY_LAST})
    list(GET PIH_LOCKED_CAPABILITIES ${PIH_CAPABILITY_INDEX}
         PIH_EXPECTED_CAPABILITY)
    string(REPLACE "|" ";" PIH_EXPECTED_CAPABILITY
           "${PIH_EXPECTED_CAPABILITY}")
    list(GET PIH_EXPECTED_CAPABILITY 0 PIH_EXPECTED_CAPABILITY_ID)
    list(GET PIH_EXPECTED_CAPABILITY 1 PIH_EXPECTED_PROVIDER_ID)
    list(GET PIH_EXPECTED_CAPABILITY 2 PIH_EXPECTED_CONTRACT_ID)
    list(GET PIH_EXPECTED_CAPABILITY 3 PIH_EXPECTED_THREADING_MODEL)
    list(GET PIH_EXPECTED_CAPABILITY 4 PIH_EXPECTED_SCOPE)
    list(GET PIH_EXPECTED_CAPABILITY 5 PIH_EXPECTED_CARDINALITY)
    foreach(PIH_CAPABILITY_FIELD IN ITEMS capability_id provider_id contract_id
                                           threading_model scope cardinality)
        string(JSON PIH_LOCKED_${PIH_CAPABILITY_FIELD} GET "${PIH_LOCK_JSON}"
               capabilities ${PIH_CAPABILITY_INDEX} ${PIH_CAPABILITY_FIELD})
    endforeach()
    if(NOT PIH_LOCKED_capability_id STREQUAL
           "${PIH_EXPECTED_CAPABILITY_ID}" OR
       NOT PIH_LOCKED_provider_id STREQUAL "${PIH_EXPECTED_PROVIDER_ID}" OR
       NOT PIH_LOCKED_contract_id STREQUAL "${PIH_EXPECTED_CONTRACT_ID}" OR
       NOT PIH_LOCKED_threading_model EQUAL "${PIH_EXPECTED_THREADING_MODEL}" OR
       NOT PIH_LOCKED_scope EQUAL "${PIH_EXPECTED_SCOPE}" OR
       NOT PIH_LOCKED_cardinality EQUAL "${PIH_EXPECTED_CARDINALITY}")
        message(FATAL_ERROR "DeepSeek PP1 Lock capability graph drifted")
    endif()
endforeach()

string(JSON PIH_LOCKED_PACK_ID GET "${PIH_LOCK_JSON}" kernel_packs 0 pack_id)
string(JSON PIH_LOCKED_PACK_VERSION GET "${PIH_LOCK_JSON}" kernel_packs 0
       pack_version)
string(JSON PIH_LOCKED_PACK_ABI GET "${PIH_LOCK_JSON}" kernel_packs 0 pack_abi)
string(JSON PIH_LOCKED_PACK_ARCHITECTURE GET "${PIH_LOCK_JSON}" kernel_packs 0
       architecture)
string(JSON PIH_LOCKED_PACK_BINARY GET "${PIH_LOCK_JSON}" kernel_packs 0 binary)
string(JSON PIH_LOCKED_PACK_SHA256 GET "${PIH_LOCK_JSON}" kernel_packs 0
       binary_sha256_hex)
file(SHA256
     "${PIH_BUNDLE_ROOT}/lib/pih_kernels_deepseek_v4_sm89.so"
     PIH_OBSERVED_PACK_SHA256)
if(NOT PIH_LOCKED_PACK_ID STREQUAL "pih.kernels.deepseek-v4.sm89" OR
   NOT PIH_LOCKED_PACK_VERSION STREQUAL "1.0.0" OR
   NOT PIH_LOCKED_PACK_ABI STREQUAL "pih.deepseek-sm89-kernel-pack.v1" OR
   NOT PIH_LOCKED_PACK_ARCHITECTURE STREQUAL "sm89" OR
   NOT PIH_LOCKED_PACK_BINARY STREQUAL
       "lib/pih_kernels_deepseek_v4_sm89.so" OR
   NOT PIH_LOCKED_PACK_SHA256 STREQUAL "${PIH_OBSERVED_PACK_SHA256}")
    message(FATAL_ERROR "DeepSeek PP1 Lock Kernel Pack drifted")
endif()

string(JSON PIH_LOCKED_ENGINE_CAPABILITY GET "${PIH_LOCK_JSON}" engine
       capability_id)
string(JSON PIH_LOCKED_ENGINE_CONTRACT GET "${PIH_LOCK_JSON}" engine
       contract_id)
string(JSON PIH_LOCKED_ENGINE_EPOCH GET "${PIH_LOCK_JSON}" engine
       activation_epoch)
string(JSON PIH_LOCKED_ENGINE_CONFIGURATION_SCHEMA GET "${PIH_LOCK_JSON}"
       engine configuration schema)
foreach(PIH_CONFIGURATION_FIELD IN ITEMS
        device_ordinal
        host_spill_enabled
        expert_slot_count
        staging_extent_count
        artifact_poll_interval_ms
        attention_reserved_tokens_per_sequence
        maximum_prefill_chunk_tokens
        maximum_decode_sequences
        maximum_verify_sequences
        maximum_shard_bytes)
    string(JSON PIH_LOCKED_CONFIGURATION_${PIH_CONFIGURATION_FIELD}
           GET "${PIH_LOCK_JSON}" engine configuration
           ${PIH_CONFIGURATION_FIELD})
endforeach()
string(JSON PIH_LOCKED_GENERATION_ROOT GET "${PIH_LOCK_JSON}" engine
       configuration target_generation_root)
string(JSON PIH_LOCKED_ARTIFACT_ROOT GET "${PIH_LOCK_JSON}" engine
       configuration artifact_root_sha256_hex)
string(JSON PIH_LOCKED_SMOKE_MODEL GET "${PIH_LOCK_JSON}" engine
       smoke_http_body model)
string(JSON PIH_LOCKED_SMOKE_FIELD_COUNT LENGTH "${PIH_LOCK_JSON}" engine
       smoke_http_body)
string(JSON PIH_LOCKED_SMOKE_PROMPT GET "${PIH_LOCK_JSON}" engine
       smoke_http_body prompt)
string(JSON PIH_LOCKED_SMOKE_MAX_TOKENS GET "${PIH_LOCK_JSON}" engine
       smoke_http_body max_tokens)
foreach(PIH_SMOKE_FIELD IN ITEMS
        temperature
        top_p
        seed
        logprobs
        top_logprobs
        stream
        n
        echo
        user)
    string(JSON PIH_LOCKED_SMOKE_${PIH_SMOKE_FIELD} GET "${PIH_LOCK_JSON}"
           engine smoke_http_body ${PIH_SMOKE_FIELD})
endforeach()
string(JSON PIH_LOCKED_SMOKE_STOP_COUNT LENGTH "${PIH_LOCK_JSON}" engine
       smoke_http_body stop)
set(PIH_LOCKED_SMOKE_STOP "")
if(PIH_LOCKED_SMOKE_STOP_COUNT EQUAL 1)
    string(JSON PIH_LOCKED_SMOKE_STOP GET "${PIH_LOCK_JSON}" engine
           smoke_http_body stop 0)
endif()
if(NOT PIH_LOCKED_ENGINE_CAPABILITY STREQUAL "engine.primary.v1" OR
   NOT PIH_LOCKED_ENGINE_CONTRACT STREQUAL "pih.engine.v1" OR
   NOT PIH_LOCKED_ENGINE_EPOCH EQUAL 1 OR
   NOT PIH_LOCKED_ENGINE_CONFIGURATION_SCHEMA STREQUAL
       "pih.deepseek-v4-flash.pp1.v1" OR
   NOT PIH_LOCKED_CONFIGURATION_device_ordinal EQUAL 0 OR
   NOT PIH_LOCKED_CONFIGURATION_host_spill_enabled EQUAL 1 OR
   NOT PIH_LOCKED_CONFIGURATION_expert_slot_count EQUAL 2 OR
   NOT PIH_LOCKED_CONFIGURATION_staging_extent_count EQUAL 2 OR
   NOT PIH_LOCKED_CONFIGURATION_artifact_poll_interval_ms EQUAL 1000 OR
   NOT PIH_LOCKED_CONFIGURATION_attention_reserved_tokens_per_sequence EQUAL
       4096 OR
   NOT PIH_LOCKED_CONFIGURATION_maximum_prefill_chunk_tokens EQUAL 8192 OR
   NOT PIH_LOCKED_CONFIGURATION_maximum_decode_sequences EQUAL 4 OR
   NOT PIH_LOCKED_CONFIGURATION_maximum_verify_sequences EQUAL 4 OR
   NOT PIH_LOCKED_CONFIGURATION_maximum_shard_bytes EQUAL 8589934592 OR
   NOT PIH_LOCKED_GENERATION_ROOT STREQUAL
       "artifacts/deepseek-v4-flash-0731" OR
   NOT PIH_LOCKED_SMOKE_FIELD_COUNT EQUAL 13 OR
   NOT PIH_LOCKED_SMOKE_MODEL STREQUAL "deepseek-v4-flash" OR
   NOT PIH_LOCKED_SMOKE_PROMPT STREQUAL "pih" OR
   NOT PIH_LOCKED_SMOKE_MAX_TOKENS EQUAL 1 OR
   NOT PIH_LOCKED_SMOKE_temperature EQUAL 0 OR
   NOT PIH_LOCKED_SMOKE_top_p EQUAL 1 OR
   NOT PIH_LOCKED_SMOKE_STOP_COUNT EQUAL 1 OR
   NOT PIH_LOCKED_SMOKE_STOP STREQUAL "END" OR
   NOT PIH_LOCKED_SMOKE_seed EQUAL 7 OR
   NOT PIH_LOCKED_SMOKE_logprobs OR
   NOT PIH_LOCKED_SMOKE_top_logprobs EQUAL 2 OR
   PIH_LOCKED_SMOKE_stream OR
   NOT PIH_LOCKED_SMOKE_n EQUAL 1 OR
   PIH_LOCKED_SMOKE_echo OR
   NOT PIH_LOCKED_SMOKE_user STREQUAL "smoke")
    message(FATAL_ERROR "DeepSeek PP1 Lock engine binding drifted")
endif()
string(LENGTH "${PIH_LOCKED_ARTIFACT_ROOT}" PIH_LOCKED_ARTIFACT_ROOT_LENGTH)
if(NOT PIH_LOCKED_ARTIFACT_ROOT_LENGTH EQUAL 64 OR
   NOT PIH_LOCKED_ARTIFACT_ROOT MATCHES "^[0-9a-f]+$" OR
   PIH_LOCKED_ARTIFACT_ROOT STREQUAL
       "0000000000000000000000000000000000000000000000000000000000000000")
    message(FATAL_ERROR "DeepSeek PP1 Lock artifact root is not sealed")
endif()

file(
    GLOB_RECURSE PIH_BUNDLE_FILES
    LIST_DIRECTORIES FALSE
    RELATIVE "${PIH_BUNDLE_ROOT}"
    "${PIH_BUNDLE_ROOT}/*"
)
foreach(PIH_BUNDLE_FILE IN LISTS PIH_BUNDLE_FILES)
    string(TOLOWER "${PIH_BUNDLE_FILE}" PIH_BUNDLE_FILE_LOWER)
    if(PIH_BUNDLE_FILE_LOWER MATCHES "qwen|nccl|sm90|dspark")
        message(
            FATAL_ERROR
            "DeepSeek PP1 bundle contains forbidden feature artifact: ${PIH_BUNDLE_FILE}"
        )
    endif()
    list(FIND PIH_REQUIRED_BUNDLE_FILES "${PIH_BUNDLE_FILE}"
         PIH_ALLOWED_BUNDLE_POSITION)
    if(PIH_ALLOWED_BUNDLE_POSITION EQUAL -1)
        message(
            FATAL_ERROR
            "DeepSeek PP1 bundle contains an undeclared artifact: ${PIH_BUNDLE_FILE}"
        )
    endif()
endforeach()
