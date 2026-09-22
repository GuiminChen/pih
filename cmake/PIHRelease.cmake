# Explicit production distribution. No tests, reference runtimes or build trees.
install(CODE "if(EXISTS \"\${CMAKE_INSTALL_PREFIX}/NATIVE-INSTALL.json\")
    message(FATAL_ERROR \"Release installation already exists; choose a new prefix\")
endif()" COMPONENT pih-release)
set(pih_release_targets)
set(pih_release_candidates pih-worker pih-cli
    pih_plugin_platform_linux pih_plugin_backend_nvidia_cuda
    pih_plugin_execution_default pih_plugin_memory_host_spill
    pih_plugin_storage_verified_artifact pih_plugin_transport_nccl
    pih_plugin_surface_text_http pih_plugin_surface_openai_http pih_plugin_surface_health)
if(TARGET pih_plugin_model_qwen3)
    list(APPEND pih_release_candidates pih_plugin_model_qwen3
        pih_kernels_qwen3_sm89 pih_kernels_qwen3_sm90 pih-qwen-int4-convert
        pih-qwen-int4-verify pih-qwen-source-verify pih-qwen-format pih-tokenize)
endif()
if(TARGET pih_plugin_model_deepseek_v4_flash)
    list(APPEND pih_release_candidates pih_plugin_model_deepseek_v4_flash
        pih_kernels_deepseek_v4_sm89 pih-deepseek-artifact-prepare
        pih-deepseek-artifact-verify pih-deepseek-generation-store
        pih-tokenize pih-deepseek-format pih-deepseek-semantic-verify)
endif()
if(TARGET pih_plugin_model_deepseek_v41)
    list(APPEND pih_release_candidates pih_plugin_model_deepseek_v41
        pih_kernels_deepseek_v41_sm103 pih-v41-rank-worker pih-v41-nccl-bootstrap
        pih-v41-supervisor pih-v41-convert pih-v41-reshard pih-v41-token-map)
endif()
list(REMOVE_DUPLICATES pih_release_candidates)
foreach(candidate IN LISTS pih_release_candidates)
    if(TARGET ${candidate})
        list(APPEND pih_release_targets ${candidate})
        install(TARGETS ${candidate}
            RUNTIME DESTINATION bin COMPONENT pih-release
            LIBRARY DESTINATION lib COMPONENT pih-release)
    endif()
endforeach()
add_custom_target(pih-release DEPENDS ${pih_release_targets})
foreach(sm 89 90)
    if(TARGET pih_kernels_qwen3_sm${sm})
        install(FILES
            "${CMAKE_BINARY_DIR}/cubin/sm_${sm}/qwen_bf16_primitives.cubin"
            "${CMAKE_BINARY_DIR}/cubin/sm_${sm}/qwen_bf16_primitives.cubin.json"
            DESTINATION lib/qwen3-sm${sm}/sm_${sm} COMPONENT pih-release)
    endif()
endforeach()
foreach(plugin IN LISTS PIH_ENABLED_PLUGINS)
    string(REGEX REPLACE "^pih\\." "" directory "${plugin}")
    string(REPLACE "." "-" directory "${directory}")
    if(EXISTS "${PROJECT_SOURCE_DIR}/plugins/${directory}/manifest.json")
        install(FILES "${PROJECT_SOURCE_DIR}/plugins/${directory}/manifest.json"
            DESTINATION "share/pih/plugins/${plugin}" COMPONENT pih-release)
    endif()
endforeach()
install(FILES LICENSE THIRD_PARTY_NOTICES.md plugins/common/DEEPSEEK_LICENSE
    DESTINATION share/pih/licenses COMPONENT pih-release)
install(PROGRAMS deploy/run-native.sh deploy/run-v41-native.sh
    DESTINATION deploy COMPONENT pih-release)
install(FILES deploy/native.py DESTINATION deploy COMPONENT pih-release)
install(FILES deploy/development/deepseek-v4-flash-rtx4090d-pp1.lock
    DESTINATION deploy/development COMPONENT pih-release)
install(FILES cmake/PIHSealV41.cmake cmake/PIHSealDevelopmentLock.cmake
    cmake/PIHStageDeepSeekPp1Artifact.cmake cmake/PIHVerifyDeepSeekPp1Bundle.cmake
    DESTINATION cmake COMPONENT pih-release)
install(SCRIPT "${CMAKE_CURRENT_LIST_DIR}/PIHWriteReleaseInventory.cmake"
    COMPONENT pih-release)
