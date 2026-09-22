if(NOT TARGET pih_deepseek_v41_supervisor_request OR NOT TARGET pih-v41-rank-worker
   OR NOT "pih.kernels.deepseek-v41.sm103" IN_LIST PIH_ENABLED_PLUGINS)
    message(FATAL_ERROR "V4.1 model requires native rank worker, tokenizer, and SM103 Pack")
endif()
foreach(provider pih.platform.linux pih.backend.nvidia-cuda pih.transport.nccl)
    if(NOT "${provider}" IN_LIST PIH_ENABLED_PLUGINS)
        message(FATAL_ERROR "V4.1 rank runtime requires the ${provider} plugin in the build")
    endif()
endforeach()
find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(pih_plugin_model_deepseek_v41 SHARED
    "${PROJECT_SOURCE_DIR}/plugins/model-deepseek-v41/entrypoint.cpp"
    "${PROJECT_SOURCE_DIR}/plugins/model-deepseek-v41/supervisor_execution.cpp"
    "${PROJECT_SOURCE_DIR}/plugins/common/text_json.cpp")
target_link_libraries(pih_plugin_model_deepseek_v41 PRIVATE PIH::plugin_sdk
    pih_deepseek_v41_supervisor_request pih_deepseek_v41_tokenizer
    pih_deepseek_v41_rank_channel pih_deepseek_v41_token_ledger pih_core_primitives ICU::uc)
pih_assert_direct_link_allowlist(pih_plugin_model_deepseek_v41 PIH::plugin_sdk
    pih_deepseek_v41_supervisor_request pih_deepseek_v41_tokenizer
    pih_deepseek_v41_rank_channel pih_deepseek_v41_token_ledger pih_core_primitives ICU::uc)
target_link_options(pih_plugin_model_deepseek_v41 PRIVATE "LINKER:-z,defs"
    "LINKER:--version-script=${PROJECT_SOURCE_DIR}/plugins/model-deepseek-v41/exports.map")
set_property(TARGET pih_plugin_model_deepseek_v41 APPEND PROPERTY LINK_DEPENDS
    "${PROJECT_SOURCE_DIR}/plugins/model-deepseek-v41/exports.map")
set_target_properties(pih_plugin_model_deepseek_v41 PROPERTIES PREFIX ""
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
install(TARGETS pih_plugin_model_deepseek_v41 LIBRARY DESTINATION lib COMPONENT pih-plugin-model-deepseek-v41)
install(TARGETS pih-v41-rank-worker pih-v41-nccl-bootstrap pih-v41-supervisor
    RUNTIME DESTINATION bin COMPONENT pih-v41-commands)
install(FILES "${PROJECT_SOURCE_DIR}/plugins/common/DEEPSEEK_LICENSE"
    DESTINATION share/pih/licenses COMPONENT pih-plugin-model-deepseek-v41)

# Build real final link products without starting a worker or loading weights.
# Kernel Pack capabilities are resolved at runtime; keep this a build dependency,
# never a direct model-to-pack link that bypasses capability resolution.
add_custom_target(pih-v41-runtime-link-check)
add_dependencies(pih-v41-runtime-link-check
    pih_plugin_model_deepseek_v41 pih_kernels_deepseek_v41_sm103
    pih-v41-rank-worker pih-v41-nccl-bootstrap pih-v41-supervisor
    pih_plugin_platform_linux pih_plugin_backend_nvidia_cuda pih_plugin_transport_nccl)
