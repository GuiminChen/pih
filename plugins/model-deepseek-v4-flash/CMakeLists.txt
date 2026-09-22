find_package(ICU REQUIRED COMPONENTS uc i18n)
include("${CMAKE_CURRENT_LIST_DIR}/../kernels/deepseek-v4-sm89/validation.cmake")
add_library(
    pih_plugin_model_deepseek_v4_flash SHARED
    entrypoint.cpp
    generation.cpp
    text_completion.cpp
    ../../src/backend/cuda/cuda_runtime_resources.cpp
    ../../src/backend/cuda/owned_cuda_runtime_resources.cpp
    ../common/bytelevel_tokenizer.cpp
    ../common/deepseek_semantic_artifact.cpp
    ../common/deepseek_encoding.cpp
    ../common/deepseek_completion_decoder.cpp
    ../common/text_json.cpp
    $<TARGET_OBJECTS:pih_model_deepseek_v4_flash_impl>
    $<TARGET_OBJECTS:pih_cuda_deepseek_v4_flash_impl>
)
target_link_libraries(
    pih_plugin_model_deepseek_v4_flash
    PRIVATE
        PIH::plugin_sdk
        pih_native_model_support
        pih_output_burst_credits
        pih_deepseek_launch_contracts
        ICU::uc ICU::i18n
)
pih_assert_direct_link_allowlist(
    pih_plugin_model_deepseek_v4_flash
    PIH::plugin_sdk
    pih_native_model_support
    pih_output_burst_credits
    pih_deepseek_launch_contracts
    ICU::uc ICU::i18n
)
target_compile_features(pih_plugin_model_deepseek_v4_flash PRIVATE cxx_std_20)
target_compile_definitions(
    pih_plugin_model_deepseek_v4_flash
    PRIVATE
        PIH_DEEPSEEK_DSPARK_DISABLED=1
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(
        pih_plugin_model_deepseek_v4_flash
        PRIVATE "LINKER:--gc-sections" "LINKER:-z,defs"
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
    set_property(TARGET pih_plugin_model_deepseek_v4_flash APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(
    pih_plugin_model_deepseek_v4_flash
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
)
install(TARGETS pih_plugin_model_deepseek_v4_flash
        LIBRARY DESTINATION lib COMPONENT pih-plugin-model-deepseek-v4-flash
        RUNTIME DESTINATION lib COMPONENT pih-plugin-model-deepseek-v4-flash)
install(FILES ../common/DEEPSEEK_LICENSE DESTINATION share/pih/licenses
        COMPONENT pih-plugin-model-deepseek-v4-flash)
