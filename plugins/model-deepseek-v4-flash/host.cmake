# Native PP1 model implementation. No monolithic/optional-feature source scan.
include_guard(GLOBAL)

set(PIH_DEEPSEEK_MULTIPROCESS_SOURCE_PATTERN
    "deepseek_rank_(artifact_|capacity_plan_instance|control_codec|exec_identity_verifier|materialization_(allocation_census|completion|exchange|grant|warmup)|model_startup_plan|post_|process_|scm_rights_|serving_|spawn_|startup_barrier|worker_)")
function(pih_assert_deepseek_pp1_source_closure PIH_SOURCE_LIST)
    foreach(PIH_PP1_SOURCE IN LISTS ${PIH_SOURCE_LIST})
        string(TOLOWER "${PIH_PP1_SOURCE}" PIH_PP1_SOURCE_LOWER)
        if(PIH_PP1_SOURCE_LOWER MATCHES
            "boundary|component_sha256|controller_artifact_catalog|dspark|mxfp4_decode|nccl|pipeline_wire|qwen|runtime_artifact_evidence|sm90|${PIH_DEEPSEEK_MULTIPROCESS_SOURCE_PATTERN}")
            message(FATAL_ERROR "DeepSeek PP1 target contains forbidden source: ${PIH_PP1_SOURCE}")
        endif()
    endforeach()
endfunction()

file(GLOB PIH_DEEPSEEK_MODEL_SOURCES CONFIGURE_DEPENDS
    RELATIVE "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_LIST_DIR}/*deepseek*.cpp")
pih_assert_deepseek_pp1_source_closure(PIH_DEEPSEEK_MODEL_SOURCES)
list(SORT PIH_DEEPSEEK_MODEL_SOURCES COMPARE STRING CASE SENSITIVE ORDER ASCENDING)
list(LENGTH PIH_DEEPSEEK_MODEL_SOURCES PIH_DEEPSEEK_PP1_MODEL_SOURCE_COUNT)
list(JOIN PIH_DEEPSEEK_MODEL_SOURCES "\n" PIH_DEEPSEEK_PP1_MODEL_SOURCE_CLOSURE)
string(SHA256 PIH_DEEPSEEK_PP1_MODEL_SOURCE_ROOT "${PIH_DEEPSEEK_PP1_MODEL_SOURCE_CLOSURE}")
if(NOT PIH_DEEPSEEK_PP1_MODEL_SOURCE_COUNT EQUAL 161 OR
   NOT PIH_DEEPSEEK_PP1_MODEL_SOURCE_ROOT STREQUAL
       "8c47647f1a5f72ef1d35df8cbfe03733dd19b93a641ae778c44243b75a117a27")
    message(FATAL_ERROR "DeepSeek PP1 model source closure drifted: count=${PIH_DEEPSEEK_PP1_MODEL_SOURCE_COUNT}, root=${PIH_DEEPSEEK_PP1_MODEL_SOURCE_ROOT}")
endif()

set(pih_deepseek_host_sources ${PIH_DEEPSEEK_MODEL_SOURCES})
list(TRANSFORM pih_deepseek_host_sources PREPEND "${PROJECT_SOURCE_DIR}/")
add_library(pih_model_deepseek_v4_flash_impl OBJECT ${pih_deepseek_host_sources})
target_compile_features(pih_model_deepseek_v4_flash_impl PUBLIC cxx_std_20)
target_include_directories(pih_model_deepseek_v4_flash_impl
    PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
pih_assert_direct_link_allowlist(pih_model_deepseek_v4_flash_impl)
target_compile_definitions(pih_model_deepseek_v4_flash_impl PRIVATE
    PIH_DEEPSEEK_DSPARK_DISABLED=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pih_model_deepseek_v4_flash_impl PRIVATE -ffunction-sections -fdata-sections)
endif()
set_target_properties(pih_model_deepseek_v4_flash_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON EXCLUDE_FROM_ALL TRUE
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
