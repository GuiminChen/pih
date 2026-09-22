# Model-owned host adapters calling startup-bound CUDA capability tables.
# No CUDA toolkit is required to compile this object target itself.
include_guard(GLOBAL)
file(GLOB PIH_CUDA_DEEPSEEK_BASE_SOURCES CONFIGURE_DEPENDS
    RELATIVE "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cpp")
pih_assert_deepseek_pp1_source_closure(PIH_CUDA_DEEPSEEK_BASE_SOURCES)
list(SORT PIH_CUDA_DEEPSEEK_BASE_SOURCES)
list(LENGTH PIH_CUDA_DEEPSEEK_BASE_SOURCES PIH_DEEPSEEK_PP1_CUDA_SOURCE_COUNT)
list(JOIN PIH_CUDA_DEEPSEEK_BASE_SOURCES "\n" PIH_DEEPSEEK_PP1_CUDA_SOURCE_CLOSURE)
string(SHA256 PIH_DEEPSEEK_PP1_CUDA_SOURCE_ROOT "${PIH_DEEPSEEK_PP1_CUDA_SOURCE_CLOSURE}")
if(NOT PIH_DEEPSEEK_PP1_CUDA_SOURCE_COUNT EQUAL 24 OR
   NOT PIH_DEEPSEEK_PP1_CUDA_SOURCE_ROOT STREQUAL
       "1c712c76584b56ca89f9b6c31d76ed26a05b49bf556d17e814c700784959b1c8")
    message(FATAL_ERROR "DeepSeek PP1 CUDA adapter source closure drifted: count=${PIH_DEEPSEEK_PP1_CUDA_SOURCE_COUNT}, root=${PIH_DEEPSEEK_PP1_CUDA_SOURCE_ROOT}")
endif()
set(pih_deepseek_cuda_sources ${PIH_CUDA_DEEPSEEK_BASE_SOURCES})
list(TRANSFORM pih_deepseek_cuda_sources PREPEND "${PROJECT_SOURCE_DIR}/")
add_library(pih_cuda_deepseek_v4_flash_impl OBJECT ${pih_deepseek_cuda_sources})
target_compile_features(pih_cuda_deepseek_v4_flash_impl PUBLIC cxx_std_20)
target_include_directories(pih_cuda_deepseek_v4_flash_impl PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_cuda_deepseek_v4_flash_impl PUBLIC pih_native_model_support)
pih_assert_direct_link_allowlist(pih_cuda_deepseek_v4_flash_impl pih_native_model_support)
target_compile_definitions(pih_cuda_deepseek_v4_flash_impl PRIVATE
    PIH_DEEPSEEK_DSPARK_DISABLED=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pih_cuda_deepseek_v4_flash_impl PRIVATE -ffunction-sections -fdata-sections)
endif()
set_target_properties(pih_cuda_deepseek_v4_flash_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON EXCLUDE_FROM_ALL TRUE
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
