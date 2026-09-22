if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT PIH_ENABLE_CUDA)
    message(FATAL_ERROR "Native Qwen plugin requires Linux and CUDA")
endif()
find_package(ICU REQUIRED COMPONENTS uc i18n)
include(sources.cmake)
foreach(qwen_source IN LISTS qwen_sources)
    if(NOT IS_ABSOLUTE "${qwen_source}" OR NOT EXISTS "${qwen_source}")
        message(FATAL_ERROR "Qwen implementation requires existing absolute source paths: ${qwen_source}")
    endif()
endforeach()
# Do not instantiate offline targets for a model-only build.
add_library(pih_qwen3_native_impl STATIC ${qwen_sources})
target_include_directories(pih_qwen3_native_impl PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/generated")
target_compile_features(pih_qwen3_native_impl PUBLIC cxx_std_20)
add_dependencies(pih_qwen3_native_impl pih_qwen_cubin_identity)
target_sources(pih_qwen3_native_impl PRIVATE "${CMAKE_BINARY_DIR}/generated/pih/qwen_cubin_digests.cpp")
target_compile_options(pih_qwen3_native_impl PRIVATE -ffunction-sections -fdata-sections)
target_link_libraries(pih_qwen3_native_impl PUBLIC CUDA::cuda_driver CUDA::cudart CUDA::cublasLt pih_native_model_support pih_packed_plan_values)
pih_assert_direct_link_allowlist(pih_qwen3_native_impl CUDA::cuda_driver CUDA::cudart CUDA::cublasLt pih_native_model_support pih_packed_plan_values)
set_target_properties(pih_qwen3_native_impl PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
add_library(pih_plugin_model_qwen3 SHARED entrypoint.cpp tokenizer.cpp ../common/bytelevel_tokenizer.cpp)
target_link_libraries(pih_plugin_model_qwen3 PRIVATE PIH::plugin_sdk pih_qwen3_native_impl ICU::uc ICU::i18n)
pih_assert_direct_link_allowlist(pih_plugin_model_qwen3 PIH::plugin_sdk pih_qwen3_native_impl ICU::uc ICU::i18n)
target_link_options(pih_plugin_model_qwen3 PRIVATE "LINKER:-z,defs" "LINKER:--gc-sections"
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_property(TARGET pih_plugin_model_qwen3 APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_target_properties(pih_plugin_model_qwen3 PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
install(TARGETS pih_plugin_model_qwen3 LIBRARY DESTINATION lib)
if(PIH_BUILD_QWEN_QUALIFICATION_TOOLS)
    add_executable(pih-qwen-cuda-qualify EXCLUDE_FROM_ALL
        qualification_main.cpp
        qwen3_teacher_forced_metric_oracle.cpp
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/qwen_int4_cuda_fixture.cpp"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_driver_status.cpp"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_status.cpp")
    target_include_directories(pih-qwen-cuda-qualify PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih-qwen-cuda-qualify PRIVATE cxx_std_20)
    target_link_libraries(pih-qwen-cuda-qualify PRIVATE
        pih_core_primitives CUDA::cuda_driver CUDA::cudart)
    pih_assert_direct_link_allowlist(pih-qwen-cuda-qualify
        pih_core_primitives CUDA::cuda_driver CUDA::cudart)
endif()
