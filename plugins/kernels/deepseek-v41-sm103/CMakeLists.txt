if(NOT PIH_V41_PACK_SOURCES)
    message(FATAL_ERROR "V4.1 native kernel ownership was not configured")
endif()
add_library(pih_kernels_deepseek_v41_sm103 SHARED kernel_pack.cpp ${PIH_V41_PACK_SOURCES})
target_include_directories(pih_kernels_deepseek_v41_sm103 PRIVATE
    "${PROJECT_SOURCE_DIR}/plugins/model-deepseek-v41")
target_link_libraries(pih_kernels_deepseek_v41_sm103 PRIVATE PIH::plugin_sdk
    pih_core_primitives pih_deepseek_v41_sampling_host CUDA::cudart)
pih_assert_direct_link_allowlist(pih_kernels_deepseek_v41_sm103 PIH::plugin_sdk
    pih_core_primitives pih_deepseek_v41_sampling_host CUDA::cudart)
target_compile_options(pih_kernels_deepseek_v41_sm103 PRIVATE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false>
    $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--pre-include=${CMAKE_CURRENT_SOURCE_DIR}/device_target.cuh>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,Clang>:SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/device_target.cuh\">")
target_compile_features(pih_kernels_deepseek_v41_sm103 PRIVATE cxx_std_20)
target_link_options(pih_kernels_deepseek_v41_sm103 PRIVATE "LINKER:-z,defs"
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_property(TARGET pih_kernels_deepseek_v41_sm103 APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_target_properties(pih_kernels_deepseek_v41_sm103 PROPERTIES PREFIX ""
    CUDA_ARCHITECTURES "103-real" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
install(TARGETS pih_kernels_deepseek_v41_sm103 LIBRARY DESTINATION lib COMPONENT pih-kernel-deepseek-v41-sm103)
