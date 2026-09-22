add_library(
    pih_kernels_deepseek_v4_sm89 SHARED
    kernel_pack.cpp
    $<TARGET_OBJECTS:pih_deepseek_sm89_kernels_impl>
    $<TARGET_OBJECTS:pih_deepseek_sm89_kernel_contract_impl>
)
target_link_libraries(
    pih_kernels_deepseek_v4_sm89
    PRIVATE PIH::plugin_sdk pih_core_primitives CUDA::cuda_driver CUDA::cudart
)
pih_assert_direct_link_allowlist(
    pih_kernels_deepseek_v4_sm89
    PIH::plugin_sdk
    pih_core_primitives
    CUDA::cuda_driver
    CUDA::cudart
)
target_compile_features(pih_kernels_deepseek_v4_sm89 PRIVATE cxx_std_20)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # A capability pack must resolve every native launcher at build time.
    target_link_options(pih_kernels_deepseek_v4_sm89 PRIVATE "LINKER:-z,defs"
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET pih_kernels_deepseek_v4_sm89 APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
target_compile_definitions(
    pih_kernels_deepseek_v4_sm89
    PRIVATE PIH_DEEPSEEK_DSPARK_DISABLED=1
)
set_target_properties(
    pih_kernels_deepseek_v4_sm89
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
        CUDA_ARCHITECTURES "89-real"
)
install(TARGETS pih_kernels_deepseek_v4_sm89
        LIBRARY DESTINATION lib COMPONENT pih-kernel-deepseek-v4-sm89
        RUNTIME DESTINATION lib COMPONENT pih-kernel-deepseek-v4-sm89)
