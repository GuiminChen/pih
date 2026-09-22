add_library(pih_kernels_deepseek_v4_sm90 SHARED
    ../deepseek-v4-sm89/kernel_pack.cpp
    $<TARGET_OBJECTS:pih_deepseek_sm90_kernels_impl>
    $<TARGET_OBJECTS:pih_deepseek_sm90_kernel_contract_impl>)
target_link_libraries(pih_kernels_deepseek_v4_sm90 PRIVATE
    PIH::plugin_sdk pih_core_primitives CUDA::cuda_driver CUDA::cudart)
pih_assert_direct_link_allowlist(pih_kernels_deepseek_v4_sm90
    PIH::plugin_sdk pih_core_primitives CUDA::cuda_driver CUDA::cudart)
target_compile_features(pih_kernels_deepseek_v4_sm90 PRIVATE cxx_std_20)
target_compile_definitions(pih_kernels_deepseek_v4_sm90 PRIVATE
    PIH_DEEPSEEK_DSPARK_DISABLED=1
    PIH_DEEPSEEK_PACK_ID="pih.kernels.deepseek-v4.sm90"
    PIH_DEEPSEEK_PACK_ABI="pih.deepseek-sm90-kernel-pack.v1"
    PIH_DEEPSEEK_PACK_ARCH="sm90")
set_target_properties(pih_kernels_deepseek_v4_sm90 PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX ""
    CUDA_ARCHITECTURES "90-real")
install(TARGETS pih_kernels_deepseek_v4_sm90
    LIBRARY DESTINATION lib COMPONENT pih-kernel-deepseek-v4-sm90
    RUNTIME DESTINATION lib COMPONENT pih-kernel-deepseek-v4-sm90)
