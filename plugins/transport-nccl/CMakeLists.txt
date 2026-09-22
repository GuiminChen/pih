if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT PIH_ENABLE_CUDA OR
   NOT PIH_BUILD_NATIVE_ENGRAM_NCCL OR NOT PIH_NATIVE_ENGRAM_NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL transport plugin requires Linux, CUDA and pinned NCCL")
endif()
add_library(pih_plugin_transport_nccl SHARED entrypoint.cpp)
target_compile_features(pih_plugin_transport_nccl PRIVATE cxx_std_20)
target_include_directories(pih_plugin_transport_nccl PRIVATE
    "${PIH_NATIVE_ENGRAM_NCCL_INCLUDE_DIR}")
target_link_libraries(pih_plugin_transport_nccl PRIVATE PIH::plugin_sdk
    CUDA::cuda_driver CUDA::cudart "${PIH_NATIVE_ENGRAM_NCCL_LIBRARY}")
pih_assert_direct_link_allowlist(pih_plugin_transport_nccl PIH::plugin_sdk
    CUDA::cuda_driver CUDA::cudart "${PIH_NATIVE_ENGRAM_NCCL_LIBRARY}")
target_link_options(pih_plugin_transport_nccl PRIVATE "LINKER:-z,defs")
set_target_properties(pih_plugin_transport_nccl PROPERTIES PREFIX ""
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
install(TARGETS pih_plugin_transport_nccl LIBRARY DESTINATION lib
    COMPONENT pih-plugin-transport-nccl)
