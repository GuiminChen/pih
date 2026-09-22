add_library(
    pih_plugin_backend_nvidia_cuda SHARED
    entrypoint.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_allocator.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_memory_copier.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_driver_status.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_status.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/cuda/nvidia_pinned_host_allocator.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/cuda/nvidia_runtime_resource_driver.cpp
)
target_link_libraries(
    pih_plugin_backend_nvidia_cuda
    PRIVATE
        PIH::plugin_sdk
        pih_core_primitives
        CUDA::cuda_driver
        CUDA::cudart
)
pih_assert_direct_link_allowlist(
    pih_plugin_backend_nvidia_cuda
    PIH::plugin_sdk
    pih_core_primitives
    CUDA::cuda_driver
    CUDA::cudart
)
target_compile_features(pih_plugin_backend_nvidia_cuda PRIVATE cxx_std_20)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pih_plugin_backend_nvidia_cuda PRIVATE
        "LINKER:-z,defs"
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET pih_plugin_backend_nvidia_cuda APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(
    pih_plugin_backend_nvidia_cuda
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
)
install(TARGETS pih_plugin_backend_nvidia_cuda
        LIBRARY DESTINATION lib COMPONENT pih-plugin-backend-nvidia-cuda
        RUNTIME DESTINATION lib COMPONENT pih-plugin-backend-nvidia-cuda)
