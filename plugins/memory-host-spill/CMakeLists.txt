add_library(
    pih_plugin_memory_host_spill SHARED
    entrypoint.cpp
)
# The capability provider submits through the bound CUDA async table. The
# historical CudaStagingLease value object is not part of that execution path;
# forcing it into this DSO imports an unrelated typed-copy implementation.
# Keep its separate source target available without publishing a legacy path.
target_link_libraries(
    pih_plugin_memory_host_spill
    PRIVATE PIH::plugin_sdk pih_core_primitives
)
pih_assert_direct_link_allowlist(
    pih_plugin_memory_host_spill
    PIH::plugin_sdk
    pih_core_primitives
)
target_compile_features(pih_plugin_memory_host_spill PRIVATE cxx_std_20)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pih_plugin_memory_host_spill PRIVATE
        "LINKER:-z,defs"
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET pih_plugin_memory_host_spill APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(
    pih_plugin_memory_host_spill
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
)
install(TARGETS pih_plugin_memory_host_spill
        LIBRARY DESTINATION lib COMPONENT pih-plugin-memory-host-spill
        RUNTIME DESTINATION lib COMPONENT pih-plugin-memory-host-spill)
