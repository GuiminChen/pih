add_library(
    pih_plugin_execution_default SHARED
    entrypoint.cpp
    controller_provider.cpp
    $<TARGET_OBJECTS:pih_execution_default_impl>
)
target_link_libraries(
    pih_plugin_execution_default
    PRIVATE PIH::plugin_sdk pih_core_primitives pih_packed_plan_values
)
pih_assert_direct_link_allowlist(
    pih_plugin_execution_default
    PIH::plugin_sdk
    pih_core_primitives pih_packed_plan_values
)
target_compile_features(pih_plugin_execution_default PRIVATE cxx_std_20)
target_link_options(pih_plugin_execution_default PRIVATE "LINKER:-z,defs")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pih_plugin_execution_default PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET pih_plugin_execution_default APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(
    pih_plugin_execution_default
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
)
install(TARGETS pih_plugin_execution_default
        LIBRARY DESTINATION lib COMPONENT pih-plugin-execution-default
        RUNTIME DESTINATION lib COMPONENT pih-plugin-execution-default)
