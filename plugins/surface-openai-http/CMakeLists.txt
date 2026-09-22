add_library(
    pih_plugin_surface_openai_http SHARED
    completion_request.cpp
    entrypoint.cpp
    http_request_framing.cpp
    http_response_framing.cpp
)
target_link_libraries(
    pih_plugin_surface_openai_http
    PRIVATE PIH::plugin_sdk pih_core_primitives
)
pih_assert_direct_link_allowlist(
    pih_plugin_surface_openai_http
    PIH::plugin_sdk
    pih_core_primitives
)
target_compile_features(pih_plugin_surface_openai_http PRIVATE cxx_std_20)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pih_plugin_surface_openai_http PRIVATE
        "LINKER:-z,defs"
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET pih_plugin_surface_openai_http APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(
    pih_plugin_surface_openai_http
    PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX ""
)
install(TARGETS pih_plugin_surface_openai_http
        LIBRARY DESTINATION lib COMPONENT pih-plugin-surface-openai-http
        RUNTIME DESTINATION lib COMPONENT pih-plugin-surface-openai-http)
