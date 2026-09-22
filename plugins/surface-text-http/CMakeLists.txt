add_library(pih_plugin_surface_text_http SHARED entrypoint.cpp ../surface-openai-http/http_request_framing.cpp)
target_link_libraries(pih_plugin_surface_text_http PRIVATE PIH::plugin_sdk pih_core_primitives)
pih_assert_direct_link_allowlist(pih_plugin_surface_text_http PIH::plugin_sdk pih_core_primitives)
target_link_options(pih_plugin_surface_text_http PRIVATE "LINKER:-z,defs")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pih_plugin_surface_text_http PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET pih_plugin_surface_text_http APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(pih_plugin_surface_text_http PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
install(TARGETS pih_plugin_surface_text_http LIBRARY DESTINATION lib COMPONENT pih-plugin-surface-text-http)
