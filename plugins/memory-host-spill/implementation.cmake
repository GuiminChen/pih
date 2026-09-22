add_library(
    pih_memory_host_spill_impl OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/staging_lease.cpp"
)
target_compile_features(pih_memory_host_spill_impl PUBLIC cxx_std_20)
target_include_directories(
    pih_memory_host_spill_impl
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
pih_assert_direct_link_allowlist(
    pih_memory_host_spill_impl
)
set_target_properties(
    pih_memory_host_spill_impl PROPERTIES POSITION_INDEPENDENT_CODE ON
)

set_target_properties(pih_memory_host_spill_impl
    PROPERTIES EXCLUDE_FROM_ALL TRUE CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
