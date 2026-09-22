add_library(pih_output_burst_credits STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/output_burst_credit_pool.cpp")
target_compile_features(pih_output_burst_credits PUBLIC cxx_std_20)
target_include_directories(pih_output_burst_credits PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(pih_output_burst_credits PRIVATE pih_core_primitives)
pih_assert_direct_link_allowlist(pih_output_burst_credits pih_core_primitives)
set_target_properties(pih_output_burst_credits PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Pure packed-plan values are shared by the execution provider and the model
# adapter. Controller state, queues and scheduling remain provider-owned.
add_library(pih_packed_plan_values STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/packed_token_plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/packed_token_metadata_arena.cpp")
target_compile_features(pih_packed_plan_values PUBLIC cxx_std_20)
target_include_directories(pih_packed_plan_values PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(pih_packed_plan_values PUBLIC pih_core_primitives)
pih_assert_direct_link_allowlist(pih_packed_plan_values pih_core_primitives)
set_target_properties(pih_packed_plan_values PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

add_library(
    pih_execution_default_impl OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/phase_tier_scheduler.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controller_sequence.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controller_mailbox.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controller_request_arena.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/output_slot_arena.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/output_burst_credit_pool.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controller_ingress.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controller_runtime.cpp"
)
target_compile_features(pih_execution_default_impl PUBLIC cxx_std_20)
target_include_directories(
    pih_execution_default_impl
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
pih_assert_direct_link_allowlist(
    pih_execution_default_impl
)
set_target_properties(
    pih_execution_default_impl PROPERTIES POSITION_INDEPENDENT_CODE ON
)

set_target_properties(pih_execution_default_impl
    PROPERTIES EXCLUDE_FROM_ALL TRUE CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
