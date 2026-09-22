if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Native DeepSeek artifact tools require Linux descriptor semantics")
endif()
add_library(pih_deepseek_identity_copy STATIC identity_copy.cpp shard_layout.cpp generation_layout.cpp tensor_inventory.cpp)
target_compile_features(pih_deepseek_identity_copy PRIVATE cxx_std_20)
target_link_libraries(pih_deepseek_identity_copy PRIVATE pih_native_model_support)
pih_assert_direct_link_allowlist(pih_deepseek_identity_copy pih_native_model_support)
add_library(pih_deepseek_generation_metadata STATIC
    generation_metadata.cpp
    prepare_generation.cpp
    source_payload.cpp
    source_artifact.cpp
    source_semantics.cpp
    source_inventory.cpp
    disposition_records.cpp
    generation_copy.cpp
    generation_verify.cpp
    generation_receipt.cpp
    generation_store.cpp
    generation_commit.cpp
    generation_pointer.cpp
    generation_activate.cpp
    active_generation.cpp
    generation_promote.cpp
    ../model-deepseek-v4-flash/deepseek_runtime_artifact_manifest.cpp
    ../model-deepseek-v4-flash/deepseek_runtime_records_manifest.cpp
    ../model-deepseek-v4-flash/deepseek_tensor_ownership_plan.cpp
    ../model-deepseek-v4-flash/deepseek_v4_config.cpp)
target_compile_features(pih_deepseek_generation_metadata PRIVATE cxx_std_20)
target_link_libraries(pih_deepseek_generation_metadata PRIVATE pih_native_model_support pih_deepseek_identity_copy)
pih_assert_direct_link_allowlist(pih_deepseek_generation_metadata pih_native_model_support pih_deepseek_identity_copy)
add_executable(pih-deepseek-artifact-verify
    verify_main.cpp)
target_compile_features(pih-deepseek-artifact-verify PRIVATE cxx_std_20)
target_link_libraries(pih-deepseek-artifact-verify PRIVATE pih_native_model_support pih_deepseek_identity_copy pih_deepseek_generation_metadata)
pih_assert_direct_link_allowlist(pih-deepseek-artifact-verify pih_native_model_support pih_deepseek_identity_copy pih_deepseek_generation_metadata)
install(TARGETS pih-deepseek-artifact-verify RUNTIME DESTINATION bin)
add_executable(pih-deepseek-artifact-prepare prepare_main.cpp)
target_compile_features(pih-deepseek-artifact-prepare PRIVATE cxx_std_20)
target_link_libraries(pih-deepseek-artifact-prepare PRIVATE pih_deepseek_generation_metadata pih_native_model_support)
pih_assert_direct_link_allowlist(pih-deepseek-artifact-prepare pih_deepseek_generation_metadata pih_native_model_support)
install(TARGETS pih-deepseek-artifact-prepare RUNTIME DESTINATION bin)
add_executable(pih-deepseek-generation-store store_main.cpp)
target_compile_features(pih-deepseek-generation-store PRIVATE cxx_std_20)
target_link_libraries(pih-deepseek-generation-store PRIVATE pih_deepseek_generation_metadata pih_native_model_support)
pih_assert_direct_link_allowlist(pih-deepseek-generation-store pih_deepseek_generation_metadata pih_native_model_support)
install(TARGETS pih-deepseek-generation-store RUNTIME DESTINATION bin)
