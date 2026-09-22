if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Native Qwen artifact conversion requires Linux file publication semantics")
endif()

# Exact offline closure, never the model engine/backend source set.
add_library(pih_qwen_artifact_impl STATIC
    ../model-qwen3/artifact/qwen3_int4_artifact_io.cpp
    ../model-qwen3/artifact/qwen3_int4_artifact_layout.cpp
    ../model-qwen3/artifact/qwen3_int4_artifact_metadata.cpp
    ../model-qwen3/artifact/qwen3_int4_disposition_plan.cpp
    ../model-qwen3/artifact/qwen3_int4_linear_shape_ledger.cpp
    ../model-qwen3/artifact/qwen3_int4_source_binding.cpp
    ../model-qwen3/artifact/qwen3_manifest.cpp
    ../model-qwen3/artifact/qwen3_source_artifact.cpp)
target_compile_features(pih_qwen_artifact_impl PUBLIC cxx_std_20)
target_compile_options(pih_qwen_artifact_impl PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(pih_qwen_artifact_impl PUBLIC
    pih_native_model_support pih_storage_verified_artifact_impl)
pih_assert_direct_link_allowlist(pih_qwen_artifact_impl
    pih_native_model_support pih_storage_verified_artifact_impl)
add_library(pih_qwen_conversion_impl STATIC
    qwen3_int4_artifact_writer.cpp
    qwen3_int4_conversion_stream.cpp
    qwen3_int4_converter.cpp
    qwen3_int4_reference_quantizer.cpp
    qwen3_int4_safetensors_source.cpp
    qwen3_int4_tensor_payload.cpp)
target_compile_features(pih_qwen_conversion_impl PUBLIC cxx_std_20)
target_compile_options(pih_qwen_conversion_impl PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(pih_qwen_conversion_impl PUBLIC pih_qwen_artifact_impl)
pih_assert_direct_link_allowlist(pih_qwen_conversion_impl pih_qwen_artifact_impl)
add_executable(pih-qwen-int4-convert convert_main.cpp)
target_link_libraries(pih-qwen-int4-convert PRIVATE pih_qwen_conversion_impl)
pih_assert_direct_link_allowlist(pih-qwen-int4-convert pih_qwen_conversion_impl)
install(TARGETS pih-qwen-int4-convert RUNTIME DESTINATION bin)
add_executable(pih-qwen-int4-verify verify_main.cpp)
target_link_libraries(pih-qwen-int4-verify PRIVATE pih_qwen_artifact_impl)
pih_assert_direct_link_allowlist(pih-qwen-int4-verify pih_qwen_artifact_impl)
install(TARGETS pih-qwen-int4-verify RUNTIME DESTINATION bin)
add_executable(pih-qwen-source-verify verify_source_main.cpp)
target_link_libraries(pih-qwen-source-verify PRIVATE pih_qwen_artifact_impl)
pih_assert_direct_link_allowlist(pih-qwen-source-verify pih_qwen_artifact_impl)
install(TARGETS pih-qwen-source-verify RUNTIME DESTINATION bin)
