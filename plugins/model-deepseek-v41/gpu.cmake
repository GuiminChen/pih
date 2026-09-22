# Included after CUDA language/toolkit admission. Host targets are declared by
# this model's CMakeLists.txt. This file does not register a model or Kernel Pack.
# Native V4.1 primitives are not a registered model/kernel-pack capability.
# Keep their build independent of the legacy monolithic implementation.
# CUDA availability alone must not add this model to another deployment's
# default build. Explicit worker/primitive targets bring in their dependency
# closure; deploy/run-v41-native.sh names the required executable targets.
add_library(pih_deepseek_v41_fp8_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/linear_fp8.cpp" "${CMAKE_CURRENT_LIST_DIR}/linear_fp4.cpp" "${CMAKE_CURRENT_LIST_DIR}/linear_fp8.cu")
target_include_directories(pih_deepseek_v41_fp8_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_fp8_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_fp8_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_fp8_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_fp8_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_engram_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/engram_launch.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/engram_completion.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/engram_cuda.cu")
target_include_directories(pih_deepseek_v41_engram_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_engram_cuda PRIVATE pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_engram_cuda pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_engram_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_engram_cuda PRIVATE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
add_library(pih_deepseek_v41_rope_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/rope_launch.cpp" "${CMAKE_CURRENT_LIST_DIR}/rope_cuda.cu")
target_include_directories(pih_deepseek_v41_rope_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_rope_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_rope_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_rope_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_rope_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_rope_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_sampling_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/sampling.cu")
target_include_directories(pih_deepseek_v41_sampling_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_sampling_cuda PRIVATE pih_deepseek_v41_sampling_host pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_sampling_cuda pih_deepseek_v41_sampling_host pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_sampling_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_sampling_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_sampling_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(pih_deepseek_v41_weight_upload STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/weight_upload.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/weight_memory_owner.cpp")
    target_include_directories(pih_deepseek_v41_weight_upload PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_weight_upload PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_weight_upload PRIVATE pih_deepseek_v41_weight_files pih_core_primitives CUDA::cudart)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_weight_upload pih_deepseek_v41_weight_files pih_core_primitives CUDA::cudart)
    set_target_properties(pih_deepseek_v41_weight_upload PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_uploaded_bindings STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/uploaded_bindings.cpp")
    target_include_directories(pih_deepseek_v41_uploaded_bindings PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_uploaded_bindings PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_uploaded_bindings PRIVATE pih_deepseek_v41_weight_upload
        pih_deepseek_v41_token_embedding_cuda pih_deepseek_v41_model_head_cuda
        pih_deepseek_v41_expert_dispatch_cuda pih_deepseek_v41_shared_expert_cuda
        pih_deepseek_v41_mhc_cuda pih_deepseek_v41_router_cuda pih_deepseek_v41_engram_cuda
        pih_deepseek_v41_attention_prepare pih_deepseek_v41_attention_assemble_cuda
        pih_deepseek_v41_compressed_prepare pih_deepseek_v41_indexer_chain pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_uploaded_bindings pih_deepseek_v41_weight_upload
        pih_deepseek_v41_token_embedding_cuda pih_deepseek_v41_model_head_cuda
        pih_deepseek_v41_expert_dispatch_cuda pih_deepseek_v41_shared_expert_cuda
        pih_deepseek_v41_mhc_cuda pih_deepseek_v41_router_cuda pih_deepseek_v41_engram_cuda
        pih_deepseek_v41_attention_prepare pih_deepseek_v41_attention_assemble_cuda
        pih_deepseek_v41_compressed_prepare pih_deepseek_v41_indexer_chain pih_core_primitives)
    set_target_properties(pih_deepseek_v41_uploaded_bindings PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_ffn_plan STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/ffn_plan.cpp")
    target_include_directories(pih_deepseek_v41_ffn_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_ffn_plan PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_ffn_plan PRIVATE pih_deepseek_v41_uploaded_bindings
        pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_residual pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_ffn_plan pih_deepseek_v41_uploaded_bindings
        pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_residual pih_core_primitives)
    set_target_properties(pih_deepseek_v41_ffn_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_attention_prepare_plan STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/attention_prepare_plan.cpp")
    target_include_directories(pih_deepseek_v41_attention_prepare_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_attention_prepare_plan PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_attention_prepare_plan PRIVATE pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_prepare_plan pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    set_target_properties(pih_deepseek_v41_attention_prepare_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_attention_output_plan STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/attention_output_plan.cpp")
    target_include_directories(pih_deepseek_v41_attention_output_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_attention_output_plan PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_attention_output_plan PRIVATE pih_deepseek_v41_uploaded_bindings
        pih_deepseek_v41_attention_prepare pih_deepseek_v41_attention_residual pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_output_plan pih_deepseek_v41_uploaded_bindings
        pih_deepseek_v41_attention_prepare pih_deepseek_v41_attention_residual pih_core_primitives)
    set_target_properties(pih_deepseek_v41_attention_output_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_engram_plan STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/engram_plan.cpp")
    target_include_directories(pih_deepseek_v41_engram_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_engram_plan PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_engram_plan PRIVATE pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_engram_plan pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    set_target_properties(pih_deepseek_v41_engram_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_compressor_plan STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/compressor_plan.cpp")
    target_include_directories(pih_deepseek_v41_compressor_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_compressor_plan PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_compressor_plan PRIVATE pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_compressor_plan pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    set_target_properties(pih_deepseek_v41_compressor_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_indexer_plan STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/indexer_plan.cpp")
    target_include_directories(pih_deepseek_v41_indexer_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_indexer_plan PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_indexer_plan PRIVATE pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_indexer_plan pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    set_target_properties(pih_deepseek_v41_indexer_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_library(pih_deepseek_v41_token_input_upload STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/token_input_upload.cpp")
target_include_directories(pih_deepseek_v41_token_input_upload PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_token_input_upload PRIVATE pih_core_primitives pih_deepseek_v41_engram_hash CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_token_input_upload pih_core_primitives pih_deepseek_v41_engram_hash CUDA::cudart)
target_compile_features(pih_deepseek_v41_token_input_upload PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_token_input_upload PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_token_embedding_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/token_embedding.cpp" "${CMAKE_CURRENT_LIST_DIR}/token_embedding.cu")
target_include_directories(pih_deepseek_v41_token_embedding_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_token_embedding_cuda PRIVATE pih_deepseek_v41_mhc_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_token_embedding_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_token_embedding_cuda PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_token_embedding_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_model_head_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/model_head.cpp" "${CMAKE_CURRENT_LIST_DIR}/model_head.cu")
target_include_directories(pih_deepseek_v41_model_head_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_model_head_cuda PRIVATE pih_deepseek_v41_mhc_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_model_head_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_model_head_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_model_head_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_model_head_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_norm_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/norm_launch.cpp" "${CMAKE_CURRENT_LIST_DIR}/norm_cuda.cu")
target_include_directories(pih_deepseek_v41_norm_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_norm_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_norm_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_norm_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_norm_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_indexer_select_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/indexer_select.cpp" "${CMAKE_CURRENT_LIST_DIR}/indexer_select.cu")
target_include_directories(pih_deepseek_v41_indexer_select_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_indexer_select_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_indexer_select_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_indexer_select_cuda PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_indexer_select_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_indexer_score_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/indexer_score.cpp" "${CMAKE_CURRENT_LIST_DIR}/indexer_score.cu")
target_include_directories(pih_deepseek_v41_indexer_score_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_indexer_score_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_indexer_score_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_indexer_score_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_indexer_score_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_indexer_score_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_indexer_input_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/indexer_input.cpp" "${CMAKE_CURRENT_LIST_DIR}/indexer_input.cu")
target_include_directories(pih_deepseek_v41_indexer_input_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_indexer_input_cuda PRIVATE pih_deepseek_v41_fp8_cuda pih_deepseek_v41_rope_cuda pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_indexer_input_cuda pih_deepseek_v41_fp8_cuda pih_deepseek_v41_rope_cuda pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_indexer_input_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_indexer_input_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_indexer_input_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_compressed_kv_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/compressed_kv.cpp" "${CMAKE_CURRENT_LIST_DIR}/compressed_kv.cu")
target_include_directories(pih_deepseek_v41_compressed_kv_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_compressed_kv_cuda PRIVATE pih_deepseek_v41_rope_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_compressed_kv_cuda pih_deepseek_v41_rope_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_compressed_kv_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_compressed_kv_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_compressed_kv_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_compressor_pool_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/compressor_pool.cpp" "${CMAKE_CURRENT_LIST_DIR}/compressor_pool.cu")
target_include_directories(pih_deepseek_v41_compressor_pool_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_compressor_pool_cuda PRIVATE pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_compressor_pool_cuda pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_compressor_pool_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_compressor_pool_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_compressor_pool_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_attention_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/sparse_attention.cpp" "${CMAKE_CURRENT_LIST_DIR}/sparse_attention.cu")
target_include_directories(pih_deepseek_v41_attention_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_attention_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_attention_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_attention_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_attention_output_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/attention_output.cpp" "${CMAKE_CURRENT_LIST_DIR}/attention_output.cu")
target_include_directories(pih_deepseek_v41_attention_output_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_output_cuda PRIVATE pih_deepseek_v41_attention_cuda
    pih_deepseek_v41_rope_cuda pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_output_cuda pih_deepseek_v41_attention_cuda
    pih_deepseek_v41_rope_cuda pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_attention_output_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_attention_output_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_attention_output_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_window_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/window_kv.cpp" "${CMAKE_CURRENT_LIST_DIR}/window_kv.cu")
target_include_directories(pih_deepseek_v41_window_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_window_cuda PRIVATE pih_deepseek_v41_norm_cuda
    pih_deepseek_v41_rope_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_window_cuda pih_deepseek_v41_norm_cuda
    pih_deepseek_v41_rope_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_window_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_window_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_window_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_attention_input STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/attention_input.cpp")
target_include_directories(pih_deepseek_v41_attention_input PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_input PRIVATE pih_deepseek_v41_fp8_cuda
    pih_deepseek_v41_norm_cuda pih_deepseek_v41_rope_cuda pih_deepseek_v41_window_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_input pih_deepseek_v41_fp8_cuda
    pih_deepseek_v41_norm_cuda pih_deepseek_v41_rope_cuda pih_deepseek_v41_window_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_attention_input PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_attention_input PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_mhc_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/mhc_launch.cpp" "${CMAKE_CURRENT_LIST_DIR}/mhc_cuda.cu")
target_include_directories(pih_deepseek_v41_mhc_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_mhc_cuda PRIVATE pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_mhc_cuda pih_deepseek_v41_norm_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_mhc_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_mhc_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_mhc_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
set_target_properties(pih_deepseek_v41_engram_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_attention_assemble_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/attention_assemble.cpp" "${CMAKE_CURRENT_LIST_DIR}/attention_assemble.cu")
target_include_directories(pih_deepseek_v41_attention_assemble_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_assemble_cuda PRIVATE pih_deepseek_v41_attention_cuda pih_deepseek_v41_attention_output_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_attention_cuda pih_deepseek_v41_attention_output_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_attention_assemble_cuda PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_attention_assemble_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_expert_dispatch_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/expert_dispatch.cpp" "${CMAKE_CURRENT_LIST_DIR}/expert_dispatch.cu")
target_include_directories(pih_deepseek_v41_expert_dispatch_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_expert_dispatch_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_dispatch_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_expert_dispatch_cuda PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_expert_dispatch_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_router_cuda STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/router.cpp" "${CMAKE_CURRENT_LIST_DIR}/router.cu")
target_include_directories(pih_deepseek_v41_router_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_router_cuda PRIVATE pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_router_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_router_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_router_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_router_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_step_phases STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/step_phases.cpp")
target_include_directories(pih_deepseek_v41_step_phases PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_step_phases PRIVATE pih_deepseek_v41_rope_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_step_phases pih_deepseek_v41_rope_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_step_phases PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_step_phases PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_attention_sources STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/attention_sources.cpp")
target_include_directories(pih_deepseek_v41_attention_sources PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_sources PRIVATE pih_deepseek_v41_attention_prepare pih_deepseek_v41_compressed_prepare pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_sources pih_deepseek_v41_attention_prepare pih_deepseek_v41_compressed_prepare pih_core_primitives)
target_compile_features(pih_deepseek_v41_attention_sources PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_attention_sources PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_compressed_prepare STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/compressed_prepare.cpp")
target_include_directories(pih_deepseek_v41_compressed_prepare PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_compressed_prepare PRIVATE pih_deepseek_v41_compressor_pool_cuda pih_deepseek_v41_compressed_kv_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_compressed_prepare pih_deepseek_v41_compressor_pool_cuda pih_deepseek_v41_compressed_kv_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_compressed_prepare PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_compressed_prepare PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_attention_prepare STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/attention_prepare.cpp")
target_include_directories(pih_deepseek_v41_attention_prepare PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_prepare PRIVATE pih_deepseek_v41_attention_input pih_deepseek_v41_mhc_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_prepare pih_deepseek_v41_attention_input pih_deepseek_v41_mhc_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_attention_prepare PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_attention_prepare PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_ffn_route STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/ffn_route.cpp")
target_include_directories(pih_deepseek_v41_ffn_route PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_ffn_route PRIVATE pih_deepseek_v41_mhc_cuda pih_deepseek_v41_router_cuda pih_deepseek_v41_expert_counts pih_deepseek_v41_engram_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_ffn_route pih_deepseek_v41_mhc_cuda pih_deepseek_v41_router_cuda pih_deepseek_v41_expert_counts pih_deepseek_v41_engram_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_ffn_route PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_ffn_route PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_expert_residual STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/expert_residual.cpp")
target_include_directories(pih_deepseek_v41_expert_residual PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_expert_residual PRIVATE pih_deepseek_v41_shared_expert_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_residual pih_deepseek_v41_shared_expert_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_expert_residual PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_expert_residual PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_expert_workspace STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/expert_workspace.cpp" "${CMAKE_CURRENT_LIST_DIR}/expert_workspace_owner.cpp")
target_include_directories(pih_deepseek_v41_expert_workspace PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_expert_workspace PRIVATE pih_deepseek_v41_expert_batch pih_deepseek_v41_expert_counts pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_workspace pih_deepseek_v41_expert_batch pih_deepseek_v41_expert_counts pih_core_primitives)
target_compile_features(pih_deepseek_v41_expert_workspace PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_expert_workspace PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_expert_batch STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/expert_batch.cpp")
target_include_directories(pih_deepseek_v41_expert_batch PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_expert_batch PRIVATE pih_deepseek_v41_expert_chain pih_deepseek_v41_expert_counts pih_deepseek_v41_engram_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_batch pih_deepseek_v41_expert_chain pih_deepseek_v41_expert_counts pih_deepseek_v41_engram_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_expert_batch PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_expert_batch PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_expert_counts STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/expert_counts.cpp")
target_include_directories(pih_deepseek_v41_expert_counts PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_expert_counts PRIVATE pih_deepseek_v41_expert_dispatch_cuda pih_deepseek_v41_engram_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_counts pih_deepseek_v41_expert_dispatch_cuda pih_deepseek_v41_engram_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_expert_counts PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_expert_counts PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_expert_chain STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/expert_chain.cpp")
target_include_directories(pih_deepseek_v41_expert_chain PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_expert_chain PRIVATE pih_deepseek_v41_expert_dispatch_cuda pih_deepseek_v41_shared_expert_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_chain pih_deepseek_v41_expert_dispatch_cuda pih_deepseek_v41_shared_expert_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_expert_chain PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_expert_chain PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pih_deepseek_v41_shared_expert_cuda STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/expert_fp8.cpp" "${CMAKE_CURRENT_LIST_DIR}/expert_fp8.cu")
target_include_directories(pih_deepseek_v41_shared_expert_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_shared_expert_cuda PRIVATE pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_v41_shared_expert_cuda pih_deepseek_v41_fp8_cuda pih_core_primitives CUDA::cudart)
target_compile_features(pih_deepseek_v41_shared_expert_cuda PRIVATE cxx_std_20)
target_compile_options(pih_deepseek_v41_shared_expert_cuda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fmad=false> $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>)
set_target_properties(pih_deepseek_v41_shared_expert_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_attention_residual STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/attention_residual.cpp")
target_include_directories(pih_deepseek_v41_attention_residual PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_attention_residual PRIVATE pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_residual pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_attention_residual PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_attention_residual PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_indexer_chain STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/indexer_chain.cpp")
target_include_directories(pih_deepseek_v41_indexer_chain PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_indexer_chain PRIVATE
    pih_deepseek_v41_indexer_input_cuda pih_deepseek_v41_indexer_score_cuda
    pih_deepseek_v41_indexer_select_cuda pih_deepseek_v41_engram_cuda pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_indexer_chain
    pih_deepseek_v41_indexer_input_cuda pih_deepseek_v41_indexer_score_cuda
    pih_deepseek_v41_indexer_select_cuda pih_deepseek_v41_engram_cuda pih_core_primitives)
target_compile_features(pih_deepseek_v41_indexer_chain PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_indexer_chain PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(PIH_BUILD_NATIVE_ENGRAM_NCCL)
    find_path(PIH_NATIVE_ENGRAM_NCCL_INCLUDE_DIR nccl.h REQUIRED)
    find_library(PIH_NATIVE_ENGRAM_NCCL_LIBRARY NAMES nccl REQUIRED)
    add_library(pih_deepseek_v41_engram_nccl STATIC EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/engram_reduce.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/engram_pipeline.cpp")
    target_include_directories(pih_deepseek_v41_engram_nccl PRIVATE
        "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_engram_nccl PRIVATE
        pih_deepseek_v41_engram_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_engram_nccl
        pih_deepseek_v41_engram_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_engram_nccl PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_engram_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_indexer_nccl STATIC EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/indexer_reduce.cpp" "${CMAKE_CURRENT_LIST_DIR}/indexer_pipeline.cpp")
    target_include_directories(pih_deepseek_v41_indexer_nccl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_indexer_nccl PRIVATE
        pih_deepseek_v41_engram_nccl pih_deepseek_v41_indexer_chain pih_deepseek_v41_indexer_input_cuda pih_deepseek_v41_indexer_score_cuda pih_deepseek_v41_indexer_select_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_indexer_nccl
        pih_deepseek_v41_engram_nccl pih_deepseek_v41_indexer_chain pih_deepseek_v41_indexer_input_cuda pih_deepseek_v41_indexer_score_cuda pih_deepseek_v41_indexer_select_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_indexer_nccl PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_indexer_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_sampling_operation STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/sampling_operation.cpp")
    target_include_directories(pih_deepseek_v41_sampling_operation PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_sampling_operation PRIVATE pih_deepseek_v41_head_operation
        pih_deepseek_v41_block_operation pih_deepseek_v41_sampling_cuda pih_deepseek_v41_engram_cuda pih_core_primitives CUDA::cudart)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_sampling_operation pih_deepseek_v41_head_operation
        pih_deepseek_v41_block_operation pih_deepseek_v41_sampling_cuda pih_deepseek_v41_engram_cuda pih_core_primitives CUDA::cudart)
    target_compile_features(pih_deepseek_v41_sampling_operation PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_sampling_operation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_embedding_operation STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/embedding_operation.cpp")
    target_include_directories(pih_deepseek_v41_embedding_operation PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_embedding_operation PRIVATE pih_deepseek_v41_uploaded_bindings pih_deepseek_v41_block_operation
        pih_deepseek_v41_token_input_upload pih_deepseek_v41_token_embedding_cuda pih_deepseek_v41_engram_nccl pih_deepseek_v41_engram_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_embedding_operation pih_deepseek_v41_uploaded_bindings pih_deepseek_v41_block_operation
        pih_deepseek_v41_token_input_upload pih_deepseek_v41_token_embedding_cuda pih_deepseek_v41_engram_nccl pih_deepseek_v41_engram_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_embedding_operation PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_embedding_operation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_head_operation STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/head_operation.cpp")
    target_include_directories(pih_deepseek_v41_head_operation PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_head_operation PRIVATE pih_deepseek_v41_uploaded_bindings pih_deepseek_v41_block_operation
        pih_deepseek_v41_model_head_cuda pih_deepseek_v41_mhc_cuda pih_deepseek_v41_engram_nccl pih_deepseek_v41_engram_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_head_operation pih_deepseek_v41_uploaded_bindings pih_deepseek_v41_block_operation
        pih_deepseek_v41_model_head_cuda pih_deepseek_v41_mhc_cuda pih_deepseek_v41_engram_nccl pih_deepseek_v41_engram_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_head_operation PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_head_operation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_block_operation STATIC EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/block_operation.cpp" "${CMAKE_CURRENT_LIST_DIR}/block_sequence.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/backbone_operation.cpp")
    target_include_directories(pih_deepseek_v41_block_operation PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_block_operation PRIVATE
        pih_deepseek_v41_engram_plan
        pih_deepseek_v41_compressor_plan
        pih_deepseek_v41_indexer_plan
        pih_deepseek_v41_attention_output_plan
        pih_deepseek_v41_attention_prepare_plan
        pih_deepseek_v41_ffn_plan pih_deepseek_v41_sequence_cache pih_deepseek_v41_uploaded_bindings
        pih_deepseek_v41_step_phases pih_deepseek_v41_block_bindings pih_deepseek_v41_indexed_sources
        pih_deepseek_v41_prepared_block pih_deepseek_v41_attention_nccl pih_deepseek_v41_expert_workspace
        pih_deepseek_v41_engram_cuda pih_deepseek_v41_engram_nccl pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_block_operation
        pih_deepseek_v41_engram_plan
        pih_deepseek_v41_compressor_plan
        pih_deepseek_v41_indexer_plan
        pih_deepseek_v41_attention_output_plan
        pih_deepseek_v41_attention_prepare_plan
        pih_deepseek_v41_ffn_plan pih_deepseek_v41_sequence_cache pih_deepseek_v41_uploaded_bindings
        pih_deepseek_v41_step_phases pih_deepseek_v41_block_bindings pih_deepseek_v41_indexed_sources
        pih_deepseek_v41_prepared_block pih_deepseek_v41_attention_nccl pih_deepseek_v41_expert_workspace
        pih_deepseek_v41_engram_cuda pih_deepseek_v41_engram_nccl pih_core_primitives)
    target_compile_features(pih_deepseek_v41_block_operation PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_block_operation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_inference_operation STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/inference_operation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/inference_memory.cpp" "${CMAKE_CURRENT_LIST_DIR}/inference_memory_owner.cpp")
    target_include_directories(pih_deepseek_v41_inference_operation PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_inference_operation PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_inference_operation PRIVATE pih_deepseek_v41_block_operation
        pih_deepseek_v41_token_ledger
        pih_deepseek_v41_step_phases pih_deepseek_v41_engram_plan pih_deepseek_v41_sequence_cache
        pih_deepseek_v41_ffn_plan pih_deepseek_v41_attention_prepare_plan pih_deepseek_v41_attention_output_plan
        pih_deepseek_v41_compressor_plan pih_deepseek_v41_indexer_plan
        pih_deepseek_v41_boundary_plan pih_deepseek_v41_embedding_operation pih_deepseek_v41_head_operation
        pih_deepseek_v41_sampling_operation pih_deepseek_v41_sampling_cuda pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_inference_operation pih_deepseek_v41_block_operation
        pih_deepseek_v41_token_ledger
        pih_deepseek_v41_step_phases pih_deepseek_v41_engram_plan pih_deepseek_v41_sequence_cache
        pih_deepseek_v41_ffn_plan pih_deepseek_v41_attention_prepare_plan pih_deepseek_v41_attention_output_plan
        pih_deepseek_v41_compressor_plan pih_deepseek_v41_indexer_plan
        pih_deepseek_v41_boundary_plan pih_deepseek_v41_embedding_operation pih_deepseek_v41_head_operation
        pih_deepseek_v41_sampling_operation pih_deepseek_v41_sampling_cuda pih_deepseek_v41_uploaded_bindings pih_core_primitives)
    set_target_properties(pih_deepseek_v41_inference_operation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_rank_worker_loop STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/rank_worker_loop.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/worker_handles.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/worker_memory.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/worker_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/worker_artifacts.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/worker_communicator.cpp")
    target_include_directories(pih_deepseek_v41_rank_worker_loop PRIVATE
        "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_rank_worker_loop PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_rank_worker_loop PRIVATE pih_deepseek_v41_inference_operation
        pih_deepseek_v41_rank_channel pih_deepseek_v41_sampling_cuda pih_core_primitives
        CUDA::cudart)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_rank_worker_loop pih_deepseek_v41_inference_operation
        pih_deepseek_v41_rank_channel pih_deepseek_v41_sampling_cuda pih_core_primitives
        CUDA::cudart)
    set_target_properties(pih_deepseek_v41_rank_worker_loop PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(pih_deepseek_v41_block_bindings STATIC EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/block_bindings.cpp" "${CMAKE_CURRENT_LIST_DIR}/block_liveness.cpp")
    target_include_directories(pih_deepseek_v41_block_bindings PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_block_bindings PRIVATE pih_deepseek_v41_step_phases pih_deepseek_v41_indexed_sources pih_deepseek_v41_attention_residual pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_residual pih_deepseek_v41_expert_workspace pih_deepseek_v41_engram_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_block_bindings pih_deepseek_v41_step_phases pih_deepseek_v41_indexed_sources pih_deepseek_v41_attention_residual pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_residual pih_deepseek_v41_expert_workspace pih_deepseek_v41_engram_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_block_bindings PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_block_bindings PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(pih_deepseek_v41_indexed_sources STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/indexed_sources.cpp")
    target_include_directories(pih_deepseek_v41_indexed_sources PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_indexed_sources PRIVATE pih_deepseek_v41_attention_sources pih_deepseek_v41_indexer_nccl pih_deepseek_v41_indexer_chain pih_deepseek_v41_engram_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_indexed_sources pih_deepseek_v41_attention_sources pih_deepseek_v41_indexer_nccl pih_deepseek_v41_indexer_chain pih_deepseek_v41_engram_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_indexed_sources PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_indexed_sources PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(pih_deepseek_v41_prepared_block STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/prepared_block.cpp")
    target_include_directories(pih_deepseek_v41_prepared_block PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_prepared_block PRIVATE pih_deepseek_v41_attention_nccl pih_deepseek_v41_attention_residual pih_deepseek_v41_ffn_nccl pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_residual pih_deepseek_v41_expert_workspace pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_prepared_block pih_deepseek_v41_attention_nccl pih_deepseek_v41_attention_residual pih_deepseek_v41_ffn_nccl pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_residual pih_deepseek_v41_expert_workspace pih_core_primitives)
    target_compile_features(pih_deepseek_v41_prepared_block PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_prepared_block PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(pih_deepseek_v41_ffn_nccl STATIC EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/ffn_continuation.cpp" "${CMAKE_CURRENT_LIST_DIR}/ffn_operation.cpp")
    target_include_directories(pih_deepseek_v41_ffn_nccl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_ffn_nccl PRIVATE pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_batch pih_deepseek_v41_expert_workspace pih_deepseek_v41_expert_residual pih_deepseek_v41_expert_nccl pih_deepseek_v41_engram_nccl pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_ffn_nccl pih_deepseek_v41_ffn_route pih_deepseek_v41_expert_batch pih_deepseek_v41_expert_workspace pih_deepseek_v41_expert_residual pih_deepseek_v41_expert_nccl pih_deepseek_v41_engram_nccl pih_core_primitives)
    target_compile_features(pih_deepseek_v41_ffn_nccl PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_ffn_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(pih_deepseek_v41_expert_nccl STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/expert_pipeline.cpp")
    target_include_directories(pih_deepseek_v41_expert_nccl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_expert_nccl PRIVATE pih_deepseek_v41_engram_nccl pih_deepseek_v41_expert_residual pih_deepseek_v41_engram_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_expert_nccl pih_deepseek_v41_engram_nccl pih_deepseek_v41_expert_residual pih_deepseek_v41_engram_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_expert_nccl PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_expert_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(pih_deepseek_v41_attention_nccl STATIC EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/attention_reduce.cpp" "${CMAKE_CURRENT_LIST_DIR}/attention_pipeline.cpp")
    target_include_directories(pih_deepseek_v41_attention_nccl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(pih_deepseek_v41_attention_nccl PRIVATE pih_deepseek_v41_engram_nccl pih_deepseek_v41_attention_output_cuda pih_deepseek_v41_attention_residual pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_attention_nccl pih_deepseek_v41_engram_nccl pih_deepseek_v41_attention_output_cuda pih_deepseek_v41_attention_residual pih_deepseek_v41_attention_assemble_cuda pih_deepseek_v41_mhc_cuda pih_core_primitives)
    target_compile_features(pih_deepseek_v41_attention_nccl PRIVATE cxx_std_20)
    set_target_properties(pih_deepseek_v41_attention_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
