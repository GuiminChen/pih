# CPU-only model tokenizer and offline Engram map command. No Python/GPU owner.
if(PIH_BUILD_TOKENIZER_TOOLS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Imported ICU targets are directory-scoped; resolve them in this owner.
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    add_library(pih_deepseek_v41_token_map STATIC EXCLUDE_FROM_ALL
        token_map_builder.cpp "${CMAKE_CURRENT_LIST_DIR}/../common/bytelevel_tokenizer.cpp")
    target_compile_features(pih_deepseek_v41_token_map PUBLIC cxx_std_20)
    target_link_libraries(pih_deepseek_v41_token_map PUBLIC pih_core_primitives ICU::uc ICU::i18n)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_token_map pih_core_primitives ICU::uc ICU::i18n)
    add_executable(pih-v41-token-map EXCLUDE_FROM_ALL token_map_main.cpp)
    target_compile_features(pih-v41-token-map PRIVATE cxx_std_20)
    set_target_properties(pih-v41-token-map PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_link_libraries(pih-v41-token-map PRIVATE pih_deepseek_v41_token_map pih_deepseek_v41_weight_files pih_native_model_support)
    pih_assert_direct_link_allowlist(pih-v41-token-map pih_deepseek_v41_token_map pih_deepseek_v41_weight_files pih_native_model_support)
    add_library(pih_deepseek_v41_tokenizer STATIC EXCLUDE_FROM_ALL
        tokenizer.cpp "${CMAKE_CURRENT_LIST_DIR}/../common/bytelevel_tokenizer.cpp")
    target_compile_features(pih_deepseek_v41_tokenizer PUBLIC cxx_std_20)
    set_target_properties(pih_deepseek_v41_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(pih_deepseek_v41_tokenizer PUBLIC
        pih_deepseek_v41_weight_files pih_core_primitives ICU::uc ICU::i18n)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_tokenizer
        pih_deepseek_v41_weight_files pih_core_primitives ICU::uc ICU::i18n)
endif()
