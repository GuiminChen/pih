foreach(sm IN LISTS PIH_QWEN_KERNEL_ARCHITECTURES)
    set(target pih_kernels_qwen3_sm${sm})
    add_library(${target} SHARED kernel_pack.cpp)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE PIH::plugin_sdk ${CMAKE_DL_LIBS})
    pih_assert_direct_link_allowlist(${target} PIH::plugin_sdk ${CMAKE_DL_LIBS})
    target_link_options(${target} PRIVATE "LINKER:-z,defs"
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    target_compile_definitions(${target} PRIVATE
        PIH_QWEN_PACK_ID="pih.kernels.qwen3.sm${sm}"
        PIH_QWEN_PACK_ARCH="sm_${sm}" PIH_QWEN_PACK_SM=${sm}
        PIH_QWEN_PACK_DIRECTORY="qwen3-sm${sm}")
    set_target_properties(${target} PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
    add_dependencies(${target} pih_qwen_sm${sm}_cubin)
    install(TARGETS ${target} LIBRARY DESTINATION lib)
    install(FILES
        "${CMAKE_BINARY_DIR}/cubin/sm_${sm}/qwen_bf16_primitives.cubin"
        "${CMAKE_BINARY_DIR}/cubin/sm_${sm}/qwen_bf16_primitives.cubin.json"
        DESTINATION lib/qwen3-sm${sm}/sm_${sm})
endforeach()
