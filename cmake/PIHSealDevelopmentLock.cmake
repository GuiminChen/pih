if(NOT DEFINED PIH_BUNDLE_ROOT OR
   NOT IS_DIRECTORY "${PIH_BUNDLE_ROOT}" OR
   IS_SYMLINK "${PIH_BUNDLE_ROOT}")
    message(FATAL_ERROR "PIH bundle root is missing")
endif()

set(
    PIH_LOCK_PATH
    "${PIH_BUNDLE_ROOT}/deepseek-v4-flash-rtx4090d-pp1.lock"
)
if(NOT EXISTS "${PIH_LOCK_PATH}" OR
   IS_DIRECTORY "${PIH_LOCK_PATH}" OR
   IS_SYMLINK "${PIH_LOCK_PATH}")
    message(FATAL_ERROR "PIH development Lock is missing or not sealable")
endif()
file(READ "${PIH_LOCK_PATH}" PIH_LOCK_JSON)

function(pih_seal_locked_artifact PIH_SECTION PIH_INDEX PIH_PATH_FIELD
         PIH_DIGEST_FIELD)
    string(JSON PIH_RELATIVE_PATH GET "${PIH_LOCK_JSON}"
           ${PIH_SECTION} ${PIH_INDEX} ${PIH_PATH_FIELD})
    if(IS_ABSOLUTE "${PIH_RELATIVE_PATH}" OR
       PIH_RELATIVE_PATH MATCHES "(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR
                "Locked PIH artifact path escapes the bundle: ${PIH_RELATIVE_PATH}")
    endif()
    get_filename_component(
        PIH_ARTIFACT_PARENT
        "${PIH_RELATIVE_PATH}"
        DIRECTORY
    )
    while(NOT PIH_ARTIFACT_PARENT STREQUAL "" AND
          NOT PIH_ARTIFACT_PARENT STREQUAL ".")
        set(
            PIH_ARTIFACT_PARENT_PATH
            "${PIH_BUNDLE_ROOT}/${PIH_ARTIFACT_PARENT}"
        )
        if(NOT IS_DIRECTORY "${PIH_ARTIFACT_PARENT_PATH}" OR
           IS_SYMLINK "${PIH_ARTIFACT_PARENT_PATH}")
            message(FATAL_ERROR
                    "Locked PIH artifact parent is not sealable: ${PIH_ARTIFACT_PARENT}")
        endif()
        get_filename_component(
            PIH_ARTIFACT_PARENT
            "${PIH_ARTIFACT_PARENT}"
            DIRECTORY
        )
    endwhile()
    set(PIH_ARTIFACT_PATH "${PIH_BUNDLE_ROOT}/${PIH_RELATIVE_PATH}")
    if(NOT EXISTS "${PIH_ARTIFACT_PATH}" OR
       IS_DIRECTORY "${PIH_ARTIFACT_PATH}" OR
       IS_SYMLINK "${PIH_ARTIFACT_PATH}")
        message(FATAL_ERROR
                "Locked PIH artifact is missing or not sealable: ${PIH_RELATIVE_PATH}")
    endif()
    file(SHA256 "${PIH_ARTIFACT_PATH}" PIH_ARTIFACT_SHA256)
    set(PIH_PATH_FRAGMENT "\"${PIH_PATH_FIELD}\":\"${PIH_RELATIVE_PATH}\"")
    set(
        PIH_SEALED_FRAGMENT
        "${PIH_PATH_FRAGMENT},\"${PIH_DIGEST_FIELD}\":\"${PIH_ARTIFACT_SHA256}\""
    )
    string(FIND "${PIH_LOCK_JSON}" "${PIH_SEALED_FRAGMENT}"
           PIH_ALREADY_SEALED)
    if(PIH_ALREADY_SEALED EQUAL -1)
        string(FIND "${PIH_LOCK_JSON}" "${PIH_PATH_FRAGMENT}"
               PIH_PATH_POSITION)
        if(PIH_PATH_POSITION EQUAL -1)
            message(FATAL_ERROR "Locked PIH artifact path is not canonical")
        endif()
        string(REPLACE "${PIH_PATH_FRAGMENT}" "${PIH_SEALED_FRAGMENT}"
               PIH_LOCK_JSON "${PIH_LOCK_JSON}")
    endif()
    set(PIH_LOCK_JSON "${PIH_LOCK_JSON}" PARENT_SCOPE)
endfunction()

string(JSON PIH_KERNEL_COUNT LENGTH "${PIH_LOCK_JSON}" kernel_packs)
if(PIH_KERNEL_COUNT GREATER 0)
    math(EXPR PIH_KERNEL_LAST "${PIH_KERNEL_COUNT} - 1")
    foreach(PIH_INDEX RANGE 0 ${PIH_KERNEL_LAST})
        pih_seal_locked_artifact(kernel_packs ${PIH_INDEX} binary
                                 binary_sha256_hex)
    endforeach()
endif()

string(JSON PIH_PLUGIN_COUNT LENGTH "${PIH_LOCK_JSON}" plugins)
if(PIH_PLUGIN_COUNT LESS 1)
    message(FATAL_ERROR "PIH development Lock has no plugins")
endif()
math(EXPR PIH_PLUGIN_LAST "${PIH_PLUGIN_COUNT} - 1")
foreach(PIH_INDEX RANGE 0 ${PIH_PLUGIN_LAST})
    pih_seal_locked_artifact(plugins ${PIH_INDEX} entrypoint
                             entrypoint_sha256_hex)
endforeach()

file(WRITE "${PIH_LOCK_PATH}" "${PIH_LOCK_JSON}")
