if(NOT DEFINED PIH_BUNDLE_ROOT OR
   NOT IS_DIRECTORY "${PIH_BUNDLE_ROOT}" OR
   IS_SYMLINK "${PIH_BUNDLE_ROOT}")
    message(FATAL_ERROR "DeepSeek PP1 bundle root is missing")
endif()
if(NOT DEFINED PIH_ARTIFACT_DIRECTORY OR
   NOT IS_ABSOLUTE "${PIH_ARTIFACT_DIRECTORY}" OR
   NOT IS_DIRECTORY "${PIH_ARTIFACT_DIRECTORY}" OR
   IS_SYMLINK "${PIH_ARTIFACT_DIRECTORY}")
    message(
        FATAL_ERROR
        "PIH_DEEPSEEK_4090D_ARTIFACT_DIRECTORY must name an existing absolute verified generation directory"
    )
endif()
foreach(PIH_REQUIRED_AUTHORITY IN ITEMS
        pih.manifest.json
        pih.runtime-records.json)
    set(
        PIH_AUTHORITY_PATH
        "${PIH_ARTIFACT_DIRECTORY}/${PIH_REQUIRED_AUTHORITY}"
    )
    if(NOT EXISTS "${PIH_AUTHORITY_PATH}" OR
       IS_DIRECTORY "${PIH_AUTHORITY_PATH}" OR
       IS_SYMLINK "${PIH_AUTHORITY_PATH}")
        message(
            FATAL_ERROR
            "DeepSeek PP1 generation authority must be a real file: ${PIH_REQUIRED_AUTHORITY}"
        )
    endif()
endforeach()

string(LENGTH "${PIH_ARTIFACT_ROOT_SHA256}" PIH_ARTIFACT_ROOT_LENGTH)
if(NOT DEFINED PIH_ARTIFACT_ROOT_SHA256 OR
   NOT PIH_ARTIFACT_ROOT_LENGTH EQUAL 64 OR
   NOT PIH_ARTIFACT_ROOT_SHA256 MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "DeepSeek PP1 expected artifact root is invalid")
endif()
if(PIH_ARTIFACT_ROOT_SHA256 STREQUAL
   "0000000000000000000000000000000000000000000000000000000000000000")
    message(FATAL_ERROR "DeepSeek PP1 placeholder artifact root is forbidden")
endif()

file(READ "${PIH_ARTIFACT_DIRECTORY}/pih.manifest.json" PIH_MANIFEST)
foreach(PIH_MANIFEST_FIELD IN ITEMS
        artifact_root
        model_family
        world_size
        dspark_enabled)
    string(
        JSON PIH_MANIFEST_${PIH_MANIFEST_FIELD}
        ERROR_VARIABLE PIH_MANIFEST_ERROR
        GET "${PIH_MANIFEST}" "${PIH_MANIFEST_FIELD}"
    )
    if(PIH_MANIFEST_ERROR)
        message(
            FATAL_ERROR
            "DeepSeek PP1 manifest has no valid ${PIH_MANIFEST_FIELD}"
        )
    endif()
endforeach()
if(NOT "${PIH_MANIFEST_artifact_root}" STREQUAL
       "${PIH_ARTIFACT_ROOT_SHA256}")
    message(
        FATAL_ERROR
        "DeepSeek PP1 manifest artifact root does not match the Lock"
    )
endif()
if(NOT "${PIH_MANIFEST_model_family}" STREQUAL "deepseek_v4_flash_0731" OR
   NOT "${PIH_MANIFEST_world_size}" EQUAL 1 OR
   PIH_MANIFEST_dspark_enabled)
    message(FATAL_ERROR "DeepSeek PP1 manifest does not describe the required profile")
endif()

set(PIH_ARTIFACT_PARENT "${PIH_BUNDLE_ROOT}/artifacts")
set(PIH_ARTIFACT_LINK "${PIH_ARTIFACT_PARENT}/deepseek-v4-flash-0731")
if(IS_SYMLINK "${PIH_ARTIFACT_PARENT}" OR
   (EXISTS "${PIH_ARTIFACT_PARENT}" AND
    NOT IS_DIRECTORY "${PIH_ARTIFACT_PARENT}"))
    message(
        FATAL_ERROR
        "DeepSeek PP1 artifact mount parent must be a real directory"
    )
endif()
file(MAKE_DIRECTORY "${PIH_ARTIFACT_PARENT}")
if(IS_SYMLINK "${PIH_ARTIFACT_LINK}")
    file(REMOVE "${PIH_ARTIFACT_LINK}")
elseif(EXISTS "${PIH_ARTIFACT_LINK}")
    message(
        FATAL_ERROR
        "DeepSeek PP1 artifact mount path exists and is not a symbolic link"
    )
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
            "${PIH_ARTIFACT_DIRECTORY}" "${PIH_ARTIFACT_LINK}"
    RESULT_VARIABLE PIH_LINK_RESULT
)
if(NOT PIH_LINK_RESULT EQUAL 0)
    message(FATAL_ERROR "DeepSeek PP1 artifact mount link could not be created")
endif()
