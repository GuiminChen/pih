string(LENGTH "${PIH_ARTIFACT_ROOT_SHA256}" PIH_ARTIFACT_ROOT_LENGTH)
if(NOT DEFINED PIH_ARTIFACT_ROOT_SHA256 OR
   NOT PIH_ARTIFACT_ROOT_LENGTH EQUAL 64 OR
   NOT PIH_ARTIFACT_ROOT_SHA256 MATCHES "^[0-9a-f]+$")
    message(
        FATAL_ERROR
        "DeepSeek PP1 bundle requires a 64-character lowercase artifact root"
    )
endif()

set(
    PIH_ZERO_SHA256
    "0000000000000000000000000000000000000000000000000000000000000000"
)
if("${PIH_ARTIFACT_ROOT_SHA256}" STREQUAL "${PIH_ZERO_SHA256}")
    message(
        FATAL_ERROR
        "DeepSeek PP1 bundle cannot be emitted with the placeholder artifact root; configure PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256"
    )
endif()

if(NOT DEFINED PIH_LOCK_PATH OR NOT EXISTS "${PIH_LOCK_PATH}")
    message(FATAL_ERROR "DeepSeek PP1 generated deployment lock is missing")
endif()
file(READ "${PIH_LOCK_PATH}" PIH_LOCK_CONTENT)
set(
    PIH_EXPECTED_ROOT_FIELD
    "\"artifact_root_sha256_hex\":\"${PIH_ARTIFACT_ROOT_SHA256}\""
)
string(FIND "${PIH_LOCK_CONTENT}" "${PIH_EXPECTED_ROOT_FIELD}"
       PIH_EXPECTED_ROOT_POSITION)
string(FIND "${PIH_LOCK_CONTENT}" "${PIH_ZERO_SHA256}"
       PIH_PLACEHOLDER_ROOT_POSITION)
if(PIH_EXPECTED_ROOT_POSITION EQUAL -1 OR
   NOT PIH_PLACEHOLDER_ROOT_POSITION EQUAL -1)
    message(
        FATAL_ERROR
        "DeepSeek PP1 generated deployment lock does not contain the configured artifact root"
    )
endif()
