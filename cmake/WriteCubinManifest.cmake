if(NOT DEFINED CUBIN_PATH OR NOT EXISTS "${CUBIN_PATH}")
  message(FATAL_ERROR "CUBIN_PATH must name an existing cubin")
endif()
if(NOT DEFINED MANIFEST_PATH OR NOT DEFINED TARGET_SM OR
   NOT DEFINED PRODUCER_TOOLKIT OR NOT DEFINED PRODUCER_FLAGS)
  message(FATAL_ERROR "cubin manifest inputs are incomplete")
endif()

file(SHA256 "${CUBIN_PATH}" cubin_sha256)
file(SIZE "${CUBIN_PATH}" cubin_bytes)
string(SHA256 producer_flags_sha256 "${PRODUCER_FLAGS}")
file(
    WRITE "${MANIFEST_PATH}"
    "{\n"
    "  \"schema\": \"pih.cubin_artifact.v1\",\n"
    "  \"target_sm\": \"${TARGET_SM}\",\n"
    "  \"producer_toolkit\": \"${PRODUCER_TOOLKIT}\",\n"
    "  \"producer_flags_sha256\": \"${producer_flags_sha256}\",\n"
    "  \"cubin_sha256\": \"${cubin_sha256}\",\n"
    "  \"cubin_bytes\": ${cubin_bytes}\n"
    "}\n"
)
