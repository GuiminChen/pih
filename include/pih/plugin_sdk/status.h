#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_STATUS_ABI_VERSION_V1 1U

enum pih_status_code_v1 {
  PIH_STATUS_OK_V1 = 0,
  PIH_STATUS_INVALID_ARGUMENT_V1 = 1,
  PIH_STATUS_FAILED_PRECONDITION_V1 = 2,
  PIH_STATUS_INTERNAL_V1 = 3,
  PIH_STATUS_UNAVAILABLE_V1 = 4,
  PIH_STATUS_RESOURCE_EXHAUSTED_V1 = 5,
  PIH_STATUS_DEADLINE_EXCEEDED_V1 = 6,
};

typedef struct pih_status_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t code;
  char message[256];
} pih_status_v1;

static inline int pih_status_is_valid_v1(const pih_status_v1* status) {
  return status != 0 && status->struct_size == sizeof(pih_status_v1) &&
         status->abi_version == PIH_STATUS_ABI_VERSION_V1 &&
         status->code <= PIH_STATUS_DEADLINE_EXCEEDED_V1;
}

static inline int pih_status_is_ok_v1(const pih_status_v1* status) {
  return pih_status_is_valid_v1(status) && status->code == PIH_STATUS_OK_V1;
}

#ifdef __cplusplus
}
#endif
