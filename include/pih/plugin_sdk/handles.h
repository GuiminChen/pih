#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_HANDLE_ABI_VERSION_V1 1U

typedef struct pih_activation_handle_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t activation_epoch;
  uint64_t owner_epoch;
} pih_activation_handle_v1;

typedef struct pih_resource_handle_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t activation_epoch;
  uint64_t resource_epoch;
} pih_resource_handle_v1;

#ifdef __cplusplus
}
#endif
