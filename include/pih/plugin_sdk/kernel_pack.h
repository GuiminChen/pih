#pragma once

#include <stdint.h>
#include "pih/plugin_sdk/status.h"

#define PIH_KERNEL_PACK_ABI_VERSION_V1 1u
#define PIH_KERNEL_PACK_IDENTITY_SYMBOL_V1 "pih_kernel_pack_identity_v1_get"
#define PIH_KERNEL_PACK_API_SYMBOL_V1 "pih_kernel_pack_api_v1_get"

typedef struct pih_kernel_pack_identity_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* pack_id;
  const char* pack_version;
  const char* pack_abi;
  const char* architecture;
} pih_kernel_pack_identity_v1;

typedef const pih_kernel_pack_identity_v1*
    (*pih_kernel_pack_identity_fn_v1)(void);

typedef struct pih_kernel_pack_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const pih_kernel_pack_identity_v1* identity;
  const void* contract_api;
  // Required before publishing the contract. The host passes the authenticated
  // library source location; dladdr sees a sealed memfd, not this path.
  // Must copy retained data; cannot activate devices. Old smaller tables fail.
  pih_status_v1 (*bind_origin)(const char* absolute_library_path);
} pih_kernel_pack_api_v1;

typedef const pih_kernel_pack_api_v1* (*pih_kernel_pack_api_fn_v1)(void);

#ifdef __cplusplus
extern "C" {
#endif

const pih_kernel_pack_identity_v1* pih_kernel_pack_identity_v1_get(void);
const pih_kernel_pack_api_v1* pih_kernel_pack_api_v1_get(void);

#ifdef __cplusplus
}
#endif
