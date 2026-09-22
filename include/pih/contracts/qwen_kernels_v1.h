#pragma once
#include <stdint.h>
#include "pih/plugin_sdk/status.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct pih_qwen_kernels_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t target_sm;
  // Pack-owned path; lives as long as the loaded pack. Contents carry the
  // native Qwen cubin manifest and are independently checked by its loader.
  pih_status_v1 (*root)(const char** path);
} pih_qwen_kernels_api_v1;
#ifdef __cplusplus
}
#endif
