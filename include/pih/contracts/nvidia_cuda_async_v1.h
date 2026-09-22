#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 1U
#define PIH_CUDA_COPY_H2D_V1 1U
#define PIH_CUDA_COPY_D2H_V1 2U
#define PIH_CUDA_COPY_D2D_V1 3U
#define PIH_CUDA_EVENT_COMPLETE_V1 1U
#define PIH_CUDA_EVENT_PENDING_V1 2U

typedef pih_status_v1 (*pih_cuda_activate_context_v1)(
    void* context, uintptr_t retained_context);

// Checks the calling thread's CUDA last-error state without clearing it or
// changing its current context. The exact retained context must be current.
typedef pih_status_v1 (*pih_cuda_require_clean_last_error_v1)(
    void* context, uintptr_t retained_context);
typedef pih_status_v1 (*pih_cuda_validate_pinned_host_v1)(
    void* context, uintptr_t retained_context, uintptr_t pointer);
typedef pih_status_v1 (*pih_cuda_copy_async_v1)(
    void* context, uintptr_t retained_context, uintptr_t destination,
    uintptr_t source, uint64_t bytes, uint32_t copy_kind, uintptr_t stream);
typedef pih_status_v1 (*pih_cuda_memset_async_v1)(
    void* context, uintptr_t retained_context, uintptr_t destination,
    uint32_t byte_value, uint64_t bytes, uintptr_t stream);
typedef pih_status_v1 (*pih_cuda_record_event_v1)(
    void* context, uintptr_t retained_context, uintptr_t event,
    uintptr_t stream);
typedef pih_status_v1 (*pih_cuda_query_event_v1)(
    void* context, uintptr_t retained_context, uintptr_t event,
    uint32_t* event_status);
typedef pih_status_v1 (*pih_cuda_synchronize_stream_v1)(
    void* context, uintptr_t retained_context, uintptr_t stream);

typedef struct pih_nvidia_cuda_async_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_cuda_activate_context_v1 activate_context;
  pih_cuda_validate_pinned_host_v1 validate_pinned_host;
  pih_cuda_copy_async_v1 copy_async;
  pih_cuda_memset_async_v1 memset_async;
  pih_cuda_record_event_v1 record_event;
  pih_cuda_query_event_v1 query_event;
  pih_cuda_synchronize_stream_v1 synchronize_stream;
  pih_cuda_require_clean_last_error_v1 require_clean_last_error;
} pih_nvidia_cuda_async_api_v1;

#ifdef __cplusplus
}
#endif
