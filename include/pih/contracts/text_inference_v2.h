#pragma once
#include <stdint.h>
#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serialized activation-owned engine. No C++ objects, Python objects or
// allocator ownership cross this boundary. All buffers belong to the caller.
#define PIH_TEXT_INFERENCE_ABI_V2 2U
// Callback table borrowed for one synchronous complete() call. Callbacks must
// be invoked on the calling thread, not throw or reenter the engine.
// start/write return 1 on success, 0 to cancel;
// cancelled returns 0 to keep going, nonzero to request cancellation.
// write receives a single UTF-8 JSON SSE data value (without data:/newlines).
// start is called only after request validation, before model submission.
// start may be called at most once. write is valid only after a successful
// streaming start. A failed start/write must terminate output; the provider
// must retire submitted work before returning and must not report success.
typedef struct pih_text_output_sink_v2 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  uint32_t (*start)(void*, uint32_t streaming);
  uint32_t (*write)(void*, const char* json, uint64_t bytes);
  uint32_t (*cancelled)(void*);
} pih_text_output_sink_v2;
typedef struct pih_text_inference_api_v2 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  const char* model_id;
  pih_status_v1 (*load)(void*, const char* configuration, uint64_t bytes);
  pih_status_v1 (*complete)(void*, uint32_t chat, const char* request,
      uint64_t request_bytes, char* response, uint64_t capacity,
      uint64_t* response_bytes, const pih_text_output_sink_v2* sink);
  // Must be safe before load, after a failed/partial load, and after completion.
  // OK means all model-owned execution has retired; UNAVAILABLE retains state
  // and permits a bounded caller retry. Other errors are terminal: the caller
  // must not unload providers that may still be in use. Repeated close after OK
  // is safe. No exception may cross this C ABI boundary.
  pih_status_v1 (*close)(void*);
} pih_text_inference_api_v2;

typedef struct pih_text_service_api_v2 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  // Loopback only. Blocks until SIGINT/SIGTERM or an unrecoverable error.
  // The caller loads the selected model before entering this operation.
  pih_status_v1 (*serve)(void*, uint16_t port);
} pih_text_service_api_v2;

#ifdef __cplusplus
}
#endif
