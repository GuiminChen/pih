#pragma once

#include <stddef.h>
#include <stdint.h>

#include "pih/contracts/engine_v1.h"
#include "pih/plugin_sdk/status.h"

#define PIH_OPENAI_HTTP_ABI_VERSION_V1 1U
#define PIH_OPENAI_HTTP_ACCEPTED_SOCKET_CALLER_OWNED_V1 1U

typedef pih_status_v1 (*pih_openai_http_bind_engine_v1)(
    void* context, const pih_engine_handle_v1* engine);
typedef pih_status_v1 (*pih_openai_http_operation_v1)(void* context);

typedef struct pih_openai_http_exchange_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* request_bytes;
  size_t request_size;
  char* response_bytes;
  size_t response_capacity;
  size_t response_size;
} pih_openai_http_exchange_v1;

typedef pih_status_v1 (*pih_openai_http_execute_smoke_exchange_v1)(
    void* context, pih_openai_http_exchange_v1* exchange);
typedef pih_status_v1 (*pih_openai_http_execute_smoke_connection_v1)(
    void* context, intptr_t accepted_socket, uint64_t deadline_monotonic_ns);

typedef struct pih_openai_http_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t accepted_socket_ownership;
  void* context;
  pih_openai_http_bind_engine_v1 bind_engine;
  pih_openai_http_operation_v1 unbind_engine;
  pih_openai_http_execute_smoke_exchange_v1 execute_smoke_exchange;
  pih_openai_http_execute_smoke_connection_v1 execute_smoke_connection;
} pih_openai_http_api_v1;
