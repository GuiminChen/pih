#pragma once

#include <stddef.h>
#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#define PIH_HEALTH_ABI_VERSION_V1 1U
#define PIH_HEALTH_READINESS_ENGINE_GENERATION_V1 1U
#define PIH_HEALTH_ACCEPTED_SOCKET_CALLER_OWNED_V1 1U

typedef struct pih_health_snapshot_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t ready;
  uint32_t live;
  uint32_t reason;
  uint64_t activation_epoch;
  uint64_t generation;
  uint64_t revision;
} pih_health_snapshot_v1;

enum {
  PIH_HEALTH_REASON_ENGINE_STARTING_V1 = 1,
  PIH_HEALTH_REASON_READY_V1 = 2,
  PIH_HEALTH_REASON_DRAINING_V1 = 3,
};

typedef pih_status_v1 (*pih_health_snapshot_operation_v1)(
    void* context, pih_health_snapshot_v1* snapshot);
typedef pih_status_v1 (*pih_health_publish_readiness_v1)(
    void* context, uint64_t engine_generation, uint32_t ready);
typedef pih_status_v1 (*pih_health_bind_activation_v1)(
    void* context, uint64_t activation_epoch);
typedef pih_status_v1 (*pih_health_publish_capabilities_v1)(
    void* context, const char* capabilities_json, size_t capabilities_size);

typedef struct pih_health_http_exchange_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* request_bytes;
  size_t request_size;
  char* response_bytes;
  size_t response_capacity;
  size_t response_size;
} pih_health_http_exchange_v1;

typedef pih_status_v1 (*pih_health_execute_http_exchange_v1)(
    void* context, pih_health_http_exchange_v1* exchange);
typedef pih_status_v1 (*pih_health_execute_http_connection_v1)(
    void* context, intptr_t accepted_socket, uint64_t deadline_monotonic_ns);

typedef struct pih_health_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t readiness_semantics;
  uint32_t accepted_socket_ownership;
  void* context;
  pih_health_snapshot_operation_v1 snapshot;
  pih_health_bind_activation_v1 bind_activation;
  pih_health_publish_capabilities_v1 publish_capabilities;
  pih_health_publish_readiness_v1 publish_readiness;
  pih_health_execute_http_exchange_v1 execute_http_exchange;
  pih_health_execute_http_connection_v1 execute_http_connection;
} pih_health_api_v1;
