#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#define PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 1U

typedef struct pih_verified_artifact_generation_request_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* generation_root;
  const char* root_sha256_hex;
  uint64_t maximum_shard_bytes;
} pih_verified_artifact_generation_request_v1;

typedef pih_status_v1 (*pih_verified_artifact_validate_generation_v1)(
    void* context,
    const pih_verified_artifact_generation_request_v1* request);

typedef struct pih_verified_artifact_lease_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t handle;
  uint64_t file_bytes;
  uint64_t filesystem_identity;
  uint64_t file_identity;
  int64_t data_mtime_seconds;
  uint32_t data_mtime_nanoseconds;
  uint32_t reserved;
} pih_verified_artifact_lease_v1;

typedef struct pih_verified_artifact_mapping_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t handle;
  uint64_t lease_handle;
  uintptr_t address;
  uint64_t file_offset;
  uint64_t bytes;
} pih_verified_artifact_mapping_v1;

typedef pih_status_v1 (*pih_verified_artifact_open_development_lease_v1)(
    void* context, const char* trusted_root, const char* member_basename,
    uint64_t maximum_bytes, pih_verified_artifact_lease_v1* lease);
typedef pih_status_v1 (*pih_verified_artifact_read_lease_v1)(
    void* context, uint64_t handle, uint64_t offset, void* destination,
    uint64_t bytes);
typedef pih_status_v1 (*pih_verified_artifact_poll_lease_v1)(
    void* context, uint64_t handle);
typedef pih_status_v1 (*pih_verified_artifact_release_lease_v1)(
    void* context, pih_verified_artifact_lease_v1* lease);
typedef pih_status_v1 (*pih_verified_artifact_map_lease_v1)(
    void* context, uint64_t lease_handle, uint64_t file_offset,
    uint64_t bytes, pih_verified_artifact_mapping_v1* mapping);
typedef pih_status_v1 (*pih_verified_artifact_release_mapping_v1)(
    void* context, pih_verified_artifact_mapping_v1* mapping);

typedef struct pih_verified_artifact_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_verified_artifact_validate_generation_v1 validate_generation;
  pih_verified_artifact_open_development_lease_v1 open_development_lease;
  pih_verified_artifact_read_lease_v1 read_lease;
  pih_verified_artifact_poll_lease_v1 poll_lease;
  pih_verified_artifact_release_lease_v1 release_lease;
  pih_verified_artifact_map_lease_v1 map_lease;
  pih_verified_artifact_release_mapping_v1 release_mapping;
} pih_verified_artifact_api_v1;
