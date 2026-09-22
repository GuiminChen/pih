#pragma once

#include <stdint.h>
#include "pih/plugin_sdk/status.h"

#define PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1 1U

typedef struct pih_artifact_snapshot_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t handle;
  uintptr_t address;
  uint64_t bytes;
} pih_artifact_snapshot_v1;

// open_snapshot authenticates the complete, bounded member against a caller-
// supplied SHA-256 before returning a read-only immutable byte extent. The
// address stays valid until release_snapshot succeeds; consumers must finish
// every read or copy before releasing it. A failed release does not transfer
// ownership back to the caller and the provider must remain loaded.
typedef pih_status_v1 (*pih_artifact_open_snapshot_v1)(
    void* context, const char* trusted_root, const char* member_basename,
    const char* expected_sha256_hex, uint64_t maximum_bytes,
    pih_artifact_snapshot_v1* snapshot);
typedef pih_status_v1 (*pih_artifact_release_snapshot_v1)(
    void* context, pih_artifact_snapshot_v1* snapshot);

typedef struct pih_artifact_snapshot_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_artifact_open_snapshot_v1 open_snapshot;
  pih_artifact_release_snapshot_v1 release_snapshot;
} pih_artifact_snapshot_api_v1;
