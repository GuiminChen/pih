#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

struct DmVerityGeometry final {
  std::uint32_t version = 0;
  std::uint64_t num_data_blocks = 0;
  std::uint32_t data_block_bytes = 0;
  std::uint32_t hash_block_bytes = 0;
  std::uint64_t hash_start_block = 0;
  std::uint64_t data_device_identity = 0;
  std::uint64_t data_device_blocks = 0;
  std::uint64_t hash_device_identity = 0;
  std::uint64_t hash_device_blocks = 0;
  std::uint32_t digest_slot_bytes = 0;
  std::uint32_t fec_roots = 0;
  std::uint64_t fec_blocks = 0;
  std::uint64_t fec_start_block = 0;
  std::uint64_t fec_device_identity = 0;
  std::uint64_t fec_device_blocks = 0;
};

struct DmVeritySupervisorReceipt final {
  DmVerityGeometry geometry;
  Sha256Digest root_digest;
  Sha256Digest table_digest;
  Sha256Digest supervisor_attestation_digest;
  std::uint64_t mount_identity = 0;
  std::uint64_t mode_overhead_bytes = 0;
  std::uint64_t integrity_cache_high_water_bytes = 0;
  bool active_table = false;
  bool read_only = false;
  bool no_writable_alias = false;
  bool authenticated = false;
};

struct DmVerityCapacityReceipt final {
  std::uint32_t tree_levels = 0;
  std::uint64_t tree_blocks = 0;
  std::uint64_t tree_bytes = 0;
  std::uint64_t fec_bytes = 0;
  std::uint64_t total_integrity_bytes = 0;
};

Result<DmVerityCapacityReceipt> validate_dm_verity_supervisor_receipt(
    const DmVeritySupervisorReceipt& receipt,
    const Sha256Digest& expected_root_digest,
    const Sha256Digest& expected_table_digest,
    const Sha256Digest& expected_supervisor_attestation_digest,
    std::uint64_t observed_mount_identity,
    std::uint64_t integrity_reserve_bytes);

}  // namespace pih
