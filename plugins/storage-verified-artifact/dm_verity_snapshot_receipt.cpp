#include "pih/io/dm_verity_snapshot_receipt.h"

#include <algorithm>

#include "pih/core/checked_math.h"

namespace pih { namespace {

bool nonzero(const Sha256Digest& digest) {
  return std::ranges::any_of(digest.bytes,
                             [](std::byte value) { return value != std::byte{}; });
}

}  // namespace

Result<DmVerityCapacityReceipt> validate_dm_verity_supervisor_receipt(
    const DmVeritySupervisorReceipt& receipt,
    const Sha256Digest& expected_root_digest,
    const Sha256Digest& expected_table_digest,
    const Sha256Digest& expected_supervisor_attestation_digest,
    std::uint64_t observed_mount_identity,
    std::uint64_t integrity_reserve_bytes) {
  const auto& geometry = receipt.geometry;
  if (geometry.version != 1 || geometry.num_data_blocks == 0 ||
      geometry.data_block_bytes != 4096 ||
      geometry.hash_block_bytes != 4096 ||
      geometry.digest_slot_bytes != 32 ||
      geometry.data_device_identity == 0 ||
      geometry.hash_device_identity == 0 ||
      geometry.data_device_blocks < geometry.num_data_blocks ||
      integrity_reserve_bytes == 0) {
    return Status::InvalidArgument("dm-verity geometry is outside V1 profile");
  }
  if (!receipt.active_table || !receipt.read_only ||
      !receipt.no_writable_alias || !receipt.authenticated ||
      receipt.mount_identity == 0 ||
      receipt.mount_identity != observed_mount_identity ||
      !nonzero(receipt.root_digest) || !nonzero(receipt.table_digest) ||
      !nonzero(receipt.supervisor_attestation_digest) ||
      receipt.root_digest != expected_root_digest ||
      receipt.table_digest != expected_table_digest ||
      receipt.supervisor_attestation_digest !=
          expected_supervisor_attestation_digest) {
    return Status::FailedPrecondition(
        "dm-verity supervisor authority or snapshot identity differs");
  }
  if ((geometry.fec_roots == 0 &&
       (geometry.fec_blocks != 0 || geometry.fec_start_block != 0 ||
        geometry.fec_device_identity != 0 ||
        geometry.fec_device_blocks != 0)) ||
      (geometry.fec_roots != 0 &&
       (geometry.fec_blocks == 0 || geometry.fec_start_block == 0 ||
        geometry.fec_device_identity == 0 ||
        geometry.fec_device_blocks == 0))) {
    return Status::InvalidArgument("dm-verity FEC geometry is inconsistent");
  }

  DmVerityCapacityReceipt result;
  const auto hashes_per_block =
      geometry.hash_block_bytes / geometry.digest_slot_bytes;
  std::uint64_t blocks = geometry.num_data_blocks;
  while (blocks > 1) {
    auto numerator = checked_add_u64(blocks, hashes_per_block - 1);
    if (!numerator.ok()) return numerator.status();
    blocks = *numerator / hashes_per_block;
    auto total = checked_add_u64(result.tree_blocks, blocks);
    if (!total.ok()) return total.status();
    result.tree_blocks = *total;
    ++result.tree_levels;
  }
  auto tree_bytes = checked_mul_u64(result.tree_blocks,
                                    geometry.hash_block_bytes);
  if (!tree_bytes.ok()) return tree_bytes.status();
  result.tree_bytes = *tree_bytes;
  auto hash_end = checked_add_u64(geometry.hash_start_block,
                                  result.tree_blocks);
  if (!hash_end.ok()) return hash_end.status();
  if (*hash_end > geometry.hash_device_blocks) {
    return Status::InvalidArgument(
        "dm-verity hash tree exceeds hash device capacity");
  }
  auto fec_bytes = checked_mul_u64(geometry.fec_blocks,
                                   geometry.data_block_bytes);
  if (!fec_bytes.ok()) return fec_bytes.status();
  result.fec_bytes = *fec_bytes;
  if (geometry.fec_roots != 0) {
    auto fec_end = checked_add_u64(geometry.fec_start_block,
                                   geometry.fec_blocks);
    if (!fec_end.ok()) return fec_end.status();
    if (*fec_end > geometry.fec_device_blocks) {
      return Status::InvalidArgument(
          "dm-verity FEC range exceeds FEC device capacity");
    }
  }
  auto total = checked_add_u64(result.tree_bytes, result.fec_bytes);
  if (!total.ok()) return total.status();
  total = checked_add_u64(*total, receipt.mode_overhead_bytes);
  if (!total.ok()) return total.status();
  total = checked_add_u64(*total,
                          receipt.integrity_cache_high_water_bytes);
  if (!total.ok()) return total.status();
  result.total_integrity_bytes = *total;
  if (result.total_integrity_bytes > integrity_reserve_bytes) {
    return Status::ResourceExhausted(
        "dm-verity full-device integrity owner exceeds storage reserve");
  }
  return result;
}

}  // namespace pih
