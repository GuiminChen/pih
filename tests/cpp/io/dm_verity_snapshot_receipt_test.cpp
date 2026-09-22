#include "pih/io/dm_verity_snapshot_receipt.h"

#include <cstddef>

#include <gtest/gtest.h>

namespace pih { namespace {

DmVeritySupervisorReceipt receipt() {
  DmVeritySupervisorReceipt value;
  value.geometry.version = 1;
  value.geometry.num_data_blocks = 128;
  value.geometry.data_block_bytes = 4096;
  value.geometry.hash_block_bytes = 4096;
  value.geometry.hash_start_block = 128;
  value.geometry.data_device_identity = 10;
  value.geometry.data_device_blocks = 1000;
  value.geometry.hash_device_identity = 11;
  value.geometry.hash_device_blocks = 1000;
  value.geometry.digest_slot_bytes = 32;
  value.geometry.fec_roots = 0;
  value.geometry.fec_blocks = 0;
  value.geometry.fec_start_block = 0;
  value.root_digest.bytes[0] = std::byte{1};
  value.table_digest.bytes[0] = std::byte{2};
  value.supervisor_attestation_digest.bytes[0] = std::byte{3};
  value.mount_identity = 42;
  value.active_table = true;
  value.read_only = true;
  value.no_writable_alias = true;
  value.authenticated = true;
  return value;
}

TEST(DmVeritySnapshotReceiptTest, ComputesTreeFromWholeDeviceBlocks) {
  auto value = receipt();
  auto validated = validate_dm_verity_supervisor_receipt(
      value, value.root_digest, value.table_digest,
      value.supervisor_attestation_digest, 42, 1536ULL * 1024 * 1024);
  ASSERT_TRUE(validated.ok()) << validated.status().message();
  EXPECT_EQ(validated->tree_levels, 1U);
  EXPECT_EQ(validated->tree_blocks, 1U);
  EXPECT_EQ(validated->tree_bytes, 4096U);

  value.geometry.num_data_blocks = 129;
  validated = validate_dm_verity_supervisor_receipt(
      value, value.root_digest, value.table_digest,
      value.supervisor_attestation_digest, 42, 1536ULL * 1024 * 1024);
  ASSERT_TRUE(validated.ok());
  EXPECT_EQ(validated->tree_levels, 2U);
  EXPECT_EQ(validated->tree_blocks, 3U);
}

TEST(DmVeritySnapshotReceiptTest, IncludesFecModeAndCacheInReserve) {
  auto value = receipt();
  value.geometry.fec_roots = 2;
  value.geometry.fec_blocks = 10;
  value.geometry.fec_start_block = 200;
  value.geometry.fec_device_identity = 12;
  value.geometry.fec_device_blocks = 1000;
  value.mode_overhead_bytes = 1000;
  value.integrity_cache_high_water_bytes = 2000;
  auto validated = validate_dm_verity_supervisor_receipt(
      value, value.root_digest, value.table_digest,
      value.supervisor_attestation_digest, 42, 100000);
  ASSERT_TRUE(validated.ok());
  EXPECT_EQ(validated->fec_bytes, 40960U);
  EXPECT_EQ(validated->total_integrity_bytes, 48056U);
  EXPECT_FALSE(validate_dm_verity_supervisor_receipt(
                   value, value.root_digest, value.table_digest,
                   value.supervisor_attestation_digest, 42, 48055)
                   .ok());
}

TEST(DmVeritySnapshotReceiptTest, RejectsUntrustedOrMismatchedSnapshot) {
  auto value = receipt();
  auto expected = value.root_digest;
  expected.bytes[1] = std::byte{1};
  EXPECT_FALSE(validate_dm_verity_supervisor_receipt(
                   value, expected, value.table_digest,
                   value.supervisor_attestation_digest, 42,
                   1536ULL * 1024 * 1024)
                   .ok());
  EXPECT_FALSE(validate_dm_verity_supervisor_receipt(
                   value, value.root_digest, value.table_digest,
                   value.supervisor_attestation_digest, 43,
                   1536ULL * 1024 * 1024)
                   .ok());
  value.no_writable_alias = false;
  EXPECT_FALSE(validate_dm_verity_supervisor_receipt(
                   value, value.root_digest, value.table_digest,
                   value.supervisor_attestation_digest, 42,
                   1536ULL * 1024 * 1024)
                   .ok());
}

TEST(DmVeritySnapshotReceiptTest, RejectsArtifactSizedOrInvalidGeometry) {
  auto value = receipt();
  value.geometry.num_data_blocks = 0;
  EXPECT_FALSE(validate_dm_verity_supervisor_receipt(
                   value, value.root_digest, value.table_digest,
                   value.supervisor_attestation_digest, 42,
                   1536ULL * 1024 * 1024)
                   .ok());
  value = receipt();
  value.geometry.digest_slot_bytes = 64;
  EXPECT_FALSE(validate_dm_verity_supervisor_receipt(
                   value, value.root_digest, value.table_digest,
                   value.supervisor_attestation_digest, 42,
                   1536ULL * 1024 * 1024)
                   .ok());
}

} }  // namespace pih
