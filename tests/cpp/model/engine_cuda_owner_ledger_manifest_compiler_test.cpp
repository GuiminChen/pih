#include "pih/model/engine_cuda_owner_ledger_manifest_compiler.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest compiler_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

std::array<EnginePinnedMemoryBinding, 2> pinned_bindings() {
  return {EnginePinnedMemoryBinding{60, compiler_digest(1), compiler_digest(2),
                                    compiler_digest(9), 3, 4096},
          EnginePinnedMemoryBinding{61, compiler_digest(3), compiler_digest(4),
                                    compiler_digest(9), 3, 8192}};
}

std::array<EngineGpuAllocationBinding, 2> allocation_bindings() {
  return {EngineGpuAllocationBinding{70, compiler_digest(5), compiler_digest(6),
                                     compiler_digest(9), 16384},
          EngineGpuAllocationBinding{71, compiler_digest(7), compiler_digest(8),
                                     compiler_digest(9), 32768}};
}

EngineCudaOwnerLedgerSnapshot owner_snapshot() {
  return {17,
          100,
          110,
          {{60, true, true, compiler_digest(9), 3, 4096},
           {61, true, true, compiler_digest(9), 3, 8192}},
          {{70, true, true, compiler_digest(9), 16384},
           {71, true, true, compiler_digest(9), 32768}}};
}

TEST(EngineCudaOwnerLedgerManifestCompilerTest,
     CompilesBothKindsFromOneAtomicSample) {
  auto result = compile_engine_cuda_owner_ledger_manifest(
      owner_snapshot(), pinned_bindings(), allocation_bindings());
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->sample_identity, 17U);
  EXPECT_EQ(result->sample_started_ns, 100U);
  EXPECT_EQ(result->sample_completed_ns, 110U);
  EXPECT_NE(result->pinned_census_digest, Sha256Digest{});
  EXPECT_NE(result->allocation_census_digest, Sha256Digest{});
  ASSERT_EQ(result->pinned_records.size(), 2U);
  ASSERT_EQ(result->allocation_records.size(), 2U);
  EXPECT_EQ(result->pinned_records[0].kind,
            EngineOwnedResourceKind::kPinnedMemory);
  EXPECT_EQ(result->pinned_records[0].backing_bytes, 4096U);
  EXPECT_EQ(result->allocation_records[1].kind,
            EngineOwnedResourceKind::kGpuAllocation);
  EXPECT_EQ(result->allocation_records[1].owner_identity,
            compiler_digest(7));
}

TEST(EngineCudaOwnerLedgerManifestCompilerTest,
     CensusDigestsBindTheirOwnAtomicRecordSet) {
  auto first = compile_engine_cuda_owner_ledger_manifest(
      owner_snapshot(), pinned_bindings(), allocation_bindings()).value();
  auto snapshot = owner_snapshot();
  snapshot.pinned[1].registered = false;
  snapshot.pinned[1].registered_bytes = 0;
  auto second = compile_engine_cuda_owner_ledger_manifest(
      snapshot, pinned_bindings(), allocation_bindings()).value();
  EXPECT_NE(first.pinned_census_digest, second.pinned_census_digest);
  EXPECT_EQ(first.allocation_census_digest, second.allocation_census_digest);
}

TEST(EngineCudaOwnerLedgerManifestCompilerTest,
     OmitsReleasedTombstonesAcrossBothKinds) {
  auto snapshot = owner_snapshot();
  snapshot.pinned[1].registered = false;
  snapshot.pinned[1].registered_bytes = 0;
  snapshot.allocations[1].allocated = false;
  snapshot.allocations[1].allocated_bytes = 0;
  auto result = compile_engine_cuda_owner_ledger_manifest(
      snapshot, pinned_bindings(), allocation_bindings());
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->pinned_records.size(), 1U);
  EXPECT_EQ(result->allocation_records.size(), 1U);
}

TEST(EngineCudaOwnerLedgerManifestCompilerTest,
     AllowsDeviceOnlyCensusWithZeroPinnedSentinel) {
  auto snapshot = owner_snapshot();
  snapshot.pinned.clear();
  const std::array<EnginePinnedMemoryBinding, 0> no_pinned{};
  auto result = compile_engine_cuda_owner_ledger_manifest(
      snapshot, no_pinned, allocation_bindings());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_TRUE(result->pinned_records.empty());
  EXPECT_EQ(result->pinned_census_digest, Sha256Digest{});
  EXPECT_NE(result->allocation_census_digest, Sha256Digest{});
}

TEST(EngineCudaOwnerLedgerManifestCompilerTest,
     RejectsSampleAndCrossCategoryManifestDrift) {
  for (int mutation = 0; mutation < 10; ++mutation) {
    auto snapshot = owner_snapshot();
    auto pinned = pinned_bindings();
    auto allocations = allocation_bindings();
    if (mutation == 0) snapshot.sample_identity = 0;
    if (mutation == 1) snapshot.sample_completed_ns = 99;
    if (mutation == 2) snapshot.pinned.pop_back();
    if (mutation == 3) ++snapshot.pinned[0].registration_identity;
    if (mutation == 4) snapshot.pinned[0].owner_counter_visible = false;
    if (mutation == 5) ++snapshot.pinned[0].registered_bytes;
    if (mutation == 6) snapshot.allocations[0].physical_gpu_identity =
                           compiler_digest(10);
    if (mutation == 7) allocations[0].allocation_identity = 60;
    if (mutation == 8) allocations[0].owner_identity = pinned[0].owner_identity,
                       allocations[0].resource_identity =
                           pinned[0].resource_identity;
    if (mutation == 9) pinned[0].registered_bytes = 0;
    auto result = compile_engine_cuda_owner_ledger_manifest(
        snapshot, pinned, allocations);
    ASSERT_FALSE(result.ok()) << mutation;
    EXPECT_EQ(result.status().code(),
              mutation == 4 ? StatusCode::kUnavailable
                            : (mutation >= 7 ? StatusCode::kInvalidArgument
                                             : StatusCode::kFailedPrecondition))
        << mutation;
  }
}

}  // namespace
}  // namespace pih
