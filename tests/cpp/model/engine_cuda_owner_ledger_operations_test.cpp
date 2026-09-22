#include "pih/model/engine_cuda_owner_ledger_operations.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest ledger_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

EngineCudaOwnerLedgerSnapshot valid_snapshot(std::uint64_t sample_identity) {
  return EngineCudaOwnerLedgerSnapshot{
      sample_identity,
      sample_identity * 100,
      sample_identity * 100 + 10,
      {EnginePinnedMemoryObservation{60, true, true, ledger_digest(9), 0,
                                     4096},
       EnginePinnedMemoryObservation{61, false, true, {}, -1, 0}},
      {EngineGpuAllocationObservation{70, true, true, ledger_digest(9),
                                      8192},
       EngineGpuAllocationObservation{71, false, true, {}, 0}}};
}

class LedgerBackend final : public EngineCudaOwnerLedgerBackend {
 public:
  Result<EngineCudaOwnerLedgerSnapshot> capture(
      std::span<const std::uint64_t> pinned_identities,
      std::span<const std::uint64_t> allocation_identities) override {
    ++calls;
    pinned_requests.emplace_back(pinned_identities.begin(),
                                 pinned_identities.end());
    allocation_requests.emplace_back(allocation_identities.begin(),
                                     allocation_identities.end());
    if (!status.ok()) return status;
    return snapshots.at(next_snapshot++);
  }

  int calls = 0;
  Status status = Status::Ok();
  std::size_t next_snapshot = 0;
  std::vector<EngineCudaOwnerLedgerSnapshot> snapshots{valid_snapshot(1),
                                                        valid_snapshot(2)};
  std::vector<std::vector<std::uint64_t>> pinned_requests;
  std::vector<std::vector<std::uint64_t>> allocation_requests;
};

std::array<std::uint64_t, 2> pinned_ids() { return {60, 61}; }
std::array<std::uint64_t, 2> allocation_ids() { return {70, 71}; }

TEST(EngineCudaOwnerLedgerOperationsTest,
     CapturesBothOwnerSetsForEveryCategoryRequest) {
  LedgerBackend backend;
  auto operations = EngineCudaOwnerLedgerOperations::Create(
                        pinned_ids(), allocation_ids(), backend)
                        .value();

  auto pinned = operations->capture_pinned(pinned_ids());
  auto allocations = operations->capture_allocations(allocation_ids());

  ASSERT_TRUE(pinned.ok());
  ASSERT_TRUE(allocations.ok());
  ASSERT_EQ(pinned->size(), 2U);
  ASSERT_EQ(allocations->size(), 2U);
  EXPECT_EQ(pinned->at(0).registration_identity, 60U);
  EXPECT_EQ(allocations->at(0).allocation_identity, 70U);
  ASSERT_EQ(backend.calls, 2);
  EXPECT_EQ(backend.pinned_requests[0],
            std::vector<std::uint64_t>({60, 61}));
  EXPECT_EQ(backend.allocation_requests[1],
            std::vector<std::uint64_t>({70, 71}));
}

TEST(EngineCudaOwnerLedgerOperationsTest,
     RejectsUnexpectedRequestBeforeSampling) {
  LedgerBackend backend;
  auto operations = EngineCudaOwnerLedgerOperations::Create(
                        pinned_ids(), allocation_ids(), backend)
                        .value();
  const std::array<std::uint64_t, 2> reversed_pinned{61, 60};
  const std::array<std::uint64_t, 1> partial_allocations{70};

  EXPECT_EQ(operations->capture_pinned(reversed_pinned).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(operations->capture_allocations(partial_allocations).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(backend.calls, 0);
}

TEST(EngineCudaOwnerLedgerOperationsTest,
     RejectsAndPoisonsPartialOrReorderedCounterpart) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    LedgerBackend backend;
    if (mutation == 0) backend.snapshots[0].allocations.pop_back();
    if (mutation == 1)
      std::swap(backend.snapshots[0].allocations[0],
                backend.snapshots[0].allocations[1]);
    auto operations = EngineCudaOwnerLedgerOperations::Create(
                          pinned_ids(), allocation_ids(), backend)
                          .value();

    auto first = operations->capture_pinned(pinned_ids());
    ASSERT_FALSE(first.ok()) << mutation;
    EXPECT_EQ(first.status().code(), StatusCode::kFailedPrecondition)
        << mutation;
    EXPECT_EQ(operations->capture_pinned(pinned_ids()).status().code(),
              StatusCode::kFailedPrecondition)
        << mutation;
    EXPECT_EQ(backend.calls, 1) << mutation;
  }
}

TEST(EngineCudaOwnerLedgerOperationsTest,
     TreatsUnavailableAsRecoverableEvidenceGap) {
  LedgerBackend backend;
  auto operations = EngineCudaOwnerLedgerOperations::Create(
                        pinned_ids(), allocation_ids(), backend)
                        .value();
  backend.status = Status::Unavailable("driver ledger unavailable");
  EXPECT_EQ(operations->capture_pinned(pinned_ids()).status().code(),
            StatusCode::kUnavailable);
  backend.status = Status::Ok();
  auto recovered = operations->capture_pinned(pinned_ids());
  ASSERT_TRUE(recovered.ok());
  EXPECT_EQ(recovered->at(0).registration_identity, 60U);
  EXPECT_EQ(backend.calls, 2);
}

TEST(EngineCudaOwnerLedgerOperationsTest,
     RejectsAndPoisonsInvalidOrRegressingSampleWindow) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    LedgerBackend backend;
    if (mutation == 0) backend.snapshots[0].sample_identity = 0;
    if (mutation == 1)
      backend.snapshots[0].sample_completed_ns =
          backend.snapshots[0].sample_started_ns - 1;
    if (mutation == 2)
      backend.snapshots[1].sample_identity =
          backend.snapshots[0].sample_identity;
    if (mutation == 3)
      backend.snapshots[1].sample_started_ns =
          backend.snapshots[0].sample_completed_ns - 1;
    auto operations = EngineCudaOwnerLedgerOperations::Create(
                          pinned_ids(), allocation_ids(), backend)
                          .value();
    auto first = operations->capture_pinned(pinned_ids());
    if (mutation < 2) {
      EXPECT_EQ(first.status().code(), StatusCode::kFailedPrecondition)
          << mutation;
    } else {
      ASSERT_TRUE(first.ok()) << mutation;
      EXPECT_EQ(operations->capture_allocations(allocation_ids()).status().code(),
                StatusCode::kFailedPrecondition)
          << mutation;
    }
  }
}

TEST(EngineCudaOwnerLedgerOperationsTest, RejectsInvalidConfiguredIdentities) {
  LedgerBackend backend;
  const std::array<std::uint64_t, 2> duplicate_pinned{60, 60};
  const std::array<std::uint64_t, 2> overlapping_allocations{60, 71};
  EXPECT_FALSE(EngineCudaOwnerLedgerOperations::Create(
                   std::span<const std::uint64_t>{}, allocation_ids(), backend)
                   .ok());
  EXPECT_FALSE(EngineCudaOwnerLedgerOperations::Create(
                   duplicate_pinned, allocation_ids(), backend)
                   .ok());
  EXPECT_FALSE(EngineCudaOwnerLedgerOperations::Create(
                   pinned_ids(), overlapping_allocations, backend)
                   .ok());
}

}  // namespace
}  // namespace pih
