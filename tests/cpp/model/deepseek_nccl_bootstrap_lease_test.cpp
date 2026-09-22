#include "pih/model/deepseek_nccl_bootstrap_lease.h"
#include "pih/model/deepseek_nccl_bootstrap_capability_issuer.h"
#include "pih/model/deepseek_nccl_boundary_warmup_gate.h"
#include "pih/model/deepseek_nccl_endpoint_manifest_plan.h"
#include "pih/model/deepseek_nccl_endpoint_reconciliation_gate.h"

#include <gtest/gtest.h>

#include <array>
#include <new>

namespace pih {
namespace {

class LeaseAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    if (bytes != 128 || alignment != 64) {
      return Status::InvalidArgument("unexpected lease allocation");
    }
    data = ::operator new(128, std::align_val_t{64});
    return Allocation{data, 128, 64, 1, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    observed_zero = true;
    for (std::size_t i = 0; i < allocation.bytes; ++i) {
      observed_zero = observed_zero &&
          static_cast<std::byte*>(allocation.data)[i] == std::byte{0};
    }
    ::operator delete(allocation.data, std::align_val_t{64});
    data = nullptr;
  }
  void* data = nullptr;
  bool observed_zero = false;
};

class IssuerAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    if (fail_after >= 0 && allocations >= fail_after) {
      return Status::ResourceExhausted("injected pinned allocation failure");
    }
    auto* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t{64});
    ++allocations;
    ++active;
    return Allocation{data, bytes, alignment,
                      static_cast<std::uint64_t>(allocations), Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    bool zero = true;
    for (std::size_t i = 0; i < allocation.bytes; ++i) {
      zero = zero && static_cast<std::byte*>(allocation.data)[i] == std::byte{0};
    }
    if (zero) ++zeroized_deallocations;
    ::operator delete(allocation.data, std::align_val_t{64});
    --active;
  }
  int fail_after = -1;
  int allocations = 0;
  int active = 0;
  int zeroized_deallocations = 0;
};

class IssuerSource final : public DeepSeekNcclUniqueIdSource {
 public:
  Result<std::array<std::byte, 128>> get_unique_id() override {
    ++calls;
    if (fail_at == calls) return Status::Internal("injected unique ID failure");
    std::array<std::byte, 128> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = static_cast<std::byte>(i + 1);
    }
    result[0] = static_cast<std::byte>(calls);
    return result;
  }
  int calls = 0;
  int fail_at = -1;
};

std::array<std::byte, 128> unique_id() {
  std::array<std::byte, 128> result{};
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<std::byte>(i + 1);
  }
  return result;
}

TEST(DeepSeekNcclBootstrapLeaseTest, BindsAndZeroizesEphemeralCapability) {
  LeaseAllocator allocator;
  const auto raw = unique_id();
  {
    auto lease = DeepSeekNcclBootstrapLease::Create(allocator, raw, 7, 2, 11);
    ASSERT_TRUE(lease.ok()) << lease.status().message();
    EXPECT_FALSE(lease->commitment().hex().empty());
    auto borrowed = lease->borrow(7, 2, 11);
    ASSERT_TRUE(borrowed.ok());
    EXPECT_TRUE(std::equal(borrowed->begin(), borrowed->end(), raw.begin()));
    EXPECT_FALSE(lease->borrow(8, 2, 11).ok());
    ASSERT_TRUE(lease->zeroize().ok());
    EXPECT_TRUE(lease->zeroized());
    EXPECT_FALSE(lease->borrow(7, 2, 11).ok());
    EXPECT_TRUE(lease->zeroize().ok());
  }
  EXPECT_TRUE(allocator.observed_zero);
}

TEST(DeepSeekNcclBootstrapLeaseTest, RejectsWrongSizeAndZeroIdentity) {
  LeaseAllocator allocator;
  std::array<std::byte, 127> short_id{};
  EXPECT_FALSE(DeepSeekNcclBootstrapLease::Create(
      allocator, short_id, 1, 0, 1).ok());
  const auto raw = unique_id();
  EXPECT_FALSE(DeepSeekNcclBootstrapLease::Create(
      allocator, raw, 0, 0, 1).ok());
  EXPECT_EQ(allocator.data, nullptr);
}

TEST(DeepSeekNcclBootstrapCapabilityIssuerTest,
     IssuesExactlyOneFreshCapabilityPerEdge) {
  IssuerAllocator allocator;
  IssuerSource source;
  {
    auto issued = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
        7, 4, 100, source, allocator);
    ASSERT_TRUE(issued.ok()) << issued.status().message();
    ASSERT_EQ(issued->size(), 3U);
    EXPECT_EQ(source.calls, 3);
    EXPECT_EQ(allocator.active, 6);
    for (std::uint32_t edge = 0; edge < 3; ++edge) {
      EXPECT_EQ((*issued)[edge].edge_id, edge);
      EXPECT_EQ((*issued)[edge].lease_id, 100U + edge);
      EXPECT_NE((*issued)[edge].commitment_id, 0U);
      EXPECT_NE((*issued)[edge].lower, nullptr);
      EXPECT_NE((*issued)[edge].upper, nullptr);
      EXPECT_EQ((*issued)[edge].lower->commitment(),
                (*issued)[edge].upper->commitment());
      if (edge > 0) {
        EXPECT_NE((*issued)[edge].commitment_id,
                  (*issued)[edge - 1].commitment_id);
      }
    }
  }
  EXPECT_EQ(allocator.active, 0);
  EXPECT_EQ(allocator.zeroized_deallocations, 6);
}

TEST(DeepSeekNcclBootstrapCapabilityIssuerTest,
     RollsBackAllPinnedCopiesWhenLaterEdgeFails) {
  IssuerAllocator allocator;
  IssuerSource source;
  source.fail_at = 3;
  auto issued = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 4, 100, source, allocator);
  EXPECT_FALSE(issued.ok());
  EXPECT_EQ(source.calls, 3);
  EXPECT_EQ(allocator.active, 0);
  EXPECT_EQ(allocator.zeroized_deallocations, 4);
}

TEST(DeepSeekNcclEndpointManifestPlanTest,
     ProjectsFreshEdgeCapabilitiesOntoCanonicalRanks) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 4, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 4> ranks{{
      {0, 10, 20}, {1, 11, 21}, {2, 12, 22}, {3, 13, 23}}};
  auto plan = DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks, capabilities);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->edges().size(), 3U);
  EXPECT_EQ(plan->incoming(0), nullptr);
  EXPECT_EQ(plan->outgoing(3), nullptr);
  for (std::uint32_t edge = 0; edge < 3; ++edge) {
    const auto& pair = plan->edges()[edge];
    EXPECT_EQ(pair.lower.local_global_rank, edge);
    EXPECT_EQ(pair.lower.peer_global_rank, edge + 1);
    EXPECT_EQ(pair.upper.local_global_rank, edge + 1);
    EXPECT_EQ(pair.upper.peer_global_rank, edge);
    EXPECT_EQ(pair.lower.communicator_local_rank, 0U);
    EXPECT_EQ(pair.upper.communicator_local_rank, 1U);
    EXPECT_EQ(pair.lower.bootstrap_lease_id,
              capabilities[edge].lease_id);
    EXPECT_EQ(pair.upper.bootstrap_commitment_id,
              capabilities[edge].commitment_id);
    EXPECT_EQ(pair.lower.device_identity, 10U + edge);
    EXPECT_EQ(pair.upper.context_identity, 21U + edge);
  }
  ASSERT_NE(plan->incoming(2), nullptr);
  ASSERT_NE(plan->outgoing(2), nullptr);
  EXPECT_EQ(plan->incoming(2)->edge_id, 1U);
  EXPECT_EQ(plan->outgoing(2)->edge_id, 2U);
}

TEST(DeepSeekNcclEndpointManifestPlanTest,
     RejectsNoncanonicalRankAndCapabilitySets) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 2, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 2> swapped{{
      {1, 10, 20}, {0, 11, 21}}};
  EXPECT_FALSE(DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, swapped, capabilities).ok());
  const std::array<DeepSeekNcclRankEndpointIdentity, 2> ranks{{
      {0, 10, 20}, {1, 11, 21}}};
  EXPECT_FALSE(DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks,
      std::span<const DeepSeekNcclIssuedEdgeCapability>{}).ok());
}

DeepSeekNcclEndpointBootstrapReceipt receipt(
    const DeepSeekNcclCommunicatorManifest& manifest) {
  return {manifest.engine_epoch,
          manifest.communicator_generation,
          manifest.bootstrap_lease_id,
          manifest.bootstrap_commitment_id,
          manifest.edge_id,
          manifest.local_global_rank,
          manifest.peer_global_rank,
          manifest.communicator_local_rank,
          manifest.device_identity,
          manifest.context_identity,
          manifest.config_identity,
          true,
          true};
}

Sha256Digest digest(std::byte value) {
  Sha256Digest result{};
  result.bytes.fill(value);
  return result;
}

DeepSeekNcclEndpointWarmupReceipt warmup_receipt(
    const DeepSeekNcclCommunicatorManifest& manifest,
    std::uint32_t maximum_token_count,
    Sha256Digest minimum_digest,
    Sha256Digest maximum_digest) {
  return {receipt(manifest),
          1,
          maximum_token_count,
          minimum_digest,
          maximum_digest,
          true,
          true};
}

TEST(DeepSeekNcclEndpointReconciliationGateTest,
     CompletesOnlyAfterEveryPlannedWorkerEndpointArrives) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 3, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 3> ranks{{
      {0, 10, 20}, {1, 11, 21}, {2, 12, 22}}};
  auto plan = DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks, capabilities).value();
  auto gate = DeepSeekNcclEndpointReconciliationGate::Create(plan).value();
  ASSERT_TRUE(gate.accept(receipt(plan.edges()[1].upper)).ok());
  ASSERT_TRUE(gate.accept(receipt(plan.edges()[0].lower)).ok());
  EXPECT_FALSE(gate.complete());
  EXPECT_EQ(gate.accepted_count(), 2U);
  ASSERT_TRUE(gate.accept(receipt(plan.edges()[1].lower)).ok());
  ASSERT_TRUE(gate.accept(receipt(plan.edges()[0].upper)).ok());
  EXPECT_TRUE(gate.complete());
  EXPECT_EQ(gate.accepted_count(), 4U);
}

TEST(DeepSeekNcclEndpointReconciliationGateTest,
     PoisonsGenerationOnReplayOrIdentityDrift) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 2, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 2> ranks{{
      {0, 10, 20}, {1, 11, 21}}};
  auto plan = DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks, capabilities).value();
  {
    auto gate = DeepSeekNcclEndpointReconciliationGate::Create(plan).value();
    auto lower = receipt(plan.edges()[0].lower);
    ASSERT_TRUE(gate.accept(lower).ok());
    EXPECT_FALSE(gate.accept(lower).ok());
    EXPECT_TRUE(gate.poisoned());
    EXPECT_FALSE(gate.accept(receipt(plan.edges()[0].upper)).ok());
  }
  {
    auto gate = DeepSeekNcclEndpointReconciliationGate::Create(plan).value();
    auto upper = receipt(plan.edges()[0].upper);
    ++upper.device_identity;
    EXPECT_FALSE(gate.accept(upper).ok());
    EXPECT_TRUE(gate.poisoned());
  }
  {
    auto gate = DeepSeekNcclEndpointReconciliationGate::Create(plan).value();
    auto upper = receipt(plan.edges()[0].upper);
    upper.bootstrap_zeroized = false;
    EXPECT_FALSE(gate.accept(upper).ok());
    EXPECT_TRUE(gate.poisoned());
  }
}

TEST(DeepSeekNcclBoundaryWarmupGateTest,
     FinalizesOnlyAfterMatchingMinAndMaxEvidenceFromEveryEndpoint) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 3, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 3> ranks{{
      {0, 10, 20}, {1, 11, 21}, {2, 12, 22}}};
  auto plan = DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks, capabilities).value();
  auto gate = DeepSeekNcclBoundaryWarmupGate::Create(plan, 8).value();

  ASSERT_TRUE(gate.accept(warmup_receipt(
      plan.edges()[1].upper, 8, digest(std::byte{3}),
      digest(std::byte{4}))).ok());
  ASSERT_TRUE(gate.accept(warmup_receipt(
      plan.edges()[0].lower, 8, digest(std::byte{1}),
      digest(std::byte{2}))).ok());
  ASSERT_TRUE(gate.accept(warmup_receipt(
      plan.edges()[1].lower, 8, digest(std::byte{3}),
      digest(std::byte{4}))).ok());
  ASSERT_TRUE(gate.accept(warmup_receipt(
      plan.edges()[0].upper, 8, digest(std::byte{1}),
      digest(std::byte{2}))).ok());

  auto finalized = gate.finalize();
  ASSERT_TRUE(finalized.ok()) << finalized.status().message();
  ASSERT_EQ(finalized->size(), 2U);
  EXPECT_EQ(gate.accepted_count(), 4U);
  EXPECT_TRUE((*finalized)[0].minimum_boundary_passed);
  EXPECT_TRUE((*finalized)[0].maximum_boundary_passed);
  EXPECT_EQ((*finalized)[1].edge_id, 1U);
}

TEST(DeepSeekNcclBoundaryWarmupGateTest,
     PoisonsGenerationWhenPeerPayloadDigestsDiffer) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 2, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 2> ranks{{
      {0, 10, 20}, {1, 11, 21}}};
  auto plan = DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks, capabilities).value();
  auto gate = DeepSeekNcclBoundaryWarmupGate::Create(plan, 8).value();
  ASSERT_TRUE(gate.accept(warmup_receipt(
      plan.edges()[0].lower, 8, digest(std::byte{1}),
      digest(std::byte{2}))).ok());
  ASSERT_TRUE(gate.accept(warmup_receipt(
      plan.edges()[0].upper, 8, digest(std::byte{1}),
      digest(std::byte{9}))).ok());
  EXPECT_FALSE(gate.finalize().ok());
  EXPECT_TRUE(gate.poisoned());
}

TEST(DeepSeekNcclBoundaryWarmupGateTest,
     RejectsUnverifiedOrWrongSizedEndpointEvidence) {
  IssuerAllocator allocator;
  IssuerSource source;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      7, 2, 100, source, allocator).value();
  const std::array<DeepSeekNcclRankEndpointIdentity, 2> ranks{{
      {0, 10, 20}, {1, 11, 21}}};
  auto plan = DeepSeekNcclEndpointManifestPlan::Create(
      7, 9, 30, ranks, capabilities).value();
  {
    auto gate = DeepSeekNcclBoundaryWarmupGate::Create(plan, 8).value();
    auto invalid = warmup_receipt(plan.edges()[0].lower, 7,
                                  digest(std::byte{1}),
                                  digest(std::byte{2}));
    EXPECT_FALSE(gate.accept(std::move(invalid)).ok());
    EXPECT_TRUE(gate.poisoned());
  }
  {
    auto gate = DeepSeekNcclBoundaryWarmupGate::Create(plan, 8).value();
    auto invalid = warmup_receipt(plan.edges()[0].lower, 8,
                                  digest(std::byte{1}),
                                  digest(std::byte{2}));
    invalid.maximum_complete_verified = false;
    EXPECT_FALSE(gate.accept(std::move(invalid)).ok());
    EXPECT_TRUE(gate.poisoned());
  }
}

}  // namespace
}  // namespace pih
