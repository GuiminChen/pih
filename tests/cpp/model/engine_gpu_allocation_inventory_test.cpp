#include "pih/model/engine_gpu_allocation_inventory.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest allocation_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

std::array<EngineGpuAllocationBinding, 2> allocation_bindings() {
  return {EngineGpuAllocationBinding{
              70, allocation_digest(1), allocation_digest(2),
              allocation_digest(9), 4096,
              EngineOwnedResourceLifecycleState::kActive},
          EngineGpuAllocationBinding{
              71, allocation_digest(1), allocation_digest(3),
              allocation_digest(9), 8192,
              EngineOwnedResourceLifecycleState::kPassive}};
}

class AllocationOperations final : public EngineGpuAllocationOperations {
 public:
  Result<std::vector<EngineGpuAllocationObservation>> capture_allocations(
      std::span<const std::uint64_t> allocation_identities) override {
    ++calls;
    if (!status.ok()) return status;
    identities.assign(allocation_identities.begin(), allocation_identities.end());
    std::vector<EngineGpuAllocationObservation> result(observations.begin(), observations.end());
    if (reverse) std::reverse(result.begin(), result.end());
    if (truncate) result.pop_back();
    return result;
  }
  int calls = 0;
  Status status = Status::Ok();
  std::vector<std::uint64_t> identities;
  bool reverse = false;
  bool truncate = false;
  std::array<EngineGpuAllocationObservation, 2> observations{
      EngineGpuAllocationObservation{70, true, true, allocation_digest(9), 4096},
      EngineGpuAllocationObservation{71, true, true, allocation_digest(9), 8192}};
};

TEST(EngineGpuAllocationInventoryTest, CapturesExactAllocationOwners) {
  AllocationOperations operations;
  auto inventory = EngineGpuAllocationInventory::Create(
      allocation_bindings(), operations).value();
  auto records = inventory.capture(EngineOwnedResourceKind::kGpuAllocation);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 2U);
  EXPECT_EQ(records->at(0).backing_bytes, 4096U);
  EXPECT_EQ(records->at(1).state,
            EngineOwnedResourceLifecycleState::kPassive);
}

TEST(EngineGpuAllocationInventoryTest, OmitsReleasedAllocation) {
  AllocationOperations operations;
  operations.observations[1].allocated = false;
  operations.observations[1].allocated_bytes = 0;
  auto inventory = EngineGpuAllocationInventory::Create(
      allocation_bindings(), operations).value();
  EXPECT_EQ(inventory.capture(EngineOwnedResourceKind::kGpuAllocation)->size(),
            1U);
}

TEST(EngineGpuAllocationInventoryTest, RejectsVisibilityAndIdentityDrift) {
  for (int mutation = 0; mutation < 5; ++mutation) {
    AllocationOperations operations;
    if (mutation == 0) ++operations.observations[0].allocation_identity;
    if (mutation == 1) operations.observations[0].owner_counter_visible = false;
    if (mutation == 2) operations.observations[0].physical_gpu_identity =
                           allocation_digest(8);
    if (mutation == 3) ++operations.observations[0].allocated_bytes;
    if (mutation == 4) {
      operations.observations[0].allocated = false;
      operations.observations[0].allocated_bytes = 1;
    }
    auto inventory = EngineGpuAllocationInventory::Create(
        allocation_bindings(), operations).value();
    auto records = inventory.capture(EngineOwnedResourceKind::kGpuAllocation);
    ASSERT_FALSE(records.ok()) << mutation;
    EXPECT_EQ(records.status().code(),
              mutation == 1 ? StatusCode::kUnavailable
                            : StatusCode::kFailedPrecondition) << mutation;
  }
}

TEST(EngineGpuAllocationInventoryTest, PreservesUnavailableAndKindBoundary) {
  AllocationOperations operations;
  auto inventory = EngineGpuAllocationInventory::Create(
      allocation_bindings(), operations).value();
  operations.status = Status::Unavailable("GPU owner ledger unavailable");
  EXPECT_EQ(inventory.capture(EngineOwnedResourceKind::kGpuAllocation)
                .status().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kPinnedMemory).ok());
}

TEST(EngineGpuAllocationInventoryTest, RejectsInvalidManifest) {
  AllocationOperations operations;
  auto bindings = allocation_bindings();
  bindings[1].allocation_identity = bindings[0].allocation_identity;
  EXPECT_FALSE(EngineGpuAllocationInventory::Create(bindings, operations).ok());
  bindings = allocation_bindings();
  bindings[0].allocated_bytes = 0;
  EXPECT_FALSE(EngineGpuAllocationInventory::Create(bindings, operations).ok());
  bindings = allocation_bindings();
  bindings[0].physical_gpu_identity = {};
  EXPECT_FALSE(EngineGpuAllocationInventory::Create(bindings, operations).ok());
}

TEST(EngineGpuAllocationInventoryTest, RejectsPartialOrReorderedSnapshot) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    AllocationOperations operations;
    operations.truncate = mutation == 0;
    operations.reverse = mutation == 1;
    auto inventory = EngineGpuAllocationInventory::Create(
        allocation_bindings(), operations).value();
    EXPECT_FALSE(inventory.capture(
        EngineOwnedResourceKind::kGpuAllocation).ok()) << mutation;
  }
}

}  // namespace
}  // namespace pih
