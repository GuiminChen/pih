#include "pih/model/engine_pinned_memory_inventory.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest pinned_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

std::array<EnginePinnedMemoryBinding, 2> pinned_bindings() {
  return {EnginePinnedMemoryBinding{
              60, pinned_digest(1), pinned_digest(2), pinned_digest(9),
              0, 4096, EngineOwnedResourceLifecycleState::kActive},
          EnginePinnedMemoryBinding{
              61, pinned_digest(1), pinned_digest(3), pinned_digest(9),
              0, 8192, EngineOwnedResourceLifecycleState::kPassive}};
}

class PinnedOperations final : public EnginePinnedMemoryOperations {
 public:
  Result<std::vector<EnginePinnedMemoryObservation>> capture_pinned(
      std::span<const std::uint64_t> registration_identities) override {
    ++calls;
    if (!status.ok()) return status;
    identities.assign(registration_identities.begin(), registration_identities.end());
    std::vector<EnginePinnedMemoryObservation> result(observations.begin(), observations.end());
    if (reverse) std::reverse(result.begin(), result.end());
    if (truncate) result.pop_back();
    return result;
  }
  int calls = 0;
  Status status = Status::Ok();
  std::vector<std::uint64_t> identities;
  bool reverse = false;
  bool truncate = false;
  std::array<EnginePinnedMemoryObservation, 2> observations{
      EnginePinnedMemoryObservation{60, true, true, pinned_digest(9), 0, 4096},
      EnginePinnedMemoryObservation{61, true, true, pinned_digest(9), 0, 8192}};
};

TEST(EnginePinnedMemoryInventoryTest, CapturesExactRegisteredOwners) {
  PinnedOperations operations;
  auto inventory = EnginePinnedMemoryInventory::Create(
      pinned_bindings(), operations).value();
  auto records = inventory.capture(EngineOwnedResourceKind::kPinnedMemory);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 2U);
  EXPECT_EQ(records->at(0).backing_bytes, 4096U);
  EXPECT_EQ(records->at(0).resource_identity, pinned_digest(2));
  EXPECT_EQ(records->at(1).state,
            EngineOwnedResourceLifecycleState::kPassive);
}

TEST(EnginePinnedMemoryInventoryTest, OmitsUnregisteredExtent) {
  PinnedOperations operations;
  operations.observations[1].registered = false;
  operations.observations[1].registered_bytes = 0;
  auto inventory = EnginePinnedMemoryInventory::Create(
      pinned_bindings(), operations).value();
  EXPECT_EQ(inventory.capture(EngineOwnedResourceKind::kPinnedMemory)->size(),
            1U);
}

TEST(EnginePinnedMemoryInventoryTest, RejectsVisibilityAndTopologyDrift) {
  for (int mutation = 0; mutation < 6; ++mutation) {
    PinnedOperations operations;
    if (mutation == 0) ++operations.observations[0].registration_identity;
    if (mutation == 1) operations.observations[0].owner_counter_visible = false;
    if (mutation == 2) operations.observations[0].physical_gpu_identity =
                           pinned_digest(8);
    if (mutation == 3) ++operations.observations[0].numa_node;
    if (mutation == 4) ++operations.observations[0].registered_bytes;
    if (mutation == 5) {
      operations.observations[0].registered = false;
      operations.observations[0].registered_bytes = 1;
    }
    auto inventory = EnginePinnedMemoryInventory::Create(
        pinned_bindings(), operations).value();
    auto records = inventory.capture(EngineOwnedResourceKind::kPinnedMemory);
    ASSERT_FALSE(records.ok()) << mutation;
    EXPECT_EQ(records.status().code(),
              mutation == 1 ? StatusCode::kUnavailable
                            : StatusCode::kFailedPrecondition)
        << mutation;
  }
}

TEST(EnginePinnedMemoryInventoryTest, PreservesUnavailableAndKindBoundary) {
  PinnedOperations operations;
  auto inventory = EnginePinnedMemoryInventory::Create(
      pinned_bindings(), operations).value();
  operations.status = Status::Unavailable("driver owner counter unavailable");
  EXPECT_EQ(inventory.capture(EngineOwnedResourceKind::kPinnedMemory)
                .status().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kShm).ok());
}

TEST(EnginePinnedMemoryInventoryTest, RejectsInvalidManifest) {
  PinnedOperations operations;
  auto bindings = pinned_bindings();
  bindings[1].registration_identity = bindings[0].registration_identity;
  EXPECT_FALSE(EnginePinnedMemoryInventory::Create(bindings, operations).ok());
  bindings = pinned_bindings();
  bindings[0].registered_bytes = 0;
  EXPECT_FALSE(EnginePinnedMemoryInventory::Create(bindings, operations).ok());
  bindings = pinned_bindings();
  bindings[0].physical_gpu_identity = {};
  EXPECT_FALSE(EnginePinnedMemoryInventory::Create(bindings, operations).ok());
}

TEST(EnginePinnedMemoryInventoryTest, RejectsPartialOrReorderedSnapshot) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    PinnedOperations operations;
    operations.truncate = mutation == 0;
    operations.reverse = mutation == 1;
    auto inventory = EnginePinnedMemoryInventory::Create(
        pinned_bindings(), operations).value();
    EXPECT_FALSE(inventory.capture(
        EngineOwnedResourceKind::kPinnedMemory).ok()) << mutation;
  }
}

}  // namespace
}  // namespace pih
