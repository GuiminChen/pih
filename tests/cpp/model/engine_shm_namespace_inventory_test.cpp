#include "pih/model/engine_shm_namespace_inventory.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest shm_inventory_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

std::array<EngineShmObjectBinding, 2> shm_bindings() {
  return {EngineShmObjectBinding{
              20, shm_inventory_digest(1), shm_inventory_digest(2),
              7, 70, 4096, EngineOwnedResourceLifecycleState::kActive},
          EngineShmObjectBinding{
              21, shm_inventory_digest(1), shm_inventory_digest(3),
              7, 71, 8192, EngineOwnedResourceLifecycleState::kRetired}};
}

class ShmOperations final : public EngineShmNamespaceOperations {
 public:
  Result<EngineShmObjectObservation> observe(
      std::uint64_t object_identity) override {
    ++calls;
    if (!status.ok()) return status;
    return observations[object_identity == 20 ? 0U : 1U];
  }

  int calls = 0;
  Status status = Status::Ok();
  std::array<EngineShmObjectObservation, 2> observations{
      EngineShmObjectObservation{20, true, true, 7, 70, 4096},
      EngineShmObjectObservation{21, true, true, 7, 71, 8192}};
};

TEST(EngineShmNamespaceInventoryTest, CapturesPresentManifestObjects) {
  ShmOperations operations;
  auto inventory = EngineShmNamespaceInventory::Create(
      shm_bindings(), operations).value();

  auto records = inventory.capture(EngineOwnedResourceKind::kShm);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 2U);
  EXPECT_EQ(records->at(0).owner_identity, shm_inventory_digest(1));
  EXPECT_EQ(records->at(0).resource_identity, shm_inventory_digest(2));
  EXPECT_EQ(records->at(0).backing_bytes, 4096U);
  EXPECT_EQ(records->at(1).state,
            EngineOwnedResourceLifecycleState::kRetired);
}

TEST(EngineShmNamespaceInventoryTest, OmitsAbsentObjectAndRecovers) {
  ShmOperations operations;
  operations.observations[1].present = false;
  auto inventory = EngineShmNamespaceInventory::Create(
      shm_bindings(), operations).value();
  auto records = inventory.capture(EngineOwnedResourceKind::kShm);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 1U);
  operations.observations[1] = {21, true, true, 7, 71, 8192};
  EXPECT_EQ(inventory.capture(EngineOwnedResourceKind::kShm)->size(), 2U);
}

TEST(EngineShmNamespaceInventoryTest, RejectsReplacementAndMetadataDrift) {
  for (int mutation = 0; mutation < 5; ++mutation) {
    ShmOperations operations;
    if (mutation == 0) ++operations.observations[0].object_identity;
    if (mutation == 1) operations.observations[0].regular_file = false;
    if (mutation == 2) ++operations.observations[0].device_identity;
    if (mutation == 3) ++operations.observations[0].inode_identity;
    if (mutation == 4) ++operations.observations[0].size_bytes;
    auto inventory = EngineShmNamespaceInventory::Create(
        shm_bindings(), operations).value();
    auto records = inventory.capture(EngineOwnedResourceKind::kShm);
    ASSERT_FALSE(records.ok()) << mutation;
    EXPECT_EQ(records.status().code(), StatusCode::kFailedPrecondition)
        << mutation;
  }
}

TEST(EngineShmNamespaceInventoryTest, PreservesUnavailableAndKindBoundary) {
  ShmOperations operations;
  auto inventory = EngineShmNamespaceInventory::Create(
      shm_bindings(), operations).value();
  operations.status = Status::Unavailable("SHM namespace unavailable");
  auto unavailable = inventory.capture(EngineOwnedResourceKind::kShm);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  operations.status = Status::Ok();
  EXPECT_TRUE(inventory.capture(EngineOwnedResourceKind::kShm).ok());
  EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kArtifact).ok());
}

TEST(EngineShmNamespaceInventoryTest, RejectsInvalidManifest) {
  ShmOperations operations;
  auto bindings = shm_bindings();
  bindings[1].object_identity = bindings[0].object_identity;
  EXPECT_FALSE(EngineShmNamespaceInventory::Create(bindings, operations).ok());
  bindings = shm_bindings();
  bindings[1].resource_identity = bindings[0].resource_identity;
  EXPECT_FALSE(EngineShmNamespaceInventory::Create(bindings, operations).ok());
  bindings = shm_bindings();
  bindings[0].expected_inode_identity = 0;
  EXPECT_FALSE(EngineShmNamespaceInventory::Create(bindings, operations).ok());
}

}  // namespace
}  // namespace pih
