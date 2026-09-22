#include "pih/model/engine_artifact_descriptor_inventory.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest artifact_inventory_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

std::array<EngineArtifactDescriptorBinding, 2> artifact_bindings() {
  return {EngineArtifactDescriptorBinding{
              10, artifact_inventory_digest(1), artifact_inventory_digest(2),
              4, 40, 4096, EngineOwnedResourceLifecycleState::kPassive},
          EngineArtifactDescriptorBinding{
              11, artifact_inventory_digest(1), artifact_inventory_digest(3),
              4, 41, 8192, EngineOwnedResourceLifecycleState::kActive}};
}

class DescriptorOperations final : public EngineArtifactDescriptorOperations {
 public:
  Result<EngineArtifactDescriptorObservation> observe(
      std::uint64_t descriptor_identity) override {
    ++calls;
    if (!status.ok()) return status;
    const auto index = descriptor_identity == 10 ? 0U : 1U;
    return observations[index];
  }

  int calls = 0;
  Status status = Status::Ok();
  std::array<EngineArtifactDescriptorObservation, 2> observations{
      EngineArtifactDescriptorObservation{10, true, true, 4, 40, 4096},
      EngineArtifactDescriptorObservation{11, true, true, 4, 41, 8192}};
};

TEST(EngineArtifactDescriptorInventoryTest, CapturesBoundDescriptorOwners) {
  DescriptorOperations operations;
  auto inventory = EngineArtifactDescriptorInventory::Create(
      artifact_bindings(), operations).value();

  auto records = inventory.capture(EngineOwnedResourceKind::kArtifact);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 2U);
  EXPECT_EQ(records->at(0).owner_identity, artifact_inventory_digest(1));
  EXPECT_EQ(records->at(0).resource_identity, artifact_inventory_digest(2));
  EXPECT_EQ(records->at(0).backing_bytes, 4096U);
  EXPECT_EQ(records->at(0).state,
            EngineOwnedResourceLifecycleState::kPassive);
}

TEST(EngineArtifactDescriptorInventoryTest, OmitsClosedDescriptor) {
  DescriptorOperations operations;
  operations.observations[1].open = false;
  auto inventory = EngineArtifactDescriptorInventory::Create(
      artifact_bindings(), operations).value();

  auto records = inventory.capture(EngineOwnedResourceKind::kArtifact);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 1U);
  EXPECT_EQ(records->at(0).resource_identity, artifact_inventory_digest(2));
}

TEST(EngineArtifactDescriptorInventoryTest, RejectsDescriptorReuseAndDrift) {
  for (int mutation = 0; mutation < 5; ++mutation) {
    DescriptorOperations operations;
    if (mutation == 0) operations.observations[0].regular_file = false;
    if (mutation == 1) ++operations.observations[0].descriptor_identity;
    if (mutation == 2) ++operations.observations[0].device_identity;
    if (mutation == 3) ++operations.observations[0].inode_identity;
    if (mutation == 4) ++operations.observations[0].size_bytes;
    auto inventory = EngineArtifactDescriptorInventory::Create(
        artifact_bindings(), operations).value();
    auto records = inventory.capture(EngineOwnedResourceKind::kArtifact);
    ASSERT_FALSE(records.ok()) << mutation;
    EXPECT_EQ(records.status().code(), StatusCode::kFailedPrecondition)
        << mutation;
  }
}

TEST(EngineArtifactDescriptorInventoryTest, PreservesUnavailableAndKindBoundary) {
  DescriptorOperations operations;
  auto inventory = EngineArtifactDescriptorInventory::Create(
      artifact_bindings(), operations).value();
  operations.status = Status::Unavailable("fstat visibility unavailable");
  auto unavailable = inventory.capture(EngineOwnedResourceKind::kArtifact);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  operations.status = Status::Ok();
  EXPECT_TRUE(inventory.capture(EngineOwnedResourceKind::kArtifact).ok());
  EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kShm).ok());
}

TEST(EngineArtifactDescriptorInventoryTest, RejectsInvalidManifest) {
  DescriptorOperations operations;
  auto bindings = artifact_bindings();
  bindings[1].descriptor_identity = bindings[0].descriptor_identity;
  EXPECT_FALSE(EngineArtifactDescriptorInventory::Create(
                   bindings, operations).ok());
  bindings = artifact_bindings();
  bindings[1].resource_identity = bindings[0].resource_identity;
  EXPECT_FALSE(EngineArtifactDescriptorInventory::Create(
                   bindings, operations).ok());
  bindings = artifact_bindings();
  bindings[0].expected_inode_identity = 0;
  EXPECT_FALSE(EngineArtifactDescriptorInventory::Create(
                   bindings, operations).ok());
}

TEST(EngineArtifactDescriptorInventoryTest, AllowsSharedArtifactAcrossOwners) {
  DescriptorOperations operations;
  auto bindings = artifact_bindings();
  bindings[1].resource_identity = bindings[0].resource_identity;
  bindings[1].owner_identity = artifact_inventory_digest(9);

  auto inventory = EngineArtifactDescriptorInventory::Create(
      bindings, operations);
  ASSERT_TRUE(inventory.ok());
  EXPECT_TRUE(inventory->capture(EngineOwnedResourceKind::kArtifact).ok());
}

}  // namespace
}  // namespace pih
