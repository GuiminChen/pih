#include "pih/model/engine_owned_resource_manifest_census_source.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest manifest_source_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

EngineOwnedResourceManifestRecord artifact_record(std::uint8_t identity) {
  return {EngineOwnedResourceKind::kArtifact, manifest_source_digest(1),
          manifest_source_digest(identity), 4096, 1,
          EngineOwnedResourceLifecycleState::kActive};
}

class Inventory final : public EngineOwnedResourceManifestInventory {
 public:
  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override {
    ++calls;
    last_kind = kind;
    return result;
  }

  int calls = 0;
  EngineOwnedResourceKind last_kind = EngineOwnedResourceKind::kShm;
  Result<std::vector<EngineOwnedResourceManifestRecord>> result =
      std::vector<EngineOwnedResourceManifestRecord>{};
};

TEST(EngineOwnedResourceManifestCensusSourceTest, EncodesInventoryRecords) {
  Inventory inventory;
  inventory.result = std::vector{artifact_record(2), artifact_record(3)};
  auto source = EngineOwnedResourceManifestCensusSource::Create(
      EngineOwnedResourceKind::kArtifact, inventory).value();

  auto census = source.capture();
  ASSERT_TRUE(census.ok());
  ASSERT_EQ(census->size(), 2U);
  EXPECT_EQ(census->at(0),
            *encode_engine_owned_resource_manifest_record(artifact_record(2)));
  EXPECT_EQ(inventory.calls, 1);
  EXPECT_EQ(inventory.last_kind, EngineOwnedResourceKind::kArtifact);
}

TEST(EngineOwnedResourceManifestCensusSourceTest, PreservesUnavailableAndRecovers) {
  Inventory inventory;
  inventory.result = Status::Unavailable("artifact fd inventory unavailable");
  auto source = EngineOwnedResourceManifestCensusSource::Create(
      EngineOwnedResourceKind::kArtifact, inventory).value();

  auto unavailable = source.capture();
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  inventory.result = std::vector{artifact_record(2)};
  EXPECT_TRUE(source.capture().ok());
}

TEST(EngineOwnedResourceManifestCensusSourceTest, RejectsKindDrift) {
  Inventory inventory;
  auto drifted = artifact_record(2);
  drifted.kind = EngineOwnedResourceKind::kShm;
  inventory.result = std::vector{drifted};
  auto source = EngineOwnedResourceManifestCensusSource::Create(
      EngineOwnedResourceKind::kArtifact, inventory).value();

  auto census = source.capture();
  ASSERT_FALSE(census.ok());
  EXPECT_EQ(census.status().code(), StatusCode::kFailedPrecondition);
}

TEST(EngineOwnedResourceManifestCensusSourceTest, RejectsInvalidBoundKind) {
  Inventory inventory;
  EXPECT_FALSE(EngineOwnedResourceManifestCensusSource::Create(
                   static_cast<EngineOwnedResourceKind>(6), inventory).ok());
  EXPECT_EQ(inventory.calls, 0);
}

}  // namespace
}  // namespace pih
