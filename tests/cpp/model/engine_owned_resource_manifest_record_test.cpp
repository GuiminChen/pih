#include "pih/model/engine_owned_resource_manifest_record.h"

#include <algorithm>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest manifest_record_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

TEST(EngineOwnedResourceManifestRecordTest, ProducesStableCanonicalIdentity) {
  const EngineOwnedResourceManifestRecord record{
      EngineOwnedResourceKind::kArtifact, manifest_record_digest(1),
      manifest_record_digest(2), 4096, 1,
      EngineOwnedResourceLifecycleState::kActive};
  auto encoded = encode_engine_owned_resource_manifest_record(record);
  ASSERT_TRUE(encoded.ok());
  ASSERT_EQ(encoded->size(), 32U);
  Sha256Digest digest{};
  std::copy(encoded->begin(), encoded->end(), digest.bytes.begin());
  EXPECT_EQ(digest.hex(),
            "99e95316b4067f68965f0ee45dc90229a77ead6bfd932cd591591272a4f90d10");
}

TEST(EngineOwnedResourceManifestRecordTest, SeparatesEveryField) {
  const EngineOwnedResourceManifestRecord original{
      EngineOwnedResourceKind::kNetwork, manifest_record_digest(1),
      manifest_record_digest(2), 4096, 1,
      EngineOwnedResourceLifecycleState::kActive};
  auto baseline = encode_engine_owned_resource_manifest_record(original).value();
  for (int mutation = 0; mutation < 6; ++mutation) {
    auto changed = original;
    if (mutation == 0) changed.kind = EngineOwnedResourceKind::kListener;
    if (mutation == 1) changed.owner_identity = manifest_record_digest(3);
    if (mutation == 2) changed.resource_identity = manifest_record_digest(3);
    if (mutation == 3) ++changed.backing_bytes;
    if (mutation == 4) ++changed.object_count;
    if (mutation == 5)
      changed.state = EngineOwnedResourceLifecycleState::kRetired;
    EXPECT_NE(encode_engine_owned_resource_manifest_record(changed).value(),
              baseline) << mutation;
  }
}

TEST(EngineOwnedResourceManifestRecordTest, RejectsInvalidAuthorityFields) {
  EngineOwnedResourceManifestRecord record{
      EngineOwnedResourceKind::kArtifact, manifest_record_digest(1),
      manifest_record_digest(2), 0, 1,
      EngineOwnedResourceLifecycleState::kActive};
  record.owner_identity = {};
  EXPECT_FALSE(encode_engine_owned_resource_manifest_record(record).ok());
  record.owner_identity = manifest_record_digest(1);
  record.resource_identity = {};
  EXPECT_FALSE(encode_engine_owned_resource_manifest_record(record).ok());
  record.resource_identity = manifest_record_digest(2);
  record.object_count = 0;
  EXPECT_FALSE(encode_engine_owned_resource_manifest_record(record).ok());
  record.object_count = 1;
  record.kind = static_cast<EngineOwnedResourceKind>(6);
  EXPECT_FALSE(encode_engine_owned_resource_manifest_record(record).ok());
  record.kind = EngineOwnedResourceKind::kArtifact;
  record.state = static_cast<EngineOwnedResourceLifecycleState>(3);
  EXPECT_FALSE(encode_engine_owned_resource_manifest_record(record).ok());
}

}  // namespace
}  // namespace pih
