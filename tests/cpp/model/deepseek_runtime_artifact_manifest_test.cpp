#include "pih/model/deepseek_runtime_artifact_manifest.h"

#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

constexpr std::string_view kFixture =
    R"({"artifact_abi":"deepseek_v4_runtime_artifact_v1","artifact_root":"e8d6f70d267f7010f9bd012e302e7caaab6683987fbdff79644b654cee5ff8d2","conversion_root":"a3464593685ff271e00a53446f4f3cc287438c590ce24f3ab19434803d58fd8e","converter_abi":"deepseek_v4_streaming_identity_converter_v1","converter_identity_root":"4444444444444444444444444444444444444444444444444444444444444444","disposition_root":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","dspark_enabled":false,"index":{"file_bytes":100,"index_object_root":"02496787f3dd3999c6ed41dfa8373d36048e8fc95c39059e63b6f14ff8875fbf","index_root":"2222222222222222222222222222222222222222222222222222222222222222","name":"model.safetensors.index.json","object_sha256":"3333333333333333333333333333333333333333333333333333333333333333"},"layout_root":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","logical_layout_root":"8888888888888888888888888888888888888888888888888888888888888888","manifest_body_sha256":"538abfb631cd23987bb960aed3302bd432ab180ca5437388f3c9cd41acb161ee","model_family":"deepseek_v4_flash_0731","resources":{"maximum_copy_chunk_bytes":1048576,"maximum_output_descriptors":1,"source_descriptor_count":50},"runtime_records":{"abi":"deepseek_v4_runtime_records_v1","body_sha256":"d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5","file_bytes":1335,"name":"pih.runtime-records.json","object_sha256":"7497c375ee3a6935185374e392a18e49abdab761ffcf325d79191057cea5cd6c","record_bytes":4,"record_count":1,"record_set_root":"fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f","runtime_records_object_root":"19da446e5c407d889c0e307c8b668db6a37efa68d3790984f42024006800ede2","runtime_records_root":"abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7"},"schema":"pih.deepseek_v4_flash_0731_runtime_artifact.v1","shard_count":1,"shards":[{"converted_shard_root":"13cbed0f507486033a8a4ce6d9c904038415111ff750394b87ba1b9c3148d429","file_bytes":104,"name":"model-endpoint.safetensors","namespace":"endpoint","object_sha256":"1111111111111111111111111111111111111111111111111111111111111111","payload_bytes":4,"shard_layout_root":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","tensor_count":1}],"source_inventory_root":"5555555555555555555555555555555555555555555555555555555555555555","source_payload_closure_root":"6666666666666666666666666666666666666666666666666666666666666666","support_state":"hardware_evidence_open","tensor_bytes":4,"tensor_count":1,"world_size":1})";

Sha256Digest expected_root() {
  return Sha256Digest::ParseHex(
             "e8d6f70d267f7010f9bd012e302e7caaab6683987fbdff79644b654cee5ff8d2")
      .value();
}

TEST(DeepSeekRuntimeArtifactManifestTest,
     ReplaysPythonProducedRootChainWithoutTrustingNestedDigests) {
  auto parsed =
      DeepSeekRuntimeArtifactManifest::Parse(kFixture, expected_root());
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->world_size(), 1U);
  EXPECT_FALSE(parsed->dspark_enabled());
  EXPECT_EQ(parsed->tensor_count(), 1U);
  EXPECT_EQ(parsed->tensor_bytes(), 4U);
  ASSERT_EQ(parsed->shards().size(), 1U);
  EXPECT_EQ(parsed->shards().front().name_space, "endpoint");
  EXPECT_EQ(parsed->index().name, "model.safetensors.index.json");
  EXPECT_EQ(parsed->runtime_records().record_count, 1U);
  EXPECT_EQ(parsed->runtime_records().object_bytes, 1335U);
  EXPECT_FALSE(parsed->validate_flash_0731_geometry().ok());
}

TEST(DeepSeekRuntimeArtifactManifestTest,
     RejectsTrustedRootCanonicalAndNestedObjectDrift) {
  auto wrong_root = Sha256Digest::ParseHex(
                        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
                        .value();
  EXPECT_FALSE(
      DeepSeekRuntimeArtifactManifest::Parse(kFixture, wrong_root).ok());

  std::string noncanonical(kFixture);
  noncanonical.insert(1, " ");
  EXPECT_FALSE(DeepSeekRuntimeArtifactManifest::Parse(
                   noncanonical, expected_root())
                   .ok());

  std::string changed(kFixture);
  const auto position = changed.find(
      "19da446e5c407d889c0e307c8b668db6a37efa68d3790984f42024006800ede2");
  ASSERT_NE(position, std::string::npos);
  changed[position] = '2';
  EXPECT_FALSE(
      DeepSeekRuntimeArtifactManifest::Parse(changed, expected_root()).ok());
}

TEST(DeepSeekRuntimeArtifactManifestTest, EncodesSyntheticCompleteLedgerAndRejectsDrift) {
  // Metadata-only synthetic ledger, not checkpoint provenance or payload evidence.
  const auto hash = expected_root();
  DeepSeekRuntimeArtifactManifest::EncodeInput input;
  input.logical_layout_root = hash;
  input.source_inventory_root = hash;
  input.source_payload_closure_root = hash;
  input.converter_identity_root = hash;
  input.maximum_copy_chunk_bytes = 1U << 20;
  input.source_descriptor_count = 50;
  input.maximum_output_descriptors = 1;
  input.records = {hash, hash, hash, hash, hash, hash, 100, 156'015'698'140ULL,
                   67'612, 1, false};
  input.index = {"model.safetensors.index.json", 100, hash, hash, {}};
  for (unsigned i = 0; i < 44; ++i) {
    const auto space = i == 0 ? std::string("endpoint") : "layers." + std::to_string(i - 1);
    const auto member = i == 0 ? std::string("model-endpoint.safetensors") :
        "model-layers-" + std::to_string(i - 1) + ".safetensors";
    const auto bytes = i == 0 ? 156'015'698'140ULL - 43 : 1ULL;
    input.shards.push_back({space, member, bytes + 64, i == 0 ? 67'612U - 43 : 1U,
                            bytes, hash, hash, {}});
  }
  auto encoded = DeepSeekRuntimeArtifactManifest::Encode(input);
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  auto parsed = DeepSeekRuntimeArtifactManifest::Parse(encoded->json, encoded->artifact_root);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_TRUE(parsed->validate_flash_0731_geometry().ok());
  EXPECT_EQ(parsed->runtime_records().object_sha256, hash);
  auto changed = input;
  changed.shards[0].converted_shard_root = hash;
  changed.index.index_object_root = hash;
  auto recomputed = DeepSeekRuntimeArtifactManifest::Encode(changed);
  ASSERT_TRUE(recomputed.ok());
  EXPECT_EQ(recomputed->json, encoded->json);
  changed = input;
  changed.source_descriptor_count = 49;
  EXPECT_FALSE(DeepSeekRuntimeArtifactManifest::Encode(changed).ok());
  changed = input;
  changed.shards[1].tensor_count++;
  EXPECT_FALSE(DeepSeekRuntimeArtifactManifest::Encode(changed).ok());
  changed = input;
  changed.source_inventory_root = {};
  EXPECT_FALSE(DeepSeekRuntimeArtifactManifest::Encode(changed).ok());
}

}  // namespace
}  // namespace pih
