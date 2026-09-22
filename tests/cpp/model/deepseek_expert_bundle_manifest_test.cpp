#include "pih/model/deepseek_expert_bundle_manifest.h"

#include <gtest/gtest.h>

namespace pih { namespace {

std::vector<DeepSeekRankTensorRecord> expert_records(std::uint32_t layer) {
  std::vector<DeepSeekRankTensorRecord> records;
  std::uint64_t offset = 4096;
  for (std::uint32_t expert = 0; expert < 256; ++expert) {
    for (const auto* matrix : {"w1", "w2", "w3"}) {
      const bool w2 = std::string_view(matrix) == "w2";
      const std::string prefix = "layers." + std::to_string(layer) +
          ".ffn.experts." + std::to_string(expert) + "." + matrix;
      const std::uint64_t packed_bytes = 4194304;
      records.push_back({prefix + ".weight", "expert.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kInt8,
                         w2 ? std::vector<std::uint64_t>{4096, 1024}
                            : std::vector<std::uint64_t>{2048, 2048},
                         offset, offset + packed_bytes});
      offset += packed_bytes;
      const std::uint64_t scale_bytes = 262144;
      records.push_back({prefix + ".scale", "expert.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kFloat8E8M0,
                         w2 ? std::vector<std::uint64_t>{4096, 64}
                            : std::vector<std::uint64_t>{2048, 128},
                         offset, offset + scale_bytes});
      offset += scale_bytes;
    }
  }
  return records;
}

std::vector<DeepSeekRankTensorRecord> dspark_expert_records() {
  auto records = expert_records(42);
  for (auto& record : records) {
    record.tensor_name.replace(0, std::string("layers.42").size(), "mtp.0");
    record.shard_name = "dspark.safetensors";
    record.role = DeepSeekTensorRole::kDspark;
  }
  return records;
}

TEST(DeepSeekExpertBundleManifestTest, CompilesCanonicalSixSegmentBundles) {
  auto records = expert_records(4);
  auto manifest = DeepSeekExpertBundleManifest::Create({4, 4}, records);
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  ASSERT_EQ(manifest->bundles().size(), 256U);
  const auto& bundle = manifest->bundles()[7];
  EXPECT_EQ(bundle.identity, (DeepSeekExpertIdentity{4, 7}));
  EXPECT_TRUE(bundle.segments[0].shard_name.ends_with(".safetensors"));
  EXPECT_EQ(bundle.segments[0].bytes, 4194304U);
  EXPECT_EQ(bundle.segments[1].bytes, 262144U);
  EXPECT_LT(bundle.segments[0].file_offset, bundle.segments[1].file_offset);
  EXPECT_EQ(manifest->payload_bytes(),
            256U * DeepSeekExpertBundleLayout::kBundleBytes);
}

TEST(DeepSeekExpertBundleManifestTest, RejectsMissingWrongDtypeAndOverlap) {
  auto missing = expert_records(4);
  missing.pop_back();
  EXPECT_FALSE(DeepSeekExpertBundleManifest::Create({4, 4}, missing).ok());

  auto wrong_dtype = expert_records(4);
  wrong_dtype[1].dtype = DType::kUInt8;
  EXPECT_FALSE(DeepSeekExpertBundleManifest::Create({4, 4}, wrong_dtype).ok());

  auto overlap = expert_records(4);
  overlap[1].file_begin = overlap[0].file_begin;
  overlap[1].file_end = overlap[1].file_begin + 262144;
  EXPECT_FALSE(DeepSeekExpertBundleManifest::Create({4, 4}, overlap).ok());
}

TEST(DeepSeekExpertBundleManifestTest, ExcludesFixedResidentDsparkExperts) {
  auto records = expert_records(42);
  auto dspark = dspark_expert_records();
  records.insert(records.end(), dspark.begin(), dspark.end());
  auto manifest = DeepSeekExpertBundleManifest::Create({42,42}, records);
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  ASSERT_EQ(manifest->bundles().size(),256U);
  EXPECT_EQ(manifest->bundles().back().identity,
            (DeepSeekExpertIdentity{42,255}));
}

} }  // namespace pih
