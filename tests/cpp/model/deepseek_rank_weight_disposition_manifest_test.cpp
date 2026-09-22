#include "pih/model/deepseek_rank_weight_disposition_manifest.h"

#include <gtest/gtest.h>

namespace pih { namespace {

std::vector<DeepSeekRankTensorRecord> disposition_records() {
  std::vector<DeepSeekRankTensorRecord> records;
  std::uint64_t offset = 0;
  for (std::uint32_t expert = 0; expert < 256; ++expert) {
    for (const auto* matrix : {"w1", "w2", "w3"}) {
      const bool w2 = std::string_view(matrix) == "w2";
      const auto prefix = "layers.4.ffn.experts." + std::to_string(expert) +
                          "." + matrix;
      records.push_back({prefix + ".weight", "a.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kInt8,
                         w2 ? std::vector<std::uint64_t>{4096, 1024}
                            : std::vector<std::uint64_t>{2048, 2048},
                         offset, offset + 4194304});
      offset += 4194304;
      records.push_back({prefix + ".scale", "a.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kFloat8E8M0,
                         w2 ? std::vector<std::uint64_t>{4096, 64}
                            : std::vector<std::uint64_t>{2048, 128},
                         offset, offset + 262144});
      offset += 262144;
    }
  }
  records.push_back({"layers.4.attn_norm.weight", "b.safetensors",
                     DeepSeekTensorRole::kMainLayer, DType::kBFloat16,
                     {4096}, 4096, 12288});
  return records;
}

std::vector<DeepSeekRankTensorRecord> disposition_records_with_dspark() {
  auto records = disposition_records();
  const auto main_count = 256U * 6U;
  for (auto& record : records) {
    record.tensor_name.replace(0, std::string("layers.4").size(), "layers.42");
  }
  records.reserve(records.size() + 3U * main_count);
  for (std::uint32_t stage = 0; stage < 3; ++stage) {
    for (std::size_t index = 0; index < main_count; ++index) {
      auto record = records[index];
      record.tensor_name.replace(
          0, std::string("layers.42").size(),
          "mtp." + std::to_string(stage));
      record.shard_name = "dspark-" + std::to_string(stage) +
                          ".safetensors";
      record.role = DeepSeekTensorRole::kDspark;
      records.push_back(std::move(record));
    }
  }
  return records;
}

TEST(DeepSeekRankWeightDispositionManifestTest,
     HostSpillAssignsEveryExpertSegmentToPagedSource) {
  auto records = disposition_records();
  auto manifest = DeepSeekRankWeightDispositionManifest::Create(
      2, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, records);
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  EXPECT_EQ(manifest->owner_rank(), 2U);
  EXPECT_EQ(manifest->records().size(), records.size());
  EXPECT_EQ(manifest->paged_source_record_count(), 256U * 6U);
  EXPECT_EQ(manifest->materialize_fixed_record_count(), 1U);
  EXPECT_EQ(manifest->paged_source_bytes(),
            256U * DeepSeekExpertBundleLayout::kBundleBytes);
  EXPECT_EQ(manifest->materialize_fixed_bytes(), 8192U);
  const auto* expert = manifest->find("layers.4.ffn.experts.7.w2.scale");
  ASSERT_NE(expert, nullptr);
  EXPECT_EQ(expert->kind, DeepSeekWeightDispositionKind::kPagedSource);
  ASSERT_TRUE(expert->expert_identity.has_value());
  EXPECT_EQ(*expert->expert_identity, (DeepSeekExpertIdentity{4, 7}));
}

TEST(DeepSeekRankWeightDispositionManifestTest,
     FullResidentMaterializesAllSegmentsWithoutPagedOwner) {
  auto records = disposition_records();
  auto manifest = DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kFullResident, records);
  ASSERT_TRUE(manifest.ok());
  EXPECT_EQ(manifest->paged_source_record_count(), 0U);
  EXPECT_EQ(manifest->materialize_fixed_record_count(), records.size());
  EXPECT_EQ(manifest->paged_source_bytes(), 0U);
  EXPECT_EQ(manifest->materialize_fixed_bytes(),
            256U * DeepSeekExpertBundleLayout::kBundleBytes + 8192U);
}

TEST(DeepSeekRankWeightDispositionManifestTest,
     HostSpillKeepsDsparkExpertsFixedResident) {
  auto records = disposition_records_with_dspark();
  auto manifest = DeepSeekRankWeightDispositionManifest::Create(
      3, {42, 42}, DeepSeekRoutedExpertResidency::kHostSpill, records);
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  EXPECT_EQ(manifest->paged_source_record_count(), 256U * 6U);
  EXPECT_EQ(manifest->materialize_fixed_record_count(),
            3U * 256U * 6U + 1U);
  for (std::uint32_t stage = 0; stage < 3; ++stage) {
    const auto name = "mtp." + std::to_string(stage) +
                      ".ffn.experts.7.w2.scale";
    const auto* expert = manifest->find(name);
    ASSERT_NE(expert, nullptr);
    EXPECT_EQ(expert->kind, DeepSeekWeightDispositionKind::kMaterializeFixed);
    EXPECT_FALSE(expert->expert_identity.has_value());
  }
}

TEST(DeepSeekRankWeightDispositionManifestTest,
     RejectsIncompleteOrWrongRankManifest) {
  auto records = disposition_records();
  records.pop_back();
  records.pop_back();
  EXPECT_FALSE(DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, records).ok());
  records = disposition_records();
  EXPECT_FALSE(DeepSeekRankWeightDispositionManifest::Create(
      4, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, records).ok());
}

} }  // namespace pih
