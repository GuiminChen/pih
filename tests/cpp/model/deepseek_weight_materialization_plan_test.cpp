#include "pih/model/deepseek_weight_materialization_plan.h"

#include <gtest/gtest.h>

namespace pih { namespace {

std::vector<DeepSeekRankTensorRecord> materialization_records() {
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

Sha256Digest test_digest(std::uint8_t tag, std::uint64_t ordinal = 0) {
  Sha256Digest result;
  result.bytes[0] = static_cast<std::byte>(tag);
  for (std::size_t index = 0; index < 8; ++index) {
    result.bytes[index + 1] =
        static_cast<std::byte>(ordinal >> (index * 8U));
  }
  return result;
}

void bind_target_authority(std::vector<DeepSeekRankTensorRecord>& records) {
  for (std::size_t ordinal = 0; ordinal < records.size(); ++ordinal) {
    auto& record = records[ordinal];
    record.logical_layer = 4;
    record.tensor_bytes = record.file_end - record.file_begin;
    if (record.dtype == DType::kInt8) {
      record.storage_semantics =
          DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits;
    } else if (record.dtype == DType::kFloat8E8M0) {
      record.storage_semantics =
          DeepSeekStorageSemantics::kDirectUe8m0ScaleBits;
    } else {
      record.storage_semantics =
          DeepSeekStorageSemantics::kDirectBf16LittleEndianBits;
    }
    record.artifact_root = test_digest(1);
    record.layout_root = test_digest(2);
    record.disposition_root = test_digest(3);
    record.target_logical_root = test_digest(4, ordinal);
    record.disposition_record_root = test_digest(5, ordinal);
    record.layout_record_root = test_digest(6, ordinal);
    record.runtime_record_root = test_digest(7, ordinal);
  }
}

TEST(DeepSeekWeightMaterializationPlanTest,
     HostSpillAllocatesOnlyFixedTensorBacking) {
  auto tensors = materialization_records();
  auto disposition = DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, tensors).value();
  auto experts = DeepSeekExpertBundleManifest::Create({4, 4}, tensors).value();
  auto plan = DeepSeekWeightMaterializationPlan::Create(
      disposition, experts, tensors);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->copies().size(), 1U);
  EXPECT_EQ(plan->owner_rank(), 0U);
  EXPECT_EQ(plan->copies()[0].tensor_name, "layers.4.attn_norm.weight");
  EXPECT_EQ(plan->copies()[0].destination_offset, 0U);
  EXPECT_EQ(plan->payload_bytes(), 8192U);
  EXPECT_EQ(plan->backing_bytes(), 8192U);
  EXPECT_EQ(plan->paged_source_bytes(),
            256U * DeepSeekExpertBundleLayout::kBundleBytes);
}

TEST(DeepSeekWeightMaterializationPlanTest,
     FullResidentPlacesEveryExpertInCanonicalContiguousBundle) {
  auto tensors = materialization_records();
  auto disposition = DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kFullResident, tensors).value();
  auto experts = DeepSeekExpertBundleManifest::Create({4, 4}, tensors).value();
  auto plan = DeepSeekWeightMaterializationPlan::Create(
      disposition, experts, tensors);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->copies().size(), tensors.size());
  const auto* scale = plan->find("layers.4.ffn.experts.7.w2.scale");
  ASSERT_NE(scale, nullptr);
  const auto bundle_base = 8192U +
      7U * DeepSeekExpertBundleLayout::kBundleBytes;
  EXPECT_EQ(scale->destination_offset, bundle_base + 8650752U);
  EXPECT_EQ(plan->payload_bytes(),
            8192U + 256U * DeepSeekExpertBundleLayout::kBundleBytes);
  EXPECT_EQ(plan->paged_source_bytes(), 0U);
  EXPECT_EQ(plan->backing_bytes(), plan->payload_bytes());
}

TEST(DeepSeekWeightMaterializationPlanTest,
     RejectsTensorSetThatDiffersFromDisposition) {
  auto tensors = materialization_records();
  auto disposition = DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, tensors).value();
  auto experts = DeepSeekExpertBundleManifest::Create({4, 4}, tensors).value();
  tensors.pop_back();
  EXPECT_FALSE(DeepSeekWeightMaterializationPlan::Create(
      disposition, experts, tensors).ok());
}

TEST(DeepSeekWeightMaterializationPlanTest,
     PreservesCompleteTargetAuthorityIntoEveryFixedCopy) {
  auto tensors = materialization_records();
  bind_target_authority(tensors);
  auto disposition = DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, tensors).value();
  auto experts = DeepSeekExpertBundleManifest::Create({4, 4}, tensors).value();
  auto plan = DeepSeekWeightMaterializationPlan::Create(
      disposition, experts, tensors);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_TRUE(plan->target_authority_bound());
  ASSERT_EQ(plan->copies().size(), 1U);
  EXPECT_EQ(plan->artifact_root(), test_digest(1));
  EXPECT_EQ(plan->layout_root(), test_digest(2));
  EXPECT_EQ(plan->disposition_root(), test_digest(3));
  EXPECT_EQ(plan->copies().front().storage_semantics,
            DeepSeekStorageSemantics::kDirectBf16LittleEndianBits);
  EXPECT_EQ(plan->copies().front().runtime_record_root,
            tensors.back().runtime_record_root);

  tensors.back().runtime_record_root = {};
  EXPECT_FALSE(DeepSeekWeightMaterializationPlan::Create(
                   disposition, experts, tensors)
                   .ok());
}

} }  // namespace pih
