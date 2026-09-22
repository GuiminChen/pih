#include "pih/model/deepseek_rank_artifact_prefault.h"

#include <array>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekRankMappingPlan tail_page_mapping() {
  DeepSeekRankMappingPlan mapping;
  mapping.rank = 0;
  mapping.owned_tensor_count = 2;
  mapping.logical_tensor_bytes = 2;
  mapping.mapped_interval_bytes = 4097;
  mapping.shards.push_back({"a.safetensors", 8193});
  mapping.intervals.push_back({"a.safetensors", 0, 4096});
  mapping.intervals.push_back({"a.safetensors", 8192, 8193});
  return mapping;
}

TEST(DeepSeekRankArtifactPrefaultTest,
     RoundsPhysicalTailPageSeparatelyFromMappedBytes) {
  auto layout = compile_deepseek_rank_artifact_prefault_layout(
      tail_page_mapping(), 4096);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactPrefaultAbi,
            "pih_deepseek_rank_artifact_prefault_v1");
  EXPECT_EQ(layout->rank, 0U);
  EXPECT_EQ(layout->interval_count, 2U);
  EXPECT_EQ(layout->page_bytes, 4096U);
  EXPECT_EQ(layout->mapped_interval_bytes, 4097U);
  EXPECT_EQ(layout->selected_page_union_bytes, 8192U);
  EXPECT_NE(layout->mapping_plan_root, Sha256Digest{});
  EXPECT_NE(layout->layout_root, Sha256Digest{});

  auto repeated = compile_deepseek_rank_artifact_prefault_layout(
      tail_page_mapping(), 4096);
  ASSERT_TRUE(repeated.ok()) << repeated.status().message();
  EXPECT_EQ(repeated->layout_root, layout->layout_root);
}

TEST(DeepSeekRankArtifactPrefaultTest,
     RejectsForeignPageGeometryAndUnalignedSelectedInterval) {
  EXPECT_FALSE(compile_deepseek_rank_artifact_prefault_layout(
                   tail_page_mapping(), 65536)
                   .ok());

  auto unaligned = tail_page_mapping();
  unaligned.intervals[0].file_begin = 1;
  unaligned.intervals[0].file_end = 4096;
  --unaligned.mapped_interval_bytes;
  EXPECT_FALSE(compile_deepseek_rank_artifact_prefault_layout(
                   unaligned, 4096)
                   .ok());

  auto drifted = tail_page_mapping();
  ++drifted.mapped_interval_bytes;
  EXPECT_FALSE(compile_deepseek_rank_artifact_prefault_layout(
                   drifted, 4096)
                   .ok());
}

TEST(DeepSeekRankArtifactPrefaultTest,
     DeduplicatesPhysicalPagesSharedByTwoRanks) {
  auto rank_zero = tail_page_mapping();
  auto rank_one = tail_page_mapping();
  rank_one.rank = 1;
  rank_one.owned_tensor_count = 1;
  rank_one.logical_tensor_bytes = 1;
  rank_one.mapped_interval_bytes = 8193;
  rank_one.intervals.clear();
  rank_one.intervals.push_back({"a.safetensors", 0, 8193});
  std::array<DeepSeekRankMappingPlan, 2> plans{
      std::move(rank_zero), std::move(rank_one)};

  auto layout = compile_deepseek_node_artifact_prefault_layout(plans);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(kDeepSeekArtifactPrefaultLayoutAbi,
            "pih_deepseek_artifact_prefault_layout_v1");
  EXPECT_EQ(layout->world_size, 2U);
  EXPECT_EQ(layout->page_bytes, 4096U);
  EXPECT_EQ(layout->summed_rank_selected_page_bytes, 20'480U);
  EXPECT_EQ(layout->node_selected_page_union_bytes, 12'288U);
  EXPECT_EQ(layout->node_duplicate_selected_page_bytes, 8192U);
  EXPECT_NE(layout->rank_layout_set_root, Sha256Digest{});
  EXPECT_NE(layout->node_page_union_root, Sha256Digest{});
  EXPECT_NE(layout->layout_root, Sha256Digest{});
}

TEST(DeepSeekRankArtifactPrefaultTest,
     RejectsOutOfOrderRanksAndCrossRankShardSizeDrift) {
  std::array<DeepSeekRankMappingPlan, 2> plans{
      tail_page_mapping(), tail_page_mapping()};
  EXPECT_FALSE(compile_deepseek_node_artifact_prefault_layout(plans).ok());

  plans[1].rank = 1;
  plans[1].shards[0].file_bytes += 1;
  EXPECT_FALSE(compile_deepseek_node_artifact_prefault_layout(plans).ok());
}

}  // namespace
}  // namespace pih
