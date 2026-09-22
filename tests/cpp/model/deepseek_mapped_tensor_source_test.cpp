#include "pih/model/deepseek_mapped_tensor_source.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekMappedTensorSourceTest, ResolvesValidatedManifestFromSharedMapping) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-mapped-tensor-source-test";
  std::filesystem::create_directory(root);
  std::ofstream(root / "a.safetensors", std::ios::binary | std::ios::trunc)
      << "abcdefgh";
  auto lease = ControllerFileLease::OpenBeneath(
      root, "a.safetensors", 8,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("a.safetensors",
                           lease.duplicate_for_worker().value());
  DeepSeekRankMappingPlan plan;
  plan.shards.push_back({"a.safetensors", 8});
  plan.intervals.push_back({"a.safetensors", 0, 8});
  plan.mapped_interval_bytes = 8;
  auto inventory = DeepSeekStageMappedInventory::Create(plan, std::move(descriptors));
  ASSERT_TRUE(inventory.ok());
  std::vector<DeepSeekRankTensorRecord> records = {
      {"embed.weight", "a.safetensors", DeepSeekTensorRole::kEmbedding,
       DType::kUInt8, {4}, 0, 4},
      {"layers.0.norm.weight", "a.safetensors", DeepSeekTensorRole::kMainLayer,
       DType::kBFloat16, {2}, 4, 8}};
  auto source = DeepSeekMappedTensorSource::Create(*inventory, records);
  ASSERT_TRUE(source.ok()) << source.status().message();
  auto tensor = source->resolve("layers.0.norm.weight");
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->record->dtype, DType::kBFloat16);
  EXPECT_EQ(tensor->bytes.size(), 4U);
  EXPECT_EQ(std::to_integer<char>(tensor->bytes[0]), 'e');
  EXPECT_FALSE(source->resolve("head.weight").ok());
  std::filesystem::remove_all(root);
}

TEST(DeepSeekMappedTensorSourceTest, RejectsByteGeometryAndOverlapDrift) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-mapped-tensor-source-invalid-test";
  std::filesystem::create_directory(root);
  std::ofstream(root / "a.safetensors", std::ios::binary | std::ios::trunc)
      << "abcdefgh";
  auto lease = ControllerFileLease::OpenBeneath(
      root, "a.safetensors", 8,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("a.safetensors", lease.duplicate_for_worker().value());
  DeepSeekRankMappingPlan plan;
  plan.shards.push_back({"a.safetensors", 8});
  plan.intervals.push_back({"a.safetensors", 0, 8});
  plan.mapped_interval_bytes = 8;
  auto inventory = DeepSeekStageMappedInventory::Create(plan, std::move(descriptors)).value();
  std::vector<DeepSeekRankTensorRecord> bad = {
      {"a", "a.safetensors", DeepSeekTensorRole::kEmbedding,
       DType::kUInt8, {5}, 0, 4}};
  EXPECT_FALSE(DeepSeekMappedTensorSource::Create(inventory, bad).ok());
  bad = {{"a", "a.safetensors", DeepSeekTensorRole::kEmbedding,
          DType::kUInt8, {4}, 0, 4},
         {"b", "a.safetensors", DeepSeekTensorRole::kMainLayer,
          DType::kUInt8, {4}, 2, 6}};
  EXPECT_FALSE(DeepSeekMappedTensorSource::Create(inventory, bad).ok());
  std::filesystem::remove_all(root);
}

} }  // namespace pih
