#include "pih/model/deepseek_stage_mapping_plan.h"
#include "pih/model/deepseek_rank_artifact_handoff_plan.h"

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace pih { namespace {

SafetensorsHeader header(std::string json, std::uint64_t payload_bytes) {
  std::vector<std::byte> prefix(8 + json.size());
  const auto size = static_cast<std::uint64_t>(json.size());
  for (std::size_t index = 0; index < 8; ++index) {
    prefix[index] = static_cast<std::byte>((size >> (index * 8U)) & 0xffU);
  }
  std::memcpy(prefix.data() + 8, json.data(), json.size());
  return SafetensorsHeader::ParsePrefix(prefix, 8 + json.size() + payload_bytes)
      .value();
}

std::uint64_t file_bytes(const SafetensorsHeader& value) {
  return 8 + value.header_bytes() + value.data_bytes();
}

TEST(DeepSeekStageMappingPlanTest, BuildsPageUnionsForOwnedTensorExtents) {
  auto first = header(
      R"({"embed.weight":{"dtype":"U8","shape":[3000],"data_offsets":[0,3000]},"layers.0.a":{"dtype":"U8","shape":[2000],"data_offsets":[3000,5000]},"layers.42.a":{"dtype":"U8","shape":[100],"data_offsets":[5000,5100]}})",
      5100);
  auto second = header(
      R"({"head.weight":{"dtype":"U8","shape":[100],"data_offsets":[0,100]},"layers.0.b":{"dtype":"U8","shape":[8900],"data_offsets":[100,9000]},"mtp.0.weight":{"dtype":"U8","shape":[100],"data_offsets":[9000,9100]}})",
      9100);
  std::vector<SafetensorsShardBinding> bindings{
      {"embed.weight", "a.safetensors"},
      {"layers.0.a", "a.safetensors"},
      {"layers.0.b", "b.safetensors"},
      {"layers.42.a", "a.safetensors"},
      {"head.weight", "b.safetensors"},
      {"mtp.0.weight", "b.safetensors"}};
  std::vector<DeepSeekShardHeaderView> headers{
      {"a.safetensors", file_bytes(first), &first},
      {"b.safetensors", file_bytes(second), &second}};
  auto pipeline = DeepSeekPipelinePlan::Create(2, true).value();
  auto plan = DeepSeekStageMappingPlan::Create(pipeline, bindings, headers,
                                                4096);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->rank(0).intervals.size(), 2U);
  EXPECT_EQ(plan->rank(0).intervals[0].shard_name, "a.safetensors");
  EXPECT_EQ(plan->rank(0).intervals[0].file_begin, 0U);
  EXPECT_EQ(plan->rank(0).intervals[0].file_end, file_bytes(first));
  ASSERT_EQ(plan->rank(0).shards.size(), 2U);
  EXPECT_EQ(plan->rank(0).shards[0].file_bytes, file_bytes(first));
  ASSERT_EQ(plan->rank(1).intervals.size(), 3U);
  EXPECT_EQ(plan->rank(1).owned_tensor_count, 3U);
}

TEST(DeepSeekStageMappingPlanTest, ExcludesDsparkAndRejectsHeaderIndexDrift) {
  auto shard = header(
      R"({"head.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"mtp.0.weight":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
      2);
  std::vector<SafetensorsShardBinding> bindings{
      {"head.weight", "a.safetensors"},
      {"mtp.0.weight", "a.safetensors"}};
  std::vector<DeepSeekShardHeaderView> headers{{"a.safetensors", file_bytes(shard),
                                                &shard}};
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto plan = DeepSeekStageMappingPlan::Create(pipeline, bindings, headers,
                                                4096);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->rank(0).owned_tensor_count, 1U);
  EXPECT_EQ(plan->excluded_tensor_count(), 1U);

  bindings.pop_back();
  EXPECT_FALSE(DeepSeekStageMappingPlan::Create(pipeline, bindings, headers,
                                                 4096).ok());
  headers[0].file_bytes += 1;
  EXPECT_FALSE(DeepSeekStageMappingPlan::Create(
                   pipeline,
                   std::vector<SafetensorsShardBinding>{
                       {"head.weight", "a.safetensors"},
                       {"mtp.0.weight", "a.safetensors"}},
                   headers, 4096)
                   .ok());
}

TEST(DeepSeekStageMappingPlanTest, RejectsNonProductionPageGeometry) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, true).value();
  std::vector<SafetensorsShardBinding> bindings{{"head.weight", "a"}};
  EXPECT_FALSE(DeepSeekStageMappingPlan::Create(pipeline, bindings, {}, 0).ok());
  EXPECT_FALSE(
      DeepSeekStageMappingPlan::Create(pipeline, bindings, {}, 8192).ok());
}

TEST(DeepSeekStageMappingPlanTest,
     CompilesDistinctCanonicalRootsForEverySupportedWorldSize) {
  std::string json = "{";
  std::vector<SafetensorsShardBinding> bindings;
  auto append = [&](std::string name, std::uint64_t offset) {
    if (json.size() != 1U) json += ',';
    json += "\"" + name +
            "\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[" +
            std::to_string(offset) + "," + std::to_string(offset + 1U) +
            "]}";
    bindings.push_back({std::move(name), "model.safetensors"});
  };
  append("embed.weight", 0);
  for (std::uint32_t layer = 0; layer < 43; ++layer) {
    append("layers." + std::to_string(layer) + ".weight", layer + 1U);
  }
  append("head.weight", 44);
  json += '}';
  auto shard = header(std::move(json), 45);
  std::vector<DeepSeekShardHeaderView> headers{
      {"model.safetensors", file_bytes(shard), &shard}};

  std::set<std::string> roots;
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto pipeline = DeepSeekPipelinePlan::Create(world_size, false).value();
    auto mapping = DeepSeekStageMappingPlan::Create(
        pipeline, bindings, headers,
        DeepSeekStageMappingPlan::kProductionPageBytes);
    ASSERT_TRUE(mapping.ok()) << mapping.status().message();
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      EXPECT_GT(mapping->rank(rank).owned_tensor_count, 0U);
    }
    auto first = compile_deepseek_stage_mapping_plan_root(*mapping);
    auto second = compile_deepseek_stage_mapping_plan_root(*mapping);
    ASSERT_TRUE(first.ok()) << first.status().message();
    ASSERT_TRUE(second.ok()) << second.status().message();
    EXPECT_EQ(*first, *second);
    EXPECT_NE(*first, Sha256Digest{});
    roots.insert(first->hex());
  }
  EXPECT_EQ(roots.size(), 4U);
}

} }  // namespace pih
