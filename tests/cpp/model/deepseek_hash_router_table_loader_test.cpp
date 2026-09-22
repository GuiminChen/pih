#include "pih/model/deepseek_hash_router_table_loader.h"

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <string>

namespace pih {
namespace {

class TableSource final : public DeepSeekWeightByteSource {
 public:
  void add(std::string name, std::vector<std::int64_t> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(std::int64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    values_.emplace(std::move(name), std::move(bytes));
  }
  Result<std::span<const std::byte>> resolve_weight_bytes(
      std::string_view name) const override {
    const auto found = values_.find(std::string(name));
    if (found == values_.end()) return Status::InvalidArgument("missing table");
    return std::span<const std::byte>(found->second);
  }
 private:
  std::map<std::string, std::vector<std::byte>> values_;
};

std::vector<std::int64_t> table(std::uint32_t vocabulary_size,
                                std::int64_t offset = 0) {
  std::vector<std::int64_t> result;
  for (std::uint32_t token = 0; token < vocabulary_size; ++token) {
    for (std::int64_t expert = 0; expert < 6; ++expert) {
      result.push_back(offset + expert);
    }
  }
  return result;
}

TEST(DeepSeekHashRouterTableLoaderTest,
     LoadsOwnedHashLayersAndNarrowsValidatedIds) {
  TableSource source;
  source.add("layers.1.ffn.gate.tid2eid", table(2, 10));
  source.add("layers.2.ffn.gate.tid2eid", table(2, 20));
  auto loaded = DeepSeekHashRouterTableLoader::Load({1, 4}, 2, source);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  ASSERT_EQ(loaded->size(), 2U);
  EXPECT_EQ((*loaded)[0].layer, 1U);
  EXPECT_EQ((*loaded)[0].token_to_experts[0], 10U);
  EXPECT_EQ((*loaded)[1].token_to_experts[11], 25U);
}

TEST(DeepSeekHashRouterTableLoaderTest,
     LearnedOnlyStageNeedsNoHashArtifact) {
  TableSource source;
  auto loaded = DeepSeekHashRouterTableLoader::Load({3, 10}, 2, source);
  ASSERT_TRUE(loaded.ok());
  EXPECT_TRUE(loaded->empty());
}

TEST(DeepSeekHashRouterTableLoaderTest,
     RejectsExtentRangeAndDuplicateExpertRows) {
  TableSource source;
  source.add("layers.0.ffn.gate.tid2eid", {0});
  EXPECT_FALSE(DeepSeekHashRouterTableLoader::Load({0, 0}, 1, source).ok());
  source = TableSource{};
  source.add("layers.0.ffn.gate.tid2eid", {0, 1, 2, 3, 4, 256});
  EXPECT_FALSE(DeepSeekHashRouterTableLoader::Load({0, 0}, 1, source).ok());
  source = TableSource{};
  source.add("layers.0.ffn.gate.tid2eid", {0, 1, 2, 3, 4, 4});
  EXPECT_FALSE(DeepSeekHashRouterTableLoader::Load({0, 0}, 1, source).ok());
}

}  // namespace
}  // namespace pih
