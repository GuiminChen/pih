#include "pih/model/deepseek_learned_router_bias_loader.h"

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <string>

namespace pih {
namespace {

class BiasSource final : public DeepSeekWeightByteSource {
 public:
  void add(std::string name, float base) {
    std::array<float, 256> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = base + static_cast<float>(i) / 1024.0F;
    }
    std::vector<std::byte> bytes(sizeof(values));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    values_.emplace(std::move(name), std::move(bytes));
  }
  void truncate(std::string name) {
    values_[std::move(name)] = std::vector<std::byte>(4);
  }
  Result<std::span<const std::byte>> resolve_weight_bytes(
      std::string_view name) const override {
    const auto found = values_.find(std::string(name));
    if (found == values_.end()) {
      return Status::InvalidArgument("missing bias");
    }
    return std::span<const std::byte>(found->second);
  }
 private:
  std::map<std::string, std::vector<std::byte>> values_;
};

TEST(DeepSeekLearnedRouterBiasLoaderTest,
     LoadsEveryOwnedNonHashLayerInCanonicalOrder) {
  BiasSource source;
  source.add("layers.3.ffn.gate.bias", 3.0F);
  source.add("layers.4.ffn.gate.bias", 4.0F);
  auto loaded = DeepSeekLearnedRouterBiasLoader::Load({2, 4}, source);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  ASSERT_EQ(loaded->size(), 2U);
  EXPECT_EQ((*loaded)[0].layer, 3U);
  EXPECT_FLOAT_EQ((*loaded)[0].bias[0], 3.0F);
  EXPECT_EQ((*loaded)[1].layer, 4U);
  EXPECT_FLOAT_EQ((*loaded)[1].bias[255],
                  4.0F + 255.0F / 1024.0F);
}

TEST(DeepSeekLearnedRouterBiasLoaderTest,
     HashOnlyStageNeedsNoBiasArtifact) {
  BiasSource source;
  auto loaded = DeepSeekLearnedRouterBiasLoader::Load({0, 2}, source);
  ASSERT_TRUE(loaded.ok());
  EXPECT_TRUE(loaded->empty());
}

TEST(DeepSeekLearnedRouterBiasLoaderTest,
     RejectsMissingOrWrongExtentBeforeFactoryCreation) {
  BiasSource source;
  EXPECT_FALSE(DeepSeekLearnedRouterBiasLoader::Load({3, 3}, source).ok());
  source.truncate("layers.3.ffn.gate.bias");
  EXPECT_FALSE(DeepSeekLearnedRouterBiasLoader::Load({3, 3}, source).ok());
}

}  // namespace
}  // namespace pih
