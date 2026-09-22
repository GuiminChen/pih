#include "pih/model/deepseek_rank_router_factory_set.h"

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <string>

namespace pih {
namespace {

class RouterSource final : public DeepSeekWeightByteSource {
 public:
  template <typename T>
  void add(std::string name, const std::vector<T>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    values_.emplace(std::move(name), std::move(bytes));
  }
  Result<std::span<const std::byte>> resolve_weight_bytes(
      std::string_view name) const override {
    const auto found = values_.find(std::string(name));
    if (found == values_.end()) return Status::InvalidArgument("missing router");
    return std::span<const std::byte>(found->second);
  }
 private:
  std::map<std::string, std::vector<std::byte>> values_;
};

class RouterOperations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status gemm(DeepSeekRouterBf16GemmLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

TEST(DeepSeekRankRouterFactorySetTest,
     BuildsHashAndLearnedFactoriesFromOneArtifactSource) {
  RouterSource source;
  const std::vector<std::int64_t> table{0, 1, 2, 3, 4, 5};
  for (std::uint32_t layer = 0; layer < 3; ++layer) {
    source.add("layers." + std::to_string(layer) +
                   ".ffn.gate.tid2eid",
               table);
  }
  source.add("layers.3.ffn.gate.bias", std::vector<float>(256, 0.25F));
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 3}, 1).value();
  RouterOperations operations;
  auto factories = DeepSeekRankRouterFactorySet::Create(
      {0, 3}, 1, 1, source, store, operations);
  ASSERT_TRUE(factories.ok()) << factories.status().message();
  EXPECT_EQ(factories->hash_router().owned_hash_layer_count(), 3U);
  EXPECT_EQ(factories->learned_router().owned_learned_layer_count(), 1U);
  EXPECT_EQ(factories->hash_router().owned_layers(),
            factories->learned_router().owned_layers());
}

TEST(DeepSeekRankRouterFactorySetTest,
     MissingEitherRouterArtifactPreventsPartialFactoryPublication) {
  RouterSource source;
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 3}, 1).value();
  RouterOperations operations;
  EXPECT_FALSE(DeepSeekRankRouterFactorySet::Create(
      {0, 3}, 1, 1, source, store, operations).ok());
}

}  // namespace
}  // namespace pih
