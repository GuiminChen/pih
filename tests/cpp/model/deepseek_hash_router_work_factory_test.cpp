#include "pih/model/deepseek_hash_router_work_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class ProjectedOperations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status gemm(DeepSeekRouterBf16GemmLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class ProjectedPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("projected staging");
    return Allocation{data, bytes, alignment, 1, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override { _aligned_free(value.data); }
};

std::vector<std::uint16_t> table(std::uint32_t vocabulary_size) {
  std::vector<std::uint16_t> result(
      static_cast<std::size_t>(vocabulary_size) *
      DeepSeekExpertSubwavePlan::kRoutesPerToken);
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::uint16_t>(
        index % DeepSeekExpertSubwavePlan::kExpertCount);
  }
  return result;
}

TEST(DeepSeekHashRouterWorkFactoryTest,
     ReusesFrozenTablesAcrossPlanOwners) {
  auto factory = DeepSeekHashRouterWorkFactory::Create(
      {0, 10}, 4, 8,
      {{0, table(8)}, {1, table(8)}, {2, table(8)}}).value();
  const std::uint16_t* first_table = nullptr;
  const std::array<std::uint32_t, 2> tokens{4, 7};
  for (std::uint32_t plan = 0; plan < 2; ++plan) {
    DeepSeekRankComputeWorkBuilder builder;
    std::vector<DeepSeekHashRouterLayerScores> scores;
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
      scores.push_back({layer, std::vector<float>(2 * 256, 0.25F)});
    }
    ASSERT_TRUE(factory.append_plan_work(tokens, std::move(scores),
                                         builder).ok());
    auto work = std::move(builder).finish().value();
    ASSERT_EQ(work.hash_router.size(), 3U);
    if (plan == 0) first_table = work.hash_router[0].token_to_experts.data();
    EXPECT_EQ(work.hash_router[0].token_to_experts.data(), first_table);
    EXPECT_EQ(work.hash_router[2].token_ids[1], 7U);
  }
}

TEST(DeepSeekHashRouterWorkFactoryTest,
     RejectsMissingTablesInvalidTokensAndIncompleteScores) {
  EXPECT_FALSE(DeepSeekHashRouterWorkFactory::Create(
      {0, 10}, 4, 8, {{0, table(8)}}).ok());
  auto factory = DeepSeekHashRouterWorkFactory::Create(
      {0, 2}, 1, 8,
      {{0, table(8)}, {1, table(8)}, {2, table(8)}}).value();
  DeepSeekRankComputeWorkBuilder builder;
  const std::array<std::uint32_t, 1> invalid_token{8};
  const std::array<std::uint32_t, 1> valid_token{1};
  EXPECT_FALSE(factory.append_plan_work(
      invalid_token, {{0, std::vector<float>(256)},
            {1, std::vector<float>(256)},
            {2, std::vector<float>(256)}}, builder).ok());
  EXPECT_FALSE(factory.append_plan_work(
      valid_token, {{0, std::vector<float>(256)}}, builder).ok());
}

TEST(DeepSeekHashRouterWorkFactoryTest,
     RankWithoutHashLayersRequiresNoTablesOrScores) {
  auto factory = DeepSeekHashRouterWorkFactory::Create(
      {3, 14}, 4, 8, {}).value();
  EXPECT_EQ(factory.owned_hash_layer_count(), 0U);
  DeepSeekRankComputeWorkBuilder builder;
  const std::array<std::uint32_t, 1> tokens{1};
  EXPECT_TRUE(factory.append_plan_work(tokens, {}, builder).ok());
}

TEST(DeepSeekHashRouterWorkFactoryTest,
     PublishesProjectedLayersWithCanonicalCoordinatorAndSharedTable) {
  const DeepSeekPipelinePlanDescriptor descriptor{
      1, 2, DeepSeekPlanPhase::kDecode, 1, 1};
  auto store = DeepSeekExpertPlanCatalog::Create(descriptor, {0, 0}).value();
  ProjectedOperations operations;
  auto factory = DeepSeekHashRouterWorkFactory::CreateProjected(
      {0, 0}, 1, 1, {{0, table(1)}}, store, operations).value();
  ProjectedPinnedAllocator allocator;
  auto allocated = DeepSeekLearnedRouterHostStaging::Allocate(1, allocator);
  ASSERT_TRUE(allocated.ok());
  auto staging = std::make_shared<DeepSeekLearnedRouterHostStaging>(
      std::move(*allocated));
  DeepSeekRankComputeWorkBuilder builder;
  const std::array<std::uint32_t, 1> tokens{0};
  ASSERT_TRUE(factory.append_projected_plan_work(
      tokens,
      {{{0, 1, 1, 2, 3, 4, 5, 6, {}, nullptr}, staging}},
      builder).ok());
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.hash_router.size(), 1U);
  EXPECT_NE(work.hash_router[0].coordinator, nullptr);
  EXPECT_EQ(work.hash_router[0].coordinator->plan_provider(), &store);
  EXPECT_EQ(work.hash_router[0].submission.projection.weight_bf16, 2U);
  EXPECT_EQ(work.hash_router[0].submission.token_to_experts.size(), 6U);

  allocated = DeepSeekLearnedRouterHostStaging::Allocate(1, allocator);
  ASSERT_TRUE(allocated.ok());
  staging = std::make_shared<DeepSeekLearnedRouterHostStaging>(
      std::move(*allocated));
  DeepSeekRankComputeWorkBuilder invalid_builder;
  EXPECT_FALSE(factory.append_projected_plan_work(
      tokens,
      {{{0, 1, 1, 2, 1, 4, 5, 6, {}, nullptr}, staging}},
      invalid_builder).ok());
}

}  // namespace
}  // namespace pih
