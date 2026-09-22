#include "pih/model/deepseek_learned_router_work_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Operations final : public DeepSeekLearnedRouterOperations {
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
  Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(
        static_cast<std::size_t>(bytes),
        static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

DeepSeekLearnedRouterSubmission submission(std::uint32_t layer,
                                           std::uint32_t tokens) {
  DeepSeekLearnedRouterSubmission value;
  value.layer = layer;
  value.token_count = tokens;
  value.input_bf16 = 1;
  value.weight_bf16 = 2;
  value.scores_f32 = 6;
  value.error_flag_u32 = 7;
  value.stream = 8;
  value.completion_event = 9;
  return value;
}

TEST(DeepSeekLearnedRouterWorkFactoryTest,
     BuildsExactOwnedLayersWithPinnedPlanStaging) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({3, 4}, 2).value();
  Operations operations;
  std::array<float, 256> bias{};
  auto factory = DeepSeekLearnedRouterWorkFactory::Create(
      {3, 4}, 2, {{3, bias}, {4, bias}}, store, operations).value();
  PinnedAllocator allocator;
  std::vector<DeepSeekLearnedRouterLayerPlanWork> layers;
  for (std::uint32_t id = 3; id <= 4; ++id) {
    auto staging = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator);
    ASSERT_TRUE(staging.ok());
    layers.push_back({
        submission(id, 2),
        std::make_shared<DeepSeekLearnedRouterHostStaging>(
            std::move(*staging))});
  }
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(factory.append_plan_work(2, std::move(layers), builder).ok());
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.learned_router.size(), 2U);
  EXPECT_EQ(work.learned_router[0].coordinator->plan_provider(), &store);
  EXPECT_EQ(work.learned_router[1].submission.host_scores.size(), 512U);
}

TEST(DeepSeekLearnedRouterWorkFactoryTest,
     RejectsBiasGapsSubmissionGapsAndIncompleteCudaResources) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({3, 4}, 2).value();
  Operations operations;
  std::array<float, 256> bias{};
  EXPECT_FALSE(DeepSeekLearnedRouterWorkFactory::Create(
      {3, 4}, 2, {{3, bias}}, store, operations).ok());
  auto factory = DeepSeekLearnedRouterWorkFactory::Create(
      {3, 4}, 2, {{3, bias}, {4, bias}}, store, operations).value();
  PinnedAllocator allocator;
  auto staging = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator);
  DeepSeekRankComputeWorkBuilder builder;
  EXPECT_FALSE(factory.append_plan_work(
      2, {{submission(3, 2),
           std::make_shared<DeepSeekLearnedRouterHostStaging>(
               std::move(*staging))}}, builder).ok());
  auto first = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator);
  auto second = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator);
  auto incomplete = submission(3, 2);
  incomplete.stream = 0;
  EXPECT_FALSE(factory.append_plan_work(
      2, {{incomplete,
           std::make_shared<DeepSeekLearnedRouterHostStaging>(
               std::move(*first))},
          {submission(4, 2),
           std::make_shared<DeepSeekLearnedRouterHostStaging>(
               std::move(*second))}}, builder).ok());
  first = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator);
  second = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator);
  auto aliased = submission(3, 2);
  aliased.scores_f32 = aliased.input_bf16;
  EXPECT_FALSE(factory.append_plan_work(
      2, {{aliased,
           std::make_shared<DeepSeekLearnedRouterHostStaging>(
               std::move(*first))},
          {submission(4, 2),
           std::make_shared<DeepSeekLearnedRouterHostStaging>(
               std::move(*second))}}, builder).ok());
}

TEST(DeepSeekLearnedRouterWorkFactoryTest,
     RejectsSyntheticDsparkLayerIdentity) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({42, 42}, 2).value();
  Operations operations;
  std::array<float, 256> bias{};
  EXPECT_FALSE(DeepSeekLearnedRouterWorkFactory::Create(
      {43, 43}, 2, {{43, bias}}, store, operations).ok());
}

}  // namespace
}  // namespace pih
