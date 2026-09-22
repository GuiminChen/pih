#include "pih/model/deepseek_learned_router_plan_input_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class PlanPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("plan staging");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

DeepSeekLearnedRouterWeightBindings weights() {
  return DeepSeekLearnedRouterWeightBindings::Resolve(
      {3, 4}, [](std::string_view name) {
        const std::array<std::int64_t, 2> shape{256, 4096};
        const auto address = static_cast<std::uintptr_t>(
            name[7] == '3' ? 0x10000 : 0x210000);
        return TensorView::Create(
            reinterpret_cast<void*>(address), DType::kBFloat16, shape, {},
            Device::Create(DeviceType::kCuda, 0).value(), 11);
      }).value();
}

TEST(DeepSeekLearnedRouterPlanInputAssemblerTest,
     BorrowsDeviceWeightsAndLeasesExactPinnedLayerSpans) {
  PlanPinnedAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate({3, 4}, 8, allocator);
  ASSERT_TRUE(pool.ok());
  auto bindings = weights();
  const std::array inputs{
      DeepSeekLearnedRouterLayerInput{3, 0x50000},
      DeepSeekLearnedRouterLayerInput{4, 0x51000}};
  auto work = DeepSeekLearnedRouterPlanInputAssembler::Assemble(
      3, inputs, bindings, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 8);
  ASSERT_TRUE(work.ok()) << work.status().message();
  ASSERT_EQ(work->size(), 2U);
  EXPECT_EQ((*work)[0].submission.layer, 3U);
  EXPECT_EQ((*work)[1].submission.layer, 4U);
  EXPECT_EQ((*work)[0].submission.weight_bf16, 0x10000U);
  EXPECT_EQ((*work)[1].submission.weight_bf16, 0x210000U);
  EXPECT_EQ((*work)[0].submission.input_bf16, 0x50000U);
  EXPECT_EQ((*work)[1].submission.input_bf16, 0x51000U);
  EXPECT_EQ((*work)[0].submission.scores_f32, 0x60000U);
  EXPECT_EQ((*work)[0].submission.host_scores.size(), 3U * 256U);
  EXPECT_EQ((*work)[0].submission.host_error_flag,
            (*work)[0].host_staging->error_flag());
  EXPECT_EQ((*work)[0].submission.stream, 0x80000U);
  EXPECT_EQ((*work)[0].submission.completion_event, 0x90000U);
  EXPECT_FALSE(pool->acquire().ok());
}

TEST(DeepSeekLearnedRouterPlanInputAssemblerTest,
     RejectsCapacityAndResourceMismatchWithoutConsumingLease) {
  PlanPinnedAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate({3, 4}, 2, allocator);
  ASSERT_TRUE(pool.ok());
  auto bindings = weights();
  const std::array inputs{
      DeepSeekLearnedRouterLayerInput{3, 0x50000},
      DeepSeekLearnedRouterLayerInput{4, 0x51000}};
  EXPECT_FALSE(DeepSeekLearnedRouterPlanInputAssembler::Assemble(
      3, inputs, bindings, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 2).ok());
  auto reordered = inputs;
  reordered[0].layer = 4;
  EXPECT_FALSE(DeepSeekLearnedRouterPlanInputAssembler::Assemble(
      2, reordered, bindings, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 2).ok());
  EXPECT_TRUE(pool->acquire().ok());
}

TEST(DeepSeekLearnedRouterPlanInputAssemblerTest,
     RejectsAliasedRouterOutputBeforeConsumingLease) {
  PlanPinnedAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate({3, 4}, 8, allocator);
  ASSERT_TRUE(pool.ok());
  auto bindings = weights();
  const std::array inputs{
      DeepSeekLearnedRouterLayerInput{3, 0x50000},
      DeepSeekLearnedRouterLayerInput{4, 0x51000}};
  EXPECT_FALSE(DeepSeekLearnedRouterPlanInputAssembler::Assemble(
      3, inputs, bindings, {0x50000, 0x70000}, *pool,
      0x80000, 0x90000, 8).ok());
  EXPECT_TRUE(pool->acquire().ok());
}

TEST(DeepSeekLearnedRouterPlanInputAssemblerTest,
     HashOnlyRankProducesCanonicalEmptyLearnedWork) {
  PlanPinnedAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate({0, 2}, 2, allocator);
  auto bindings = DeepSeekLearnedRouterWeightBindings::Resolve(
      {0, 2}, [](std::string_view) -> Result<TensorView> {
        return Status::Internal("resolver must not be called");
      });
  ASSERT_TRUE(pool.ok()); ASSERT_TRUE(bindings.ok());
  auto work = DeepSeekLearnedRouterPlanInputAssembler::Assemble(
      1, {}, *bindings, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 2);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_TRUE(work->empty());
}

}  // namespace
}  // namespace pih
