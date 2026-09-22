#include "pih/model/deepseek_hash_router_plan_input_assembler.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class HashPlanPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("hash plan staging");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override { _aligned_free(value.data); }
 private:
  std::uint64_t generation_ = 0;
};

DeepSeekHashRouterWeightBindings hash_weights() {
  return DeepSeekHashRouterWeightBindings::Resolve(
      {0, 1}, [](std::string_view name) {
        const std::array<std::int64_t, 2> shape{256, 4096};
        const auto address = static_cast<std::uintptr_t>(
            name[7] == '0' ? 0x10000 : 0x210000);
        return TensorView::Create(
            reinterpret_cast<void*>(address), DType::kBFloat16, shape, {},
            Device::Create(DeviceType::kCuda, 0).value(), 11);
      }).value();
}

TEST(DeepSeekHashRouterPlanInputAssemblerTest,
     ResolvesWeightsAndLeasesExactPinnedHashLayers) {
  HashPlanPinnedAllocator allocator;
  auto pool = DeepSeekHashRouterStagingPool::Allocate({0, 1}, 8, allocator);
  ASSERT_TRUE(pool.ok());
  auto weights = hash_weights();
  const std::array inputs{
      DeepSeekLearnedRouterLayerInput{0, 0x50000},
      DeepSeekLearnedRouterLayerInput{1, 0x51000}};
  auto work = DeepSeekHashRouterPlanInputAssembler::Assemble(
      3, inputs, weights, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 8);
  ASSERT_TRUE(work.ok()) << work.status().message();
  ASSERT_EQ(work->size(), 2U);
  EXPECT_EQ((*work)[0].submission.layer, 0U);
  EXPECT_EQ((*work)[1].submission.layer, 1U);
  EXPECT_EQ((*work)[0].submission.weight_bf16, 0x10000U);
  EXPECT_EQ((*work)[1].submission.weight_bf16, 0x210000U);
  EXPECT_EQ((*work)[0].submission.input_bf16, 0x50000U);
  EXPECT_EQ((*work)[0].submission.host_scores.size(), 3U * 256U);
  EXPECT_EQ((*work)[0].submission.host_error_flag,
            (*work)[0].host_staging->error_flag());
  EXPECT_FALSE(pool->acquire().ok());
}

TEST(DeepSeekHashRouterPlanInputAssemblerTest,
     RejectsLayerMismatchWithoutConsumingLease) {
  HashPlanPinnedAllocator allocator;
  auto pool = DeepSeekHashRouterStagingPool::Allocate({0, 1}, 2, allocator);
  ASSERT_TRUE(pool.ok());
  auto weights = hash_weights();
  const std::array inputs{
      DeepSeekLearnedRouterLayerInput{1, 0x50000},
      DeepSeekLearnedRouterLayerInput{0, 0x51000}};
  EXPECT_FALSE(DeepSeekHashRouterPlanInputAssembler::Assemble(
      2, inputs, weights, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 2).ok());
  EXPECT_TRUE(pool->acquire().ok());
}

TEST(DeepSeekHashRouterPlanInputAssemblerTest,
     RejectsAliasedRouterOutputWithoutConsumingLease) {
  HashPlanPinnedAllocator allocator;
  auto pool = DeepSeekHashRouterStagingPool::Allocate({0, 1}, 8, allocator);
  ASSERT_TRUE(pool.ok());
  auto weights = hash_weights();
  const std::array inputs{
      DeepSeekLearnedRouterLayerInput{0, 0x50000},
      DeepSeekLearnedRouterLayerInput{1, 0x51000}};
  EXPECT_FALSE(DeepSeekHashRouterPlanInputAssembler::Assemble(
      3, inputs, weights, {0x50000, 0x70000}, *pool,
      0x80000, 0x90000, 8).ok());
  EXPECT_TRUE(pool->acquire().ok());
}

TEST(DeepSeekHashRouterPlanInputAssemblerTest,
     LearnedOnlyRankProducesCanonicalEmptyHashWork) {
  HashPlanPinnedAllocator allocator;
  auto pool = DeepSeekHashRouterStagingPool::Allocate({3, 9}, 2, allocator);
  auto weights = DeepSeekHashRouterWeightBindings::Resolve(
      {3, 9}, [](std::string_view) -> Result<TensorView> {
        return Status::Internal("resolver must not be called");
      });
  ASSERT_TRUE(pool.ok()); ASSERT_TRUE(weights.ok());
  auto work = DeepSeekHashRouterPlanInputAssembler::Assemble(
      1, {}, *weights, {0x60000, 0x70000}, *pool,
      0x80000, 0x90000, 2);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_TRUE(work->empty());
}

}}  // namespace pih::<anonymous>
