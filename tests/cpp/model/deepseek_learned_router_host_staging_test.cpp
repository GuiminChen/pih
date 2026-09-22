#include "pih/model/deepseek_learned_router_host_staging.h"
#include "pih/model/deepseek_rank_compute_work_builder.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

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

TEST(DeepSeekLearnedRouterHostStagingTest,
     AllocatesAlignedScoresAndSeparateErrorFlag) {
  PinnedAllocator allocator;
  auto staging = DeepSeekLearnedRouterHostStaging::Allocate(4, allocator);
  ASSERT_TRUE(staging.ok()) << staging.status().message();
  auto scores = staging->scores(2);
  ASSERT_TRUE(scores.ok());
  EXPECT_EQ(scores->size(), 2U * 256U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(scores->data()) % 256U, 0U);
  EXPECT_GT(reinterpret_cast<std::uintptr_t>(staging->error_flag()),
            reinterpret_cast<std::uintptr_t>(scores->data()));
  EXPECT_NE(staging->generation(), 0U);
}

TEST(DeepSeekLearnedRouterHostStagingTest,
     RejectsInvalidCapacityAndTokenViews) {
  PinnedAllocator allocator;
  EXPECT_FALSE(DeepSeekLearnedRouterHostStaging::Allocate(0, allocator).ok());
  EXPECT_FALSE(
      DeepSeekLearnedRouterHostStaging::Allocate(4097, allocator).ok());
  auto staging = DeepSeekLearnedRouterHostStaging::Allocate(2, allocator)
                     .value();
  EXPECT_FALSE(staging.scores(0).ok());
  EXPECT_FALSE(staging.scores(3).ok());
}

TEST(DeepSeekLearnedRouterHostStagingTest,
     PlanOwnerRetainsPinnedScoresAndErrorStorage) {
  PinnedAllocator allocator;
  auto staging = std::make_shared<DeepSeekLearnedRouterHostStaging>(
      DeepSeekLearnedRouterHostStaging::Allocate(2, allocator).value());
  const auto scores = staging->scores(2).value();
  auto* error = staging->error_flag();
  DeepSeekLearnedRouterSubmission submission;
  submission.layer = 3;
  submission.token_count = 2;
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(builder.add_learned_router_with_host_staging(
      3, reinterpret_cast<DeepSeekLearnedRouterCoordinator*>(0x1000),
      submission, staging).ok());
  staging.reset();
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.learned_router.size(), 1U);
  EXPECT_EQ(work.learned_router[0].submission.host_scores.data(),
            scores.data());
  EXPECT_EQ(work.learned_router[0].submission.host_error_flag, error);
  EXPECT_EQ(work.learned_router[0].submission.host_scores.size(), 512U);
}

}  // namespace
}  // namespace pih
