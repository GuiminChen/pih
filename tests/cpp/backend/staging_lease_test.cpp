#include "pih/backend/cuda/staging_lease.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class CopyDriver final : public TypedCopyDriver {
 public:
  [[nodiscard]] std::uintptr_t context_identity() const noexcept override {
    return 17;
  }
  Status copy(CudaCopyKind, std::uintptr_t, std::uintptr_t, std::uint64_t,
              DriverStreamHandle) override {
    ++calls;
    return result;
  }
  Status result = Status::Ok();
  int calls = 0;
};

CudaCopyEndpoint host_endpoint(std::uint64_t owner, std::uint64_t generation,
                               std::uint64_t bytes) {
  return {0x1000, bytes, 0, owner, generation,
          CudaCopyMemoryType::kRegisteredPinnedHost, 0, 0};
}

CudaCopyEndpoint device_endpoint(std::uint64_t owner,
                                 std::uint64_t generation,
                                 std::uint64_t bytes) {
  return {0x4000, bytes, 0, owner, generation, CudaCopyMemoryType::kDevice, 0,
          0};
}

CudaTypedCopyPlan input_plan(std::uint64_t staging_generation,
                             std::uint64_t bytes = 1024) {
  return CudaTypedCopyPlan::Create(
             41, CudaCopyPurpose::kInput, CudaCopyKind::kHostToDevice,
             host_endpoint(7, staging_generation, bytes),
             device_endpoint(9, 3, bytes), bytes, 128, 17, 19, 23)
      .value();
}

CudaCompletionFrontier successful_frontier(std::uint64_t plan_generation) {
  auto frontier = CudaCompletionFrontier::Create(
                      {11, 0, plan_generation, CudaCompletionPhase::kCopy, 29},
                      23, 100, 200)
                      .value();
  EXPECT_TRUE(frontier
                  .observe(23, CudaEventQueryResult::kSuccess, true, 0, false)
                  .ok());
  return frontier;
}

TEST(CudaStagingLeaseTest, CpuFillCopyCompletionConsumeAndReuseIsGenerational) {
  auto lease = CudaStagingLease::Create(7, 4096).value();
  ASSERT_TRUE(lease.begin_cpu_fill(31, 1024).ok());
  EXPECT_EQ(lease.state(), CudaStagingState::kCpuFilling);
  ASSERT_TRUE(lease.mark_ready(31).ok());
  auto plan = input_plan(31);
  CopyDriver driver;
  ASSERT_TRUE(lease.submit_copy(plan, driver).ok());
  EXPECT_EQ(lease.state(), CudaStagingState::kInFlight);
  auto frontier = successful_frontier(41);
  ASSERT_TRUE(lease.complete_copy(frontier).ok());
  ASSERT_TRUE(lease.consume(31).ok());
  ASSERT_TRUE(lease.release(31).ok());
  EXPECT_EQ(lease.state(), CudaStagingState::kFree);
  EXPECT_FALSE(lease.begin_cpu_fill(31, 1024).ok());
  EXPECT_TRUE(lease.begin_gpu_fill(32, 2048, 51).ok());
}

TEST(CudaStagingLeaseTest, RejectsEarlyReadReuseAndCopyGenerationDrift) {
  auto lease = CudaStagingLease::Create(7, 4096).value();
  ASSERT_TRUE(lease.begin_cpu_fill(31, 1024).ok());
  EXPECT_FALSE(lease.consume(31).ok());
  EXPECT_FALSE(lease.release(31).ok());
  EXPECT_FALSE(lease.begin_gpu_fill(32, 1024, 51).ok());
  ASSERT_TRUE(lease.mark_ready(31).ok());
  auto stale_plan = input_plan(30);
  CopyDriver driver;
  EXPECT_FALSE(lease.submit_copy(stale_plan, driver).ok());
  EXPECT_EQ(driver.calls, 0);
  EXPECT_EQ(lease.state(), CudaStagingState::kReadyToSubmit);
}

TEST(CudaStagingLeaseTest, CompletionRequiresMatchingAuthorizedFrontier) {
  auto lease = CudaStagingLease::Create(7, 4096).value();
  ASSERT_TRUE(lease.begin_cpu_fill(31, 1024).ok());
  ASSERT_TRUE(lease.mark_ready(31).ok());
  auto plan = input_plan(31);
  CopyDriver driver;
  ASSERT_TRUE(lease.submit_copy(plan, driver).ok());

  auto pending = CudaCompletionFrontier::Create(
                     {11, 0, 41, CudaCompletionPhase::kCopy, 29}, 23, 100, 200)
                     .value();
  EXPECT_FALSE(lease.complete_copy(pending).ok());
  EXPECT_EQ(lease.state(), CudaStagingState::kInFlight);
  auto wrong_plan = successful_frontier(42);
  EXPECT_FALSE(lease.complete_copy(wrong_plan).ok());
  auto success = successful_frontier(41);
  EXPECT_TRUE(lease.complete_copy(success).ok());
}

TEST(CudaStagingLeaseTest, CopyDriverFailureMakesSlotSuspect) {
  auto lease = CudaStagingLease::Create(7, 4096).value();
  ASSERT_TRUE(lease.begin_cpu_fill(31, 1024).ok());
  ASSERT_TRUE(lease.mark_ready(31).ok());
  auto plan = input_plan(31);
  CopyDriver driver;
  driver.result = Status::Internal("injected copy failure");
  EXPECT_FALSE(lease.submit_copy(plan, driver).ok());
  EXPECT_EQ(driver.calls, 1);
  EXPECT_TRUE(plan.submitted());
  EXPECT_TRUE(lease.suspect());
}

TEST(CudaStagingLeaseTest, FailureMakesSlotPermanentlySuspect) {
  auto lease = CudaStagingLease::Create(7, 4096).value();
  ASSERT_TRUE(lease.begin_gpu_fill(31, 1024, 51).ok());
  EXPECT_FALSE(lease.fail(31).ok());
  EXPECT_EQ(lease.state(), CudaStagingState::kSuspect);
  EXPECT_TRUE(lease.suspect());
  EXPECT_FALSE(lease.begin_cpu_fill(32, 1024).ok());
  EXPECT_FALSE(lease.release(31).ok());
  EXPECT_FALSE(lease.fail(31).ok());
}

TEST(CudaStagingLeaseTest, GpuFillRequiresMatchingAuthorizedProducer) {
  auto lease = CudaStagingLease::Create(7, 4096).value();
  ASSERT_TRUE(lease.begin_gpu_fill(31, 1024, 51).ok());
  EXPECT_FALSE(lease.mark_ready(31).ok());
  auto wrong_plan = successful_frontier(52);
  EXPECT_FALSE(lease.complete_gpu_fill(wrong_plan).ok());
  auto producer = successful_frontier(51);
  EXPECT_TRUE(lease.complete_gpu_fill(producer).ok());
  EXPECT_EQ(lease.state(), CudaStagingState::kReadyToSubmit);
}

TEST(CudaStagingLeaseTest, RejectsMalformedSlotAndExtent) {
  EXPECT_FALSE(CudaStagingLease::Create(0, 4096).ok());
  EXPECT_FALSE(CudaStagingLease::Create(7, 0).ok());
  auto lease = CudaStagingLease::Create(7, 4096).value();
  EXPECT_FALSE(lease.begin_cpu_fill(0, 1024).ok());
  EXPECT_FALSE(lease.begin_cpu_fill(31, 0).ok());
  EXPECT_FALSE(lease.begin_cpu_fill(31, 4097).ok());
  EXPECT_FALSE(lease.begin_gpu_fill(31, 1024, 0).ok());
}

}  // namespace
}  // namespace pih
