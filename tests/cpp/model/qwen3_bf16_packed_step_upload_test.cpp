#include "pih/model/qwen3_bf16_packed_step_upload.h"

#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

CudaCopyEndpoint host(std::uint64_t bytes) {
  return {0x100000, bytes, 0, 11, 3,
          CudaCopyMemoryType::kRegisteredPinnedHost, 0, 0};
}

CudaCopyEndpoint device(std::uint64_t bytes) {
  return {0x200000, bytes, 0, 12, 5, CudaCopyMemoryType::kDevice, 0, 0};
}

class Driver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind, std::uintptr_t, std::uintptr_t,
              std::uint64_t bytes, DriverStreamHandle stream) override {
    copied_bytes.push_back(bytes);
    streams.push_back(stream);
    if (copied_bytes.size() == fail_on) return Status::Internal("copy failed");
    return Status::Ok();
  }
  std::size_t fail_on = 0;
  std::vector<std::uint64_t> copied_bytes;
  std::vector<DriverStreamHandle> streams;
};

TEST(QwenBf16PackedStepUploadTest, SubmitsFourteenExactSpansInFrozenOrder) {
  auto staging =
      QwenBf16PackedStepStagingLayout::CreateBounded(5, 2, 4).value();
  auto upload = QwenBf16PackedStepUpload::Create(
      staging, host(staging.total_bytes()), device(staging.total_bytes()),
      17, 19, 23, 100);
  ASSERT_TRUE(upload.ok()) << upload.status().message();
  Driver driver;
  ASSERT_TRUE(upload->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes,
            (std::vector<std::uint64_t>{20, 40, 20, 12, 8, 4,
                                        40, 10, 12, 32, 8, 8, 224, 8}));
  EXPECT_EQ(driver.streams, (std::vector<DriverStreamHandle>(14, 19)));
  EXPECT_EQ(upload->submitted_copies(), 14U);
  EXPECT_EQ(upload->state(), QwenBf16PackedStepUploadState::kSubmitted);
  EXPECT_FALSE(upload->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes.size(), 14U);
}

TEST(QwenBf16PackedStepUploadTest, PartialFailurePoisonsWithoutReplay) {
  auto staging =
      QwenBf16PackedStepStagingLayout::CreateBounded(5, 2, 4).value();
  auto upload = QwenBf16PackedStepUpload::Create(
                    staging, host(staging.total_bytes()),
                    device(staging.total_bytes()), 17, 19, 23, 100)
                    .value();
  Driver driver;
  driver.fail_on = 6;
  EXPECT_FALSE(upload.submit(driver).ok());
  EXPECT_EQ(upload.submitted_copies(), 5U);
  EXPECT_EQ(upload.state(), QwenBf16PackedStepUploadState::kPoisoned);
  EXPECT_FALSE(upload.submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes.size(), 6U);
}

TEST(QwenBf16PackedStepUploadTest, RejectsRangeAndPlanIdentityOverflow) {
  auto staging =
      QwenBf16PackedStepStagingLayout::CreateBounded(5, 2, 4).value();
  EXPECT_FALSE(QwenBf16PackedStepUpload::Create(
                   staging, host(staging.total_bytes() - 1),
                   device(staging.total_bytes()), 17, 19, 23, 100)
                   .ok());
  EXPECT_FALSE(QwenBf16PackedStepUpload::Create(
                   staging, host(staging.total_bytes()),
                   device(staging.total_bytes()), 17, 19, 23,
                   UINT64_MAX - 9)
                   .ok());
}

}  // namespace
}  // namespace pih
