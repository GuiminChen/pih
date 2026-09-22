#include "pih/model/qwen3_bf16_step_upload.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

QwenBf16StepStagingLayout layout() {
  const QwenKvBlockHandle handles[] = {{4, 9}, {7, 3}};
  auto table = QwenKvBlockTable::Create(2, 6, 32, handles).value();
  auto append = table.prepare_append(17).value();
  std::vector<std::int64_t> tokens(17, 4);
  auto input = QwenBf16StepInputPlan::Create(tokens, 0, table, append).value();
  return QwenBf16StepStagingLayout::Create(input).value();
}

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

TEST(QwenBf16StepUploadTest, SubmitsFiveExactSpansInFrozenOrder) {
  auto staging = layout();
  auto upload = QwenBf16StepUpload::Create(
      staging, host(staging.total_bytes()), device(staging.total_bytes()),
      17, 19, 23, 100);
  ASSERT_TRUE(upload.ok()) << upload.status().message();
  Driver driver;
  ASSERT_TRUE(upload->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes,
            (std::vector<std::uint64_t>{17 * 8, 17 * 8, 17 * 8, 17 * 2,
                                        2 * 8}));
  EXPECT_EQ(driver.streams, (std::vector<DriverStreamHandle>(5, 19)));
  EXPECT_EQ(upload->submitted_copies(), 5);
  EXPECT_EQ(upload->state(), QwenBf16StepUploadState::kSubmitted);
  EXPECT_FALSE(upload->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes.size(), 5);
}

TEST(QwenBf16StepUploadTest, PartialFailurePoisonsWithoutReplay) {
  auto staging = layout();
  auto upload = QwenBf16StepUpload::Create(
                    staging, host(staging.total_bytes()),
                    device(staging.total_bytes()), 17, 19, 23, 100)
                    .value();
  Driver driver;
  driver.fail_on = 3;
  EXPECT_FALSE(upload.submit(driver).ok());
  EXPECT_EQ(upload.submitted_copies(), 2);
  EXPECT_EQ(upload.state(), QwenBf16StepUploadState::kPoisoned);
  EXPECT_FALSE(upload.submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes.size(), 3);
}

TEST(QwenBf16StepUploadTest, RejectsArenaRangeContextAndPlanOverflow) {
  auto staging = layout();
  EXPECT_FALSE(QwenBf16StepUpload::Create(
                   staging, host(staging.total_bytes() - 1),
                   device(staging.total_bytes()), 17, 19, 23, 100)
                   .ok());
  EXPECT_FALSE(QwenBf16StepUpload::Create(
                   staging, host(staging.total_bytes()),
                   device(staging.total_bytes()), 0, 19, 23, 100)
                   .ok());
  EXPECT_FALSE(QwenBf16StepUpload::Create(
                   staging, host(staging.total_bytes()),
                   device(staging.total_bytes()), 17, 19, 23,
                   UINT64_MAX - 3)
                   .ok());
}

}  // namespace
}  // namespace pih
