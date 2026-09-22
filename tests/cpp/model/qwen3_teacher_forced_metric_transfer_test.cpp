#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_metric_transfer.h"

namespace pih {
namespace {

CudaCopyEndpoint endpoint(std::uintptr_t address, std::uint64_t bytes,
                          std::uint64_t owner, CudaCopyMemoryType type) {
  return {address, bytes, 0, owner, 3, type, 0, 0};
}

class Driver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t bytes, DriverStreamHandle stream) override {
    kinds[calls] = kind; sizes[calls] = bytes; streams[calls] = stream;
    ++calls;
    return calls == fail_at ? Status::Internal("injected copy") : Status::Ok();
  }
  int calls = 0, fail_at = 99;
  CudaCopyKind kinds[3]{};
  std::uint64_t sizes[3]{};
  DriverStreamHandle streams[3]{};
};

Result<QwenTeacherForcedMetricTransfer> transfer(std::uint32_t rows = 17) {
  auto layout = QwenTeacherForcedMetricResultLayout::Create(32).value();
  return QwenTeacherForcedMetricTransfer::Create(
      layout, rows,
      endpoint(0x080000, 32 * 4, 5,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      endpoint(0x180000, 32 * 4, 6, CudaCopyMemoryType::kDevice),
      endpoint(0x100000, 32 * 4, 1,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      endpoint(0x200000, 32 * 4, 2, CudaCopyMemoryType::kDevice),
      endpoint(0x300000, layout.total_bytes(), 3,
               CudaCopyMemoryType::kDevice),
      endpoint(0x400000, layout.total_bytes(), 4,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      17, 19, 23, 100, 101, 102);
}

TEST(QwenTeacherForcedMetricTransferTest, OrdersBoundedUploadBeforeReadback) {
  auto value = transfer();
  ASSERT_TRUE(value.ok()) << value.status().message();
  Driver driver;
  EXPECT_FALSE(value->submit_readback(driver).ok());
  ASSERT_TRUE(value->submit_upload(driver).ok());
  ASSERT_TRUE(value->submit_readback(driver).ok());
  EXPECT_EQ(driver.calls, 3);
  EXPECT_EQ(driver.kinds[0], CudaCopyKind::kHostToDevice);
  EXPECT_EQ(driver.kinds[1], CudaCopyKind::kHostToDevice);
  EXPECT_EQ(driver.kinds[2], CudaCopyKind::kDeviceToHost);
  EXPECT_EQ(driver.sizes[0], 17U * 4U);
  EXPECT_EQ(driver.sizes[1], 17U * 4U);
  EXPECT_EQ(driver.streams[2], 19U);
  EXPECT_EQ(value->state(),
            QwenTeacherForcedMetricTransferState::kReadbackSubmitted);
}

TEST(QwenTeacherForcedMetricTransferTest, CopyFailurePoisonsAndPreventsReplay) {
  auto value = transfer().value();
  Driver driver; driver.fail_at = 1;
  EXPECT_FALSE(value.submit_upload(driver).ok());
  EXPECT_EQ(value.state(), QwenTeacherForcedMetricTransferState::kPoisoned);
  EXPECT_FALSE(value.submit_upload(driver).ok());
  EXPECT_FALSE(value.submit_readback(driver).ok());
  EXPECT_EQ(driver.calls, 1);
}

TEST(QwenTeacherForcedMetricTransferTest, RejectsCapacityAndEndpointDrift) {
  EXPECT_FALSE(transfer(0).ok());
  EXPECT_FALSE(transfer(33).ok());
  auto layout = QwenTeacherForcedMetricResultLayout::Create(1).value();
  EXPECT_FALSE(QwenTeacherForcedMetricTransfer::Create(
      layout, 1,
      endpoint(5, 4, 5, CudaCopyMemoryType::kRegisteredPinnedHost),
      endpoint(6, 4, 6, CudaCopyMemoryType::kDevice),
      endpoint(1, 4, 1, CudaCopyMemoryType::kDevice),
      endpoint(2, 4, 2, CudaCopyMemoryType::kDevice),
      endpoint(3, layout.total_bytes(), 3, CudaCopyMemoryType::kDevice),
      endpoint(4, layout.total_bytes(), 4,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      17, 19, 23, 1, 2, 3).ok());
}

}  // namespace
}  // namespace pih
