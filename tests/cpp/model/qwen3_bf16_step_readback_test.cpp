#include "pih/model/qwen3_bf16_step_readback.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

CudaCopyEndpoint device(std::uintptr_t address, std::uint64_t bytes,
                        std::uint64_t owner) {
  return {address, bytes, 0, owner, 5, CudaCopyMemoryType::kDevice, 0, 0};
}

CudaCopyEndpoint host(std::uint64_t bytes) {
  return {0x300000, bytes, 0, 20, 7,
          CudaCopyMemoryType::kRegisteredPinnedHost, 0, 0};
}

class Driver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t destination,
              std::uintptr_t source, std::uint64_t bytes,
              DriverStreamHandle stream) override {
    kinds.push_back(kind);
    destinations.push_back(destination);
    sources.push_back(source);
    copied_bytes.push_back(bytes);
    streams.push_back(stream);
    if (copied_bytes.size() == fail_on) return Status::Internal("copy failed");
    return Status::Ok();
  }
  std::size_t fail_on = 0;
  std::vector<CudaCopyKind> kinds;
  std::vector<std::uintptr_t> destinations;
  std::vector<std::uintptr_t> sources;
  std::vector<std::uint64_t> copied_bytes;
  std::vector<DriverStreamHandle> streams;
};

TEST(QwenBf16StepReadbackTest, SubmitsTokenThenErrorOnExecutionStream) {
  auto readback = QwenBf16StepReadback::Create(
      device(0x100000, 8, 11), device(0x200000, 4, 12),
      host(QwenBf16StepResultLayout::kTotalBytes), 17, 19, 23, 200);
  ASSERT_TRUE(readback.ok()) << readback.status().message();
  Driver driver;
  ASSERT_TRUE(readback->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes, (std::vector<std::uint64_t>{8, 4}));
  EXPECT_EQ(driver.destinations,
            (std::vector<std::uintptr_t>{0x300000, 0x300100}));
  EXPECT_EQ(driver.streams, (std::vector<DriverStreamHandle>{19, 19}));
  EXPECT_EQ(readback->state(), QwenBf16StepReadbackState::kSubmitted);
  EXPECT_FALSE(readback->submit(driver).ok());
}

TEST(QwenBf16StepReadbackTest, PartialFailurePoisonsWithoutReplay) {
  auto readback = QwenBf16StepReadback::Create(
                      device(0x100000, 8, 11), device(0x200000, 4, 12),
                      host(QwenBf16StepResultLayout::kTotalBytes), 17, 19, 23,
                      200)
                      .value();
  Driver driver;
  driver.fail_on = 2;
  EXPECT_FALSE(readback.submit(driver).ok());
  EXPECT_EQ(readback.submitted_copies(), 1);
  EXPECT_EQ(readback.state(), QwenBf16StepReadbackState::kPoisoned);
  EXPECT_FALSE(readback.submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes.size(), 2);
}

TEST(QwenBf16StepReadbackTest, RejectsOwnerRangeAndIdentityDrift) {
  EXPECT_FALSE(QwenBf16StepReadback::Create(
                   device(0x100000, 7, 11), device(0x200000, 4, 12),
                   host(QwenBf16StepResultLayout::kTotalBytes), 17, 19, 23,
                   200)
                   .ok());
  EXPECT_FALSE(QwenBf16StepReadback::Create(
                   device(0x100000, 8, 11), device(0x200000, 4, 12),
                   host(QwenBf16StepResultLayout::kTotalBytes - 1), 17, 19,
                   23, 200)
                   .ok());
  EXPECT_FALSE(QwenBf16StepReadback::Create(
                   device(0x100000, 8, 11), device(0x200000, 4, 12),
                   host(QwenBf16StepResultLayout::kTotalBytes), 0, 19, 23,
                   200)
                   .ok());
}

}  // namespace
}  // namespace pih
