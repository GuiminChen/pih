#include "pih/model/qwen3_bf16_packed_readback.h"

#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

CudaCopyEndpoint packed_device(std::uintptr_t address, std::uint64_t bytes,
                               std::uint64_t owner) {
  return {address, bytes, 0, owner, 5, CudaCopyMemoryType::kDevice, 0, 0};
}

CudaCopyEndpoint packed_host(std::uint64_t bytes) {
  return {0x300000, bytes, 0, 20, 7,
          CudaCopyMemoryType::kRegisteredPinnedHost, 0, 0};
}

class PackedCopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind, std::uintptr_t destination,
              std::uintptr_t, std::uint64_t bytes,
              DriverStreamHandle) override {
    destinations.push_back(destination);
    copied_bytes.push_back(bytes);
    if (copied_bytes.size() == fail_on) return Status::Internal("injected");
    return Status::Ok();
  }
  std::size_t fail_on = 0;
  std::vector<std::uintptr_t> destinations;
  std::vector<std::uint64_t> copied_bytes;
};

TEST(QwenBf16PackedReadbackTest, CopiesOnlyActiveTokensThenError) {
  auto layout = QwenBf16PackedResultLayout::Create(7).value();
  auto readback = QwenBf16PackedReadback::Create(
      layout, 3, packed_device(0x100000, 28, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes()),
      17, 19, 23, 200);
  ASSERT_TRUE(readback.ok()) << readback.status().message();
  PackedCopyDriver driver;
  ASSERT_TRUE(readback->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes, (std::vector<std::uint64_t>{12, 4}));
  EXPECT_EQ(driver.destinations,
            (std::vector<std::uintptr_t>{
                0x300000,
                0x300000 + layout.device_error().offset_bytes}));
  EXPECT_FALSE(readback->submit(driver).ok());
}

TEST(QwenBf16PackedReadbackTest, ZeroSampleCopiesOnlyDeviceError) {
  auto layout = QwenBf16PackedResultLayout::Create(7).value();
  auto readback = QwenBf16PackedReadback::Create(
      layout, 0, packed_device(0x100000, 28, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes()),
      17, 19, 23, 200);
  ASSERT_TRUE(readback.ok()) << readback.status().message();
  EXPECT_EQ(readback->size(), 1U);
  PackedCopyDriver driver;
  ASSERT_TRUE(readback->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes, (std::vector<std::uint64_t>{4}));
  EXPECT_EQ(driver.destinations,
            (std::vector<std::uintptr_t>{
                0x300000 + layout.device_error().offset_bytes}));
}

TEST(QwenBf16PackedReadbackTest,
     CopiesCompleteActiveSamplingReceiptBeforeError) {
  auto layout = QwenBf16PackedResultLayout::Create(7).value();
  auto readback = QwenBf16PackedReadback::CreateSampling(
      layout, 3,
      packed_device(0x100000, layout.device_error().offset_bytes, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes()),
      17, 19, 23, 200);
  ASSERT_TRUE(readback.ok()) << readback.status().message();
  PackedCopyDriver driver;
  ASSERT_TRUE(readback->submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes,
            (std::vector<std::uint64_t>{12, 12, 12, 240, 240, 12, 4}));
  EXPECT_EQ(driver.destinations,
            (std::vector<std::uintptr_t>{
                0x300000 + layout.sampled_token_ids().offset_bytes,
                0x300000 + layout.selected_logprobs().offset_bytes,
                0x300000 + layout.rng_words().offset_bytes,
                0x300000 + layout.top_logprob_token_ids().offset_bytes,
                0x300000 + layout.top_logprobs().offset_bytes,
                0x300000 + layout.top_logprob_counts().offset_bytes,
                0x300000 + layout.device_error().offset_bytes}));
  EXPECT_EQ(readback->size(), 7U);
}

TEST(QwenBf16PackedReadbackTest, PartialFailurePoisonsWithoutReplay) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  auto readback = QwenBf16PackedReadback::Create(
      layout, 2, packed_device(0x100000, 8, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes()),
      17, 19, 23, 200).value();
  PackedCopyDriver driver;
  driver.fail_on = 2;
  EXPECT_FALSE(readback.submit(driver).ok());
  EXPECT_EQ(readback.submitted_copies(), 1);
  EXPECT_EQ(readback.state(), QwenBf16PackedReadbackState::kPoisoned);
  EXPECT_FALSE(readback.submit(driver).ok());
  EXPECT_EQ(driver.copied_bytes.size(), 2U);
}

TEST(QwenBf16PackedReadbackTest, RejectsCapacityAndOwnerRangeDrift) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  EXPECT_FALSE(QwenBf16PackedReadback::Create(
      layout, 3, packed_device(0x100000, 12, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes()),
      17, 19, 23, 200).ok());
  EXPECT_FALSE(QwenBf16PackedReadback::Create(
      layout, 2, packed_device(0x100000, 7, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes()),
      17, 19, 23, 200).ok());
  EXPECT_FALSE(QwenBf16PackedReadback::Create(
      layout, 2, packed_device(0x100000, 8, 11),
      packed_device(0x200000, 4, 12), packed_host(layout.total_bytes() - 1),
      17, 19, 23, 200).ok());
}

}  // namespace
}  // namespace pih
