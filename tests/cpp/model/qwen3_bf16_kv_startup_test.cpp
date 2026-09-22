#include "pih/model/qwen3_bf16_kv_startup.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Driver final : public QwenBf16KvScrubDriver {
 public:
  Status clear_and_wait(std::uintptr_t destination, std::uint64_t bytes,
                        DriverStreamHandle stream,
                        DriverEventHandle event) override {
    calls.push_back({destination, bytes, stream, event});
    if (calls.size() == fail_on_call) {
      return Status::Internal("injected startup clear failure");
    }
    return Status::Ok();
  }
  struct Call {
    std::uintptr_t destination;
    std::uint64_t bytes;
    DriverStreamHandle stream;
    DriverEventHandle event;
  };
  std::vector<Call> calls;
  std::size_t fail_on_call = 0;
};

TEST(QwenBf16KvStartupTest, PublishesCreditsOnlyAfterBothDeviceBackingsClear) {
  constexpr std::uint32_t kSlots = 3;
  auto pool = QwenKvSlotPool::Create(
      kSlots, kSlots * QwenKvSlotPool::kSlotPayloadBytes,
      kSlots * sizeof(QwenKvSlotState)).value();
  Driver driver;
  const QwenBf16DeviceArenaOwner kv{
      0x10000000, kSlots * QwenKvSlotPool::kSlotPayloadBytes, 7};
  const QwenBf16DeviceArenaOwner metadata{
      0x20000000, kSlots * sizeof(QwenKvSlotState), 11};
  ASSERT_TRUE(QwenBf16KvStartup::SanitizeAndPublish(
      pool, kv, metadata, 13, 17, driver).ok());
  ASSERT_EQ(driver.calls.size(), 2U);
  EXPECT_EQ(driver.calls[0].destination, kv.base);
  EXPECT_EQ(driver.calls[1].destination, metadata.base);
  EXPECT_TRUE(pool.ready());
  EXPECT_FALSE(pool.failed());
  EXPECT_EQ(pool.clean_credits(), kSlots);
}

TEST(QwenBf16KvStartupTest, SecondClearFailurePermanentlyFailsPool) {
  constexpr std::uint32_t kSlots = 2;
  auto pool = QwenKvSlotPool::Create(
      kSlots, kSlots * QwenKvSlotPool::kSlotPayloadBytes,
      kSlots * sizeof(QwenKvSlotState)).value();
  Driver driver;
  driver.fail_on_call = 2;
  const QwenBf16DeviceArenaOwner kv{
      0x10000000, kSlots * QwenKvSlotPool::kSlotPayloadBytes, 7};
  const QwenBf16DeviceArenaOwner metadata{
      0x20000000, kSlots * sizeof(QwenKvSlotState), 11};
  EXPECT_FALSE(QwenBf16KvStartup::SanitizeAndPublish(
      pool, kv, metadata, 13, 17, driver).ok());
  EXPECT_FALSE(pool.ready());
  EXPECT_TRUE(pool.failed());
  EXPECT_EQ(pool.clean_credits(), 0U);
}

TEST(QwenBf16KvStartupTest, RejectsBackingDriftBeforeAnyClear) {
  auto pool = QwenKvSlotPool::Create(
      1, QwenKvSlotPool::kSlotPayloadBytes,
      sizeof(QwenKvSlotState)).value();
  Driver driver;
  const QwenBf16DeviceArenaOwner kv{
      0x10000000, QwenKvSlotPool::kSlotPayloadBytes - 1, 7};
  const QwenBf16DeviceArenaOwner metadata{
      0x20000000, sizeof(QwenKvSlotState), 11};
  EXPECT_FALSE(QwenBf16KvStartup::SanitizeAndPublish(
      pool, kv, metadata, 13, 17, driver).ok());
  EXPECT_TRUE(driver.calls.empty());
  EXPECT_FALSE(pool.ready());
  EXPECT_FALSE(pool.failed());
}

}  // namespace
}  // namespace pih
