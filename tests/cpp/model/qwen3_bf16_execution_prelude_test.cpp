#include "pih/model/qwen3_bf16_execution_prelude.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace pih {
namespace {

TensorView error_view(std::uint64_t generation = 7,
                      std::int32_t rank = 0,
                      std::int64_t bytes = 4) {
  const std::array<std::int64_t, 1> shape{bytes};
  auto device = Device::Create(DeviceType::kCuda, rank);
  auto view = TensorView::Create(reinterpret_cast<void*>(0x10000),
                                 DType::kUInt8, shape, {}, *device,
                                 generation);
  EXPECT_TRUE(view.ok());
  return std::move(view).value();
}

class ClearDriver final : public QwenBf16DeviceErrorClearDriver {
 public:
  Status clear_u32_async(const TensorView& target, std::int32_t owning_rank,
                         DriverStreamHandle stream) override {
    ++calls;
    seen_generation = target.generation();
    seen_rank = owning_rank;
    seen_stream = stream;
    return result;
  }

  Status result = Status::Ok();
  std::uint32_t calls = 0;
  std::uint64_t seen_generation = 0;
  std::int32_t seen_rank = -1;
  DriverStreamHandle seen_stream = 0;
};

TEST(QwenBf16ExecutionPreludeTest, ClearsExactGenerationOnceBeforeCommands) {
  auto prelude = QwenBf16ExecutionPrelude::Create(error_view(), 7, 0);
  ASSERT_TRUE(prelude.ok()) << prelude.status().message();
  ClearDriver driver;
  EXPECT_TRUE(prelude->submit(driver, 0x55).ok());
  EXPECT_EQ(driver.calls, 1);
  EXPECT_EQ(driver.seen_generation, 7);
  EXPECT_EQ(driver.seen_rank, 0);
  EXPECT_EQ(driver.seen_stream, 0x55);
  EXPECT_TRUE(prelude->submitted());
  EXPECT_FALSE(prelude->submit(driver, 0x55).ok());
  EXPECT_EQ(driver.calls, 1);
}

TEST(QwenBf16ExecutionPreludeTest, DriverFailureStillConsumesOneShotPrelude) {
  auto prelude = QwenBf16ExecutionPrelude::Create(error_view(), 7, 0);
  ASSERT_TRUE(prelude.ok());
  ClearDriver driver;
  driver.result = Status::Internal("injected clear failure");
  EXPECT_FALSE(prelude->submit(driver, 1).ok());
  EXPECT_TRUE(prelude->submitted());
  EXPECT_FALSE(prelude->submit(driver, 1).ok());
  EXPECT_EQ(driver.calls, 1);
}

TEST(QwenBf16ExecutionPreludeTest, RejectsExtentDeviceGenerationAndStreamDrift) {
  EXPECT_FALSE(QwenBf16ExecutionPrelude::Create(error_view(7, 0, 8), 7, 0)
                   .ok());
  EXPECT_FALSE(
      QwenBf16ExecutionPrelude::Create(error_view(7, 1), 7, 0).ok());
  EXPECT_FALSE(
      QwenBf16ExecutionPrelude::Create(error_view(8, 0), 7, 0).ok());
  auto prelude = QwenBf16ExecutionPrelude::Create(error_view(), 7, 0);
  ASSERT_TRUE(prelude.ok());
  ClearDriver driver;
  EXPECT_FALSE(prelude->submit(driver, 0).ok());
  EXPECT_EQ(driver.calls, 0);
  EXPECT_FALSE(prelude->submitted());
}

}  // namespace
}  // namespace pih
