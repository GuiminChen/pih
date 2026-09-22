#include "pih/model/qwen3_bf16_step_result_layout.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "pih/model/qwen3_cuda_invariant.h"

namespace pih {
namespace {

template <typename T>
void write(std::vector<std::byte>& backing, QwenBf16ArenaSpan span, T value) {
  std::memcpy(backing.data() + span.offset_bytes, &value, sizeof(value));
}

TEST(QwenBf16StepResultLayoutTest, PublishesOnlyCompletedCleanValidToken) {
  std::vector<std::byte> backing(QwenBf16StepResultLayout::kTotalBytes);
  ASSERT_TRUE(QwenBf16StepResultLayout::initialize(backing).ok());
  write(backing, QwenBf16StepResultLayout::sampled_token(),
        std::int64_t{151935});
  write(backing, QwenBf16StepResultLayout::device_error(), std::uint32_t{0});
  EXPECT_FALSE(QwenBf16StepResultLayout::parse(backing, false).ok());
  auto token = QwenBf16StepResultLayout::parse(backing, true);
  ASSERT_TRUE(token.ok());
  EXPECT_EQ(*token, 151935);
}

TEST(QwenBf16StepResultLayoutTest, InitializationMakesEarlyReadInvalid) {
  std::vector<std::byte> backing(QwenBf16StepResultLayout::kTotalBytes);
  ASSERT_TRUE(QwenBf16StepResultLayout::initialize(backing).ok());
  EXPECT_FALSE(QwenBf16StepResultLayout::parse(backing, true).ok());
}

TEST(QwenBf16StepResultLayoutTest, RejectsDeviceAndTokenFailures) {
  std::vector<std::byte> backing(QwenBf16StepResultLayout::kTotalBytes);
  ASSERT_TRUE(QwenBf16StepResultLayout::initialize(backing).ok());
  write(backing, QwenBf16StepResultLayout::sampled_token(), std::int64_t{7});
  write(backing, QwenBf16StepResultLayout::device_error(),
        static_cast<std::uint32_t>(QwenCudaInvariant::kPagedGqaSlotRange));
  EXPECT_FALSE(QwenBf16StepResultLayout::parse(backing, true).ok());
  write(backing, QwenBf16StepResultLayout::device_error(), UINT32_MAX);
  EXPECT_FALSE(QwenBf16StepResultLayout::parse(backing, true).ok());
  write(backing, QwenBf16StepResultLayout::device_error(), std::uint32_t{0});
  write(backing, QwenBf16StepResultLayout::sampled_token(),
        QwenBf16StepResultLayout::kVocabularySize);
  EXPECT_FALSE(QwenBf16StepResultLayout::parse(backing, true).ok());
}

TEST(QwenBf16StepResultLayoutTest, RejectsNonexactBacking) {
  std::vector<std::byte> short_backing(
      QwenBf16StepResultLayout::kTotalBytes - 1);
  EXPECT_FALSE(QwenBf16StepResultLayout::initialize(short_backing).ok());
  EXPECT_FALSE(QwenBf16StepResultLayout::parse(short_backing, true).ok());
}

}  // namespace
}  // namespace pih
