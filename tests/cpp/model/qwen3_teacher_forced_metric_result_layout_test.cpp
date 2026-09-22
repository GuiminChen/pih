#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_metric_result_layout.h"

namespace pih {
namespace {

template <typename T>
void store(std::vector<std::byte>& bytes, std::uint64_t offset, T value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

TEST(QwenTeacherForcedMetricResultLayoutTest, ParsesCompactFiniteAndNonfiniteRows) {
  auto layout = QwenTeacherForcedMetricResultLayout::Create(3);
  ASSERT_TRUE(layout.ok());
  std::vector<std::byte> bytes(layout->total_bytes());
  ASSERT_TRUE(layout->initialize(bytes).ok());
  store(bytes, layout->device_error().offset_bytes, std::uint32_t{0});
  const std::array<std::uint32_t, 3> tokens{7, 8, 9};
  const std::array<double, 3> nll{1.25, 0.0, 2.75};
  const std::array<std::uint32_t, 3> flags{0, 1, 0};
  for (std::size_t row = 0; row < 3; ++row) {
    store(bytes, layout->argmax_tokens().offset_bytes + row * 4, tokens[row]);
    store(bytes, layout->target_nll().offset_bytes + row * 8, nll[row]);
    store(bytes, layout->nonfinite_rows().offset_bytes + row * 4, flags[row]);
  }
  auto parsed = layout->parse(bytes, 3, true);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->rows.size(), 3U);
  EXPECT_EQ(parsed->nonfinite_count, 1U);
  EXPECT_DOUBLE_EQ(parsed->nll_sum, 4.0);
  EXPECT_FALSE(parsed->rows[1].finite);
}

TEST(QwenTeacherForcedMetricResultLayoutTest, RejectsUnpublishedAndMalformedRows) {
  auto layout = QwenTeacherForcedMetricResultLayout::Create(1).value();
  std::vector<std::byte> bytes(layout.total_bytes());
  ASSERT_TRUE(layout.initialize(bytes).ok());
  EXPECT_FALSE(layout.parse(bytes, 1, false).ok());
  store(bytes, layout.device_error().offset_bytes, std::uint32_t{0});
  store(bytes, layout.argmax_tokens().offset_bytes, std::uint32_t{151936});
  store(bytes, layout.target_nll().offset_bytes, 1.0);
  store(bytes, layout.nonfinite_rows().offset_bytes, std::uint32_t{0});
  EXPECT_FALSE(layout.parse(bytes, 1, true).ok());
  store(bytes, layout.argmax_tokens().offset_bytes, std::uint32_t{3});
  store(bytes, layout.nonfinite_rows().offset_bytes, std::uint32_t{1});
  EXPECT_FALSE(layout.parse(bytes, 1, true).ok());
}

TEST(QwenTeacherForcedMetricResultLayoutTest, FreezesCapacityAndAlignmentBounds) {
  EXPECT_FALSE(QwenTeacherForcedMetricResultLayout::Create(0).ok());
  EXPECT_FALSE(QwenTeacherForcedMetricResultLayout::Create(4097).ok());
  auto layout = QwenTeacherForcedMetricResultLayout::Create(4096).value();
  EXPECT_EQ(layout.target_nll().offset_bytes % 256, 0U);
  EXPECT_EQ(layout.nonfinite_rows().offset_bytes % 256, 0U);
  EXPECT_EQ(layout.device_error().offset_bytes % 256, 0U);
  EXPECT_EQ(layout.total_bytes() % 256, 0U);
}

}  // namespace
}  // namespace pih
