#include "pih/model/qwen3_bf16_step_staging_layout.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace pih {
namespace {

QwenBf16StepInputPlan input_plan(std::uint32_t target) {
  const QwenKvBlockHandle handles[] = {{4, 9}, {7, 3}};
  auto table = QwenKvBlockTable::Create(2, 6, 32, handles).value();
  auto append = table.prepare_append(target).value();
  std::vector<std::int64_t> tokens(target);
  for (std::uint32_t index = 0; index < target; ++index) tokens[index] = index;
  return QwenBf16StepInputPlan::Create(tokens, 0, table, append).value();
}

template <typename T>
T read(const std::vector<std::byte>& backing, QwenBf16ArenaSpan span,
       std::size_t index) {
  T result{};
  std::memcpy(&result,
              backing.data() + span.offset_bytes + index * sizeof(T),
              sizeof(T));
  return result;
}

TEST(QwenBf16StepStagingLayoutTest, FreezesFiveAlignedTypedSpans) {
  auto input = input_plan(17);
  auto layout = QwenBf16StepStagingLayout::Create(input);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  for (const auto span : {layout->token_ids(), layout->positions(),
                          layout->append_handles(), layout->token_offsets(),
                          layout->visible_handles()}) {
    EXPECT_EQ(span.offset_bytes % QwenBf16StepStagingLayout::kAlignment, 0);
  }
  EXPECT_EQ(layout->token_ids().size_bytes, 17 * sizeof(std::int64_t));
  EXPECT_EQ(layout->visible_handles().size_bytes,
            2 * sizeof(QwenKvBlockHandle));
  EXPECT_EQ(layout->total_bytes() % QwenBf16StepStagingLayout::kAlignment, 0);
}

TEST(QwenBf16StepStagingLayoutTest, MaterializesExactCrossBlockBytes) {
  auto input = input_plan(17);
  auto layout = QwenBf16StepStagingLayout::Create(input).value();
  std::vector<std::byte> backing(layout.total_bytes(), std::byte{0xa5});
  ASSERT_TRUE(layout.materialize(input, backing).ok());
  EXPECT_EQ(read<std::int64_t>(backing, layout.token_ids(), 16), 16);
  EXPECT_EQ(read<std::int64_t>(backing, layout.positions(), 16), 16);
  EXPECT_EQ(read<QwenKvBlockHandle>(backing, layout.append_handles(), 16),
            (QwenKvBlockHandle{7, 3}));
  EXPECT_EQ(read<std::uint16_t>(backing, layout.token_offsets(), 16), 0);
  EXPECT_EQ(read<QwenKvBlockHandle>(backing, layout.visible_handles(), 1),
            (QwenKvBlockHandle{7, 3}));
}

TEST(QwenBf16StepStagingLayoutTest, RejectsBackingAndGenerationShapeDrift) {
  auto input = input_plan(1);
  auto layout = QwenBf16StepStagingLayout::Create(input).value();
  std::vector<std::byte> short_backing(layout.total_bytes() - 1);
  EXPECT_FALSE(layout.materialize(input, short_backing).ok());
  auto other = input_plan(2);
  std::vector<std::byte> backing(layout.total_bytes());
  EXPECT_FALSE(layout.materialize(other, backing).ok());
}

}  // namespace
}  // namespace pih
