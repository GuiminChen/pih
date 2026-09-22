#include "pih/model/qwen3_bf16_packed_step_staging_layout.h"

#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

template <typename T>
T read(std::span<const std::byte> backing, QwenBf16ArenaSpan span,
       std::size_t index = 0) {
  T result{};
  std::memcpy(&result,
              backing.data() + span.offset_bytes + index * sizeof(T),
              sizeof(T));
  return result;
}

struct Metadata final {
  std::array<std::uint32_t, 5> tokens{7, 8, 9, 0, 0};
  std::array<std::uint64_t, 5> positions{15, 16, 16, 0, 0};
  std::array<std::uint32_t, 5> request_index{
      0, 0, 1, kInvalidPackedRequestIndex, kInvalidPackedRequestIndex};
  std::array<std::uint32_t, 3> query_offsets{0, 2, 3};
  std::array<std::uint32_t, 1> sample_rows{2};
  std::array<QwenKvBlockHandle, 5> append_handles{
      QwenKvBlockHandle{10, 1}, QwenKvBlockHandle{11, 1},
      QwenKvBlockHandle{21, 1}, kInvalidPackedKvBlockHandle,
      kInvalidPackedKvBlockHandle};
  std::array<std::uint16_t, 5> token_offsets{15, 0, 0, 0, 0};
  std::array<std::uint32_t, 3> visible_offsets{0, 2, 4};
  std::array<QwenKvBlockHandle, 4> visible_handles{
      QwenKvBlockHandle{10, 1}, QwenKvBlockHandle{11, 1},
      QwenKvBlockHandle{20, 1}, QwenKvBlockHandle{21, 1}};
  std::array<std::uint32_t, 2> key_counts{17, 17};
  std::array<std::uint32_t, 2> owner_indices{4, 9};

  PackedTokenMetadataView token_view() {
    return {1, tokens, positions, request_index, query_offsets, sample_rows, 3};
  }
  QwenBf16PackedKvMetadataView kv_view() {
    return {1, append_handles, token_offsets, visible_offsets,
            visible_handles, key_counts, owner_indices};
  }
};

TEST(QwenBf16PackedStepStagingLayoutTest,
     FreezesFourteenAlignedTypedSpans) {
  Metadata metadata;
  auto layout = QwenBf16PackedStepStagingLayout::Create(
      metadata.token_view(), metadata.kv_view());
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  for (const auto span : layout->copy_spans()) {
    EXPECT_EQ(span.offset_bytes % QwenBf16PackedStepStagingLayout::kAlignment,
              0U);
    EXPECT_GT(span.size_bytes, 0U);
  }
  EXPECT_EQ(layout->copy_spans().size(), 14U);
  EXPECT_EQ(layout->token_ids().size_bytes, 5 * sizeof(std::uint32_t));
  EXPECT_EQ(layout->sample_row_index().size_bytes,
            2 * sizeof(std::uint32_t));
  EXPECT_EQ(layout->total_bytes() %
                QwenBf16PackedStepStagingLayout::kAlignment,
            0U);
}

TEST(QwenBf16PackedStepStagingLayoutTest,
     MaterializesStablePerSequenceSamplingWire) {
  Metadata metadata;
  std::array<Qwen3SamplingDescriptor, 2> sampling{};
  sampling[0].mode = Qwen3SamplingMode::kStochastic;
  sampling[0].temperature = 0.75F;
  sampling[0].top_p = 0.9F;
  sampling[0].top_k = 32U;
  sampling[0].seed = 101U;
  sampling[0].sample_ordinal = 7U;
  sampling[0].top_logprobs_count = 5U;
  sampling[0].suppressed_token_count = 2U;
  sampling[0].suppressed_token_ids[0] = 151645U;
  sampling[0].suppressed_token_ids[1] = 17U;
  auto layout = QwenBf16PackedStepStagingLayout::Create(
                    metadata.token_view(), metadata.kv_view()).value();
  std::vector<std::byte> backing(layout.total_bytes(), std::byte{0xa5});
  ASSERT_TRUE(layout.materialize(metadata.token_view(), metadata.kv_view(),
                                 sampling, backing).ok());
  const auto first = read<Qwen3PackedSamplingWire>(
      backing, layout.sampling_descriptors(), 0);
  EXPECT_EQ(first.mode, 1U);
  EXPECT_FLOAT_EQ(first.temperature, 0.75F);
  EXPECT_FLOAT_EQ(first.top_p, 0.9F);
  EXPECT_EQ(first.top_k, 32U);
  EXPECT_EQ(first.seed, 101U);
  EXPECT_EQ(first.sample_ordinal, 7U);
  EXPECT_EQ(first.top_logprobs_count, 5U);
  EXPECT_EQ(first.suppressed_token_count, 2U);
  EXPECT_EQ(first.suppressed_token_ids[0], 151645U);
  EXPECT_EQ(first.suppressed_token_ids[1], 17U);
  const auto second = read<Qwen3PackedSamplingWire>(
      backing, layout.sampling_descriptors(), 1);
  EXPECT_EQ(second.mode, 0U);
  EXPECT_EQ(second.top_k, 0U);
  EXPECT_EQ(second.suppressed_token_count, 0U);
}

TEST(QwenBf16PackedStepStagingLayoutTest,
     MaterializesExactPackedBytesAndSampleSentinel) {
  Metadata metadata;
  auto layout = QwenBf16PackedStepStagingLayout::Create(
                    metadata.token_view(), metadata.kv_view()).value();
  std::vector<std::byte> backing(layout.total_bytes(), std::byte{0xa5});
  ASSERT_TRUE(layout.materialize(metadata.token_view(), metadata.kv_view(),
                                 backing).ok());
  EXPECT_EQ(read<std::uint32_t>(backing, layout.token_ids(), 2), 9U);
  EXPECT_EQ(read<std::uint64_t>(backing, layout.positions(), 1), 16U);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.request_index(), 3),
            kInvalidPackedRequestIndex);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.query_start_offsets(), 2), 3U);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.sample_row_index(), 0), 2U);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.sample_row_index(), 1),
            kInvalidPackedRequestIndex);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.sample_sequence_indices(), 0),
            1U);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.sample_sequence_indices(), 1),
            kInvalidPackedRequestIndex);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.sample_count()), 1U);
  EXPECT_EQ(read<QwenKvBlockHandle>(backing, layout.append_handles(), 1),
            (QwenKvBlockHandle{11, 1}));
  EXPECT_EQ(read<std::uint16_t>(backing, layout.token_offsets(), 0), 15U);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.visible_handle_offsets(), 2),
            4U);
  EXPECT_EQ(read<QwenKvBlockHandle>(backing, layout.visible_handles(), 2),
            (QwenKvBlockHandle{20, 1}));
  EXPECT_EQ(read<std::uint32_t>(backing, layout.key_token_counts(), 1), 17U);
  EXPECT_EQ(read<std::uint32_t>(backing, layout.owner_sequence_indices(), 1),
            9U);
}

TEST(QwenBf16PackedStepStagingLayoutTest,
     RejectsShapeDriftWithoutWritingBacking) {
  Metadata metadata;
  auto layout = QwenBf16PackedStepStagingLayout::Create(
                    metadata.token_view(), metadata.kv_view()).value();
  std::vector<std::byte> backing(layout.total_bytes(), std::byte{0xa5});
  auto kv = metadata.kv_view();
  kv.key_token_counts = kv.key_token_counts.first(1);
  EXPECT_FALSE(layout.materialize(metadata.token_view(), kv, backing).ok());
  EXPECT_EQ(backing.front(), std::byte{0xa5});
  EXPECT_EQ(backing.back(), std::byte{0xa5});
}

}  // namespace
}  // namespace pih
