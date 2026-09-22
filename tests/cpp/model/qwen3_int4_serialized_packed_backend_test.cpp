#include "pih/model/qwen3_int4_serialized_packed_backend.h"

#include <array>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_packed_kv_metadata.h"

namespace pih {
namespace {

Sha256Digest digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

class SequenceBackend final : public QwenBf16SequenceBackend,
                              public QwenBf16CompletionIdentityProvider {
 public:
  Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable&,
      const QwenKvAppendPlan&) override {
    calls.push_back({tokens.size(), first_position});
    return static_cast<std::int64_t>(40 + calls.size());
  }
  Result<QwenKvCompletionEvent> last_completion_event() const override {
    return QwenKvCompletionEvent{9, 10};
  }
  std::vector<std::array<std::uint64_t, 2>> calls;
};

QwenKvBlockTable table(std::uint32_t owner, std::uint32_t generation,
                       std::uint32_t slot) {
  const std::array handles{QwenKvBlockHandle{slot, 1}};
  return QwenKvBlockTable::Create(owner, generation, 16, handles).value();
}

TEST(QwenInt4SerializedPackedBackendTest,
     ExecutesSequencesInCanonicalOrderAndPublishesSampleRows) {
  SequenceBackend sequence;
  auto backend = QwenInt4SerializedPackedBackend::Create(sequence, sequence, 2)
                     .value();
  const std::array<std::uint32_t, 2> first{1, 2};
  const std::array<std::uint32_t, 1> second{3};
  const std::array inputs{
      PackedSequenceInput{11, 5, 2, 1,
                          packed_token_input_digest(first).value()},
      PackedSequenceInput{12, 8, 1, 1,
                          packed_token_input_digest(second).value()}};
  auto plan = PackedTokenPlan::Create(
      7, 1, PackedTokenPhase::kPrefill, "r1", digest("resources"),
      inputs, 3, {2, 3, 3}).value();
  auto arena = PackedTokenMetadataArena::Create({2, 3}).value();
  const std::array payloads{PackedSequenceTokens{11, first, false},
                            PackedSequenceTokens{12, second, true}};
  auto metadata = arena.materialize(plan, payloads).value();
  auto first_table = table(0, 1, 1);
  auto second_table = table(1, 2, 2);
  const std::array bindings{QwenBf16PackedKvBinding{11, &first_table},
                            QwenBf16PackedKvBinding{12, &second_table}};
  const std::array appends{first_table.prepare_append(7).value(),
                           second_table.prepare_append(9).value()};
  const std::array<Qwen3SamplingDescriptor, 2> sampling{};
  QwenBf16PackedKvMetadataView unused{};

  auto result = backend.execute_packed(plan, metadata, unused, bindings,
                                       appends, sampling);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(sequence.calls,
            (std::vector<std::array<std::uint64_t, 2>>{{2, 5}, {1, 8}}));
  ASSERT_EQ(result->sampled_token_ids.size(), 1U);
  EXPECT_EQ(result->sampled_token_ids[0], 42U);
  ASSERT_EQ(result->sampling_receipts.size(), 1U);
  EXPECT_EQ(result->sampling_receipts[0].token_id, 42U);
  EXPECT_EQ(result->completion_event.handle, 9U);
  EXPECT_EQ(result->completion_event.generation, 10U);
  EXPECT_EQ(backend.state(), QwenInt4SerializedPackedBackendState::kReady);
}

TEST(QwenInt4SerializedPackedBackendTest,
     RejectsStochasticSamplingWithoutExecutingAnySequence) {
  SequenceBackend sequence;
  auto backend = QwenInt4SerializedPackedBackend::Create(sequence, sequence, 1)
                     .value();
  const std::array<std::uint32_t, 1> tokens{1};
  const std::array inputs{PackedSequenceInput{
      11, 0, 1, 1, packed_token_input_digest(tokens).value()}};
  auto plan = PackedTokenPlan::Create(
      7, 1, PackedTokenPhase::kDecode, "r1", digest("resources"),
      inputs, 1, {1, 1, 1}).value();
  auto arena = PackedTokenMetadataArena::Create({1, 1}).value();
  const std::array payloads{PackedSequenceTokens{11, tokens, true}};
  auto metadata = arena.materialize(plan, payloads).value();
  auto block_table = table(0, 1, 1);
  const std::array bindings{QwenBf16PackedKvBinding{11, &block_table}};
  const std::array appends{block_table.prepare_append(1).value()};
  std::array<Qwen3SamplingDescriptor, 1> sampling{};
  sampling[0].mode = Qwen3SamplingMode::kStochastic;
  sampling[0].temperature = 1.0F;

  auto result = backend.execute_packed(plan, metadata, {}, bindings, appends,
                                       sampling);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(sequence.calls.empty());
  EXPECT_EQ(backend.state(), QwenInt4SerializedPackedBackendState::kReady);
}

}  // namespace
}  // namespace pih
