#include "pih/model/qwen3_bf16_packed_batch_transaction.h"
#include "pih/model/qwen3_bf16_packed_kv_metadata.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

QwenKvBlockTable table(std::uint32_t owner, std::uint32_t generation,
                       std::uint32_t slot) {
  const std::array handles{QwenKvBlockHandle{slot, 1}};
  return QwenKvBlockTable::Create(owner, generation, 16, handles).value();
}

struct PackedFixture final {
  PackedTokenPlan plan;
  PackedTokenMetadataArena arena;
  PackedTokenMetadataView metadata;
  std::array<Qwen3SamplingDescriptor, 2> sampling;
};

PackedFixture fixture() {
  const std::array<std::uint32_t, 2> first{1, 2};
  const std::array<std::uint32_t, 1> second{3};
  const std::array inputs{
      PackedSequenceInput{11, 0, 2, 1,
                          packed_token_input_digest(first).value()},
      PackedSequenceInput{12, 0, 1, 1,
                          packed_token_input_digest(second).value()}};
  auto plan = PackedTokenPlan::Create(
      7, 1, PackedTokenPhase::kPrefill, "profile-r1", digest("resources"),
      inputs, 3, {2, 3, 3}).value();
  auto arena = PackedTokenMetadataArena::Create({2, 3}).value();
  const std::array tokens{
      PackedSequenceTokens{11, first, false},
      PackedSequenceTokens{12, second, true}};
  auto metadata = arena.materialize(plan, tokens).value();
  return {std::move(plan), std::move(arena), metadata, {}};
}

class Backend final : public QwenBf16PackedBatchBackend {
 public:
  Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const QwenKvAppendPlan> append_plans,
      std::span<const Qwen3SamplingDescriptor> sampling) override {
    ++calls;
    observed_plan_sequence = plan.plan_sequence();
    observed_real_tokens = metadata.real_token_count;
    observed_bindings = bindings.size();
    observed_kv_rows = kv_metadata.append_handles.size();
    observed_targets = {append_plans[0].target_committed_tokens,
                        append_plans[1].target_committed_tokens};
    observed_sampling.assign(sampling.begin(), sampling.end());
    if (fail) return Status::Unavailable("backend failed after launch");
    return QwenBf16PackedBatchExecutionView{sampled, {9, 10}};
  }

  std::array<std::uint32_t, 1> sampled{42};
  std::array<std::uint32_t, 2> observed_targets{};
  std::vector<Qwen3SamplingDescriptor> observed_sampling;
  std::uint64_t observed_plan_sequence = 0;
  std::uint32_t observed_real_tokens = 0;
  std::size_t observed_bindings = 0;
  std::size_t observed_kv_rows = 0;
  std::uint32_t calls = 0;
  bool fail = false;
};

TEST(QwenBf16PackedBatchTransactionTest,
     CommitsEveryKvFrontierAfterOnePackedBackendCall) {
  auto data = fixture();
  auto first = table(0, 1, 1);
  auto second = table(1, 2, 2);
  const std::array bindings{
      QwenBf16PackedKvBinding{11, &first},
      QwenBf16PackedKvBinding{12, &second}};
  auto transaction = QwenBf16PackedBatchTransaction::Create(2).value();
  auto kv_metadata = QwenBf16PackedKvMetadataArena::Create({2, 3, 2}).value();
  Backend backend;

  data.sampling[1].sample_ordinal = 7;
  auto result = transaction.execute(data.plan, data.metadata, bindings,
                                    data.sampling,
                                    kv_metadata, backend);
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_EQ(result->sampled_token_ids.size(), 1U);
  EXPECT_EQ(result->sampled_token_ids[0], 42U);
  EXPECT_EQ(result->completion_event.handle, 9U);
  EXPECT_EQ(result->completion_event.generation, 10U);
  EXPECT_EQ(backend.calls, 1U);
  EXPECT_EQ(backend.observed_plan_sequence, 1U);
  EXPECT_EQ(backend.observed_real_tokens, 3U);
  EXPECT_EQ(backend.observed_bindings, 2U);
  EXPECT_EQ(backend.observed_kv_rows, 3U);
  EXPECT_EQ(backend.observed_targets,
            (std::array<std::uint32_t, 2>{2, 1}));
  ASSERT_EQ(backend.observed_sampling.size(), 2U);
  EXPECT_EQ(backend.observed_sampling[1].sample_ordinal, 7U);
  EXPECT_EQ(first.descriptor().committed_tokens, 2U);
  EXPECT_EQ(second.descriptor().committed_tokens, 1U);
  EXPECT_EQ(transaction.state(), QwenBf16PackedBatchTransactionState::kReady);
}

TEST(QwenBf16PackedBatchTransactionTest,
     BackendFailureLeavesHostCutsUncommittedAndPoisonsTransaction) {
  auto data = fixture();
  auto first = table(0, 1, 1);
  auto second = table(1, 2, 2);
  const std::array bindings{
      QwenBf16PackedKvBinding{11, &first},
      QwenBf16PackedKvBinding{12, &second}};
  auto transaction = QwenBf16PackedBatchTransaction::Create(2).value();
  auto kv_metadata = QwenBf16PackedKvMetadataArena::Create({2, 3, 2}).value();
  Backend backend;
  backend.fail = true;

  EXPECT_FALSE(
      transaction.execute(data.plan, data.metadata, bindings, data.sampling,
                          kv_metadata,
                          backend).ok());
  EXPECT_EQ(first.descriptor().committed_tokens, 0U);
  EXPECT_EQ(second.descriptor().committed_tokens, 0U);
  EXPECT_EQ(transaction.state(),
            QwenBf16PackedBatchTransactionState::kPoisoned);
  EXPECT_FALSE(
      transaction.execute(data.plan, data.metadata, bindings, data.sampling,
                          kv_metadata,
                          backend).ok());
  EXPECT_EQ(backend.calls, 1U);
}

TEST(QwenBf16PackedBatchTransactionTest,
     ValidationFailureDoesNotCallBackendOrPoisonTransaction) {
  auto data = fixture();
  auto first = table(0, 1, 1);
  auto second = table(1, 2, 2);
  const std::array bindings{
      QwenBf16PackedKvBinding{99, &first},
      QwenBf16PackedKvBinding{12, &second}};
  auto transaction = QwenBf16PackedBatchTransaction::Create(2).value();
  auto kv_metadata = QwenBf16PackedKvMetadataArena::Create({2, 3, 2}).value();
  Backend backend;

  EXPECT_FALSE(
      transaction.execute(data.plan, data.metadata, bindings, data.sampling,
                          kv_metadata,
                          backend).ok());
  EXPECT_EQ(backend.calls, 0U);
  EXPECT_EQ(transaction.state(), QwenBf16PackedBatchTransactionState::kReady);
}

TEST(QwenBf16PackedBatchTransactionTest,
     ForgedRequestIndexIsRejectedBeforeBackendExecution) {
  auto data = fixture();
  auto first = table(0, 1, 1);
  auto second = table(1, 2, 2);
  const std::array bindings{
      QwenBf16PackedKvBinding{11, &first},
      QwenBf16PackedKvBinding{12, &second}};
  std::vector<std::uint32_t> request_index(
      data.metadata.request_index.begin(), data.metadata.request_index.end());
  request_index[0] = 1;
  const PackedTokenMetadataView forged{
      data.metadata.generation,
      data.metadata.input_token_ids,
      data.metadata.positions,
      request_index,
      data.metadata.query_start_offsets,
      data.metadata.sample_row_index,
      data.metadata.real_token_count};
  auto transaction = QwenBf16PackedBatchTransaction::Create(2).value();
  auto kv_metadata = QwenBf16PackedKvMetadataArena::Create({2, 3, 2}).value();
  Backend backend;

  EXPECT_FALSE(transaction.execute(data.plan, forged, bindings, data.sampling,
                                   kv_metadata, backend).ok());
  EXPECT_EQ(backend.calls, 0U);
  EXPECT_EQ(first.descriptor().committed_tokens, 0U);
  EXPECT_EQ(second.descriptor().committed_tokens, 0U);
}

}  // namespace
}  // namespace pih
