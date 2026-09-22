#include "pih/scheduler/controller_runtime.h"
#include "pih/scheduler/controller_packed_driver.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

class RecordingAdmissionParticipant final
    : public ControllerAdmissionParticipant {
 public:
  Status reserve(const ControllerAdmissionView& admission) override {
    calls.push_back("reserve:" +
                    std::to_string(admission.request.request_generation));
    return reserve_status;
  }
  Status publish(std::uint64_t sequence_generation) override {
    calls.push_back("publish:" + std::to_string(sequence_generation));
    return publish_status;
  }
  Status rollback(std::uint64_t request_generation) override {
    calls.push_back("rollback:" + std::to_string(request_generation));
    return rollback_status;
  }

  std::vector<std::string> calls;
  Status reserve_status = Status::Ok();
  Status publish_status = Status::Ok();
  Status rollback_status = Status::Ok();
};
Sha256Digest resource_root() {
  constexpr std::string_view value = "runtime-resources";
  return sha256(std::as_bytes(std::span(value))).value();
}
ControllerRuntime runtime(std::uint32_t event_capacity = 2) {
  return ControllerRuntime::Create(
      {7, 2, {{2, event_capacity}, {2, 4, 8}},
       {2, 2, 4, 2, 2, 100}, {2, 8}, 2, false,
       "profile-r1", resource_root()}).value();
}

ControllerRuntime runtime_with_eos(std::uint32_t eos_token_id) {
  return ControllerRuntime::Create(
      {7, 2, {{2, 4}, {2, 4, 8}}, {2, 2, 4, 2, 2, 100},
       {2, 8}, 2, false, "profile-r1", resource_root(), eos_token_id}).value();
}

class ControllerFakePackedBackend final : public QwenBf16PackedBatchBackend {
 public:
  Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan&, PackedTokenMetadataView metadata,
      QwenBf16PackedKvMetadataView,
      std::span<const QwenBf16PackedKvBinding>,
      std::span<const QwenKvAppendPlan>,
      std::span<const Qwen3SamplingDescriptor> sampling) override {
    ++calls;
    observed_sampling.assign(sampling.begin(), sampling.end());
    sampling_history.push_back(observed_sampling);
    tokens.clear();
    receipts.clear();
    if (!metadata.sample_row_index.empty()) {
      tokens.push_back(42);
      QwenBf16PackedSampleReceipt receipt{};
      receipt.token_id = 42;
      receipt.selected_logprob = -0.25F;
      receipt.rng_word = 17;
      receipt.top_logprob_count = 1;
      receipt.top_token_ids[0] = 42;
      receipt.top_logprobs[0] = -0.25F;
      receipts.push_back(receipt);
    }
    return QwenBf16PackedBatchExecutionView{
        tokens, {31, static_cast<std::uint64_t>(calls)}, receipts};
  }
  int calls = 0;
  std::vector<std::uint32_t> tokens;
  std::vector<QwenBf16PackedSampleReceipt> receipts;
  std::vector<Qwen3SamplingDescriptor> observed_sampling;
  std::vector<std::vector<Qwen3SamplingDescriptor>> sampling_history;
};

class ControllerFakeRecycler final : public QwenBf16KvRecycler {
 public:
  Status recycle(QwenKvSlotPool&, std::span<const QwenKvBlockHandle> handles,
                 QwenKvCompletionEvent event) override {
    ++calls;
    observed_handles.assign(handles.begin(), handles.end());
    observed_event = event;
    return Status::Ok();
  }
  int calls = 0;
  std::vector<QwenKvBlockHandle> observed_handles;
  QwenKvCompletionEvent observed_event{};
};
}

TEST(ControllerRuntimeTest, ConvertsAdmissionIntoSequenceAndImmutableEvent) {
  auto controller = runtime();
  const std::array<std::uint32_t, 3> tokens{1, 2, 3};
  ASSERT_TRUE(controller.submit_admit(11, tokens, 2).ok());
  EXPECT_EQ(controller.step().value(), ControllerRuntimeStep::kAdmitted);
  EXPECT_EQ(controller.active_sequence_count(), 1U);
  EXPECT_EQ(controller.queued_command_count(), 0U);
  auto event = controller.try_take_event().value();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->kind, ControllerOutputEventKind::kAdmitted);
  EXPECT_EQ(event->epoch, 7U);
  EXPECT_EQ(event->request_generation, 11U);
}

TEST(ControllerRuntimeTest, PublishesPersistentAdmissionBeforeAdmittedEvent) {
  auto controller = runtime(2);
  RecordingAdmissionParticipant participant;
  const std::array<std::uint32_t, 2> prompt{1, 2};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 3).ok());
  ASSERT_EQ(controller.step(10, participant).value(),
            ControllerRuntimeStep::kAdmitted);
  EXPECT_EQ(participant.calls,
            (std::vector<std::string>{"reserve:11", "publish:11"}));
  auto admitted = controller.try_take_event().value();
  ASSERT_TRUE(admitted.has_value());
  EXPECT_EQ(admitted->kind, ControllerOutputEventKind::kAdmitted);
}

TEST(ControllerRuntimeTest, ReserveFailureReleasesIngressWithoutPublication) {
  auto controller = runtime(2);
  RecordingAdmissionParticipant participant;
  participant.reserve_status =
      Status::ResourceExhausted("persistent credits unavailable");
  const std::array<std::uint32_t, 1> prompt{1};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 1).ok());
  auto result = controller.step(10, participant);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, ControllerRuntimeStep::kAdmissionRejected);
  EXPECT_EQ(participant.calls,
            (std::vector<std::string>{"reserve:11"}));
  EXPECT_EQ(controller.active_sequence_count(), 0U);
  auto rejected = controller.try_take_event().value();
  ASSERT_TRUE(rejected.has_value());
  EXPECT_EQ(rejected->kind, ControllerOutputEventKind::kFailed);
  EXPECT_EQ(rejected->request_generation, 11U);
  EXPECT_EQ(controller.state(), ControllerRuntimeState::kReady);
}

TEST(ControllerRuntimeTest, PersistentLedgerDivergenceFailsControllerEpoch) {
  auto controller = runtime(2);
  RecordingAdmissionParticipant participant;
  participant.reserve_status =
      Status::FailedPrecondition("persistent generation drifted");
  const std::array<std::uint32_t, 1> prompt{1};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 1).ok());
  auto result = controller.step(10, participant);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(controller.state(), ControllerRuntimeState::kFailed);
}

TEST(ControllerRuntimeTest, OutputBackpressureDoesNotConsumeNextCommand) {
  auto controller = runtime(1);
  const std::array<std::uint32_t, 1> token{1};
  ASSERT_TRUE(controller.submit_admit(1, token, 1).ok());
  ASSERT_EQ(controller.step().value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.submit_admit(2, token, 1).ok());
  EXPECT_EQ(controller.step().value(),
            ControllerRuntimeStep::kOutputBackpressured);
  EXPECT_EQ(controller.queued_command_count(), 1U);
  EXPECT_EQ(controller.active_sequence_count(), 1U);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  EXPECT_EQ(controller.step().value(), ControllerRuntimeStep::kAdmitted);
  EXPECT_EQ(controller.active_sequence_count(), 2U);
}

TEST(ControllerRuntimeTest, EmptyStepIsStableAndLimitsMustMatch) {
  auto controller = runtime();
  EXPECT_EQ(controller.step().value(), ControllerRuntimeStep::kIdle);
  EXPECT_EQ(controller.state(), ControllerRuntimeState::kReady);
  EXPECT_FALSE(ControllerRuntime::Create(
      {1, 1, {{1, 1}, {2, 4, 8}},
       {1, 1, 2, 1, 1, 1}, {1, 2}, 1, false, "p",
       resource_root()}).ok());
  EXPECT_FALSE(ControllerRuntime::Create(
      {1, 1, {{1, 1}, {1, 4, 8}},
       {1, 1, 2, 1, 1, 1}, {1, 2}, 1, false, "p",
       Sha256Digest{}}).ok());
  EXPECT_FALSE(controller.step(-1).ok());
}

TEST(ControllerRuntimeTest, PreparesFairPackedPrefillFromActiveSequences) {
  auto controller = runtime();
  const std::array<std::uint32_t, 3> first{1, 2, 3};
  const std::array<std::uint32_t, 1> second{9};
  ASSERT_TRUE(controller.submit_admit(11, first, 2).ok());
  ASSERT_TRUE(controller.submit_admit(12, second, 2).ok());
  ASSERT_EQ(controller.step(10).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_EQ(controller.step(20).value(), ControllerRuntimeStep::kAdmitted);
  auto prepared = controller.prepare_next_plan(30);
  ASSERT_TRUE(prepared.ok()) << prepared.status().message();
  ASSERT_TRUE(prepared->has_value());
  const auto& view = **prepared;
  ASSERT_NE(view.plan, nullptr);
  EXPECT_EQ(view.plan->phase(), PackedTokenPhase::kPrefill);
  EXPECT_EQ(view.plan->ordered_sequence_generations(),
            (std::vector<std::uint64_t>{11, 12}));
  EXPECT_EQ(view.plan->real_token_counts(),
            (std::vector<std::uint32_t>{2, 1}));
  EXPECT_EQ(view.plan->packed_offsets(),
            (std::vector<std::uint32_t>{0, 2, 3}));
  ASSERT_EQ(view.metadata.input_token_ids.size(), 3U);
  EXPECT_EQ(view.metadata.input_token_ids[0], 1U);
  EXPECT_EQ(view.metadata.input_token_ids[1], 2U);
  EXPECT_EQ(view.metadata.input_token_ids[2], 9U);
  ASSERT_EQ(view.metadata.sample_row_index.size(), 1U);
  EXPECT_EQ(view.metadata.sample_row_index[0], 2U);
  EXPECT_EQ(view.selected_slot_indices.size(), 2U);
  EXPECT_FALSE(controller.prepare_next_plan(31).ok());
}

TEST(ControllerRuntimeTest, ControllerTimeCannotMoveBackwards) {
  auto controller = runtime();
  const std::array<std::uint32_t, 1> token{1};
  ASSERT_TRUE(controller.submit_admit(1, token, 1).ok());
  ASSERT_EQ(controller.step(10).value(), ControllerRuntimeStep::kAdmitted);
  EXPECT_FALSE(controller.prepare_next_plan(9).ok());
  EXPECT_TRUE(controller.prepare_next_plan(10).ok());
}

TEST(ControllerRuntimeTest, CommitsCompletesAndRequeuesSuccessivePrefillChunks) {
  auto controller = runtime();
  const std::array<std::uint32_t, 3> prompt{1, 2, 3};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 2).ok());
  ASSERT_EQ(controller.step(10).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());

  auto first = controller.prepare_next_plan(20).value().value();
  const auto first_sequence = first.plan->plan_sequence();
  const auto first_digest = first.plan_digest;
  ASSERT_TRUE(controller.commit_current_plan(first_sequence, first_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      first_sequence, first_digest).ok());
  const std::array first_results{ControllerBackendSequenceResult{
      {first_sequence, 11, 2, 0, 2, PackedTokenPhase::kDecode, false}, {}}};
  ASSERT_TRUE(controller.complete_current_plan(first_results, 30).ok());
  EXPECT_FALSE(controller.try_take_event().value().has_value());
  ASSERT_TRUE(controller.acknowledge_output_plan(first_sequence).ok());

  auto second = controller.prepare_next_plan(31).value().value();
  EXPECT_EQ(second.plan->plan_sequence(), first_sequence + 1);
  ASSERT_EQ(second.metadata.input_token_ids.size(), 1U);
  EXPECT_EQ(second.metadata.input_token_ids[0], 3U);
  ASSERT_EQ(second.metadata.sample_row_index.size(), 1U);
  const auto second_sequence = second.plan->plan_sequence();
  const auto second_digest = second.plan_digest;
  ASSERT_TRUE(controller.commit_current_plan(second_sequence, second_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      second_sequence, second_digest).ok());
  const std::array<std::uint32_t, 1> accepted{42};
  const std::array second_results{ControllerBackendSequenceResult{
      {second_sequence, 11, 3, 1, 3, PackedTokenPhase::kDecode, false},
      accepted}};
  ASSERT_TRUE(controller.complete_current_plan(second_results, 40).ok());
  auto token = controller.try_take_event().value();
  ASSERT_TRUE(token.has_value());
  EXPECT_EQ(token->kind, ControllerOutputEventKind::kTokenCommitted);
  EXPECT_EQ(token->token_ordinal, 1U);
  EXPECT_EQ(token->token_id, 42U);
  ASSERT_TRUE(controller.acknowledge_output_plan(second_sequence).ok());

  auto decode = controller.prepare_next_plan(41).value().value();
  EXPECT_EQ(decode.plan->phase(), PackedTokenPhase::kDecode);
  ASSERT_EQ(decode.metadata.input_token_ids.size(), 1U);
  EXPECT_EQ(decode.metadata.input_token_ids[0], 42U);
  ASSERT_EQ(decode.metadata.sample_row_index.size(), 1U);
  EXPECT_EQ(decode.metadata.sample_row_index[0], 0U);
  const auto decode_sequence = decode.plan->plan_sequence();
  ASSERT_TRUE(
      controller.commit_current_plan(decode_sequence, decode.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      decode_sequence, decode.plan_digest).ok());
  const std::array<std::uint32_t, 1> final_token{43};
  const std::array decode_results{ControllerBackendSequenceResult{
      {decode_sequence, 11, 4, 2, 4, PackedTokenPhase::kDecode, true},
      final_token}};
  ASSERT_TRUE(controller.complete_current_plan(decode_results, 42).ok());
  auto final = controller.try_take_event().value();
  ASSERT_TRUE(final.has_value());
  EXPECT_EQ(final->kind, ControllerOutputEventKind::kTokenCommitted);
  EXPECT_EQ(final->token_ordinal, 2U);
  EXPECT_EQ(final->token_id, 43U);
  auto draining = controller.try_take_event().value();
  ASSERT_TRUE(draining.has_value());
  EXPECT_EQ(draining->finish_reason, ControllerFinishReason::kLength);
  EXPECT_EQ(draining->kind, ControllerOutputEventKind::kDraining);
  ASSERT_TRUE(controller.acknowledge_output_plan(decode_sequence).ok());
  EXPECT_FALSE(controller.prepare_next_plan(43).value().has_value());
}

TEST(ControllerRuntimeTest,
     SampledCompletionDerivesZeroSampleChunkAndGenerationLimit) {
  auto controller = runtime(4);
  const std::array<std::uint32_t, 3> prompt{1, 2, 3};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 1).ok());
  ASSERT_EQ(controller.step(10).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());

  auto first = controller.prepare_next_plan(20).value().value();
  ASSERT_TRUE(controller.commit_current_plan(
      first.plan->plan_sequence(), first.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      first.plan->plan_sequence(), first.plan_digest).ok());
  EXPECT_TRUE(controller.complete_current_sampled_tokens({}, 21).ok());
  EXPECT_FALSE(controller.try_take_event().value().has_value());
  ASSERT_TRUE(controller.acknowledge_output_plan(
      first.plan->plan_sequence()).ok());

  auto final = controller.prepare_next_plan(22).value().value();
  ASSERT_TRUE(controller.commit_current_plan(
      final.plan->plan_sequence(), final.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      final.plan->plan_sequence(), final.plan_digest).ok());
  const std::array<std::uint32_t, 1> sampled{42};
  EXPECT_TRUE(controller.complete_current_sampled_tokens(sampled, 23).ok());
  auto token = controller.try_take_event().value();
  ASSERT_TRUE(token.has_value());
  EXPECT_EQ(token->kind, ControllerOutputEventKind::kTokenCommitted);
  EXPECT_EQ(token->token_id, 42U);
  auto draining = controller.try_take_event().value();
  ASSERT_TRUE(draining.has_value());
  EXPECT_EQ(draining->kind, ControllerOutputEventKind::kDraining);
}

TEST(ControllerRuntimeTest, SampledEosTerminatesBeforeGenerationLimit) {
  auto controller = runtime_with_eos(42);
  const std::array<std::uint32_t, 1> prompt{1};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 4).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  auto plan = controller.prepare_next_plan(2).value().value();
  ASSERT_TRUE(controller.commit_current_plan(
      plan.plan->plan_sequence(), plan.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      plan.plan->plan_sequence(), plan.plan_digest).ok());
  const std::array<std::uint32_t, 1> sampled{42};
  ASSERT_TRUE(controller.complete_current_sampled_tokens(sampled, 3).ok());
  auto token = controller.try_take_event().value();
  ASSERT_TRUE(token.has_value());
  EXPECT_EQ(token->kind, ControllerOutputEventKind::kTokenCommitted);
  auto draining = controller.try_take_event().value();
  ASSERT_TRUE(draining.has_value());
  EXPECT_EQ(draining->kind, ControllerOutputEventKind::kDraining);
  EXPECT_EQ(draining->finish_reason, ControllerFinishReason::kStop);
  EXPECT_FALSE(controller.prepare_next_plan(4).value().has_value());
}

TEST(ControllerRuntimeTest, SampledRequestLocalStopTokenTerminates) {
  auto controller = runtime(4);
  const std::array<std::uint32_t, 1> prompt{1};
  ControllerRequestSampling sampling;
  sampling.stop_token_ids[0] = 77;
  sampling.stop_token_ids[1] = 88;
  sampling.stop_token_count = 2;
  ASSERT_TRUE(controller.submit_admit(11, prompt, 4, sampling).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  auto plan = controller.prepare_next_plan(2).value().value();
  ASSERT_TRUE(controller.commit_current_plan(
      plan.plan->plan_sequence(), plan.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      plan.plan->plan_sequence(), plan.plan_digest).ok());
  const std::array<std::uint32_t, 1> sampled{88};
  ASSERT_TRUE(controller.complete_current_sampled_tokens(sampled, 3).ok());
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  auto draining = controller.try_take_event().value();
  ASSERT_TRUE(draining.has_value());
  EXPECT_EQ(draining->kind, ControllerOutputEventKind::kDraining);
  EXPECT_EQ(draining->finish_reason, ControllerFinishReason::kStop);
  EXPECT_FALSE(controller.prepare_next_plan(4).value().has_value());
}

TEST(ControllerPackedDriverTest,
     DrivesChunkedPrefillThroughKvCommitAndControllerPublication) {
  auto controller = runtime(2);
  ControllerFakePackedBackend backend;
  auto driver = ControllerPackedDriver::Create(
      controller, backend, 2, {2, 8, 2});
  ASSERT_TRUE(driver.ok()) << driver.status().message();
  const std::array<QwenKvBlockHandle, 1> handles{{0, 1}};
  auto table_result = QwenKvBlockTable::Create(0, 11, 16, handles);
  ASSERT_TRUE(table_result.ok()) << table_result.status().message();
  auto table = std::move(*table_result);
  ASSERT_TRUE(driver->bind_sequence(11, table, {30, 1}).ok());
  const std::array<std::uint32_t, 3> prompt{1, 2, 3};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 1).ok());
  ASSERT_EQ(controller.step(10).value(), ControllerRuntimeStep::kAdmitted);

  auto intermediate = driver->execute_next(20);
  ASSERT_TRUE(intermediate.ok()) << intermediate.status().message();
  EXPECT_EQ(*intermediate, ControllerPackedDriverStep::kCompleted);
  EXPECT_EQ(table.descriptor().committed_tokens, 2U);
  ASSERT_TRUE(controller.acknowledge_output_plan(1).ok());

  auto final = driver->execute_next(21);
  ASSERT_TRUE(final.ok()) << final.status().message();
  EXPECT_EQ(*final, ControllerPackedDriverStep::kOutputBackpressured);
  EXPECT_EQ(table.descriptor().committed_tokens, 3U);
  EXPECT_EQ(backend.calls, 2);
  EXPECT_EQ(driver->state(), ControllerPackedDriverState::kCompletionPending);
  auto admitted = controller.try_take_event().value();
  ASSERT_TRUE(admitted.has_value());
  EXPECT_EQ(admitted->kind, ControllerOutputEventKind::kAdmitted);
  auto retried = driver->execute_next(22);
  ASSERT_TRUE(retried.ok()) << retried.status().message();
  EXPECT_EQ(*retried, ControllerPackedDriverStep::kCompleted);
  ASSERT_TRUE(controller.acknowledge_output_plan(2).ok());
  EXPECT_EQ(backend.calls, 2);
  auto pool = QwenKvSlotPool::Create(
                  1, QwenKvSlotPool::kSlotPayloadBytes,
                  sizeof(QwenKvSlotState))
                  .value();
  ControllerFakeRecycler recycler;
  auto blocked_drain = driver->drain_sequence(11, pool, recycler);
  EXPECT_EQ(blocked_drain.code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(recycler.calls, 1);
  auto token = controller.try_take_event().value();
  ASSERT_TRUE(token.has_value());
  EXPECT_EQ(token->kind, ControllerOutputEventKind::kTokenCommitted);
  EXPECT_EQ(token->token_id, 42U);
  auto receipt = driver->try_take_sampling_receipt();
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  ASSERT_TRUE(receipt->has_value());
  EXPECT_EQ((*receipt)->request_generation, 11U);
  EXPECT_EQ((*receipt)->token_ordinal, 1U);
  EXPECT_EQ((*receipt)->sample.token_id, 42U);
  EXPECT_FLOAT_EQ((*receipt)->sample.selected_logprob, -0.25F);
  EXPECT_FALSE(driver->try_take_sampling_receipt().value().has_value());
  ASSERT_TRUE(driver->drain_sequence(11, pool, recycler).ok());
  EXPECT_EQ(recycler.calls, 1);
  auto draining = controller.try_take_event().value();
  ASSERT_TRUE(draining.has_value());
  EXPECT_EQ(draining->kind, ControllerOutputEventKind::kDraining);
  EXPECT_EQ(recycler.observed_handles, (std::vector<QwenKvBlockHandle>{{0, 1}}));
  EXPECT_EQ(recycler.observed_event.handle, 31U);
  EXPECT_EQ(recycler.observed_event.generation, 2U);
  auto completed = controller.try_take_event().value();
  ASSERT_TRUE(completed.has_value());
  EXPECT_EQ(completed->kind, ControllerOutputEventKind::kCompleted);
  EXPECT_EQ(controller.active_sequence_count(), 0U);
  EXPECT_EQ(driver->state(), ControllerPackedDriverState::kReady);
}

TEST(ControllerPackedDriverTest,
     ConvertsSamplingAndSuppressesEosUntilMinimumTokensAreAccepted) {
  constexpr std::uint32_t eos = 151645;
  auto controller = runtime_with_eos(eos);
  ControllerFakePackedBackend backend;
  auto driver = ControllerPackedDriver::Create(controller, backend, 2,
                                                {2, 8, 2}).value();
  const std::array<QwenKvBlockHandle, 1> handles{{0, 1}};
  auto table = QwenKvBlockTable::Create(0, 11, 16, handles).value();
  ASSERT_TRUE(driver.bind_sequence(11, table, {30, 1}).ok());
  const std::array<std::uint32_t, 1> prompt{1};
  const ControllerRequestSampling sampling{
      ControllerSamplingMode::kStochastic, 0.75F, 0.9F, 32U, 101U, 0U,
      1U, true, 5U};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 2, sampling).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());

  ASSERT_EQ(driver.execute_next(2).value(),
            ControllerPackedDriverStep::kCompleted);
  ASSERT_EQ(backend.sampling_history.size(), 1U);
  ASSERT_EQ(backend.sampling_history[0].size(), 1U);
  const auto& first = backend.sampling_history[0][0];
  EXPECT_EQ(first.mode, Qwen3SamplingMode::kStochastic);
  EXPECT_FLOAT_EQ(first.temperature, 0.75F);
  EXPECT_EQ(first.seed, 101U);
  EXPECT_EQ(first.sample_ordinal, 0U);
  EXPECT_EQ(first.top_logprobs_count, 5U);
  EXPECT_EQ(first.suppressed_token_count, 1U);
  EXPECT_EQ(first.suppressed_token_ids[0], eos);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  ASSERT_TRUE(controller.acknowledge_output_plan(1).ok());

  ASSERT_EQ(driver.execute_next(3).value(),
            ControllerPackedDriverStep::kCompleted);
  ASSERT_EQ(backend.sampling_history.size(), 2U);
  const auto& second = backend.sampling_history[1][0];
  EXPECT_EQ(second.sample_ordinal, 1U);
  EXPECT_EQ(second.suppressed_token_count, 0U);
}

TEST(ControllerRuntimeTest,
     CancelAtSchedulingBoundaryDrainsAndReleasesRequestCredit) {
  auto controller = runtime(4);
  const std::array<std::uint32_t, 2> prompt{1, 2};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 2).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  ASSERT_TRUE(controller.submit_cancel(11).ok());
  ASSERT_EQ(controller.step(2).value(),
            ControllerRuntimeStep::kCancelRequested);
  auto draining = controller.try_take_event().value();
  ASSERT_TRUE(draining.has_value());
  EXPECT_EQ(draining->kind, ControllerOutputEventKind::kDraining);
  EXPECT_EQ(controller.active_sequence_count(), 1U);
  ASSERT_TRUE(controller.finalize_draining(11).ok());
  auto cancelled = controller.try_take_event().value();
  ASSERT_TRUE(cancelled.has_value());
  EXPECT_EQ(cancelled->kind, ControllerOutputEventKind::kCancelled);
  EXPECT_EQ(controller.active_sequence_count(), 0U);

  ASSERT_TRUE(controller.submit_admit(12, prompt, 1).ok());
  EXPECT_EQ(controller.step(3).value(), ControllerRuntimeStep::kAdmitted);
  EXPECT_EQ(controller.active_sequence_count(), 1U);
}

TEST(ControllerRuntimeTest, GenerationLimitRequiresTerminalCompletion) {
  auto controller = runtime();
  const std::array<std::uint32_t, 1> prompt{1};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 1).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  auto plan = controller.prepare_next_plan(2).value().value();
  ASSERT_TRUE(
      controller.commit_current_plan(plan.plan->plan_sequence(), plan.plan_digest)
          .ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      plan.plan->plan_sequence(), plan.plan_digest).ok());
  const std::array<std::uint32_t, 1> accepted{42};
  std::array results{ControllerBackendSequenceResult{
      {plan.plan->plan_sequence(), 11, 1, 1, 2,
       PackedTokenPhase::kDecode, false},
      accepted}};
  EXPECT_FALSE(controller.complete_current_plan(results, 3).ok());
  results[0].completion.terminal = true;
  EXPECT_TRUE(controller.complete_current_plan(results, 3).ok());
}

TEST(ControllerRuntimeTest, InvalidBatchCompletionPublishesNothing) {
  auto controller = runtime();
  const std::array<std::uint32_t, 1> first{1};
  const std::array<std::uint32_t, 1> second{2};
  ASSERT_TRUE(controller.submit_admit(11, first, 2).ok());
  ASSERT_TRUE(controller.submit_admit(12, second, 2).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_EQ(controller.step(2).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  auto plan = controller.prepare_next_plan(3).value().value();
  const auto plan_sequence = plan.plan->plan_sequence();
  ASSERT_TRUE(controller.commit_current_plan(plan_sequence, plan.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      plan_sequence, plan.plan_digest).ok());
  const std::array<std::uint32_t, 1> token_a{7};
  const std::array<std::uint32_t, 1> token_b{8};
  std::array results{
      ControllerBackendSequenceResult{
          {plan_sequence, 11, 1, 1, 2, PackedTokenPhase::kDecode, false},
          std::span<const std::uint32_t>(token_a)},
      ControllerBackendSequenceResult{
          {plan_sequence, 99, 1, 1, 2, PackedTokenPhase::kDecode, false},
          std::span<const std::uint32_t>(token_b)}};
  EXPECT_FALSE(controller.complete_current_plan(results, 4).ok());
  EXPECT_FALSE(controller.try_take_event().value().has_value());
  results[1].completion.sequence_generation = 12;
  EXPECT_TRUE(controller.complete_current_plan(results, 4).ok());
  EXPECT_TRUE(controller.try_take_event().value().has_value());
  EXPECT_TRUE(controller.try_take_event().value().has_value());
}

TEST(ControllerRuntimeTest, CompletionWaitsForOutputCreditWithoutStateAdvance) {
  auto controller = runtime(1);
  const std::array<std::uint32_t, 1> prompt{1};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 2).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  auto plan = controller.prepare_next_plan(2).value().value();
  EXPECT_EQ(controller.available_output_burst_credits(), 0U);
  const auto plan_sequence = plan.plan->plan_sequence();
  ASSERT_TRUE(controller.commit_current_plan(plan_sequence, plan.plan_digest).ok());
  ASSERT_TRUE(controller.mark_current_plan_in_flight(
      plan_sequence, plan.plan_digest).ok());
  const std::array<std::uint32_t, 1> accepted{42};
  const std::array results{ControllerBackendSequenceResult{
      {plan_sequence, 11, 1, 1, 2, PackedTokenPhase::kDecode, false},
      accepted}};
  EXPECT_FALSE(controller.complete_current_plan(results, 3).ok());
  EXPECT_EQ(controller.available_output_burst_credits(), 0U);
  EXPECT_FALSE(controller.prepare_next_plan(3).ok());
  auto admitted = controller.try_take_event().value();
  ASSERT_TRUE(admitted.has_value());
  EXPECT_EQ(admitted->kind, ControllerOutputEventKind::kAdmitted);
  EXPECT_TRUE(controller.complete_current_plan(results, 3).ok());
  EXPECT_EQ(controller.available_output_burst_credits(), 0U);
  EXPECT_FALSE(controller.prepare_next_plan(4).ok());
  auto token = controller.try_take_event().value();
  ASSERT_TRUE(token.has_value());
  EXPECT_EQ(token->token_id, 42U);
  EXPECT_EQ(token->plan_sequence, plan_sequence);
  EXPECT_EQ(token->plan_event_index, 0U);
  EXPECT_EQ(token->plan_event_count, 1U);
  EXPECT_FALSE(controller.acknowledge_output_plan(plan_sequence + 1).ok());
  EXPECT_TRUE(controller.acknowledge_output_plan(plan_sequence).ok());
  EXPECT_EQ(controller.available_output_burst_credits(), 1U);
  EXPECT_FALSE(controller.acknowledge_output_plan(plan_sequence).ok());
  EXPECT_TRUE(controller.prepare_next_plan(4).value().has_value());
}

TEST(ControllerRuntimeTest, PreparedPlanAbortReturnsOutputBurstCredit) {
  auto controller = runtime();
  const std::array<std::uint32_t, 1> prompt{1};
  ASSERT_TRUE(controller.submit_admit(11, prompt, 2).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  auto plan = controller.prepare_next_plan(2).value().value();
  EXPECT_EQ(controller.available_output_burst_credits(), 0U);
  ASSERT_TRUE(controller.abort_current_plan(
      plan.plan->plan_sequence(), plan.plan_digest).ok());
  EXPECT_EQ(controller.available_output_burst_credits(), 1U);
  EXPECT_TRUE(controller.prepare_next_plan(3).value().has_value());
}

TEST(ControllerRuntimeTest, PreparedPlanProjectsSamplingInSelectedSequenceOrder) {
  auto controller = runtime();
  const std::array<std::uint32_t, 1> first{1};
  const std::array<std::uint32_t, 1> second{2};
  const ControllerRequestSampling first_sampling{
      ControllerSamplingMode::kStochastic, 0.75F, 0.9F, 32U, 101U, 0U,
      1U, true, 5U};
  const ControllerRequestSampling second_sampling{
      ControllerSamplingMode::kStochastic, 1.25F, 0.8F, 16U, 202U, 0U,
      0U, false, 0U};
  ASSERT_TRUE(controller.submit_admit(11, first, 2, first_sampling).ok());
  ASSERT_TRUE(controller.submit_admit(12, second, 2, second_sampling).ok());
  ASSERT_EQ(controller.step(1).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_EQ(controller.step(2).value(), ControllerRuntimeStep::kAdmitted);
  ASSERT_TRUE(controller.try_take_event().value().has_value());
  ASSERT_TRUE(controller.try_take_event().value().has_value());

  auto plan = controller.prepare_next_plan(3).value().value();
  ASSERT_EQ(plan.selected_sampling.size(), 2U);
  EXPECT_EQ(plan.selected_sampling[0], first_sampling);
  EXPECT_EQ(plan.selected_sampling[1], second_sampling);
}

}  // namespace pih
