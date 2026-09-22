#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_observation_pipeline.h"

namespace pih {
namespace {

class CopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 41; }
  Status copy(CudaCopyKind, std::uintptr_t, std::uintptr_t, std::uint64_t,
              DriverStreamHandle) override {
    ++calls;
    return Status::Ok();
  }
  std::size_t calls = 0;
};

class EventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle event, DriverStreamHandle stream) override {
    ++record_calls;
    last_event = event;
    last_stream = stream;
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    ++query_calls;
    const auto current = result;
    if (on_query) on_query();
    return current;
  }
  CudaEventQueryResult result = CudaEventQueryResult::kNotReady;
  std::size_t record_calls = 0;
  std::size_t query_calls = 0;
  DriverEventHandle last_event = 0;
  DriverStreamHandle last_stream = 0;
  std::function<void()> on_query;
};

class Evidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    ++calls;
    return value;
  }
  CompletionPublicationEvidence value{true, 0, false};
  std::size_t calls = 0;
};

class Clock final : public QwenBf16MonotonicClock {
 public:
  Result<std::uint64_t> now_ns() override {
    if (index >= values.size()) return Status::Internal("clock exhausted");
    return values[index++];
  }
  std::vector<std::uint64_t> values;
  std::size_t index = 0;
};

class Waiter final : public QwenBf16PollWaiter {
 public:
  Status wait() override {
    ++calls;
    return result;
  }
  Status result = Status::Ok();
  std::size_t calls = 0;
};

CudaCopyEndpoint endpoint(std::uintptr_t base, std::uint64_t bytes,
                          std::uint64_t owner, CudaCopyMemoryType type,
                          std::int32_t locality) {
  return {base, bytes, 0, owner, 1, type, 0, locality};
}

QwenKvSemanticObservationPlan kv_plan() {
  const std::array<QwenKvBlockHandle, 1> handles{{{2, 11}}};
  auto table = QwenKvBlockTable::Create(4, 7, 1, handles).value();
  const auto append = table.prepare_append(1).value();
  EXPECT_TRUE(table.commit_append(append).ok());
  std::vector<QwenKvSlotState> states(
      4, {1, QwenKvSlotPool::kNoOwner, 0,
          QwenKvSlotLifecycle::kFreeClean, 0, 0});
  states[2] = {11, 4, 1, QwenKvSlotLifecycle::kOwned, 0, 0};
  return QwenKvSemanticObservationPlan::Create(
             table, states, 4 * QwenKvSlotPool::kSlotPayloadBytes)
      .value();
}

QwenSemanticObservationPipeline pipeline() {
  const auto kv = kv_plan();
  auto transfer = QwenSemanticObservationTransfer::Create(
      endpoint(0x20000000, 1 << 20, 1, CudaCopyMemoryType::kDevice, 0),
      endpoint(0x30000000, 1 << 20, 2,
               CudaCopyMemoryType::kRegisteredPinnedHost, 1),
      endpoint(0x40000000, 4 * QwenKvSlotPool::kSlotPayloadBytes, 3,
               CudaCopyMemoryType::kDevice, 0),
      endpoint(0x50000000, kv.payload_bytes(), 4,
               CudaCopyMemoryType::kRegisteredPinnedHost, 1),
      kv, {100, 41, 43, 47});
  return QwenSemanticObservationPipeline::Create(
             std::move(*transfer), 53, 1, 0, 61)
      .value();
}

TEST(QwenSemanticObservationPipelineTest,
     PublishesOnlyAfterAuthorizedCompletion) {
  auto value = pipeline();
  CopyDriver copies;
  EventDriver events;
  Evidence evidence;
  ASSERT_TRUE(value.submit(copies, events, 100, 200).ok());
  EXPECT_EQ(copies.calls, 57);
  EXPECT_EQ(events.record_calls, 1);
  EXPECT_EQ(events.last_event, 53);
  EXPECT_EQ(events.last_stream, 43);
  EXPECT_EQ(value.poll(events, evidence).code(), StatusCode::kUnavailable);
  EXPECT_EQ(evidence.calls, 0);
  EXPECT_EQ(value.state(), QwenSemanticObservationPipelineState::kRecorded);

  events.result = CudaEventQueryResult::kSuccess;
  ASSERT_TRUE(value.poll(events, evidence).ok());
  EXPECT_EQ(evidence.calls, 1);
  std::vector<std::byte> logits(
      QwenSemanticObservationTransfer::kFinalLogitsBytes, std::byte{1});
  std::vector<std::byte> kv(1 * QwenKvAddressMapper::kBytesPerToken * 2 * 28,
                            std::byte{2});
  QwenSemanticOutcomeRecorder recorder;
  const std::array payload{std::byte{3}};
  ASSERT_TRUE(recorder.record_dispatch(payload).ok());
  ASSERT_TRUE(recorder.record_greedy_trajectory(payload).ok());
  ASSERT_TRUE(recorder.record_accepted_token_ledger(payload).ok());
  ASSERT_TRUE(recorder.observe_capacity(100, 20).ok());
  ASSERT_TRUE(value.publish(logits, kv, recorder).ok());
  EXPECT_EQ(value.state(), QwenSemanticObservationPipelineState::kPublished);
  EXPECT_TRUE(recorder.seal().ok());
}

TEST(QwenSemanticObservationPipelineTest, DeviceEvidenceFailurePoisons) {
  auto value = pipeline();
  CopyDriver copies;
  EventDriver events;
  Evidence evidence;
  events.result = CudaEventQueryResult::kSuccess;
  evidence.value.device_error_code = 6;
  ASSERT_TRUE(value.submit(copies, events, 100, 200).ok());
  EXPECT_FALSE(value.poll(events, evidence).ok());
  EXPECT_EQ(value.state(), QwenSemanticObservationPipelineState::kPoisoned);
}

TEST(QwenSemanticObservationPipelineTest, AwaitCompletesAfterRetryablePoll) {
  auto value = pipeline();
  CopyDriver copies;
  EventDriver events;
  Evidence evidence;
  Clock clock;
  Waiter waiter;
  clock.values = {100, 110, 120};
  ASSERT_TRUE(value.submit(copies, events, 100, 200).ok());
  events.result = CudaEventQueryResult::kNotReady;
  events.on_query = [&] {
    if (events.query_calls == 1) events.result = CudaEventQueryResult::kSuccess;
  };
  ASSERT_TRUE(value.await(events, evidence, clock, waiter).ok());
  EXPECT_EQ(waiter.calls, 1);
  EXPECT_EQ(value.state(), QwenSemanticObservationPipelineState::kComplete);
}

TEST(QwenSemanticObservationPipelineTest, AwaitPoisonsAtDeadline) {
  auto value = pipeline();
  CopyDriver copies;
  EventDriver events;
  Evidence evidence;
  Clock clock;
  Waiter waiter;
  clock.values = {200};
  ASSERT_TRUE(value.submit(copies, events, 100, 200).ok());
  const Status status = value.await(events, evidence, clock, waiter);
  EXPECT_EQ(status.code(), StatusCode::kInternal);
  EXPECT_EQ(waiter.calls, 0);
  EXPECT_EQ(value.state(), QwenSemanticObservationPipelineState::kPoisoned);
}

}  // namespace
}  // namespace pih
