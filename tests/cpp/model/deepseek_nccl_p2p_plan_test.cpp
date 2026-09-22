#include "pih/model/deepseek_nccl_p2p_plan.h"
#include "pih/model/deepseek_nccl_endpoint_warmup_runner.h"

#include <gtest/gtest.h>

#include <deque>

namespace pih {
namespace {

DeepSeekNcclP2pManifest manifest(DeepSeekNcclRole role) {
  return {.operation_plan_id = 10,
          .engine_epoch = 2,
          .communicator_generation = 3,
          .pipeline_plan_sequence = 4,
          .operation_ordinal = 5,
          .global_issue_ordinal = 7,
          .directed_boundary_id = 1,
          .role = role,
          .local_global_rank = role == DeepSeekNcclRole::kSend ? 1U : 2U,
          .peer_global_rank = role == DeepSeekNcclRole::kSend ? 2U : 1U,
          .communicator_local_rank = role == DeepSeekNcclRole::kSend ? 0U : 1U,
          .communicator_peer_rank = role == DeepSeekNcclRole::kSend ? 1U : 0U,
          .buffer_owner_id = 20,
          .buffer_offset_bytes = 256,
          .buffer_capacity_bytes = 131072,
          .buffer_generation = 6,
          .context_identity = 77,
          .token_count = 2};
}

class FakeNcclDriver final : public DeepSeekBoundaryTransportDriver {
 public:
  Status bind_p2p(DeepSeekNcclRole role, void* buffer,
                  std::uint64_t bytes, std::uintptr_t stream) override {
    bound_roles.push_back(role); bound_buffer = buffer;
    bound_bytes = bytes; bound_stream = stream; return Status::Ok();
  }
  Status group_start() override { calls.push_back("start"); return start_status; }
  Status send(std::uint64_t count, std::uint32_t peer) override {
    calls.push_back("send"); observed_count = count; observed_peer = peer; return op_status;
  }
  Status recv(std::uint64_t count, std::uint32_t peer) override {
    calls.push_back("recv"); observed_count = count; observed_peer = peer; return op_status;
  }
  Result<DeepSeekNcclAsyncStatus> group_end() override {
    calls.push_back("end");
    if (!end_status.ok()) return end_status;
    return end_result;
  }
  Result<DeepSeekNcclAsyncStatus> async_status() override {
    calls.push_back("poll");
    if (polls.empty()) return DeepSeekNcclAsyncStatus::kSuccess;
    auto value = polls.front(); polls.pop_front(); return value;
  }
  std::vector<std::string> calls;
  Status start_status = Status::Ok();
  Status op_status = Status::Ok();
  Status end_status = Status::Ok();
  DeepSeekNcclAsyncStatus end_result = DeepSeekNcclAsyncStatus::kSuccess;
  std::deque<DeepSeekNcclAsyncStatus> polls;
  std::uint64_t observed_count = 0;
  std::uint32_t observed_peer = 0;
  std::vector<DeepSeekNcclRole> bound_roles;
  void* bound_buffer = nullptr;
  std::uint64_t bound_bytes = 0;
  std::uintptr_t bound_stream = 0;
};

class FakeEventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override { ++records; return Status::Ok(); }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
  int records = 0;
};

class CleanEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class FakeWarmupPayload final : public DeepSeekNcclWarmupPayloadOperations {
 public:
  Status prepare(DeepSeekNcclRole role, void*, std::uint64_t bytes,
                 DriverStreamHandle, std::uint64_t pattern) override {
    roles.push_back(role); prepared_bytes.push_back(bytes);
    patterns.push_back(pattern); return Status::Ok();
  }
  Result<Sha256Digest> digest(const void*, std::uint64_t bytes) override {
    digested_bytes.push_back(bytes);
    Sha256Digest result{};
    result.bytes[0] = static_cast<std::byte>(digested_bytes.size());
    return result;
  }
  std::vector<DeepSeekNcclRole> roles;
  std::vector<std::uint64_t> prepared_bytes;
  std::vector<std::uint64_t> patterns;
  std::vector<std::uint64_t> digested_bytes;
};

TEST(DeepSeekNcclP2pPlanTest, ValidatesMatchingDirectedPair) {
  auto send = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kSend));
  auto recv = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kRecv));
  ASSERT_TRUE(send.ok());
  ASSERT_TRUE(recv.ok());
  EXPECT_TRUE(DeepSeekNcclP2pPlan::ValidatePair(send->manifest(), recv->manifest()).ok());
  auto wrong = recv->manifest();
  ++wrong.token_count;
  EXPECT_FALSE(DeepSeekNcclP2pPlan::ValidatePair(send->manifest(), wrong).ok());
}

TEST(DeepSeekNcclP2pPlanTest,
     PairSharesGlobalOrdinalButAllowsDifferentLocalProjection) {
  auto send = manifest(DeepSeekNcclRole::kSend);
  auto recv = manifest(DeepSeekNcclRole::kRecv);
  send.operation_ordinal = 2;
  recv.operation_ordinal = 1;
  EXPECT_TRUE(DeepSeekNcclP2pPlan::ValidatePair(send, recv).ok());
  ++recv.global_issue_ordinal;
  EXPECT_FALSE(DeepSeekNcclP2pPlan::ValidatePair(send, recv).ok());
}

TEST(DeepSeekNcclP2pPlanTest, IssuesExactlyOneOperationAndClosesFrontier) {
  auto plan = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kSend));
  ASSERT_TRUE(plan.ok());
  FakeNcclDriver nccl;
  ASSERT_TRUE(plan->issue(nccl).ok());
  EXPECT_EQ(nccl.calls, (std::vector<std::string>{"start", "send", "end"}));
  EXPECT_EQ(nccl.observed_count, 32768U);
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kIssuedToStream);

  auto event = CompletionEventSlot::Create(1, 2);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 1, 4, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ASSERT_TRUE(event.ok()); ASSERT_TRUE(frontier.ok());
  FakeEventDriver event_driver;
  CleanEvidence evidence;
  ASSERT_TRUE(plan->record_completion(*event, event_driver, 3).ok());
  ASSERT_TRUE(plan->poll_completion(nccl, *event, event_driver, *frontier,
                                    evidence).ok());
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kCompleteVerified);
}

TEST(DeepSeekNcclP2pPlanTest, PollsInProgressBeforeRecordingCudaEvent) {
  auto plan = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kRecv));
  ASSERT_TRUE(plan.ok());
  FakeNcclDriver nccl;
  nccl.end_result = DeepSeekNcclAsyncStatus::kInProgress;
  nccl.polls = {DeepSeekNcclAsyncStatus::kInProgress,
                DeepSeekNcclAsyncStatus::kSuccess};
  ASSERT_TRUE(plan->issue(nccl).ok());
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kGroupEndPending);
  auto event = CompletionEventSlot::Create(1, 2);
  FakeEventDriver event_driver;
  ASSERT_TRUE(event.ok());
  EXPECT_FALSE(plan->record_completion(*event, event_driver, 3).ok());
  EXPECT_TRUE(plan->poll_issue(nccl).ok());
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kGroupEndPending);
  EXPECT_TRUE(plan->poll_issue(nccl).ok());
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kIssuedToStream);
}

TEST(DeepSeekNcclP2pPlanTest, AnyCommittedIssueErrorPoisonsPlan) {
  auto plan = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kSend));
  ASSERT_TRUE(plan.ok());
  FakeNcclDriver nccl;
  nccl.op_status = Status::Internal("send failed");
  EXPECT_FALSE(plan->issue(nccl).ok());
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kPoisoned);
  EXPECT_FALSE(plan->issue(nccl).ok());
}

TEST(DeepSeekNcclP2pPlanTest, RejectsUnknownRoleAndPoisonsFinalAsyncError) {
  auto invalid = manifest(DeepSeekNcclRole::kSend);
  invalid.role = static_cast<DeepSeekNcclRole>(255);
  EXPECT_FALSE(DeepSeekNcclP2pPlan::Create(invalid).ok());

  auto plan = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kSend));
  ASSERT_TRUE(plan.ok());
  FakeNcclDriver nccl;
  ASSERT_TRUE(plan->issue(nccl).ok());
  nccl.polls = {DeepSeekNcclAsyncStatus::kError};
  auto event = CompletionEventSlot::Create(1, 2);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 1, 4, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ASSERT_TRUE(event.ok()); ASSERT_TRUE(frontier.ok());
  FakeEventDriver event_driver;
  CleanEvidence evidence;
  ASSERT_TRUE(plan->record_completion(*event, event_driver, 3).ok());
  EXPECT_FALSE(plan->poll_completion(nccl, *event, event_driver, *frontier,
                                     evidence).ok());
  EXPECT_EQ(plan->state(), DeepSeekNcclP2pState::kPoisoned);
}

TEST(DeepSeekNcclP2pPlanTest, SequencerAllowsOnlyOneVerifiedOrdinal) {
  auto first = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kSend));
  auto second_manifest = manifest(DeepSeekNcclRole::kSend);
  second_manifest.operation_plan_id = 11;
  second_manifest.operation_ordinal = 6;
  auto second = DeepSeekNcclP2pPlan::Create(second_manifest);
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  ASSERT_TRUE(first.ok()); ASSERT_TRUE(second.ok()); ASSERT_TRUE(sequencer.ok());
  FakeNcclDriver nccl;
  ASSERT_TRUE(sequencer->issue(*first, nccl).ok());
  EXPECT_FALSE(sequencer->issue(*second, nccl).ok());
  EXPECT_FALSE(sequencer->close(*first).ok());

  auto event = CompletionEventSlot::Create(1, 2);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 1, 4, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ASSERT_TRUE(event.ok()); ASSERT_TRUE(frontier.ok());
  FakeEventDriver event_driver;
  CleanEvidence evidence;
  ASSERT_TRUE(first->record_completion(*event, event_driver, 3).ok());
  ASSERT_TRUE(first->poll_completion(nccl, *event, event_driver, *frontier,
                                     evidence).ok());
  ASSERT_TRUE(sequencer->close(*first).ok());
  EXPECT_EQ(sequencer->next_ordinal(), 6U);
  EXPECT_TRUE(sequencer->issue(*second, nccl).ok());
}

TEST(DeepSeekNcclP2pPlanTest, SequencerPoisonsAfterCommittedIssueError) {
  auto plan = DeepSeekNcclP2pPlan::Create(manifest(DeepSeekNcclRole::kSend));
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(sequencer.ok());
  FakeNcclDriver nccl;
  nccl.end_status = Status::Internal("group end failed");
  EXPECT_FALSE(sequencer->issue(*plan, nccl).ok());
  EXPECT_TRUE(sequencer->poisoned());
}

TEST(DeepSeekNcclEndpointWarmupRunnerTest,
     ExecutesVerifiedMinimumThenMaximumSend) {
  std::array<std::byte, 4> storage{};
  DeepSeekNcclCommunicatorManifest endpoint{
      2, 3, 4, 5, 1, 1, 2, 0, 8, 9, 10};
  FakeNcclDriver nccl;
  FakeWarmupPayload payload;
  auto runner = DeepSeekNcclEndpointWarmupRunner::Create(
      {endpoint, 2, 20, 30, 31, storage.data(), 65536, 40, {50, 51},
       100, 200}, nccl, payload);
  ASSERT_TRUE(runner.ok()) << runner.status().message();
  ASSERT_TRUE(runner->begin().ok());
  FakeEventDriver events;
  CleanEvidence evidence;
  auto first = runner->poll(events, evidence, 101);
  ASSERT_TRUE(first.ok()) << first.status().message();
  EXPECT_FALSE(first->has_value());
  auto second = runner->poll(events, evidence, 102);
  ASSERT_TRUE(second.ok()) << second.status().message();
  ASSERT_TRUE(second->has_value());
  EXPECT_EQ((*second)->minimum_token_count, 1U);
  EXPECT_EQ((*second)->maximum_token_count, 2U);
  EXPECT_TRUE((*second)->minimum_complete_verified);
  EXPECT_EQ(payload.prepared_bytes,
            (std::vector<std::uint64_t>{32768, 65536}));
  EXPECT_EQ(payload.digested_bytes, payload.prepared_bytes);
  EXPECT_EQ(nccl.bound_roles,
            (std::vector<DeepSeekNcclRole>{DeepSeekNcclRole::kSend,
                                           DeepSeekNcclRole::kSend}));
  EXPECT_EQ(events.records, 2);
}

TEST(DeepSeekNcclEndpointWarmupRunnerTest,
     PoisonsBeforeDigestWhenCompletionEvidenceFails) {
  class DirtyEvidence final : public CompletionEvidenceProvider {
   public:
    Result<CompletionPublicationEvidence> collect() override {
      return CompletionPublicationEvidence{false, 0, false};
    }
  } dirty;
  std::array<std::byte, 4> storage{};
  DeepSeekNcclCommunicatorManifest endpoint{
      2, 3, 4, 5, 0, 1, 0, 1, 8, 9, 10};
  FakeNcclDriver nccl;
  FakeWarmupPayload payload;
  auto runner = DeepSeekNcclEndpointWarmupRunner::Create(
      {endpoint, 1, 20, 30, 31, storage.data(), 32768, 40, {50, 51},
       100, 200}, nccl, payload).value();
  ASSERT_TRUE(runner.begin().ok());
  FakeEventDriver events;
  EXPECT_FALSE(runner.poll(events, dirty, 101).ok());
  EXPECT_TRUE(runner.poisoned());
  EXPECT_TRUE(payload.digested_bytes.empty());
}

TEST(DeepSeekNcclEndpointWarmupRunnerTest,
     PoisonsWhenCompletionRemainsNotReadyAtDeadline) {
  class NotReadyEventDriver final : public CompletionEventDriver {
   public:
    Status record(DriverEventHandle, DriverStreamHandle) override {
      return Status::Ok();
    }
    Result<CudaEventQueryResult> query(DriverEventHandle) override {
      return CudaEventQueryResult::kNotReady;
    }
  } events;
  std::array<std::byte, 4> storage{};
  DeepSeekNcclCommunicatorManifest endpoint{
      2, 3, 4, 5, 0, 0, 1, 0, 8, 9, 10};
  FakeNcclDriver nccl;
  FakeWarmupPayload payload;
  auto runner = DeepSeekNcclEndpointWarmupRunner::Create(
      {endpoint, 1, 20, 30, 31, storage.data(), 32768, 40, {50, 51},
       100, 200}, nccl, payload).value();
  ASSERT_TRUE(runner.begin().ok());
  CleanEvidence evidence;
  auto result = runner.poll(events, evidence, 200);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInternal);
  EXPECT_TRUE(runner.poisoned());
}

}  // namespace
}  // namespace pih
