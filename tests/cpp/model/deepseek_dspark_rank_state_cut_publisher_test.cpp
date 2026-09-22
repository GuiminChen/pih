#include "pih/model/deepseek_dspark_rank_state_cut_publisher.h"
#include "pih/model/deepseek_attention_terminal_state_drainer.h"

#include <gtest/gtest.h>

#include <optional>

namespace pih { namespace {
class BankOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};
class StageCompute final : public DeepSeekStageComputeDriver {
 public:
  Status launch(const DeepSeekPipelinePlanDescriptor&,
                const DeepSeekStagePlan&) override { return Status::Ok(); }
  Result<DeepSeekStageComputeStatus> poll() override {
    return DeepSeekStageComputeStatus::kSuccess;
  }
};
class DigestOperations final : public DeepSeekDsparkGpuStateDigestOperations {
 public:
  Status launch_component_sha256(
      const DeepSeekDsparkGpuStateDigestSubmission& value) override {
    for (std::size_t i = 0; i < value.components.size(); ++i) {
      const auto& component = value.components[i];
      auto digest = sha256({reinterpret_cast<const std::byte*>(
                                component.device_address),
                            static_cast<std::size_t>(component.bytes)});
      if (!digest.ok()) return digest.status();
      value.host_component_digests[i] = *digest;
    }
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};
struct Fixture final {
  BankOperations operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 5, operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(4).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 1, 1, banks, ratio4, ratio128).value();
  DeepSeekPipelineResourceSet pipeline_resources;
  std::optional<DeepSeekPipelineTransaction> pipeline_transaction;
  std::optional<DeepSeekPipelineStageExecutor> stage_executor;
  Fixture() {
    EXPECT_TRUE(transaction.begin(9).ok());
    EXPECT_TRUE(transaction.reserve_ratio4_append(2, 0).ok());
    EXPECT_TRUE(transaction.reserve_ratio128_append(3, 0).ok());
    EXPECT_TRUE(transaction.seal(9).ok());
    EXPECT_TRUE(transaction.poll().ok());
    auto capacity = DeepSeekPipelineCapacity::Create(1, 8, 8, 1, true);
    EXPECT_TRUE(capacity.ok());
    pipeline_resources = DeepSeekPipelineResourceSet::Create(*capacity).value();
    pipeline_transaction.emplace(pipeline_resources.prepare(
        {1, 1, DeepSeekPlanPhase::kVerify, 2, 1}).value());
    EXPECT_TRUE(pipeline_transaction->commit().ok());
    auto plan = DeepSeekPipelinePlan::Create(1, true).value();
    stage_executor.emplace(DeepSeekPipelineStageExecutor::Create(
        *pipeline_transaction, plan.rank(0), 1, nullptr, nullptr).value());
    StageCompute compute;
    DeepSeekStageExecutionDrivers drivers{&compute, {}, {}};
    EXPECT_TRUE(stage_executor->advance(drivers).ok());
    EXPECT_TRUE(stage_executor->advance(drivers).ok());
    EXPECT_EQ(stage_executor->state(),
              DeepSeekPipelineStageExecutorState::kComplete);
  }
};
DeepSeekDsparkGpuStateDigestPoll gpu_state_digest() {
  static std::array<std::byte, 8> state{
      std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
      std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
  std::array<DeepSeekDsparkDeviceStateComponent, 1> components{{
      {DeepSeekDsparkStateComponentKind::kFixed, 0,
       reinterpret_cast<std::uintptr_t>(state.data()), state.size()}}};
  std::array<Sha256Digest, 1> digests{};
  std::uint32_t error = 0;
  DigestOperations operations;
  auto coordinator = DeepSeekDsparkGpuStateDigestCoordinator::Create(
      1, operations).value();
  EXPECT_TRUE(coordinator.launch(
      {0, 1, 5, 6, 2, components, 1, 32, digests, 2, &error, 3, 4}).ok());
  auto result = coordinator.poll();
  EXPECT_TRUE(result.ok());
  return *result;
}
DeepSeekDsparkPrefixCommitment commitment() {
  std::array<Sha256Digest, 1> hashes{gpu_state_digest().local_state_hash()};
  return compile_deepseek_dspark_prefix_commitment(
      1, 5, 6, hashes).value();
}
DeepSeekDsparkStateCutDecision cut(bool terminal = false) {
  return {1, 5, 6, 2, 2,
          terminal ? std::optional<std::uint32_t>{}
                   : std::optional<std::uint32_t>{17},
          terminal, commitment().root};
}
DeepSeekDsparkRankCutReceipt receipt(bool terminal = false) {
  return DeepSeekDsparkRankCutReceipt::Create(
      terminal, gpu_state_digest(), commitment().proofs[0]).value();
}
class Drainer final : public DeepSeekDsparkTerminalStateDrainer {
 public:
  Status drain_committed_sequence_state(
      const DeepSeekDsparkStateCutDecision&) override {
    ++calls; return result;
  }
  int calls = 0;
  Status result = Status::Ok();
};
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     CommitsExactPrefixBeforeProducingAcknowledgement) {
  Fixture fixture;
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor, receipt()).value();
  auto ack = publisher.publish(cut());
  ASSERT_TRUE(ack.ok());
  EXPECT_TRUE(ack->prefix_published);
  EXPECT_EQ(ack->rank, 0U);
  EXPECT_EQ(fixture.banks.generation(), 2U);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 1U);
  EXPECT_EQ(fixture.ratio128.published_pages(), 1U);
  EXPECT_FALSE(publisher.publish(cut()).ok());
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     TerminalCutRollsBackTentativeThenDrainsCommittedState) {
  Fixture fixture; Drainer drainer;
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor,
      receipt(true), &drainer).value();
  ASSERT_TRUE(publisher.publish(cut(true)).ok());
  EXPECT_EQ(drainer.calls, 1);
  EXPECT_EQ(fixture.banks.generation(), 1U);
  EXPECT_EQ(fixture.ratio4.free_pairs(), 4U);
  EXPECT_EQ(fixture.ratio128.free_pages(), 4U);
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     RefusesTerminalAckWithoutCommittedStateDrainer) {
  Fixture fixture;
  EXPECT_FALSE(DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor, receipt(true)).ok());
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     RefusesMismatchedCutWithoutPublishingState) {
  Fixture fixture;
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor, receipt()).value();
  auto wrong = cut(); wrong.new_generation = 7;
  EXPECT_FALSE(publisher.publish(wrong).ok());
  EXPECT_EQ(fixture.banks.generation(), 1U);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     RefusesCutWithDifferentRetainedPrefixLength) {
  Fixture fixture;
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor, receipt()).value();
  auto wrong = cut();
  wrong.retained_record_count = 1;
  EXPECT_FALSE(publisher.publish(wrong).ok());
  EXPECT_EQ(fixture.banks.generation(), 1U);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     ReceiptRejectsProofNotProducedFromGpuStateDigest) {
  auto proof = commitment().proofs[0];
  proof.local_state_hash.bytes[0] ^= std::byte{1};
  EXPECT_FALSE(DeepSeekDsparkRankCutReceipt::Create(
      false, gpu_state_digest(), proof).ok());
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     RefusesUnprovenPrefixRootBeforePublishingState) {
  Fixture fixture;
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor, receipt()).value();
  auto wrong = cut();
  wrong.per_rank_prefix_hash_merkle_root.bytes[0] ^= std::byte{1};
  EXPECT_FALSE(publisher.publish(wrong).ok());
  EXPECT_EQ(fixture.banks.generation(), 1U);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
  EXPECT_EQ(fixture.ratio128.published_pages(), 0U);
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     TerminalDrainFailurePoisonsAndProducesNoAcknowledgement) {
  Fixture fixture; Drainer drainer;
  drainer.result = Status::Internal("drain failed");
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor,
      receipt(true), &drainer).value();
  EXPECT_FALSE(publisher.publish(cut(true)).ok());
  EXPECT_TRUE(publisher.poisoned());
  drainer.result = Status::Ok();
  EXPECT_FALSE(publisher.publish(cut(true)).ok());
  EXPECT_EQ(drainer.calls, 1);
}
TEST(DeepSeekDsparkRankStateCutPublisherTest,
     TerminalAckRequiresConcreteTentativeAndCommittedStateDrain) {
  Fixture fixture;
  auto committed4 = fixture.ratio4.reserve(7, 2, 3).value();
  auto committed128 = fixture.ratio128.reserve(7, 3, 3).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed4).ok());
  ASSERT_TRUE(fixture.ratio128.publish(committed128).ok());
  auto stage = DeepSeekPipelinePlan::Create(1, false).value().rank(0);
  auto bytes = DeepSeekAttentionStateGeometry::StageLogicalBytes(
      stage, 4096).value();
  DeepSeekAttentionStatePool state_pool(bytes);
  auto reservation = state_pool.reserve(7, stage, 4096).value();
  ASSERT_TRUE(reservation.publish().ok());
  auto drainer = DeepSeekAttentionTerminalStateDrainer::Create(
      7, reservation, fixture.ratio4, fixture.ratio128).value();
  auto publisher = DeepSeekDsparkRankStateCutPublisher::Create(
      0, fixture.transaction, *fixture.stage_executor,
      receipt(true), &drainer).value();
  auto ack = publisher.publish(cut(true));
  ASSERT_TRUE(ack.ok());
  EXPECT_TRUE(ack->prefix_published);
  EXPECT_TRUE(drainer.drained());
  EXPECT_EQ(state_pool.owned_bytes(), 0U);
  EXPECT_EQ(fixture.ratio4.free_pairs(), 4U);
  EXPECT_EQ(fixture.ratio128.free_pages(), 4U);
}
}}  // namespace pih::<anonymous>
