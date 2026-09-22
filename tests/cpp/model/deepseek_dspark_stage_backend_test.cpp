#include "pih/model/deepseek_dspark_stage_backend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace pih { namespace {
class Fixed final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};
class EmbedOps final : public DeepSeekDsparkEmbedOperations {
 public:
  explicit EmbedOps(std::vector<std::string>& c) : calls(&c) {}
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { calls->push_back("zero"); return Status::Ok(); }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override { calls->push_back("embed"); return Status::Ok(); }
  Status gemm(DeepSeekFp8GemmLaunch) override { return Status::Ok(); }
  Status rms(DeepSeekRmsNormLaunch) override { return Status::Ok(); }
  Status draft_init(DeepSeekDsparkDraftInitLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  std::vector<std::string>* calls;
};
class HeadOps final : public DeepSeekDsparkHeadOperations {
 public:
  explicit HeadOps(std::vector<std::string>& c) : calls(&c) {}
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status hc_head(DeepSeekHcHeadLaunch) override {
    calls->push_back("hc");
    return fail ? Status::Internal("injected head failure") : Status::Ok();
  }
  Status rms_norm(DeepSeekRmsNormLaunch) override { calls->push_back("rms"); return Status::Ok(); }
  Status lm_head(DeepSeekLmHeadLaunch) override { calls->push_back("lm"); return Status::Ok(); }
  Status markov(DeepSeekDsparkMarkovLaunch) override { calls->push_back("head"); return Status::Ok(); }
  Status argmax(DeepSeekArgmaxLaunch) override { return Status::Ok(); }
  Status confidence(DeepSeekDsparkConfidenceLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  std::vector<std::string>* calls;
  bool fail = false;
};
class Blocks final : public DeepSeekDsparkBlockExecutor {
 public:
  explicit Blocks(std::vector<std::string>& c) : calls(&c) {}
  Status launch(DeepSeekDsparkStageId stage,
                const DeepSeekPipelinePlanDescriptor&) override {
    active = deepseek_dspark_stage_index(stage);
    calls->push_back("block" + std::to_string(active)); return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    calls->push_back("poll" + std::to_string(active));
    return active == fail_stage ? DeepSeekStageComputeStatus::kError
                                : DeepSeekStageComputeStatus::kSuccess;
  }
  Status cancel() override {
    calls->push_back("cancel" + std::to_string(active));
    return Status::Ok();
  }
  std::vector<std::string>* calls;
  std::uint32_t active = 0;
  std::uint32_t fail_stage = UINT32_MAX;
};
class Fallback final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand&, const DeepSeekPipelinePlanDescriptor&) override { ++launches; return Status::Ok(); }
  Result<DeepSeekStageComputeStatus> poll() override { return DeepSeekStageComputeStatus::kSuccess; }
  int launches = 0;
};
DeepSeekDsparkEmbedSubmission embed_submission() {
  return {{1,2,3,90,11,1,12288},{2,3,4,5,6,90,11,1,4096,12288},
          {6,7,8,90,11,1,4096,1e-6F},
          {9,10,11,12,90,11,1,17,5,129280,4096,4}};
}
DeepSeekDsparkHeadSubmission head_submission() {
  DeepSeekDsparkHeadSubmission s;
  constexpr std::uintptr_t raw=0x1000000,bias=0x2000000,emb=0x3000000,tok=0x4000000;
  s.hc={0x5000000,12,13,14,0x6000000,90,11,5,4096,4,1e-6F,1e-6F};
  s.rms={0x6000000,15,0x7000000,90,11,5,4096,1e-6F};
  s.lm={0x7000000,16,raw,90,11,5,129280,4096};
  for(std::size_t i=0;i<5;++i){
    s.markov[i]={tok+i*4,2,3,raw+i*129280ULL*4,emb+i*512,
                 bias+i*129280ULL*4,90,11,1,129280,256};
    s.argmax[i]={s.markov[i].biased_logits_f32,tok+(i+1)*4,90,11,129280};
  }
  s.confidence={5,emb,6,7,90,11,5,4096,256}; return s;
}
struct Fixture final {
  std::vector<std::string> calls;
  Fixed fixed;
  DeepSeekFixedStateBanks banks=DeepSeekFixedStateBanks::Create({0x1000,256},{0x2000,256},121,fixed).value();
  DeepSeekRatio4PagePool r4=DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool r128=DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction tx=DeepSeekAttentionSequenceTransaction::Create(1,0,0,banks,r4,r128).value();
  EmbedOps embed_ops{calls}; HeadOps head_ops{calls}; std::uint32_t error=0;
  DeepSeekDsparkEmbedCoordinator embed=DeepSeekDsparkEmbedCoordinator::Create(embed_ops,&error).value();
  DeepSeekDsparkHeadExecutor head=DeepSeekDsparkHeadExecutor::Create(head_ops,&error).value();
  Blocks blocks{calls};
  DeepSeekDsparkStageWork work{
      DeepSeekDsparkStageWorkKind::kDecodeProposal,
      &embed,&head,&tx,embed_submission(),head_submission()};
};
class Provider final : public DeepSeekDsparkStageWorkProvider {
 public: Result<const DeepSeekDsparkStageWork*> resolve(const DeepSeekPipelinePlanDescriptor&) override { return work; }
  const DeepSeekDsparkStageWork* work=nullptr;
};
DeepSeekPipelinePlanDescriptor plan(){return {3,7,DeepSeekPlanPhase::kDecode,1,1};}
TEST(DeepSeekDsparkStageBackendTest,
     RunsEmbedThreeStagesAndHeadWithOrderedEvidence) {
  Fixture f; ASSERT_TRUE(f.tx.begin(11).ok()); Provider p; p.work=&f.work; Fallback fallback;
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(fallback,p,f.blocks).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kDspark,0},plan()).ok());
  EXPECT_EQ(f.calls[1],"embed"); EXPECT_EQ(f.calls[2],"block0");
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(f.calls, std::vector<std::string>({
      "zero","embed","block0","poll0","block1","poll1","block2",
      "poll2","hc","rms","lm","head","head","head","head","head"}));
  ASSERT_EQ(backend.evidence().size(), 3U);
  for (std::uint32_t index = 0; index < 3; ++index) {
    EXPECT_EQ(deepseek_dspark_stage_index(backend.evidence()[index].stage),
              index);
    EXPECT_EQ(backend.evidence()[index].state,
              DeepSeekDsparkStageEvidenceState::kSucceeded);
  }
  EXPECT_TRUE(backend.head_submitted());
}
TEST(DeepSeekDsparkStageBackendTest,
     EarlierFailureBlocksLaterStageAndRollsBackWholeTransaction) {
  Fixture f; ASSERT_TRUE(f.tx.begin(11).ok());
  f.blocks.fail_stage = 1;
  Provider p; p.work=&f.work; Fallback fallback;
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(
      fallback,p,f.blocks).value();
  ASSERT_TRUE(backend.launch(
      {DeepSeekStageOperatorKind::kDspark,0},plan()).ok());
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kError);
  EXPECT_EQ(f.tx.state(), DeepSeekAttentionSequenceTransactionState::kIdle);
  EXPECT_FALSE(backend.head_submitted());
  EXPECT_EQ(std::ranges::count(f.calls, "block2"), 0);
  EXPECT_EQ(std::ranges::count(f.calls, "head"), 0);
  EXPECT_EQ(backend.evidence()[1].state,
            DeepSeekDsparkStageEvidenceState::kFailed);
}
TEST(DeepSeekDsparkStageBackendTest,
     HeadFailureRollsBackAfterAllStagesWithoutClaimingCompletion) {
  Fixture f; ASSERT_TRUE(f.tx.begin(11).ok());
  f.head_ops.fail = true;
  Provider p; p.work=&f.work; Fallback fallback;
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(
      fallback,p,f.blocks).value();
  ASSERT_TRUE(backend.launch(
      {DeepSeekStageOperatorKind::kDspark,0},plan()).ok());
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_FALSE(backend.poll().ok());
  EXPECT_EQ(f.tx.state(), DeepSeekAttentionSequenceTransactionState::kIdle);
  EXPECT_FALSE(backend.head_submitted());
  for (const auto& evidence : backend.evidence()) {
    EXPECT_EQ(evidence.state, DeepSeekDsparkStageEvidenceState::kSucceeded);
  }
}
TEST(DeepSeekDsparkStageBackendTest,
     CancellationRollsBackTheWholeTransactionAndBlocksRedispatch) {
  Fixture f; ASSERT_TRUE(f.tx.begin(11).ok());
  Provider p; p.work=&f.work; Fallback fallback;
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(
      fallback,p,f.blocks).value();
  ASSERT_TRUE(backend.launch(
      {DeepSeekStageOperatorKind::kDspark,0},plan()).ok());
  ASSERT_TRUE(backend.cancel().ok());
  EXPECT_EQ(f.tx.state(), DeepSeekAttentionSequenceTransactionState::kIdle);
  EXPECT_EQ(backend.evidence()[0].state,
            DeepSeekDsparkStageEvidenceState::kCancelled);
  EXPECT_FALSE(backend.head_submitted());
  EXPECT_NE(std::ranges::find(f.calls, "cancel0"), f.calls.end());
  EXPECT_EQ(std::ranges::count(f.calls, "block1"), 0);
  EXPECT_EQ(std::ranges::count(f.calls, "head"), 0);
  EXPECT_FALSE(backend.poll().ok());
  EXPECT_FALSE(backend.launch(
      {DeepSeekStageOperatorKind::kDspark,0},plan()).ok());
}
TEST(DeepSeekDsparkStageBackendTest,
     PrefillInitializesThreeStageStateWithoutDraftOrHeadSubmission) {
  Fixture f; ASSERT_TRUE(f.tx.begin(11).ok());
  Provider p; p.work=&f.work; Fallback fallback;
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(
      fallback,p,f.blocks).value();
  auto prefill = plan();
  prefill.phase = DeepSeekPlanPhase::kPrefill;
  f.work.kind = DeepSeekDsparkStageWorkKind::kPrefillStateInitialization;
  f.work.head_executor = nullptr;
  ASSERT_TRUE(backend.launch(
      {DeepSeekStageOperatorKind::kDspark,0},prefill).ok());
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(f.calls, std::vector<std::string>({
      "zero","embed","block0","poll0","block1","poll1",
      "block2","poll2"}));
  EXPECT_TRUE(backend.prefill_state_initialized());
  EXPECT_FALSE(backend.head_submitted());
  for (const auto& evidence : backend.evidence()) {
    EXPECT_EQ(evidence.state,
              DeepSeekDsparkStageEvidenceState::kSucceeded);
  }
}
TEST(DeepSeekDsparkStageBackendTest,
     RejectsEmptyPlanBeforeProviderOrEmbedSubmission) {
  Fixture f; Provider p; p.work=&f.work; Fallback fallback;
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(
      fallback,p,f.blocks).value();
  auto empty = plan();
  empty.sequence_count = 0;
  EXPECT_EQ(backend.launch(
      {DeepSeekStageOperatorKind::kDspark,0},empty).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(f.calls.empty());
}
TEST(DeepSeekDsparkStageBackendTest, PassesNonDsparkCommandsToFallback) {
  Provider p; Fallback fallback; std::vector<std::string> calls; Blocks blocks{calls};
  auto backend=DeepSeekDsparkStageOperatorBackend::Create(fallback,p,blocks).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kHead,0},plan()).ok());
  EXPECT_EQ(fallback.launches,1); EXPECT_EQ(backend.poll().value(),DeepSeekStageComputeStatus::kSuccess);
}
}}  // namespace pih::<anonymous>
