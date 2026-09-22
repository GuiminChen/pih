#include "pih/model/deepseek_attention_stage_backend.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class Fallback final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor&) override {
    last = command;
    ++launches;
    return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override { return status; }
  DeepSeekStageOperatorCommand last;
  DeepSeekStageComputeStatus status = DeepSeekStageComputeStatus::kSuccess;
  std::uint32_t launches = 0;
};

class Provider final : public DeepSeekDecodeAttentionWorkProvider {
 public:
  Result<const DeepSeekDecodeAttentionWork*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override {
    resolved_layer = layer;
    resolved_sequence = plan.plan_sequence;
    return work;
  }
  const DeepSeekDecodeAttentionWork* work = nullptr;
  std::uint32_t resolved_layer = 99;
  std::uint64_t resolved_sequence = 0;
};

class ChunkProvider final : public DeepSeekChunkAttentionWorkProvider {
 public:
  Result<const DeepSeekChunkAttentionWork*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override {
    resolved_layer = layer;
    resolved_phase = plan.phase;
    return work;
  }
  const DeepSeekChunkAttentionWork* work = nullptr;
  std::uint32_t resolved_layer = 99;
  DeepSeekPlanPhase resolved_phase = DeepSeekPlanPhase::kDrain;
};

DeepSeekPipelinePlanDescriptor plan(DeepSeekPlanPhase phase) {
  return {7, 11, phase, 1, 1};
}

TEST(DeepSeekAttentionStageBackendTest,
     DelegatesNonDecodeAndNonAttentionOperators) {
  Fallback fallback;
  Provider provider;
  auto backend =
      DeepSeekAttentionStageOperatorBackend::Create(fallback, provider).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kEmbedding, 0},
                             plan(DeepSeekPlanPhase::kDecode)).ok());
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kAttention, 4},
                             plan(DeepSeekPlanPhase::kPrefill)).ok());
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(fallback.launches, 2U);
  EXPECT_EQ(provider.resolved_layer, 99U);
}

TEST(DeepSeekAttentionStageBackendTest, RejectsMissingDecodeWorkFailStop) {
  Fallback fallback;
  Provider provider;
  auto backend =
      DeepSeekAttentionStageOperatorBackend::Create(fallback, provider).value();
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kAttention, 8},
                              plan(DeepSeekPlanPhase::kDecode)).ok());
  EXPECT_EQ(provider.resolved_layer, 8U);
  EXPECT_EQ(provider.resolved_sequence, 11U);
  EXPECT_FALSE(backend.poll().ok());
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kHead, 0},
                              plan(DeepSeekPlanPhase::kDecode)).ok());
}

TEST(DeepSeekAttentionStageBackendTest,
     ConfiguredChunkProviderFailStopsInsteadOfFallingBack) {
  Fallback fallback;
  Provider decode;
  ChunkProvider chunk;
  auto backend = DeepSeekAttentionStageOperatorBackend::Create(
      fallback, decode, chunk).value();
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kAttention, 5},
                              plan(DeepSeekPlanPhase::kPrefill)).ok());
  EXPECT_EQ(chunk.resolved_layer, 5U);
  EXPECT_EQ(chunk.resolved_phase, DeepSeekPlanPhase::kPrefill);
  EXPECT_EQ(fallback.launches, 0U);
}

} }  // namespace pih
