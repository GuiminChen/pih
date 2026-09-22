#include "pih/model/deepseek_rank_plan_compiler.h"
#include "pih/model/deepseek_attention_projection_submission_assembler.h"
#include "pih/model/deepseek_mhc_submission_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekDenseMhcLayerSubmissionInput valid_dense_mhc_layer(
    std::uint32_t layer, std::uint32_t token_count) {
  const DeepSeekAttentionWeightBindings attention_weights{
      0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107,
      0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D, 71};
  const DeepSeekAttentionProjectionDeviceView attention_workspace{
      0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x208,
      0x209, 0x20A, 0x20B, 0x20C, 0x20D, 0x20E, 0x20F, 0x305,
      0x304, 0x210};
  auto attention =
      DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
          attention_weights, attention_workspace, 8, token_count, 0x301,
          0x302, 0x303, 104, 17)
          .value();
  const DeepSeekMhcWeightBindings mhc_weights{
      0x401, 0x402, 0x403, 0x404,
      0x405, 0x406, 0x407, 0x408, 79};
  const DeepSeekMhcDeviceView mhc_workspace{
      0x501, 0x502, 0x503, 0x504, 0x505, 0x506};
  auto mhc = DeepSeekMhcSubmissionAssembler::Assemble(
                 layer, mhc_weights, mhc_workspace, 8, token_count,
                 0x501, attention.branch_output_bf16, 0x210, 17)
                 .value();
  return {
      .layer = layer,
      .attention_input = attention.input,
      .attention_output = attention.output,
      .sparse_query_bf16 = attention.sparse_query_bf16,
      .sparse_kv_bf16 = attention.sparse_kv_bf16,
      .sparse_output_bf16 = attention.sparse_output_bf16,
      .mhc_attention = mhc.attention,
      .mhc_feed_forward = mhc.feed_forward};
}

class Operations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status gemm(DeepSeekRouterBf16GemmLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class DeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("device allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class FixedOperations final : public DeepSeekFixedStateBankOperations {
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

class DensePinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("dense pinned");
    return Allocation{data, bytes, alignment, 77, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
};

class DenseProjectionOperations final
    : public DeepSeekAttentionProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override { return Status::Ok(); }
  Status gemm(DeepSeekFp8GemmLaunch) override { return Status::Ok(); }
  Status rms(DeepSeekRmsNormLaunch) override { return Status::Ok(); }
  Status head_rms(DeepSeekHeadRmsLaunch) override { return Status::Ok(); }
  Status rotary(DeepSeekRotaryLaunch) override { return Status::Ok(); }
  Status kv_simulate(DeepSeekKvFp8SimulateLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};

class DenseOutputOperations final
    : public DeepSeekAttentionOutputProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status rotary(DeepSeekRotaryLaunch) override { return Status::Ok(); }
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch) override { return Status::Ok(); }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override { return Status::Ok(); }
  Status gemm(DeepSeekFp8GemmLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};

class DenseMhcOperations final : public DeepSeekMhcSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status pre(DeepSeekMhcPreLaunch) override { return Status::Ok(); }
  Status branch(DeepSeekMhcBranchLaunch) override { return Status::Ok(); }
  Status post(DeepSeekMhcPostLaunch) override { return Status::Ok(); }
  Status target_hidden_tap(DeepSeekMhcTargetHiddenTapLaunch) override {
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};

template <typename T>
T* fake(std::uintptr_t value) { return reinterpret_cast<T*>(value); }

Result<DeepSeekRankPlanCompiler> compiler(
    DeepSeekStagePlan stage, DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations,
    DeepSeekDenseMhcRuntimeResources* dense_runtime = nullptr) {
  auto hash = DeepSeekHashRouterWorkFactory::Create(
      stage.layers, 2, 1,
      {{0, std::vector<std::uint16_t>(
               DeepSeekExpertSubwavePlan::kRoutesPerToken)}});
  if (!hash.ok()) return hash.status();
  auto learned = DeepSeekLearnedRouterWorkFactory::Create(
      stage.layers, 2, {}, store, operations);
  if (!learned.ok()) return learned.status();
  auto attention = DeepSeekAttentionWorkFactory::Create(
      stage.layers,
      {{0, fake<DeepSeekRecentStateWriter>(0x1000),
        fake<DeepSeekCompressedLayerUpdateCoordinator>(0x2000),
        fake<DeepSeekAttentionLayerCoordinator>(0x3000),
        fake<DeepSeekPrefillLayerCoordinator>(0x4000)}});
  if (!attention.ok()) return attention.status();
  auto dense = DeepSeekDenseMhcWorkFactory::Create(stage.layers, 1, 2);
  if (!dense.ok()) return dense.status();
  auto endpoint = DeepSeekEndpointWorkFactory::Create(stage, 1);
  if (!endpoint.ok()) return endpoint.status();
  return DeepSeekRankPlanCompiler::Create(
      stage, std::move(*hash), std::move(*learned),
      std::move(*attention), std::move(*dense), std::move(*endpoint),
      dense_runtime);
}

DeepSeekRankPlanCompilerInput decode_input(
    DeepSeekRankAttentionStatePool& state_pool) {
  DeepSeekRankPlanCompilerInput input;
  input.token_ids = {0};
  input.hash_router_scores = {{0, std::vector<float>(256)}};
  std::vector<std::uint32_t> positions{0};
  std::vector<std::uint32_t> visible{0};
  DeepSeekDecodeAttentionLayerPlanInput attention;
  attention.layer = 0;
  attention.attention.query_positions = positions;
  attention.attention.selection.visible_slot_counts = visible;
  attention.transaction = fake<DeepSeekAttentionSequenceTransaction>(0x5000);
  input.decode_attention.push_back(attention);

  DeepSeekDenseAttentionStageSequenceWork dense;
  dense.input_coordinator = fake<DeepSeekAttentionProjectionCoordinator>(0x6000);
  dense.output_coordinator =
      fake<DeepSeekAttentionOutputProjectionCoordinator>(0x7000);
  dense.transaction = attention.transaction;
  dense.input.input_quant.token_count = 1;
  dense.sparse_query_bf16 = 1;
  dense.sparse_kv_bf16 = 2;
  dense.sparse_output_bf16 = 3;
  input.dense_attention = {{0, {dense}}};

  DeepSeekMhcStageSequenceWork mhc;
  mhc.executor = fake<DeepSeekMhcSequenceExecutor>(0x8000);
  mhc.transaction = attention.transaction;
  mhc.submission.kind = DeepSeekMhcBranchKind::kAttention;
  mhc.submission.layer_id = 0;
  mhc.submission.token_count = 1;
  input.mhc_attention = {{0, {mhc}}};
  mhc.executor = fake<DeepSeekMhcSequenceExecutor>(0x9000);
  mhc.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
  input.mhc_feed_forward = {{0, {mhc}}};
  input.attention_state_pool = &state_pool;
  input.attention_bindings = {{71, 0}};
  return input;
}

TEST(DeepSeekRankPlanCompilerTest,
     AtomicallyCompilesEveryDecodeWorkCategory) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 0}, 2).value();
  Operations operations;
  DeviceAllocator device_allocator;
  FixedOperations fixed_operations;
  auto state_pool = DeepSeekRankAttentionStatePool::Allocate(
      device_allocator, {0, {2, 3}, false, false, false}, 1, 1, 19,
      fixed_operations, 17, 0).value();
  auto value = compiler({0, {0, 0}, false, false, false},
                        store, operations).value();
  auto work = value.compile(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, decode_input(state_pool));
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_EQ(work->hash_router.size(), 1U);
  EXPECT_EQ(work->decode_attention.size(), 1U);
  EXPECT_EQ(work->dense_attention.size(), 1U);
  EXPECT_EQ(work->mhc_attention.size(), 1U);
  EXPECT_EQ(work->mhc_feed_forward.size(), 1U);
  EXPECT_NE(work->lifetime_owner, nullptr);
  auto* owned_transaction = work->decode_attention[0].work.transaction;
  EXPECT_NE(owned_transaction,
            fake<DeepSeekAttentionSequenceTransaction>(0x5000));
  EXPECT_EQ(work->dense_attention[0].sequences[0].transaction,
            owned_transaction);
  EXPECT_EQ(work->mhc_attention[0].sequences[0].transaction,
            owned_transaction);
  EXPECT_TRUE(owned_transaction->begin(23).ok());
}

TEST(DeepSeekRankPlanCompilerTest,
     RejectsPhaseMixingAndFactoryStageDisagreement) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 0}, 2).value();
  Operations operations;
  DeviceAllocator device_allocator;
  FixedOperations fixed_operations;
  auto state_pool = DeepSeekRankAttentionStatePool::Allocate(
      device_allocator, {0, {2, 3}, false, false, false}, 1, 1, 19,
      fixed_operations, 17, 0).value();
  auto value = compiler({0, {0, 0}, false, false, false},
                        store, operations).value();
  auto input = decode_input(state_pool);
  input.chunk_attention.resize(1);
  EXPECT_FALSE(value.compile(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, std::move(input)).ok());

  auto foreign_input_pointer = decode_input(state_pool);
  foreign_input_pointer.dense_attention[0].sequences[0].transaction =
      fake<DeepSeekAttentionSequenceTransaction>(0xDEAD);
  auto rebound = value.compile(
      {7, 2, DeepSeekPlanPhase::kDecode, 1, 1},
      std::move(foreign_input_pointer));
  ASSERT_TRUE(rebound.ok()) << rebound.status().message();
  EXPECT_NE(rebound->dense_attention[0].sequences[0].transaction,
            fake<DeepSeekAttentionSequenceTransaction>(0xDEAD));

  auto multi_sequence = decode_input(state_pool);
  multi_sequence.attention_bindings.push_back({72, 0});
  EXPECT_FALSE(value.compile(
      {7, 3, DeepSeekPlanPhase::kDecode, 1, 2},
      std::move(multi_sequence)).ok());

  auto hash = DeepSeekHashRouterWorkFactory::Create(
      {0, 0}, 2, 1,
      {{0, std::vector<std::uint16_t>(
               DeepSeekExpertSubwavePlan::kRoutesPerToken)}}).value();
  auto learned = DeepSeekLearnedRouterWorkFactory::Create(
      {0, 0}, 2, {}, store, operations).value();
  auto attention = DeepSeekAttentionWorkFactory::Create(
      {0, 0}, {{0, fake<DeepSeekRecentStateWriter>(0x1000),
                fake<DeepSeekCompressedLayerUpdateCoordinator>(0x2000),
                fake<DeepSeekAttentionLayerCoordinator>(0x3000),
                fake<DeepSeekPrefillLayerCoordinator>(0x4000)}}).value();
  auto dense = DeepSeekDenseMhcWorkFactory::Create({0, 0}, 1, 2).value();
  auto wrong_endpoint = DeepSeekEndpointWorkFactory::Create(
      {1, {0, 0}, false, false, false}, 1).value();
  EXPECT_FALSE(DeepSeekRankPlanCompiler::Create(
      {0, {0, 0}, false, false, false}, std::move(hash),
      std::move(learned), std::move(attention), std::move(dense),
      std::move(wrong_endpoint)).ok());
}

TEST(DeepSeekRankPlanCompilerTest,
     AssemblesRawDenseMhcLayersWithPlanOwnedTransaction) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 0}, 2).value();
  Operations operations;
  DeviceAllocator device_allocator;
  FixedOperations fixed_operations;
  auto state_pool = DeepSeekRankAttentionStatePool::Allocate(
      device_allocator, {0, {2, 3}, false, false, false}, 1, 1, 19,
      fixed_operations, 17, 0).value();
  DensePinnedAllocator pinned;
  DenseProjectionOperations projection;
  DenseOutputOperations output;
  DenseMhcOperations mhc_operations;
  auto runtime = DeepSeekDenseMhcRuntimeResources::Allocate(
      {0, 0}, {&projection, &output, &mhc_operations}, pinned).value();
  auto value = compiler({0, {0, 0}, false, false, false}, store,
                        operations, &runtime).value();
  auto input = decode_input(state_pool);
  input.dense_attention.clear();
  input.mhc_attention.clear();
  input.mhc_feed_forward.clear();
  auto raw = valid_dense_mhc_layer(0, 1);
  input.dense_mhc_layers.push_back(raw);
  auto work = value.compile(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, std::move(input));
  ASSERT_TRUE(work.ok()) << work.status().message();
  const auto borrowed = runtime.borrow(0).value();
  EXPECT_EQ(work->dense_attention[0].sequences[0].input_coordinator,
            borrowed.input);
  EXPECT_EQ(work->mhc_attention[0].sequences[0].executor,
            borrowed.mhc_attention);
  EXPECT_EQ(work->mhc_feed_forward[0].sequences[0].executor,
            borrowed.mhc_feed_forward);
  EXPECT_EQ(work->dense_attention[0].sequences[0].transaction,
            work->mhc_attention[0].sequences[0].transaction);

  auto mixed = decode_input(state_pool);
  mixed.dense_mhc_layers.push_back(raw);
  EXPECT_FALSE(value.compile(
      {7, 10, DeepSeekPlanPhase::kDecode, 1, 1}, std::move(mixed)).ok());
}

}  // namespace
}  // namespace pih
