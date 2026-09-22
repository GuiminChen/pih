#include "pih/model/deepseek_dense_mhc_runtime_resources.h"
#include "pih/model/deepseek_dense_mhc_plan_input_assembler.h"
#include "pih/model/deepseek_attention_projection_submission_assembler.h"
#include "pih/model/deepseek_mhc_submission_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class DensePinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("dense runtime");
    return Allocation{data, bytes, alignment, 9, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
};

class ProjectionOps final : public DeepSeekAttentionProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t* value) override {
    return value == nullptr ? Status::InvalidArgument("null") : Status::Ok();
  }
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

class OutputOps final : public DeepSeekAttentionOutputProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t* value) override {
    return value == nullptr ? Status::InvalidArgument("null") : Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status rotary(DeepSeekRotaryLaunch) override { return Status::Ok(); }
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch) override { return Status::Ok(); }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override { return Status::Ok(); }
  Status gemm(DeepSeekFp8GemmLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};

class MhcOps final : public DeepSeekMhcSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t* value) override {
    return value == nullptr ? Status::InvalidArgument("null") : Status::Ok();
  }
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

class FixedOps final : public DeepSeekFixedStateBankOperations {
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

DeepSeekDenseMhcLayerSubmissionInput valid_layer(std::uint32_t layer) {
  const DeepSeekAttentionWeightBindings attention_weights{
      0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107,
      0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D, 71};
  const DeepSeekAttentionProjectionDeviceView attention_workspace{
      0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x208,
      0x209, 0x20A, 0x20B, 0x20C, 0x20D, 0x20E, 0x20F, 0x305,
      0x304, 0x210};
  auto attention =
      DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
          attention_weights, attention_workspace, 8, 4, 0x301, 0x302,
          0x303, 104, 0x8)
          .value();
  const DeepSeekMhcWeightBindings mhc_weights{
      0x401, 0x402, 0x403, 0x404,
      0x405, 0x406, 0x407, 0x408, 79};
  const DeepSeekMhcDeviceView mhc_workspace{
      0x501, 0x502, 0x503, 0x504, 0x505, 0x506};
  auto mhc = DeepSeekMhcSubmissionAssembler::Assemble(
                 layer, mhc_weights, mhc_workspace, 8, 4, 0x501,
                 attention.branch_output_bf16, 0x210, 0x8)
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

TEST(DeepSeekDenseMhcRuntimeResourcesTest,
     OwnsFourIndependentExecutorsPerLayer) {
  DensePinnedAllocator allocator;
  ProjectionOps projection;
  OutputOps output;
  MhcOps mhc;
  auto runtime = DeepSeekDenseMhcRuntimeResources::Allocate(
      {7, 8}, {&projection, &output, &mhc}, allocator);
  ASSERT_TRUE(runtime.ok()) << runtime.status().message();
  EXPECT_EQ(runtime->layer_count(), 2U);
  auto layer7 = runtime->borrow(7);
  auto layer8 = runtime->borrow(8);
  ASSERT_TRUE(layer7.ok());
  ASSERT_TRUE(layer8.ok());
  EXPECT_NE(layer7->input, nullptr);
  EXPECT_NE(layer7->output, nullptr);
  EXPECT_NE(layer7->mhc_attention, nullptr);
  EXPECT_NE(layer7->mhc_feed_forward, nullptr);
  EXPECT_NE(layer7->mhc_attention, layer7->mhc_feed_forward);
  EXPECT_NE(layer7->input, layer8->input);
  EXPECT_FALSE(runtime->borrow(6).ok());
}

TEST(DeepSeekDenseMhcRuntimeResourcesTest, RejectsIncompleteOperations) {
  DensePinnedAllocator allocator;
  ProjectionOps projection;
  OutputOps output;
  EXPECT_FALSE(DeepSeekDenseMhcRuntimeResources::Allocate(
      {7, 8}, {&projection, &output, nullptr}, allocator).ok());
}

TEST(DeepSeekDenseMhcRuntimeResourcesTest,
     RejectsSyntheticDsparkLayerIdentity) {
  DensePinnedAllocator allocator;
  ProjectionOps projection;
  OutputOps output;
  MhcOps mhc;
  EXPECT_FALSE(DeepSeekDenseMhcRuntimeResources::Allocate(
      {43, 43}, {&projection, &output, &mhc}, allocator).ok());
}

TEST(DeepSeekDenseMhcRuntimeResourcesTest,
     AssemblesCanonicalLayerWorkFromOwnedExecutors) {
  DensePinnedAllocator allocator;
  ProjectionOps projection;
  OutputOps output;
  MhcOps mhc;
  auto runtime = DeepSeekDenseMhcRuntimeResources::Allocate(
      {7, 8}, {&projection, &output, &mhc}, allocator).value();
  FixedOps fixed_ops;
  auto banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 31, fixed_ops).value();
  auto ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  auto ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 0, 0, banks, ratio4, ratio128).value();
  std::vector<DeepSeekDenseMhcLayerSubmissionInput> layers(2);
  for (std::uint32_t offset = 0; offset < 2; ++offset) {
    layers[offset] = valid_layer(7 + offset);
  }
  auto assembled = DeepSeekDenseMhcPlanInputAssembler::Assemble(
      {7, 8}, layers, runtime, transaction);
  ASSERT_TRUE(assembled.ok()) << assembled.status().message();
  EXPECT_EQ(assembled->dense_attention.size(), 2U);
  EXPECT_EQ(assembled->dense_attention[1].sequences[0].transaction,
            &transaction);
  EXPECT_EQ(assembled->mhc_attention[0].sequences[0].submission.kind,
            DeepSeekMhcBranchKind::kAttention);
  EXPECT_EQ(assembled->mhc_feed_forward[1].sequences[0].submission.layer_id,
            8U);
  std::swap(layers[0], layers[1]);
  EXPECT_FALSE(DeepSeekDenseMhcPlanInputAssembler::Assemble(
      {7, 8}, layers, runtime, transaction).ok());
}

TEST(DeepSeekDenseMhcRuntimeResourcesTest,
     RejectsInvalidInnerLaunchBeforeBorrowingPlanWork) {
  DensePinnedAllocator allocator;
  ProjectionOps projection;
  OutputOps output;
  MhcOps mhc;
  auto runtime = DeepSeekDenseMhcRuntimeResources::Allocate(
      {7, 7}, {&projection, &output, &mhc}, allocator).value();
  FixedOps fixed_ops;
  auto banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 31, fixed_ops).value();
  auto ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  auto ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 0, 0, banks, ratio4, ratio128).value();
  auto layer = valid_layer(7);
  layer.attention_input.q_norm.input_bf16 = 0;
  EXPECT_FALSE(DeepSeekDenseMhcPlanInputAssembler::Assemble(
                   {7, 7}, std::span(&layer, 1), runtime, transaction)
                   .ok());
  layer = valid_layer(7);
  layer.mhc_feed_forward.branch_output_bf16 = 0;
  EXPECT_FALSE(DeepSeekDenseMhcPlanInputAssembler::Assemble(
                   {7, 7}, std::span(&layer, 1), runtime, transaction)
                   .ok());
}

}  // namespace
}  // namespace pih
