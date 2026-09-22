#include "pih/model/deepseek_dense_mhc_layer_submission_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class DenseLayerAllocator final : public Allocator {
 public:
  explicit DenseLayerAllocator(std::int32_t device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("dense layer");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, device_).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::int32_t device_;
  std::uint64_t generation_ = 0;
};

DeepSeekAttentionWeightBindings attention_weights() {
  return {0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107,
          0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D, 71};
}

DeepSeekMhcWeightBindings mhc_weights() {
  return {0x201, 0x202, 0x203, 0x204,
          0x205, 0x206, 0x207, 0x208, 71};
}

TEST(DeepSeekDenseMhcLayerSubmissionAssemblerTest,
     BuildsOneAtomicLayerFromGenerationOwnedResources) {
  DenseLayerAllocator allocator(2);
  auto attention = DeepSeekAttentionProjectionDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  auto mhc = DeepSeekMhcDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  const auto attention_view = attention.view();
  const auto mhc_view = mhc.view();

  auto assembled = DeepSeekDenseMhcLayerSubmissionAssembler::Assemble(
      7, attention_weights(), mhc_weights(), attention, mhc, 4,
      mhc_view.residual_a_bf16, 0x301, 104, 0x401);
  ASSERT_TRUE(assembled.ok()) << assembled.status().message();
  EXPECT_EQ(assembled->input.layer, 7U);
  EXPECT_EQ(assembled->input.attention_input.input_quant.input_bf16,
            mhc_view.layer_input_bf16);
  EXPECT_EQ(assembled->input.attention_output.wo_b.output_bf16,
            attention_view.branch_output_bf16);
  EXPECT_EQ(assembled->input.sparse_query_bf16,
            attention_view.query_bf16);
  EXPECT_EQ(assembled->input.sparse_kv_bf16, attention_view.kv_bf16);
  EXPECT_EQ(assembled->input.sparse_output_bf16,
            attention_view.attention_output_bf16);
  EXPECT_EQ(assembled->input.mhc_attention.branch_output_bf16,
            attention_view.branch_output_bf16);
  EXPECT_EQ(assembled->input.mhc_feed_forward.branch_output_bf16,
            mhc_view.ffn_branch_output_bf16);
  EXPECT_EQ(assembled->residual_output_bf16, mhc_view.residual_a_bf16);
}

TEST(DeepSeekDenseMhcLayerSubmissionAssemblerTest,
     RejectsCrossContextResourcesAndInvalidResidualCarrier) {
  DenseLayerAllocator allocator(2);
  auto attention = DeepSeekAttentionProjectionDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  auto foreign_mhc = DeepSeekMhcDeviceResources::Allocate(
      allocator, 8, 92, 2).value();
  EXPECT_FALSE(DeepSeekDenseMhcLayerSubmissionAssembler::Assemble(
                   7, attention_weights(), mhc_weights(), attention,
                   foreign_mhc, 4, foreign_mhc.view().residual_a_bf16,
                   0x301, 104, 0x401)
                   .ok());
  auto mhc = DeepSeekMhcDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  EXPECT_FALSE(DeepSeekDenseMhcLayerSubmissionAssembler::Assemble(
                   7, attention_weights(), mhc_weights(), attention, mhc,
                   4, mhc.view().residual_b_bf16, 0x301, 104, 0x401)
                   .ok());
  auto foreign_weights = mhc_weights();
  foreign_weights.generation = 72;
  EXPECT_FALSE(DeepSeekDenseMhcLayerSubmissionAssembler::Assemble(
                   7, attention_weights(), foreign_weights, attention, mhc,
                   4, mhc.view().residual_a_bf16, 0x301, 104, 0x401)
                   .ok());
}

}  // namespace
}  // namespace pih
