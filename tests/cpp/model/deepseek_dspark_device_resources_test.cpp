#include "pih/model/deepseek_dspark_device_resources.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class DsparkDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("DSpark device");
    return Allocation{data, bytes, alignment, 31,
                      Device::Create(DeviceType::kCuda, 1).value()};
  }
  void deallocate(Allocation value) noexcept override { _aligned_free(value.data); }
};

TEST(DeepSeekDsparkDeviceResourcesTest, AllocatesSingleSequenceDraftWorkspace) {
  DsparkDeviceAllocator allocator;
  DeepSeekStagePlan stage{0, {0, 42}, true, true, true};
  auto resources = DeepSeekDsparkDeviceResources::Allocate(
      stage, 4096, allocator, 77, 1);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  const auto view = resources->view();
  EXPECT_NE(view.draft_token_ids_u32, 0U);
  EXPECT_NE(view.draft_input_hc_bf16, 0U);
  EXPECT_NE(view.main_quant_fp8, 0U);
  EXPECT_NE(view.main_quant_scale_ue8m0, 0U);
  EXPECT_NE(view.main_projected_bf16, 0U);
  EXPECT_NE(view.raw_logits_f32, 0U);
  EXPECT_NE(view.biased_logits_f32, 0U);
  EXPECT_NE(view.markov_embeddings_bf16, 0U);
  EXPECT_NE(view.confidence_f32, 0U);
  EXPECT_NE(view.error_flag_u32, 0U);
  EXPECT_NE(view.target_hidden_bf16, 0U);
  EXPECT_NE(view.prefill_kv_bf16, 0U);
  EXPECT_NE(view.stage_residual_a_bf16, 0U);
  EXPECT_NE(view.stage_residual_b_bf16, 0U);
  EXPECT_NE(view.draft_positions_u32, 0U);
  EXPECT_NE(view.stage_residual_a_bf16, view.stage_residual_b_bf16);
  EXPECT_EQ(resources->maximum_tokens(), 4096U);
  EXPECT_GT(resources->backing_bytes(),
            4096ULL * 3ULL * 4096ULL * sizeof(std::uint16_t));
}

TEST(DeepSeekDsparkDeviceResourcesTest, NonDsparkStageAllocatesNothing) {
  DsparkDeviceAllocator allocator;
  DeepSeekStagePlan stage{0, {0, 42}, true, true, false};
  auto resources=DeepSeekDsparkDeviceResources::Allocate(
      stage, 8, allocator, 77, 1);
  ASSERT_TRUE(resources.ok());
  EXPECT_EQ(resources->backing_bytes(),0U);
}

} }  // namespace pih
