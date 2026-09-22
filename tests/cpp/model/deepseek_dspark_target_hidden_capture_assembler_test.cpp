#include "pih/model/deepseek_dspark_target_hidden_capture_assembler.h"

#include <gtest/gtest.h>

#include "pih/model/deepseek_mhc_submission_assembler.h"

namespace pih { namespace {

class TargetHiddenDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("target hidden");
    return Allocation{data, bytes, alignment, 73,
                      Device::Create(DeviceType::kCuda, 1).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
};

std::vector<DeepSeekDenseMhcLayerSubmissionInput> target_stage_layers() {
  std::vector<DeepSeekDenseMhcLayerSubmissionInput> layers;
  const DeepSeekMhcWeightBindings weights{
      0x401, 0x402, 0x403, 0x404,
      0x405, 0x406, 0x407, 0x408, 79};
  const DeepSeekMhcDeviceView workspace{
      0x501, 0x502, 0x503, 0x504, 0x505, 0x506};
  for (std::uint32_t layer = 33; layer <= 42; ++layer) {
    auto mhc = DeepSeekMhcSubmissionAssembler::Assemble(
        layer, weights, workspace, 8, 4, 0x501, 0x601, 0x602, 0x603);
    EXPECT_TRUE(mhc.ok()) << mhc.status().message();
    DeepSeekDenseMhcLayerSubmissionInput input;
    input.layer = layer;
    input.mhc_attention = mhc->attention;
    input.mhc_feed_forward = mhc->feed_forward;
    layers.push_back(input);
  }
  return layers;
}

TEST(DeepSeekDsparkTargetHiddenCaptureAssemblerTest,
     BindsExactlyThreePostMhcTargetLayerMeans) {
  TargetHiddenDeviceAllocator allocator;
  const DeepSeekStagePlan stage{3, {33, 42}, false, true, true};
  auto resources = DeepSeekDsparkDeviceResources::Allocate(
      stage, 8, allocator, 77, 1);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  auto layers = target_stage_layers();

  auto status = DeepSeekDsparkTargetHiddenCaptureAssembler::Bind(
      stage, *resources, 4, layers);
  ASSERT_TRUE(status.ok()) << status.message();
  for (std::uint32_t layer = 33; layer < 40; ++layer) {
    EXPECT_EQ(layers[layer - 33].mhc_feed_forward.target_hidden_bf16, 0U);
  }
  for (std::uint32_t layer = 40; layer <= 42; ++layer) {
    const auto& submission = layers[layer - 33].mhc_feed_forward;
    EXPECT_EQ(submission.target_hidden_bf16,
              resources->view().target_hidden_bf16);
    EXPECT_EQ(submission.target_stage_index, layer - 40);
    EXPECT_TRUE(validate_deepseek_mhc_sequence_submission(submission).ok());
  }
}

TEST(DeepSeekDsparkTargetHiddenCaptureAssemblerTest,
     FailedBindingDoesNotPublishPartialTargetTaps) {
  TargetHiddenDeviceAllocator allocator;
  const DeepSeekStagePlan stage{3, {33, 42}, false, true, true};
  auto resources = DeepSeekDsparkDeviceResources::Allocate(
      stage, 8, allocator, 77, 1).value();
  auto layers = target_stage_layers();
  layers[41 - 33].mhc_feed_forward.token_count = 3;

  EXPECT_FALSE(DeepSeekDsparkTargetHiddenCaptureAssembler::Bind(
      stage, resources, 4, layers).ok());
  for (const auto& layer : layers) {
    EXPECT_EQ(layer.mhc_feed_forward.target_hidden_bf16, 0U);
    EXPECT_EQ(layer.mhc_feed_forward.target_stage_index, 0U);
  }
}

}}  // namespace pih::<anonymous>
