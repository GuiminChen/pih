#include "pih/model/deepseek_endpoint_device_resources.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class EndpointDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("endpoint device");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 2).value()};
  }
  void deallocate(Allocation value) noexcept override { _aligned_free(value.data); }
 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekEndpointDeviceResourcesTest, AllocatesStageExactEndpointSpans) {
  EndpointDeviceAllocator allocator;
  DeepSeekStagePlan all{0, {0, 42}, true, true, true};
  auto resources = DeepSeekEndpointDeviceResources::Allocate(
      all, 8, allocator, 77, 2);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  const auto view = resources->view();
  EXPECT_NE(view.embedding_output_hc_bf16, 0U);
  EXPECT_NE(view.head_output_bf16, 0U);
  EXPECT_NE(view.normalized_bf16, 0U);
  EXPECT_NE(view.logits_f32, 0U);
  EXPECT_NE(view.sampled_token_u32, 0U);
  EXPECT_NE(view.selected_logprob_f32, 0U);
  EXPECT_NE(view.top_logprobs_ids_u32, 0U);
  EXPECT_NE(view.top_logprobs_f32, 0U);
  EXPECT_NE(view.rng_word_u32, 0U);
  EXPECT_NE(view.sampling_workspace_values_f32, 0U);
  EXPECT_NE(view.sampling_workspace_ids_u32, 0U);
  EXPECT_NE(view.error_flag_u32, 0U);
  EXPECT_EQ(resources->maximum_tokens(), 8U);
  EXPECT_GE(resources->backing_bytes(),
            8U * 4U * 4096U * sizeof(std::uint16_t) +
                2U * 4096U * sizeof(std::uint16_t) +
                2U * 129280U * sizeof(float) +
                129280U * sizeof(std::uint32_t) +
                3U * sizeof(std::uint32_t));
}

TEST(DeepSeekEndpointDeviceResourcesTest, MiddleStageAllocatesNothing) {
  EndpointDeviceAllocator allocator;
  DeepSeekStagePlan middle{1, {12, 24}, false, false, false};
  auto resources = DeepSeekEndpointDeviceResources::Allocate(
      middle, 8, allocator, 77, 2);
  ASSERT_TRUE(resources.ok());
  EXPECT_EQ(resources->backing_bytes(), 0U);
  EXPECT_EQ(resources->view().embedding_output_hc_bf16, 0U);
}

}  // namespace
}  // namespace pih
