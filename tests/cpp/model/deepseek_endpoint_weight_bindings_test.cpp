#include "pih/model/deepseek_endpoint_weight_bindings.h"

#include <gtest/gtest.h>

#include <map>

namespace pih {
namespace {

using Shape = std::vector<std::int64_t>;

Result<TensorView> endpoint_tensor(std::string_view name) {
  const std::map<std::string, std::pair<DType, Shape>, std::less<>> tensors{
      {"embed.weight", {DType::kBFloat16, {129280, 4096}}},
      {"hc_head_fn", {DType::kFloat32, {4, 16384}}},
      {"hc_head_scale", {DType::kFloat32, {1}}},
      {"hc_head_base", {DType::kFloat32, {4}}},
      {"norm.weight", {DType::kBFloat16, {4096}}},
      {"head.weight", {DType::kBFloat16, {129280, 4096}}}};
  const auto found = tensors.find(name);
  if (found == tensors.end()) return Status::InvalidArgument("missing endpoint tensor");
  const auto address = static_cast<std::uintptr_t>(
      0x10000 + std::distance(tensors.begin(), found) * 0x10000);
  return TensorView::Create(reinterpret_cast<void*>(address),
                            found->second.first, found->second.second, {},
                            Device::Create(DeviceType::kCuda, 0).value(), 23);
}

DeepSeekStagePlan first_stage() { return {0, {0, 20}, true, false, false}; }
DeepSeekStagePlan last_stage() { return {1, {21, 42}, false, true, true}; }

TEST(DeepSeekEndpointWeightBindingsTest, ResolvesOnlyStageOwnedEndpointWeights) {
  auto first = DeepSeekEndpointWeightBindings::Resolve(first_stage(), endpoint_tensor);
  ASSERT_TRUE(first.ok()) << first.status().message();
  EXPECT_NE(first->embedding_weight_bf16, 0U);
  EXPECT_EQ(first->head_weight_bf16, 0U);
  EXPECT_EQ(first->generation, 23U);

  auto last = DeepSeekEndpointWeightBindings::Resolve(last_stage(), endpoint_tensor);
  ASSERT_TRUE(last.ok()) << last.status().message();
  EXPECT_EQ(last->embedding_weight_bf16, 0U);
  EXPECT_NE(last->hc_head_fn_f32, 0U);
  EXPECT_NE(last->hc_head_scale_f32, 0U);
  EXPECT_NE(last->hc_head_base_f32, 0U);
  EXPECT_NE(last->norm_weight_bf16, 0U);
  EXPECT_NE(last->head_weight_bf16, 0U);
}

TEST(DeepSeekEndpointWeightBindingsTest, RejectsWrongCheckpointShape) {
  auto result = DeepSeekEndpointWeightBindings::Resolve(
      last_stage(), [](std::string_view name) -> Result<TensorView> {
        auto view = endpoint_tensor(name);
        if (!view.ok() || name != "head.weight") return view;
        const std::array<std::int64_t, 2> wrong{4096, 129280};
        return TensorView::Create(view->data(), view->dtype(), wrong, {},
                                  view->device(), view->generation());
      });
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace pih
