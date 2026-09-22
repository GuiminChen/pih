#include "pih/model/deepseek_dspark_resident_expert_bindings.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace pih { namespace {

Result<TensorView> tensor(std::string_view name, bool wrong = false,
                          std::vector<std::string>* requested = nullptr) {
  if (requested != nullptr) requested->emplace_back(name);
  const bool scale = name.ends_with(".scale");
  const bool w2 = name.find(".w2.") != std::string_view::npos;
  const std::array<std::int64_t, 2> shape = scale
      ? std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 64 : 128}
      : std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 1024 : 2048};
  const auto hash = std::hash<std::string_view>{}(name);
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000U + (hash & 0x0fffffffU)),
      wrong ? DType::kBFloat16
            : (scale ? DType::kFloat8E8M0 : DType::kInt8),
      shape, {}, Device::Create(DeviceType::kCuda, 0).value(), 17);
}

DeepSeekStagePlan stage() { return {0, {0, 42}, true, true, true}; }

TEST(DeepSeekDsparkResidentExpertBindingsTest,
     ResolvesAllThreeBy256FixedResidentBundles) {
  std::vector<std::string> requested;
  auto bindings = DeepSeekDsparkResidentExpertBindings::Resolve(
      stage(), [&requested](std::string_view name) {
        return tensor(name, false, &requested);
      });
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  EXPECT_EQ(bindings->generation(), 17U);
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto dspark_stage = deepseek_dspark_stage_id(index).value();
    EXPECT_NE(bindings->expert(dspark_stage, 0).w1.packed.address, 0U);
    EXPECT_EQ(bindings->expert(dspark_stage, 255).w2.scales.bytes,
              DeepSeekExpertBundleLayout::kScaleBytesPerMatrix);
  }
  EXPECT_EQ(requested.size(), 3U * 256U * 3U * 2U);
  EXPECT_TRUE(std::ranges::any_of(requested, [](const auto& name) {
    return name.starts_with("mtp.0.ffn.experts.");
  }));
  EXPECT_TRUE(std::ranges::any_of(requested, [](const auto& name) {
    return name.starts_with("mtp.1.ffn.experts.");
  }));
  EXPECT_TRUE(std::ranges::any_of(requested, [](const auto& name) {
    return name.starts_with("mtp.2.ffn.experts.");
  }));
}

TEST(DeepSeekDsparkResidentExpertBindingsTest,
     RejectsWrongTensorAbiAndNonOwningStage) {
  EXPECT_FALSE(DeepSeekDsparkResidentExpertBindings::Resolve(
      stage(), [](std::string_view name) { return tensor(name, true); }).ok());
  auto non_owner = stage();
  non_owner.owns_dspark = false;
  EXPECT_FALSE(DeepSeekDsparkResidentExpertBindings::Resolve(
      non_owner, [](std::string_view name) { return tensor(name); }).ok());
}

} }  // namespace pih
