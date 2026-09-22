#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string_view>

#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_resident_weight_arena.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

class DeepSeekResidentExpertBindings final {
 public:
  static constexpr std::uint32_t kExpertCount = 256;
  using Resolver = std::function<Result<TensorView>(std::string_view)>;

  static Result<DeepSeekResidentExpertBindings> Resolve(
      DeepSeekStagePlan stage, const Resolver& resolver);
  static Result<DeepSeekResidentExpertBindings> Resolve(
      DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena);

  [[nodiscard]] bool contains(std::uint32_t layer) const noexcept;
  [[nodiscard]] const DeepSeekExpertBundleDeviceView& expert(
      std::uint32_t layer, std::uint32_t id) const;
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

 private:
  using LayerBindings =
      std::array<DeepSeekExpertBundleDeviceView, kExpertCount>;
  std::map<std::uint32_t, LayerBindings> layers_;
  std::uint64_t generation_ = 0;
};

}  // namespace pih
