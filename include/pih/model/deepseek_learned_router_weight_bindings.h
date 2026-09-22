#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_resident_weight_arena.h"

namespace pih {

struct DeepSeekLearnedRouterWeightBinding final {
  std::uint32_t layer = 0;
  std::uintptr_t weight_bf16 = 0;
};

class DeepSeekLearnedRouterWeightBindings final {
 public:
  using Resolver = std::function<Result<TensorView>(std::string_view)>;

  static Result<DeepSeekLearnedRouterWeightBindings> Resolve(
      DeepSeekStageRange owned_layers, const Resolver& resolver);
  static Result<DeepSeekLearnedRouterWeightBindings> Resolve(
      DeepSeekStageRange owned_layers,
      const DeepSeekResidentWeightArena& arena);

  [[nodiscard]] std::size_t size() const noexcept { return bindings_.size(); }
  [[nodiscard]] const DeepSeekLearnedRouterWeightBinding& at(
      std::size_t index) const { return bindings_.at(index); }
  [[nodiscard]] const std::vector<DeepSeekLearnedRouterWeightBinding>&
  bindings() const noexcept { return bindings_; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

 private:
  std::vector<DeepSeekLearnedRouterWeightBinding> bindings_;
  std::uint64_t generation_ = 0;
};

}  // namespace pih
