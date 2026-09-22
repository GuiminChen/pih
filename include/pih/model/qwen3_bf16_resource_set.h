#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "pih/core/result.h"
#include "pih/core/tensor_view.h"
#include "pih/model/qwen3_bf16_activation_route.h"
#include "pih/model/qwen3_manifest.h"

namespace pih {

struct QwenBf16SlotResource final {
  QwenBf16ActivationSlot slot;
  TensorView view;
};

class QwenBf16ResourceSet final {
 public:
  static constexpr std::size_t kSlotCount =
      static_cast<std::size_t>(QwenBf16ActivationSlot::kCount);

  static Result<QwenBf16ResourceSet> Create(
      std::uint64_t request_generation, std::int32_t owning_rank,
      std::span<const QwenBf16SlotResource> resources);

  [[nodiscard]] std::uint64_t request_generation() const noexcept {
    return request_generation_;
  }
  [[nodiscard]] std::int32_t owning_rank() const noexcept {
    return owning_rank_;
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return kSlotCount;
  }
  Result<TensorView> view(QwenBf16ActivationSlot slot) const;

 private:
  std::uint64_t request_generation_ = 0;
  std::int32_t owning_rank_ = -1;
  std::array<std::optional<TensorView>, kSlotCount> resources_{};
};

class QwenBf16WeightResourceSet final {
 public:
  static constexpr std::size_t kWeightCount =
      Qwen3Manifest::kOfficialTensorCount;

  static Result<QwenBf16WeightResourceSet> Create(
      std::int32_t owning_rank, std::span<const TensorView> weights);

  [[nodiscard]] std::int32_t owning_rank() const noexcept {
    return owning_rank_;
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return kWeightCount;
  }
  Result<TensorView> view(std::size_t tensor_index) const;

 private:
  std::int32_t owning_rank_ = -1;
  std::array<std::optional<TensorView>, kWeightCount> weights_{};
};

}  // namespace pih
