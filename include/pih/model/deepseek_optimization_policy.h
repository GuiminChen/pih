#pragma once

#include <cstdint>

#include "pih/core/sha256.h"
#include "pih/model/deepseek_gpu_architecture.h"

namespace pih {

enum class DeepSeekAttentionPhysicalLayoutVersion : std::uint8_t {
  kBaseline,
  kV1,
};

enum class DeepSeekFusedKernelSet : std::uint8_t {
  kUnfusedControl,
  kVerifiedV1,
};

struct DeepSeekOptimizationProfile final {
  DeepSeekGpuArchitecture architecture = DeepSeekGpuArchitecture::kSm89;
  std::uint32_t world_size = 0;
  bool chunked_prefill = false;
  DeepSeekAttentionPhysicalLayoutVersion attention_layout =
      DeepSeekAttentionPhysicalLayoutVersion::kBaseline;
  DeepSeekFusedKernelSet fused_kernel_set =
      DeepSeekFusedKernelSet::kUnfusedControl;
  bool observed_miss_debit = false;
  bool dspark = false;
  bool cuda_graph = false;
  bool expert_locality_reorder = false;
};

class DeepSeekOptimizationPolicy final {
 public:
  static Result<DeepSeekOptimizationPolicy> Create(
      DeepSeekOptimizationProfile profile);

  [[nodiscard]] const Sha256Digest& identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] DeepSeekGpuArchitecture architecture() const noexcept {
    return profile_.architecture;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return profile_.world_size;
  }
  [[nodiscard]] bool chunked_prefill() const noexcept {
    return profile_.chunked_prefill;
  }
  [[nodiscard]] DeepSeekAttentionPhysicalLayoutVersion attention_layout()
      const noexcept { return profile_.attention_layout; }
  [[nodiscard]] DeepSeekFusedKernelSet fused_kernel_set() const noexcept {
    return profile_.fused_kernel_set;
  }
  [[nodiscard]] bool observed_miss_debit() const noexcept {
    return profile_.observed_miss_debit;
  }
  [[nodiscard]] bool dspark() const noexcept { return profile_.dspark; }
  [[nodiscard]] bool cuda_graph() const noexcept {
    return profile_.cuda_graph;
  }
  [[nodiscard]] bool expert_locality_reorder() const noexcept {
    return profile_.expert_locality_reorder;
  }

 private:
  DeepSeekOptimizationPolicy(DeepSeekOptimizationProfile profile,
                             Sha256Digest identity) noexcept
      : profile_(profile), identity_(identity) {}

  DeepSeekOptimizationProfile profile_;
  Sha256Digest identity_{};
};

}  // namespace pih
