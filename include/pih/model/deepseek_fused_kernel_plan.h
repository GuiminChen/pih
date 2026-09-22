#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "pih/model/deepseek_optimization_policy.h"

namespace pih {

struct DeepSeekFusedKernelManifest final {
  std::string logical_id;
  DeepSeekGpuArchitecture architecture = DeepSeekGpuArchitecture::kSm89;
  std::uint32_t maximum_tokens = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t routed_experts = 0;
  std::uint32_t activated_experts = 0;
  Sha256Digest cubin_digest{};
  Sha256Digest parameter_abi_root{};
};

class DeepSeekFusedKernelPlan final {
 public:
  static Result<DeepSeekFusedKernelPlan> Compile(
      const DeepSeekOptimizationPolicy& policy, std::uint32_t token_count,
      const DeepSeekFusedKernelManifest* manifest,
      const Sha256Digest* observed_cubin_digest);
  static Result<Sha256Digest> ExpectedAbiRoot(
      DeepSeekGpuArchitecture architecture);

  [[nodiscard]] bool fused() const noexcept { return fused_; }
  [[nodiscard]] std::string_view logical_id() const noexcept {
    return logical_id_;
  }
  [[nodiscard]] const Sha256Digest& cubin_digest() const noexcept {
    return cubin_digest_;
  }
  [[nodiscard]] const Sha256Digest& identity() const noexcept {
    return identity_;
  }

 private:
  bool fused_ = false;
  std::string logical_id_;
  Sha256Digest cubin_digest_{};
  Sha256Digest identity_{};
};

}  // namespace pih
