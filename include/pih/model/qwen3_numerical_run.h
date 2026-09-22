#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "pih/core/sha256.h"
#include "pih/model/qwen3_numerical_comparison.h"

namespace pih {

enum class QwenTargetGpu : std::uint8_t { kRtx4090D = 0, kH100Pcie80Gb };

class QwenNumericalRunIdentity final {
 public:
  static constexpr std::uint64_t kMaximumManifestBytes = 2048;
  static Result<QwenNumericalRunIdentity> Parse(std::string_view json);

  [[nodiscard]] QwenTargetGpu target_gpu() const noexcept { return target_gpu_; }
  [[nodiscard]] std::uint32_t target_sm() const noexcept { return target_sm_; }
  [[nodiscard]] std::uint64_t run_generation() const noexcept {
    return run_generation_;
  }
  Result<Sha256Digest> semantic_digest() const;
  friend bool operator==(const QwenNumericalRunIdentity&,
                         const QwenNumericalRunIdentity&) = default;

 private:
  QwenNumericalRunIdentity(Sha256Digest model, Sha256Digest fixture,
                           Sha256Digest tolerance, Sha256Digest kernels,
                           Sha256Digest build, Sha256Digest environment,
                           QwenTargetGpu gpu, std::uint32_t sm,
                           std::uint64_t generation)
      : model_(model), fixture_(fixture), tolerance_(tolerance),
        kernels_(kernels), build_(build), environment_(environment),
        target_gpu_(gpu), target_sm_(sm), run_generation_(generation) {}

  Sha256Digest model_;
  Sha256Digest fixture_;
  Sha256Digest tolerance_;
  Sha256Digest kernels_;
  Sha256Digest build_;
  Sha256Digest environment_;
  QwenTargetGpu target_gpu_;
  std::uint32_t target_sm_;
  std::uint64_t run_generation_;
};

Result<QwenNumericalReport> compare_qwen_numerical_run(
    const QwenNumericalRunIdentity& reference_identity,
    std::span<const float> reference,
    const QwenNumericalRunIdentity& candidate_identity,
    std::span<const float> candidate, const QwenNumericalPolicy& policy);

}  // namespace pih
