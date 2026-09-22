#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_handoff_plan.h"
#include "pih/model/deepseek_rank_artifact_transfer_manifest.h"
#include "pih/model/deepseek_rank_model_startup_plan.h"
#include "pih/model/deepseek_runtime_artifact_evidence.h"

namespace pih {

class DeepSeekRankArtifactTransferTransaction;
class DeepSeekRankArtifactMetadataTransferTransaction;

inline constexpr std::string_view kDeepSeekRankArtifactTransferPlanAbi =
    "pih_deepseek_rank_artifact_transfer_plan_v1";

// Exclusive controller-side join for the exact profile, process generation,
// pipeline ownership and descriptor inventory. It owns every antecedent but
// intentionally exposes no descriptor, send, acknowledgement or ready API.
class DeepSeekRankArtifactTransferPlan final {
 public:
  static Result<DeepSeekRankArtifactTransferPlan> Compile(
      DeepSeekRankModelStartupPlan model_startup,
      DeepSeekRuntimeArtifactAdmissionBinding artifact_binding,
      DeepSeekRankArtifactHandoffPlan handoff);

  DeepSeekRankArtifactTransferPlan(
      const DeepSeekRankArtifactTransferPlan&) = delete;
  DeepSeekRankArtifactTransferPlan& operator=(
      const DeepSeekRankArtifactTransferPlan&) = delete;
  DeepSeekRankArtifactTransferPlan(
      DeepSeekRankArtifactTransferPlan&&) noexcept = default;
  DeepSeekRankArtifactTransferPlan& operator=(
      DeepSeekRankArtifactTransferPlan&&) noexcept = default;

  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return model_startup_.world_size();
  }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_binding_.artifact_root();
  }
  [[nodiscard]] const Sha256Digest& artifact_binding_root() const noexcept {
    return artifact_binding_.binding_root();
  }
  [[nodiscard]] const Sha256Digest& plan_root() const noexcept {
    return plan_root_;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return model_startup_.admission_anchor_ != nullptr &&
           model_startup_.admission_anchor_->dspark_enabled();
  }
  [[nodiscard]] const DeepSeekRankArtifactTransferManifest& rank_manifest(
      std::uint32_t rank) const;
  [[nodiscard]] std::array<std::byte,
                           kDeepSeekRankArtifactTransferManifestBytes>
  encode_rank_manifest(std::uint32_t rank) const;
  [[nodiscard]] bool owns_all_antecedents() const noexcept {
    return model_startup_.admission_authority_retained() &&
           artifact_binding_.admission_authority_retained() &&
           manifests_.size() == model_startup_.world_size();
  }

 private:
  friend class DeepSeekRankArtifactTransferTransaction;
  friend class DeepSeekRankArtifactMetadataTransferTransaction;

  DeepSeekRankArtifactTransferPlan(
      DeepSeekRankModelStartupPlan model_startup,
      DeepSeekRuntimeArtifactAdmissionBinding artifact_binding,
      DeepSeekRankArtifactHandoffPlan handoff,
      std::vector<DeepSeekRankArtifactTransferManifest> manifests,
      Sha256Digest plan_root) noexcept;

  DeepSeekRankModelStartupPlan model_startup_;
  DeepSeekRuntimeArtifactAdmissionBinding artifact_binding_;
  DeepSeekRankArtifactHandoffPlan handoff_;
  std::vector<DeepSeekRankArtifactTransferManifest> manifests_;
  Sha256Digest plan_root_{};
};

}  // namespace pih
