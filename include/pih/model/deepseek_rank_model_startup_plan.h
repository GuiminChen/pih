#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_pipeline_identity.h"
#include "pih/model/deepseek_rank_post_exec_resource_receipt.h"
#include "pih/model/runtime_profile_readiness_receipt.h"

namespace pih {

class DeepSeekRankArtifactTransferPlan;

inline constexpr std::string_view kDeepSeekRankModelStartupPlanAbi =
    "pih_deepseek_rank_model_startup_plan_v1";

struct DeepSeekRankModelStartupSeed final {
  std::uint32_t rank = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  std::uint64_t physical_device_identity = 0;
  Sha256Digest physical_device_uuid_commitment{};
  std::int32_t startup_device_ordinal = -1;
  Sha256Digest seed_root{};
};

// Immutable controller-side join between the admitted model profile, exact
// exec generation, all-rank post-exec resource seal, and canonical pipeline
// geometry. This is deliberately not a model-load authority: artifact
// descriptor handoff and NCCL/CUDA readiness must still be joined later.
class DeepSeekRankModelStartupPlan final {
 public:
  static Result<DeepSeekRankModelStartupPlan> Compile(
      const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
      const RuntimeProfileReadinessReceipt& profile_readiness,
      const DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankPostExecResourceSeal& resource_seal,
      const DeepSeekPipelinePlan& pipeline,
      const DeepSeekPipelineCapacity& capacity,
      std::uint64_t model_startup_deadline_ns);

  DeepSeekRankModelStartupPlan(const DeepSeekRankModelStartupPlan&) = delete;
  DeepSeekRankModelStartupPlan& operator=(
      const DeepSeekRankModelStartupPlan&) = delete;
  DeepSeekRankModelStartupPlan(DeepSeekRankModelStartupPlan&&) noexcept =
      default;
  DeepSeekRankModelStartupPlan& operator=(
      DeepSeekRankModelStartupPlan&&) noexcept = default;

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint64_t model_startup_deadline_ns() const noexcept {
    return model_startup_deadline_ns_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& profile_readiness_root() const noexcept {
    return profile_readiness_root_;
  }
  [[nodiscard]] const Sha256Digest& post_exec_resource_seal_root()
      const noexcept {
    return post_exec_resource_seal_root_;
  }
  [[nodiscard]] const Sha256Digest& pipeline_plan_root() const noexcept {
    return pipeline_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& pipeline_capacity_root() const noexcept {
    return pipeline_capacity_root_;
  }
  [[nodiscard]] const Sha256Digest& plan_root() const noexcept {
    return plan_root_;
  }
  [[nodiscard]] const Sha256Digest& rank_seed_root(
      std::uint32_t rank) const;
  [[nodiscard]] const DeepSeekRankModelStartupSeed& rank_seed(
      std::uint32_t rank) const;
  [[nodiscard]] bool admission_authority_retained() const noexcept {
    return admission_anchor_ != nullptr && admission_anchor_->production_ready();
  }

 private:
  friend class DeepSeekRankArtifactTransferPlan;
  DeepSeekRankModelStartupPlan(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, std::uint64_t model_startup_deadline_ns,
      Sha256Digest manifest_root,
      Sha256Digest capacity_plan_instance_root,
      Sha256Digest profile_readiness_root,
      Sha256Digest post_exec_resource_seal_root,
      Sha256Digest pipeline_plan_root,
      Sha256Digest pipeline_capacity_root, Sha256Digest plan_root,
      std::vector<DeepSeekRankModelStartupSeed> rank_seeds,
      std::shared_ptr<const RuntimeEngineAdmission> admission_anchor) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint64_t model_startup_deadline_ns_ = 0;
  Sha256Digest manifest_root_{};
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest profile_readiness_root_{};
  Sha256Digest post_exec_resource_seal_root_{};
  Sha256Digest pipeline_plan_root_{};
  Sha256Digest pipeline_capacity_root_{};
  Sha256Digest plan_root_{};
  std::vector<DeepSeekRankModelStartupSeed> rank_seeds_;
  std::shared_ptr<const RuntimeEngineAdmission> admission_anchor_;
};

}  // namespace pih
