#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pih/model/deepseek_rank_spawn_coordinator.h"
#include "pih/model/deepseek_rank_spawn_resource_collector.h"
#include "pih/model/deepseek_rank_startup_barrier.h"
#include "pih/model/deepseek_runtime_artifact_evidence.h"
#include "pih/platform/linux/linux_deepseek_rank_artifact_metadata_transfer_controller_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_artifact_transfer_controller_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_process_driver.h"
#include "pih/platform/linux/linux_deepseek_rank_post_mapping_resource_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_materialization_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_spawn_authority_probe.h"

namespace pih {

class LinuxDeepSeekRankSpawnController final {
 public:
  static constexpr std::uint64_t kPostMappingResourceWaitTimeoutNs =
      UINT64_C(30'000'000'000);
  static constexpr std::uint64_t kMaterializationTransactionTimeoutNs =
      UINT64_C(1'800'000'000'000);
  static Result<std::unique_ptr<LinuxDeepSeekRankSpawnController>> Create(
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan plan,
      std::span<const DeepSeekRankPostExecResourcePlan> post_exec_plans,
      std::uint64_t post_exec_deadline_ns,
      DeepSeekRankCapacityPlanInstance capacity_plan_instance,
      std::string worker_executable,
      std::vector<std::string> fixed_arguments,
      int inherited_controller_pidfd,
      std::uint32_t maximum_usage_samples = 4,
      std::uint64_t maximum_metadata_reassembly_bytes =
          kDeepSeekRankArtifactMetadataBlobMaximumBytes);

  LinuxDeepSeekRankSpawnController(
      const LinuxDeepSeekRankSpawnController&) = delete;
  LinuxDeepSeekRankSpawnController& operator=(
      const LinuxDeepSeekRankSpawnController&) = delete;
  LinuxDeepSeekRankSpawnController(
      LinuxDeepSeekRankSpawnController&&) = delete;
  LinuxDeepSeekRankSpawnController& operator=(
      LinuxDeepSeekRankSpawnController&&) = delete;

  Status launch();
  Status advance_startup(std::uint64_t now_ns);
  Status begin_artifact_transfer(
      const RuntimeProfileSupervisorBootstrapManifest& profile_bootstrap,
      const RuntimeProfileReadinessReceipt& profile_readiness,
      const DeepSeekPipelinePlan& pipeline,
      const DeepSeekPipelineCapacity& pipeline_capacity,
      std::uint64_t model_startup_deadline_ns,
      DeepSeekRuntimeArtifactAdmissionBinding artifact_binding,
      DeepSeekRankArtifactHandoffPlan handoff,
      std::span<const DeepSeekRankMaterializationAllocationAuthority>
          allocation_authorities);
  Status advance_artifact_transfer();
  [[nodiscard]] bool startup_ready() const noexcept;
  [[nodiscard]] bool artifact_transfer_attempted() const noexcept {
    return artifact_transfer_attempted_;
  }
  [[nodiscard]] bool artifact_transfer_started() const noexcept {
    return artifact_transfer_transaction_ != nullptr ||
           artifact_metadata_transfer_transaction_ != nullptr ||
           materialization_warmup_coordinator_ != nullptr;
  }
  [[nodiscard]] bool artifact_transfer_complete() const noexcept {
    return materialization_warmup_coordinator_ != nullptr &&
           materialization_warmup_coordinator_->complete();
  }
  [[nodiscard]] const Sha256Digest* artifact_transfer_transaction_root()
      const noexcept {
    if (artifact_metadata_transfer_transaction_) {
      return &artifact_metadata_transfer_transaction_
                  ->descriptor_transaction_root();
    }
    return artifact_transfer_transaction_
               ? &artifact_transfer_transaction_->transaction_root()
               : nullptr;
  }
  [[nodiscard]] const Sha256Digest*
  artifact_metadata_transfer_transaction_root() const noexcept {
    return artifact_metadata_transfer_transaction_
               ? &artifact_metadata_transfer_transaction_->transaction_root()
               : nullptr;
  }
  [[nodiscard]] const DeepSeekRankPostExecResourceSeal* resource_seal()
      const noexcept;
  [[nodiscard]] const DeepSeekRankPostMappingResourceSeal*
  post_mapping_resource_seal() const noexcept {
    return post_mapping_resource_coordinator_
               ? post_mapping_resource_coordinator_->seal()
               : nullptr;
  }
  [[nodiscard]] const DeepSeekRankMaterializationWarmSeal*
  materialization_warm_seal() const noexcept {
    return materialization_warmup_coordinator_
               ? materialization_warmup_coordinator_->seal()
               : nullptr;
  }
  [[nodiscard]] bool launch_attempted() const noexcept;
  [[nodiscard]] const Sha256Digest& preflight_receipt_root() const noexcept;
  [[nodiscard]] const Sha256Digest& spawn_authorization_root() const noexcept;

 private:
  LinuxDeepSeekRankSpawnController(
      std::shared_ptr<const RuntimeEngineAdmission> capacity_authority_anchor,
      std::uint64_t controller_process_identity,
      std::vector<DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan> resource_plans,
      std::unique_ptr<LinuxDeepSeekRankProcessDriver> driver,
      std::unique_ptr<LinuxDeepSeekRankSpawnAuthorityProbe> probe,
      std::unique_ptr<StableDeepSeekRankSpawnResourceCollector> collector,
      std::unique_ptr<DeepSeekRankSpawnCoordinator> coordinator,
      std::unique_ptr<DeepSeekRankStartupBarrier> startup_barrier) noexcept;

  // Declared first so authority leases outlive driver cleanup.
  std::shared_ptr<const RuntimeEngineAdmission> capacity_authority_anchor_;
  std::uint64_t controller_process_identity_ = 0;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan spawn_plan_;
  std::vector<DeepSeekRankPostExecResourcePlan> resource_plans_;
  std::unique_ptr<LinuxDeepSeekRankProcessDriver> driver_;
  std::unique_ptr<LinuxDeepSeekRankSpawnAuthorityProbe> probe_;
  std::unique_ptr<StableDeepSeekRankSpawnResourceCollector> collector_;
  std::unique_ptr<DeepSeekRankSpawnCoordinator> coordinator_;
  std::unique_ptr<DeepSeekRankStartupBarrier> startup_barrier_;
  std::unique_ptr<
      LinuxDeepSeekRankArtifactTransferControllerOperations>
      artifact_transfer_operations_;
  std::unique_ptr<DeepSeekRankArtifactTransferTransaction>
      artifact_transfer_transaction_;
  std::unique_ptr<
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations>
      artifact_metadata_transfer_operations_;
  std::unique_ptr<DeepSeekRankArtifactMetadataTransferTransaction>
      artifact_metadata_transfer_transaction_;
  std::unique_ptr<
      LinuxDeepSeekRankPostMappingResourceControllerOperations>
      post_mapping_resource_operations_;
  std::unique_ptr<DeepSeekRankPostMappingResourceCoordinator>
      post_mapping_resource_coordinator_;
  std::uint64_t post_mapping_resource_deadline_ns_ = 0;
  std::unique_ptr<
      LinuxDeepSeekRankMaterializationControllerOperations>
      materialization_grant_operations_;
  std::unique_ptr<DeepSeekRankMaterializationGrantCoordinator>
      materialization_grant_coordinator_;
  std::unique_ptr<DeepSeekRankMaterializationWarmupCoordinator>
      materialization_warmup_coordinator_;
  std::uint64_t materialization_grant_deadline_ns_ = 0;
  std::vector<DeepSeekRankMaterializationAllocationAuthority>
      materialization_allocation_authorities_;
  bool artifact_transfer_attempted_ = false;
};

}  // namespace pih
