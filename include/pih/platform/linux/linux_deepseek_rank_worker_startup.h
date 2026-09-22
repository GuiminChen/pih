#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_mapping_owner.h"
#include "pih/model/deepseek_rank_materialization_completion.h"
#include "pih/model/deepseek_rank_scm_rights_ledger.h"
#include "pih/model/deepseek_rank_worker_bootstrap.h"
#include "pih/model/deepseek_rank_worker_materialization_lifecycle.h"
#include "pih/platform/linux/linux_deepseek_rank_post_exec_resource_probe.h"
#include "pih/platform/linux/linux_deepseek_rank_post_mapping_resource_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_materialization_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_artifact_prefault_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_artifact_transfer_receiver_operations.h"
#include "pih/platform/linux/linux_deepseek_rank_worker_bootstrap_runtime.h"
#include "pih/platform/linux/linux_deepseek_rank_worker_handshake_operations.h"

namespace pih {

inline constexpr std::string_view kLinuxDeepSeekRankWorkerStartupAbi =
    "pih_linux_deepseek_rank_worker_startup_v1";

class LinuxDeepSeekRankWorkerStartup final
    : public DeepSeekRankWorkerStartupProtocol {
 public:
  static constexpr std::uint64_t kAuthorityWaitTimeoutNs =
      UINT64_C(30'000'000'000);
  static constexpr std::uint64_t kArtifactTransferWaitTimeoutNs =
      UINT64_C(30'000'000'000);
  static constexpr std::uint64_t kPostMappingResourceWaitTimeoutNs =
      UINT64_C(30'000'000'000);
  // Bounds grant delivery, selected-page prefault and the authorized
  // materialization transaction. Target profiles must tighten this using
  // measured cold-start evidence; V1 never leaves it unbounded.
  static constexpr std::uint64_t kMaterializationTransactionTimeoutNs =
      UINT64_C(1'800'000'000'000);

  static Result<std::unique_ptr<LinuxDeepSeekRankWorkerStartup>> Create(
      std::span<const std::string_view> arguments,
      std::unique_ptr<DeepSeekPhysicalDeviceIdentityProbe> device_probe,
      std::uint32_t maximum_usage_samples = 4);

  LinuxDeepSeekRankWorkerStartup(
      const LinuxDeepSeekRankWorkerStartup&) = delete;
  LinuxDeepSeekRankWorkerStartup& operator=(
      const LinuxDeepSeekRankWorkerStartup&) = delete;
  LinuxDeepSeekRankWorkerStartup(
      LinuxDeepSeekRankWorkerStartup&&) = delete;
  LinuxDeepSeekRankWorkerStartup& operator=(
      LinuxDeepSeekRankWorkerStartup&&) = delete;

  Status advance_resource_barrier();
  Status run_resource_barrier() override;
  Status advance_artifact_transfer();
  Status run_artifact_transfer() override;
  [[nodiscard]] bool resource_reported() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool artifact_received() const noexcept {
    return artifact_mapping_owner_ != nullptr ||
           artifact_mapping_transferred_;
  }
  [[nodiscard]] bool artifact_mapped() const noexcept {
    return artifact_mapping_owner_ != nullptr ||
           artifact_mapping_transferred_;
  }
  [[nodiscard]] bool post_mapping_resource_reported() const noexcept {
    return post_mapping_reporter_ != nullptr &&
           post_mapping_reporter_->reported();
  }
  [[nodiscard]] bool materialization_grant_accepted() const noexcept {
    return materialization_receiver_ != nullptr &&
           materialization_receiver_->complete();
  }
  Result<DeepSeekRankPrefaultedMaterializationInputs>
  prefault_materialization_inputs();
  Status begin_materialization_completion(
      const DeepSeekRankEngineResources& resources,
      const DeepSeekRankMaterializationCompletionObservation& observation);
  Status advance_materialization_completion();
  Status run_materialization_completion() override;
  [[nodiscard]] bool materialization_completion_sent() const noexcept override {
    return materialization_completion_sender_ != nullptr &&
           materialization_completion_sender_->complete();
  }
  [[nodiscard]] const DeepSeekRankArtifactMappingOwner*
  artifact_mapping_owner() const noexcept {
    return artifact_mapping_owner_.get();
  }
  [[nodiscard]] const std::vector<std::string>& application_arguments()
      const noexcept {
    return bootstrap_result_.arguments.application_arguments;
  }
  [[nodiscard]] const DeepSeekRankExecReady& exec_ready() const noexcept {
    return bootstrap_result_.exec_ready;
  }

 private:
  LinuxDeepSeekRankWorkerStartup(
      std::unique_ptr<DeepSeekPhysicalDeviceIdentityProbe> device_probe,
      std::unique_ptr<LinuxDeepSeekRankWorkerHandshakeOperations> operations,
      std::unique_ptr<LinuxDeepSeekRankWorkerBootstrapRuntime>
          bootstrap_runtime,
      std::unique_ptr<DeepSeekRankScmRightsInFlightLedger>
          scm_rights_ledger,
      std::unique_ptr<
          LinuxDeepSeekRankArtifactTransferReceiverOperations>
          artifact_transfer_operations,
      std::unique_ptr<DeepSeekRankArtifactTransferReceiver>
          artifact_transfer_receiver,
      std::unique_ptr<LinuxDeepSeekRankPostExecResourceProbe> resource_probe,
      std::unique_ptr<StableDeepSeekRankPostExecResourceCollector> collector,
      DeepSeekRankWorkerBootstrapResult bootstrap_result,
      std::unique_ptr<DeepSeekRankPostExecResourceReporter> reporter,
      std::uint64_t authority_wait_deadline_ns,
      std::uint64_t post_mapping_authority_wait_deadline_ns,
      std::uint64_t materialization_grant_wait_deadline_ns) noexcept;
  Status wait_for_resource_progress();
  Status wait_for_artifact_progress();
  Status wait_for_materialization_completion_progress();

  std::unique_ptr<DeepSeekPhysicalDeviceIdentityProbe> device_probe_;
  std::unique_ptr<LinuxDeepSeekRankWorkerHandshakeOperations> operations_;
  std::unique_ptr<LinuxDeepSeekRankWorkerBootstrapRuntime>
      bootstrap_runtime_;
  std::unique_ptr<DeepSeekRankScmRightsInFlightLedger> scm_rights_ledger_;
  std::unique_ptr<
      LinuxDeepSeekRankArtifactTransferReceiverOperations>
      artifact_transfer_operations_;
  std::unique_ptr<DeepSeekRankArtifactTransferReceiver>
      artifact_transfer_receiver_;
  std::unique_ptr<DeepSeekRankArtifactMetadataReceiver>
      artifact_metadata_receiver_;
  std::unique_ptr<LinuxDeepSeekRankPostExecResourceProbe> resource_probe_;
  std::unique_ptr<StableDeepSeekRankPostExecResourceCollector> collector_;
  DeepSeekRankWorkerBootstrapResult bootstrap_result_;
  std::unique_ptr<DeepSeekRankPostExecResourceReporter> reporter_;
  std::unique_ptr<DeepSeekRankArtifactMappingOwner>
      artifact_mapping_owner_;
  std::unique_ptr<
      LinuxDeepSeekRankPostMappingResourceReporterOperations>
      post_mapping_operations_;
  std::unique_ptr<DeepSeekRankPostMappingResourceReporter>
      post_mapping_reporter_;
  std::unique_ptr<LinuxDeepSeekRankMaterializationReceiverOperations>
      materialization_operations_;
  std::unique_ptr<DeepSeekRankMaterializationGrantReceiver>
      materialization_receiver_;
  std::unique_ptr<DeepSeekRankMaterializationCompletionSender>
      materialization_completion_sender_;
  std::uint64_t authority_wait_deadline_ns_ = 0;
  std::uint64_t post_mapping_authority_wait_deadline_ns_ = 0;
  std::uint64_t materialization_grant_wait_deadline_ns_ = 0;
  bool artifact_mapping_transferred_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
