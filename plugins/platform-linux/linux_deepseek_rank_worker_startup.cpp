#include "pih/platform/linux/linux_deepseek_rank_worker_startup.h"

#include <cerrno>
#include <climits>
#include <utility>

#include <poll.h>
#include <sys/prctl.h>

#include "pih/core/checked_math.h"
#include "pih/model/deepseek_rank_engine_resources.h"

namespace pih {
namespace {

Status system_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

}  // namespace

LinuxDeepSeekRankWorkerStartup::LinuxDeepSeekRankWorkerStartup(
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
    std::uint64_t materialization_grant_wait_deadline_ns) noexcept
    : device_probe_(std::move(device_probe)),
      operations_(std::move(operations)),
      bootstrap_runtime_(std::move(bootstrap_runtime)),
      scm_rights_ledger_(std::move(scm_rights_ledger)),
      artifact_transfer_operations_(
          std::move(artifact_transfer_operations)),
      artifact_transfer_receiver_(std::move(artifact_transfer_receiver)),
      resource_probe_(std::move(resource_probe)),
      collector_(std::move(collector)),
      bootstrap_result_(std::move(bootstrap_result)),
      reporter_(std::move(reporter)),
      authority_wait_deadline_ns_(authority_wait_deadline_ns),
      post_mapping_authority_wait_deadline_ns_(
          post_mapping_authority_wait_deadline_ns),
      materialization_grant_wait_deadline_ns_(
          materialization_grant_wait_deadline_ns) {}

Result<std::unique_ptr<LinuxDeepSeekRankWorkerStartup>>
LinuxDeepSeekRankWorkerStartup::Create(
    std::span<const std::string_view> arguments,
    std::unique_ptr<DeepSeekPhysicalDeviceIdentityProbe> device_probe,
    std::uint32_t maximum_usage_samples) {
  if (device_probe == nullptr) {
    return Status::InvalidArgument(
        "Linux DeepSeek rank worker device probe is absent");
  }
  auto parsed = DeepSeekRankWorkerArguments::Parse(arguments);
  if (!parsed.ok()) return parsed.status();
  if (::prctl(PR_SET_DUMPABLE, 0) != 0) {
    return system_failure("prctl PR_SET_DUMPABLE");
  }
  auto operations =
      std::make_unique<LinuxDeepSeekRankWorkerHandshakeOperations>(
          *device_probe);
  auto bootstrap_runtime =
      std::make_unique<LinuxDeepSeekRankWorkerBootstrapRuntime>();
  auto bootstrap = DeepSeekRankWorkerBootstrap::Run(
      std::move(*parsed), *operations, *bootstrap_runtime);
  if (!bootstrap.ok()) return bootstrap.status();
  auto now = operations->monotonic_now_ns();
  if (!now.ok()) return now.status();
  auto authority_wait_deadline = checked_add_u64(
      *now, kAuthorityWaitTimeoutNs);
  if (!authority_wait_deadline.ok()) {
    return authority_wait_deadline.status();
  }
  auto scm_rights_ledger =
      std::make_unique<DeepSeekRankScmRightsInFlightLedger>();
  auto artifact_wait_deadline = checked_add_u64(
      bootstrap->arguments.manifest.startup_deadline_ns,
      kArtifactTransferWaitTimeoutNs);
  if (!artifact_wait_deadline.ok()) return artifact_wait_deadline.status();
  auto post_mapping_wait_deadline = checked_add_u64(
      *artifact_wait_deadline, kPostMappingResourceWaitTimeoutNs);
  if (!post_mapping_wait_deadline.ok()) {
    return post_mapping_wait_deadline.status();
  }
  auto materialization_wait_deadline = checked_add_u64(
      *post_mapping_wait_deadline, kMaterializationTransactionTimeoutNs);
  if (!materialization_wait_deadline.ok()) {
    return materialization_wait_deadline.status();
  }
  auto artifact_operations =
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          bootstrap->arguments.control_fd,
          bootstrap->arguments.controller_process_identity);
  if (!artifact_operations.ok()) return artifact_operations.status();
  auto owned_artifact_operations = std::make_unique<
      LinuxDeepSeekRankArtifactTransferReceiverOperations>(
      std::move(*artifact_operations));
  auto artifact_receiver = DeepSeekRankArtifactTransferReceiver::Create(
      bootstrap->arguments.manifest, bootstrap->exec_ready,
      bootstrap->arguments.control_fd, *artifact_wait_deadline,
      *scm_rights_ledger, *owned_artifact_operations);
  if (!artifact_receiver.ok()) return artifact_receiver.status();
  auto owned_artifact_receiver =
      std::make_unique<DeepSeekRankArtifactTransferReceiver>(
          std::move(*artifact_receiver));
  auto resource_probe =
      std::make_unique<LinuxDeepSeekRankPostExecResourceProbe>(
          *scm_rights_ledger);
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      *resource_probe, maximum_usage_samples);
  if (!collector.ok()) return collector.status();
  auto owned_collector =
      std::make_unique<StableDeepSeekRankPostExecResourceCollector>(
          std::move(*collector));
  auto reporter = DeepSeekRankPostExecResourceReporter::Create(
      bootstrap->arguments.manifest, bootstrap->exec_ready,
      bootstrap->arguments.control_fd, *owned_collector, *operations);
  if (!reporter.ok()) return reporter.status();
  auto owned_reporter =
      std::make_unique<DeepSeekRankPostExecResourceReporter>(
          std::move(*reporter));
  return std::unique_ptr<LinuxDeepSeekRankWorkerStartup>(
      new LinuxDeepSeekRankWorkerStartup(
          std::move(device_probe), std::move(operations),
          std::move(bootstrap_runtime), std::move(scm_rights_ledger),
          std::move(owned_artifact_operations),
          std::move(owned_artifact_receiver),
          std::move(resource_probe), std::move(owned_collector),
          std::move(*bootstrap), std::move(owned_reporter),
          *authority_wait_deadline, *post_mapping_wait_deadline,
          *materialization_wait_deadline));
}

Status LinuxDeepSeekRankWorkerStartup::advance_resource_barrier() {
  if (poisoned_ || reporter_->reported()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek rank worker resource barrier is closed");
  }
  auto status = reporter_->advance();
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    poisoned_ = true;
  }
  return status;
}

Status LinuxDeepSeekRankWorkerStartup::wait_for_resource_progress() {
  const auto event = reporter_->wait_event();
  if (!event) {
    return Status::FailedPrecondition(
        "Linux DeepSeek rank worker reporter has no wait event");
  }
  for (;;) {
    auto now = operations_->monotonic_now_ns();
    if (!now.ok()) return now.status();
    const auto deadline = reporter_->deadline_ns().value_or(
        authority_wait_deadline_ns_);
    if (*now >= deadline) {
      return Status::DeadlineExceeded(
          "Linux DeepSeek rank worker resource wait expired");
    }
    const auto remaining = deadline - *now;
    const auto milliseconds = remaining / UINT64_C(1'000'000) +
                              (remaining % UINT64_C(1'000'000) != 0);
    const auto timeout = static_cast<int>(
        milliseconds > static_cast<std::uint64_t>(INT_MAX)
            ? INT_MAX
            : milliseconds);
    const auto control_events =
        *event == DeepSeekRankPostExecResourceReporterWaitEvent::
                      kAuthorityReadable
            ? POLLIN
            : POLLOUT;
    pollfd descriptors[2]{{bootstrap_result_.arguments.control_fd,
                           static_cast<short>(control_events), 0},
                          {bootstrap_result_.arguments.controller_pidfd,
                           POLLIN, 0}};
    const auto observed = ::poll(descriptors, 2, timeout);
    if (observed < 0 && errno == EINTR) continue;
    if (observed < 0) {
      return system_failure("poll DeepSeek rank worker resource barrier");
    }
    if (observed == 0) continue;
    if ((descriptors[1].revents &
         (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return Status::FailedPrecondition(
          "DeepSeek rank controller exited during resource barrier");
    }
    if ((descriptors[0].revents &
         (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return Status::FailedPrecondition(
          "DeepSeek rank control channel closed during resource barrier");
    }
    if ((descriptors[0].revents & control_events) != 0) {
      return Status::Ok();
    }
  }
}

Status LinuxDeepSeekRankWorkerStartup::run_resource_barrier() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "Linux DeepSeek rank worker resource barrier is poisoned");
  }
  while (!reporter_->reported()) {
    auto status = advance_resource_barrier();
    if (status.ok()) continue;
    if (status.code() != StatusCode::kUnavailable) return status;
    status = wait_for_resource_progress();
    if (!status.ok()) {
      poisoned_ = true;
      return status;
    }
  }
  return Status::Ok();
}

bool LinuxDeepSeekRankWorkerStartup::resource_reported() const noexcept {
  return reporter_->reported();
}

Status LinuxDeepSeekRankWorkerStartup::advance_artifact_transfer() {
  if (poisoned_ || !reporter_->reported()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek worker artifact transfer is not advanceable");
  }
  if (materialization_receiver_ != nullptr) {
    auto status = materialization_receiver_->advance();
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      poisoned_ = true;
    }
    return status;
  }
  if (post_mapping_reporter_ != nullptr &&
      post_mapping_reporter_->reported()) {
    const auto* report = post_mapping_reporter_->report();
    if (artifact_mapping_owner_ == nullptr || report == nullptr) {
      poisoned_ = true;
      return Status::FailedPrecondition(
          "Linux DeepSeek materialization antecedent disappeared");
    }
    auto operations =
        LinuxDeepSeekRankMaterializationReceiverOperations::Create(
            bootstrap_result_.arguments.control_fd,
            bootstrap_result_.arguments.controller_process_identity);
    if (!operations.ok()) {
      poisoned_ = true;
      return operations.status();
    }
    auto owned_operations = std::make_unique<
        LinuxDeepSeekRankMaterializationReceiverOperations>(
        std::move(*operations));
    auto receiver = DeepSeekRankMaterializationGrantReceiver::Create(
        bootstrap_result_.arguments.manifest,
        bootstrap_result_.exec_ready,
        bootstrap_result_.arguments.control_fd,
        materialization_grant_wait_deadline_ns_,
        *artifact_mapping_owner_, *report, *owned_operations);
    if (!receiver.ok()) {
      poisoned_ = true;
      return receiver.status();
    }
    materialization_operations_ = std::move(owned_operations);
    materialization_receiver_ =
        std::make_unique<DeepSeekRankMaterializationGrantReceiver>(
            std::move(*receiver));
    auto status = materialization_receiver_->advance();
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      poisoned_ = true;
    }
    return status;
  }
  if (artifact_mapping_owner_ != nullptr) {
    if (post_mapping_reporter_ == nullptr) {
      auto operations =
          LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
              bootstrap_result_.arguments.control_fd,
              bootstrap_result_.arguments.controller_process_identity);
      if (!operations.ok()) {
        poisoned_ = true;
        return operations.status();
      }
      auto owned_operations = std::make_unique<
          LinuxDeepSeekRankPostMappingResourceReporterOperations>(
          std::move(*operations));
      auto reporter = DeepSeekRankPostMappingResourceReporter::Create(
          bootstrap_result_.arguments.manifest,
          bootstrap_result_.exec_ready,
          bootstrap_result_.arguments.control_fd,
          *artifact_mapping_owner_, *collector_, *owned_operations);
      if (!reporter.ok()) {
        poisoned_ = true;
        return reporter.status();
      }
      post_mapping_operations_ = std::move(owned_operations);
      post_mapping_reporter_ =
          std::make_unique<DeepSeekRankPostMappingResourceReporter>(
              std::move(*reporter));
    }
    auto status = post_mapping_reporter_->advance();
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      poisoned_ = true;
    }
    return status;
  }
  if (artifact_metadata_receiver_ != nullptr) {
    auto status = artifact_metadata_receiver_->advance();
    if (!status.ok()) {
      if (status.code() != StatusCode::kUnavailable) poisoned_ = true;
      return status;
    }
    if (!artifact_metadata_receiver_->complete()) return Status::Ok();
    auto receipt = DeepSeekRankArtifactMetadataReceipt::Create(
        std::move(*artifact_metadata_receiver_));
    artifact_metadata_receiver_.reset();
    if (!receipt.ok()) {
      poisoned_ = true;
      return receipt.status();
    }
    auto mapping_owner = DeepSeekRankArtifactMappingOwner::Create(
        std::move(*receipt));
    if (!mapping_owner.ok()) {
      poisoned_ = true;
      return mapping_owner.status();
    }
    artifact_mapping_owner_ = std::move(*mapping_owner);
    return Status::Ok();
  }
  if (artifact_transfer_receiver_ == nullptr) {
    poisoned_ = true;
    return Status::Internal(
        "Linux DeepSeek worker descriptor receiver is absent");
  }
  auto status = artifact_transfer_receiver_->advance();
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable) poisoned_ = true;
    return status;
  }
  if (!artifact_transfer_receiver_->complete()) return Status::Ok();
  auto receipt = DeepSeekRankArtifactAdoptionReceipt::Create(
      std::move(*artifact_transfer_receiver_));
  artifact_transfer_receiver_.reset();
  if (!receipt.ok()) {
    poisoned_ = true;
    return receipt.status();
  }
  auto metadata_receiver = DeepSeekRankArtifactMetadataReceiver::Create(
      std::move(*receipt),
      bootstrap_result_.arguments.expected_dspark_enabled,
      bootstrap_result_.arguments.maximum_metadata_reassembly_bytes,
      *artifact_transfer_operations_);
  if (!metadata_receiver.ok()) {
    poisoned_ = true;
    return metadata_receiver.status();
  }
  artifact_metadata_receiver_ =
      std::make_unique<DeepSeekRankArtifactMetadataReceiver>(
          std::move(*metadata_receiver));
  return artifact_metadata_receiver_->advance();
}

Status LinuxDeepSeekRankWorkerStartup::wait_for_artifact_progress() {
  const auto waiting_for_materialization =
      materialization_receiver_ != nullptr &&
      !materialization_receiver_->complete();
  const auto waiting_for_post_mapping =
      post_mapping_reporter_ != nullptr &&
      !post_mapping_reporter_->reported();
  if (artifact_transfer_receiver_ == nullptr &&
      artifact_metadata_receiver_ == nullptr &&
      !waiting_for_post_mapping && !waiting_for_materialization) {
    return Status::FailedPrecondition(
        "Linux DeepSeek worker artifact receiver is absent");
  }
  const auto waiting_for_metadata = artifact_metadata_receiver_ != nullptr;
  const auto post_mapping_event =
      waiting_for_post_mapping && !waiting_for_materialization
          ? post_mapping_reporter_->wait_event()
          : std::nullopt;
  const auto materialization_event = waiting_for_materialization
      ? materialization_receiver_->wait_event()
      : std::nullopt;
  const auto descriptor_event =
      waiting_for_materialization || waiting_for_post_mapping ||
              waiting_for_metadata
          ? std::nullopt
          : artifact_transfer_receiver_->wait_event();
  const auto metadata_event =
      waiting_for_metadata ? artifact_metadata_receiver_->wait_event()
                           : std::nullopt;
  if ((waiting_for_materialization && !materialization_event) ||
      (!waiting_for_materialization && waiting_for_post_mapping &&
       !post_mapping_event) ||
      (!waiting_for_materialization && !waiting_for_post_mapping &&
       !waiting_for_metadata &&
       !descriptor_event) ||
      (!waiting_for_materialization && !waiting_for_post_mapping &&
       waiting_for_metadata &&
       !metadata_event)) {
    return Status::FailedPrecondition(
        "Linux DeepSeek worker artifact receiver has no wait event");
  }
  for (;;) {
    auto now = waiting_for_materialization
                   ? materialization_operations_->monotonic_now_ns()
               : waiting_for_post_mapping
                   ? post_mapping_operations_->monotonic_now_ns()
                   : artifact_transfer_operations_->monotonic_now_ns();
    if (!now.ok()) return now.status();
    const auto deadline = waiting_for_materialization
        ? materialization_receiver_->wait_deadline_ns()
        : waiting_for_post_mapping
        ? post_mapping_reporter_->deadline_ns().value_or(
              post_mapping_authority_wait_deadline_ns_)
        : (waiting_for_metadata
               ? artifact_metadata_receiver_->wait_deadline_ns()
               : artifact_transfer_receiver_->wait_deadline_ns());
    if (*now >= deadline) {
      return Status::DeadlineExceeded(
          "Linux DeepSeek worker artifact wait expired");
    }
    const auto remaining = deadline - *now;
    const auto milliseconds = remaining / UINT64_C(1'000'000) +
                              (remaining % UINT64_C(1'000'000) != 0);
    const auto timeout = static_cast<int>(
        milliseconds > static_cast<std::uint64_t>(INT_MAX)
            ? INT_MAX
            : milliseconds);
    const auto control_events = waiting_for_materialization
        ? (*materialization_event ==
                   DeepSeekRankMaterializationGrantReceiverWaitEvent::
                       kGrantReadable
               ? POLLIN
               : POLLOUT)
        : waiting_for_post_mapping
        ? (*post_mapping_event ==
                   DeepSeekRankPostMappingResourceReporterWaitEvent::
                       kAuthorityReadable
               ? POLLIN
               : POLLOUT)
        : waiting_for_metadata
        ? (*metadata_event ==
                   DeepSeekRankArtifactMetadataReceiverWaitEvent::
                       kPacketReadable
               ? POLLIN
               : POLLOUT)
        : (*descriptor_event ==
                   DeepSeekRankArtifactTransferReceiverWaitEvent::
                       kPacketReadable
               ? POLLIN
               : POLLOUT);
    pollfd descriptors[2]{{bootstrap_result_.arguments.control_fd,
                           static_cast<short>(control_events), 0},
                          {bootstrap_result_.arguments.controller_pidfd,
                           POLLIN, 0}};
    const auto observed = ::poll(descriptors, 2, timeout);
    if (observed < 0 && errno == EINTR) continue;
    if (observed < 0) {
      return system_failure("poll DeepSeek worker artifact transfer");
    }
    if (observed == 0) continue;
    if ((descriptors[1].revents &
         (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return Status::FailedPrecondition(
          "DeepSeek rank controller exited during artifact transfer");
    }
    if ((descriptors[0].revents &
         (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return Status::FailedPrecondition(
          "DeepSeek rank control channel closed during artifact transfer");
    }
    if ((descriptors[0].revents & control_events) != 0) {
      return Status::Ok();
    }
  }
}

Status LinuxDeepSeekRankWorkerStartup::run_artifact_transfer() {
  if (poisoned_ || !reporter_->reported()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek worker artifact transfer is not runnable");
  }
  while (!materialization_grant_accepted()) {
    auto status = advance_artifact_transfer();
    if (status.ok()) continue;
    if (status.code() != StatusCode::kUnavailable) return status;
    status = wait_for_artifact_progress();
    if (!status.ok()) {
      poisoned_ = true;
      return status;
    }
  }
  return Status::Ok();
}

Result<DeepSeekRankPrefaultedMaterializationInputs>
LinuxDeepSeekRankWorkerStartup::prefault_materialization_inputs() {
  if (poisoned_ || materialization_receiver_ == nullptr ||
      !materialization_receiver_->complete() ||
      artifact_mapping_owner_ == nullptr || artifact_mapping_transferred_) {
    return Status::FailedPrecondition(
        "Linux DeepSeek prefault inputs are absent");
  }
  auto operations =
      LinuxDeepSeekRankArtifactPrefaultOperations::Create();
  if (!operations.ok()) {
    poisoned_ = true;
    return operations.status();
  }
  auto admission = materialization_receiver_->take_admission();
  if (!admission.ok()) {
    poisoned_ = true;
    return admission.status();
  }
  auto mapping_owner = std::move(artifact_mapping_owner_);
  artifact_mapping_transferred_ = true;
  auto result = DeepSeekRankArtifactPrefaultTransaction::Run(
      DeepSeekRankAuthorizedMaterializationInputs(
          std::move(*admission), std::move(mapping_owner)),
      **operations);
  if (!result.ok()) poisoned_ = true;
  return result;
}

Status LinuxDeepSeekRankWorkerStartup::begin_materialization_completion(
    const DeepSeekRankEngineResources& resources,
    const DeepSeekRankMaterializationCompletionObservation& observation) {
  if (poisoned_ || materialization_operations_ == nullptr ||
      materialization_receiver_ == nullptr ||
      !materialization_receiver_->complete() ||
      !artifact_mapping_transferred_ ||
      materialization_completion_sender_ != nullptr) {
    return Status::FailedPrecondition(
        "Linux DeepSeek materialization completion is not startable");
  }
  const auto* admission = resources.materialization_admission();
  if (admission == nullptr ||
      !validate_deepseek_rank_materialization_completion_worker_identity(
           *admission, bootstrap_result_.exec_ready)
           .ok()) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "Linux DeepSeek completion resources differ from worker identity");
  }
  auto completion = compile_deepseek_rank_materialization_completion(
      resources, observation);
  if (!completion.ok()) {
    poisoned_ = true;
    return completion.status();
  }
  auto sender = DeepSeekRankMaterializationCompletionSender::Create(
      std::move(*completion), bootstrap_result_.arguments.control_fd,
      *materialization_operations_);
  if (!sender.ok()) {
    poisoned_ = true;
    return sender.status();
  }
  materialization_completion_sender_ = std::make_unique<
      DeepSeekRankMaterializationCompletionSender>(std::move(*sender));
  return Status::Ok();
}

Status LinuxDeepSeekRankWorkerStartup::advance_materialization_completion() {
  if (poisoned_ || materialization_completion_sender_ == nullptr) {
    return Status::FailedPrecondition(
        "Linux DeepSeek materialization completion sender is absent");
  }
  auto status = materialization_completion_sender_->advance();
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    poisoned_ = true;
  }
  return status;
}

Status LinuxDeepSeekRankWorkerStartup::
    wait_for_materialization_completion_progress() {
  if (poisoned_ || materialization_completion_sender_ == nullptr ||
      materialization_operations_ == nullptr) {
    return Status::FailedPrecondition(
        "Linux DeepSeek materialization completion is not waitable");
  }
  for (;;) {
    auto now = materialization_operations_->monotonic_now_ns();
    if (!now.ok()) return now.status();
    const auto deadline = materialization_completion_sender_->deadline_ns();
    if (*now >= deadline) {
      return Status::DeadlineExceeded(
          "Linux DeepSeek materialization completion wait expired");
    }
    const auto remaining = deadline - *now;
    const auto milliseconds = remaining / UINT64_C(1'000'000) +
                              (remaining % UINT64_C(1'000'000) != 0);
    const auto timeout = static_cast<int>(
        milliseconds > static_cast<std::uint64_t>(INT_MAX)
            ? INT_MAX
            : milliseconds);
    pollfd descriptors[2]{{bootstrap_result_.arguments.control_fd,
                           POLLOUT, 0},
                          {bootstrap_result_.arguments.controller_pidfd,
                           POLLIN, 0}};
    const auto observed = ::poll(descriptors, 2, timeout);
    if (observed < 0 && errno == EINTR) continue;
    if (observed < 0) {
      return system_failure(
          "poll DeepSeek worker materialization completion");
    }
    if (observed == 0) continue;
    if ((descriptors[1].revents &
         (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return Status::FailedPrecondition(
          "DeepSeek controller exited before materialization completion");
    }
    if ((descriptors[0].revents &
         (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return Status::FailedPrecondition(
          "DeepSeek control channel closed before materialization completion");
    }
    if ((descriptors[0].revents & POLLOUT) != 0) return Status::Ok();
  }
}

Status LinuxDeepSeekRankWorkerStartup::run_materialization_completion() {
  if (poisoned_ || materialization_completion_sender_ == nullptr) {
    return Status::FailedPrecondition(
        "Linux DeepSeek materialization completion is not runnable");
  }
  while (!materialization_completion_sender_->complete()) {
    auto status = advance_materialization_completion();
    if (status.ok()) continue;
    if (status.code() != StatusCode::kUnavailable) return status;
    status = wait_for_materialization_completion_progress();
    if (!status.ok()) {
      poisoned_ = true;
      return status;
    }
  }
  return Status::Ok();
}

}  // namespace pih
