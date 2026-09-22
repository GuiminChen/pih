#include "pih/platform/linux/linux_deepseek_rank_spawn_controller.h"

#include <cerrno>
#include <set>
#include <sys/random.h>
#include <utility>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::vector<std::uint64_t>> random_challenge_identities(
    std::uint32_t count) {
  if (count == 0 || count > 4) {
    return Status::InvalidArgument(
        "Linux DeepSeek rank challenge count is invalid");
  }
  std::vector<std::uint64_t> values;
  std::set<std::uint64_t> unique;
  values.reserve(count);
  const auto maximum_attempts = static_cast<std::size_t>(count) * 8U;
  for (std::size_t attempt = 0;
       attempt < maximum_attempts && values.size() < count; ++attempt) {
    std::uint64_t value = 0;
    ssize_t received = -1;
    do {
      received = ::getrandom(&value, sizeof(value), GRND_NONBLOCK);
    } while (received < 0 && errno == EINTR);
    if (received != static_cast<ssize_t>(sizeof(value))) {
      return Status::Unavailable(
          "Linux DeepSeek rank challenge entropy is unavailable");
    }
    if (value != 0 && unique.insert(value).second) values.push_back(value);
  }
  if (values.size() != count) {
    return Status::Unavailable(
        "Linux DeepSeek rank challenge entropy was not unique");
  }
  return values;
}

}  // namespace

Result<std::unique_ptr<LinuxDeepSeekRankSpawnController>>
LinuxDeepSeekRankSpawnController::Create(
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan plan,
    std::span<const DeepSeekRankPostExecResourcePlan> post_exec_plans,
    std::uint64_t post_exec_deadline_ns,
    DeepSeekRankCapacityPlanInstance capacity_plan_instance,
    std::string worker_executable,
    std::vector<std::string> fixed_arguments,
    int inherited_controller_pidfd,
    std::uint32_t maximum_usage_samples,
    std::uint64_t maximum_metadata_reassembly_bytes) {
  const auto expected_controller_process_identity =
      capacity_plan_instance.expected_controller_process_identity();
  auto capacity_authority_anchor = capacity_plan_instance.admission_anchor_;
  if (capacity_authority_anchor == nullptr ||
      !capacity_authority_anchor->production_ready()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek rank capacity authority lease is not retained");
  }
  auto driver = LinuxDeepSeekRankProcessDriver::Create(
      std::move(worker_executable), std::move(fixed_arguments),
      expected_controller_process_identity, inherited_controller_pidfd,
      capacity_authority_anchor->dspark_enabled(),
      maximum_metadata_reassembly_bytes);
  if (!driver.ok()) return driver.status();
  auto owned_driver = std::make_unique<LinuxDeepSeekRankProcessDriver>(
      std::move(*driver));
  auto owned_probe =
      std::make_unique<LinuxDeepSeekRankSpawnAuthorityProbe>();
  auto collector = StableDeepSeekRankSpawnResourceCollector::Create(
      *owned_probe, maximum_usage_samples);
  if (!collector.ok()) return collector.status();
  auto owned_collector =
      std::make_unique<StableDeepSeekRankSpawnResourceCollector>(
          std::move(*collector));
  auto coordinator = DeepSeekRankSpawnCoordinator::Create(
      manifests, plan, std::move(capacity_plan_instance),
      *owned_collector, *owned_driver);
  if (!coordinator.ok()) return coordinator.status();
  auto owned_coordinator = std::make_unique<DeepSeekRankSpawnCoordinator>(
      std::move(*coordinator));
  auto startup_barrier = DeepSeekRankStartupBarrier::Create(
      *owned_coordinator, manifests, plan, post_exec_plans,
      post_exec_deadline_ns, *owned_driver);
  if (!startup_barrier.ok()) return startup_barrier.status();
  auto owned_startup_barrier =
      std::make_unique<DeepSeekRankStartupBarrier>(
          std::move(*startup_barrier));
  return std::unique_ptr<LinuxDeepSeekRankSpawnController>(
      new LinuxDeepSeekRankSpawnController(
          std::move(capacity_authority_anchor),
          expected_controller_process_identity,
          std::vector<DeepSeekRankProcessManifest>(manifests.begin(),
                                                   manifests.end()),
          plan,
          std::vector<DeepSeekRankPostExecResourcePlan>(
              post_exec_plans.begin(), post_exec_plans.end()),
          std::move(owned_driver),
          std::move(owned_probe),
          std::move(owned_collector), std::move(owned_coordinator),
          std::move(owned_startup_barrier)));
}

LinuxDeepSeekRankSpawnController::LinuxDeepSeekRankSpawnController(
    std::shared_ptr<const RuntimeEngineAdmission> capacity_authority_anchor,
    std::uint64_t controller_process_identity,
    std::vector<DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan spawn_plan,
    std::vector<DeepSeekRankPostExecResourcePlan> resource_plans,
    std::unique_ptr<LinuxDeepSeekRankProcessDriver> driver,
    std::unique_ptr<LinuxDeepSeekRankSpawnAuthorityProbe> probe,
    std::unique_ptr<StableDeepSeekRankSpawnResourceCollector> collector,
    std::unique_ptr<DeepSeekRankSpawnCoordinator> coordinator,
    std::unique_ptr<DeepSeekRankStartupBarrier> startup_barrier) noexcept
    : capacity_authority_anchor_(std::move(capacity_authority_anchor)),
      controller_process_identity_(controller_process_identity),
      manifests_(std::move(manifests)), spawn_plan_(spawn_plan),
      resource_plans_(std::move(resource_plans)),
      driver_(std::move(driver)),
      probe_(std::move(probe)),
      collector_(std::move(collector)), coordinator_(std::move(coordinator)),
      startup_barrier_(std::move(startup_barrier)) {}

Status LinuxDeepSeekRankSpawnController::launch() {
  auto status = coordinator_->launch();
  if (!status.ok()) return status;
  auto* launched = coordinator_->supervisor();
  if (launched == nullptr) {
    return Status::Internal(
        "Linux DeepSeek rank supervisor is absent after launch");
  }
  auto challenges = random_challenge_identities(
      capacity_authority_anchor_->device_ordinals().size());
  if (!challenges.ok()) return launched->abort_startup(challenges.status());
  return launched->dispatch_challenges(
      controller_process_identity_, *challenges);
}

Status LinuxDeepSeekRankSpawnController::advance_startup(
    std::uint64_t now_ns) {
  return startup_barrier_->advance(now_ns);
}

Status LinuxDeepSeekRankSpawnController::begin_artifact_transfer(
    const RuntimeProfileSupervisorBootstrapManifest& profile_bootstrap,
    const RuntimeProfileReadinessReceipt& profile_readiness,
    const DeepSeekPipelinePlan& pipeline,
    const DeepSeekPipelineCapacity& pipeline_capacity,
    std::uint64_t model_startup_deadline_ns,
    DeepSeekRuntimeArtifactAdmissionBinding artifact_binding,
    DeepSeekRankArtifactHandoffPlan handoff,
    std::span<const DeepSeekRankMaterializationAllocationAuthority>
        allocation_authorities) {
  if (artifact_transfer_attempted_ || !startup_barrier_->ready() ||
      startup_barrier_->poisoned() ||
      allocation_authorities.size() != manifests_.size()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek artifact transfer is not beginable");
  }
  auto* supervisor = coordinator_->supervisor();
  const auto* seal = startup_barrier_->resource_seal();
  if (supervisor == nullptr || seal == nullptr || !supervisor->ready() ||
      supervisor->failed()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek artifact transfer lacks a sealed generation");
  }
  const auto fail = [supervisor](Status cause) {
    return supervisor->failed() ? cause
                                : supervisor->abort_post_exec(cause);
  };
  for (const auto& authority : allocation_authorities) {
    auto root =
        compile_deepseek_rank_materialization_allocation_authority_root(
            authority);
    if (!root.ok()) return fail(root.status());
  }
  artifact_transfer_attempted_ = true;
  materialization_allocation_authorities_.assign(
      allocation_authorities.begin(), allocation_authorities.end());
  auto model_startup = DeepSeekRankModelStartupPlan::Compile(
      profile_bootstrap, profile_readiness, *supervisor, manifests_, *seal,
      pipeline, pipeline_capacity, model_startup_deadline_ns);
  if (!model_startup.ok()) return fail(model_startup.status());
  auto post_mapping_deadline = checked_add_u64(
      model_startup_deadline_ns, kPostMappingResourceWaitTimeoutNs);
  if (!post_mapping_deadline.ok()) {
    return fail(post_mapping_deadline.status());
  }
  post_mapping_resource_deadline_ns_ = *post_mapping_deadline;
  auto materialization_deadline = checked_add_u64(
      *post_mapping_deadline, kMaterializationTransactionTimeoutNs);
  if (!materialization_deadline.ok()) {
    return fail(materialization_deadline.status());
  }
  materialization_grant_deadline_ns_ = *materialization_deadline;
  auto plan = DeepSeekRankArtifactTransferPlan::Compile(
      std::move(*model_startup), std::move(artifact_binding),
      std::move(handoff));
  if (!plan.ok()) return fail(plan.status());

  std::vector<DeepSeekRankProcessHandle> handles;
  handles.reserve(manifests_.size());
  for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
    const auto* handle = supervisor->process_handle(rank);
    if (handle == nullptr) {
      return fail(Status::FailedPrecondition(
          "Linux DeepSeek artifact transfer process handle is absent"));
    }
    handles.push_back(*handle);
  }
  auto operations =
      LinuxDeepSeekRankArtifactTransferControllerOperations::Create(
          *driver_, *supervisor, handles);
  if (!operations.ok()) return fail(operations.status());
  auto owned_operations = std::make_unique<
      LinuxDeepSeekRankArtifactTransferControllerOperations>(
      std::move(*operations));
  auto transaction = DeepSeekRankArtifactTransferTransaction::Create(
      std::move(*plan), *owned_operations);
  if (!transaction.ok()) return fail(transaction.status());
  auto owned_transaction =
      std::make_unique<DeepSeekRankArtifactTransferTransaction>(
          std::move(*transaction));
  artifact_transfer_operations_ = std::move(owned_operations);
  artifact_transfer_transaction_ = std::move(owned_transaction);
  return Status::Ok();
}

Status LinuxDeepSeekRankSpawnController::advance_artifact_transfer() {
  // Every transfer phase can be waiting for a credentialed seqpacket.  A
  // closed peer is not an ordinary backpressure condition: observe all pidfds
  // before consuming or producing another protocol frame so a worker exit
  // immediately poisons and terminates the complete rank generation.
  auto* supervisor = coordinator_->supervisor();
  if (supervisor == nullptr || !supervisor->ready() || supervisor->failed()) {
    return Status::FailedPrecondition(
        "Linux DeepSeek artifact transfer lacks a ready supervisor");
  }
  auto health = supervisor->poll();
  if (!health.ok()) return health;
  if (materialization_warmup_coordinator_ != nullptr) {
    return materialization_warmup_coordinator_->advance();
  }
  if (materialization_grant_coordinator_ != nullptr) {
    auto status = materialization_grant_coordinator_->advance();
    if (!status.ok()) return status;
    if (!materialization_grant_coordinator_->complete()) {
      return Status::Unavailable(
          "Linux DeepSeek materialization grant ACKs are pending");
    }
    if (artifact_metadata_transfer_transaction_ == nullptr ||
        materialization_grant_operations_ == nullptr) {
      return Status::FailedPrecondition(
          "Linux DeepSeek materialization warmup authority is absent");
    }
    auto warmup = DeepSeekRankMaterializationWarmupCoordinator::Create(
        *materialization_grant_coordinator_,
        *artifact_metadata_transfer_transaction_,
        *materialization_grant_operations_);
    if (!warmup.ok()) return warmup.status();
    materialization_warmup_coordinator_ = std::make_unique<
        DeepSeekRankMaterializationWarmupCoordinator>(
        std::move(*warmup));
    return materialization_warmup_coordinator_->advance();
  }
  if (post_mapping_resource_coordinator_ != nullptr) {
    auto status = post_mapping_resource_coordinator_->advance();
    if (!status.ok()) return status;
    if (!post_mapping_resource_coordinator_->sealed()) {
      return Status::Unavailable(
          "Linux DeepSeek post-mapping resource seal is pending");
    }
    if (materialization_grant_deadline_ns_ == 0) {
      return Status::FailedPrecondition(
          "Linux DeepSeek materialization grant authority is absent");
    }
    std::vector<DeepSeekRankProcessHandle> handles;
    handles.reserve(manifests_.size());
    for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
      const auto* handle = supervisor->process_handle(rank);
      if (handle == nullptr) {
        return supervisor->abort_post_exec(Status::FailedPrecondition(
            "Linux DeepSeek materialization process handle is absent"));
      }
      handles.push_back(*handle);
    }
    auto operations =
        LinuxDeepSeekRankMaterializationControllerOperations::Create(
            *driver_, *supervisor, handles);
    if (!operations.ok()) {
      return supervisor->abort_post_exec(operations.status());
    }
    auto owned_operations = std::make_unique<
        LinuxDeepSeekRankMaterializationControllerOperations>(
        std::move(*operations));
    auto grant_coordinator =
        DeepSeekRankMaterializationGrantCoordinator::Create(
            *capacity_authority_anchor_, *supervisor, manifests_,
            *post_mapping_resource_coordinator_,
            materialization_grant_deadline_ns_,
            materialization_allocation_authorities_, *owned_operations);
    if (!grant_coordinator.ok()) return grant_coordinator.status();
    auto owned_grant_coordinator =
        std::make_unique<DeepSeekRankMaterializationGrantCoordinator>(
            std::move(*grant_coordinator));
    materialization_grant_operations_ = std::move(owned_operations);
    materialization_grant_coordinator_ =
        std::move(owned_grant_coordinator);
    return materialization_grant_coordinator_->advance();
  }
  if (artifact_metadata_transfer_transaction_ != nullptr) {
    auto status = artifact_metadata_transfer_transaction_->advance();
    if (!status.ok()) return status;
    if (!artifact_metadata_transfer_transaction_->complete()) {
      return Status::Unavailable(
          "Linux DeepSeek metadata transfer is pending");
    }
    const auto* first_seal = startup_barrier_->resource_seal();
    if (first_seal == nullptr ||
        post_mapping_resource_deadline_ns_ == 0) {
      return Status::FailedPrecondition(
          "Linux DeepSeek post-mapping resource authority is absent");
    }
    std::vector<DeepSeekRankProcessHandle> handles;
    handles.reserve(manifests_.size());
    for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
      const auto* handle = supervisor->process_handle(rank);
      if (handle == nullptr) {
        return supervisor->abort_post_exec(Status::FailedPrecondition(
            "Linux DeepSeek post-mapping process handle is absent"));
      }
      handles.push_back(*handle);
    }
    auto operations =
        LinuxDeepSeekRankPostMappingResourceControllerOperations::Create(
            *driver_, *supervisor, handles);
    if (!operations.ok()) {
      return supervisor->abort_post_exec(operations.status());
    }
    auto owned_operations = std::make_unique<
        LinuxDeepSeekRankPostMappingResourceControllerOperations>(
        std::move(*operations));
    auto resource_coordinator =
        DeepSeekRankPostMappingResourceCoordinator::Create(
            *supervisor, manifests_, spawn_plan_, resource_plans_,
            *first_seal, *artifact_metadata_transfer_transaction_,
            post_mapping_resource_deadline_ns_, *owned_operations);
    if (!resource_coordinator.ok()) {
      return resource_coordinator.status();
    }
    auto owned_resource_coordinator =
        std::make_unique<DeepSeekRankPostMappingResourceCoordinator>(
            std::move(*resource_coordinator));
    post_mapping_resource_operations_ = std::move(owned_operations);
    post_mapping_resource_coordinator_ =
        std::move(owned_resource_coordinator);
    return post_mapping_resource_coordinator_->advance();
  }
  if (artifact_transfer_transaction_ == nullptr) {
    return Status::FailedPrecondition(
        "Linux DeepSeek artifact transfer has not begun");
  }
  auto status = artifact_transfer_transaction_->advance();
  if (!status.ok()) return status;
  if (!artifact_transfer_transaction_->complete()) {
    return Status::Unavailable(
        "Linux DeepSeek descriptor transfer is pending");
  }
  std::vector<DeepSeekRankProcessHandle> handles;
  handles.reserve(manifests_.size());
  for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
    const auto* handle = supervisor->process_handle(rank);
    if (handle == nullptr) {
      const auto cause = Status::FailedPrecondition(
          "Linux DeepSeek metadata transfer process handle is absent");
      return supervisor->abort_post_exec(cause);
    }
    handles.push_back(*handle);
  }
  auto operations =
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations::Create(
          *driver_, *supervisor, handles);
  if (!operations.ok()) {
    return supervisor->abort_post_exec(operations.status());
  }
  auto owned_operations = std::make_unique<
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations>(
      std::move(*operations));
  auto transaction =
      DeepSeekRankArtifactMetadataTransferTransaction::Create(
          std::move(*artifact_transfer_transaction_), *owned_operations);
  artifact_transfer_transaction_.reset();
  if (!transaction.ok()) return transaction.status();
  artifact_metadata_transfer_operations_ = std::move(owned_operations);
  artifact_metadata_transfer_transaction_ = std::make_unique<
      DeepSeekRankArtifactMetadataTransferTransaction>(
      std::move(*transaction));
  return artifact_metadata_transfer_transaction_->advance();
}

bool LinuxDeepSeekRankSpawnController::startup_ready() const noexcept {
  return startup_barrier_->ready();
}

const DeepSeekRankPostExecResourceSeal*
LinuxDeepSeekRankSpawnController::resource_seal() const noexcept {
  return startup_barrier_->resource_seal();
}

bool LinuxDeepSeekRankSpawnController::launch_attempted() const noexcept {
  return coordinator_->launch_attempted();
}

const Sha256Digest&
LinuxDeepSeekRankSpawnController::preflight_receipt_root() const noexcept {
  return coordinator_->preflight_receipt_root();
}

const Sha256Digest&
LinuxDeepSeekRankSpawnController::spawn_authorization_root() const noexcept {
  return coordinator_->spawn_authorization_root();
}

}  // namespace pih
