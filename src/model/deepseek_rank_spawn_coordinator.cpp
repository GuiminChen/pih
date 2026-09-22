#include "pih/model/deepseek_rank_spawn_coordinator.h"

#include <utility>

namespace pih {

Result<DeepSeekRankSpawnCoordinator> DeepSeekRankSpawnCoordinator::Create(
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan plan,
    DeepSeekRankCapacityPlanInstance capacity_plan_instance,
    DeepSeekRankSpawnResourceCollector& collector,
    DeepSeekRankProcessDriver& driver) {
  if (manifests.empty() || manifests.size() > 4) {
    return Status::InvalidArgument(
        "DeepSeek rank spawn coordinator inputs are invalid");
  }
  auto status = DeepSeekRankSpawnPreflightReceipt::ValidatePlan(
      manifests, plan);
  if (!status.ok()) return status;
  status = capacity_plan_instance.validate_static_binding(manifests, plan);
  if (!status.ok()) return status;
  return DeepSeekRankSpawnCoordinator(
      std::vector<DeepSeekRankProcessManifest>(manifests.begin(),
                                               manifests.end()),
      plan, std::move(capacity_plan_instance), collector, driver);
}

DeepSeekRankSpawnCoordinator::DeepSeekRankSpawnCoordinator(
    std::vector<DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan plan,
    DeepSeekRankCapacityPlanInstance capacity_plan_instance,
    DeepSeekRankSpawnResourceCollector& collector,
    DeepSeekRankProcessDriver& driver) noexcept
    : manifests_(std::move(manifests)), plan_(plan),
      capacity_plan_instance_(std::move(capacity_plan_instance)),
      collector_(&collector), driver_(&driver) {}

Status DeepSeekRankSpawnCoordinator::launch() {
  if (launch_attempted_) {
    return Status::FailedPrecondition(
        "DeepSeek rank spawn transaction cannot be replayed");
  }
  launch_attempted_ = true;
  auto observation = collector_->collect();
  if (!observation.ok()) return observation.status();
  auto preflight = DeepSeekRankSpawnPreflightReceipt::Compile(
      manifests_, plan_, *observation);
  if (!preflight.ok()) return preflight.status();
  preflight_receipt_root_ = preflight->receipt_root();
  auto capacity_plan_instance = std::move(*capacity_plan_instance_);
  capacity_plan_instance_.reset();
  auto authorization = DeepSeekRankSpawnAuthorization::Create(
      manifests_, std::move(capacity_plan_instance), *preflight);
  if (!authorization.ok()) return authorization.status();
  spawn_authorization_root_ = authorization->authorization_root();
  auto supervisor = DeepSeekRankProcessSupervisor::Create(
      manifests_, std::move(*authorization), *driver_);
  if (!supervisor.ok()) return supervisor.status();
  auto status = supervisor->launch();
  if (!status.ok()) return status;
  supervisor_.emplace(std::move(*supervisor));
  return Status::Ok();
}

}  // namespace pih
