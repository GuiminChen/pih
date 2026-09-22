#include "pih/model/deepseek_rank_startup_barrier.h"

#include <utility>

namespace pih {
namespace {

Status validate_static_inputs(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> post_exec_plans,
    std::uint64_t post_exec_deadline_ns) {
  auto status = DeepSeekRankSpawnPreflightReceipt::ValidatePlan(
      manifests, spawn_plan);
  if (!status.ok()) return status;
  if (post_exec_plans.size() != manifests.size() ||
      post_exec_deadline_ns == 0) {
    return Status::InvalidArgument(
        "DeepSeek rank startup barrier inputs are invalid");
  }
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    if (post_exec_deadline_ns <= manifests[rank].startup_deadline_ns ||
        post_exec_plans[rank].os_resource_envelope_root !=
            post_exec_plans[0].os_resource_envelope_root) {
      return Status::InvalidArgument(
          "DeepSeek rank startup barrier deadline or envelope is invalid");
    }
    auto root = compile_deepseek_rank_post_exec_resource_plan_root(
        manifests, spawn_plan, rank, post_exec_plans[rank]);
    if (!root.ok()) return root.status();
  }
  return Status::Ok();
}

}  // namespace

DeepSeekRankStartupBarrier::DeepSeekRankStartupBarrier(
    DeepSeekRankSpawnCoordinator& spawn_coordinator,
    std::vector<DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan spawn_plan,
    std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans,
    std::uint64_t post_exec_deadline_ns,
    DeepSeekRankPostExecResourceChannel& channel) noexcept
    : spawn_coordinator_(&spawn_coordinator),
      manifests_(std::move(manifests)), spawn_plan_(spawn_plan),
      post_exec_plans_(std::move(post_exec_plans)),
      post_exec_deadline_ns_(post_exec_deadline_ns), channel_(&channel) {}

Result<DeepSeekRankStartupBarrier> DeepSeekRankStartupBarrier::Create(
    DeepSeekRankSpawnCoordinator& spawn_coordinator,
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> post_exec_plans,
    std::uint64_t post_exec_deadline_ns,
    DeepSeekRankPostExecResourceChannel& channel) {
  auto status = validate_static_inputs(
      manifests, spawn_plan, post_exec_plans, post_exec_deadline_ns);
  if (!status.ok()) return status;
  return DeepSeekRankStartupBarrier(
      spawn_coordinator,
      std::vector<DeepSeekRankProcessManifest>(manifests.begin(),
                                               manifests.end()),
      spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan>(
          post_exec_plans.begin(), post_exec_plans.end()),
      post_exec_deadline_ns, channel);
}

Status DeepSeekRankStartupBarrier::fail(Status cause) noexcept {
  poisoned_ = true;
  auto* supervisor = spawn_coordinator_->supervisor();
  if (supervisor != nullptr && supervisor->ready() &&
      !supervisor->failed()) {
    return supervisor->abort_post_exec(
        cause.ok() ? Status::Internal(
                         "DeepSeek rank startup barrier failed")
                   : cause);
  }
  return cause.ok() ? Status::Internal(
                          "DeepSeek rank startup barrier failed")
                    : cause;
}

Status DeepSeekRankStartupBarrier::advance(std::uint64_t now_ns) {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek rank startup barrier is poisoned");
  }
  if (ready()) return Status::Ok();
  auto* supervisor = spawn_coordinator_->supervisor();
  if (supervisor == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank startup barrier has not launched");
  }
  // A resource-channel read can remain backpressured after its peer exits.
  // Poll the supervised pidfds before advancing either the exec handshake or
  // the post-exec exchange so a lost worker immediately poisons the complete
  // PP generation instead of being mistaken for a transient channel stall.
  auto status = supervisor->poll();
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  if (!supervisor->ready()) {
    status = supervisor->advance_exec_startup(now_ns);
    if (!status.ok()) {
      if (status.code() == StatusCode::kUnavailable) return status;
      poisoned_ = true;
      return status;
    }
  }
  if (!post_exec_coordinator_) {
    auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
        *supervisor, manifests_, spawn_plan_, post_exec_plans_,
        post_exec_deadline_ns_, *channel_);
    if (!coordinator.ok()) return fail(coordinator.status());
    post_exec_coordinator_.emplace(std::move(*coordinator));
  }
  status = post_exec_coordinator_->advance(now_ns);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    poisoned_ = true;
  }
  return status;
}

bool DeepSeekRankStartupBarrier::exec_ready() const noexcept {
  const auto* supervisor = spawn_coordinator_->supervisor();
  return supervisor != nullptr && supervisor->ready();
}

bool DeepSeekRankStartupBarrier::ready() const noexcept {
  return post_exec_coordinator_ && post_exec_coordinator_->sealed();
}

const DeepSeekRankPostExecResourceSeal*
DeepSeekRankStartupBarrier::resource_seal() const noexcept {
  return post_exec_coordinator_ ? post_exec_coordinator_->seal() : nullptr;
}

}  // namespace pih
