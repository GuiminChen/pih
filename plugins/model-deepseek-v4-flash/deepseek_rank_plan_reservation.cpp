#include "pih/model/deepseek_rank_plan_reservation.h"

#include <utility>

namespace pih { namespace {

Status validate_topology(const DeepSeekPipelinePlanDescriptor& descriptor,
                         const DeepSeekStagePlan& stage,
                         std::uint32_t world_size) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.sequence_count == 0 || world_size == 0 || world_size > 4 ||
      stage.rank >= world_size ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 ||
      stage.owns_embedding != (stage.rank == 0) ||
      stage.owns_lm_head != (stage.rank + 1 == world_size) ||
      (stage.owns_dspark && stage.rank + 1 != world_size) ||
      (descriptor.phase == DeepSeekPlanPhase::kDrain
           ? descriptor.token_count != 0
           : descriptor.token_count == 0)) {
    return Status::InvalidArgument(
        "DeepSeek rank plan reservation topology is invalid");
  }
  return Status::Ok();
}

void add(std::vector<DeepSeekRankPlanResourceRequirement>& result,
         DeepSeekRankPlanResourceKind kind, std::uint64_t units) {
  if (units != 0) result.push_back({kind, units});
}

void rollback(
    std::vector<std::unique_ptr<DeepSeekRankPlanResourceLease>>& leases) {
  for (auto lease = leases.rbegin(); lease != leases.rend(); ++lease) {
    lease->reset();
  }
  leases.clear();
}

} }  // namespace pih::<anonymous>

namespace pih {

Result<DeepSeekRankPlanReservation> DeepSeekRankPlanReservation::Prepare(
    DeepSeekPipelinePlanDescriptor descriptor, DeepSeekStagePlan stage,
    std::uint32_t world_size,
    DeepSeekRoutedExpertResidency expert_residency,
    std::uint64_t verify_prefix_workspace_bytes,
    std::uint64_t operator_workspace_bytes,
    std::uint64_t expert_workspace_bytes,
    DeepSeekRankPlanResourceProvider& provider) {
  auto topology = validate_topology(descriptor, stage, world_size);
  if (!topology.ok()) return topology;
  const bool spill =
      expert_residency == DeepSeekRoutedExpertResidency::kHostSpill;
  const bool resident =
      expert_residency == DeepSeekRoutedExpertResidency::kFullResident;
  const bool verify = descriptor.phase == DeepSeekPlanPhase::kVerify;
  if ((!spill && !resident) || operator_workspace_bytes == 0 ||
      expert_workspace_bytes == 0 ||
      (verify ? verify_prefix_workspace_bytes == 0
              : verify_prefix_workspace_bytes != 0)) {
    return Status::InvalidArgument(
        "DeepSeek rank plan reservation capacity is invalid");
  }
  std::vector<DeepSeekRankPlanResourceRequirement> requirements;
  requirements.reserve(10);
  add(requirements, DeepSeekRankPlanResourceKind::kIncomingBoundary,
      stage.rank == 0 ? 0 : 1);
  add(requirements, DeepSeekRankPlanResourceKind::kOutgoingActivation,
      stage.rank + 1 == world_size ? 0 : 1);
  add(requirements, DeepSeekRankPlanResourceKind::kControlSlot, 1);
  add(requirements,
      DeepSeekRankPlanResourceKind::kAttentionStateTransaction,
      descriptor.sequence_count);
  add(requirements,
      DeepSeekRankPlanResourceKind::kVerifyPrefixWorkspaceBytes,
      verify_prefix_workspace_bytes);
  add(requirements, DeepSeekRankPlanResourceKind::kOperatorWorkspaceBytes,
      operator_workspace_bytes);
  add(requirements, DeepSeekRankPlanResourceKind::kExpertWorkspaceBytes,
      expert_workspace_bytes);
  add(requirements,
      DeepSeekRankPlanResourceKind::kPagerForwardProgressSlot,
      spill ? 2 : 0);
  add(requirements, DeepSeekRankPlanResourceKind::kPinnedStagingCredit,
      spill ? 2 : 0);
  add(requirements, DeepSeekRankPlanResourceKind::kOutputSamplingCredit,
      stage.owns_lm_head ? descriptor.sequence_count : 0);

  std::vector<std::unique_ptr<DeepSeekRankPlanResourceLease>> leases;
  leases.reserve(requirements.size());
  for (const auto requirement : requirements) {
    auto lease = provider.reserve(descriptor, stage, requirement);
    if (!lease.ok()) {
      rollback(leases);
      return lease.status();
    }
    if (*lease == nullptr) {
      rollback(leases);
      return Status::Internal(
          "DeepSeek rank resource provider returned a null lease");
    }
    leases.push_back(std::move(*lease));
  }
  return DeepSeekRankPlanReservation(std::move(requirements),
                                     std::move(leases));
}

DeepSeekRankPlanReservation::~DeepSeekRankPlanReservation() { release(); }

DeepSeekRankPlanReservation::DeepSeekRankPlanReservation(
    DeepSeekRankPlanReservation&& other) noexcept
    : requirements_(std::move(other.requirements_)),
      leases_(std::move(other.leases_)), state_(other.state_) {
  other.leases_.clear();
  other.state_ = DeepSeekRankPlanReservationState::kReleased;
}

DeepSeekRankPlanReservation& DeepSeekRankPlanReservation::operator=(
    DeepSeekRankPlanReservation&& other) noexcept {
  if (this == &other) return *this;
  release();
  requirements_ = std::move(other.requirements_);
  leases_ = std::move(other.leases_);
  state_ = other.state_;
  other.leases_.clear();
  other.state_ = DeepSeekRankPlanReservationState::kReleased;
  return *this;
}

Status DeepSeekRankPlanReservation::validate_commit() const {
  if (state_ != DeepSeekRankPlanReservationState::kPrepared ||
      leases_.size() != requirements_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek rank plan reservation is not committable");
  }
  return Status::Ok();
}

Status DeepSeekRankPlanReservation::commit() {
  const auto status = validate_commit();
  if (!status.ok()) return status;
  state_ = DeepSeekRankPlanReservationState::kCommitted;
  return Status::Ok();
}

void DeepSeekRankPlanReservation::release() noexcept {
  if (state_ == DeepSeekRankPlanReservationState::kReleased) return;
  rollback(leases_);
  state_ = DeepSeekRankPlanReservationState::kReleased;
}

}  // namespace pih
