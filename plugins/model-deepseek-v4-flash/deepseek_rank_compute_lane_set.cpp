#include "pih/model/deepseek_rank_compute_lane_set.h"

#include <new>
#include <utility>

namespace pih {

DeepSeekRankComputeLaneSet& DeepSeekRankComputeLaneSet::operator=(
    DeepSeekRankComputeLaneSet&& other) noexcept {
  if (this != &other) {
    // bundle_ and both lane drivers borrow infrastructure_. Apply the class's
    // reverse destruction order before replacing the ownership graph.
    this->~DeepSeekRankComputeLaneSet();
    ::new (static_cast<void*>(this))
        DeepSeekRankComputeLaneSet(std::move(other));
  }
  return *this;
}

namespace {

bool dspark_bindings_match(
    const DeepSeekStagePlan& stage,
    DeepSeekRankComputeInfrastructureOwner& infrastructure) noexcept {
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  return stage.owns_dspark ==
         (infrastructure.dspark_resident_expert_bindings() != nullptr);
#else
  (void)infrastructure;
  return !stage.owns_dspark;
#endif
}

}  // namespace

Result<DeepSeekRankComputeLaneSet> DeepSeekRankComputeLaneSet::Create(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens, DeepSeekExpertPager& pager,
    std::unique_ptr<DeepSeekExpertTransferLaneOwner> transfer_lane,
    std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane) {
  if (transfer_lane == nullptr || kernel_lane == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek rank compute lane owners are required");
  }
  auto bundle = DeepSeekRankComputeBundle::Create(
      stage, maximum_sequences, maximum_tokens, pager,
      transfer_lane->transfer(), kernel_lane->kernel());
  if (!bundle.ok()) return bundle.status();
  auto owned_bundle =
      std::make_unique<DeepSeekRankComputeBundle>(std::move(*bundle));
  DeepSeekRankComputeLaneSet result;
  result.transfer_lane_ = std::move(transfer_lane);
  result.kernel_lane_ = std::move(kernel_lane);
  result.pager_ = &pager;
  result.bundle_ = std::move(owned_bundle);
  return result;
}

Result<DeepSeekRankComputeLaneSet> DeepSeekRankComputeLaneSet::CreateResident(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens,
    const DeepSeekResidentExpertBindings& resident_experts,
    std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane) {
  if (kernel_lane == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek resident rank compute kernel lane owner is required");
  }
  auto bundle = DeepSeekRankComputeBundle::CreateResident(
      stage, maximum_sequences, maximum_tokens, kernel_lane->kernel(),
      resident_experts);
  if (!bundle.ok()) return bundle.status();
  auto owned_bundle =
      std::make_unique<DeepSeekRankComputeBundle>(std::move(*bundle));
  DeepSeekRankComputeLaneSet result;
  result.kernel_lane_ = std::move(kernel_lane);
  result.bundle_ = std::move(owned_bundle);
  return result;
}

Result<DeepSeekRankComputeLaneSet>
DeepSeekRankComputeLaneSet::CreateOwnedInfrastructure(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens, DeepSeekExpertPager& pager,
    std::unique_ptr<DeepSeekRankComputeInfrastructureOwner> infrastructure,
    std::unique_ptr<DeepSeekExpertTransferLaneOwner> transfer_lane,
    std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane) {
  if (infrastructure == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek rank compute infrastructure owner is required");
  }
  if (!dspark_bindings_match(stage, *infrastructure)) {
    return Status::InvalidArgument(
        "DeepSeek owned compute infrastructure DSpark bindings disagree "
        "with the stage plan");
  }
  if (transfer_lane == nullptr || kernel_lane == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek rank compute lane owners are required");
  }
  auto bundle = DeepSeekRankComputeBundle::Create(
      stage, maximum_sequences, maximum_tokens, pager,
      transfer_lane->transfer(), kernel_lane->kernel(),
      infrastructure->attention_compute_stream());
  if (!bundle.ok()) return bundle.status();
  if (auto* shared = infrastructure->shared_expert_provider()) {
    auto status = bundle->bind_shared_experts(*shared);
    if (!status.ok()) return status;
  }
  auto owned_bundle =
      std::make_unique<DeepSeekRankComputeBundle>(std::move(*bundle));
  DeepSeekRankComputeLaneSet result;
  result.infrastructure_ = std::move(infrastructure);
  result.transfer_lane_ = std::move(transfer_lane);
  result.kernel_lane_ = std::move(kernel_lane);
  result.pager_ = &pager;
  result.bundle_ = std::move(owned_bundle);
  return result;
}

Result<DeepSeekRankComputeLaneSet>
DeepSeekRankComputeLaneSet::CreateOwnedResidentInfrastructure(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens,
    std::unique_ptr<DeepSeekRankComputeInfrastructureOwner> infrastructure,
    std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane) {
  if (infrastructure == nullptr || kernel_lane == nullptr ||
      infrastructure->resident_expert_bindings() == nullptr ||
      !dspark_bindings_match(stage, *infrastructure)) {
    return Status::InvalidArgument(
        "DeepSeek resident compute infrastructure is incomplete");
  }
  auto bundle = DeepSeekRankComputeBundle::CreateResident(
      stage, maximum_sequences, maximum_tokens, kernel_lane->kernel(),
      *infrastructure->resident_expert_bindings(),
      infrastructure->attention_compute_stream());
  if (!bundle.ok()) return bundle.status();
  if (auto* shared = infrastructure->shared_expert_provider()) {
    auto status = bundle->bind_shared_experts(*shared);
    if (!status.ok()) return status;
  }
  auto owned_bundle =
      std::make_unique<DeepSeekRankComputeBundle>(std::move(*bundle));
  DeepSeekRankComputeLaneSet result;
  result.infrastructure_ = std::move(infrastructure);
  result.kernel_lane_ = std::move(kernel_lane);
  result.bundle_ = std::move(owned_bundle);
  return result;
}

}  // namespace pih
