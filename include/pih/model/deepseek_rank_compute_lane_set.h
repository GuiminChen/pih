#pragma once

#include <memory>

#include "pih/model/deepseek_expert_lane_owner.h"
#include "pih/model/deepseek_rank_compute_bundle.h"

namespace pih {

class DeepSeekRankComputeLaneSet final {
 public:
  static Result<DeepSeekRankComputeLaneSet> Create(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, DeepSeekExpertPager& pager,
      std::unique_ptr<DeepSeekExpertTransferLaneOwner> transfer_lane,
      std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane);
  static Result<DeepSeekRankComputeLaneSet> CreateResident(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens,
      const DeepSeekResidentExpertBindings& resident_experts,
      std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane);
  static Result<DeepSeekRankComputeLaneSet> CreateOwnedInfrastructure(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, DeepSeekExpertPager& pager,
      std::unique_ptr<DeepSeekRankComputeInfrastructureOwner> infrastructure,
      std::unique_ptr<DeepSeekExpertTransferLaneOwner> transfer_lane,
      std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane);
  static Result<DeepSeekRankComputeLaneSet>
  CreateOwnedResidentInfrastructure(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens,
      std::unique_ptr<DeepSeekRankComputeInfrastructureOwner> infrastructure,
      std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane);

  DeepSeekRankComputeLaneSet(const DeepSeekRankComputeLaneSet&) = delete;
  DeepSeekRankComputeLaneSet& operator=(const DeepSeekRankComputeLaneSet&) =
      delete;
  DeepSeekRankComputeLaneSet(DeepSeekRankComputeLaneSet&&) noexcept = default;
  DeepSeekRankComputeLaneSet& operator=(
      DeepSeekRankComputeLaneSet&& other) noexcept;

  [[nodiscard]] DeepSeekRankComputeBundle& bundle() noexcept {
    return *bundle_;
  }
  [[nodiscard]] DeepSeekStagePlan stage() const noexcept {
    return bundle_->stage();
  }
  [[nodiscard]] DeepSeekExpertPager* pager_identity() const noexcept {
    return pager_;
  }
  [[nodiscard]] DeepSeekRankAttentionResources* attention_resources()
      noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->attention_resources();
  }
  [[nodiscard]] DeepSeekRankAttentionStatePool* attention_state_pool()
      noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->attention_state_pool();
  }
  [[nodiscard]] DeepSeekLearnedRouterDeviceResources*
  learned_router_device_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->learned_router_device_resources();
  }
  [[nodiscard]] DeepSeekLearnedRouterStagingPool*
  learned_router_staging_pool() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->learned_router_staging_pool();
  }
  [[nodiscard]] DeepSeekHashRouterStagingPool*
  hash_router_staging_pool() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->hash_router_staging_pool();
  }
  [[nodiscard]] DeepSeekEndpointRuntimeResources*
  endpoint_runtime_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->endpoint_runtime_resources();
  }
  [[nodiscard]] DeepSeekEndpointDeviceResources*
  endpoint_device_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->endpoint_device_resources();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkRuntimeResources*
  dspark_runtime_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->dspark_runtime_resources();
  }
  [[nodiscard]] DeepSeekDsparkDeviceResources*
  dspark_device_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->dspark_device_resources();
  }
#endif
  [[nodiscard]] DeepSeekDenseMhcRuntimeResources*
  dense_mhc_runtime_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->dense_mhc_runtime_resources();
  }
  [[nodiscard]] DeepSeekAttentionProjectionDeviceResources*
  attention_projection_device_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->attention_projection_device_resources();
  }
  [[nodiscard]] DeepSeekMhcDeviceResources* mhc_device_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->mhc_device_resources();
  }
  [[nodiscard]] DeepSeekRopeTableDeviceResources*
  rope_table_device_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->rope_table_device_resources();
  }
  [[nodiscard]] DeepSeekRequestInputStagingResources*
  request_input_staging_resources() noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->request_input_staging_resources();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkExpertKernelDriver*
  dspark_expert_kernel() noexcept {
    return kernel_lane_ == nullptr ? nullptr
                                   : kernel_lane_->dspark_kernel();
  }
#endif
  [[nodiscard]] DeepSeekExpertComputeArena expert_compute_arena()
      const noexcept {
    return infrastructure_ == nullptr
               ? DeepSeekExpertComputeArena{}
               : infrastructure_->expert_compute_arena();
  }
  [[nodiscard]] std::uintptr_t expert_accumulator_f32() const noexcept {
    return infrastructure_ == nullptr
               ? 0
               : infrastructure_->expert_accumulator_f32();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] const DeepSeekDsparkResidentExpertBindings*
  dspark_resident_expert_bindings() const noexcept {
    return infrastructure_ == nullptr
               ? nullptr
               : infrastructure_->dspark_resident_expert_bindings();
  }
#endif

 private:
  DeepSeekRankComputeLaneSet() = default;
  // Outlives both lane drivers, which may borrow runtime/device resources.
  std::unique_ptr<DeepSeekRankComputeInfrastructureOwner> infrastructure_;
  std::unique_ptr<DeepSeekExpertTransferLaneOwner> transfer_lane_;
  std::unique_ptr<DeepSeekExpertKernelLaneOwner> kernel_lane_;
  DeepSeekExpertPager* pager_ = nullptr;
  // Destroyed before its lane drivers.
  std::unique_ptr<DeepSeekRankComputeBundle> bundle_;
};

}  // namespace pih
