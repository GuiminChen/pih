#pragma once

#include <memory>

#include "pih/model/deepseek_expert_lane_owner.h"
#include "pih/model/deepseek_rank_expert_device_resources.h"
#include "pih/model/deepseek_rank_attention_resources.h"
#include "pih/model/deepseek_rank_attention_state_pool.h"
#include "pih/model/deepseek_learned_router_device_resources.h"
#include "pih/model/deepseek_learned_router_staging_pool.h"
#include "pih/model/deepseek_hash_router_staging_pool.h"
#include "pih/model/deepseek_endpoint_runtime_resources.h"
#include "pih/model/deepseek_endpoint_device_resources.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_runtime_resources.h"
#include "pih/model/deepseek_dspark_device_resources.h"
#include "pih/model/deepseek_dspark_resident_expert_bindings.h"
#endif
#include "pih/model/deepseek_resident_expert_bindings.h"
#include "pih/model/deepseek_dense_mhc_runtime_resources.h"
#include "pih/model/deepseek_attention_projection_device_resources.h"
#include "pih/model/deepseek_mhc_device_resources.h"
#include "pih/model/deepseek_rope_table_device_resources.h"
#include "pih/model/deepseek_request_input_staging_resources.h"

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkRuntimeResources;
class DeepSeekDsparkDeviceResources;
class DeepSeekDsparkResidentExpertBindings;
#endif

class DeepSeekRankComputeInfrastructure final
    : public DeepSeekRankComputeInfrastructureOwner {
 public:
  static Result<std::unique_ptr<DeepSeekRankComputeInfrastructure>> Create(
      DeepSeekRankExpertDeviceResources device_resources) {
    return std::unique_ptr<DeepSeekRankComputeInfrastructure>(
        new DeepSeekRankComputeInfrastructure(
            std::move(device_resources)));
  }
  static Result<std::unique_ptr<DeepSeekRankComputeInfrastructure>>
  CreateComplete(
      DeepSeekRankExpertDeviceResources device_resources,
      DeepSeekRankAttentionStatePool attention_state,
      DeepSeekRankAttentionResources attention,
      DeepSeekLearnedRouterDeviceResources learned_router_device,
      DeepSeekHashRouterStagingPool hash_router_staging,
      DeepSeekLearnedRouterStagingPool learned_router_staging,
      DeepSeekEndpointRuntimeResources endpoint,
      DeepSeekEndpointDeviceResources endpoint_device,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      std::unique_ptr<DeepSeekDsparkRuntimeResources> dspark,
      std::unique_ptr<DeepSeekDsparkDeviceResources> dspark_device,
#endif
      DeepSeekDenseMhcRuntimeResources dense_mhc,
      DeepSeekAttentionProjectionDeviceResources attention_projection_device,
      DeepSeekMhcDeviceResources mhc_device,
      DeepSeekRopeTableDeviceResources rope_tables,
      DeepSeekRequestInputStagingResources request_input_staging,
      std::uintptr_t attention_compute_stream,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      std::unique_ptr<DeepSeekDsparkResidentExpertBindings>
          dspark_resident_experts = nullptr,
#endif
      std::unique_ptr<DeepSeekResidentExpertBindings>
          resident_experts = nullptr) {
    if (attention_compute_stream == 0) {
      return Status::InvalidArgument(
          "DeepSeek attention compute stream is required");
    }
    return std::unique_ptr<DeepSeekRankComputeInfrastructure>(
        new DeepSeekRankComputeInfrastructure(
            std::move(device_resources), std::move(attention_state),
            std::move(attention), std::move(learned_router_device),
            std::move(hash_router_staging),
            std::move(learned_router_staging), std::move(endpoint),
            std::move(endpoint_device),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
            std::move(dspark), std::move(dspark_device),
#endif
            std::move(dense_mhc),
            std::move(attention_projection_device),
            std::move(mhc_device),
            std::move(rope_tables),
            std::move(request_input_staging),
            attention_compute_stream,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
            std::move(dspark_resident_experts),
#endif
            std::move(resident_experts)));
  }

  DeepSeekRankComputeInfrastructure(
      const DeepSeekRankComputeInfrastructure&) = delete;
  DeepSeekRankComputeInfrastructure& operator=(
      const DeepSeekRankComputeInfrastructure&) = delete;

  [[nodiscard]] DeepSeekRankExpertDeviceResources& device_resources()
      noexcept {
    return device_resources_;
  }
  Status attach_shared_experts(std::unique_ptr<DeepSeekSharedExpertRuntimeResources> resources) {
    if (resources == nullptr || shared_experts_ != nullptr) {
      return Status::FailedPrecondition("Shared expert resources are missing or already attached");
    }
    shared_experts_ = std::move(resources);
    return Status::Ok();
  }
  DeepSeekSharedExpertProvider* shared_expert_provider() noexcept override {
    return shared_experts_.get();
  }
  DeepSeekRankAttentionResources* attention_resources() noexcept override {
    return attention_ == nullptr ? nullptr : attention_.get();
  }
  DeepSeekRankAttentionStatePool* attention_state_pool() noexcept override {
    return attention_state_ == nullptr ? nullptr : attention_state_.get();
  }
  std::uintptr_t attention_compute_stream() const noexcept override {
    return attention_compute_stream_;
  }
  DeepSeekLearnedRouterDeviceResources* learned_router_device_resources()
      noexcept override { return learned_router_device_.get(); }
  DeepSeekLearnedRouterStagingPool* learned_router_staging_pool()
      noexcept override { return learned_router_staging_.get(); }
  DeepSeekHashRouterStagingPool* hash_router_staging_pool()
      noexcept override { return hash_router_staging_.get(); }
  DeepSeekEndpointRuntimeResources* endpoint_runtime_resources()
      noexcept override { return endpoint_.get(); }
  DeepSeekEndpointDeviceResources* endpoint_device_resources()
      noexcept override { return endpoint_device_.get(); }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  DeepSeekDsparkRuntimeResources* dspark_runtime_resources()
      noexcept override {
    return dspark_.get();
  }
  DeepSeekDsparkDeviceResources* dspark_device_resources()
      noexcept override {
    return dspark_device_.get();
  }
  const DeepSeekDsparkResidentExpertBindings*
  dspark_resident_expert_bindings() const noexcept override {
    return dspark_resident_experts_.get();
  }
#endif
  const DeepSeekResidentExpertBindings*
  resident_expert_bindings() const noexcept override {
    return resident_experts_.get();
  }
  DeepSeekDenseMhcRuntimeResources* dense_mhc_runtime_resources()
      noexcept override { return dense_mhc_.get(); }
  DeepSeekAttentionProjectionDeviceResources*
  attention_projection_device_resources() noexcept override {
    return attention_projection_device_.get();
  }
  DeepSeekMhcDeviceResources* mhc_device_resources() noexcept override {
    return mhc_device_.get();
  }
  DeepSeekRopeTableDeviceResources* rope_table_device_resources()
      noexcept override { return rope_tables_.get(); }
  DeepSeekRequestInputStagingResources* request_input_staging_resources()
      noexcept override { return request_input_staging_.get(); }
  DeepSeekExpertComputeArena expert_compute_arena() const noexcept override {
    return device_resources_.arena();
  }
  std::uintptr_t expert_accumulator_f32() const noexcept override {
    return device_resources_.accumulator_f32();
  }

 private:
  DeepSeekRankComputeInfrastructure(
      DeepSeekRankExpertDeviceResources device_resources)
      : device_resources_(std::move(device_resources)) {}
  DeepSeekRankComputeInfrastructure(
      DeepSeekRankExpertDeviceResources device_resources,
      DeepSeekRankAttentionStatePool attention_state,
      DeepSeekRankAttentionResources attention,
      DeepSeekLearnedRouterDeviceResources learned_router_device,
      DeepSeekHashRouterStagingPool hash_router_staging,
      DeepSeekLearnedRouterStagingPool learned_router_staging,
      DeepSeekEndpointRuntimeResources endpoint,
      DeepSeekEndpointDeviceResources endpoint_device,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      std::unique_ptr<DeepSeekDsparkRuntimeResources> dspark,
      std::unique_ptr<DeepSeekDsparkDeviceResources> dspark_device,
#endif
      DeepSeekDenseMhcRuntimeResources dense_mhc,
      DeepSeekAttentionProjectionDeviceResources attention_projection_device,
      DeepSeekMhcDeviceResources mhc_device,
      DeepSeekRopeTableDeviceResources rope_tables,
      DeepSeekRequestInputStagingResources request_input_staging,
      std::uintptr_t attention_compute_stream,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      std::unique_ptr<DeepSeekDsparkResidentExpertBindings>
          dspark_resident_experts,
#endif
      std::unique_ptr<DeepSeekResidentExpertBindings> resident_experts)
      : device_resources_(std::move(device_resources)),
        attention_state_(std::make_unique<DeepSeekRankAttentionStatePool>(
            std::move(attention_state))),
        attention_(std::make_unique<DeepSeekRankAttentionResources>(
            std::move(attention))),
        learned_router_device_(
            std::make_unique<DeepSeekLearnedRouterDeviceResources>(
                std::move(learned_router_device))),
        hash_router_staging_(
            std::make_unique<DeepSeekHashRouterStagingPool>(
                std::move(hash_router_staging))),
        learned_router_staging_(
            std::make_unique<DeepSeekLearnedRouterStagingPool>(
                std::move(learned_router_staging))),
        endpoint_(std::make_unique<DeepSeekEndpointRuntimeResources>(
            std::move(endpoint))),
        endpoint_device_(std::make_unique<DeepSeekEndpointDeviceResources>(
            std::move(endpoint_device))),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        dspark_(std::move(dspark)),
        dspark_device_(std::move(dspark_device)),
#endif
        dense_mhc_(std::make_unique<DeepSeekDenseMhcRuntimeResources>(
            std::move(dense_mhc))),
        attention_projection_device_(
            std::make_unique<DeepSeekAttentionProjectionDeviceResources>(
                std::move(attention_projection_device))),
        mhc_device_(std::make_unique<DeepSeekMhcDeviceResources>(
            std::move(mhc_device))),
        rope_tables_(std::make_unique<DeepSeekRopeTableDeviceResources>(
            std::move(rope_tables))),
        request_input_staging_(
            std::make_unique<DeepSeekRequestInputStagingResources>(
                std::move(request_input_staging))),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        dspark_resident_experts_(std::move(dspark_resident_experts)),
#endif
        resident_experts_(std::move(resident_experts)),
        attention_compute_stream_(attention_compute_stream) {}

  DeepSeekRankExpertDeviceResources device_resources_;
  std::unique_ptr<DeepSeekRankAttentionStatePool> attention_state_;
  std::unique_ptr<DeepSeekRankAttentionResources> attention_;
  std::unique_ptr<DeepSeekLearnedRouterDeviceResources>
      learned_router_device_;
  std::unique_ptr<DeepSeekHashRouterStagingPool> hash_router_staging_;
  std::unique_ptr<DeepSeekLearnedRouterStagingPool>
      learned_router_staging_;
  std::unique_ptr<DeepSeekEndpointRuntimeResources> endpoint_;
  std::unique_ptr<DeepSeekEndpointDeviceResources> endpoint_device_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekDsparkRuntimeResources> dspark_;
  std::unique_ptr<DeepSeekDsparkDeviceResources> dspark_device_;
#endif
  std::unique_ptr<DeepSeekDenseMhcRuntimeResources> dense_mhc_;
  std::unique_ptr<DeepSeekAttentionProjectionDeviceResources>
      attention_projection_device_;
  std::unique_ptr<DeepSeekMhcDeviceResources> mhc_device_;
  std::unique_ptr<DeepSeekRopeTableDeviceResources> rope_tables_;
  std::unique_ptr<DeepSeekRequestInputStagingResources>
      request_input_staging_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekDsparkResidentExpertBindings>
      dspark_resident_experts_;
#endif
  std::unique_ptr<DeepSeekResidentExpertBindings> resident_experts_;
  std::uintptr_t attention_compute_stream_ = 0;
  // Drivers/adapter are destroyed before borrowed device resources.
  std::unique_ptr<DeepSeekSharedExpertRuntimeResources> shared_experts_;
};

}  // namespace pih
