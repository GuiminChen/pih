#pragma once

#include "pih/model/deepseek_expert_subwave_executor.h"
#include "pih/model/deepseek_expert_compute_arena.h"

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkExpertKernelDriver;
#endif

class DeepSeekRankAttentionResources;
class DeepSeekRankAttentionStatePool;
class DeepSeekLearnedRouterOperations;
class DeepSeekLearnedRouterDeviceResources;
class DeepSeekLearnedRouterStagingPool;
class DeepSeekHashRouterStagingPool;
class DeepSeekEndpointRuntimeResources;
class DeepSeekEndpointDeviceResources;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkRuntimeResources;
class DeepSeekDsparkDeviceResources;
class DeepSeekDsparkResidentExpertBindings;
#endif
class DeepSeekResidentExpertBindings;
class DeepSeekDenseMhcRuntimeResources;
class DeepSeekAttentionProjectionDeviceResources;
class DeepSeekMhcDeviceResources;
class DeepSeekRopeTableDeviceResources;
class DeepSeekRequestInputStagingResources;
class DeepSeekRequestInputCopyOperations;
class DeepSeekSharedExpertProvider;

class DeepSeekRankComputeInfrastructureOwner {
 public:
  virtual ~DeepSeekRankComputeInfrastructureOwner() = default;
  virtual DeepSeekSharedExpertProvider* shared_expert_provider() noexcept { return nullptr; }
  virtual DeepSeekRankAttentionResources* attention_resources() noexcept {
    return nullptr;
  }
  virtual DeepSeekRankAttentionStatePool* attention_state_pool() noexcept {
    return nullptr;
  }
  virtual std::uintptr_t attention_compute_stream() const noexcept {
    return 0;
  }
  virtual DeepSeekLearnedRouterDeviceResources*
  learned_router_device_resources() noexcept { return nullptr; }
  virtual DeepSeekLearnedRouterStagingPool* learned_router_staging_pool()
      noexcept { return nullptr; }
  virtual DeepSeekHashRouterStagingPool* hash_router_staging_pool()
      noexcept { return nullptr; }
  virtual DeepSeekEndpointRuntimeResources* endpoint_runtime_resources()
      noexcept { return nullptr; }
  virtual DeepSeekEndpointDeviceResources* endpoint_device_resources()
      noexcept { return nullptr; }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  virtual DeepSeekDsparkRuntimeResources* dspark_runtime_resources()
      noexcept { return nullptr; }
  virtual DeepSeekDsparkDeviceResources* dspark_device_resources()
      noexcept { return nullptr; }
  virtual const DeepSeekDsparkResidentExpertBindings*
  dspark_resident_expert_bindings() const noexcept { return nullptr; }
#endif
  virtual const DeepSeekResidentExpertBindings*
  resident_expert_bindings() const noexcept { return nullptr; }
  virtual DeepSeekDenseMhcRuntimeResources* dense_mhc_runtime_resources()
      noexcept { return nullptr; }
  virtual DeepSeekAttentionProjectionDeviceResources*
  attention_projection_device_resources() noexcept { return nullptr; }
  virtual DeepSeekMhcDeviceResources* mhc_device_resources() noexcept {
    return nullptr;
  }
  virtual DeepSeekRopeTableDeviceResources* rope_table_device_resources()
      noexcept { return nullptr; }
  virtual DeepSeekRequestInputStagingResources* request_input_staging_resources()
      noexcept { return nullptr; }
  virtual DeepSeekExpertComputeArena expert_compute_arena() const noexcept {
    return {};
  }
  virtual std::uintptr_t expert_accumulator_f32() const noexcept {
    return 0;
  }
};

class DeepSeekRankRuntimeOwner {
 public:
  virtual ~DeepSeekRankRuntimeOwner() = default;
  virtual Status activate() = 0;
  virtual DeepSeekLearnedRouterOperations* learned_router_operations()
      noexcept {
    return nullptr;
  }
  virtual std::uintptr_t plan_compute_stream() const noexcept { return 0; }
  virtual std::uintptr_t plan_completion_event() const noexcept { return 0; }
  virtual DeepSeekRequestInputCopyOperations* request_input_copy_operations()
      noexcept { return nullptr; }
};

class DeepSeekExpertTransferLaneOwner {
 public:
  virtual ~DeepSeekExpertTransferLaneOwner() = default;
  virtual DeepSeekExpertTransferDriver& transfer() noexcept = 0;
};

class DeepSeekExpertKernelLaneOwner {
 public:
  virtual ~DeepSeekExpertKernelLaneOwner() = default;
  virtual DeepSeekExpertKernelDriver& kernel() noexcept = 0;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  virtual DeepSeekDsparkExpertKernelDriver* dspark_kernel() noexcept {
    return nullptr;
  }
#endif
};

}  // namespace pih
