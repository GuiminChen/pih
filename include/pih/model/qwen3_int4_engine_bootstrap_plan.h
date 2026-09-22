#pragma once
#include "pih/model/qwen3_int4_engine_model_plan.h"
#include "pih/model/qwen3_int4_engine_resource_plan.h"
#include "pih/backend/cuda/cuda_runtime_resources.h"
#include "pih/model/qwen3_int4_synchronous_backend.h"
namespace pih{
class QwenInt4EngineBootstrapPlan final{public:
 static Result<QwenInt4EngineBootstrapPlan>Create(const Qwen3Config& config,
  std::uint32_t maximum_step_tokens,std::uint32_t maximum_sequence_tokens,
  std::uint32_t slot_count,std::uint64_t linear_workspace_bytes,
  std::uint64_t available_device_bytes,std::uint64_t available_pinned_bytes,
  std::uint64_t device_reserve_bytes,
  std::uint32_t maximum_batch_sequences=1);
 static Result<QwenInt4EngineBootstrapPlan>Create(QwenInt4EngineModelPlan model,
  std::uint32_t maximum_step_tokens,std::uint32_t maximum_sequence_tokens,
  std::uint32_t slot_count,std::uint64_t linear_workspace_bytes,
  std::uint64_t available_device_bytes,std::uint64_t available_pinned_bytes,
  std::uint64_t device_reserve_bytes,
  std::uint32_t maximum_batch_sequences=1);
 [[nodiscard]]const QwenInt4EngineModelPlan& model()const noexcept{return model_;}
 [[nodiscard]]const QwenInt4EngineResourcePlan& resources()const noexcept{return resources_;}
 Result<QwenInt4SynchronousBackendIdentity> backend_identity(
  const CudaRuntimeResourceIdentity& runtime,std::uint64_t epoch,
  std::uint64_t timeout_ns)const;
 private:QwenInt4EngineBootstrapPlan(QwenInt4EngineModelPlan model,
  QwenInt4EngineResourcePlan resources):model_(std::move(model)),resources_(std::move(resources)){}
 QwenInt4EngineModelPlan model_;QwenInt4EngineResourcePlan resources_;
};}
