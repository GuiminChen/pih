#pragma once
#include "pih/model/qwen3_bf16_engine_resource_plan.h"
#include "pih/model/qwen3_int4_artifact_layout.h"
namespace pih{
class QwenInt4EngineResourcePlan final{public:
 static Result<QwenInt4EngineResourcePlan>Create(std::uint32_t maximum_step_tokens,
  std::uint32_t maximum_sequence_tokens,std::uint32_t slot_count,
  std::uint64_t linear_workspace_bytes,
  std::uint32_t maximum_batch_sequences=1);
 [[nodiscard]]const QwenBf16EngineResourcePlan& runtime()const noexcept{return runtime_;}
 [[nodiscard]]std::uint64_t resident_weight_bytes()const noexcept{return resident_weight_bytes_;}
 [[nodiscard]]std::uint64_t canonical_pinned_bytes()const noexcept{return canonical_pinned_bytes_;}
 [[nodiscard]]std::uint64_t steady_pinned_bytes()const noexcept{return steady_pinned_bytes_;}
 [[nodiscard]]std::uint64_t startup_pinned_peak_bytes()const noexcept{return startup_pinned_peak_bytes_;}
 [[nodiscard]]std::uint64_t total_device_bytes()const noexcept{return total_device_bytes_;}
 Status admit(std::uint64_t available_device_bytes,
              std::uint64_t available_pinned_bytes,
              std::uint64_t device_reserve_bytes)const;
 private:QwenInt4EngineResourcePlan(QwenBf16EngineResourcePlan runtime,
  std::uint64_t steady,std::uint64_t startup,std::uint64_t device)
  :runtime_(std::move(runtime)),resident_weight_bytes_(QwenInt4ArtifactLayout::kOfficialLogicalPayloadBytes),
   canonical_pinned_bytes_(QwenInt4ArtifactLayout::kOfficialFileBytes),steady_pinned_bytes_(steady),
   startup_pinned_peak_bytes_(startup),total_device_bytes_(device){}
 QwenBf16EngineResourcePlan runtime_;std::uint64_t resident_weight_bytes_,canonical_pinned_bytes_,
  steady_pinned_bytes_,startup_pinned_peak_bytes_,total_device_bytes_;
};}
