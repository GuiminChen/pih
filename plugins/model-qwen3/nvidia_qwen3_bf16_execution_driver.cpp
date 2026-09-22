#include "pih/model/nvidia_qwen3_bf16_execution_driver.h"

#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"

#if CUDA_VERSION < 13020
#error "PIH Qwen execution driver requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {

Result<NvidiaQwenBf16ExecutionDriver>
NvidiaQwenBf16ExecutionDriver::Create(const QwenBf16LinearPlanSet& plans,
                                      void* workspace,
                                      std::uint64_t workspace_bytes,
                                      std::int32_t owning_rank) {
  return Create(plans, plans, workspace, workspace_bytes, owning_rank);
}

Result<NvidiaQwenBf16ExecutionDriver>
NvidiaQwenBf16ExecutionDriver::Create(
    const QwenBf16LinearPlanSet& prefill_plans,
    const QwenBf16LinearPlanSet& decode_plans, void* workspace,
    std::uint64_t workspace_bytes, std::int32_t owning_rank) {
  return Create(prefill_plans, prefill_plans, decode_plans, workspace,
                workspace_bytes, owning_rank);
}

Result<NvidiaQwenBf16ExecutionDriver>
NvidiaQwenBf16ExecutionDriver::Create(
    const QwenBf16LinearPlanSet& prefill_plans,
    const QwenBf16LinearPlanSet& decode_plans,
    std::span<const QwenBf16LinearPlanSet* const> packed_lm_head_plans,
    void* workspace, std::uint64_t workspace_bytes,
    std::int32_t owning_rank) {
  return Create(prefill_plans, prefill_plans, decode_plans,
                packed_lm_head_plans, workspace, workspace_bytes,
                owning_rank);
}

Result<NvidiaQwenBf16ExecutionDriver>
NvidiaQwenBf16ExecutionDriver::Create(
    const QwenBf16LinearPlanSet& prefill_plans,
    const QwenBf16LinearPlanSet& tail_prefill_plans,
    const QwenBf16LinearPlanSet& decode_plans, void* workspace,
    std::uint64_t workspace_bytes, std::int32_t owning_rank) {
  return Create(prefill_plans, tail_prefill_plans, decode_plans, {}, workspace,
                workspace_bytes, owning_rank);
}

Result<NvidiaQwenBf16ExecutionDriver>
NvidiaQwenBf16ExecutionDriver::Create(
    const QwenBf16LinearPlanSet& prefill_plans,
    const QwenBf16LinearPlanSet& tail_prefill_plans,
    const QwenBf16LinearPlanSet& decode_plans,
    std::span<const QwenBf16LinearPlanSet* const> packed_lm_head_plans,
    void* workspace, std::uint64_t workspace_bytes,
    std::int32_t owning_rank) {
  auto selector = QwenBf16LinearShapeSelector::Create(
      prefill_plans.tokens(), tail_prefill_plans.tokens(),
      decode_plans.tokens());
  if (!selector.ok()) return selector.status();
  for (std::size_t index = 0; index < packed_lm_head_plans.size(); ++index) {
    const auto* plan = packed_lm_head_plans[index];
    if (plan == nullptr || plan->tokens() != index + 1 ||
        workspace_bytes < plan->workspace_bytes()) {
      return Status::InvalidArgument(
          "NVIDIA Qwen packed LM-head plan bank is invalid");
    }
  }
  if (owning_rank < 0) {
    return Status::InvalidArgument(
        "NVIDIA Qwen execution driver requires a nonnegative rank");
  }
  if (decode_plans.tokens() != 1 || prefill_plans.tokens() == 0 ||
      workspace_bytes < prefill_plans.workspace_bytes() ||
      workspace_bytes < tail_prefill_plans.workspace_bytes() ||
      workspace_bytes < decode_plans.workspace_bytes() ||
      ((prefill_plans.workspace_bytes() != 0 ||
        tail_prefill_plans.workspace_bytes() != 0 ||
        decode_plans.workspace_bytes() != 0) &&
       workspace == nullptr)) {
    return Status::InvalidArgument(
        "NVIDIA Qwen execution workspace is smaller than the plan requires");
  }
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "NVIDIA Qwen execution driver requires a current context");
  }
  CUdevice device = 0;
  result = cuCtxGetDevice(&device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetDevice");
  }
  if (device != owning_rank) {
    return Status::FailedPrecondition(
        "current CUDA device differs from Qwen execution owner");
  }
  return NvidiaQwenBf16ExecutionDriver(
      prefill_plans, tail_prefill_plans, decode_plans, workspace,
      workspace_bytes, owning_rank, reinterpret_cast<std::uintptr_t>(context),
      std::move(*selector), packed_lm_head_plans);
}

Status NvidiaQwenBf16ExecutionDriver::require_current_owner() const {
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (reinterpret_cast<std::uintptr_t>(context) != context_identity_) {
    return Status::FailedPrecondition(
        "current CUDA context differs from Qwen execution owner");
  }
  CUdevice device = 0;
  result = cuCtxGetDevice(&device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetDevice");
  }
  if (device != owning_rank_) {
    return Status::FailedPrecondition(
        "current CUDA device differs from Qwen execution owner");
  }
  return Status::Ok();
}


Status NvidiaQwenBf16ExecutionDriver::execute(
    const QwenBf16LinearBinding& binding, DriverStreamHandle stream) {
  if (stream == 0) {
    return Status::InvalidArgument(
        "NVIDIA Qwen linear execution requires an explicit stream");
  }
  const Status owner = require_current_owner();
  if (!owner.ok()) return owner;
  const auto rows = static_cast<std::uint64_t>(binding.input().dim(0));
  if (binding.kind() == QwenBf16LinearKind::kLmHead &&
      !packed_lm_head_plans_.empty()) {
    if (rows == 0 || rows > packed_lm_head_plans_.size()) {
      return Status::InvalidArgument(
          "packed Qwen LM-head rows exceed the frozen plan bank");
    }
    return packed_lm_head_plans_[rows - 1]->execute(
        binding.kind(), binding.input(), binding.weight(), binding.output(),
        workspace_, workspace_bytes_, reinterpret_cast<cudaStream_t>(stream));
  }
  auto role = selector_.select(rows);
  if (!role.ok()) return role.status();
  const QwenBf16LinearPlanSet* plans =
      *role == QwenBf16LinearShapeRole::kPrefill
          ? prefill_plans_
          : (*role == QwenBf16LinearShapeRole::kTailPrefill
                 ? tail_prefill_plans_ : decode_plans_);
  return plans->execute(binding.kind(), binding.input(), binding.weight(),
                        binding.output(), workspace_, workspace_bytes_,
                        reinterpret_cast<cudaStream_t>(stream));
}

}  // namespace pih
