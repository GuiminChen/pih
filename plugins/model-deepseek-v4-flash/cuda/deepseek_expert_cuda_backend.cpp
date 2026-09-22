#include "pih/backend/cuda/deepseek_expert_cuda_backend.h"

namespace pih {

Result<DeepSeekExpertCudaBackend> DeepSeekExpertCudaBackend::Create(
    DeepSeekExpertCudaOperations& operations, DeepSeekExpertHostStaging staging,
    std::uintptr_t completion_event, std::uint64_t context_identity) {
  if (staging.route_weights == nullptr || staging.token_indices == nullptr ||
      staging.error_flag == nullptr || staging.capacity == 0 ||
      completion_event == 0 || context_identity == 0) {
    return Status::InvalidArgument("DeepSeek CUDA backend resources are invalid");
  }
  const auto validated = operations.validate_host_staging(staging);
  if (!validated.ok()) return validated;
  DeepSeekExpertCudaBackend backend;
  backend.operations_ = &operations;
  backend.staging_ = staging;
  backend.completion_event_ = completion_event;
  backend.context_identity_ = context_identity;
  return backend;
}

Status DeepSeekExpertCudaBackend::submit(
    const DeepSeekExpertComputeSubmission& submission) {
  if (poisoned_ || inflight_) {
    return Status::FailedPrecondition("DeepSeek CUDA backend is not launchable");
  }
  if (submission.context_identity != context_identity_ ||
      submission.routes == nullptr || submission.route_count == 0 ||
      submission.route_count > staging_.capacity || submission.stream == 0) {
    return Status::InvalidArgument("DeepSeek CUDA submission is incompatible");
  }
  for (std::uint32_t i = 0; i < submission.route_count; ++i) {
    staging_.route_weights[i] = submission.routes[i].weight;
    staging_.token_indices[i] = submission.routes[i].token_index;
  }
  *staging_.error_flag = 0;

  const auto& a = submission.arena;
  const auto stream = submission.stream;
  auto run = [this](Status status) {
    if (!status.ok()) poisoned_ = true;
    return status;
  };
  Status status = run(operations_->zero_u32_async(a.error_flag_u32.address, stream));
  if (!status.ok()) return status;
  status = run(operations_->copy_h2d_async(
      a.route_weights_f32.address, staging_.route_weights,
      submission.route_count * sizeof(float), stream));
  if (!status.ok()) return status;
  status = run(operations_->copy_h2d_async(
      a.token_indices_u32.address, staging_.token_indices,
      submission.route_count * sizeof(std::uint32_t), stream));
  if (!status.ok()) return status;
  status = run(operations_->gather({submission.source_hidden_bf16,
                                    a.token_indices_u32.address,
                                    a.route_input_bf16.address,
                                    a.error_flag_u32.address, stream,
                                    submission.route_count,
                                    submission.packed_token_count}));
  if (!status.ok()) return status;

  auto quantize = [&](std::uintptr_t input, std::uint32_t k) {
    return run(operations_->quantize(
        {input, a.activation_e4m3.address, a.activation_scale_bits.address,
         a.error_flag_u32.address, stream, submission.route_count, k}));
  };
  auto gemm = [&](const DeepSeekExpertMatrixDeviceView& matrix,
                  std::uintptr_t output, std::uint32_t n, std::uint32_t k) {
    return run(operations_->gemm(
        {a.activation_e4m3.address, a.activation_scale_bits.address,
         matrix.packed.address, matrix.scales.address, output,
         a.error_flag_u32.address, stream, submission.route_count, n, k}));
  };
  status = quantize(a.route_input_bf16.address, 4096);
  if (!status.ok()) return status;
  status = gemm(submission.bundle.w1, a.gate_or_middle_bf16.address, 2048, 4096);
  if (!status.ok()) return status;
  status = gemm(submission.bundle.w3, a.up_bf16.address, 2048, 4096);
  if (!status.ok()) return status;
  status = run(operations_->swiglu(
      {a.gate_or_middle_bf16.address, a.up_bf16.address,
       a.route_weights_f32.address, a.gate_or_middle_bf16.address,
       a.error_flag_u32.address, stream, submission.route_count}));
  if (!status.ok()) return status;
  status = quantize(a.gate_or_middle_bf16.address, 2048);
  if (!status.ok()) return status;
  status = gemm(submission.bundle.w2, a.expert_output_bf16.address, 4096, 2048);
  if (!status.ok()) return status;
  status = run(operations_->accumulate(
      {a.expert_output_bf16.address, a.token_indices_u32.address,
       submission.accumulator_f32, a.error_flag_u32.address, stream,
       submission.route_count, submission.packed_token_count}));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      staging_.error_flag, a.error_flag_u32.address,
      sizeof(std::uint32_t), stream));
  if (!status.ok()) return status;
  status = run(operations_->record_event(completion_event_, stream));
  if (!status.ok()) return status;
  inflight_ = true;
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus> DeepSeekExpertCudaBackend::poll() {
  if (poisoned_) return DeepSeekExpertAsyncStatus::kError;
  if (!inflight_) {
    return Status::FailedPrecondition("DeepSeek CUDA backend has no submission");
  }
  auto event = operations_->query_event(completion_event_);
  if (!event.ok()) {
    poisoned_ = true;
    return event.status();
  }
  if (*event == DeepSeekExpertAsyncStatus::kInProgress) return *event;
  if (*event == DeepSeekExpertAsyncStatus::kError || *staging_.error_flag != 0) {
    poisoned_ = true;
    return DeepSeekExpertAsyncStatus::kError;
  }
  if (*event != DeepSeekExpertAsyncStatus::kSuccess) {
    poisoned_ = true;
    return Status::Internal(
        "DeepSeek CUDA backend received invalid event state");
  }
  inflight_ = false;
  return DeepSeekExpertAsyncStatus::kSuccess;
}

}  // namespace pih
