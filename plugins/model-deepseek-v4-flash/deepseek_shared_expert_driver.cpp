#include "pih/model/deepseek_shared_expert_driver.h"

#include <array>
#include <string>

#include "pih/model/deepseek_resident_weight_arena.h"

namespace pih {
namespace {

bool fits(const DeepSeekExpertArenaSpan& span, std::uint64_t bytes) {
  return span.address != 0 && span.bytes >= bytes;
}

}  // namespace

Result<DeepSeekSharedExpertWeights> DeepSeekSharedExpertWeights::Resolve(
    std::uint32_t layer, const Resolver& resolver) {
  if (layer >= 43 || !resolver) {
    return Status::InvalidArgument("DeepSeek shared expert layer or resolver is invalid");
  }
  DeepSeekSharedExpertWeights result;
  const auto prefix = "layers." + std::to_string(layer) + ".ffn.shared_experts.";
  struct Matrix {
    const char* name;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uintptr_t* weight;
    std::uintptr_t* scale;
  };
  const std::array matrices{
      Matrix{"w1", 2048, 4096, &result.w1_e4m3, &result.w1_scale_bits},
      Matrix{"w3", 2048, 4096, &result.w3_e4m3, &result.w3_scale_bits},
      Matrix{"w2", 4096, 2048, &result.w2_e4m3, &result.w2_scale_bits}};
  for (const auto& matrix : matrices) {
    for (const bool scale : {false, true}) {
      auto view = resolver(prefix + matrix.name + (scale ? ".scale" : ".weight"));
      if (!view.ok()) return view.status();
      const auto rows = matrix.rows / (scale ? 128U : 1U);
      const auto columns = matrix.columns / (scale ? 128U : 1U);
      if (view->data() == nullptr || view->rank() != 2 ||
          view->dtype() != (scale ? DType::kFloat8E8M0 : DType::kFloat8E4M3) ||
          view->dim(0) != rows || view->dim(1) != columns ||
          view->stride(0) != columns || view->stride(1) != 1 ||
          view->device().type() != DeviceType::kCuda ||
          view->device().index() < 0 || view->generation() == 0 ||
          (result.generation != 0 && result.generation != view->generation()) ||
          (result.device_ordinal >= 0 && result.device_ordinal != view->device().index())) {
        return Status::FailedPrecondition(
            "DeepSeek shared expert weight violates FP8 checkpoint geometry or identity");
      }
      result.generation = view->generation();
      result.device_ordinal = view->device().index();
      *(scale ? matrix.scale : matrix.weight) = reinterpret_cast<std::uintptr_t>(view->data());
    }
  }
  return result;
}

Result<DeepSeekSharedExpertWeights> DeepSeekSharedExpertWeights::Resolve(
    std::uint32_t layer, const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(layer, [&arena](std::string_view name) { return arena.tensor(name); });
  if (!result.ok()) return result.status();
  if (result->generation != arena.generation() ||
      result->device_ordinal != arena.device().index()) {
    return Status::FailedPrecondition("DeepSeek shared expert resident identity differs");
  }
  return result;
}

Result<DeepSeekSharedExpertDriver> DeepSeekSharedExpertDriver::Create(
    std::uint32_t maximum_tokens, DeepSeekSharedExpertWeights weights,
    DeepSeekSharedExpertArena arena, std::uintptr_t input_bf16,
    std::uintptr_t accumulator_f32, std::uintptr_t output_bf16,
    std::uintptr_t stream, std::uintptr_t completion_event,
    std::uint32_t* host_error, DeepSeekSharedExpertOperations& operations) {
  const auto tokens = static_cast<std::uint64_t>(maximum_tokens);
  if (maximum_tokens == 0 || maximum_tokens > kMaximumTokens ||
      weights.w1_e4m3 == 0 || weights.w1_scale_bits == 0 ||
      weights.w3_e4m3 == 0 || weights.w3_scale_bits == 0 ||
      weights.w2_e4m3 == 0 || weights.w2_scale_bits == 0 ||
      input_bf16 == 0 || accumulator_f32 == 0 || output_bf16 == 0 ||
      stream == 0 || completion_event == 0 || host_error == nullptr ||
      !fits(arena.activation_e4m3, tokens * kHiddenSize) ||
      !fits(arena.activation_scale_bits, tokens * (kHiddenSize / 128U)) ||
      !fits(arena.gate_or_middle_bf16,
            tokens * kIntermediateSize * sizeof(std::uint16_t)) ||
      !fits(arena.up_bf16,
            tokens * kIntermediateSize * sizeof(std::uint16_t)) ||
      !fits(arena.shared_output_bf16,
            tokens * kHiddenSize * sizeof(std::uint16_t)) ||
      !fits(arena.error_flag_u32, sizeof(std::uint32_t))) {
    return Status::InvalidArgument(
        "DeepSeek shared expert driver resources are invalid");
  }
  auto valid_host = operations.validate_host_error(host_error);
  if (!valid_host.ok()) return valid_host;
  DeepSeekSharedExpertDriver driver;
  driver.maximum_tokens_ = maximum_tokens;
  driver.weights_ = weights;
  driver.arena_ = arena;
  driver.input_bf16_ = input_bf16;
  driver.accumulator_f32_ = accumulator_f32;
  driver.output_bf16_ = output_bf16;
  driver.stream_ = stream;
  driver.completion_event_ = completion_event;
  driver.host_error_ = host_error;
  driver.operations_ = &operations;
  return driver;
}

Status DeepSeekSharedExpertDriver::execute(std::uint32_t token_count) {
  if (!can_execute(token_count)) {
    return Status::FailedPrecondition(
        "DeepSeek shared expert driver is not executable");
  }
  auto run = [this](Status status) {
    if (!status.ok()) poisoned_ = true;
    return status;
  };
  inflight_ = true;
  *host_error_ = 0;
  auto status = run(operations_->zero_u32_async(arena_.error_flag_u32.address,
                                                 stream_));
  if (!status.ok()) return status;
  status = run(operations_->quantize(
      {input_bf16_, arena_.activation_e4m3.address,
       arena_.activation_scale_bits.address, arena_.error_flag_u32.address,
       stream_, token_count, kHiddenSize}));
  if (!status.ok()) return status;
  status = run(operations_->gemm(
      {arena_.activation_e4m3.address, arena_.activation_scale_bits.address,
       weights_.w1_e4m3, weights_.w1_scale_bits,
       arena_.gate_or_middle_bf16.address, arena_.error_flag_u32.address,
       stream_, token_count, kIntermediateSize, kHiddenSize}));
  if (!status.ok()) return status;
  status = run(operations_->gemm(
      {arena_.activation_e4m3.address, arena_.activation_scale_bits.address,
       weights_.w3_e4m3, weights_.w3_scale_bits, arena_.up_bf16.address,
       arena_.error_flag_u32.address, stream_, token_count,
       kIntermediateSize, kHiddenSize}));
  if (!status.ok()) return status;
  status = run(operations_->swiglu(
      {arena_.gate_or_middle_bf16.address, arena_.up_bf16.address,
       arena_.gate_or_middle_bf16.address, arena_.error_flag_u32.address,
       stream_, token_count}));
  if (!status.ok()) return status;
  status = run(operations_->quantize(
      {arena_.gate_or_middle_bf16.address, arena_.activation_e4m3.address,
       arena_.activation_scale_bits.address, arena_.error_flag_u32.address,
       stream_, token_count, kIntermediateSize}));
  if (!status.ok()) return status;
  status = run(operations_->gemm(
      {arena_.activation_e4m3.address, arena_.activation_scale_bits.address,
       weights_.w2_e4m3, weights_.w2_scale_bits,
       arena_.shared_output_bf16.address, arena_.error_flag_u32.address,
       stream_, token_count, kHiddenSize, kIntermediateSize}));
  if (!status.ok()) return status;
  status = run(operations_->finalize(
      {accumulator_f32_, arena_.shared_output_bf16.address, output_bf16_,
       arena_.error_flag_u32.address, stream_, token_count}));
  if (!status.ok()) return status;
  status = run(operations_->copy_error_d2h_async(
      host_error_, arena_.error_flag_u32.address, stream_));
  if (!status.ok()) return status;
  return run(operations_->record_event(completion_event_, stream_));
}

Result<DeepSeekExpertAsyncStatus> DeepSeekSharedExpertDriver::poll() {
  if (poisoned_ || !inflight_) {
    return Status::FailedPrecondition("DeepSeek shared expert driver is not pollable");
  }
  auto status = operations_->query_event(completion_event_);
  if (!status.ok()) {
    poisoned_ = true;
    return status.status();
  }
  if (*status == DeepSeekExpertAsyncStatus::kInProgress) return *status;
  // The pinned host error word is readable only after the event completes the
  // preceding D2H copy. Any terminal error prevents this lane from being reused.
  if (*status != DeepSeekExpertAsyncStatus::kSuccess || *host_error_ != 0) {
    poisoned_ = true;
    return DeepSeekExpertAsyncStatus::kError;
  }
  inflight_ = false;
  return DeepSeekExpertAsyncStatus::kSuccess;
}

Status DeepSeekSharedExpertStageBackend::poison(Status status) {
  state_ = State::kPoisoned;
  return status;
}

Result<std::unique_ptr<DeepSeekSharedExpertRuntimeResources>>
DeepSeekSharedExpertRuntimeResources::Create(
    std::uint32_t first_layer, std::uint32_t last_layer,
    std::uint32_t maximum_tokens, const DeepSeekResidentWeightArena& weights,
    DeepSeekSharedExpertArena scratch, std::uintptr_t input_bf16,
    std::uintptr_t routed_input_bf16,
    std::uintptr_t accumulator_f32, std::uintptr_t output_bf16,
    std::uintptr_t stream, std::uintptr_t completion_event,
    std::unique_ptr<DeepSeekSharedExpertOperations> operations, Allocator& pinned_allocator) {
  if (operations == nullptr || first_layer > last_layer || last_layer >= 43 || maximum_tokens == 0 ||
      maximum_tokens > DeepSeekSharedExpertDriver::kMaximumTokens ||
      input_bf16 == 0 || accumulator_f32 == 0 || output_bf16 == 0 ||
      stream == 0 || completion_event == 0 ||
      input_bf16 == accumulator_f32 || output_bf16 == accumulator_f32 ||
      routed_input_bf16 == 0 || routed_input_bf16 == input_bf16 ||
      routed_input_bf16 == accumulator_f32 || routed_input_bf16 == output_bf16) {
    return Status::InvalidArgument("Invalid shared expert runtime layer range or capacity");
  }
  auto host = Buffer::Allocate(pinned_allocator, sizeof(std::uint32_t), 256);
  if (!host.ok()) return host.status();
  if (host->data() == nullptr || host->size_bytes() != sizeof(std::uint32_t) ||
      host->generation() == 0 || host->device().type() != DeviceType::kCpu) {
    return Status::FailedPrecondition("Shared expert error allocation is not a valid host buffer");
  }
  auto result = std::unique_ptr<DeepSeekSharedExpertRuntimeResources>(
      new DeepSeekSharedExpertRuntimeResources(std::move(*host)));
  result->first_layer_ = first_layer;
  result->owned_operations_ = std::move(operations);
  result->operations_ = result->owned_operations_.get();
  result->input_bf16_ = input_bf16;
  result->routed_input_bf16_ = routed_input_bf16;
  result->accumulator_f32_ = accumulator_f32;
  result->stream_ = stream;
  result->drivers_.reserve(last_layer - first_layer + 1U);
  for (auto layer = first_layer; layer <= last_layer; ++layer) {
    auto bound = DeepSeekSharedExpertWeights::Resolve(layer, weights);
    if (!bound.ok()) return bound.status();
    auto driver = DeepSeekSharedExpertDriver::Create(maximum_tokens, *bound,
        scratch, routed_input_bf16, accumulator_f32, output_bf16, stream, completion_event,
        static_cast<std::uint32_t*>(result->host_error_.data()), *result->operations_);
    if (!driver.ok()) return driver.status();
    result->drivers_.push_back(std::move(*driver));
  }
  return result;
}

Result<DeepSeekSharedExpertDriver*> DeepSeekSharedExpertRuntimeResources::resolve(
    std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) {
  if (poisoned_ || plan.engine_epoch == 0 || plan.plan_sequence == 0 || layer < first_layer_ ||
      static_cast<std::size_t>(layer - first_layer_) >= drivers_.size()) {
    return Status::InvalidArgument("Shared expert plan or owned layer is invalid");
  }
  // Every layer borrows the same scratch, event and error word. In-flight or
  // poisoned siblings prohibit resolution, including after partial submission.
  for (const auto& driver : drivers_) {
    if (!driver.can_execute(plan.token_count)) {
      return Status::FailedPrecondition("Shared expert runtime is busy, failed or undersized");
    }
  }
  return &drivers_[layer - first_layer_];
}

Status DeepSeekSharedExpertRuntimeResources::prepare(std::uint32_t layer,
    const DeepSeekPipelinePlanDescriptor& plan) {
  auto driver = resolve(layer, plan);
  if (!driver.ok()) return driver.status();
  auto status = operations_->prepare_moe(input_bf16_, routed_input_bf16_,
      accumulator_f32_, plan.token_count, stream_);
  if (!status.ok()) poisoned_ = true;
  return status;
}

Status DeepSeekSharedExpertStageBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (state_ != State::kIdle || plan.engine_epoch == 0 || plan.plan_sequence == 0 ||
      plan.token_count == 0 || plan.token_count > DeepSeekSharedExpertDriver::kMaximumTokens) {
    return Status::FailedPrecondition("DeepSeek shared stage is not launchable");
  }
  const bool moe = command.kind == DeepSeekStageOperatorKind::kMoe;
  if (moe) {
    if (command.layer >= 43) return poison(Status::InvalidArgument("Invalid main MoE layer"));
    auto driver = provider_->resolve(command.layer, plan);
    if (!driver.ok()) return poison(driver.status());
    if (*driver == nullptr) return poison(Status::FailedPrecondition("Shared expert driver is missing"));
    if (!(*driver)->can_execute(plan.token_count)) {
      return poison(Status::FailedPrecondition("Shared expert driver is busy, failed or undersized"));
    }
    shared_ = *driver;
    tokens_ = plan.token_count;
    auto prepared = provider_->prepare(command.layer, plan);
    if (!prepared.ok()) return poison(prepared);
  }
  auto status = routed_->launch(command, plan);
  if (!status.ok()) return poison(status);
  state_ = moe ? State::kRouted : State::kPassthrough;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus> DeepSeekSharedExpertStageBackend::poll() {
  if (state_ == State::kIdle || state_ == State::kPoisoned) {
    return Status::FailedPrecondition("DeepSeek shared stage is not pollable");
  }
  if (state_ != State::kShared) {
    auto status = routed_->poll();
    if (!status.ok()) return poison(status.status());
    if (*status == DeepSeekStageComputeStatus::kInProgress) return *status;
    if (*status != DeepSeekStageComputeStatus::kSuccess) {
      return poison(Status::Internal("DeepSeek routed branch failed before shared expert"));
    }
    if (state_ == State::kPassthrough) {
      state_ = State::kIdle;
      return DeepSeekStageComputeStatus::kSuccess;
    }
    auto submitted = shared_->execute(tokens_);
    if (!submitted.ok()) return poison(submitted);
    state_ = State::kShared;
    return DeepSeekStageComputeStatus::kInProgress;
  }
  auto status = shared_->poll();
  if (!status.ok()) return poison(status.status());
  if (*status == DeepSeekExpertAsyncStatus::kInProgress) return DeepSeekStageComputeStatus::kInProgress;
  if (*status != DeepSeekExpertAsyncStatus::kSuccess) {
    return poison(Status::Internal("DeepSeek shared expert finalization failed"));
  }
  shared_ = nullptr;
  tokens_ = 0;
  state_ = State::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
