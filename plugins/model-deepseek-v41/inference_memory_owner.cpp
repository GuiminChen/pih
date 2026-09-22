#include "inference_memory_owner.h"
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Status Provider(pih_status_v1 status) {
  if (!pih_status_is_valid_v1(&status)) return Status::Internal("Malformed inference memory provider status; retain ledger");
  if (!pih_status_is_ok_v1(&status)) return Status::FailedPrecondition("Inference memory provider failed; retain ledger");
  return Status::Ok();
}
Status Allocation(const pih_cuda_allocation_v1& a, unsigned kind, int device, std::uint64_t bytes) {
  if (a.struct_size != sizeof(a) || a.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      a.memory_kind != kind || a.device_ordinal != device || !a.generation || a.bytes != bytes ||
      a.alignment < 256 || (a.alignment & (a.alignment - 1)) || !a.address || a.address % a.alignment ||
      a.bytes > std::numeric_limits<std::uintptr_t>::max() - a.address)
    return Status::FailedPrecondition("Inference allocation identity or extent invalid; retain ledger");
  return Status::Ok();
}
}
InferenceMemoryOwner::InferenceMemoryOwner(const InferenceMemoryPlan& plan, const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
    std::uintptr_t context, std::uintptr_t stream, std::uintptr_t event)
    : plan_(plan), memory_(memory), async_(async), device_(device), context_(context), stream_(stream), event_(event) {}
Status InferenceMemoryOwner::Allocate() {
  if (state_ != InferenceMemoryState::kEmpty) return Status::FailedPrecondition("Inference owner is not empty");
  if (device_ < 0 || !context_ || !stream_ || !event_ || memory_.struct_size != sizeof(memory_) ||
      memory_.contract_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 || !memory_.context ||
      !memory_.allocate_device || !memory_.deallocate_device || !memory_.allocate_pinned_host || !memory_.deallocate_pinned_host ||
      async_.struct_size != sizeof(async_) || async_.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      !async_.context || !async_.activate_context || !async_.memset_async || !async_.record_event || !async_.query_event)
    return Status::InvalidArgument("Inference memory capabilities or handles invalid");
  state_ = InferenceMemoryState::kQuarantined;
  auto status = Provider(async_.activate_context(async_.context, context_)); if (!status.ok()) return status;
  std::uint32_t idle = 0;
  status = Provider(async_.query_event(async_.context, context_, event_, &idle)); if (!status.ok()) return status;
  if (idle != PIH_CUDA_EVENT_COMPLETE_V1) return Status::FailedPrecondition("Inference retirement event is not idle");
  status = Provider(memory_.allocate_device(memory_.context, device_, plan_.device_bytes(), 256, &device_allocation_));
  if (!status.ok()) return status;
  status = Allocation(device_allocation_, PIH_CUDA_MEMORY_DEVICE_V1, device_, plan_.device_bytes()); if (!status.ok()) return status;
  status = Provider(memory_.allocate_pinned_host(memory_.context, -1, plan_.host_bytes(), 256, &host_allocation_));
  if (!status.ok()) return status;
  status = Allocation(host_allocation_, PIH_CUDA_MEMORY_PINNED_HOST_V1, -1, plan_.host_bytes()); if (!status.ok()) return status;
  if (device_allocation_.address < host_allocation_.address + host_allocation_.bytes &&
      host_allocation_.address < device_allocation_.address + device_allocation_.bytes)
    return Status::FailedPrecondition("Inference device and host allocations overlap; retain ledgers");
  // Initializes the shared error flag and all persistent state once, before
  // the first sequence. Never zero the arena again between decode steps.
  status = Provider(async_.memset_async(async_.context, context_, device_allocation_.address, 0, device_allocation_.bytes, stream_));
  if (!status.ok()) return status;
  status = Provider(async_.record_event(async_.context, context_, event_, stream_)); if (!status.ok()) return status;
  state_ = InferenceMemoryState::kInitializing; return Status::Ok();
}
Status InferenceMemoryOwner::StartStep(const FlashConfig& config, BlockSequence& sequence, EngramHashState& hashes,
    std::span<const std::uint32_t> tokens, const BackboneWeightUpload& weights,
    ExpertWorkspaceOwner& workspace, std::uintptr_t communicator, std::uintptr_t completion_event,
    const SamplingParameters& parameters, const SamplingIdentity& identity, InferenceOperation::Clock::time_point deadline) {
  if (state_ != InferenceMemoryState::kReady || !completion_event || completion_event == event_)
    return Status::FailedPrecondition("Inference memory is not ready or completion event is shared");
  if (tokens.empty() || tokens.size() > 4096) return Status::InvalidArgument("Inference step token count invalid");
  const auto workspace_binding = workspace.ValidateBinding(static_cast<std::uint32_t>(tokens.size()), stream_, completion_event);
  if (!workspace_binding.ok()) return workspace_binding;
  auto views = plan_.Bind({device_allocation_.address, device_allocation_.bytes}, {host_allocation_.address, host_allocation_.bytes},
      weights, workspace, communicator, completion_event); if (!views.ok()) return views.status();
  state_ = InferenceMemoryState::kQuarantined;
  operation_.reset();
  const auto active = Provider(async_.activate_context(async_.context, context_)); if (!active.ok()) return active;
  auto started = InferenceOperation::Start(config, sequence, hashes, tokens, plan_.boundary(),
      views->boundary_device, views->boundary_host, stream_, views->backbone, parameters, identity, deadline);
  if (!started.ok()) return started.status();
  operation_ = std::move(*started); state_ = InferenceMemoryState::kRunning; return Status::Ok();
}
Result<bool> InferenceMemoryOwner::Advance() {
  try {
    auto result = AdvanceImpl();
    if (!result.ok()) { state_ = InferenceMemoryState::kQuarantined; operation_.reset(); }
    return result;
  } catch (const std::bad_alloc&) {
    state_ = InferenceMemoryState::kQuarantined;
    operation_.reset();
    return Status::ResourceExhausted("Inference memory continuation allocation failed");
  }
}
Status InferenceMemoryOwner::StartRequest(const FlashConfig& config, BlockSequence& sequence, EngramHashState& hashes,
    const InferenceRequest& request, const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace,
    std::uintptr_t communicator, std::uintptr_t completion_event, InferenceOperation::Clock::time_point deadline) {
  const auto valid = request.Validate(); if (!valid.ok()) return valid;
  if (request.finish) return Status::InvalidArgument("Terminal request cannot launch inference");
  if (sequence.next_position() > FlashConfig::kMaximumPositions - request.count ||
      sequence.next_position() + request.count != request.sampling.processed_length)
    return Status::FailedPrecondition("Worker inference request differs from local sequence position");
  return StartStep(config, sequence, hashes, {request.tokens.data(), request.count}, weights, workspace,
      communicator, completion_event, request.sampling.parameters, request.sampling.identity, deadline);
}
Result<bool> InferenceMemoryOwner::AdvanceImpl() {
  if (state_ == InferenceMemoryState::kReady) return true;
  if (state_ == InferenceMemoryState::kRunning) {
    const auto active = Provider(async_.activate_context(async_.context, context_)); if (!active.ok()) return active;
    auto ready = operation_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != InferenceState::kComplete) return false;
    state_ = InferenceMemoryState::kQuarantined;
    auto status = Provider(async_.record_event(async_.context, context_, event_, stream_)); if (!status.ok()) return status;
    state_ = InferenceMemoryState::kRetiring; return false;
  }
  if (state_ != InferenceMemoryState::kInitializing && state_ != InferenceMemoryState::kRetiring)
    return Status::FailedPrecondition("Inference owner has no pending work");
  std::uint32_t event_status = 0;
  auto status = Provider(async_.query_event(async_.context, context_, event_, &event_status)); if (!status.ok()) return status;
  if (event_status == PIH_CUDA_EVENT_PENDING_V1) return false;
  if (event_status != PIH_CUDA_EVENT_COMPLETE_V1) return Status::Internal("Invalid inference retirement event status");
  state_ = InferenceMemoryState::kReady; return true;
}
Result<EngramDeviceRegion> InferenceMemoryOwner::Logits() const {
  if (state_ != InferenceMemoryState::kReady || !operation_) return Status::FailedPrecondition("Inference output has not retired");
  return operation_->Logits();
}
Result<SamplingObservation> InferenceMemoryOwner::Candidate() const {
  if (state_ != InferenceMemoryState::kReady || !operation_) return Status::FailedPrecondition("Inference candidate has not retired");
  return operation_->Candidate();
}
Status InferenceMemoryOwner::Release() {
  if (state_ == InferenceMemoryState::kReleased) return Status::Ok();
  if (state_ != InferenceMemoryState::kReady) return Status::FailedPrecondition("Inference memory must retire before release");
  state_ = InferenceMemoryState::kQuarantined;
  operation_.reset();
  auto status = Provider(memory_.deallocate_pinned_host(memory_.context, &host_allocation_)); if (!status.ok()) return status;
  host_released_ = true;
  status = Provider(memory_.deallocate_device(memory_.context, &device_allocation_)); if (!status.ok()) return status;
  device_released_ = true; state_ = InferenceMemoryState::kReleased; return Status::Ok();
}
Result<RankStepReceipt> InferenceMemoryOwner::Receipt() const {
  if (state_ != InferenceMemoryState::kReady || !operation_)
    return Status::FailedPrecondition("Rank receipt requires completed inference retirement");
  return operation_->Receipt();
}
}  // namespace pih::deepseek_v41
