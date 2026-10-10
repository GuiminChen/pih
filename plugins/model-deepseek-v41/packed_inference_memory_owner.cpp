#include "packed_inference_memory_owner.h"
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
PackedInferenceMemoryOwner::PackedInferenceMemoryOwner(const PackedInferenceMemoryPlan& plan, const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
    std::uintptr_t context, std::uintptr_t stream, std::uintptr_t event)
    : plan_(plan), memory_(memory), async_(async), device_(device), context_(context), stream_(stream), event_(event) {}
Status PackedInferenceMemoryOwner::Allocate() {
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
Status PackedInferenceMemoryOwner::StartRequest(const FlashConfig& config, const PackedRequest& request,
    std::span<BlockSequence* const> sequences, std::span<EngramHashState* const> hashes,
    const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace,
    std::uintptr_t communicator, std::uintptr_t completion_event, InferenceOperation::Clock::time_point deadline) {
  if (state_ != InferenceMemoryState::kReady || !completion_event || completion_event == event_ ||
      request.spans().size() > plan_.maximum_sequences() || request.tokens().size() > plan_.maximum_tokens())
    return Status::FailedPrecondition("Packed inference memory not ready or batch exceeds reservation");
  const auto binding = workspace.ValidateBinding(request.tokens().size(), stream_, completion_event); if (!binding.ok()) return binding;
  auto views = plan_.Bind({device_allocation_.address, device_allocation_.bytes}, {host_allocation_.address, host_allocation_.bytes},
      weights, workspace, communicator, completion_event); if (!views.ok()) return views.status();
  std::vector<BackboneResources> members; members.reserve(request.spans().size());
  for (const auto& span : request.spans()) {
    if (span.finish || span.cache_slot >= views->slots.size()) return Status::InvalidArgument("Packed inference request contains terminal/invalid slot");
    members.push_back(views->slots[span.cache_slot].backbone);
  }
  PackedBackboneResources resources{members, plan_.ffn(), views->ffn_arenas,
      {views->attention_residual.address, request.tokens().size() * 4ULL * 5120 * 2},
      {views->attention_pre.address, request.tokens().size() * 4ULL * 4}};
  state_ = InferenceMemoryState::kQuarantined; operation_.reset();
  const auto active = Provider(async_.activate_context(async_.context, context_)); if (!active.ok()) return active;
  auto started = PackedInferenceOperation::Start(config, request, sequences, hashes, plan_.boundary(),
      views->boundary_device, views->boundary_host, stream_, resources, deadline);
  if (!started.ok()) return started.status();
  operation_ = std::move(*started); state_ = InferenceMemoryState::kRunning; return Status::Ok();
}
Status PackedInferenceMemoryOwner::ResetSlot(std::uint32_t slot, const BackboneWeightUpload& weights,
    ExpertWorkspaceOwner& workspace, std::uintptr_t communicator, std::uintptr_t completion_event) {
  if (state_ != InferenceMemoryState::kReady || slot >= plan_.maximum_sequences())
    return Status::FailedPrecondition("Packed cache reset requires retired plan and reserved slot");
  operation_.reset();
  auto views = plan_.Bind({device_allocation_.address, device_allocation_.bytes}, {host_allocation_.address, host_allocation_.bytes},
      weights, workspace, communicator, completion_event); if (!views.ok()) return views.status();
  auto active = Provider(async_.activate_context(async_.context, context_)); if (!active.ok()) { Quarantine(); return active; }
  const auto cache = views->slots[slot].backbone.cache_arena;
  auto reset = Provider(async_.memset_async(async_.context, context_, cache.address, 0, cache.bytes, stream_));
  if (!reset.ok()) Quarantine(); return reset;
}
Result<bool> PackedInferenceMemoryOwner::Advance() {
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
Result<bool> PackedInferenceMemoryOwner::AdvanceImpl() {
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
Status PackedInferenceMemoryOwner::Release() {
  if (state_ == InferenceMemoryState::kReleased) return Status::Ok();
  if (state_ != InferenceMemoryState::kReady) return Status::FailedPrecondition("Inference memory must retire before release");
  state_ = InferenceMemoryState::kQuarantined;
  operation_.reset();
  auto status = Provider(memory_.deallocate_pinned_host(memory_.context, &host_allocation_)); if (!status.ok()) return status;
  host_released_ = true;
  status = Provider(memory_.deallocate_device(memory_.context, &device_allocation_)); if (!status.ok()) return status;
  device_released_ = true; state_ = InferenceMemoryState::kReleased; return Status::Ok();
}
Result<RankStepReceipt> PackedInferenceMemoryOwner::Receipt(std::uint32_t member) const {
  if (state_ != InferenceMemoryState::kReady || !operation_)
    return Status::FailedPrecondition("Rank receipt requires completed inference retirement");
  return operation_->Receipt(member);
}
}  // namespace pih::deepseek_v41
