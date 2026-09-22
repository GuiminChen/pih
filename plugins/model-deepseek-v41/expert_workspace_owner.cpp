#include "expert_workspace_owner.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
Status ProviderStatus(pih_status_v1 status, const char* operation) {
  if (!pih_status_is_valid_v1(&status)) return Status::Internal(std::string(operation) + ": malformed provider status");
  if (!pih_status_is_ok_v1(&status)) return Status::FailedPrecondition(std::string(operation) + ": provider failure; retain allocation ledger");
  return Status::Ok();
}
}
ExpertWorkspaceOwner::ExpertWorkspaceOwner(const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
    std::uintptr_t context, std::uintptr_t stream, std::uintptr_t event)
    : memory_(&memory), async_(&async), device_(device), context_(context), stream_(stream), event_(event) {}
Status ExpertWorkspaceOwner::Allocate(std::uint32_t tokens) {
  if (state_ != ExpertWorkspaceState::kEmpty) return Status::FailedPrecondition("Expert workspace allocation is not empty");
  if (!context_ || !stream_ || !event_ || device_ < 0 || async_->struct_size != sizeof(*async_) ||
      async_->contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 || !async_->context ||
      !async_->activate_context || !async_->record_event || !async_->query_event)
    return Status::InvalidArgument("Expert workspace retirement capability or resources invalid");
  const auto size = ExpertWorkspaceBytes(tokens); if (!size.ok()) return size.status();
  state_ = ExpertWorkspaceState::kQuarantined;
  const auto active = ProviderStatus(async_->activate_context(async_->context, context_), "Activate workspace context");
  if (!active.ok()) return active;
  std::uint32_t event_status = 0;
  const auto idle = ProviderStatus(async_->query_event(async_->context, context_, event_, &event_status), "Query workspace retirement event");
  if (!idle.ok()) return idle;
  if (event_status != PIH_CUDA_EVENT_COMPLETE_V1) return Status::FailedPrecondition("Workspace retirement event is not idle");
  const auto allocated = AllocateExpertWorkspace(*memory_, device_, tokens, allocation_); if (!allocated.ok()) return allocated;
  tokens_ = tokens; state_ = ExpertWorkspaceState::kReady;
  return Status::Ok();
}
Result<std::uint64_t> ExpertWorkspaceOwner::Reserve() {
  if (state_ != ExpertWorkspaceState::kReady) return Status::FailedPrecondition("Expert workspace cannot be reserved");
  if (next_reservation_ == std::numeric_limits<std::uint64_t>::max())
    return Status::ResourceExhausted("Expert workspace reservation identities exhausted");
  reservation_ = ++next_reservation_; state_ = ExpertWorkspaceState::kReserved;
  return reservation_;
}
Result<EngramDeviceRegion> ExpertWorkspaceOwner::BeginUse(std::uint64_t reservation) {
  if (!((state_ == ExpertWorkspaceState::kReady && !reservation) ||
        (state_ == ExpertWorkspaceState::kReserved && reservation && reservation == reservation_)))
    return Status::FailedPrecondition("Expert workspace use reservation mismatch");
  const auto valid = ValidateExpertWorkspaceAllocation(allocation_, device_, tokens_);
  if (!valid.ok()) { state_ = ExpertWorkspaceState::kQuarantined; return valid; }
  state_ = ExpertWorkspaceState::kInUse;
  return EngramDeviceRegion{allocation_.address, allocation_.bytes};
}
Status ExpertWorkspaceOwner::ValidateBinding(std::uint32_t tokens, std::uintptr_t stream,
    std::uintptr_t computation_event, std::uint64_t reservation) const {
  const bool admitted = (state_ == ExpertWorkspaceState::kReady && !reservation) ||
      (state_ == ExpertWorkspaceState::kReserved && reservation && reservation == reservation_);
  if (!admitted || tokens != tokens_ || stream != stream_ ||
      !computation_event || computation_event == event_)
    return Status::FailedPrecondition("Workspace capacity, stream, state or exclusive retirement event mismatch");
  return ValidateExpertWorkspaceAllocation(allocation_, device_, tokens);
}
Status ExpertWorkspaceOwner::RecordRetirementFence(std::uint64_t reservation) {
  if ((state_ != ExpertWorkspaceState::kInUse && state_ != ExpertWorkspaceState::kReserved) || reservation != reservation_)
    return Status::FailedPrecondition("Expert workspace has no matching use/reservation to fence");
  state_ = ExpertWorkspaceState::kQuarantined;
  const auto recorded = ProviderStatus(async_->record_event(async_->context, context_, event_, stream_), "Record workspace retirement fence");
  if (!recorded.ok()) return recorded;
  state_ = ExpertWorkspaceState::kWaitingFence; return Status::Ok();
}
Result<bool> ExpertWorkspaceOwner::PollRetirement() {
  if (state_ == ExpertWorkspaceState::kReady) return true;
  if (state_ != ExpertWorkspaceState::kWaitingFence) return Status::FailedPrecondition("Expert workspace has no retirement fence");
  state_ = ExpertWorkspaceState::kQuarantined;
  std::uint32_t event_status = 0;
  const auto queried = ProviderStatus(async_->query_event(async_->context, context_, event_, &event_status), "Query workspace retirement fence");
  if (!queried.ok()) return queried;
  if (event_status == PIH_CUDA_EVENT_PENDING_V1) { state_ = ExpertWorkspaceState::kWaitingFence; return false; }
  if (event_status != PIH_CUDA_EVENT_COMPLETE_V1) return Status::Internal("Invalid workspace retirement event state");
  reservation_ = 0; state_ = ExpertWorkspaceState::kReady; return true;
}
Status ExpertWorkspaceOwner::Release() {
  if (state_ == ExpertWorkspaceState::kReleased) return Status::Ok();
  if (state_ != ExpertWorkspaceState::kReady) return Status::FailedPrecondition("Expert workspace must retire before release");
  state_ = ExpertWorkspaceState::kQuarantined;
  const auto released = ProviderStatus(memory_->deallocate_device(memory_->context, &allocation_), "Release expert workspace");
  if (!released.ok()) return released;
  // Retain the original handle as an audit record, never use it again.
  state_ = ExpertWorkspaceState::kReleased; return Status::Ok();
}
}  // namespace pih::deepseek_v41
