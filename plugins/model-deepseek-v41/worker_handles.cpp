#include "worker_handles.h"

namespace pih::deepseek_v41 {
namespace {
Status Provider(pih_status_v1 s) {
  if (!pih_status_is_valid_v1(&s)) return Status::Internal("Malformed worker resource provider response; retain ledger");
  if (!pih_status_is_ok_v1(&s)) return Status::FailedPrecondition("Worker resource provider failed; retain ledger");
  return Status::Ok();
}
}
Status WorkerHandles::Create(std::uint32_t major, std::uint32_t minor) {
  if (state_ != WorkerHandlesState::kEmpty) return Status::FailedPrecondition("Worker handles are not empty");
  if (ordinal_ < 0 || !major || major > 99 || minor > 9 ||
      device_.struct_size != sizeof(device_) || device_.contract_version != PIH_NVIDIA_CUDA_ABI_VERSION_V1 ||
      !device_.context || !device_.prepare_device || resources_.struct_size != sizeof(resources_) ||
      resources_.contract_version != PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 || !resources_.context ||
      !resources_.retain_primary_context || !resources_.bind_runtime || !resources_.create_nonblocking_stream ||
      !resources_.create_disable_timing_event || !resources_.destroy_event || !resources_.destroy_stream || !resources_.release_primary_context ||
      async_.struct_size != sizeof(async_) || async_.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      !async_.context || !async_.activate_context || !async_.record_event || !async_.query_event)
    return Status::InvalidArgument("Worker device/resource/async capabilities invalid");
  state_ = WorkerHandlesState::kQuarantined;
  auto status = Provider(device_.prepare_device(device_.context, ordinal_, major, minor)); if (!status.ok()) return status;
  status = Provider(resources_.retain_primary_context(resources_.context, ordinal_, PIH_CUDA_CONTEXT_SCHED_YIELD_V1, &ledger_.context));
  if (!status.ok()) return status;
  if (!ledger_.context) return Status::Internal("Worker context provider returned a null handle");
  status = Provider(resources_.bind_runtime(resources_.context, ordinal_, ledger_.context)); if (!status.ok()) return status;
  status = Provider(resources_.create_nonblocking_stream(resources_.context, ledger_.context, &ledger_.stream)); if (!status.ok()) return status;
  if (!ledger_.stream) return Status::Internal("Worker stream provider returned a null handle");
  for (unsigned i = 0; i < ledger_.events.size(); ++i) {
    status = Provider(resources_.create_disable_timing_event(resources_.context, ledger_.context, &ledger_.events[i]));
    if (!status.ok()) return status;
    if (!ledger_.events[i]) return Status::Internal("Worker event provider returned a null handle");
    for (unsigned j = 0; j < i; ++j) if (ledger_.events[i] == ledger_.events[j])
      return Status::FailedPrecondition("Worker provider returned duplicate event handles");
  }
  state_ = WorkerHandlesState::kReady; return Status::Ok();
}
Status WorkerHandles::BeginRetirement() {
  if (state_ != WorkerHandlesState::kReady) return Status::FailedPrecondition("Worker handles cannot begin retirement");
  state_ = WorkerHandlesState::kQuarantined;
  auto status = Provider(async_.activate_context(async_.context, ledger_.context)); if (!status.ok()) return status;
  status = Provider(async_.record_event(async_.context, ledger_.context, ledger_.events[0], ledger_.stream)); if (!status.ok()) return status;
  state_ = WorkerHandlesState::kRetiring; return Status::Ok();
}
Result<bool> WorkerHandles::PollRetirement() {
  if (state_ == WorkerHandlesState::kRetired) return true;
  if (state_ != WorkerHandlesState::kRetiring) return Status::FailedPrecondition("Worker handles have no pending retirement");
  state_ = WorkerHandlesState::kQuarantined;
  std::uint32_t event = 0;
  const auto status = Provider(async_.query_event(async_.context, ledger_.context, ledger_.events[0], &event)); if (!status.ok()) return status;
  if (event == PIH_CUDA_EVENT_PENDING_V1) { state_ = WorkerHandlesState::kRetiring; return false; }
  if (event != PIH_CUDA_EVENT_COMPLETE_V1) return Status::Internal("Invalid worker retirement event observation");
  state_ = WorkerHandlesState::kRetired; return true;
}
Status WorkerHandles::Release() {
  if (state_ == WorkerHandlesState::kReleased) return Status::Ok();
  if (state_ != WorkerHandlesState::kRetired) return Status::FailedPrecondition("Worker handles must retire before release");
  state_ = WorkerHandlesState::kQuarantined;
  auto status = Provider(async_.activate_context(async_.context, ledger_.context)); if (!status.ok()) return status;
  for (unsigned i = 4; i-- > 0;) {
    status = Provider(resources_.destroy_event(resources_.context, ledger_.events[i])); if (!status.ok()) return status;
    ledger_.events_released[i] = true;
  }
  status = Provider(resources_.destroy_stream(resources_.context, ledger_.stream)); if (!status.ok()) return status;
  ledger_.stream_released = true;
  status = Provider(resources_.release_primary_context(resources_.context, ordinal_, ledger_.context)); if (!status.ok()) return status;
  ledger_.context_released = true; state_ = WorkerHandlesState::kReleased; return Status::Ok();
}
}  // namespace pih::deepseek_v41
