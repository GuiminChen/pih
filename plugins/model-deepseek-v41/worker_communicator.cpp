#include "worker_communicator.h"

#include <atomic>
#include <cstring>
#include <string>

namespace pih::deepseek_v41 {
namespace {
std::atomic<const pih_transport_collective_api_v1*> bound_transport{nullptr};

Status CoreStatus(const pih_status_v1& wire) {
  if (!pih_status_is_valid_v1(&wire))
    return Status::Internal("Collective transport returned invalid status");
  const std::string message(wire.message,
      ::strnlen(wire.message, sizeof(wire.message)));
  switch (wire.code) {
    case PIH_STATUS_OK_V1: return Status::Ok();
    case PIH_STATUS_INVALID_ARGUMENT_V1: return Status::InvalidArgument(message);
    case PIH_STATUS_FAILED_PRECONDITION_V1: return Status::FailedPrecondition(message);
    case PIH_STATUS_RESOURCE_EXHAUSTED_V1: return Status::ResourceExhausted(message);
    case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable(message);
    case PIH_STATUS_DEADLINE_EXCEEDED_V1: return Status::DeadlineExceeded(message);
    case PIH_STATUS_INTERNAL_V1: return Status::Internal(message);
  }
  return Status::Internal("Collective transport status code invalid");
}
bool ValidApi(const pih_transport_collective_api_v1& api) noexcept {
  return api.struct_size == sizeof(api) &&
      api.contract_version == PIH_TRANSPORT_COLLECTIVE_ABI_V1 && api.context &&
      api.start && api.poll && api.validate_buffer && api.all_reduce_sum &&
      api.all_gather && api.poll_collective && api.begin_finalize &&
      api.release && api.abort;
}
}

Status BindCollectiveTransport(const pih_transport_collective_api_v1& api) {
  if (!ValidApi(api))
    return Status::FailedPrecondition("Collective transport capability invalid");
  const pih_transport_collective_api_v1* expected = nullptr;
  if (!bound_transport.compare_exchange_strong(expected, &api))
    return Status::FailedPrecondition("Collective transport already bound");
  return Status::Ok();
}
void UnbindCollectiveTransport() noexcept { bound_transport.store(nullptr); }
const pih_transport_collective_api_v1* CollectiveTransportApi() noexcept {
  return bound_transport.load();
}

Status WorkerCommunicator::Fail(Status status) {
  state_ = WorkerCommunicatorState::kQuarantined;
  return status;
}
Status WorkerCommunicator::Start(std::span<const std::byte> id,
    std::uint32_t world, std::uint32_t rank, std::int32_t device,
    Clock::time_point deadline) {
  if (state_ != WorkerCommunicatorState::kEmpty)
    return Status::FailedPrecondition("Communicator initialization is single-use");
  if (id.size() != PIH_TRANSPORT_NCCL_ID_BYTES_V1 ||
      (world != 2 && world != 4 && world != 8) || rank >= world ||
      device < 0 || deadline <= Clock::now())
    return Status::InvalidArgument("Invalid worker communicator bootstrap");
  const auto* api = CollectiveTransportApi();
  if (!api) return Status::FailedPrecondition("Collective transport not bound");
  world_ = world;
  rank_ = rank;
  device_ = device;
  deadline_ = deadline;
  const auto wire = api->start(api->context,
      reinterpret_cast<const uint8_t*>(id.data()), world, rank, device,
      &handle_);
  auto status = CoreStatus(wire);
  if (!status.ok()) {
    state_ = WorkerCommunicatorState::kQuarantined;
    if (handle_) {
      abort_attempted_ = true;
      const auto aborted = CoreStatus(api->abort(api->context, &handle_));
      if (!aborted.ok()) return aborted;
      if (handle_) return Status::Internal("Collective abort did not consume its handle");
    }
    return status;
  }
  if (!handle_) return Fail(Status::Internal("Transport returned no communicator"));
  state_ = WorkerCommunicatorState::kInitializing;
  return Status::Ok();
}
Result<bool> WorkerCommunicator::Poll() {
  if (state_ != WorkerCommunicatorState::kInitializing &&
      state_ != WorkerCommunicatorState::kFinalizing)
    return Status::FailedPrecondition("No communicator transition to poll");
  if (Clock::now() >= deadline_)
    return Fail(Status::DeadlineExceeded("Collective transport lifecycle deadline expired"));
  const auto* api = CollectiveTransportApi();
  if (!api || !handle_)
    return Fail(Status::FailedPrecondition("Collective transport unavailable"));
  uint32_t transition = PIH_TRANSPORT_PENDING_V1;
  const auto status = CoreStatus(api->poll(api->context, handle_, &transition));
  if (!status.ok()) return Fail(status);
  if (transition != PIH_TRANSPORT_PENDING_V1 && transition != PIH_TRANSPORT_READY_V1)
    return Fail(Status::Internal("Collective transport transition invalid"));
  if (transition == PIH_TRANSPORT_PENDING_V1) return false;
  if (Clock::now() >= deadline_)
    return Fail(Status::DeadlineExceeded("Collective transport lifecycle deadline expired"));
  state_ = state_ == WorkerCommunicatorState::kInitializing
      ? WorkerCommunicatorState::kReady : WorkerCommunicatorState::kFinalized;
  return true;
}
Result<std::uintptr_t> WorkerCommunicator::Borrow() const {
  if (state_ != WorkerCommunicatorState::kReady || !handle_)
    return Status::FailedPrecondition("Communicator is not ready");
  return reinterpret_cast<std::uintptr_t>(handle_);
}
Status WorkerCommunicator::BeginFinalize(Clock::time_point deadline) {
  if (state_ != WorkerCommunicatorState::kReady || deadline <= Clock::now())
    return Status::FailedPrecondition("Communicator is not ready for finalization");
  const auto* api = CollectiveTransportApi();
  if (!api || !handle_) return Fail(Status::FailedPrecondition("Collective transport unavailable"));
  state_ = WorkerCommunicatorState::kQuarantined;
  const auto status = CoreStatus(api->begin_finalize(api->context, handle_));
  if (!status.ok()) return status;
  deadline_ = deadline;
  state_ = WorkerCommunicatorState::kFinalizing;
  return Status::Ok();
}
Status WorkerCommunicator::Release() {
  if (state_ != WorkerCommunicatorState::kFinalized)
    return Status::FailedPrecondition("Communicator must finish finalizing before release");
  const auto* api = CollectiveTransportApi();
  if (!api || !handle_) return Fail(Status::FailedPrecondition("Collective transport unavailable"));
  destroy_attempted_ = true;
  state_ = WorkerCommunicatorState::kQuarantined;
  const auto status = CoreStatus(api->release(api->context, &handle_));
  if (!status.ok()) return status;
  if (handle_) return Status::Internal("Collective release did not consume its handle");
  state_ = WorkerCommunicatorState::kReleased;
  return Status::Ok();
}
Status WorkerCommunicator::Abort() {
  if (!handle_ || abort_attempted_ || destroy_attempted_ ||
      state_ == WorkerCommunicatorState::kReleased ||
      state_ == WorkerCommunicatorState::kAborted)
    return Status::FailedPrecondition("Communicator abort has no retained handle");
  const auto* api = CollectiveTransportApi();
  if (!api) return Fail(Status::FailedPrecondition("Collective transport unavailable"));
  abort_attempted_ = true;
  state_ = WorkerCommunicatorState::kQuarantined;
  const auto status = CoreStatus(api->abort(api->context, &handle_));
  if (!status.ok()) return status;
  if (handle_) return Status::Internal("Collective abort did not consume its handle");
  state_ = WorkerCommunicatorState::kAborted;
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
