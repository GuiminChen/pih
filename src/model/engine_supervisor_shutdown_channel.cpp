#include "pih/model/engine_supervisor_shutdown_channel.h"

namespace pih {

Result<EngineSupervisorShutdownChannel>
EngineSupervisorShutdownChannel::Create(
    EngineSupervisorShutdownRequest request,
    EngineSupervisorShutdownChannelDriver& driver) {
  const auto encoded = encode_engine_supervisor_shutdown_request(request);
  const auto decoded = decode_engine_supervisor_shutdown_request(encoded);
  if (!decoded.ok()) return decoded.status();
  auto ack_gate = EngineSupervisorShutdownAckGate::Create(
      request.engine_generation, request.engine_epoch,
      request.request_identity);
  if (!ack_gate.ok()) return ack_gate.status();
  return EngineSupervisorShutdownChannel(request, std::move(*ack_gate), driver);
}

Result<bool> EngineSupervisorShutdownChannel::advance(std::uint64_t now_ns) {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown channel is poisoned");
  }
  if (request_acknowledged_) return true;
  if (now_ns < last_now_ns_) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine supervisor shutdown channel clock regressed");
  }
  last_now_ns_ = now_ns;
  if (now_ns >= request_.deadline_ns) {
    poisoned_ = true;
    return Status::DeadlineExceeded(
        "engine supervisor shutdown acknowledgement deadline expired");
  }
  if (!request_sent_) {
    const auto frame = encode_engine_supervisor_shutdown_request(request_);
    const auto status = driver_->send_request(frame);
    if (!status.ok()) {
      if (status.code() == StatusCode::kUnavailable) return false;
      poisoned_ = true;
      return status;
    }
    request_sent_ = true;
  }
  auto frame = driver_->poll_ack();
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) return false;
    poisoned_ = true;
    return frame.status();
  }
  if (!frame->has_value()) return false;
  auto ack = decode_engine_supervisor_shutdown_ack(**frame);
  if (!ack.ok()) {
    poisoned_ = true;
    return ack.status();
  }
  if (ack->acknowledged_ns > now_ns) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine supervisor shutdown acknowledgement is from the future");
  }
  const auto accepted = ack_gate_.accept(*ack);
  if (!accepted.ok()) {
    poisoned_ = true;
    return accepted;
  }
  request_acknowledged_ = true;
  return true;
}

}  // namespace pih
