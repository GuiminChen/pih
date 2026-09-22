#include "pih/model/engine_supervisor_shutdown_server.h"

namespace pih {

Result<EngineSupervisorShutdownServer>
EngineSupervisorShutdownServer::Create(
    EngineSupervisorShutdownServerDriver& driver) {
  return EngineSupervisorShutdownServer(driver);
}

Result<bool> EngineSupervisorShutdownServer::flush_ack() {
  const auto status = driver_->send_ack(*pending_ack_);
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return false;
    poisoned_ = true;
    return status;
  }
  pending_ack_.reset();
  complete_ = true;
  return true;
}

Result<bool> EngineSupervisorShutdownServer::advance(
    std::uint64_t now_ns, EngineSupervisorShutdownHandler& handler,
    EngineSupervisionCoordinator& coordinator,
    EngineShutdownController& shutdown,
    EngineGenerationDomainTermination& termination) {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown server is poisoned");
  }
  if (complete_) return true;
  if (pending_ack_.has_value()) return flush_ack();

  if (!pending_request_.has_value()) {
    auto polled = driver_->poll_request();
    if (!polled.ok()) {
      if (polled.status().code() == StatusCode::kUnavailable) return false;
      poisoned_ = true;
      return polled.status();
    }
    if (!polled->has_value()) return false;
    auto decoded = decode_engine_supervisor_shutdown_request(**polled);
    if (!decoded.ok()) {
      poisoned_ = true;
      return decoded.status();
    }
    pending_request_ = *decoded;
  }

  auto handled = handler.accept(*pending_request_, now_ns, coordinator,
                                shutdown, termination);
  if (!handled.ok()) {
    if (handled.status().code() == StatusCode::kUnavailable) return false;
    poisoned_ = true;
    return handled.status();
  }
  pending_ack_ = encode_engine_supervisor_shutdown_ack(handled->ack);
  pending_request_.reset();
  return flush_ack();
}

}  // namespace pih
