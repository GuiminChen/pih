#include "pih/model/deepseek_rank_worker_handshake.h"

namespace pih {

Result<DeepSeekRankWorkerHandshake> DeepSeekRankWorkerHandshake::Create(
    DeepSeekRankWorkerArguments arguments,
    DeepSeekRankWorkerHandshakeOperations& operations) {
  if (arguments.manifest.engine_epoch == 0 ||
      arguments.manifest.worker_generation == 0 ||
      arguments.manifest.world_size == 0 || arguments.manifest.world_size > 4 ||
      arguments.manifest.rank >= arguments.manifest.world_size ||
      arguments.manifest.physical_device_identity == 0 ||
      arguments.manifest.process_manifest_identity == 0 ||
      arguments.manifest.physical_device_uuid_commitment != Sha256Digest{} ||
      arguments.manifest.startup_device_ordinal < 0 ||
      arguments.manifest.startup_deadline_ns == 0 ||
      arguments.controller_process_identity == 0 ||
      arguments.controller_pidfd < 0 || arguments.control_fd < 0) {
    return Status::InvalidArgument(
        "DeepSeek rank worker handshake arguments are invalid");
  }
  return DeepSeekRankWorkerHandshake(std::move(arguments), operations);
}

Status DeepSeekRankWorkerHandshake::fail(Status cause) noexcept {
  poisoned_ = true;
  exec_ready_.reset();
  ready_frame_.reset();
  return cause.ok() ? Status::Internal("DeepSeek rank worker handshake poisoned")
                    : cause;
}

Status DeepSeekRankWorkerHandshake::poll(std::uint64_t now_ns) {
  if (ready_ || poisoned_)
    return Status::FailedPrecondition(
        "DeepSeek rank worker handshake is closed");
  if (now_ns >= arguments_.manifest.startup_deadline_ns)
    return fail(Status::DeadlineExceeded(
        "DeepSeek rank worker handshake deadline expired"));
  if (!challenge_consumed_) {
    auto received = operations_->receive_challenge(arguments_.control_fd);
    if (!received.ok()) {
      if (received.status().code() == StatusCode::kUnavailable)
        return received.status();
      return fail(received.status());
    }
    if (!received->has_value())
      return Status::Unavailable("DeepSeek rank challenge is pending");
    challenge_consumed_ = true;
    auto challenge = decode_deepseek_rank_challenge(**received);
    if (!challenge.ok()) return fail(challenge.status());
    if (challenge->controller_process_identity !=
        arguments_.controller_process_identity)
      return fail(Status::FailedPrecondition(
          "DeepSeek rank challenge controller identity drifted"));
    auto observation = operations_->collect_observation(arguments_);
    if (!observation.ok()) return fail(observation.status());
    auto verified = verify_deepseek_rank_exec_identity(*challenge, *observation);
    if (!verified.ok()) return fail(verified.status());
    challenge_identity_ = verified->challenge_identity;
    exec_ready_ = std::move(*verified);
    ready_frame_ = encode_deepseek_rank_ready(*exec_ready_);
  }
  const auto status = operations_->send_ready(arguments_.control_fd,
                                               *ready_frame_);
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return status;
    return fail(status);
  }
  ready_ = true;
  ready_frame_.reset();
  return Status::Ok();
}

}  // namespace pih
