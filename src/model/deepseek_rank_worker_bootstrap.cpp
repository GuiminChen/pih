#include "pih/model/deepseek_rank_worker_bootstrap.h"

#include <utility>

namespace pih {

Result<DeepSeekRankWorkerBootstrapResult> DeepSeekRankWorkerBootstrap::Run(
    std::span<const std::string_view> arguments,
    DeepSeekRankWorkerHandshakeOperations& handshake_operations,
    DeepSeekRankWorkerBootstrapRuntime& runtime) {
  auto parsed = DeepSeekRankWorkerArguments::Parse(arguments);
  if (!parsed.ok()) return parsed.status();
  return Run(std::move(*parsed), handshake_operations, runtime);
}

Result<DeepSeekRankWorkerBootstrapResult> DeepSeekRankWorkerBootstrap::Run(
    DeepSeekRankWorkerArguments result,
    DeepSeekRankWorkerHandshakeOperations& handshake_operations,
    DeepSeekRankWorkerBootstrapRuntime& runtime) {
  auto handshake = DeepSeekRankWorkerHandshake::Create(
      result, handshake_operations);
  if (!handshake.ok()) return handshake.status();

  while (!handshake->ready()) {
    auto now = runtime.monotonic_now_ns();
    if (!now.ok()) return now.status();
    const auto status = handshake->poll(*now);
    if (status.ok()) continue;
    if (status.code() != StatusCode::kUnavailable) return status;
    const auto event = handshake->challenge_identity() == 0
                           ? DeepSeekRankWorkerWaitEvent::kChallengeReadable
                           : DeepSeekRankWorkerWaitEvent::kReadyWritable;
    const auto waited = runtime.wait_for_control(
        result.control_fd, event, result.manifest.startup_deadline_ns);
    if (!waited.ok()) return waited;
  }
  const auto* exec_ready = handshake->exec_ready();
  if (exec_ready == nullptr) {
    return Status::Internal(
        "DeepSeek rank worker bootstrap lost exec-ready identity");
  }
  return DeepSeekRankWorkerBootstrapResult{
      std::move(result), *exec_ready};
}

}  // namespace pih
