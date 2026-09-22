#pragma once

#include <span>
#include <string_view>

#include "pih/model/deepseek_rank_worker_handshake.h"

namespace pih {

enum class DeepSeekRankWorkerWaitEvent { kChallengeReadable, kReadyWritable };

class DeepSeekRankWorkerBootstrapRuntime {
 public:
  virtual ~DeepSeekRankWorkerBootstrapRuntime() = default;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Status wait_for_control(
      std::int32_t control_fd, DeepSeekRankWorkerWaitEvent event,
      std::uint64_t startup_deadline_ns) = 0;
};

struct DeepSeekRankWorkerBootstrapResult final {
  DeepSeekRankWorkerArguments arguments;
  DeepSeekRankExecReady exec_ready;
};

class DeepSeekRankWorkerBootstrap final {
 public:
  static Result<DeepSeekRankWorkerBootstrapResult> Run(
      std::span<const std::string_view> arguments,
      DeepSeekRankWorkerHandshakeOperations& handshake_operations,
      DeepSeekRankWorkerBootstrapRuntime& runtime);
  static Result<DeepSeekRankWorkerBootstrapResult> Run(
      DeepSeekRankWorkerArguments arguments,
      DeepSeekRankWorkerHandshakeOperations& handshake_operations,
      DeepSeekRankWorkerBootstrapRuntime& runtime);
};

}  // namespace pih
