#pragma once

#include <optional>
#include <vector>

#include "pih/model/deepseek_rank_control_codec.h"
#include "pih/model/deepseek_rank_worker_arguments.h"

namespace pih {

class DeepSeekRankWorkerHandshakeOperations {
 public:
  virtual ~DeepSeekRankWorkerHandshakeOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> receive_challenge(
      std::int32_t control_fd) = 0;
  virtual Result<DeepSeekRankExecObservation> collect_observation(
      const DeepSeekRankWorkerArguments& arguments) = 0;
  virtual Status send_ready(std::int32_t control_fd,
                            std::span<const std::byte> frame) = 0;
};

class DeepSeekRankWorkerHandshake final {
 public:
  static Result<DeepSeekRankWorkerHandshake> Create(
      DeepSeekRankWorkerArguments arguments,
      DeepSeekRankWorkerHandshakeOperations& operations);
  Status poll(std::uint64_t now_ns);
  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint64_t challenge_identity() const noexcept {
    return challenge_identity_;
  }
  [[nodiscard]] const DeepSeekRankExecReady* exec_ready() const noexcept {
    return exec_ready_ ? &*exec_ready_ : nullptr;
  }

 private:
  DeepSeekRankWorkerHandshake(DeepSeekRankWorkerArguments arguments,
      DeepSeekRankWorkerHandshakeOperations& operations) noexcept
      : arguments_(std::move(arguments)), operations_(&operations) {}
  Status fail(Status cause) noexcept;
  DeepSeekRankWorkerArguments arguments_;
  DeepSeekRankWorkerHandshakeOperations* operations_ = nullptr;
  std::optional<DeepSeekRankExecReady> exec_ready_;
  std::optional<std::array<std::byte, kDeepSeekRankReadyBytes>> ready_frame_;
  std::uint64_t challenge_identity_ = 0;
  bool challenge_consumed_ = false;
  bool ready_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
