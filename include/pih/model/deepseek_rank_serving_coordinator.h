#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pih/model/deepseek_rank_control_codec.h"

namespace pih {

// The Linux implementation binds this transport to credentialed seqpacket
// sockets. The coordinator remains platform-neutral so the ordering and
// fail-stop behavior can be tested without a GPU or Linux process domain.
class DeepSeekRankServingChannel {
 public:
  virtual ~DeepSeekRankServingChannel() = default;
  // Unavailable means no bytes were accepted and the exact frame can retry.
  virtual Status send_command(std::uint32_t rank,
                              std::span<const std::byte> frame) = 0;
  virtual Result<std::optional<std::vector<std::byte>>> poll_completion(
      std::uint32_t rank) = 0;
  virtual Status abort_generation(std::uint64_t engine_epoch,
                                  std::uint64_t worker_generation,
                                  const Status& cause) = 0;
};

// V1 admits one pipeline plan at a time across the complete rank domain. It
// creates rank-local command frames from opaque ingress/egress lease ids and
// retains exact frames across backpressure. A higher-throughput window needs a
// new protocol revision because cancellation and final-output order would no
// longer be scalar state.
class DeepSeekRankServingCoordinator final {
 public:
  static Result<DeepSeekRankServingCoordinator> Create(
      std::span<const DeepSeekRankServingSessionBinding> sessions,
      DeepSeekRankServingChannel& channel);

  DeepSeekRankServingCoordinator(const DeepSeekRankServingCoordinator&) =
      delete;
  DeepSeekRankServingCoordinator& operator=(
      const DeepSeekRankServingCoordinator&) = delete;
  DeepSeekRankServingCoordinator(DeepSeekRankServingCoordinator&&) noexcept =
      default;
  DeepSeekRankServingCoordinator& operator=(
      DeepSeekRankServingCoordinator&&) noexcept = default;

  Status begin(std::uint64_t request_id, std::uint64_t request_generation,
               DeepSeekPipelinePlanDescriptor plan,
               std::uint64_t input_lease_identity,
               std::uint64_t output_lease_identity);
  Status cancel(std::uint64_t request_id, std::uint64_t request_generation);
  Status advance();

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool cancellation_requested() const noexcept {
    return cancellation_requested_;
  }

 private:
  DeepSeekRankServingCoordinator(
      std::vector<DeepSeekRankServingSessionBinding> sessions,
      std::vector<DeepSeekRankServingWorkerGate> gates,
      DeepSeekRankServingChannel& channel) noexcept;

  Status fail(Status cause) noexcept;
  Status send_pending(std::vector<DeepSeekRankServingCommand>& commands,
                      std::vector<bool>& sent);
  Status poll_completions();

  std::vector<DeepSeekRankServingSessionBinding> sessions_;
  std::vector<DeepSeekRankServingWorkerGate> gates_;
  DeepSeekRankServingChannel* channel_ = nullptr;
  std::vector<std::uint64_t> next_command_sequences_;
  std::vector<DeepSeekRankServingCommand> execute_commands_;
  std::vector<DeepSeekRankServingCommand> cancel_commands_;
  std::vector<bool> execute_sent_;
  std::vector<bool> cancel_sent_;
  std::vector<bool> completion_received_;
  bool active_ = false;
  bool complete_ = false;
  bool cancellation_requested_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
