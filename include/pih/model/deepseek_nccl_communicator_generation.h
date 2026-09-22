#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_nccl_edge_pair.h"

namespace pih {

enum class DeepSeekNcclGenerationState : std::uint8_t {
  kPrepared,
  kInitializing,
  kReconciled,
  kSealed,
  kFinalizing,
  kDestroyed,
  kAborted,
};

struct DeepSeekNcclGenerationIdentity final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t communicator_generation = 0;
  std::uint32_t world_size = 0;
  std::uint64_t edge_timeout_ns = 0;
  std::uint64_t set_timeout_ns = 0;
};

class DeepSeekNcclCommunicatorGeneration final {
 public:
  static Result<DeepSeekNcclCommunicatorGeneration> Create(
      DeepSeekNcclGenerationIdentity identity,
      std::vector<DeepSeekNcclEdgePair> edges);

  DeepSeekNcclCommunicatorGeneration(
      const DeepSeekNcclCommunicatorGeneration&) = delete;
  DeepSeekNcclCommunicatorGeneration& operator=(
      const DeepSeekNcclCommunicatorGeneration&) = delete;
  DeepSeekNcclCommunicatorGeneration(
      DeepSeekNcclCommunicatorGeneration&&) noexcept = default;
  DeepSeekNcclCommunicatorGeneration& operator=(
      DeepSeekNcclCommunicatorGeneration&&) noexcept = default;
  ~DeepSeekNcclCommunicatorGeneration();

  Status begin_init(std::uint64_t now_ns);
  Status advance_init(std::uint64_t now_ns);
  Status seal(std::span<const DeepSeekNcclWarmupReceipt> receipts);
  Status begin_teardown(std::uint64_t now_ns);
  Status advance_teardown(std::uint64_t now_ns);
  Status abort() noexcept;

  [[nodiscard]] DeepSeekNcclGenerationState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const DeepSeekNcclGenerationIdentity& identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] DeepSeekBoundaryTransportDriver* incoming_transport(
      std::uint32_t rank) noexcept;
  [[nodiscard]] DeepSeekBoundaryTransportDriver* outgoing_transport(
      std::uint32_t rank) noexcept;
  [[nodiscard]] DeepSeekBoundaryTransportDriver* incoming_warmup_transport(
      std::uint32_t rank) noexcept;
  [[nodiscard]] DeepSeekBoundaryTransportDriver* outgoing_warmup_transport(
      std::uint32_t rank) noexcept;

 private:
  DeepSeekNcclCommunicatorGeneration(
      DeepSeekNcclGenerationIdentity identity,
      std::vector<DeepSeekNcclEdgePair> edges) noexcept
      : identity_(identity), edges_(std::move(edges)) {}
  Status fail(Status cause) noexcept;
  Status check_time(std::uint64_t now_ns) noexcept;

  DeepSeekNcclGenerationIdentity identity_;
  std::vector<DeepSeekNcclEdgePair> edges_;
  DeepSeekNcclGenerationState state_ = DeepSeekNcclGenerationState::kPrepared;
  std::size_t active_edge_ = 0;
  std::uint64_t last_observed_ns_ = 0;
  std::uint64_t set_deadline_ns_ = 0;
  std::uint64_t edge_deadline_ns_ = 0;
};

}  // namespace pih
