#pragma once

#include <memory>

#include "pih/model/deepseek_nccl_edge_endpoint.h"

namespace pih {

struct DeepSeekNcclWarmupReceipt final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t communicator_generation = 0;
  std::uint32_t edge_id = 0;
  bool minimum_boundary_passed = false;
  bool maximum_boundary_passed = false;
};

enum class DeepSeekNcclEdgePairState : std::uint8_t {
  kPrepared,
  kInitInProgress,
  kReconciled,
  kSealed,
  kFinalizeInProgress,
  kDestroyed,
  kAborted,
};

class DeepSeekNcclEdgePair final {
 public:
  static Result<DeepSeekNcclEdgePair> Create(
      std::unique_ptr<DeepSeekNcclEdgeEndpoint> lower,
      std::unique_ptr<DeepSeekNcclEdgeEndpoint> upper);

  DeepSeekNcclEdgePair(const DeepSeekNcclEdgePair&) = delete;
  DeepSeekNcclEdgePair& operator=(const DeepSeekNcclEdgePair&) = delete;
  DeepSeekNcclEdgePair(DeepSeekNcclEdgePair&&) noexcept = default;
  DeepSeekNcclEdgePair& operator=(DeepSeekNcclEdgePair&&) noexcept = default;
  ~DeepSeekNcclEdgePair();

  Status begin_init();
  Status poll_init();
  Status seal(const DeepSeekNcclWarmupReceipt& receipt);
  Status begin_finalize();
  Status poll_finalize();
  Status destroy();
  Status abort() noexcept;

  [[nodiscard]] DeepSeekNcclEdgePairState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t edge_id() const noexcept {
    return lower_->manifest().edge_id;
  }
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return lower_->manifest().engine_epoch;
  }
  [[nodiscard]] std::uint64_t communicator_generation() const noexcept {
    return lower_->manifest().communicator_generation;
  }
  [[nodiscard]] std::uint64_t config_identity() const noexcept {
    return lower_->manifest().config_identity;
  }
  [[nodiscard]] std::uint64_t bootstrap_lease_id() const noexcept {
    return lower_->manifest().bootstrap_lease_id;
  }
  [[nodiscard]] std::uint64_t bootstrap_commitment_id() const noexcept {
    return lower_->manifest().bootstrap_commitment_id;
  }
  [[nodiscard]] DeepSeekBoundaryTransportDriver* lower_transport() noexcept {
    return lower_->transport();
  }
  [[nodiscard]] DeepSeekBoundaryTransportDriver* upper_transport() noexcept {
    return upper_->transport();
  }
  [[nodiscard]] DeepSeekBoundaryTransportDriver* lower_warmup_transport()
      noexcept { return lower_->warmup_transport(); }
  [[nodiscard]] DeepSeekBoundaryTransportDriver* upper_warmup_transport()
      noexcept { return upper_->warmup_transport(); }

 private:
  DeepSeekNcclEdgePair(std::unique_ptr<DeepSeekNcclEdgeEndpoint> lower,
                       std::unique_ptr<DeepSeekNcclEdgeEndpoint> upper) noexcept
      : lower_(std::move(lower)), upper_(std::move(upper)) {}
  Status reconcile_if_ready();
  Status fail(Status cause) noexcept;

  std::unique_ptr<DeepSeekNcclEdgeEndpoint> lower_;
  std::unique_ptr<DeepSeekNcclEdgeEndpoint> upper_;
  DeepSeekNcclEdgePairState state_ = DeepSeekNcclEdgePairState::kPrepared;
};

}  // namespace pih
