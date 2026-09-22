#pragma once

#include <optional>
#include <vector>

#include "pih/model/deepseek_nccl_edge_pair.h"
#include "pih/model/deepseek_nccl_endpoint_reconciliation_gate.h"

namespace pih {

struct DeepSeekNcclEndpointWarmupReceipt final {
  DeepSeekNcclEndpointBootstrapReceipt endpoint;
  std::uint32_t minimum_token_count = 0;
  std::uint32_t maximum_token_count = 0;
  Sha256Digest minimum_payload_digest{};
  Sha256Digest maximum_payload_digest{};
  bool minimum_complete_verified = false;
  bool maximum_complete_verified = false;
};

class DeepSeekNcclBoundaryWarmupGate final {
 public:
  static Result<DeepSeekNcclBoundaryWarmupGate> Create(
      const DeepSeekNcclEndpointManifestPlan& plan,
      std::uint32_t maximum_token_count);

  Status accept(DeepSeekNcclEndpointWarmupReceipt receipt);
  Result<std::vector<DeepSeekNcclWarmupReceipt>> finalize();
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t accepted_count() const noexcept {
    return accepted_count_;
  }

 private:
  DeepSeekNcclBoundaryWarmupGate(
      std::vector<DeepSeekNcclEdgeManifestPair> expected,
      std::uint32_t maximum_token_count) noexcept
      : expected_(std::move(expected)),
        receipts_(expected_.size() * 2),
        maximum_token_count_(maximum_token_count) {}
  Status poison(Status cause) noexcept;

  std::vector<DeepSeekNcclEdgeManifestPair> expected_;
  std::vector<std::optional<DeepSeekNcclEndpointWarmupReceipt>> receipts_;
  std::uint32_t maximum_token_count_ = 0;
  std::size_t accepted_count_ = 0;
  bool poisoned_ = false;
  bool finalized_ = false;
};

}  // namespace pih
