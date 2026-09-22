#pragma once

#include <cstddef>
#include <vector>

#include "pih/model/deepseek_nccl_edge_endpoint.h"
#include "pih/model/deepseek_nccl_endpoint_manifest_plan.h"

namespace pih {

struct DeepSeekNcclEndpointBootstrapReceipt final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t communicator_generation = 0;
  std::uint64_t bootstrap_lease_id = 0;
  std::uint64_t bootstrap_commitment_id = 0;
  std::uint32_t edge_id = 0;
  std::uint32_t local_global_rank = 0;
  std::uint32_t peer_global_rank = 0;
  std::uint32_t communicator_local_rank = 0;
  std::uint64_t device_identity = 0;
  std::uintptr_t context_identity = 0;
  std::uint64_t config_identity = 0;
  bool bootstrap_zeroized = false;
  bool communicator_reconciled = false;

  static Result<DeepSeekNcclEndpointBootstrapReceipt> Create(
      const DeepSeekNcclEdgeEndpoint& endpoint);
};

class DeepSeekNcclEndpointReconciliationGate final {
 public:
  static Result<DeepSeekNcclEndpointReconciliationGate> Create(
      const DeepSeekNcclEndpointManifestPlan& plan);

  Status accept(const DeepSeekNcclEndpointBootstrapReceipt& receipt);
  [[nodiscard]] bool complete() const noexcept {
    return !poisoned_ && accepted_count_ == accepted_.size();
  }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t accepted_count() const noexcept {
    return accepted_count_;
  }

 private:
  explicit DeepSeekNcclEndpointReconciliationGate(
      std::vector<DeepSeekNcclEdgeManifestPair> expected) noexcept
      : expected_(std::move(expected)), accepted_(expected_.size() * 2, false) {}
  Status poison(Status cause) noexcept;

  std::vector<DeepSeekNcclEdgeManifestPair> expected_;
  std::vector<bool> accepted_;
  std::size_t accepted_count_ = 0;
  bool poisoned_ = false;
};

}  // namespace pih
