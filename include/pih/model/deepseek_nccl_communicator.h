#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "pih/model/deepseek_gpu_architecture.h"
#include "pih/model/deepseek_nccl_p2p_plan.h"

namespace pih {

struct DeepSeekNcclReleaseConfig final {
  std::uint32_t nccl_release = 0;
  std::uint32_t config_struct_bytes = 0;
  std::int32_t blocking = 0;
  std::int32_t min_ctas = 1;
  std::int32_t max_ctas = 32;
  std::int32_t cga_cluster_size = 0;
  std::int32_t split_share = 0;
  std::int32_t shrink_share = 0;
  std::int32_t launch_order_implicit = 0;
  std::int32_t graph_usage_mode = 0;
  std::int32_t graph_stream_ordering = 0;
  std::int32_t collnet_enable = 0;
  std::int32_t cta_policy = 0;
  std::int32_t nvls_ctas = std::numeric_limits<std::int32_t>::min();
  std::int32_t channels_per_net_peer = std::numeric_limits<std::int32_t>::min();
  std::int32_t nvlink_centric_sched = 0;
  std::int32_t num_rma_contexts = 1;
  std::int32_t num_rma_signals = 1;
  std::int32_t rma_eager_init = 0;
  std::int32_t host_cft_disabled = 1;
  std::int32_t max_p2p_peers = 1;
  std::int32_t traffic_class = std::numeric_limits<std::int32_t>::min();
  bool communicator_name_present = false;
  std::string net_name = "Socket";

  static Result<DeepSeekNcclReleaseConfig> Create(
      DeepSeekGpuArchitecture architecture, std::uint32_t nccl_release,
      std::uint32_t config_struct_bytes);
};

struct DeepSeekNcclCommunicatorManifest final {
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
};

class DeepSeekNcclCommunicatorDriver {
 public:
  virtual ~DeepSeekNcclCommunicatorDriver() = default;
  virtual Result<DeepSeekNcclAsyncStatus> init_rank_config(
      std::uint64_t bootstrap_lease_id, std::uint32_t nranks,
      std::uint32_t communicator_rank) = 0;
  virtual Result<DeepSeekNcclAsyncStatus> async_status() = 0;
  virtual Result<std::uint32_t> communicator_count() = 0;
  virtual Result<std::uint32_t> communicator_user_rank() = 0;
  virtual Result<std::uint64_t> device_identity() = 0;
  virtual Result<std::uintptr_t> context_identity() = 0;
  virtual Result<DeepSeekNcclAsyncStatus> finalize() = 0;
  virtual Status destroy() = 0;
  virtual Status abort() = 0;
};

enum class DeepSeekNcclCommunicatorState : std::uint8_t {
  kAbsent,
  kBothEndpointsValidated,
  kInitInProgress,
  kCommunicatorReady,
  kReconciled,
  kWarmed,
  kSealed,
  kFinalizeInProgress,
  kFinalizedSuccess,
  kDestroyed,
  kFailed,
  kAborted,
};

class DeepSeekNcclCommunicator final {
 public:
  static Result<DeepSeekNcclCommunicator> Create(
      DeepSeekNcclCommunicatorManifest manifest);
  Status accept_bootstrap(bool both_endpoints_validated);
  Status begin_init(DeepSeekNcclCommunicatorDriver& driver);
  Status poll_init(DeepSeekNcclCommunicatorDriver& driver);
  Status reconcile(DeepSeekNcclCommunicatorDriver& driver);
  Status mark_warmed(bool min_boundary_passed, bool max_boundary_passed);
  Status seal();
  Status begin_finalize(DeepSeekNcclCommunicatorDriver& driver);
  Status poll_finalize(DeepSeekNcclCommunicatorDriver& driver);
  Status destroy(DeepSeekNcclCommunicatorDriver& driver);
  Status abort(DeepSeekNcclCommunicatorDriver& driver);
  [[nodiscard]] DeepSeekNcclCommunicatorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const DeepSeekNcclCommunicatorManifest& manifest() const noexcept {
    return manifest_;
  }

 private:
  explicit DeepSeekNcclCommunicator(DeepSeekNcclCommunicatorManifest manifest)
      : manifest_(manifest) {}
  Status fail(Status status);
  DeepSeekNcclCommunicatorManifest manifest_;
  DeepSeekNcclCommunicatorState state_ = DeepSeekNcclCommunicatorState::kAbsent;
  bool abort_called_ = false;
  bool init_attempted_ = false;
};

}  // namespace pih
