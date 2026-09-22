#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_pipeline_transaction.h"

namespace pih {

struct DeepSeekRankReadyReceipt;
class DeepSeekRankMaterializationWarmSeal;

// The rank-service protocol starts only after the worker has emitted the
// materialization warm seal. It deliberately carries opaque lease identities,
// never a host pointer, CUDA pointer, file descriptor, or in-process resource
// object. The Linux adapter is responsible for authenticated framing.
inline constexpr std::string_view kDeepSeekRankServingProtocolAbi =
    "pih_deepseek_rank_serving_protocol_v1";

struct DeepSeekRankServingSessionBinding final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  Sha256Digest materialization_warm_seal_root{};

  friend bool operator==(const DeepSeekRankServingSessionBinding&,
                         const DeepSeekRankServingSessionBinding&) = default;
};

// Controller-only derivation. The receipt originates from the supervised
// worker and the seal is produced only after every worker's materialization
// completion has been accepted. Callers must not construct a production
// session by copying fields from a profile or a caller-selected process.
Result<DeepSeekRankServingSessionBinding>
compile_deepseek_rank_serving_session_binding(
    const DeepSeekRankReadyReceipt& ready,
    const DeepSeekRankMaterializationWarmSeal& warm_seal);
Result<std::vector<DeepSeekRankServingSessionBinding>>
compile_deepseek_rank_serving_session_set(
    std::span<const DeepSeekRankReadyReceipt> ready_receipts,
    const DeepSeekRankMaterializationWarmSeal& warm_seal);

enum class DeepSeekRankServingCommandKind : std::uint8_t {
  kExecute,
  kCancel,
};

struct DeepSeekRankServingCommand final {
  DeepSeekRankServingSessionBinding session;
  std::uint64_t command_sequence = 0;
  DeepSeekRankServingCommandKind kind =
      DeepSeekRankServingCommandKind::kExecute;
  std::uint64_t request_id = 0;
  std::uint64_t request_generation = 0;
  DeepSeekPipelinePlanDescriptor plan;

  // Execute commands use zero. A cancellation targets the one active execute
  // command on this rank, and therefore cannot accidentally cancel a reused
  // request identity from a previous generation.
  std::uint64_t target_execution_sequence = 0;

  // These values name controller/worker-owned staging leases. They are not
  // virtual addresses. Only rank zero may own input and only the last rank may
  // own final output; intermediate activation transport remains rank-local.
  std::uint64_t input_lease_identity = 0;
  std::uint64_t output_lease_identity = 0;
};

enum class DeepSeekRankServingCompletionOutcome : std::uint8_t {
  kCompleted,
  kCancelled,
  kFailed,
};

struct DeepSeekRankServingCompletion final {
  DeepSeekRankServingSessionBinding session;
  std::uint64_t execution_command_sequence = 0;
  std::uint64_t request_id = 0;
  std::uint64_t request_generation = 0;
  DeepSeekPipelinePlanDescriptor plan;
  DeepSeekRankServingCompletionOutcome outcome =
      DeepSeekRankServingCompletionOutcome::kCompleted;

  // Only a successful terminal-rank completion can release an output lease.
  std::uint64_t output_lease_identity = 0;
};

// A worker-local fail-stop gate for the service protocol. V1 permits one
// active execute command per rank. This deliberately conservative window makes
// cancellation and output ownership unambiguous; a future concurrent window
// must replace this ABI and preserve the same session/lease bindings. The
// admission/profile layer, rather than this transport-neutral gate, limits V1
// deployments to one through four ranks.
class DeepSeekRankServingWorkerGate final {
 public:
  static Result<DeepSeekRankServingWorkerGate> Create(
      DeepSeekRankServingSessionBinding session);

  DeepSeekRankServingWorkerGate(const DeepSeekRankServingWorkerGate&) =
      delete;
  DeepSeekRankServingWorkerGate& operator=(
      const DeepSeekRankServingWorkerGate&) = delete;
  DeepSeekRankServingWorkerGate(DeepSeekRankServingWorkerGate&&) noexcept =
      default;
  DeepSeekRankServingWorkerGate& operator=(
      DeepSeekRankServingWorkerGate&&) noexcept = default;

  Status accept(const DeepSeekRankServingCommand& command);
  Status complete(const DeepSeekRankServingCompletion& completion);

  [[nodiscard]] const DeepSeekRankServingSessionBinding& session()
      const noexcept {
    return session_;
  }
  [[nodiscard]] bool execution_active() const noexcept {
    return active_execution_.has_value();
  }
  [[nodiscard]] bool cancellation_requested() const noexcept {
    return cancellation_command_sequence_.has_value();
  }
  [[nodiscard]] bool failed() const noexcept { return failed_; }

 private:
  explicit DeepSeekRankServingWorkerGate(
      DeepSeekRankServingSessionBinding session) noexcept
      : session_(std::move(session)) {}

  static Status ValidateSession(
      const DeepSeekRankServingSessionBinding& session);
  Status ValidateEnvelope(const DeepSeekRankServingCommand& command) const;
  Status Reject(Status cause) noexcept;

  DeepSeekRankServingSessionBinding session_;
  std::uint64_t last_command_sequence_ = 0;
  std::optional<DeepSeekRankServingCommand> active_execution_;
  std::optional<std::uint64_t> cancellation_command_sequence_;
  bool failed_ = false;
};

}  // namespace pih
