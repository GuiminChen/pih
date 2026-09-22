#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_event_slot.h"

namespace pih {

enum class DeepSeekNcclRole : std::uint8_t { kSend, kRecv };
enum class DeepSeekNcclAsyncStatus : std::uint8_t {
  kSuccess,
  kInProgress,
  kError,
};

struct DeepSeekNcclP2pManifest final {
  std::uint64_t operation_plan_id = 0;
  std::uint64_t engine_epoch = 0;
  std::uint64_t communicator_generation = 0;
  std::uint64_t pipeline_plan_sequence = 0;
  std::uint64_t operation_ordinal = 0;
  std::uint64_t global_issue_ordinal = 0;
  std::uint32_t directed_boundary_id = 0;
  DeepSeekNcclRole role = DeepSeekNcclRole::kSend;
  std::uint32_t local_global_rank = 0;
  std::uint32_t peer_global_rank = 0;
  std::uint32_t communicator_local_rank = 0;
  std::uint32_t communicator_peer_rank = 0;
  std::uint64_t buffer_owner_id = 0;
  std::uint64_t buffer_offset_bytes = 0;
  std::uint64_t buffer_capacity_bytes = 0;
  std::uint64_t buffer_generation = 0;
  std::uintptr_t context_identity = 0;
  std::uint32_t token_count = 0;
};

class DeepSeekNcclP2pDriver {
 public:
  virtual ~DeepSeekNcclP2pDriver() = default;
  virtual Status group_start() = 0;
  virtual Status send(std::uint64_t element_count, std::uint32_t peer) = 0;
  virtual Status recv(std::uint64_t element_count, std::uint32_t peer) = 0;
  virtual Result<DeepSeekNcclAsyncStatus> group_end() = 0;
  virtual Result<DeepSeekNcclAsyncStatus> async_status() = 0;
};

class DeepSeekNcclP2pBindingTarget {
 public:
  virtual ~DeepSeekNcclP2pBindingTarget() = default;
  virtual Status bind_p2p(DeepSeekNcclRole role, void* buffer,
                          std::uint64_t buffer_bytes,
                          std::uintptr_t stream) = 0;
};

class DeepSeekBoundaryTransportDriver : public DeepSeekNcclP2pBindingTarget,
                                        public DeepSeekNcclP2pDriver {
 public:
  ~DeepSeekBoundaryTransportDriver() override = default;
};

enum class DeepSeekNcclP2pState : std::uint8_t {
  kPlanned,
  kGroupEndPending,
  kIssuedToStream,
  kDeviceInFlight,
  kCompleteVerified,
  kPoisoned,
};

class DeepSeekNcclP2pPlan final {
 public:
  static Result<DeepSeekNcclP2pPlan> Create(DeepSeekNcclP2pManifest manifest);
  static Status ValidatePair(const DeepSeekNcclP2pManifest& send,
                             const DeepSeekNcclP2pManifest& recv);

  Status issue(DeepSeekNcclP2pDriver& driver);
  Status poll_issue(DeepSeekNcclP2pDriver& driver);
  Status record_completion(CompletionEventSlot& event,
                           CompletionEventDriver& driver,
                           DriverStreamHandle stream);
  Status poll_completion(DeepSeekNcclP2pDriver& nccl,
                         CompletionEventSlot& event,
                         CompletionEventDriver& event_driver,
                         CudaCompletionFrontier& frontier,
                         CompletionEvidenceProvider& evidence);

  [[nodiscard]] DeepSeekNcclP2pState state() const noexcept { return state_; }
  [[nodiscard]] const DeepSeekNcclP2pManifest& manifest() const noexcept {
    return manifest_;
  }
  [[nodiscard]] std::uint64_t element_count() const noexcept {
    return std::uint64_t{manifest_.token_count} * 4U * 4096U;
  }
  [[nodiscard]] std::uint64_t wire_bytes() const noexcept {
    return std::uint64_t{manifest_.token_count} * 32768U;
  }

 private:
  explicit DeepSeekNcclP2pPlan(DeepSeekNcclP2pManifest manifest)
      : manifest_(manifest) {}
  Status poison(Status status);
  DeepSeekNcclP2pManifest manifest_;
  DeepSeekNcclP2pState state_ = DeepSeekNcclP2pState::kPlanned;
};

class DeepSeekNcclOperationSequencer final {
 public:
  static Result<DeepSeekNcclOperationSequencer> Create(
      std::uint64_t first_operation_ordinal);
  Status issue(DeepSeekNcclP2pPlan& plan, DeepSeekNcclP2pDriver& driver);
  Status close(const DeepSeekNcclP2pPlan& plan);
  void poison_epoch() noexcept { poisoned_ = true; }
  [[nodiscard]] std::uint64_t next_ordinal() const noexcept {
    return next_ordinal_;
  }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  explicit DeepSeekNcclOperationSequencer(std::uint64_t first) noexcept
      : next_ordinal_(first) {}
  std::uint64_t next_ordinal_ = 0;
  std::uint64_t active_plan_id_ = 0;
  bool poisoned_ = false;
};

}  // namespace pih
