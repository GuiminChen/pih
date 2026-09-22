#pragma once

#include <array>
#include <optional>

#include "pih/model/deepseek_nccl_boundary_warmup_gate.h"

namespace pih {

class DeepSeekNcclWarmupPayloadOperations {
 public:
  virtual ~DeepSeekNcclWarmupPayloadOperations() = default;
  virtual Status prepare(DeepSeekNcclRole role, void* device_buffer,
                         std::uint64_t bytes, DriverStreamHandle stream,
                         std::uint64_t pattern_identity) = 0;
  virtual Result<Sha256Digest> digest(const void* device_buffer,
                                      std::uint64_t bytes) = 0;
};

struct DeepSeekNcclEndpointWarmupRunnerConfig final {
  DeepSeekNcclCommunicatorManifest endpoint;
  std::uint32_t maximum_token_count = 0;
  std::uint64_t first_operation_ordinal = 0;
  std::uint64_t buffer_owner_id = 0;
  std::uint64_t buffer_generation = 0;
  void* device_buffer = nullptr;
  std::uint64_t buffer_capacity_bytes = 0;
  DriverStreamHandle stream = 0;
  std::array<DriverEventHandle, 2> completion_events{};
  std::uint64_t submit_ns = 0;
  std::uint64_t deadline_ns = 0;
};

class DeepSeekNcclEndpointWarmupRunner final {
 public:
  static Result<DeepSeekNcclEndpointWarmupRunner> Create(
      DeepSeekNcclEndpointWarmupRunnerConfig config,
      DeepSeekBoundaryTransportDriver& transport,
      DeepSeekNcclWarmupPayloadOperations& payload);

  Status begin();
  Result<std::optional<DeepSeekNcclEndpointWarmupReceipt>> poll(
      CompletionEventDriver& event_driver,
      CompletionEvidenceProvider& evidence, std::uint64_t now_ns);
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  DeepSeekNcclEndpointWarmupRunner(
      DeepSeekNcclEndpointWarmupRunnerConfig config,
      DeepSeekBoundaryTransportDriver& transport,
      DeepSeekNcclWarmupPayloadOperations& payload,
      DeepSeekNcclOperationSequencer sequencer) noexcept;
  Status start_phase(std::uint32_t phase);
  Status fail(Status status) noexcept;

  DeepSeekNcclEndpointWarmupRunnerConfig config_;
  DeepSeekBoundaryTransportDriver* transport_ = nullptr;
  DeepSeekNcclWarmupPayloadOperations* payload_ = nullptr;
  DeepSeekNcclOperationSequencer sequencer_;
  std::optional<DeepSeekNcclP2pPlan> plan_;
  std::optional<CompletionEventSlot> event_;
  std::optional<CudaCompletionFrontier> frontier_;
  std::array<Sha256Digest, 2> digests_{};
  std::uint32_t phase_ = 0;
  bool begun_ = false;
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
