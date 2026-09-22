#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pih/scheduler/controller_ingress.h"
#include "pih/scheduler/controller_sequence.h"
#include "pih/scheduler/output_burst_credit_pool.h"
#include "pih/scheduler/phase_tier_scheduler.h"

namespace pih {

enum class ControllerRuntimeState : std::uint8_t {
  kReady = 1,
  kFailed = 2,
  kClosed = 3,
};

enum class ControllerRuntimeStep : std::uint8_t {
  kIdle = 1,
  kOutputBackpressured = 2,
  kAdmitted = 3,
  kCancelRequested = 4,
  kAdmissionRejected = 5,
};

struct ControllerRuntimeLimits final {
  std::uint64_t epoch;
  std::uint32_t maximum_sequences;
  ControllerIngressLimits ingress;
  PhaseTierProfile scheduler;
  PackedTokenMetadataLimits metadata;
  std::uint32_t maximum_prefill_chunk_tokens;
  bool pad_to_maximum_execution_bucket;
  std::string profile_revision;
  Sha256Digest resource_vector_hash;
  std::uint32_t eos_token_id = UINT32_MAX;
  std::uint32_t output_burst_credit_count = 1;
  std::uint32_t output_maximum_slots_per_credit = 64;
  std::uint64_t output_maximum_bytes_per_credit = 1U << 20U;
  OutputPlanKind output_plan_kind = OutputPlanKind::kQwen;
};

struct ControllerPreparedPlanView final {
  const PackedTokenPlan* plan;
  PackedTokenMetadataView metadata;
  std::span<const std::uint32_t> selected_slot_indices;
  std::span<const ControllerRequestSampling> selected_sampling;
  Sha256Digest plan_digest;
};

struct ControllerBackendSequenceResult final {
  ControllerPlanCompletion completion;
  std::span<const std::uint32_t> accepted_token_ids;
};

class ControllerAdmissionParticipant {
 public:
  virtual ~ControllerAdmissionParticipant() = default;
  virtual Status reserve(const ControllerAdmissionView& admission) = 0;
  virtual Status publish(std::uint64_t sequence_generation) = 0;
  virtual Status rollback(std::uint64_t request_generation) = 0;
};

class ControllerRuntime final {
 public:
  static constexpr std::string_view kAbi = "single_writer_controller_runtime_v1";

  static Result<ControllerRuntime> Create(ControllerRuntimeLimits limits);
  ControllerRuntime(const ControllerRuntime&) = delete;
  ControllerRuntime& operator=(const ControllerRuntime&) = delete;
  ControllerRuntime(ControllerRuntime&& other) noexcept;
  ControllerRuntime& operator=(ControllerRuntime&&) noexcept = delete;

  Result<ControllerCommand> submit_admit(
      std::uint64_t request_generation,
      std::span<const std::uint32_t> prompt_token_ids,
      std::uint32_t maximum_new_tokens,
      const ControllerRequestSampling& sampling = {});
  Result<ControllerCommand> submit_cancel(
      std::uint64_t request_generation);
  Result<ControllerRuntimeStep> step(std::int64_t now_ns = 0);
  Result<ControllerRuntimeStep> step(
      std::int64_t now_ns, ControllerAdmissionParticipant& admission);
  Result<std::optional<ControllerPreparedPlanView>> prepare_next_plan(
      std::int64_t now_ns);
  Status commit_current_plan(std::uint64_t plan_sequence,
                             Sha256Digest plan_digest);
  Status mark_current_plan_in_flight(std::uint64_t plan_sequence,
                                     Sha256Digest plan_digest);
  Status abort_current_plan(std::uint64_t plan_sequence,
                            Sha256Digest plan_digest);
  Status complete_current_plan(
      std::span<const ControllerBackendSequenceResult> results,
      std::int64_t now_ns);
  Status complete_current_sampled_tokens(
      std::span<const std::uint32_t> sampled_token_ids,
      std::int64_t now_ns);
  Status acknowledge_output_plan(std::uint64_t plan_sequence);
  Status finalize_draining(std::uint64_t request_generation);
  Result<std::optional<ControllerOutputEvent>> try_take_event();

  [[nodiscard]] ControllerRuntimeState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint32_t active_sequence_count() const noexcept;
  [[nodiscard]] std::uint32_t queued_command_count() const;
  [[nodiscard]] std::uint32_t available_output_burst_credits() const noexcept {
    return output_credits_.available();
  }
  [[nodiscard]] std::uint32_t eos_token_id() const noexcept {
    return eos_token_id_;
  }
  [[nodiscard]] bool retirable() const;

 private:
  struct Slot final {
    std::optional<ControllerSequence> sequence;
    ControllerRequestView request{};
    std::optional<std::uint32_t> pending_token;
    std::uint64_t last_service_plan_sequence = 0;
    std::uint64_t ready_enqueue_sequence = 0;
    std::int64_t prefill_eligible_since_ns = 0;
  };

  ControllerRuntime(ControllerRuntimeLimits limits, ControllerIngress ingress,
                    PhaseTierScheduler scheduler,
                    PackedTokenMetadataArena metadata,
                    OutputBurstCreditPool output_credits)
      : epoch_(limits.epoch),
        ingress_(std::move(ingress)),
        slots_(limits.maximum_sequences),
        scheduler_(std::move(scheduler)),
        metadata_(std::move(metadata)),
        output_credits_(std::move(output_credits)),
        pending_output_bursts_(limits.output_burst_credit_count),
        candidates_(limits.maximum_sequences),
        candidate_states_(limits.maximum_sequences),
        candidate_tokens_(limits.maximum_sequences),
        candidate_slot_indices_(limits.maximum_sequences),
        selected_slot_indices_(limits.scheduler.maximum_sequences_per_plan),
        selected_sampling_(limits.scheduler.maximum_sequences_per_plan),
        completion_digests_(limits.scheduler.maximum_sequences_per_plan),
        sampled_completions_(limits.scheduler.maximum_sequences_per_plan),
        sampled_token_storage_(limits.scheduler.maximum_sequences_per_plan),
        output_maximum_slots_per_credit_(
            limits.output_maximum_slots_per_credit),
        output_maximum_bytes_per_credit_(
            limits.output_maximum_bytes_per_credit),
        output_plan_kind_(limits.output_plan_kind),
        maximum_prefill_chunk_tokens_(limits.maximum_prefill_chunk_tokens),
        pad_to_maximum_execution_bucket_(
            limits.pad_to_maximum_execution_bucket),
        profile_revision_(std::move(limits.profile_revision)),
        resource_vector_hash_(limits.resource_vector_hash),
        eos_token_id_(limits.eos_token_id) {}

  Result<Sha256Digest> admitted_state_digest(
      const ControllerAdmissionView& admission) const;
  Result<Sha256Digest> completion_state_digest(
      const ControllerPlanCompletion& completion) const;
  void clear_current_plan() noexcept;
  Result<ControllerRuntimeStep> step_impl(
      std::int64_t now_ns, ControllerAdmissionParticipant* admission);

  std::uint64_t epoch_ = 0;
  ControllerIngress ingress_;
  std::vector<Slot> slots_;
  PhaseTierScheduler scheduler_;
  PackedTokenMetadataArena metadata_;
  OutputBurstCreditPool output_credits_;
  std::vector<std::optional<OutputBurstLease>> pending_output_bursts_;
  std::vector<SchedulerCandidate> candidates_;
  std::vector<ScheduledSequenceState> candidate_states_;
  std::vector<ScheduledSequenceTokens> candidate_tokens_;
  std::vector<std::uint32_t> candidate_slot_indices_;
  std::vector<std::uint32_t> selected_slot_indices_;
  std::vector<ControllerRequestSampling> selected_sampling_;
  std::vector<Sha256Digest> completion_digests_;
  std::vector<ControllerBackendSequenceResult> sampled_completions_;
  std::vector<std::uint32_t> sampled_token_storage_;
  std::uint32_t output_maximum_slots_per_credit_ = 0;
  std::uint64_t output_maximum_bytes_per_credit_ = 0;
  OutputPlanKind output_plan_kind_ = OutputPlanKind::kQwen;
  std::optional<PackedTokenPlan> current_plan_;
  std::optional<OutputBurstLease> current_output_burst_;
  Sha256Digest current_plan_digest_{};
  SchedulerTier current_tier_ = SchedulerTier::kIdle;
  std::uint32_t current_selected_count_ = 0;
  std::uint32_t maximum_prefill_chunk_tokens_ = 0;
  std::uint32_t maximum_execution_bucket_tokens_ = 0;
  bool pad_to_maximum_execution_bucket_ = false;
  std::string profile_revision_;
  Sha256Digest resource_vector_hash_{};
  std::uint32_t eos_token_id_ = UINT32_MAX;
  std::uint64_t next_ready_enqueue_sequence_ = 1;
  std::uint64_t next_plan_sequence_ = 1;
  std::int64_t last_controller_now_ns_ = 0;
  std::uint32_t consecutive_decode_rounds_ = 0;
  bool last_successful_plan_was_protected_prefill_ = false;
  std::atomic<ControllerRuntimeState> state_{ControllerRuntimeState::kReady};
  std::atomic<std::uint32_t> active_sequence_count_{0};
};

}  // namespace pih
