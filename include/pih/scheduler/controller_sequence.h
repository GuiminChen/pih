#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/packed_token_plan.h"

namespace pih {

enum class ControllerSequenceState : std::uint8_t {
  kQueuedPrefill = 1,
  kPrefillPrepared,
  kPrefillCommitted,
  kPrefillInFlight,
  kReadyDecode,
  kDecodePrepared,
  kDecodeCommitted,
  kDecodeInFlight,
  kVerifyPrepared,
  kVerifyCommitted,
  kVerifyInFlight,
  kCancelRequested,
  kDraining,
  kCancelled,
  kCompleted,
  kFailed,
};

struct ControllerPlanCompletion final {
  std::uint64_t plan_sequence;
  std::uint64_t sequence_generation;
  std::uint64_t committed_model_processed_length;
  std::uint64_t accepted_completion_count;
  std::uint64_t state_generation;
  PackedTokenPhase next_decode_phase;
  bool terminal;
};

class ControllerSequence final {
 public:
  static Result<ControllerSequence> Admit(std::uint64_t sequence_generation,
                                          std::uint64_t prompt_token_count,
                                          std::uint64_t initial_state_generation);

  Status prepare(std::uint64_t plan_sequence, PackedTokenPhase phase,
                 std::uint32_t real_token_count);
  Status commit(std::uint64_t plan_sequence);
  Status mark_in_flight(std::uint64_t plan_sequence);
  Status abort_prepared(std::uint64_t plan_sequence);
  Status complete(const ControllerPlanCompletion& completion);
  Status validate_completion(const ControllerPlanCompletion& completion) const;
  Status request_cancel();
  Status begin_draining();
  Status finish_cancel();
  Status finish_drain();
  Status fail();
  Result<PackedTokenPhase> ready_phase() const;
  Result<std::uint32_t> next_real_token_count(
      std::uint32_t maximum_prefill_chunk_tokens,
      std::uint32_t verify_token_count) const;
  Result<bool> next_plan_produces_logits(
      std::uint32_t maximum_prefill_chunk_tokens,
      std::uint32_t verify_token_count) const;

  [[nodiscard]] ControllerSequenceState state() const noexcept { return state_; }
  [[nodiscard]] std::uint64_t sequence_generation() const noexcept {
    return sequence_generation_;
  }
  [[nodiscard]] std::uint64_t prompt_token_count() const noexcept {
    return prompt_token_count_;
  }
  [[nodiscard]] std::uint64_t model_processed_length() const noexcept {
    return model_processed_length_;
  }
  [[nodiscard]] std::uint64_t accepted_completion_count() const noexcept {
    return accepted_completion_count_;
  }
  [[nodiscard]] std::uint64_t state_generation() const noexcept {
    return state_generation_;
  }
  [[nodiscard]] bool has_active_plan() const noexcept {
    return active_plan_sequence_ != 0;
  }

 private:
  ControllerSequence(std::uint64_t sequence_generation,
                     std::uint64_t prompt_token_count,
                     std::uint64_t state_generation)
      : sequence_generation_(sequence_generation),
        prompt_token_count_(prompt_token_count),
        state_generation_(state_generation) {}

  static bool terminal(ControllerSequenceState state) noexcept;
  bool prepared() const noexcept;
  bool committed_or_in_flight() const noexcept;
  void clear_plan() noexcept;

  std::uint64_t sequence_generation_ = 0;
  std::uint64_t prompt_token_count_ = 0;
  std::uint64_t model_processed_length_ = 0;
  std::uint64_t accepted_completion_count_ = 0;
  std::uint64_t state_generation_ = 0;
  std::uint64_t active_plan_sequence_ = 0;
  std::uint32_t active_real_token_count_ = 0;
  PackedTokenPhase active_phase_ = PackedTokenPhase::kPrefill;
  PackedTokenPhase next_decode_phase_ = PackedTokenPhase::kDecode;
  ControllerSequenceState state_ = ControllerSequenceState::kQueuedPrefill;
  ControllerSequenceState state_before_cancel_ =
      ControllerSequenceState::kQueuedPrefill;
  bool drain_is_cancel_ = false;
};

}  // namespace pih
