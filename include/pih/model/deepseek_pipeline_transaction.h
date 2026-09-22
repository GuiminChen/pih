#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_v4_config.h"

namespace pih {

enum class DeepSeekPlanPhase : std::uint8_t {
  kPrefill,
  kDecode,
  kVerify,
  kDrain,
};

struct DeepSeekPipelineRankCapacity final {
  std::uint64_t recv_bytes = 0;
  std::uint64_t output_hold_bytes = 0;
};

struct DeepSeekPipelineCapacity final {
  static constexpr std::uint32_t kBoundaryCredits = 2;
  std::uint32_t world_size = 0;
  std::uint32_t max_pipeline_tokens = 0;
  std::uint32_t expert_tokens = 0;
  std::uint32_t max_sequences = 0;
  std::uint32_t max_prefill_chunk_tokens = 0;
  std::uint32_t max_decode_sequences = 0;
  std::uint32_t max_verify_sequences = 0;
  bool dspark_enabled = false;
  std::uint64_t boundary_slot_bytes = 0;
  std::uint64_t control_payload_bytes = 0;
  std::uint64_t expert_workspace_bytes = 0;
  std::vector<DeepSeekPipelineRankCapacity> ranks;

  static Result<DeepSeekPipelineCapacity> Create(
      std::uint32_t world_size, std::uint32_t max_prefill_chunk_tokens,
      std::uint32_t max_decode_sequences,
      std::uint32_t max_verify_sequences, bool dspark_enabled);
  [[nodiscard]] const DeepSeekPipelineRankCapacity& rank(
      std::uint32_t rank) const;
};

struct DeepSeekPipelinePlanDescriptor final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t plan_sequence = 0;
  DeepSeekPlanPhase phase = DeepSeekPlanPhase::kPrefill;
  std::uint32_t token_count = 0;
  std::uint32_t sequence_count = 0;
};

class DeepSeekPipelineStagePool final {
 public:
  [[nodiscard]] std::uint32_t incoming_available() const noexcept {
    return incoming_available_;
  }
  [[nodiscard]] std::uint32_t outgoing_available() const noexcept {
    return outgoing_available_;
  }
  [[nodiscard]] std::uint32_t control_available() const noexcept {
    return control_available_;
  }
  void set_prepare_failure_for_test(bool fail) noexcept { inject_failure_ = fail; }

 private:
  friend class DeepSeekPipelineResourceSet;
  friend class DeepSeekPipelineTransaction;
  Status reserve();
  void restore() noexcept;
  bool has_incoming_ = false;
  bool has_outgoing_ = false;
  bool inject_failure_ = false;
  std::uint32_t incoming_available_ = 0;
  std::uint32_t outgoing_available_ = 0;
  std::uint32_t control_available_ = 2;
  std::uint32_t moe_lane_available_ = 1;
};

enum class DeepSeekPipelineTransactionState : std::uint8_t {
  kPrepared,
  kCommitted,
  kAborted,
  kCompleted,
};

class DeepSeekPipelineResourceSet;

class DeepSeekPipelineTransaction final {
 public:
  DeepSeekPipelineTransaction(const DeepSeekPipelineTransaction&) = delete;
  DeepSeekPipelineTransaction& operator=(const DeepSeekPipelineTransaction&) = delete;
  DeepSeekPipelineTransaction(DeepSeekPipelineTransaction&& other) noexcept;
  DeepSeekPipelineTransaction& operator=(DeepSeekPipelineTransaction&& other) noexcept;
  ~DeepSeekPipelineTransaction();

  [[nodiscard]] Status validate_commit() const;
  Status commit();
  [[nodiscard]] Status validate_abort_prepare() const;
  Status abort_prepare();
  [[nodiscard]] Status validate_complete() const;
  Status complete();
  [[nodiscard]] DeepSeekPipelineTransactionState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const DeepSeekPipelinePlanDescriptor& descriptor() const noexcept {
    return descriptor_;
  }

 private:
  friend class DeepSeekPipelineResourceSet;
  DeepSeekPipelineTransaction(DeepSeekPipelineResourceSet& owner,
                              DeepSeekPipelinePlanDescriptor descriptor) noexcept;
  void rollback_if_prepared() noexcept;
  DeepSeekPipelineResourceSet* owner_ = nullptr;
  DeepSeekPipelinePlanDescriptor descriptor_;
  DeepSeekPipelineTransactionState state_ =
      DeepSeekPipelineTransactionState::kPrepared;
};

class DeepSeekPipelineResourceSet final {
 public:
  static Result<DeepSeekPipelineResourceSet> Create(
      DeepSeekPipelineCapacity capacity);
  Result<DeepSeekPipelineTransaction> prepare(
      DeepSeekPipelinePlanDescriptor descriptor);
  [[nodiscard]] DeepSeekPipelineStagePool& rank(std::uint32_t rank);
  [[nodiscard]] const DeepSeekPipelineCapacity& capacity() const noexcept {
    return capacity_;
  }

 private:
  friend class DeepSeekPipelineTransaction;
  void restore_all() noexcept;
  DeepSeekPipelineCapacity capacity_;
  std::vector<DeepSeekPipelineStagePool> stages_;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t last_plan_sequence_ = 0;
  bool transaction_live_ = false;
};

}  // namespace pih
