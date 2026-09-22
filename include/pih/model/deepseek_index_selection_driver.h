#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/backend/cuda/deepseek_index_score.h"
#include "pih/model/deepseek_expert_subwave_executor.h"
#include "pih/model/deepseek_online_topk.h"

namespace pih {

struct DeepSeekIndexSelectionArena final {
  std::uintptr_t score_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t page_slots_u32 = 0;
  std::uint32_t page_slot_capacity = 0;
};

struct DeepSeekIndexSelectionSubmission final {
  std::uintptr_t query_bf16 = 0;
  std::uintptr_t index_kv_bf16 = 0;
  std::uintptr_t head_weight_f32 = 0;
  DeepSeekIndexSelectionArena arena;
  std::span<const std::uint32_t> visible_slot_counts;
  std::uint32_t total_slot_count = 0;
  std::uint32_t query_count = 0;
  std::uint32_t head_count = 0;  // V1 PP rank owns all 64 index heads.
  std::uintptr_t stream = 0;
};

struct DeepSeekIndexSelectionHostStaging final {
  float* scores = nullptr;
  std::uint32_t* error_flag = nullptr;
  std::uint32_t score_capacity = 0;
  std::uint32_t* page_slots = nullptr;
  std::uint32_t page_slot_capacity = 0;
};

class DeepSeekIndexSelectionOperations {
 public:
  virtual ~DeepSeekIndexSelectionOperations() = default;
  virtual Status validate_host_staging(
      const DeepSeekIndexSelectionHostStaging& staging) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status score(DeepSeekIndexScoreLaunch launch) = 0;
  virtual Status copy_h2d_async(std::uintptr_t device, const void* host,
                                std::size_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status copy_d2h_async(void* host, std::uintptr_t device,
                                std::size_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) = 0;
};

class DeepSeekIndexSelectionDriver final {
 public:
  static Result<DeepSeekIndexSelectionDriver> Create(
      DeepSeekIndexSelectionOperations& operations,
      DeepSeekIndexSelectionHostStaging staging,
      std::uintptr_t completion_event);

  Status validate(const DeepSeekIndexSelectionSubmission& submission) const;
  Status begin(const DeepSeekIndexSelectionSubmission& submission);
  Status begin_paged(const DeepSeekIndexSelectionSubmission& submission,
                     std::span<const std::uint32_t> page_slots,
                     std::uint32_t physical_page_count);
  Status launch_next_tile();
  Result<DeepSeekExpertAsyncStatus> poll_tile();
  bool complete() const;
  Result<std::vector<std::vector<std::uint32_t>>> finish() const;

 private:
  enum class State : std::uint8_t { kIdle, kReady, kInflight, kComplete, kPoisoned };
  DeepSeekIndexSelectionOperations* operations_ = nullptr;
  DeepSeekIndexSelectionHostStaging staging_;
  std::uintptr_t completion_event_ = 0;
  DeepSeekIndexSelectionSubmission submission_;
  std::vector<std::uint32_t> visible_slot_counts_;
  std::vector<DeepSeekOnlineTopK> selectors_;
  std::uint32_t next_slot_ = 0;
  std::uint32_t inflight_slot_count_ = 0;
  std::uint32_t logical_page_count_ = 0;
  std::uint32_t physical_page_count_ = 0;
  bool paged_ = false;
  State state_ = State::kIdle;
};

}  // namespace pih
