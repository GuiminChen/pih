#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/deepseek_attention_page_pool.h"
#include "pih/model/deepseek_fixed_state_banks.h"

namespace pih {

enum class DeepSeekAttentionSequenceTransactionState : std::uint8_t {
  kIdle,
  kPreparing,
  kAwaitingCompletion,
  kReadyToResolve,
  kPoisoned,
};

class DeepSeekAttentionSequenceTransaction final {
 public:
  static Result<DeepSeekAttentionSequenceTransaction> Create(
      std::uint32_t sequence, std::uint32_t maximum_ratio4_mutations,
      std::uint32_t maximum_ratio128_mutations,
      DeepSeekFixedStateBanks& fixed_banks,
      DeepSeekRatio4PagePool& ratio4_pool,
      DeepSeekRatio128PagePool& ratio128_pool);

  Status validate_begin(std::uintptr_t stream) const;
  Status begin(std::uintptr_t stream);
  // Returns true only to the first owner in this prepare epoch; that owner
  // must clear both host and device error storage before launching work.
  Result<bool> claim_external_error_channel(
      std::uint32_t* host_error_flag,
      std::uintptr_t device_error_flag_u32);
  Result<DeepSeekExpertArenaSpan> tentative_fixed_state() const;
  Result<DeepSeekRatio4PagePair> reserve_ratio4_append(
      std::uint32_t layer_id, std::uint32_t logical_page);
  Result<DeepSeekRatio4PagePair> reserve_ratio4_tail_cow(
      const DeepSeekRatio4PagePair& committed);
  Result<DeepSeekRatio128Page> reserve_ratio128_append(
      std::uint32_t layer_id, std::uint32_t logical_page);
  Result<DeepSeekRatio128Page> reserve_ratio128_tail_cow(
      const DeepSeekRatio128Page& committed);
  Result<std::optional<DeepSeekRatio4PagePair>> find_published_ratio4(
      std::uint32_t layer_id, std::uint32_t logical_page) const;
  Result<std::vector<std::uint32_t>> ratio4_index_page_slots(
      std::uint32_t layer_id, std::uint32_t logical_page_count) const;
  Result<std::vector<std::uint32_t>> ratio4_main_page_slots(
      std::uint32_t layer_id, std::uint32_t logical_page_count) const;
  [[nodiscard]] std::uint32_t ratio4_physical_page_count() const noexcept;
  Result<std::optional<DeepSeekRatio128Page>> find_published_ratio128(
      std::uint32_t layer_id, std::uint32_t logical_page) const;
  Result<std::vector<std::uint32_t>> ratio128_main_page_slots(
      std::uint32_t layer_id, std::uint32_t logical_page_count) const;
  [[nodiscard]] std::uint32_t ratio128_physical_page_count() const noexcept;
  Status validate_ratio4_append_target(
      const DeepSeekRatio4PagePair& target) const;
  Status validate_ratio4_cow_target(
      const DeepSeekRatio4PagePair& committed,
      const DeepSeekRatio4PagePair& target) const;
  Status validate_ratio128_append_target(
      const DeepSeekRatio128Page& target) const;
  Status validate_ratio128_cow_target(
      const DeepSeekRatio128Page& committed,
      const DeepSeekRatio128Page& target) const;
  Status seal(std::uintptr_t stream);
  Result<DeepSeekExpertAsyncStatus> poll();
  Status validate_commit() const;
  Status commit();
  Status cancel_preparing();
  Status abort();

  [[nodiscard]] DeepSeekAttentionSequenceTransactionState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t prepare_epoch() const noexcept {
    return prepare_epoch_;
  }
  [[nodiscard]] std::uintptr_t stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t sequence() const noexcept { return sequence_; }

 private:
  struct Ratio4Mutation final {
    bool cow = false;
    DeepSeekRatio4PagePair committed;
    DeepSeekRatio4PagePair replacement;
  };
  struct Ratio128Mutation final {
    bool cow = false;
    DeepSeekRatio128Page committed;
    DeepSeekRatio128Page replacement;
  };
  struct ExternalErrorChannel final {
    std::uint32_t* host = nullptr;
    std::uintptr_t device = 0;
  };

  std::uint32_t sequence_ = 0;
  std::vector<Ratio4Mutation> ratio4_;
  std::vector<Ratio128Mutation> ratio128_;
  std::uint32_t ratio4_count_ = 0;
  std::uint32_t ratio128_count_ = 0;
  DeepSeekFixedStateBanks* fixed_banks_ = nullptr;
  DeepSeekRatio4PagePool* ratio4_pool_ = nullptr;
  DeepSeekRatio128PagePool* ratio128_pool_ = nullptr;
  std::vector<ExternalErrorChannel> external_error_channels_;
  std::uint64_t prepare_epoch_ = 0;
  std::uintptr_t stream_ = 0;
  DeepSeekAttentionSequenceTransactionState state_ =
      DeepSeekAttentionSequenceTransactionState::kIdle;
};

}  // namespace pih
