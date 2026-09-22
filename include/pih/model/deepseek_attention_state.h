#pragma once

#include <cstdint>
#include <unordered_map>

#include "pih/model/deepseek_v4_config.h"

namespace pih {

class DeepSeekAttentionStateGeometry final {
 public:
  static Result<std::uint64_t> MainLogicalBytes(std::uint32_t context);
  static Result<std::uint64_t> StageLogicalBytes(const DeepSeekStagePlan& stage,
                                                 std::uint32_t context);
};

enum class DeepSeekAttentionReservationState : std::uint8_t {
  kReserved,
  kPublished,
  kRolledBack,
  kReleased,
  kFailed,
};

class DeepSeekAttentionStatePool;

class DeepSeekAttentionStateReservation final {
 public:
  DeepSeekAttentionStateReservation(const DeepSeekAttentionStateReservation&) = delete;
  DeepSeekAttentionStateReservation& operator=(const DeepSeekAttentionStateReservation&) = delete;
  DeepSeekAttentionStateReservation(DeepSeekAttentionStateReservation&& other) noexcept;
  DeepSeekAttentionStateReservation& operator=(DeepSeekAttentionStateReservation&& other) noexcept;
  ~DeepSeekAttentionStateReservation();

  Status publish();
  Status rollback();
  Status validate_release() const;
  Status release();
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::uint64_t logical_bytes() const noexcept { return bytes_; }
  [[nodiscard]] DeepSeekAttentionReservationState state() const noexcept { return state_; }

 private:
  friend class DeepSeekAttentionStatePool;
  DeepSeekAttentionStateReservation(DeepSeekAttentionStatePool& pool,
                                    std::uint32_t sequence,
                                    std::uint64_t generation,
                                    std::uint64_t bytes) noexcept;
  void abandon() noexcept;

  DeepSeekAttentionStatePool* pool_ = nullptr;
  std::uint32_t sequence_ = 0;
  std::uint64_t generation_ = 0;
  std::uint64_t bytes_ = 0;
  DeepSeekAttentionReservationState state_ =
      DeepSeekAttentionReservationState::kReserved;
};

class DeepSeekAttentionStatePool final {
 public:
  explicit DeepSeekAttentionStatePool(std::uint64_t capacity_bytes)
      : capacity_bytes_(capacity_bytes) {}
  Result<DeepSeekAttentionStateReservation> reserve(
      std::uint32_t sequence, const DeepSeekStagePlan& stage,
      std::uint32_t context);
  [[nodiscard]] std::uint64_t available_bytes() const noexcept {
    return capacity_bytes_ - reserved_bytes_ - owned_bytes_;
  }
  [[nodiscard]] std::uint64_t reserved_bytes() const noexcept { return reserved_bytes_; }
  [[nodiscard]] std::uint64_t owned_bytes() const noexcept { return owned_bytes_; }

 private:
  friend class DeepSeekAttentionStateReservation;
  std::uint64_t capacity_bytes_ = 0;
  std::uint64_t reserved_bytes_ = 0;
  std::uint64_t owned_bytes_ = 0;
  std::uint64_t next_generation_ = 1;
  std::unordered_map<std::uint32_t, std::uint64_t> live_sequences_;
};

}  // namespace pih
