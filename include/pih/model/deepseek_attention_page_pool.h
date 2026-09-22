#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/core/result.h"
#include "pih/model/deepseek_attention_pool_geometry.h"

namespace pih {

struct DeepSeekBlockHandle final {
  std::uint64_t value = 0;
  friend bool operator==(const DeepSeekBlockHandle&,
                         const DeepSeekBlockHandle&) = default;

  static Result<DeepSeekBlockHandle> Create(DeepSeekStatePoolKind kind,
                                             std::uint32_t slot,
                                             std::uint32_t generation);
  [[nodiscard]] DeepSeekStatePoolKind pool_kind() const noexcept;
  [[nodiscard]] std::uint32_t slot() const noexcept;
  [[nodiscard]] std::uint32_t generation() const noexcept;
};

static_assert(sizeof(DeepSeekBlockHandle) == 8);

struct DeepSeekRatio4PagePair final {
  DeepSeekBlockHandle main;
  DeepSeekBlockHandle index;
  std::uint32_t sequence = 0;
  std::uint32_t layer_id = 0;
  std::uint32_t logical_page = 0;
  std::uint32_t generation = 0;
};

enum class DeepSeekAttentionPageState : std::uint8_t {
  kFree = 1,
  kReserved = 2,
  kPublished = 3,
};

class DeepSeekRatio4PagePool final {
 public:
  static Result<DeepSeekRatio4PagePool> Create(std::uint32_t page_pairs);

  Result<DeepSeekRatio4PagePair> reserve(std::uint32_t sequence,
                                         std::uint32_t layer_id,
                                         std::uint32_t logical_page);
  Result<DeepSeekRatio4PagePair> reserve_tail_cow(
      const DeepSeekRatio4PagePair& committed);
  Status publish(const DeepSeekRatio4PagePair& pair);
  Status validate_publish(const DeepSeekRatio4PagePair& pair) const;
  Result<std::optional<DeepSeekRatio4PagePair>> find_published(
      std::uint32_t sequence, std::uint32_t layer_id,
      std::uint32_t logical_page) const;
  Status publish_tail_cow(const DeepSeekRatio4PagePair& committed,
                          const DeepSeekRatio4PagePair& replacement);
  Status validate_publish_tail_cow(
      const DeepSeekRatio4PagePair& committed,
      const DeepSeekRatio4PagePair& replacement) const;
  Status rollback(const DeepSeekRatio4PagePair& pair);
  Status release(const DeepSeekRatio4PagePair& pair);
  Status validate_release_published_sequence(std::uint32_t sequence) const;
  Status release_published_sequence(std::uint32_t sequence);

  [[nodiscard]] std::uint32_t free_pairs() const noexcept {
    return free_pairs_;
  }
  [[nodiscard]] std::uint32_t reserved_pairs() const noexcept {
    return reserved_pairs_;
  }
  [[nodiscard]] std::uint32_t published_pairs() const noexcept {
    return published_pairs_;
  }
  [[nodiscard]] std::uint32_t page_pair_capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }

 private:
  struct Slot final {
    std::uint32_t generation = 0;
    std::uint32_t sequence = 0;
    std::uint32_t layer_id = 0;
    std::uint32_t logical_page = 0;
    DeepSeekAttentionPageState main = DeepSeekAttentionPageState::kFree;
    DeepSeekAttentionPageState index = DeepSeekAttentionPageState::kFree;
  };

  explicit DeepSeekRatio4PagePool(std::vector<Slot> slots)
      : slots_(std::move(slots)),
        free_pairs_(static_cast<std::uint32_t>(slots_.size())) {}
  Result<std::uint32_t> validate_pair(
      const DeepSeekRatio4PagePair& pair,
      DeepSeekAttentionPageState expected) const;
  Result<DeepSeekRatio4PagePair> reserve_free(
      std::uint32_t sequence, std::uint32_t layer_id,
      std::uint32_t logical_page);

  std::vector<Slot> slots_;
  std::uint32_t free_pairs_ = 0;
  std::uint32_t reserved_pairs_ = 0;
  std::uint32_t published_pairs_ = 0;
  bool generation_exhausted_ = false;
};

struct DeepSeekRatio128Page final {
  DeepSeekBlockHandle handle;
  std::uint32_t sequence = 0;
  std::uint32_t layer_id = 0;
  std::uint32_t logical_page = 0;
};

class DeepSeekRatio128PagePool final {
 public:
  static Result<DeepSeekRatio128PagePool> Create(std::uint32_t pages);
  Result<DeepSeekRatio128Page> reserve(std::uint32_t sequence,
                                       std::uint32_t layer_id,
                                       std::uint32_t logical_page);
  Result<DeepSeekRatio128Page> reserve_tail_cow(
      const DeepSeekRatio128Page& committed);
  Status publish(const DeepSeekRatio128Page& page);
  Status validate_publish(const DeepSeekRatio128Page& page) const;
  Result<std::optional<DeepSeekRatio128Page>> find_published(
      std::uint32_t sequence, std::uint32_t layer_id,
      std::uint32_t logical_page) const;
  Status publish_tail_cow(const DeepSeekRatio128Page& committed,
                          const DeepSeekRatio128Page& replacement);
  Status validate_publish_tail_cow(
      const DeepSeekRatio128Page& committed,
      const DeepSeekRatio128Page& replacement) const;
  Status rollback(const DeepSeekRatio128Page& page);
  Status release(const DeepSeekRatio128Page& page);
  Status validate_release_published_sequence(std::uint32_t sequence) const;
  Status release_published_sequence(std::uint32_t sequence);

  [[nodiscard]] std::uint32_t free_pages() const noexcept { return free_; }
  [[nodiscard]] std::uint32_t reserved_pages() const noexcept {
    return reserved_;
  }
  [[nodiscard]] std::uint32_t published_pages() const noexcept {
    return published_;
  }
  [[nodiscard]] std::uint32_t page_capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }

 private:
  struct Slot final {
    std::uint32_t generation = 0;
    std::uint32_t sequence = 0;
    std::uint32_t layer_id = 0;
    std::uint32_t logical_page = 0;
    DeepSeekAttentionPageState state = DeepSeekAttentionPageState::kFree;
  };
  explicit DeepSeekRatio128PagePool(std::vector<Slot> slots)
      : slots_(std::move(slots)), free_(static_cast<std::uint32_t>(slots_.size())) {}
  Result<DeepSeekRatio128Page> reserve_free(
      std::uint32_t sequence, std::uint32_t layer_id,
      std::uint32_t logical_page);
  Result<std::uint32_t> validate(
      const DeepSeekRatio128Page& page,
      DeepSeekAttentionPageState expected) const;
  std::vector<Slot> slots_;
  std::uint32_t free_ = 0;
  std::uint32_t reserved_ = 0;
  std::uint32_t published_ = 0;
  bool generation_exhausted_ = false;
};

}  // namespace pih
