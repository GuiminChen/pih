#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_mlp_arena.h"

namespace pih {

struct QwenBf16ArenaSpan final {
  std::uint64_t offset_bytes;
  std::uint64_t size_bytes;

  bool operator==(const QwenBf16ArenaSpan&) const = default;
};

class QwenBf16ExecutionArenaLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static constexpr std::uint64_t kVocabularySize = 151936;
  static constexpr std::uint64_t kFp32Bytes = 4;

  static Result<QwenBf16ExecutionArenaLayout> Create(
      std::uint64_t tokens, std::uint64_t active_logit_rows);

  [[nodiscard]] std::uint64_t tokens() const noexcept { return tokens_; }
  [[nodiscard]] std::uint64_t active_logit_rows() const noexcept {
    return active_logit_rows_;
  }
  [[nodiscard]] QwenBf16ArenaSpan hidden() const noexcept { return hidden_; }
  [[nodiscard]] QwenBf16ArenaSpan normalized() const noexcept {
    return normalized_;
  }
  [[nodiscard]] QwenBf16ArenaSpan query() const noexcept { return query_; }
  [[nodiscard]] QwenBf16ArenaSpan key() const noexcept { return key_; }
  [[nodiscard]] QwenBf16ArenaSpan value() const noexcept { return value_; }
  [[nodiscard]] QwenBf16ArenaSpan attention() const noexcept {
    return attention_;
  }
  [[nodiscard]] std::uint64_t activation_arena_bytes() const noexcept {
    return activation_arena_bytes_;
  }
  [[nodiscard]] const QwenBf16MlpArenaLayout& mlp() const noexcept {
    return mlp_;
  }
  [[nodiscard]] std::uint64_t rope_workspace_bytes() const noexcept {
    return rope_workspace_bytes_;
  }
  [[nodiscard]] std::uint64_t logit_workspace_bytes() const noexcept {
    return logit_workspace_bytes_;
  }

 private:
  QwenBf16ExecutionArenaLayout(
      std::uint64_t tokens, std::uint64_t active_logit_rows,
      QwenBf16ArenaSpan hidden, QwenBf16ArenaSpan normalized,
      QwenBf16ArenaSpan query, QwenBf16ArenaSpan key,
      QwenBf16ArenaSpan value, QwenBf16ArenaSpan attention,
      std::uint64_t activation_arena_bytes, QwenBf16MlpArenaLayout mlp,
      std::uint64_t rope_workspace_bytes, std::uint64_t logit_workspace_bytes)
      : tokens_(tokens),
        active_logit_rows_(active_logit_rows),
        hidden_(hidden),
        normalized_(normalized),
        query_(query),
        key_(key),
        value_(value),
        attention_(attention),
        activation_arena_bytes_(activation_arena_bytes),
        mlp_(mlp),
        rope_workspace_bytes_(rope_workspace_bytes),
        logit_workspace_bytes_(logit_workspace_bytes) {}

  std::uint64_t tokens_;
  std::uint64_t active_logit_rows_;
  QwenBf16ArenaSpan hidden_;
  QwenBf16ArenaSpan normalized_;
  QwenBf16ArenaSpan query_;
  QwenBf16ArenaSpan key_;
  QwenBf16ArenaSpan value_;
  QwenBf16ArenaSpan attention_;
  std::uint64_t activation_arena_bytes_;
  QwenBf16MlpArenaLayout mlp_;
  std::uint64_t rope_workspace_bytes_;
  std::uint64_t logit_workspace_bytes_;
};

}  // namespace pih
