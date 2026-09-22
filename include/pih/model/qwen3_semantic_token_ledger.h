#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_semantic_outcome_recorder.h"

namespace pih {

enum class QwenSemanticTokenLedgerState : std::uint8_t {
  kCollecting,
  kSealed,
  kPoisoned,
};

class QwenSemanticTokenLedger final {
 public:
  static constexpr std::uint32_t kMaximumTokens = 40960;
  static constexpr std::int64_t kVocabularySize = 151936;

  Status commit(std::uint32_t first_position,
                std::span<const std::int64_t> accepted_tokens,
                std::uint32_t decision_position,
                std::int64_t sampled_token);
  Status seal(QwenSemanticOutcomeRecorder& recorder);

  [[nodiscard]] std::uint32_t committed_tokens() const noexcept {
    return committed_tokens_;
  }
  [[nodiscard]] std::uint32_t decision_count() const noexcept {
    return decision_count_;
  }
  [[nodiscard]] QwenSemanticTokenLedgerState state() const noexcept {
    return state_;
  }

 private:
  Status poison(const char* message);
  static void append_u32(std::vector<std::byte>& wire, std::uint32_t value);
  static void set_u32(std::vector<std::byte>& wire, std::size_t offset,
                      std::uint32_t value);
  void ensure_headers();

  std::vector<std::byte> trajectory_wire_;
  std::vector<std::byte> ledger_wire_;
  std::uint32_t committed_tokens_ = 0;
  std::uint32_t decision_count_ = 0;
  std::uint32_t commit_count_ = 0;
  QwenSemanticTokenLedgerState state_ =
      QwenSemanticTokenLedgerState::kCollecting;
};

}  // namespace pih
