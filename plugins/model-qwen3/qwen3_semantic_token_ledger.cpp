#include "pih/model/qwen3_semantic_token_ledger.h"

#include <algorithm>
#include <array>

namespace pih {
namespace {

constexpr std::array<std::byte, 8> kTrajectoryMagic{
    std::byte{'X'}, std::byte{'Q'}, std::byte{'T'}, std::byte{'R'},
    std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
constexpr std::array<std::byte, 8> kLedgerMagic{
    std::byte{'X'}, std::byte{'Q'}, std::byte{'L'}, std::byte{'D'},
    std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};

}  // namespace

Status QwenSemanticTokenLedger::poison(const char* message) {
  state_ = QwenSemanticTokenLedgerState::kPoisoned;
  return Status::FailedPrecondition(message);
}

void QwenSemanticTokenLedger::append_u32(std::vector<std::byte>& wire,
                                         std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    wire.push_back(static_cast<std::byte>(value >> (index * 8U)));
  }
}

void QwenSemanticTokenLedger::set_u32(std::vector<std::byte>& wire,
                                      std::size_t offset,
                                      std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    wire[offset + index] = static_cast<std::byte>(value >> (index * 8U));
  }
}

void QwenSemanticTokenLedger::ensure_headers() {
  if (!trajectory_wire_.empty()) return;
  trajectory_wire_.insert(trajectory_wire_.end(), kTrajectoryMagic.begin(),
                          kTrajectoryMagic.end());
  ledger_wire_.insert(ledger_wire_.end(), kLedgerMagic.begin(),
                      kLedgerMagic.end());
  trajectory_wire_.resize(16);
  ledger_wire_.resize(16);
}

Status QwenSemanticTokenLedger::commit(
    std::uint32_t first_position,
    std::span<const std::int64_t> accepted_tokens,
    std::uint32_t decision_position, std::int64_t sampled_token) {
  if (state_ != QwenSemanticTokenLedgerState::kCollecting ||
      accepted_tokens.empty() || first_position != committed_tokens_ ||
      accepted_tokens.size() > kMaximumTokens - committed_tokens_ ||
      decision_position != first_position + accepted_tokens.size() - 1 ||
      sampled_token < 0 || sampled_token >= kVocabularySize ||
      std::any_of(accepted_tokens.begin(), accepted_tokens.end(),
                  [](std::int64_t token) {
                    return token < 0 || token >= kVocabularySize;
                  })) {
    return poison("Qwen semantic token commit is invalid");
  }
  ensure_headers();
  append_u32(ledger_wire_, first_position);
  append_u32(ledger_wire_, static_cast<std::uint32_t>(accepted_tokens.size()));
  for (const auto token : accepted_tokens) {
    append_u32(ledger_wire_, static_cast<std::uint32_t>(token));
  }
  append_u32(trajectory_wire_, decision_position);
  append_u32(trajectory_wire_, static_cast<std::uint32_t>(sampled_token));
  committed_tokens_ += static_cast<std::uint32_t>(accepted_tokens.size());
  ++decision_count_;
  ++commit_count_;
  return Status::Ok();
}

Status QwenSemanticTokenLedger::seal(
    QwenSemanticOutcomeRecorder& recorder) {
  if (state_ != QwenSemanticTokenLedgerState::kCollecting ||
      committed_tokens_ == 0 || decision_count_ == 0 || commit_count_ == 0) {
    return poison("Qwen semantic token ledger is incomplete");
  }
  set_u32(trajectory_wire_, 8, decision_count_);
  set_u32(trajectory_wire_, 12, committed_tokens_);
  set_u32(ledger_wire_, 8, commit_count_);
  set_u32(ledger_wire_, 12, committed_tokens_);
  Status status = recorder.record_greedy_trajectory(trajectory_wire_);
  if (status.ok()) {
    status = recorder.record_accepted_token_ledger(ledger_wire_);
  }
  if (!status.ok()) {
    state_ = QwenSemanticTokenLedgerState::kPoisoned;
    return status;
  }
  state_ = QwenSemanticTokenLedgerState::kSealed;
  return Status::Ok();
}

}  // namespace pih
