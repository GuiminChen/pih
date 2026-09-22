#pragma once
#include "sampling.h"
#include <array>
#include <string>
#include <string_view>

namespace pih::deepseek_v41 {
enum class TokenFinish { kNone, kStopToken, kStopString, kLength };
struct TokenStopConfig final {
  std::uint32_t minimum = 0, maximum = 0;
  std::uint32_t token_count = 0, tokens[17]{};
  std::uint32_t pattern_count = 0;
  std::array<std::string, 16> patterns;
};
class TokenStopState;
class TokenStopTransition final {
 public:
  TokenFinish finish() const noexcept { return finish_; }
  std::string_view visible_bytes() const noexcept { return {visible_.data(), visible_size_}; }
  std::uint32_t accepted_count() const noexcept { return accepted_; }
  const SamplingObservation& observation() const noexcept { return observation_; }
 private:
  friend class TokenStopState;
  const TokenStopState* owner_ = nullptr;
  std::uint64_t owner_generation_ = 0;
  SamplingObservation observation_{};
  std::uint32_t predecessor_ = 0, accepted_ = 0, visible_size_ = 0, pending_size_ = 0;
  TokenFinish finish_ = TokenFinish::kNone;
  std::array<char, 8448> visible_{};
  std::array<char, 256> pending_{};
};
// Raw tokenizer-byte stopping only; tool/parser classification and atomic
// distributed acceptance are controller responsibilities. Preview never commits.
class TokenStopState final {
 public:
  static Result<TokenStopState> Create(TokenStopConfig config);
  TokenStopState(const TokenStopState&) = delete;
  TokenStopState& operator=(const TokenStopState&) = delete;
  TokenStopState(TokenStopState&& other) noexcept;
  TokenStopState& operator=(TokenStopState&&) = delete;
  Result<TokenStopTransition> Preview(const SamplingObservation& observation, std::string_view token_bytes) const;
  Status Commit(const TokenStopTransition& transition);
  Status ApplySuppression(SamplingParameters& parameters) const;
  std::uint32_t accepted_count() const noexcept { return accepted_; }
  TokenFinish finish() const noexcept { return finish_; }
 private:
  TokenStopState() = default;
  TokenStopConfig config_;
  std::array<char, 256> pending_{};
  std::uint32_t accepted_ = 0, pending_size_ = 0, longest_ = 0;
  TokenFinish finish_ = TokenFinish::kNone;
  bool valid_ = false;
  std::uint64_t instance_ = 0;
};
}  // namespace pih::deepseek_v41
