#include "token_stop.h"
#include <algorithm>
#include <cstring>
#include <atomic>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Utf8(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const auto c = static_cast<unsigned char>(text[i++]);
    if (c < 0x80) continue;
    unsigned count = 0, cp = 0, minimum = 0;
    if (c >= 0xc2 && c <= 0xdf) { count = 1; cp = c & 31; minimum = 0x80; }
    else if (c >= 0xe0 && c <= 0xef) { count = 2; cp = c & 15; minimum = 0x800; }
    else if (c >= 0xf0 && c <= 0xf4) { count = 3; cp = c & 7; minimum = 0x10000; }
    else return false;
    if (count > text.size() - i) return false;
    while (count--) {
      const auto next = static_cast<unsigned char>(text[i++]);
      if ((next & 0xc0) != 0x80) return false;
      cp = (cp << 6) | (next & 63);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
  }
  return true;
}
}
Result<TokenStopState> TokenStopState::Create(TokenStopConfig c) {
  if (!c.maximum || c.maximum > 1048576 || c.minimum > c.maximum || c.token_count > 17 || c.pattern_count > 16)
    return Status::InvalidArgument("Stop policy bounds invalid");
  for (unsigned i = 0; i < 17; ++i) {
    if (i >= c.token_count) {
      if (c.tokens[i]) return Status::InvalidArgument("Stop token padding must be zero");
    } else {
      if (c.tokens[i] >= 129280) return Status::InvalidArgument("Stop token outside vocabulary");
      for (unsigned j = 0; j < i; ++j) if (c.tokens[i] == c.tokens[j])
        return Status::InvalidArgument("Stop tokens must be normalized and deduplicated");
    }
  }
  std::uint32_t total = 0, longest = 0;
  for (unsigned i = 0; i < 16; ++i) {
    const auto& p = c.patterns[i];
    if (i >= c.pattern_count) {
      if (!p.empty()) return Status::InvalidArgument("Unused stop pattern must be empty");
    } else {
      if (p.empty() || p.size() > 256 || !Utf8(p)) return Status::InvalidArgument("Stop pattern must be bounded nonempty UTF-8");
      for (unsigned j = 0; j < i; ++j) if (p == c.patterns[j])
        return Status::InvalidArgument("Stop patterns must be normalized and deduplicated");
      total += static_cast<unsigned>(p.size()); longest = std::max(longest, static_cast<unsigned>(p.size()));
    }
  }
  if (total > 2048) return Status::InvalidArgument("Stop patterns exceed total byte budget");
  static std::atomic<std::uint64_t> next_instance{1};
  auto instance = next_instance.load(std::memory_order_relaxed);
  do {
    if (instance == std::numeric_limits<std::uint64_t>::max()) return Status::FailedPrecondition("Stop state identity space exhausted");
  } while (!next_instance.compare_exchange_weak(instance, instance + 1, std::memory_order_relaxed));
  TokenStopState result; result.config_ = std::move(c); result.longest_ = longest; result.valid_ = true;
  result.instance_ = instance; return result;
}
TokenStopState::TokenStopState(TokenStopState&& other) noexcept
    : config_(std::move(other.config_)), pending_(other.pending_), accepted_(other.accepted_), pending_size_(other.pending_size_),
      longest_(other.longest_), finish_(other.finish_), valid_(other.valid_), instance_(other.instance_) { other.valid_ = false; }
Status TokenStopState::ApplySuppression(SamplingParameters& p) const {
  if (!valid_ || finish_ != TokenFinish::kNone) return Status::FailedPrecondition("Stop state is terminal or moved from");
  auto next = p;
  next.suppressed_count = 0; std::fill(std::begin(next.suppressed), std::end(next.suppressed), 0U);
  if (accepted_ + 1 < config_.minimum) {
    next.suppressed_count = config_.token_count;
    std::copy_n(config_.tokens, config_.token_count, next.suppressed);
  }
  const auto valid = ValidateSamplingParameters(next); if (!valid.ok()) return valid;
  p = next; return Status::Ok();
}
Result<TokenStopTransition> TokenStopState::Preview(const SamplingObservation& o, std::string_view bytes) const {
  if (!valid_ || finish_ != TokenFinish::kNone || accepted_ >= config_.maximum)
    return Status::FailedPrecondition("Stop state is terminal or moved from");
  if (o.candidate.token_id >= 129280 || bytes.size() > 8192)
    return Status::InvalidArgument("Stop preview token or tokenizer-byte extent invalid");
  TokenStopTransition t;
  t.owner_ = this; t.owner_generation_ = instance_; t.observation_ = o; t.predecessor_ = accepted_; t.accepted_ = accepted_ + 1;
  t.pending_ = pending_; t.pending_size_ = pending_size_;
  const auto emit = [&](const char* data, std::size_t size) {
    std::copy_n(data, size, t.visible_.data() + t.visible_size_); t.visible_size_ += static_cast<unsigned>(size);
  };
  const auto flush = [&]() { emit(t.pending_.data(), t.pending_size_); t.pending_size_ = 0; };
  bool stop_token = false;
  for (unsigned i = 0; i < config_.token_count; ++i) stop_token |= o.candidate.token_id == config_.tokens[i];
  if (t.accepted_ < config_.minimum) {
    if (stop_token) return Status::FailedPrecondition("Sampling selected a min-token-suppressed stop token");
    // Early bytes never seed the matcher that starts at the threshold token.
    flush(); emit(bytes.data(), bytes.size());
  } else if (stop_token) {
    flush(); t.finish_ = TokenFinish::kStopToken;
  } else if (!longest_) {
    emit(bytes.data(), bytes.size());
  } else {
    for (const char byte : bytes) {
      t.pending_[t.pending_size_++] = byte;
      std::size_t match = 0;
      for (unsigned i = 0; i < config_.pattern_count; ++i) {
        const auto& pattern = config_.patterns[i];
        if (pattern.size() <= t.pending_size_ && pattern.size() > match &&
            std::memcmp(t.pending_.data() + t.pending_size_ - pattern.size(), pattern.data(), pattern.size()) == 0)
          match = pattern.size();
      }
      if (match) {
        emit(t.pending_.data(), t.pending_size_ - match); t.pending_size_ = 0;
        t.finish_ = TokenFinish::kStopString; break;
      }
      if (t.pending_size_ == longest_) {
        emit(t.pending_.data(), 1);
        std::move(t.pending_.begin() + 1, t.pending_.begin() + t.pending_size_, t.pending_.begin()); --t.pending_size_;
      }
    }
  }
  if (t.finish_ == TokenFinish::kNone && t.accepted_ == config_.maximum) { flush(); t.finish_ = TokenFinish::kLength; }
  std::fill(t.pending_.begin() + t.pending_size_, t.pending_.end(), 0);
  return t;
}
Status TokenStopState::Commit(const TokenStopTransition& t) {
  if (!valid_ || finish_ != TokenFinish::kNone || t.owner_ != this || t.owner_generation_ != instance_ ||
      t.predecessor_ != accepted_ || t.accepted_ != accepted_ + 1)
    return Status::FailedPrecondition("Stop transition is stale, foreign, terminal or already committed");
  pending_ = t.pending_; pending_size_ = t.pending_size_; accepted_ = t.accepted_; finish_ = t.finish_;
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
