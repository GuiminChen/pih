#include "engram_hash.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
constexpr std::uint32_t kCompressedVocabulary = 99092;
constexpr std::array<std::uint32_t, 2> kTableRows{384006168, 384016682};
// Frozen PCG64/default_rng results from the pinned reference formula: seeds
// 10007 * layer (1,14), high=(INT64_MAX//99092)//2, then values*2+1.
// Generated with NumPy 2.5.3 offline; the native runtime has no RNG/Python import.
constexpr std::array<std::array<std::uint64_t, 4>, 2> kMultipliers{{
    {76632096046245ULL, 4839876093313ULL, 35959672319349ULL, 73987337458391ULL},
    {67716810739261ULL, 51510806800915ULL, 30921347202721ULL, 82619226485591ULL}}};
static_assert([] {
  for (const auto& layer : kMultipliers)
    for (const auto multiplier : layer)
      if (multiplier % 2 != 1 || multiplier > static_cast<std::uint64_t>(INT64_MAX) / kCompressedVocabulary)
        return false;
  return true;
}());
bool Prime(std::uint32_t value) {
  if (value < 2) return false;
  if (value % 2 == 0) return value == 2;
  for (std::uint32_t divisor = 3; divisor <= value / divisor; divisor += 2)
    if (value % divisor == 0) return false;
  return true;
}
}  // namespace
Result<EngramHashState> EngramHashState::Create(const FlashConfig& config,
    std::span<const std::byte> map, const Sha256Digest& expected) {
  if (config.config_sha256() == Sha256Digest{} || expected == Sha256Digest{} ||
      map.size() != FlashConfig::kVocabularySize * sizeof(std::uint32_t))
    return Status::InvalidArgument("Engram requires admitted V4.1 config and complete U32 token map");
  auto digest = sha256(map);
  if (!digest.ok()) return digest.status();
  if (*digest != expected) return Status::FailedPrecondition("Engram compressed token map digest differs");
  EngramHashState result;
  result.token_map_.reserve(FlashConfig::kVocabularySize);
  std::uint32_t next_id = 0;
  for (std::size_t i = 0; i < FlashConfig::kVocabularySize; ++i) {
    std::uint32_t value = 0;
    for (unsigned byte = 0; byte < 4; ++byte)
      value |= static_cast<std::uint32_t>(std::to_integer<unsigned>(map[i * 4 + byte])) << (byte * 8);
    if (value >= kCompressedVocabulary || value > next_id)
      return Status::InvalidArgument("Engram token map violates compressed first-appearance IDs");
    if (value == next_id) ++next_id;
    result.token_map_.push_back(value);
  }
  if (next_id != kCompressedVocabulary)
    return Status::InvalidArgument("Engram token map does not cover the frozen compressed vocabulary");
  result.pad_id_ = result.token_map_[2];
  // The reference restarts each head-group search at 16000000-1 and skips all
  // previously selected primes. This is exactly the first 48 distinct primes
  // above that start, allocated in layer/ngram/head order.
  std::uint32_t candidate = 15999999;
  for (std::size_t layer = 0; layer < result.buckets_.size(); ++layer) {
    auto& buckets = result.buckets_[layer];
    std::uint64_t rows = 0;
    for (std::size_t column = 0; column < buckets.primes.size(); ++column) {
      do { ++candidate; } while (!Prime(candidate));
      buckets.primes[column] = candidate;
      buckets.offsets[column] = static_cast<std::uint32_t>(rows);
      rows += candidate;
    }
    if (rows != kTableRows[layer]) return Status::Internal("Engram prime layout differs from frozen table rows");
    buckets.rows = static_cast<std::uint32_t>(rows);
  }
  return result;
}
void EngramHashState::Reset() noexcept {
  history_.fill(UINT32_MAX);
  position_ = 0;
}
Result<std::vector<EngramTokenHashes>> EngramHashState::Append(std::uint64_t start_position,
    std::span<const std::uint32_t> tokens, std::span<const std::uint8_t> participation) {
  if (start_position != position_ || tokens.empty() || tokens.size() > 4096 ||
      position_ > FlashConfig::kMaximumPositions || tokens.size() > FlashConfig::kMaximumPositions - position_ ||
      (!participation.empty() && participation.size() != tokens.size()))
    return Status::InvalidArgument("Engram append position, length or mask size invalid");
  for (std::size_t i = 0; i < tokens.size(); ++i)
    if (tokens[i] >= token_map_.size() || (!participation.empty() && participation[i] > 1))
      return Status::InvalidArgument("Engram token ID or participation value invalid");
  std::vector<EngramTokenHashes> output(tokens.size());
  auto history = history_;
  for (std::size_t token = 0; token < tokens.size(); ++token) {
    const auto compressed = !participation.empty() && !participation[token] ? UINT32_MAX : token_map_[tokens[token]];
    const std::array<std::uint32_t, 4> sequence{compressed, history[0], history[1], history[2]};
    std::array<std::uint32_t, 4> ids{};
    bool blocked = false;
    for (std::size_t shift = 0; shift < ids.size(); ++shift) {
      blocked = blocked || sequence[shift] == UINT32_MAX;
      ids[shift] = blocked ? pad_id_ : sequence[shift];
    }
    for (std::size_t layer = 0; layer < buckets_.size(); ++layer) {
      std::uint64_t rolling = ids[0] * kMultipliers[layer][0];
      for (std::size_t ngram = 1; ngram < 4; ++ngram) {
        rolling ^= ids[ngram] * kMultipliers[layer][ngram];
        for (std::size_t head = 0; head < 8; ++head) {
          const auto column = (ngram - 1) * 8 + head;
          output[token].ids[layer][column] = static_cast<std::uint32_t>(rolling % buckets_[layer].primes[column]) +
              buckets_[layer].offsets[column];
        }
      }
    }
    history = {compressed, history[0], history[1]};
  }
  history_ = history;
  position_ += tokens.size();
  return output;
}
}  // namespace pih::deepseek_v41
