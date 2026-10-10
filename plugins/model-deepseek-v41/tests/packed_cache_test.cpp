#include "../sequence_cache.h"
#include "../engram_hash.h"
#include <cstdlib>
#include <fstream>
#include <iterator>

using namespace pih;
using namespace pih::deepseek_v41;
static void Require(bool condition) { if (!condition) std::abort(); }
int main(int argc, char** argv) {
  Require(argc == 2);
  std::ifstream input(argv[1]); Require(input.good());
  std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  auto config = FlashConfig::Parse(json); Require(config.ok());
  constexpr std::uint64_t budget = 64ULL << 20;
  for (const unsigned sequences : {1U, 2U, 4U}) {
    auto cache = PackedSequenceCachePlan::Create(*config, 32, sequences, budget); Require(cache.ok());
    const EngramDeviceRegion allocation{0x10000000, cache->bytes()};
    for (unsigned slot = 0; slot < sequences; ++slot) {
      auto arena = cache->SequenceArena(allocation, slot); Require(arena.ok());
      Require(arena->bytes == cache->sequence_plan().bytes());
      auto layer = cache->Bind(allocation, slot, 2); Require(layer.ok());
      Require(layer->window.bytes == 128ULL * 512 * 2 && layer->compressed_capacity == 16);
      for (unsigned previous = 0; previous < slot; ++previous) {
        auto other = cache->Bind(allocation, previous, 2); Require(other.ok());
        Require(other->window.address + other->window.bytes <= layer->window.address);
        Require(other->compressed.address + other->compressed.bytes <= layer->compressed.address);
        Require(other->pool_values.address + other->pool_values.bytes <= layer->pool_values.address);
      }
    }
    Require(!cache->SequenceArena(allocation, sequences).ok());
    Require(!PackedSequenceCachePlan::Create(*config, 32, sequences, cache->bytes() - 1).ok());
  }
  // Actual Engram n-gram append results match independent execution even when
  // differently sized member appends interleave and another member is reset.
  std::vector<std::byte> map(FlashConfig::kVocabularySize * 4ULL);
  for (unsigned token = 0; token < FlashConfig::kVocabularySize; ++token) {
    const auto compressed = std::min(token, 99091U);
    for (unsigned byte = 0; byte < 4; ++byte) map[token * 4 + byte] = std::byte((compressed >> (byte * 8)) & 255);
  }
  auto digest = sha256(map); Require(digest.ok());
  auto base = EngramHashState::Create(*config, map, *digest); Require(base.ok());
  auto a = *base, b = *base, solo = *base;
  const std::array<std::uint32_t, 2> prefix{10, 20};
  auto packed_a = a.Append(0, prefix); auto solo_a = solo.Append(0, prefix);
  Require(packed_a.ok() && solo_a.ok() && (*packed_a)[1].ids == (*solo_a)[1].ids);
  const std::array<std::uint32_t, 3> other_tokens{30, 40, 50};
  Require(b.Append(0, other_tokens).ok());
  const std::array<std::uint32_t, 1> next{60};
  auto packed_next = a.Append(2, next); auto solo_next = solo.Append(2, next);
  Require(packed_next.ok() && solo_next.ok() && packed_next->front().ids == solo_next->front().ids);
  b.Reset(); Require(b.position() == 0 && a.position() == 3);
  Require(!a.Append(0, next).ok() && a.position() == 3);
}
