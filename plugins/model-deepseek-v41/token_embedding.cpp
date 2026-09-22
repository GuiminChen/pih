#include "token_embedding.h"
#include <array>
#include <limits>

namespace pih::deepseek_v41 {
Status ValidateTokenEmbedding(const TokenEmbeddingLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || x.rank >= x.world_size ||
      (x.world_size != 1 && x.world_size != 2 && x.world_size != 4 && x.world_size != 8))
    return Status::InvalidArgument("Token embedding shape, rank or stream invalid");
  const std::array regions{x.ids, x.weight, x.hidden, x.residual, x.pre, x.error_flag};
  const std::array<std::uint64_t, 6> sizes{x.tokens * 4ULL, (129280ULL / x.world_size) * 5120 * 2,
      x.tokens * 5120ULL * 2, x.tokens * 4ULL * 5120 * 2, x.tokens * 4ULL * 4, 4};
  const std::array<unsigned, 6> alignments{4, 2, 2, 2, 4, 4};
  for (unsigned i = 0; i < regions.size(); ++i) {
    const auto r = regions[i];
    if (!r.address || r.address % alignments[i] || r.bytes != sizes[i] || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
      return Status::InvalidArgument("Token embedding storage extent or alignment invalid");
    for (unsigned j = 0; j < i; ++j)
      if (r.address < regions[j].address + regions[j].bytes && regions[j].address < r.address + r.bytes)
        return Status::InvalidArgument("Token embedding buffers must be disjoint");
  }
  return ValidateMhcInitialPre(x.pre, x.tokens, x.stream);
}
Status LaunchTokenEmbeddingSingleRank(const TokenEmbeddingLaunch& x) {
  if (x.world_size != 1 || x.rank) return Status::InvalidArgument("Single-rank embedding requires world=1/rank=0");
  const auto lookup = LaunchTokenEmbeddingLookup(x); if (!lookup.ok()) return lookup;
  return LaunchTokenEmbeddingExpand(x);
}
}  // namespace pih::deepseek_v41
