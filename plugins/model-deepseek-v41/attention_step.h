#pragma once
#include "config.h"
#include <vector>

namespace pih::deepseek_v41 {
struct CacheWrite final { std::uint32_t input_token, slot; };
struct CompressedPosition final { std::uint32_t slot, original_position; };
struct AttentionStep final {
  bool prefill = false;
  std::uint32_t layer = 0, start = 0, end = 0, window_columns = 0;
  // Prefill indices address this step's KV array; decode indices address the ring.
  std::vector<std::uint32_t> query_positions;
  std::vector<std::int32_t> window_indices;
  std::vector<CacheWrite> window_writes;
  std::uint32_t compression_ratio = 0, compressed_length = 0;
  std::uint32_t kv_source = UINT32_MAX, index_key_source = UINT32_MAX, topk_source = UINT32_MAX;
  // Causal count per query, not the full end-of-step compressed cache length.
  std::vector<std::uint32_t> visible_compressed;
  // Only actual KV/index-key owners publish new positions. Later top-k owners
  // read index keys from kv_source; they do not own a separate index-key table.
  std::vector<CompressedPosition> compressed_writes;
  std::vector<CacheWrite> compressor_tail_writes;
};
// Pure ordinary autoregressive step planning; no mutation or cache admission.
// Prefill starts at zero (1..4096 tokens); later steps are single-token decode.
// Not a DSpark speculative-block plan or a chunked-prefill implementation.
Result<AttentionStep> BuildAttentionStep(const FlashConfig& config, std::uint32_t layer,
    std::uint32_t start, std::uint32_t tokens);
}  // namespace pih::deepseek_v41
