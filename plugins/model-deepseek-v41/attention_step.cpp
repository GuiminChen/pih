#include "attention_step.h"
#include <algorithm>

namespace pih::deepseek_v41 {
Result<AttentionStep> BuildAttentionStep(const FlashConfig& config, std::uint32_t layer,
    std::uint32_t start, std::uint32_t tokens) {
  if (config.config_sha256() == Sha256Digest{} || layer >= FlashConfig::kMainLayers ||
      !tokens || tokens > 4096 || (start && tokens != 1) || start >= FlashConfig::kMaximumPositions ||
      tokens > FlashConfig::kMaximumPositions - start)
    return Status::InvalidArgument("V4.1 ordinary attention step bounds/config invalid");
  const auto& sharing = config.attention_sharing()[layer];
  AttentionStep step;
  step.prefill = start == 0;
  step.layer = layer; step.start = start; step.end = start + tokens;
  step.window_columns = step.prefill ? std::min(tokens, 128U) : 128U;
  step.query_positions.reserve(tokens);
  step.window_indices.reserve(static_cast<std::size_t>(tokens) * step.window_columns);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto position = start + token;
    step.query_positions.push_back(position);
    for (std::uint32_t column = 0; column < step.window_columns; ++column) {
      if (step.prefill) {
        const auto index = (token >= 127 ? token - 127 : 0) + column;
        step.window_indices.push_back(index <= token ? static_cast<std::int32_t>(index) : -1);
      } else {
        const auto slot = (start % 128 + 1 + column) % 128;
        step.window_indices.push_back(slot <= start ? static_cast<std::int32_t>(slot) : -1);
      }
    }
  }
  // For long prefill only the last 128 positions survive, in absolute-position
  // modulo slots. Earlier prefill queries still address the full local KV array.
  const auto first_retained = step.prefill && tokens > 128 ? tokens - 128 : 0;
  for (std::uint32_t token = first_retained; token < tokens; ++token)
    step.window_writes.push_back({token, (start + token) % 128});
  step.compression_ratio = sharing.compression_ratio;
  if (!step.compression_ratio) return step;
  const auto ratio = step.compression_ratio;
  step.kv_source = sharing.kv_source;
  step.index_key_source = sharing.kv_source;
  step.topk_source = sharing.index_source;
  step.compressed_length = step.end / ratio;
  step.visible_compressed.reserve(tokens);
  for (const auto position : step.query_positions)
    step.visible_compressed.push_back((position + 1) / ratio);
  if (!sharing.owns_kv) return step;
  const auto first_group = start / ratio;
  for (std::uint32_t group = first_group; group < step.compressed_length; ++group)
    step.compressed_writes.push_back({group, group * ratio});
  if (ratio > 1) {
    if (step.prefill) {
      const auto cutoff = tokens - tokens % ratio;
      for (std::uint32_t token = cutoff; token < tokens; ++token)
        step.compressor_tail_writes.push_back({token, token - cutoff});
    } else {
      step.compressor_tail_writes.push_back({0, start % ratio});
    }
  }
  return step;
}
}  // namespace pih::deepseek_v41
