#pragma once
#include "pih/contracts/deepseek_v41_sm103_kernels_v1.h"
#include "linear_fp8.h"
#include "linear_fp4.h"
#include "mhc_launch.h"
#include "rope_launch.h"
#include "expert_dispatch.h"
#include "expert_fp8.h"
#include "indexer_input.h"
#include "indexer_score.h"
#include "indexer_select.h"
#include "attention_assemble.h"
#include "compressor_pool.h"
#include "compressed_kv.h"
#include "window_kv.h"
#include "model_head.h"
#include "token_embedding.h"
#include "router.h"
#include "sampling.h"
#include <bit>
#include <limits>
#include <type_traits>

namespace pih::deepseek_v41::kernel_wire {
struct MhcInitialLaunch { EngramDeviceRegion pre; std::uint32_t tokens{}; std::uintptr_t stream{}; };
// Field order is part of ABI v1. Change ABI version when changing any mapping.
template<class V> void Fields(V& v, EngramDeviceRegion& x) { v(x.address, x.bytes); }
template<class V> void Fields(V& v, Fp8LinearLaunch& x) { v(x.input, x.weight, x.weight_scales, x.quantized, x.activation_scales, x.output, x.error_flag, x.stream, x.rows, x.in_features, x.out_features); }
template<class V> void Fields(V& v, Fp4LinearLaunch& x) { v(x.input, x.weight, x.weight_scales, x.quantized, x.activation_scales, x.output, x.error_flag, x.stream, x.rows, x.in_features, x.out_features); }
template<class V> void Fields(V& v, EngramLookupLaunch& x) { v(x.table, x.scales, x.ids, x.output, x.error_flag, x.stream, x.tokens, x.layer, x.world_size, x.rank); }
template<class V> void Fields(V& v, EngramGateLaunch& x) { v(x.input, x.projected_kv, x.q_weight, x.k_weight, x.mask, x.output, x.error_flag, x.gate_storage, x.stream, x.tokens); }
template<class V> void Fields(V& v, EngramProjectionLaunch& x) { v(x.input, x.weight, x.weight_scales, x.quantized, x.activation_scales, x.output, x.error_flag, x.stream, x.tokens); }
template<class V> void Fields(V& v, EngramLaunch& x) { v(x.lookup, x.projection, x.gate); }
template<class V> void Fields(V& v, RmsNormLaunch& x) { v(x.input, x.weight, x.output, x.error_flag, x.weight_storage, x.stream, x.rows, x.width); }
template<class V> void Fields(V& v, MhcMixLaunch& x) { v(x.residual, x.fn, x.scale, x.base, x.pre, x.post, x.comb, x.error_flag, x.stream, x.tokens); }
template<class V> void Fields(V& v, MhcPreLaunch& x) { v(x.residual, x.pre, x.output, x.error_flag, x.stream, x.tokens); }
template<class V> void Fields(V& v, MhcPostLaunch& x) { v(x.sublayer, x.residual, x.post, x.comb, x.output, x.error_flag, x.stream, x.tokens); }
template<class V> void Fields(V& v, RopeTableLaunch& x) { v(x.positions, x.output, x.error_flag, x.stream, x.tokens, x.layer); }
template<class V> void Fields(V& v, RopeApplyLaunch& x) { v(x.input, x.phases, x.output, x.error_flag, x.stream, x.tokens, x.heads, x.width, x.inverse); }
template<class V> void Fields(V& v, RopeSequenceLaunch& x) { v(x.table, x.first_position, x.stride); }
template<class V> void Fields(V& v, ExpertDispatchLaunch& x) { v(x.indices, x.counts, x.slots, x.error_flag, x.stream, x.tokens, x.layer, x.world_size, x.rank); }
template<class V> void Fields(V& v, ExpertGatherLaunch& x) { v(x.dispatch, x.input, x.route_weights, x.output, x.gathered_weights, x.expert, x.rows); }
template<class V> void Fields(V& v, ExpertScatterLaunch& x) { v(x.dispatch, x.input, x.accumulator, x.expert, x.rows); }
template<class V> void Fields(V& v, ExpertActivationLaunch& x) { v(x.gate, x.up, x.route_weights, x.output, x.error_flag, x.stream, x.rows); }
template<class V> void Fields(V& v, ExpertMergeLaunch& x) { v(x.routed, x.shared, x.output, x.error_flag, x.stream, x.tokens); }
template<class V> void Fields(V& v, IndexerQuantizeLaunch& x) { v(x.values, x.error_flag, x.stream, x.tokens, x.heads); }
template<class V> void Fields(V& v, IndexerKeyProjectionLaunch& x) { v(x.input, x.weight, x.output, x.error_flag, x.stream, x.rows); }
template<class V> void Fields(V& v, IndexerKeyCacheLaunch& x) { v(x.quantize, x.cache, x.first_slot, x.capacity); }
template<class V> void Fields(V& v, IndexerWeightsLaunch& x) { v(x.input, x.weight, x.output, x.error_flag, x.stream, x.tokens, x.heads); }
template<class V> void Fields(V& v, IndexerScoreLaunch& x) { v(x.query, x.key, x.weights, x.output, x.error_flag, x.stream, x.tokens, x.heads, x.positions); }
template<class V> void Fields(V& v, IndexerSelectLaunch& x) { v(x.scores, x.output, x.error_flag, x.candidates, x.stream, x.start, x.tokens, x.ratio, x.positions, x.offset); }
template<class V> void Fields(V& v, IndexerCandidatesLaunch& x) { v(x.scores, x.output, x.error_flag, x.stream, x.start, x.tokens, x.positions); }
template<class V> void Fields(V& v, GroupedOutputLaunch& x) { v(x.input, x.weight, x.output, x.error_flag, x.stream, x.tokens, x.groups); }
template<class V> void Fields(V& v, SparseAttentionLaunch& x) { v(x.query, x.kv, x.sink, x.indices, x.output, x.error_flag, x.stream, x.tokens, x.heads, x.kv_rows, x.picks); }
template<class V> void Fields(V& v, AttentionOutputLaunch& x) { v(x.attention, x.inverse_rope, x.projection); }
template<class V> void Fields(V& v, AttentionLocalOutputLaunch& x) { v(x.grouped, x.linear, x.reduction); }
template<class V> void Fields(V& v, AttentionAssemblyLaunch& x) { v(x.window, x.compressed, x.selected, x.kv, x.indices, x.error_flag, x.stream, x.start, x.tokens, x.ratio); }
template<class V> void Fields(V& v, CompressorPoolLaunch& x) { v(x.values, x.scores, x.state_values, x.state_scores, x.output, x.error_flag, x.stream, x.start, x.tokens); }
template<class V> void Fields(V& v, CompressorProjectionLaunch& x) { v(x.input, x.value_weight, x.gate_weight, x.values, x.scores, x.error_flag, x.stream, x.tokens, x.ratio); }
template<class V> void Fields(V& v, CompressedKvLaunch& x) { v(x.kv, x.cache, x.error_flag, x.stream, x.rows, x.first_slot, x.capacity); }
template<class V> void Fields(V& v, WindowKvLaunch& x) { v(x.kv, x.ring, x.error_flag, x.stream, x.start, x.tokens); }
template<class V> void Fields(V& v, HeadProjectionLaunch& x) { v(x.input, x.weight, x.output, x.error_flag, x.stream, x.world_size, x.rank); }
template<class V> void Fields(V& v, TokenEmbeddingLaunch& x) { v(x.ids, x.weight, x.hidden, x.residual, x.pre, x.error_flag, x.stream, x.tokens, x.world_size, x.rank); }
template<class V> void Fields(V& v, RouterLaunch& x) { v(x.input, x.weight, x.bias, x.image_bias, x.image_mask, x.logits, x.indices, x.route_weights, x.error_flag, x.stream, x.tokens, x.layer); }
template<class V> void Fields(V& v, SamplingParameters& x) { v(x.temperature, x.top_p, x.top_k, x.seed, x.ordinal, x.logprobs, x.top_count, x.suppressed_count, x.suppressed); }
template<class V> void Fields(V& v, SamplingLaunch& x) { v(x.logits, x.scores, x.ids, x.weights, x.reduction, x.stats, x.candidate, x.error_flag, x.stream, x.parameters); }
template<class V> void Fields(V& v, MhcInitialLaunch& x) { v(x.pre, x.tokens, x.stream); }

struct Encoder {
  pih_v41_kernel_command_v1 command{sizeof(command), PIH_V41_KERNEL_ABI_V1, 0, 0, {}};
  bool valid = true;
  template<class... T> void operator()(T&... x) { (One(x), ...); }
  template<class T> void One(T& x) {
    if constexpr (std::is_array_v<T>) { for (auto& value : x) One(value); }
    else if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
      if (command.word_count == PIH_V41_KERNEL_WORDS_V1) { valid = false; return; }
      if constexpr (std::is_same_v<T, float>)
        command.words[command.word_count++] = std::bit_cast<uint32_t>(x);
      else command.words[command.word_count++] = static_cast<uint64_t>(x);
    } else Fields(*this, x);
  }
};
struct Decoder {
  const pih_v41_kernel_command_v1& command;
  uint32_t cursor{}; bool valid = true;
  template<class... T> void operator()(T&... x) { (One(x), ...); }
  template<class T> void One(T& x) {
    if constexpr (std::is_array_v<T>) { for (auto& value : x) One(value); }
    else if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
      if (cursor >= command.word_count) { valid = false; return; }
      const auto word = command.words[cursor++];
      if constexpr (std::is_same_v<T, float>) {
        if (word > UINT32_MAX) { valid = false; return; }
        x = std::bit_cast<float>(static_cast<uint32_t>(word));
      } else if constexpr (std::is_same_v<T, EngramStorage>) {
        if (word > static_cast<uint64_t>(EngramStorage::kF32)) { valid = false; return; }
        x = static_cast<T>(word);
      } else {
        if (word > static_cast<uint64_t>(std::numeric_limits<T>::max())) { valid = false; return; }
        x = static_cast<T>(word);
      }
    } else Fields(*this, x);
  }
};
}  // namespace pih::deepseek_v41::kernel_wire

