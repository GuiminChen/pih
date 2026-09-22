#include "uploaded_bindings.h"
#include <new>

namespace pih::deepseek_v41 {
namespace {
Status Rank(const BackboneWeightUpload& weights, std::uint32_t world, std::uint32_t rank) {
  if (weights.state() != WeightUploadState::kComplete || weights.catalog().world_size() != world ||
      weights.catalog().rank() != rank)
    return Status::FailedPrecondition("Uploaded weight rank does not match execution descriptor");
  return Status::Ok();
}
Result<ExpertProjectionWeights> Projection(const BackboneWeightUpload& weights, const std::string& name) {
  auto w = weights.Find(name + ".weight"); if (!w.ok()) return w.status();
  auto s = weights.Find(name + ".scale"); if (!s.ok()) return s.status();
  return ExpertProjectionWeights{*w, *s};
}
Result<ExpertWeights> Expert(const BackboneWeightUpload& weights, const std::string& name) {
  auto gate = Projection(weights, name + ".w1"); if (!gate.ok()) return gate.status();
  auto up = Projection(weights, name + ".w3"); if (!up.ok()) return up.status();
  auto down = Projection(weights, name + ".w2"); if (!down.ok()) return down.status();
  return ExpertWeights{*gate, *up, *down};
}
}
Result<TokenEmbeddingLaunch> BindUploadedEmbedding(const BackboneWeightUpload& weights, TokenEmbeddingLaunch x) {
  auto status = Rank(weights, x.world_size, x.rank); if (!status.ok()) return status;
  auto weight = weights.Find("embed.weight"); if (!weight.ok()) return weight.status();
  x.weight = *weight;
  status = ValidateTokenEmbedding(x); if (!status.ok()) return status;
  status = weights.ValidateScratch(std::array{x.hidden, x.residual, x.pre, x.error_flag});
  if (!status.ok()) return status;
  return x;
}
Result<ModelHeadLaunch> BindUploadedHead(const BackboneWeightUpload& weights, ModelHeadLaunch x) {
  auto status = Rank(weights, x.projection.world_size, x.projection.rank); if (!status.ok()) return status;
  auto weight = weights.Find("head.weight"); if (!weight.ok()) return weight.status();
  auto norm = weights.Find("norm.weight"); if (!norm.ok()) return norm.status();
  x.projection.weight = *weight; x.input.norm.weight = *norm; x.input.norm.weight_storage = EngramStorage::kBF16;
  status = ValidateModelHead(x); if (!status.ok()) return status;
  status = weights.ValidateScratch(std::array{x.input.collapse.output, x.input.norm.output,
      x.projection.output, x.projection.error_flag});
  if (!status.ok()) return status;
  return x;
}
Result<std::vector<ExpertWeights>> UploadedRoutedExperts(const BackboneWeightUpload& weights,
    const FlashConfig& config, const ExpertDispatchLaunch& d) {
  auto status = Rank(weights, d.world_size, d.rank); if (!status.ok()) return status;
  if (d.layer >= 40 || config.config_sha256() != weights.catalog().config_sha256())
    return Status::InvalidArgument("Uploaded expert config/layer differs");
  status = ValidateExpertDispatch(d); if (!status.ok()) return status;
  status = weights.ValidateScratch(std::array{d.counts, d.slots, d.error_flag}); if (!status.ok()) return status;
  try {
    const auto count = 384 / d.world_size;
    std::vector<ExpertWeights> result; result.reserve(count);
    const auto prefix = "layers." + std::to_string(d.layer) + ".ffn.experts.";
    for (std::uint32_t id = d.rank * count; id < (d.rank + 1) * count; ++id) {
      auto expert = Expert(weights, prefix + std::to_string(id)); if (!expert.ok()) return expert.status();
      result.push_back(*expert);
    }
    return result;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Expert binding allocation failed"); }
}
Result<SharedExpertLaunch> BindUploadedSharedExpert(const BackboneWeightUpload& weights,
    std::uint32_t layer, SharedExpertLaunch x) {
  if (layer >= 40) return Status::InvalidArgument("Uploaded shared expert layer outside backbone");
  try {
    auto expert = Expert(weights, "layers." + std::to_string(layer) + ".ffn.shared_experts");
    if (!expert.ok()) return expert.status();
    x.gate.weight = expert->gate.weight; x.gate.weight_scales = expert->gate.scales;
    x.up.weight = expert->up.weight; x.up.weight_scales = expert->up.scales;
    x.down.weight = expert->down.weight; x.down.weight_scales = expert->down.scales;
    auto status = ValidateSharedExpert(x); if (!status.ok()) return status;
    for (const auto* p : {&x.gate, &x.up, &x.down}) {
      status = weights.ValidateScratch(std::array{p->quantized, p->activation_scales, p->output, p->error_flag});
      if (!status.ok()) return status;
    }
    status = weights.ValidateScratch(std::array{x.activation.output}); if (!status.ok()) return status;
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Shared expert binding allocation failed"); }
}
Result<MhcSublayerInputLaunch> BindUploadedMhc(const BackboneWeightUpload& weights,
    std::uint32_t layer, UploadedSublayer sublayer, MhcSublayerInputLaunch x) {
  if (layer >= 40 || (sublayer != UploadedSublayer::kAttention && sublayer != UploadedSublayer::kFfn))
    return Status::InvalidArgument("Uploaded mHC layer or sublayer invalid");
  try {
    const auto prefix = "layers." + std::to_string(layer) + ".";
    const std::string branch = sublayer == UploadedSublayer::kAttention ? "attn" : "ffn";
    auto fn = weights.Find(prefix + "hc_" + branch + "_fn"); if (!fn.ok()) return fn.status();
    auto scale = weights.Find(prefix + "hc_" + branch + "_scale"); if (!scale.ok()) return scale.status();
    auto base = weights.Find(prefix + "hc_" + branch + "_base"); if (!base.ok()) return base.status();
    auto norm = weights.Find(prefix + branch + "_norm.weight"); if (!norm.ok()) return norm.status();
    x.mix.fn = *fn; x.mix.scale = *scale; x.mix.base = *base;
    x.input.norm.weight = *norm; x.input.norm.weight_storage = EngramStorage::kBF16;
    auto status = ValidateMhcSublayerInput(x); if (!status.ok()) return status;
    status = weights.ValidateScratch(std::array{x.mix.pre, x.mix.post, x.mix.comb,
        x.input.collapse.output, x.input.norm.output, x.mix.error_flag});
    if (!status.ok()) return status;
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("mHC binding allocation failed"); }
}
Result<RouterLaunch> BindUploadedRouter(const BackboneWeightUpload& weights, RouterLaunch x) {
  if (x.layer >= 40 || x.image_mask.address || x.image_mask.bytes || x.image_bias.address || x.image_bias.bytes)
    return Status::InvalidArgument("Uploaded router admits only text backbone without image inputs");
  try {
    const auto prefix = "layers." + std::to_string(x.layer) + ".ffn.gate.";
    auto weight = weights.Find(prefix + "weight"); if (!weight.ok()) return weight.status();
    auto bias = weights.Find(prefix + "bias"); if (!bias.ok()) return bias.status();
    x.weight = *weight; x.bias = *bias;
    auto status = ValidateRouter(x); if (!status.ok()) return status;
    status = weights.ValidateScratch(std::array{x.logits, x.indices, x.route_weights, x.error_flag});
    if (!status.ok()) return status;
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Router binding allocation failed"); }
}
Result<EngramLaunch> BindUploadedEngram(const BackboneWeightUpload& weights, EngramLaunch x) {
  auto status = Rank(weights, x.lookup.world_size, x.lookup.rank); if (!status.ok()) return status;
  if ((x.lookup.layer != 1 && x.lookup.layer != 14) || x.gate.mask.address || x.gate.mask.bytes)
    return Status::InvalidArgument("Uploaded Engram requires configured text layer without image mask");
  try {
    const auto prefix = "layers." + std::to_string(x.lookup.layer) + ".engram.";
    // Stage into a descriptor copy; no partially bound descriptor escapes.
    const std::array names{"embed.weight", "embed.scale", "wkv.weight", "wkv.scale", "q_weight", "k_weight"};
    const std::array destinations{&x.lookup.table, &x.lookup.scales, &x.projection.weight,
        &x.projection.weight_scales, &x.gate.q_weight, &x.gate.k_weight};
    for (std::size_t i = 0; i < names.size(); ++i) {
      auto value = weights.Find(prefix + names[i]); if (!value.ok()) return value.status();
      *destinations[i] = *value;
    }
    x.gate.gate_storage = EngramStorage::kBF16;
    status = ValidateEngramChain(x); if (!status.ok()) return status;
    status = weights.ValidateScratch(std::array{x.lookup.output, x.projection.quantized,
        x.projection.activation_scales, x.projection.output, x.gate.output, x.lookup.error_flag});
    if (!status.ok()) return status;
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Engram binding allocation failed"); }
}
Result<AttentionPrepareLaunch> BindUploadedAttentionPrepare(const BackboneWeightUpload& weights,
    const FlashConfig& config, AttentionPrepareLaunch x) {
  if (config.config_sha256() != weights.catalog().config_sha256() || x.world_size != weights.catalog().world_size())
    return Status::InvalidArgument("Attention weight configuration or TP size differs");
  auto input = BindUploadedMhc(weights, x.layer, UploadedSublayer::kAttention, x.input);
  if (!input.ok()) return input.status();
  x.input = *input;
  try {
    const auto prefix = "layers." + std::to_string(x.layer) + ".attn.";
    const std::array names{"wq_a", "wq_b", "wkv"};
    const std::array projections{&x.query.low_rank, &x.query.expand, &x.window.projection};
    for (std::size_t i = 0; i < names.size(); ++i) {
      auto p = Projection(weights, prefix + names[i]); if (!p.ok()) return p.status();
      projections[i]->weight = p->weight; projections[i]->weight_scales = p->scales;
    }
    auto qnorm = weights.Find(prefix + "q_norm.weight"); if (!qnorm.ok()) return qnorm.status();
    auto knorm = weights.Find(prefix + "kv_norm.weight"); if (!knorm.ok()) return knorm.status();
    x.query.norm.weight = *qnorm; x.query.norm.weight_storage = EngramStorage::kBF16;
    x.window.prepare.norm.weight = *knorm; x.window.prepare.norm.weight_storage = EngramStorage::kBF16;
    auto status = ValidateAttentionPrepare(config, x); if (!status.ok()) return status;
    for (const auto* p : projections) {
      status = weights.ValidateScratch(std::array{p->quantized, p->activation_scales, p->output, p->error_flag});
      if (!status.ok()) return status;
    }
    status = weights.ValidateScratch(std::array{x.query.norm.output, x.query.rope.output,
        x.window.prepare.norm.output, x.window.prepare.rope.output, x.window.prepare.cache.ring});
    if (!status.ok()) return status;
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Attention preparation binding allocation failed"); }
}
Result<AssembledAttentionOutputLaunch> BindUploadedAttentionOutput(const BackboneWeightUpload& weights,
    const FlashConfig& config, std::uint32_t layer, AssembledAttentionOutputLaunch x) {
  if (layer >= 40 || config.config_sha256() != weights.catalog().config_sha256() ||
      x.output.grouped.projection.groups != 8 / weights.catalog().world_size() ||
      x.assembly.ratio != config.attention_sharing()[layer].compression_ratio)
    return Status::InvalidArgument("Attention output weight configuration, layer or TP size differs");
  try {
    const auto prefix = "layers." + std::to_string(layer) + ".attn.";
    auto sink = weights.Find(prefix + "attn_sink"); if (!sink.ok()) return sink.status();
    auto grouped = weights.Find(prefix + "wo_a.weight"); if (!grouped.ok()) return grouped.status();
    auto linear = Projection(weights, prefix + "wo_b"); if (!linear.ok()) return linear.status();
    x.output.grouped.attention.sink = *sink; x.output.grouped.projection.weight = *grouped;
    x.output.linear.weight = linear->weight; x.output.linear.weight_scales = linear->scales;
    auto status = ValidateAssembledAttentionOutput(x); if (!status.ok()) return status;
    status = weights.ValidateScratch(std::array{x.assembly.kv, x.assembly.indices,
        x.output.grouped.attention.output, x.output.grouped.inverse_rope.output,
        x.output.grouped.projection.output, x.output.linear.quantized, x.output.linear.activation_scales,
        x.output.linear.output, x.output.reduction, x.assembly.error_flag});
    if (!status.ok()) return status;
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Attention output binding allocation failed"); }
}
Result<CompressedPrepareLaunch> BindUploadedCompressed(const BackboneWeightUpload& weights,
    const FlashConfig& config, CompressedPrepareLaunch x) {
  if (x.layer >= 40 || config.config_sha256() != weights.catalog().config_sha256() ||
      !config.attention_sharing()[x.layer].owns_kv)
    return Status::InvalidArgument("Uploaded compressor requires matching configuration and KV owner");
  try {
    const auto prefix = "layers." + std::to_string(x.layer) + ".attn.compressor.";
    auto value = weights.Find(prefix + "wkv.weight"); if (!value.ok()) return value.status();
    auto norm = weights.Find(prefix + "norm.weight"); if (!norm.ok()) return norm.status();
    auto& c = x.compressor;
    c.projection.value_weight = *value;
    if (config.attention_sharing()[x.layer].compression_ratio == 2) {
      auto gate = weights.Find(prefix + "wgate.weight"); if (!gate.ok()) return gate.status();
      c.projection.gate_weight = *gate;
    } else if (c.projection.gate_weight.address || c.projection.gate_weight.bytes) {
      return Status::InvalidArgument("Ratio-one compressor cannot have a gate weight");
    }
    if (c.direct_norm) { c.direct_norm->weight = *norm; c.direct_norm->weight_storage = EngramStorage::kBF16; }
    if (c.pooled && c.pooled->norm) {
      c.pooled->norm->weight = *norm; c.pooled->norm->weight_storage = EngramStorage::kBF16;
    }
    auto status = ValidateCompressedPrepare(config, x); if (!status.ok()) return status;
    status = weights.ValidateScratch(std::array{c.projection.values, c.projection.scores, c.projection.error_flag});
    if (!status.ok()) return status;
    if (c.direct_norm) {
      status = weights.ValidateScratch(std::array{c.direct_norm->output}); if (!status.ok()) return status;
    }
    if (c.pooled) {
      const auto& p = c.pooled->pool;
      status = weights.ValidateScratch(std::array{p.state_values, p.state_scores, p.output,
          c.pooled->norm ? c.pooled->norm->output : EngramDeviceRegion{}});
      if (!status.ok()) return status;
    }
    if (x.cache) {
      status = weights.ValidateScratch(std::array{x.cache->rope.output, x.cache->cache.kv, x.cache->cache.cache});
      if (!status.ok()) return status;
    }
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Compressor binding allocation failed"); }
}
Result<IndexerPipelineLaunch> BindUploadedIndexer(const BackboneWeightUpload& weights,
    const FlashConfig& config, std::uint32_t layer, IndexerPipelineLaunch x) {
  if (layer >= 40 || config.config_sha256() != weights.catalog().config_sha256() ||
      !config.attention_sharing()[layer].owns_index ||
      x.query.quantize.heads != 32 / weights.catalog().world_size())
    return Status::InvalidArgument("Uploaded indexer configuration, owner or head partition differs");
  try {
    const auto prefix = "layers." + std::to_string(layer) + ".attn.indexer.";
    auto query = Projection(weights, prefix + "wq_b"); if (!query.ok()) return query.status();
    auto score = weights.Find(prefix + "weights_proj.weight"); if (!score.ok()) return score.status();
    x.query.projection.weight = query->weight; x.query.projection.weight_scales = query->scales;
    x.scoring.weights.weight = *score;
    if (x.key) {
      if (!config.attention_sharing()[layer].owns_kv)
        return Status::InvalidArgument("Shared index consumer cannot bind key projection weights");
      auto key = weights.Find(prefix + "wk.weight"); if (!key.ok()) return key.status();
      auto norm = weights.Find(prefix + "k_norm.weight"); if (!norm.ok()) return norm.status();
      x.key->projection.weight = *key; x.key->norm.weight = *norm; x.key->norm.weight_storage = EngramStorage::kBF16;
    }
    auto status = ValidateIndexerLayer(config, layer, x); if (!status.ok()) return status;
    status = weights.ValidateScratch(std::array{x.query.projection.quantized, x.query.projection.activation_scales,
        x.query.projection.output, x.query.rope.output, x.query.quantize.values, x.scoring.weights.output,
        x.scoring.score.output, x.selection.output, x.selection.error_flag,
        x.candidates ? x.candidates->output : EngramDeviceRegion{}});
    if (!status.ok()) return status;
    if (x.key) {
      status = weights.ValidateScratch(std::array{x.key->projection.output, x.key->norm.output,
          x.key->rope.output, x.key->cache.quantize.values, x.key->cache.cache});
      if (!status.ok()) return status;
    }
    return x;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Indexer binding allocation failed"); }
}
}  // namespace pih::deepseek_v41
