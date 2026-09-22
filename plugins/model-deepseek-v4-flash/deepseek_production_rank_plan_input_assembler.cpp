#include "pih/model/deepseek_production_rank_plan_input_assembler.h"

#include "pih/model/deepseek_engine_resources.h"
#include "pih/model/deepseek_projected_attention_shape_assembler.h"
#include "pih/model/deepseek_rank_attention_resources.h"

#include <algorithm>
#include <limits>

namespace pih {

Result<std::unique_ptr<DeepSeekProductionRankPlanInputAssembler>>
DeepSeekProductionRankPlanInputAssembler::Create(
    DeepSeekEngineResources& resources, std::uint32_t rank,
    DeepSeekPipelinePlanDescriptor descriptor,
    DeepSeekProductionRankPlanSeed seed) {
  bool positions_contiguous = !seed.positions.empty();
  for (std::size_t index = 0; positions_contiguous &&
                              index < seed.positions.size(); ++index) {
    const auto first = seed.positions.front();
    positions_contiguous =
        index <= std::numeric_limits<std::uint32_t>::max() - first &&
        seed.positions[index] == first + static_cast<std::uint32_t>(index);
  }
  if (rank >= resources.world_size() || descriptor.engine_epoch == 0 ||
      descriptor.plan_sequence == 0 || descriptor.phase == DeepSeekPlanPhase::kDrain ||
      descriptor.token_count == 0 || descriptor.sequence_count != 1 ||
      seed.token_ids.size() != descriptor.token_count ||
      seed.positions.size() != descriptor.token_count ||
      !positions_contiguous ||
      seed.table_position_count == 0 ||
      seed.table_position_count <= seed.positions.back()) {
    return Status::InvalidArgument(
        "DeepSeek production rank plan seed is invalid");
  }
  if (resources.world_size() != 1 || rank != 0) {
    return Status::FailedPrecondition(
        "DeepSeek PP1 input assembler requires the single local rank");
  }
  if (descriptor.token_count != 1) {
    return Status::FailedPrecondition(
        "DeepSeek PP1 recent-KV ring requires sequential single-token plans");
  }
  return std::unique_ptr<DeepSeekProductionRankPlanInputAssembler>(
      new DeepSeekProductionRankPlanInputAssembler(
          resources, rank, descriptor, std::move(seed)
          ));
}


Result<DeepSeekRankPlanCompilerInput>
DeepSeekProductionRankPlanInputAssembler::assemble(
    std::uintptr_t incoming_activation_bf16) {
  if (assembled_ || resources_ == nullptr ||
      (rank_ == 0) != (incoming_activation_bf16 == 0)) {
    return Status::InvalidArgument(
        "DeepSeek production rank assembly state is invalid");
  }
  auto* bundle = resources_->rank(rank_).compute_bundle();
  auto* state_pool = resources_->rank(rank_).attention_state_pool();
  if (bundle == nullptr || state_pool == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek production rank compute resources are incomplete");
  }
  auto token_ids_u32 = resources_->rank_request_token_ids_u32(rank_);
  if (!token_ids_u32.ok()) return token_ids_u32.status();
  Result<std::uintptr_t> initial = rank_ == 0
      ? resources_->rank_initial_residual_bf16(rank_)
      : Result<std::uintptr_t>(incoming_activation_bf16);
  if (!initial.ok()) return initial.status();
  auto dense = resources_->assemble_rank_dense_mhc_stage_submissions(
      rank_, descriptor_.token_count, *initial,
      seed_.table_position_count);
  if (!dense.ok()) return dense.status();

  const bool needs_attention_resources =
      !seed_.attention_shape_seeds.empty();
  auto* attention_resources = needs_attention_resources
      ? resources_->rank(rank_).attention_resources()
      : nullptr;
  if (needs_attention_resources && attention_resources == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek production attention resources are incomplete");
  }
  for (auto& shape : seed_.attention_shape_seeds) {
    if (shape.ratio != 0 && shape.ratio != 4 && shape.ratio != 128) {
      return Status::InvalidArgument(
          "DeepSeek projected attention shape ratio is invalid");
    }
    const auto layer = std::find_if(
        dense->layers.begin(), dense->layers.end(),
        [&shape](const auto& value) { return value.layer == shape.layer; });
    if (layer == dense->layers.end()) {
      return Status::InvalidArgument(
          "DeepSeek projected attention shape layer is not owned");
    }
    if (shape.stream == 0) {
      shape.stream = layer->mhc_attention.stream;
    }
    if (shape.recent.empty()) {
      constexpr std::uintptr_t kLatentRowBytes =
          512U * sizeof(std::uint16_t);
      if (layer->sparse_kv_bf16 == 0 ||
          shape.positions.size() >
              (std::numeric_limits<std::uintptr_t>::max() -
               layer->sparse_kv_bf16) /
                  kLatentRowBytes) {
        return Status::InvalidArgument(
            "DeepSeek recent-state source range is invalid");
      }
      shape.recent.reserve(shape.positions.size());
      for (std::size_t token = 0; token < shape.positions.size(); ++token) {
        shape.recent.push_back(
            {shape.layer,
             layer->sparse_kv_bf16 + token * kLatentRowBytes,
             shape.stream, 1, shape.positions[token]});
      }
    }
    auto attention_weights = DeepSeekAttentionWeightBindings::Resolve(
        shape.layer, resources_->rank(rank_).resident_weights());
    if (!attention_weights.ok()) return attention_weights.status();
    auto sparse_status =
        DeepSeekProjectedAttentionShapeAssembler::BindSparseRuntime(
            shape, layer->sparse_query_bf16, layer->sparse_kv_bf16,
            layer->sparse_output_bf16,
            attention_weights->attention_sink_f32,
            attention_resources->device_scratch());
    if (!sparse_status.ok()) return sparse_status;
    if (shape.ratio == 0) continue;
    if (shape.ratio == 4) {
      shape.index_kv_bf16 = state_pool->page_arena().ratio4_index_base();
    }
    if (!shape.updates.empty()) continue;
    const auto kind = shape.ratio == 4
        ? DeepSeekCompressedAttentionKind::kRatio4
        : DeepSeekCompressedAttentionKind::kRatio128;
    auto weights = DeepSeekCompressorWeightBindings::Resolve(
        shape.layer, kind, resources_->rank(rank_).resident_weights());
    if (!weights.ok()) return weights.status();
    auto status = DeepSeekProjectedAttentionShapeAssembler::Populate(
        shape, layer->mhc_attention.layer_input_bf16,
        layer->attention_input.q_norm.output_bf16,
        layer->attention_input.q_rope.positions_u32, *weights,
        attention_resources->compressor_projection(),
        attention_resources->device_scratch(),
        attention_resources->device_scratch().compressor_error_u32(),
        layer->mhc_attention.stream,
        layer->attention_input.q_rope.frequencies_f32,
        seed_.table_position_count);
    if (!status.ok()) return status;
  }

  std::vector<DeepSeekLearnedRouterLayerInput> router_inputs;
  std::vector<DeepSeekLearnedRouterLayerInput> hash_router_inputs;
  for (const auto& layer : dense->layers) {
    if (layer.layer < 3) {
      hash_router_inputs.push_back(
          {layer.layer, layer.mhc_feed_forward.layer_input_bf16});
    } else {
      router_inputs.push_back(
          {layer.layer, layer.mhc_feed_forward.layer_input_bf16});
    }
  }
  auto hash = resources_->assemble_rank_hash_router_plan_input(
      rank_, descriptor_.token_count, hash_router_inputs);
  if (!hash.ok()) return hash.status();
  auto learned = resources_->assemble_rank_learned_router_plan_input(
      rank_, descriptor_.token_count, router_inputs);
  if (!learned.ok()) return learned.status();

  const auto final_residual_bf16 = dense->final_residual_bf16;
  const auto stage = bundle->stage();
  std::optional<DeepSeekEndpointStageSequenceWork> endpoint;
  if (stage.owns_embedding || stage.owns_lm_head) {
    auto assembled = resources_->assemble_rank_endpoint_plan_input(
        rank_, descriptor_.token_count, *token_ids_u32,
        final_residual_bf16, seed_.sampling);
    if (!assembled.ok()) return assembled.status();
    endpoint = std::move(*assembled);
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<DeepSeekDsparkStageWork> dspark;
  std::vector<DeepSeekBoundDsparkMtpStageWork> dspark_mtp;
  if (stage.owns_dspark &&
      (descriptor_.phase == DeepSeekPlanPhase::kPrefill ||
       descriptor_.phase == DeepSeekPlanPhase::kDecode)) {
    auto assembled = resources_->assemble_rank_dspark_plan_input(
        rank_, descriptor_.phase, descriptor_.token_count, *token_ids_u32);
    if (!assembled.ok()) return assembled.status();
    dspark = std::move(*assembled);
    if (descriptor_.phase == DeepSeekPlanPhase::kPrefill) {
      auto mtp = resources_->assemble_rank_dspark_prefill_mtp_plan_input(
          rank_, descriptor_.token_count, seed_.table_position_count);
      if (!mtp.ok()) return mtp.status();
      dspark_mtp = std::move(*mtp);
    } else {
      if (seed_.positions.size() != 1) {
        return Status::InvalidArgument(
            "DeepSeek DSpark decode requires one current position");
      }
      auto mtp = resources_->assemble_rank_dspark_decode_mtp_plan_input(
          rank_, seed_.positions.front(), seed_.table_position_count);
      if (!mtp.ok()) return mtp.status();
      dspark_mtp = std::move(*mtp);
    }
  }
#else
  if (stage.owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this model bundle");
  }
#endif

  // This is the first operation that submits device work. Keep it after all
  // deterministic descriptor, weight and dataflow validation so PREPARE
  // failures cannot leave request-local H2D writes in flight.
  auto lease = resources_->stage_rank_request_inputs(
      rank_, seed_.token_ids, seed_.positions);
  if (!lease.ok()) return lease.status();

  DeepSeekProductionRankResolvedInput resolved;
  resolved.stage = stage;
  resolved.attention_state_pool = state_pool;
  resolved.request_input_lease = std::move(*lease);
  resolved.hash_router = std::move(*hash);
  resolved.learned_router = std::move(*learned);
  resolved.dense_mhc = std::move(*dense);
  resolved.endpoint = std::move(endpoint);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  resolved.dspark = std::move(dspark);
  resolved.dspark_mtp = std::move(dspark_mtp);
#endif
  auto input = PublishResolved(descriptor_, std::move(seed_),
                               std::move(resolved));
  if (!input.ok()) return input.status();
  assembled_ = true;
  return input;
}

Status DeepSeekProductionRankPlanInputAssembler::
AssembleSchedulerAttentionShapes(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    DeepSeekProductionRankPlanSeed& seed,
    DeepSeekProductionRankResolvedInput& resolved) {
  if (seed.attention_shape_seeds.empty()) return Status::Ok();
  if (!seed.decode_attention.empty() || !seed.chunk_attention.empty() ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek scheduler and native attention inputs are ambiguous");
  }
  auto owned =
      std::make_shared<std::vector<DeepSeekOwnedAttentionLayerPlanInput>>();
  owned->reserve(seed.attention_shape_seeds.size());
  for (auto& shape : seed.attention_shape_seeds) {
    auto assembled = descriptor.phase == DeepSeekPlanPhase::kDecode
        ? DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
              std::move(shape))
        : DeepSeekOwnedAttentionLayerPlanInput::AssembleChunk(
              std::move(shape));
    if (!assembled.ok()) return assembled.status();
    owned->push_back(std::move(*assembled));
  }
  for (const auto& layer : *owned) {
    if (descriptor.phase == DeepSeekPlanPhase::kDecode) {
      seed.decode_attention.push_back(layer.decode());
    } else {
      seed.chunk_attention.push_back(layer.chunk());
    }
  }
  seed.attention_shape_seeds.clear();
  resolved.additional_lifetime_backings.push_back(std::move(owned));
  return Status::Ok();
}

Result<DeepSeekRankPlanCompilerInput>
DeepSeekProductionRankPlanInputAssembler::PublishResolved(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    DeepSeekProductionRankPlanSeed seed,
    DeepSeekProductionRankResolvedInput resolved) {
  const bool endpoint_required =
      resolved.stage.owns_embedding || resolved.stage.owns_lm_head;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  const bool dspark_required =
      resolved.stage.owns_dspark &&
      (descriptor.phase == DeepSeekPlanPhase::kPrefill ||
       descriptor.phase == DeepSeekPlanPhase::kDecode);
  const bool dspark_mtp_required =
      resolved.stage.owns_dspark &&
      descriptor.phase == DeepSeekPlanPhase::kPrefill;
#endif
  bool input_incomplete =
      descriptor.token_count == 0 || descriptor.sequence_count != 1 ||
      seed.token_ids.size() != descriptor.token_count ||
      seed.positions.size() != descriptor.token_count ||
      resolved.attention_state_pool == nullptr ||
      resolved.request_input_lease.owner == nullptr ||
      resolved.request_input_lease.token_count != descriptor.token_count ||
      resolved.dense_mhc.layers.empty() ||
      endpoint_required != resolved.endpoint.has_value();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  input_incomplete = input_incomplete ||
      dspark_required != resolved.dspark.has_value() ||
      (dspark_mtp_required ? resolved.dspark_mtp.size() != 3
                           : !resolved.dspark_mtp.empty());
#else
  input_incomplete = input_incomplete || resolved.stage.owns_dspark;
#endif
  if (input_incomplete) {
    return Status::InvalidArgument(
        "DeepSeek resolved production rank input is incomplete");
  }
  auto status = AssembleSchedulerAttentionShapes(descriptor, seed, resolved);
  if (!status.ok()) return status;
  DeepSeekRankPlanCompilerInput input;
  input.token_ids = std::move(seed.token_ids);
  input.projected_hash_router = std::move(resolved.hash_router);
  input.learned_router = std::move(resolved.learned_router);
  input.decode_attention = std::move(seed.decode_attention);
  input.chunk_attention = std::move(seed.chunk_attention);
  input.dense_mhc_layers = std::move(resolved.dense_mhc.layers);
  input.attention_state_pool = resolved.attention_state_pool;
  input.attention_bindings = {seed.attention_binding};
  input.lifetime_backings.push_back(
      std::move(resolved.request_input_lease.owner));
  for (auto& backing : resolved.additional_lifetime_backings) {
    if (backing == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek production additional lifetime backing is null");
    }
    input.lifetime_backings.push_back(std::move(backing));
  }
  if (resolved.endpoint.has_value()) {
    input.endpoint.push_back(std::move(*resolved.endpoint));
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  input.dspark = std::move(resolved.dspark);
  input.dspark_mtp = std::move(resolved.dspark_mtp);
#endif
  return input;
}

}  // namespace pih
