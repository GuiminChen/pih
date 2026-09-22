#include "pih/model/deepseek_rank_plan_compiler.h"

namespace pih {

Result<DeepSeekRankPlanCompiler> DeepSeekRankPlanCompiler::Create(
    DeepSeekStagePlan stage,
    DeepSeekHashRouterWorkFactory hash_router,
    DeepSeekLearnedRouterWorkFactory learned_router,
    DeepSeekAttentionWorkFactory attention,
    DeepSeekDenseMhcWorkFactory dense_mhc,
    DeepSeekEndpointWorkFactory endpoint,
    DeepSeekDenseMhcRuntimeResources* dense_mhc_runtime) {
  if (hash_router.owned_layers() != stage.layers ||
      learned_router.owned_layers() != stage.layers ||
      attention.owned_layers() != stage.layers ||
      dense_mhc.owned_layers() != stage.layers ||
      endpoint.stage().rank != stage.rank ||
      endpoint.stage().layers != stage.layers ||
      endpoint.stage().owns_embedding != stage.owns_embedding ||
      endpoint.stage().owns_lm_head != stage.owns_lm_head ||
      endpoint.stage().owns_dspark != stage.owns_dspark) {
    return Status::InvalidArgument(
        "DeepSeek rank plan compiler factories disagree on stage");
  }
  DeepSeekRankPlanCompiler result;
  result.stage_ = stage;
  result.hash_router_ = std::move(hash_router);
  result.learned_router_ = std::move(learned_router);
  result.attention_ = std::move(attention);
  result.dense_mhc_ = std::move(dense_mhc);
  result.endpoint_ = std::move(endpoint);
  result.dense_mhc_runtime_ = dense_mhc_runtime;
  return result;
}

Result<DeepSeekRankComputePlanWork> DeepSeekRankPlanCompiler::compile(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    DeepSeekRankPlanCompilerInput input) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count != 1 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain ||
      input.token_ids.size() != descriptor.token_count ||
      input.attention_state_pool == nullptr ||
      input.attention_bindings.size() != descriptor.sequence_count) {
    return Status::InvalidArgument(
        "DeepSeek rank plan compiler descriptor is invalid");
  }
#if defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (stage_.owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this model bundle");
  }
#endif
  DeepSeekRankComputeWorkBuilder builder;
  for (auto& backing : input.lifetime_backings) {
    auto status = builder.own_lifetime_backing(std::move(backing));
    if (!status.ok()) return status;
  }
  DeepSeekRankAttentionTransactionFactory transaction_factory(
      *input.attention_state_pool);
  auto transactions = transaction_factory.Create(input.attention_bindings);
  if (!transactions.ok()) return transactions.status();
  auto* plan_transaction = (*transactions)[0].get();
  if (!input.dense_mhc_layers.empty()) {
    if (dense_mhc_runtime_ == nullptr || !input.dense_attention.empty() ||
        !input.mhc_attention.empty() ||
        !input.mhc_feed_forward.empty()) {
      return Status::InvalidArgument(
          "DeepSeek raw dense mHC input is not uniquely assemblable");
    }
    auto dense = DeepSeekDenseMhcPlanInputAssembler::Assemble(
        stage_.layers, input.dense_mhc_layers, *dense_mhc_runtime_,
        *plan_transaction);
    if (!dense.ok()) return dense.status();
    input.dense_attention = std::move(dense->dense_attention);
    input.mhc_attention = std::move(dense->mhc_attention);
    input.mhc_feed_forward = std::move(dense->mhc_feed_forward);
  }
  const auto rebind = [&](DeepSeekAttentionSequenceTransaction*& transaction) {
    transaction = plan_transaction;
    return Status::Ok();
  };
  for (auto& layer : input.decode_attention) {
    auto status = rebind(layer.transaction);
    if (!status.ok()) return status;
  }
  for (auto& layer : input.chunk_attention) {
    auto status = rebind(layer.transaction);
    if (!status.ok()) return status;
  }
  for (auto& layer : input.dense_attention) {
    for (auto& sequence : layer.sequences) {
      auto status = rebind(sequence.transaction);
      if (!status.ok()) return status;
    }
  }
  const auto rebind_mhc = [&](auto& layers) -> Status {
    for (auto& layer : layers) {
      for (auto& sequence : layer.sequences) {
        auto status = rebind(sequence.transaction);
        if (!status.ok()) return status;
      }
    }
    return Status::Ok();
  };
  auto status = rebind_mhc(input.mhc_attention);
  if (!status.ok()) return status;
  status = rebind_mhc(input.mhc_feed_forward);
  if (!status.ok()) return status;
  for (auto& sequence : input.endpoint) {
    status = rebind(sequence.transaction);
    if (!status.ok()) return status;
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (input.dspark.has_value()) {
    status = rebind(input.dspark->transaction);
    if (!status.ok()) return status;
  }
  for (auto& stage : input.dspark_mtp) {
    if (descriptor.phase == DeepSeekPlanPhase::kPrefill) {
      status = rebind(stage.prefill_resources.transaction);
      stage.prefill_resources.state_layout =
          &input.attention_state_pool->fixed_layout();
      stage.prefill_resources.prepare_epoch = 0;
      stage.prefill_resources.recent_state = {
          stage.prefill_resources.stage, {}};
    } else {
      status = rebind(stage.resources.transaction);
      stage.resources.state_layout =
          &input.attention_state_pool->fixed_layout();
      stage.resources.prepare_epoch = 0;
      stage.resources.recent_state = {stage.resources.stage, {}};
    }
    if (!status.ok()) return status;
  }
#endif
  status = builder.own_attention_transactions(std::move(*transactions));
  if (!status.ok()) return status;
  if (!input.hash_router_scores.empty() &&
      !input.projected_hash_router.empty()) {
    return Status::InvalidArgument(
        "DeepSeek hash router inputs are ambiguous");
  }
  status = input.projected_hash_router.empty()
      ? hash_router_.append_plan_work(
            input.token_ids, std::move(input.hash_router_scores), builder)
      : hash_router_.append_projected_plan_work(
            input.token_ids, std::move(input.projected_hash_router), builder);
  if (!status.ok()) return status;
  status = learned_router_.append_plan_work(
      descriptor.token_count, std::move(input.learned_router), builder);
  if (!status.ok()) return status;
  if (descriptor.phase == DeepSeekPlanPhase::kDecode) {
    if (!input.chunk_attention.empty()) {
      return Status::InvalidArgument(
          "DeepSeek decode compiler input contains chunk attention");
    }
    status = attention_.append_decode_plan_work(
        std::move(input.decode_attention), builder);
  } else {
    if (!input.decode_attention.empty()) {
      return Status::InvalidArgument(
          "DeepSeek chunk compiler input contains decode attention");
    }
    status = attention_.append_chunk_plan_work(
        std::move(input.chunk_attention), builder);
  }
  if (!status.ok()) return status;
  status = dense_mhc_.append_plan_work(
      descriptor.token_count, descriptor.sequence_count,
      std::move(input.dense_attention), std::move(input.mhc_attention),
      std::move(input.mhc_feed_forward), builder);
  if (!status.ok()) return status;
  status = endpoint_.append_plan_work(
      descriptor.phase, descriptor.sequence_count,
      std::move(input.endpoint), builder);
  if (!status.ok()) return status;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  const bool dspark_required = stage_.owns_dspark &&
      (descriptor.phase == DeepSeekPlanPhase::kPrefill ||
       descriptor.phase == DeepSeekPlanPhase::kDecode);
  if (input.dspark.has_value() != dspark_required) {
    return Status::InvalidArgument(
        "DeepSeek DSpark work differs from stage phase ownership");
  }
  if (input.dspark.has_value()) {
    const auto expected_kind = descriptor.phase == DeepSeekPlanPhase::kPrefill
        ? DeepSeekDsparkStageWorkKind::kPrefillStateInitialization
        : DeepSeekDsparkStageWorkKind::kDecodeProposal;
    if (input.dspark->kind != expected_kind ||
        input.dspark->embed_coordinator == nullptr ||
        input.dspark->transaction == nullptr ||
        (descriptor.phase == DeepSeekPlanPhase::kDecode &&
         input.dspark->head_executor == nullptr)) {
      return Status::InvalidArgument(
          "DeepSeek DSpark work resources are incomplete");
    }
    status = builder.set_dspark(std::move(*input.dspark));
    if (!status.ok()) return status;
  }
  if (!input.dspark_mtp.empty()) {
    status = builder.set_dspark_mtp(std::move(input.dspark_mtp));
    if (!status.ok()) return status;
  }
#endif
  return std::move(builder).finish();
}

}  // namespace pih
