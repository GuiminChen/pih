#include "block_operation.h"

namespace pih::deepseek_v41 {
BlockOperation::BlockOperation(BlockOperation&& other) noexcept
    : config_(other.config_), phases_(other.phases_), source_launch_(other.source_launch_),
      block_launch_(other.block_launch_), weights_(std::move(other.weights_)),
      workspace_(other.workspace_), sequence_(other.sequence_), host_counts_(other.host_counts_), communicator_(other.communicator_),
      reservation_(other.reservation_), resources_(other.resources_), deadline_(other.deadline_),
      sources_(std::move(other.sources_)), engram_(std::move(other.engram_)), block_(std::move(other.block_)), state_(other.state_) {
  other.workspace_ = nullptr; other.sequence_ = nullptr; other.state_ = BlockOperationState::kFailed;
}
BlockOperation::~BlockOperation() {
  if (sequence_ && state_ != BlockOperationState::kComplete) sequence_->Fail();
}
Result<BlockOperation> BlockOperation::Start(const FlashConfig& config, BlockSequence& sequence,
    const StepPhasesPlan& phase_plan, EngramDeviceRegion phase_arena,
    const EngramPlan& engram_plan, EngramDeviceRegion engram_arena,
    const BackboneWeightUpload& uploaded, ExpertWorkspaceOwner& workspace,
    const SequenceCachePlan& cache_plan, EngramDeviceRegion cache_arena,
    const FfnPlan& ffn_plan, EngramDeviceRegion ffn_arena,
    const AttentionPreparePlan& attention_plan, EngramDeviceRegion attention_arena,
    const AttentionOutputPlan& output_plan, EngramDeviceRegion output_arena,
    const CompressorPlan& compressor_plan, EngramDeviceRegion compressor_arena,
    const IndexerPlan& indexer_plan, EngramDeviceRegion indexer_arena,
    EngramDeviceRegion host_counts, std::uintptr_t communicator,
    const EngramCompletionResources& resources, Clock::time_point deadline) {
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Block deadline expired before preparation");
  if (sequence.failed_ || sequence.active_ || (!sequence.layer_ && !sequence.input_ready_))
    return Status::FailedPrecondition("Block requires an idle sequence with completed step input");
  auto generated_phases = phase_plan.Bind(config, phase_arena, sequence.layer_, sequence.start_,
      sequence.layer_ ? sequence.tokens_ : sequence.input_tokens_, sequence.error_, sequence.stream_);
  if (!generated_phases.ok()) return generated_phases.status();
  const auto phases = *generated_phases;
  const auto phase_safe = uploaded.ValidateScratch(std::array{phase_arena}); if (!phase_safe.ok()) return phase_safe;
  IndexedSourcesLaunch sources;
  auto& initial = sources.sources.attention;
  initial.layer = phases.query.table.layer; initial.world_size = ffn_plan.world_size();
  initial.input.mix.tokens = phases.query.table.tokens;
  initial.input.mix.stream = phases.query.table.stream; initial.input.mix.error_flag = phases.query.table.error_flag;
  initial.window.prepare.cache.start = phases.start;
  PreparedBlockLaunch block_template;
  block_template.ffn.dispatch.world_size = ffn_plan.world_size();
  block_template.ffn.dispatch.rank = ffn_plan.rank();
  if (cache_plan.config_sha256() != config.config_sha256() ||
      phases.start >= cache_plan.maximum_positions() || sources.sources.attention.input.mix.tokens > cache_plan.maximum_positions() - phases.start ||
      (sequence.cache_arena_.bytes && (sequence.cache_arena_.address != cache_arena.address ||
          sequence.cache_arena_.bytes != cache_arena.bytes || sequence.cache_capacity_ != cache_plan.maximum_positions())))
    return Status::InvalidArgument("Block cache plan config, capacity or allocation changed");
  auto cache = cache_plan.Bind(cache_arena, sources.sources.attention.layer); if (!cache.ok()) return cache.status();
  auto& assembly = block_template.attention.attention.assembly;
  assembly.start = phases.start; assembly.tokens = sources.sources.attention.input.mix.tokens;
  assembly.stream = sources.sources.attention.input.mix.stream; assembly.error_flag = sources.sources.attention.input.mix.error_flag;
  assembly.ratio = config.attention_sharing()[sources.sources.attention.layer].compression_ratio;
  const auto cache_safe = uploaded.ValidateScratch(std::array{cache_arena}); if (!cache_safe.ok()) return cache_safe;
  sources.sources.attention.window.prepare.cache.ring = cache->window;
  // Generate owner/cache topology before sequence admission. The complete
  // transient descriptor is built once normalized attention input is bound.
  sources.sources.compressed.reset();
  const auto& role = config.attention_sharing()[sources.sources.attention.layer];
  if (role.owns_kv && role.compression_ratio) {
    const auto step = CompressorOutputRows(phases.start, assembly.tokens); if (!step.ok()) return step.status();
    CompressedPrepareLaunch c;
    c.layer = sources.sources.attention.layer; c.start = phases.start;
    if (role.compression_ratio == 2) {
      c.compressor.pooled.emplace();
      c.compressor.pooled->pool.state_values = cache->pool_values;
      c.compressor.pooled->pool.state_scores = cache->pool_scores;
    }
    if (role.compression_ratio == 1 || *step) {
      c.cache.emplace(); c.cache->cache.cache = cache->compressed; c.cache->cache.capacity = cache->compressed_capacity;
    }
    sources.sources.compressed = c;
  }
  const auto positions = role.compression_ratio ? (phases.start + assembly.tokens) / role.compression_ratio : 0;
  if (role.owns_index && positions) {
    sources.indexer.emplace();
    if (role.owns_kv && positions > phases.start / role.compression_ratio) {
      sources.indexer->key.emplace();
      sources.indexer->key->cache.cache = cache->index_keys;
      sources.indexer->key->cache.capacity = cache->compressed_capacity;
    }
  }
  auto arena = uploaded.Arena(); if (!arena.ok()) return arena.status();
  auto embedding = uploaded.Find("embed.weight"); if (!embedding.ok()) return embedding.status();
  if (config.config_sha256() != uploaded.catalog().config_sha256() ||
      sources.sources.attention.world_size != uploaded.catalog().world_size() ||
      block_template.ffn.dispatch.rank != uploaded.catalog().rank() ||
      sequence.embedding_weight_.address != embedding->address || sequence.embedding_weight_.bytes != embedding->bytes ||
      (sequence.weight_arena_.bytes && (sequence.weight_arena_.address != arena->address || sequence.weight_arena_.bytes != arena->bytes)))
    return Status::InvalidArgument("Block weights differ from sequence embedding, config or rank");
  const auto sequencing = sequence.Prepare(config, phases, sources, block_template, communicator); if (!sequencing.ok()) return sequencing;
  std::optional<EngramLaunch> engram;
  if (config.attention_sharing()[sources.sources.attention.layer].has_engram) {
    const auto layer = sources.sources.attention.layer;
    auto& input = sources.sources.attention.input;
    auto generated = engram_plan.Bind(uploaded, config, engram_arena, layer, input.mix.tokens,
        sequence.engram_ids_[layer == 1 ? 0 : 1], input.mix.residual, input.mix.error_flag, input.mix.stream);
    if (!generated.ok()) return generated.status();
    engram = *generated;
    input.mix.residual = engram->gate.output;
    input.input.collapse.residual = engram->gate.output;
  }
  auto phased = WireSourcePhases(config, phases, std::move(sources)); if (!phased.ok()) return phased.status();
  sources = std::move(*phased);
  const auto layer = sources.sources.attention.layer;
  const auto& declared = sources.sources.attention;
  auto attention = attention_plan.Bind(uploaded, config, attention_arena, layer, phases.start,
      declared.input.mix.tokens, declared.input.mix.residual, declared.input.input.collapse.pre,
      phases.query.table.output, cache->window, declared.input.mix.error_flag, declared.input.mix.stream);
  if (!attention.ok()) return attention.status(); sources.sources.attention = *attention;
  if (sources.sources.compressed) {
    auto compressed = compressor_plan.Bind(uploaded, config, compressor_arena, layer, phases.start,
        attention->input.mix.tokens, attention->input.input.norm.output,
        phases.compressed ? phases.compressed->table.output : EngramDeviceRegion{},
        *cache, attention->input.mix.error_flag, attention->input.mix.stream);
    if (!compressed.ok()) return compressed.status(); sources.sources.compressed = *compressed;
  }
  if (sources.indexer) {
    EngramDeviceRegion latent, compressed_phases;
    if (sources.indexer->key) {
      if (!sources.sources.compressed)
        return Status::InvalidArgument("Index-key producer requires a local compressor");
      const auto& c = sources.sources.compressed->compressor;
      const auto* norm = c.direct_norm ? &*c.direct_norm : c.pooled && c.pooled->norm ? &*c.pooled->norm : nullptr;
      if (!norm) return Status::InvalidArgument("Index-key producer requires emitted normalized compressor rows");
      latent = norm->output;
      if (phases.compressed) compressed_phases = phases.compressed->table.output;
    }
    auto indexer = indexer_plan.Bind(uploaded, config, indexer_arena, layer, phases.start, attention->input.mix.tokens,
        attention->input.input.norm.output, attention->query.norm.output, phases.query.table.output,
        latent, compressed_phases, *cache, sources.indexer->scoring.score.key, sources.indexer->selection.candidates,
        attention->input.mix.error_flag, attention->input.mix.stream);
    if (!indexer.ok()) return indexer.status(); sources.indexer = *indexer;
  }
  const auto phase_valid = ValidateSourcePhases(config, phases, sources); if (!phase_valid.ok()) return phase_valid;
  auto wired = WirePreparedBlockSources(sources, std::move(block_template)); if (!wired.ok()) return wired.status();
  block_template = std::move(*wired);
  auto output = output_plan.Bind(uploaded, config, output_arena, sources.sources.attention,
      block_template.attention.attention.assembly.compressed, block_template.attention.attention.assembly.selected);
  if (!output.ok()) return output.status(); block_template.attention = *output;
  const auto& source_mix = sources.sources.attention.input.mix;
  const auto& workspace_record = workspace.allocation_record();
  auto accumulator = ExpertWorkspaceAccumulator({workspace_record.address, workspace_record.bytes}, source_mix.tokens);
  if (!accumulator.ok()) return accumulator.status();
  auto ffn = ffn_plan.Bind(uploaded, config, ffn_arena, layer, source_mix.tokens,
      block_template.attention.residual.output, source_mix.pre, *accumulator, source_mix.error_flag, source_mix.stream);
  if (!ffn.ok()) return ffn.status();
  block_template.ffn = ffn->route; block_template.tail = ffn->tail;
  auto weights = UploadedRoutedExperts(uploaded, config, block_template.ffn.dispatch); if (!weights.ok()) return weights.status();
  auto bound = BindPreparedBlockSources(config, sources, std::move(block_template)); if (!bound.ok()) return bound.status();
  const auto& d = bound->ffn.dispatch;
  if (d.world_size == 1) return Status::InvalidArgument("Block operation requires 2, 4 or 8 ranks");
  const auto binding = workspace.ValidateBinding(d.tokens, d.stream, resources.event); if (!binding.ok()) return binding;
  const auto& allocation = workspace.allocation_record();
  auto retained = sequence.Retained(); retained.push_back(*arena);
  const auto liveness = ValidateBlockCrossStageBuffers(config, sources, *bound, *weights, {allocation.address, allocation.bytes}, &phases,
      engram ? &*engram : nullptr, retained);
  if (!liveness.ok()) return liveness;
  const auto completion = ValidateEngramCompletionResources(resources); if (!completion.ok()) return completion;
  const auto transport = ValidateAttentionReduction(bound->attention.attention.output, d.rank, communicator);
  if (!transport.ok()) return transport;
  if (engram) {
    const auto admitted = ValidateEngramReduction(engram->lookup, communicator); if (!admitted.ok()) return admitted;
  }
  BlockOperation operation;
  operation.config_ = config; operation.block_launch_ = *bound;
  operation.phases_ = phases; operation.source_launch_ = sources;
  operation.weights_ = std::move(*weights); operation.workspace_ = &workspace;
  operation.host_counts_ = host_counts; operation.communicator_ = communicator;
  operation.resources_ = resources; operation.deadline_ = deadline;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Block deadline expired during preflight");
  const auto reserved = workspace.Reserve(); if (!reserved.ok()) return reserved.status();
  operation.reservation_ = *reserved;
  const auto claimed = sequence.Reserve(sources, *bound, communicator); if (!claimed.ok()) return claimed;
  operation.sequence_ = &sequence;
  sequence.weight_arena_ = *arena;
  sequence.cache_arena_ = cache_arena; sequence.cache_capacity_ = cache_plan.maximum_positions();
  if (engram) {
    auto started = EngramTensorParallel::Start(*engram, communicator, resources, deadline);
    if (!started.ok()) return started.status();
    operation.engram_.emplace(std::move(*started)); operation.state_ = BlockOperationState::kWaitingEngram;
  } else {
    const auto started = operation.StartSources(); if (!started.ok()) return started;
  }
  return operation;
}
Status BlockOperation::StartSources() {
  const auto generated = LaunchStepPhases(config_, phases_); if (!generated.ok()) return generated;
  auto prepared = IndexedSourcesOperation::Start(config_, source_launch_, block_launch_.ffn.dispatch.rank,
      communicator_, resources_, deadline_);
  if (!prepared.ok()) return prepared.status();
  sources_.emplace(std::move(*prepared)); state_ = BlockOperationState::kWaitingSources;
  return Status::Ok();
}
Result<BlockOperationState> BlockOperation::Advance() {
  auto result = AdvanceImpl();
  if (!result.ok() && sequence_) sequence_->Fail();
  return result;
}
Result<BlockOperationState> BlockOperation::AdvanceImpl() {
  if (state_ == BlockOperationState::kFailed) return Status::FailedPrecondition("Block operation failed or moved from");
  if (state_ == BlockOperationState::kComplete) return state_;
  const auto previous = state_; state_ = BlockOperationState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Block operation deadline expired");
  if (previous == BlockOperationState::kWaitingEngram) {
    const auto ready = engram_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != EngramPipelineState::kComplete) { state_ = previous; return state_; }
    const auto started = StartSources(); if (!started.ok()) return started;
    return state_;
  }
  if (previous == BlockOperationState::kWaitingSources) {
    const auto ready = sources_->Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) { state_ = previous; return state_; }
    auto block = PreparedBlockOperation::Start(config_, block_launch_, weights_, *workspace_, host_counts_,
        communicator_, resources_, deadline_, reservation_);
    if (!block.ok()) return block.status();
    block_.emplace(std::move(*block)); state_ = BlockOperationState::kWaitingBlock; return state_;
  }
  const auto ready = block_->Advance(); if (!ready.ok()) return ready.status();
  state_ = *ready == PreparedBlockState::kComplete ? BlockOperationState::kComplete : BlockOperationState::kWaitingBlock;
  if (state_ == BlockOperationState::kComplete) sequence_->Complete();
  return state_;
}
}  // namespace pih::deepseek_v41
