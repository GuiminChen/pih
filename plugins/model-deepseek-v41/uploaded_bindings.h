#pragma once
#include "weight_upload.h"
#include "token_embedding.h"
#include "model_head.h"
#include "expert_workspace.h"
#include "router.h"
#include "attention_prepare.h"
#include "attention_assemble.h"
#include "compressed_prepare.h"
#include "indexer_chain.h"

namespace pih::deepseek_v41 {
// Return validated copies; never mutate caller descriptors or enqueue work.
// Runtime token/state/scratch connections are supplied by the caller.
Result<TokenEmbeddingLaunch> BindUploadedEmbedding(const BackboneWeightUpload& weights, TokenEmbeddingLaunch launch);
Result<ModelHeadLaunch> BindUploadedHead(const BackboneWeightUpload& weights, ModelHeadLaunch launch);
// Global expert IDs increase contiguously within the admitted rank partition.
// The result has exactly 384/world entries, in dispatch-local expert order.
Result<std::vector<ExpertWeights>> UploadedRoutedExperts(const BackboneWeightUpload& weights,
    const FlashConfig& config, const ExpertDispatchLaunch& dispatch);
Result<SharedExpertLaunch> BindUploadedSharedExpert(const BackboneWeightUpload& weights,
    std::uint32_t layer, SharedExpertLaunch launch);
enum class UploadedSublayer { kAttention, kFfn };
Result<MhcSublayerInputLaunch> BindUploadedMhc(const BackboneWeightUpload& weights,
    std::uint32_t layer, UploadedSublayer sublayer, MhcSublayerInputLaunch launch);
Result<RouterLaunch> BindUploadedRouter(const BackboneWeightUpload& weights, RouterLaunch launch);
Result<EngramLaunch> BindUploadedEngram(const BackboneWeightUpload& weights, EngramLaunch launch);
Result<AttentionPrepareLaunch> BindUploadedAttentionPrepare(const BackboneWeightUpload& weights,
    const FlashConfig& config, AttentionPrepareLaunch launch);
Result<AssembledAttentionOutputLaunch> BindUploadedAttentionOutput(const BackboneWeightUpload& weights,
    const FlashConfig& config, std::uint32_t layer, AssembledAttentionOutputLaunch launch);
Result<CompressedPrepareLaunch> BindUploadedCompressed(const BackboneWeightUpload& weights,
    const FlashConfig& config, CompressedPrepareLaunch launch);
Result<IndexerPipelineLaunch> BindUploadedIndexer(const BackboneWeightUpload& weights,
    const FlashConfig& config, std::uint32_t layer, IndexerPipelineLaunch launch);
}  // namespace pih::deepseek_v41
