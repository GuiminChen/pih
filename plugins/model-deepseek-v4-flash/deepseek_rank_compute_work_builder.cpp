#include "pih/model/deepseek_rank_compute_work_builder.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "pih/model/deepseek_expert_subwave_plan.h"

namespace pih {
namespace {

struct HashRouterBacking final {
  std::uint32_t layer = 0;
  std::vector<std::uint32_t> token_ids;
  std::vector<float> raw_scores;
  std::uint32_t vocabulary_size = 0;
  std::vector<std::uint16_t> token_to_experts;
  std::shared_ptr<const std::vector<std::uint16_t>> shared_token_to_experts;
  DeepSeekRouteScratchArena* scratch = nullptr;
  DeepSeekHashRouterCoordinator* coordinator = nullptr;
  DeepSeekLearnedRouterSubmission submission;
  std::shared_ptr<DeepSeekLearnedRouterHostStaging> host_staging;
};

struct LearnedRouterBacking final {
  std::uint32_t layer = 0;
  DeepSeekLearnedRouterCoordinator* coordinator = nullptr;
  DeepSeekLearnedRouterSubmission submission;
  std::vector<float> host_scores;
  std::shared_ptr<DeepSeekLearnedRouterHostStaging> host_staging;
};

struct DecodeAttentionBacking final {
  std::uint32_t layer = 0;
  DeepSeekDecodeAttentionWork work;
  std::vector<std::uint32_t> query_positions;
  std::vector<std::uint32_t> visible_slot_counts;
};

struct ChunkAttentionBacking final {
  std::uint32_t layer = 0;
  DeepSeekChunkAttentionWork work;
  std::vector<DeepSeekRecentStateSubmission> recent;
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  std::vector<std::uint32_t> query_positions;
  std::vector<std::uint32_t> visible_slot_counts;
};

template <typename Sequence>
struct LayerSequenceBacking final {
  std::uint32_t layer = 0;
  std::vector<Sequence> sequences;
};

class WorkBacking final {
 public:
  std::vector<std::shared_ptr<const void>> lifetime_backings;
  std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>
      attention_transactions;
  std::vector<HashRouterBacking> hash_backing;
  std::vector<DeepSeekBoundHashRouterWork> hash_views;
  std::vector<LearnedRouterBacking> learned_backing;
  std::vector<DeepSeekBoundLearnedRouterWork> learned_views;
  std::vector<DecodeAttentionBacking> decode_backing;
  std::vector<DeepSeekBoundDecodeAttentionLayerWork> decode_views;
  std::vector<ChunkAttentionBacking> chunk_backing;
  std::vector<DeepSeekBoundChunkAttentionLayerWork> chunk_views;
  std::vector<LayerSequenceBacking<DeepSeekDenseAttentionStageSequenceWork>>
      dense_backing;
  std::vector<DeepSeekBoundDenseAttentionLayerWork> dense_views;
  std::vector<LayerSequenceBacking<DeepSeekMhcStageSequenceWork>>
      mhc_attention_backing;
  std::vector<DeepSeekBoundMhcLayerWork> mhc_attention_views;
  std::vector<LayerSequenceBacking<DeepSeekMhcStageSequenceWork>>
      mhc_feed_forward_backing;
  std::vector<DeepSeekBoundMhcLayerWork> mhc_feed_forward_views;
  std::vector<DeepSeekEndpointStageSequenceWork> endpoint;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<DeepSeekDsparkStageWork> dspark;
  std::vector<DeepSeekBoundDsparkMtpStageWork> dspark_mtp;
#endif
};

bool multiply_fits(std::uint64_t left, std::uint64_t right) {
  return right == 0 ||
         left <= std::numeric_limits<std::size_t>::max() / right;
}

}  // namespace

Status DeepSeekRankComputeWorkBuilder::own_lifetime_backing(
    std::shared_ptr<const void> backing) {
  if (finished_ || backing == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek plan lifetime backing is invalid");
  }
  lifetime_backings_.push_back(std::move(backing));
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_hash_router(
    std::uint32_t layer, std::vector<std::uint32_t> token_ids,
    std::vector<float> raw_scores, std::uint32_t vocabulary_size,
    std::vector<std::uint16_t> token_to_experts,
    DeepSeekRouteScratchArena* scratch) {
  auto shared = std::make_shared<const std::vector<std::uint16_t>>(
      std::move(token_to_experts));
  return add_hash_router_with_shared_table(
      layer, std::move(token_ids), std::move(raw_scores), vocabulary_size,
      std::move(shared), scratch);
}

Status DeepSeekRankComputeWorkBuilder::add_hash_router_with_shared_table(
    std::uint32_t layer, std::vector<std::uint32_t> token_ids,
    std::vector<float> raw_scores, std::uint32_t vocabulary_size,
    std::shared_ptr<const std::vector<std::uint16_t>> token_to_experts,
    DeepSeekRouteScratchArena* scratch) {
  constexpr auto kExperts = DeepSeekExpertSubwavePlan::kExpertCount;
  constexpr auto kRoutes = DeepSeekExpertSubwavePlan::kRoutesPerToken;
  if (finished_ || layer >= 3 || token_ids.empty() ||
      vocabulary_size == 0 || scratch == nullptr ||
      token_to_experts == nullptr ||
      !multiply_fits(token_ids.size(), kExperts) ||
      !multiply_fits(vocabulary_size, kRoutes) ||
      raw_scores.size() != token_ids.size() * kExperts ||
      token_to_experts->size() !=
          static_cast<std::size_t>(vocabulary_size) * kRoutes ||
      std::ranges::any_of(hash_router_, [layer](const auto& work) {
        return work.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned hash router work shape or identity is invalid");
  }
  hash_router_.push_back({
      layer, std::move(token_ids), std::move(raw_scores), vocabulary_size,
      {}, std::move(token_to_experts), scratch, nullptr, {}, nullptr});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::
add_projected_hash_router_with_shared_table(
    std::uint32_t layer, std::vector<std::uint32_t> token_ids,
    DeepSeekHashRouterCoordinator* coordinator,
    DeepSeekLearnedRouterSubmission submission,
    std::uint32_t vocabulary_size,
    std::shared_ptr<const std::vector<std::uint16_t>> token_to_experts,
    std::shared_ptr<DeepSeekLearnedRouterHostStaging> staging) {
  constexpr auto kRoutes = DeepSeekExpertSubwavePlan::kRoutesPerToken;
  if (finished_ || layer >= 3 || token_ids.empty() ||
      coordinator == nullptr || staging == nullptr ||
      coordinator->plan_provider() == nullptr ||
      submission.layer != layer ||
      submission.token_count != token_ids.size() ||
      submission.input_bf16 == 0 || submission.weight_bf16 == 0 ||
      submission.scores_f32 == 0 || submission.error_flag_u32 == 0 ||
      submission.stream == 0 || submission.completion_event == 0 ||
      vocabulary_size == 0 || token_to_experts == nullptr ||
      !multiply_fits(vocabulary_size, kRoutes) ||
      token_to_experts->size() !=
          static_cast<std::size_t>(vocabulary_size) * kRoutes ||
      std::ranges::any_of(hash_router_, [layer](const auto& work) {
        return work.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek projected hash router work shape or identity is invalid");
  }
  submission.host_scores = {};
  submission.host_error_flag = nullptr;
  hash_router_.push_back(
      {layer, std::move(token_ids), {}, vocabulary_size, {},
       std::move(token_to_experts), nullptr, coordinator, submission,
       std::move(staging)});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_learned_router(
    std::uint32_t layer, DeepSeekLearnedRouterCoordinator* coordinator,
    DeepSeekLearnedRouterSubmission submission) {
  if (finished_ || layer < 3 || layer > 42 || coordinator == nullptr ||
      submission.layer != layer || submission.token_count == 0 ||
      submission.host_scores.size() !=
          static_cast<std::size_t>(submission.token_count) *
              DeepSeekLearnedRouterCoordinator::kExpertCount ||
      submission.host_error_flag == nullptr ||
      std::ranges::any_of(learned_router_, [layer](const auto& work) {
        return work.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned learned router work shape or identity is invalid");
  }
  std::vector<float> host_scores(
      submission.host_scores.begin(), submission.host_scores.end());
  submission.host_scores = {};
  learned_router_.push_back({
      layer, coordinator, submission, std::move(host_scores), nullptr});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_learned_router_with_host_staging(
    std::uint32_t layer, DeepSeekLearnedRouterCoordinator* coordinator,
    DeepSeekLearnedRouterSubmission submission,
    std::shared_ptr<DeepSeekLearnedRouterHostStaging> staging) {
  if (staging == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek learned router host staging is missing");
  }
  auto scores = staging->scores(submission.token_count);
  if (!scores.ok()) return scores.status();
  submission.host_scores = *scores;
  submission.host_error_flag = staging->error_flag();
  if (finished_ || layer < 3 || layer > 42 || coordinator == nullptr ||
      submission.layer != layer || submission.token_count == 0 ||
      std::ranges::any_of(learned_router_, [layer](const auto& work) {
        return work.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek pinned learned router work is invalid or duplicated");
  }
  learned_router_.push_back({
      layer, coordinator, submission, {}, std::move(staging)});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_decode_attention(
    std::uint32_t layer, DeepSeekDecodeAttentionWork work) {
  if (finished_ || layer > 42 || work.recent_writer == nullptr ||
      work.update_coordinator == nullptr || work.coordinator == nullptr ||
      work.transaction == nullptr || work.attention.query_positions.empty() ||
      (work.attention.kind == DeepSeekCompressedAttentionKind::kRatio4 &&
       work.attention.compressed_slot_count != 0 &&
       work.attention.selection.visible_slot_counts.empty()) ||
      !chunk_attention_.empty() ||
      std::ranges::any_of(decode_attention_, [layer](const auto& item) {
        return item.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned decode attention work is invalid or duplicated");
  }
  std::vector<std::uint32_t> query_positions(
      work.attention.query_positions.begin(),
      work.attention.query_positions.end());
  std::vector<std::uint32_t> visible_slot_counts(
      work.attention.selection.visible_slot_counts.begin(),
      work.attention.selection.visible_slot_counts.end());
  work.attention.query_positions = {};
  work.attention.selection.visible_slot_counts = {};
  decode_attention_.push_back({
      layer, work, std::move(query_positions),
      std::move(visible_slot_counts)});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_chunk_attention(
    std::uint32_t layer, DeepSeekChunkAttentionWork work) {
  if (finished_ || layer > 42 || work.chunk_coordinator == nullptr ||
      work.attention_coordinator == nullptr || work.transaction == nullptr ||
      work.submission.recent.empty() ||
      (work.submission.attention.kind == DeepSeekCompressedAttentionKind::kRecentOnly
           ? !work.submission.updates.empty() : work.submission.updates.empty()) ||
      work.submission.attention.query_positions.empty() ||
      (work.submission.attention.kind == DeepSeekCompressedAttentionKind::kRatio4 &&
       work.submission.attention.compressed_slot_count != 0 &&
       work.submission.attention.selection.visible_slot_counts.empty()) ||
      !decode_attention_.empty() ||
      std::ranges::any_of(chunk_attention_, [layer](const auto& item) {
        return item.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned chunk attention work is invalid or duplicated");
  }
  std::vector<DeepSeekRecentStateSubmission> recent(
      work.submission.recent.begin(), work.submission.recent.end());
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates(
      work.submission.updates.begin(), work.submission.updates.end());
  std::vector<std::uint32_t> query_positions(
      work.submission.attention.query_positions.begin(),
      work.submission.attention.query_positions.end());
  std::vector<std::uint32_t> visible_slot_counts(
      work.submission.attention.selection.visible_slot_counts.begin(),
      work.submission.attention.selection.visible_slot_counts.end());
  work.submission.recent = {};
  work.submission.updates = {};
  work.submission.attention.query_positions = {};
  work.submission.attention.selection.visible_slot_counts = {};
  chunk_attention_.push_back({
      layer, work, std::move(recent), std::move(updates),
      std::move(query_positions), std::move(visible_slot_counts)});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_dense_attention(
    std::uint32_t layer,
    std::vector<DeepSeekDenseAttentionStageSequenceWork> sequences) {
  if (finished_ || layer > 42 || sequences.empty() ||
      std::ranges::any_of(sequences, [](const auto& item) {
        return item.input_coordinator == nullptr ||
               item.output_coordinator == nullptr ||
               item.transaction == nullptr;
      }) ||
      std::ranges::any_of(dense_attention_, [layer](const auto& item) {
        return item.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned dense attention work is invalid or duplicated");
  }
  dense_attention_.push_back({layer, std::move(sequences)});
  return Status::Ok();
}

namespace {

Status validate_mhc_builder_work(
    std::uint32_t layer, DeepSeekMhcBranchKind kind,
    std::span<const DeepSeekMhcStageSequenceWork> sequences) {
  if (layer > 42 || sequences.empty() ||
      std::ranges::any_of(sequences, [layer, kind](const auto& item) {
        return item.executor == nullptr || item.transaction == nullptr ||
               item.submission.layer_id != layer ||
               item.submission.kind != kind;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned mHC work is invalid");
  }
  return Status::Ok();
}

}  // namespace

Status DeepSeekRankComputeWorkBuilder::add_mhc_attention(
    std::uint32_t layer,
    std::vector<DeepSeekMhcStageSequenceWork> sequences) {
  if (finished_ ||
      std::ranges::any_of(mhc_attention_, [layer](const auto& item) {
        return item.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned mHC attention work is duplicated");
  }
  const auto valid = validate_mhc_builder_work(
      layer, DeepSeekMhcBranchKind::kAttention, sequences);
  if (!valid.ok()) return valid;
  mhc_attention_.push_back({layer, std::move(sequences)});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::add_mhc_feed_forward(
    std::uint32_t layer,
    std::vector<DeepSeekMhcStageSequenceWork> sequences) {
  if (finished_ ||
      std::ranges::any_of(mhc_feed_forward_, [layer](const auto& item) {
        return item.layer == layer;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned mHC feed-forward work is duplicated");
  }
  const auto valid = validate_mhc_builder_work(
      layer, DeepSeekMhcBranchKind::kFeedForward, sequences);
  if (!valid.ok()) return valid;
  mhc_feed_forward_.push_back({layer, std::move(sequences)});
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::set_endpoint(
    std::vector<DeepSeekEndpointStageSequenceWork> sequences) {
  if (finished_ || !endpoint_.empty() || sequences.empty() ||
      std::ranges::any_of(sequences, [](const auto& item) {
        return item.executor == nullptr || item.transaction == nullptr;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned endpoint work is invalid or already set");
  }
  endpoint_ = std::move(sequences);
  return Status::Ok();
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status DeepSeekRankComputeWorkBuilder::set_dspark(
    DeepSeekDsparkStageWork work) {
  if (finished_ || dspark_.has_value() || work.embed_coordinator == nullptr ||
      work.transaction == nullptr ||
      (work.kind == DeepSeekDsparkStageWorkKind::kDecodeProposal &&
       work.head_executor == nullptr) ||
      (work.kind != DeepSeekDsparkStageWorkKind::kDecodeProposal &&
       work.kind !=
           DeepSeekDsparkStageWorkKind::kPrefillStateInitialization)) {
    return Status::InvalidArgument(
        "DeepSeek owned DSpark work is invalid or already set");
  }
  dspark_ = std::move(work);
  return Status::Ok();
}

Status DeepSeekRankComputeWorkBuilder::set_dspark_mtp(
    std::vector<DeepSeekBoundDsparkMtpStageWork> stages) {
  if (finished_ || !dspark_mtp_.empty() || stages.size() != 3) {
    return Status::InvalidArgument(
        "DeepSeek owned DSpark MTP work is invalid or already set");
  }
  dspark_mtp_ = std::move(stages);
  return Status::Ok();
}
#endif


Status DeepSeekRankComputeWorkBuilder::own_attention_transactions(
    std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>
        transactions) {
  if (finished_ || !attention_transactions_.empty() ||
      transactions.empty() ||
      std::ranges::any_of(transactions, [](const auto& transaction) {
        return transaction == nullptr;
      })) {
    return Status::InvalidArgument(
        "DeepSeek owned attention transactions are invalid or already set");
  }
  std::vector<DeepSeekAttentionSequenceTransaction*> identities;
  identities.reserve(transactions.size());
  for (const auto& transaction : transactions) {
    if (std::ranges::find(identities, transaction.get()) != identities.end()) {
      return Status::InvalidArgument(
          "DeepSeek owned attention transaction is duplicated");
    }
    identities.push_back(transaction.get());
  }
  attention_transactions_ = std::move(transactions);
  return Status::Ok();
}

Result<DeepSeekRankComputePlanWork>
DeepSeekRankComputeWorkBuilder::finish() && {
  bool no_publishable_work =
      hash_router_.empty() && learned_router_.empty() &&
      decode_attention_.empty() && chunk_attention_.empty() &&
      dense_attention_.empty() && mhc_attention_.empty() &&
      mhc_feed_forward_.empty() && endpoint_.empty();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  no_publishable_work = no_publishable_work && !dspark_.has_value() &&
                        dspark_mtp_.empty();
#endif
  if (finished_ || no_publishable_work) {
    return Status::FailedPrecondition(
        "DeepSeek compute work builder has no publishable work");
  }
  finished_ = true;
  auto backing = std::make_shared<WorkBacking>();
  backing->attention_transactions = std::move(attention_transactions_);
  backing->hash_backing.reserve(hash_router_.size());
  for (auto& source : hash_router_) {
    backing->hash_backing.push_back({
        source.layer, std::move(source.token_ids),
        std::move(source.raw_scores), source.vocabulary_size,
        std::move(source.token_to_experts),
        std::move(source.shared_token_to_experts), source.scratch,
        source.coordinator, source.submission,
        std::move(source.host_staging)});
  }
  backing->hash_views.reserve(backing->hash_backing.size());
  for (auto& source : backing->hash_backing) {
    const std::span<const std::uint16_t> table =
        source.shared_token_to_experts != nullptr
            ? std::span<const std::uint16_t>(*source.shared_token_to_experts)
            : std::span<const std::uint16_t>(source.token_to_experts);
    if (source.host_staging != nullptr) {
      source.submission.host_scores =
          source.host_staging->scores(source.submission.token_count).value();
      source.submission.host_error_flag = source.host_staging->error_flag();
    }
    backing->hash_views.push_back({
        source.layer, source.token_ids, source.raw_scores,
        source.vocabulary_size, table, source.scratch,
        source.coordinator,
        {source.submission, source.token_ids, source.vocabulary_size, table}});
  }
  backing->learned_backing.reserve(learned_router_.size());
  for (auto& source : learned_router_) {
    backing->learned_backing.push_back({
        source.layer, source.coordinator, source.submission,
        std::move(source.host_scores), std::move(source.host_staging)});
  }
  backing->learned_views.reserve(backing->learned_backing.size());
  for (auto& source : backing->learned_backing) {
    if (source.host_staging != nullptr) {
      source.submission.host_scores =
          source.host_staging->scores(source.submission.token_count).value();
      source.submission.host_error_flag = source.host_staging->error_flag();
    } else {
      source.submission.host_scores = source.host_scores;
    }
    backing->learned_views.push_back({
        source.layer, source.coordinator, source.submission});
  }
  backing->decode_backing.reserve(decode_attention_.size());
  for (auto& source : decode_attention_) {
    backing->decode_backing.push_back({
        source.layer, source.work, std::move(source.query_positions),
        std::move(source.visible_slot_counts)});
  }
  backing->decode_views.reserve(backing->decode_backing.size());
  for (auto& source : backing->decode_backing) {
    source.work.attention.query_positions = source.query_positions;
    source.work.attention.selection.visible_slot_counts =
        source.visible_slot_counts;
    backing->decode_views.push_back({source.layer, source.work});
  }
  backing->chunk_backing.reserve(chunk_attention_.size());
  for (auto& source : chunk_attention_) {
    backing->chunk_backing.push_back({
        source.layer, source.work, std::move(source.recent),
        std::move(source.updates), std::move(source.query_positions),
        std::move(source.visible_slot_counts)});
  }
  backing->chunk_views.reserve(backing->chunk_backing.size());
  for (auto& source : backing->chunk_backing) {
    source.work.submission.recent = source.recent;
    source.work.submission.updates = source.updates;
    source.work.submission.attention.query_positions = source.query_positions;
    source.work.submission.attention.selection.visible_slot_counts =
        source.visible_slot_counts;
    backing->chunk_views.push_back({source.layer, source.work});
  }
  backing->dense_backing.reserve(dense_attention_.size());
  for (auto& source : dense_attention_) {
    backing->dense_backing.push_back(
        {source.layer, std::move(source.sequences)});
  }
  backing->dense_views.reserve(backing->dense_backing.size());
  for (const auto& source : backing->dense_backing) {
    backing->dense_views.push_back({source.layer, source.sequences});
  }
  backing->mhc_attention_backing.reserve(mhc_attention_.size());
  for (auto& source : mhc_attention_) {
    backing->mhc_attention_backing.push_back(
        {source.layer, std::move(source.sequences)});
  }
  backing->mhc_attention_views.reserve(
      backing->mhc_attention_backing.size());
  for (const auto& source : backing->mhc_attention_backing) {
    backing->mhc_attention_views.push_back(
        {source.layer, source.sequences});
  }
  backing->mhc_feed_forward_backing.reserve(mhc_feed_forward_.size());
  for (auto& source : mhc_feed_forward_) {
    backing->mhc_feed_forward_backing.push_back(
        {source.layer, std::move(source.sequences)});
  }
  backing->mhc_feed_forward_views.reserve(
      backing->mhc_feed_forward_backing.size());
  for (const auto& source : backing->mhc_feed_forward_backing) {
    backing->mhc_feed_forward_views.push_back(
        {source.layer, source.sequences});
  }
  backing->endpoint = std::move(endpoint_);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  backing->dspark = std::move(dspark_);
  backing->dspark_mtp = std::move(dspark_mtp_);
#endif
  backing->lifetime_backings = std::move(lifetime_backings_);
  DeepSeekRankComputePlanWork work;
  work.lifetime_owner = std::shared_ptr<const DeepSeekRankComputePlanWorkOwner>(
      new DeepSeekRankComputePlanWorkOwner(backing));
  work.hash_router = backing->hash_views;
  work.learned_router = backing->learned_views;
  work.decode_attention = backing->decode_views;
  work.chunk_attention = backing->chunk_views;
  work.dense_attention = backing->dense_views;
  work.mhc_attention = backing->mhc_attention_views;
  work.mhc_feed_forward = backing->mhc_feed_forward_views;
  work.endpoint = backing->endpoint;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (backing->dspark.has_value()) work.dspark = &*backing->dspark;
  work.dspark_mtp = backing->dspark_mtp;
#endif
  return work;
}

}  // namespace pih
