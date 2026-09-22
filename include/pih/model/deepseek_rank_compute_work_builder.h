#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_rank_compute_bundle.h"
#include "pih/model/deepseek_learned_router_host_staging.h"

namespace pih {

// Builds plan work and its immutable span backing as one ownership unit.
// Pointer-valued operator dependencies remain rank-runtime resources; callers
// cannot use this builder to manufacture or extend their lifetime.
class DeepSeekRankComputeWorkBuilder final {
 public:
  Status add_hash_router(
      std::uint32_t layer, std::vector<std::uint32_t> token_ids,
      std::vector<float> raw_scores, std::uint32_t vocabulary_size,
      std::vector<std::uint16_t> token_to_experts,
      DeepSeekRouteScratchArena* scratch);
  Status add_hash_router_with_shared_table(
      std::uint32_t layer, std::vector<std::uint32_t> token_ids,
      std::vector<float> raw_scores, std::uint32_t vocabulary_size,
      std::shared_ptr<const std::vector<std::uint16_t>> token_to_experts,
      DeepSeekRouteScratchArena* scratch);
  Status add_projected_hash_router_with_shared_table(
      std::uint32_t layer, std::vector<std::uint32_t> token_ids,
      DeepSeekHashRouterCoordinator* coordinator,
      DeepSeekLearnedRouterSubmission submission,
      std::uint32_t vocabulary_size,
      std::shared_ptr<const std::vector<std::uint16_t>> token_to_experts,
      std::shared_ptr<DeepSeekLearnedRouterHostStaging> staging);
  Status add_learned_router(
      std::uint32_t layer, DeepSeekLearnedRouterCoordinator* coordinator,
      DeepSeekLearnedRouterSubmission submission);
  Status add_learned_router_with_host_staging(
      std::uint32_t layer, DeepSeekLearnedRouterCoordinator* coordinator,
      DeepSeekLearnedRouterSubmission submission,
      std::shared_ptr<DeepSeekLearnedRouterHostStaging> staging);
  Status add_decode_attention(
      std::uint32_t layer, DeepSeekDecodeAttentionWork work);
  Status add_chunk_attention(
      std::uint32_t layer, DeepSeekChunkAttentionWork work);
  Status add_dense_attention(
      std::uint32_t layer,
      std::vector<DeepSeekDenseAttentionStageSequenceWork> sequences);
  Status add_mhc_attention(
      std::uint32_t layer,
      std::vector<DeepSeekMhcStageSequenceWork> sequences);
  Status add_mhc_feed_forward(
      std::uint32_t layer,
      std::vector<DeepSeekMhcStageSequenceWork> sequences);
  Status set_endpoint(
      std::vector<DeepSeekEndpointStageSequenceWork> sequences);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  Status set_dspark(DeepSeekDsparkStageWork work);
  Status set_dspark_mtp(
      std::vector<DeepSeekBoundDsparkMtpStageWork> stages);
#endif
  Status own_attention_transactions(
      std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>
          transactions);
  Status own_lifetime_backing(std::shared_ptr<const void> backing);

  Result<DeepSeekRankComputePlanWork> finish() &&;

 private:
  struct OwnedHashRouterWork final {
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
  struct OwnedLearnedRouterWork final {
    std::uint32_t layer = 0;
    DeepSeekLearnedRouterCoordinator* coordinator = nullptr;
    DeepSeekLearnedRouterSubmission submission;
    std::vector<float> host_scores;
    std::shared_ptr<DeepSeekLearnedRouterHostStaging> host_staging;
  };
  struct OwnedDecodeAttentionWork final {
    std::uint32_t layer = 0;
    DeepSeekDecodeAttentionWork work;
    std::vector<std::uint32_t> query_positions;
    std::vector<std::uint32_t> visible_slot_counts;
  };
  struct OwnedChunkAttentionWork final {
    std::uint32_t layer = 0;
    DeepSeekChunkAttentionWork work;
    std::vector<DeepSeekRecentStateSubmission> recent;
    std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
    std::vector<std::uint32_t> query_positions;
    std::vector<std::uint32_t> visible_slot_counts;
  };
  struct OwnedDenseAttentionWork final {
    std::uint32_t layer = 0;
    std::vector<DeepSeekDenseAttentionStageSequenceWork> sequences;
  };
  struct OwnedMhcWork final {
    std::uint32_t layer = 0;
    std::vector<DeepSeekMhcStageSequenceWork> sequences;
  };

  std::vector<OwnedHashRouterWork> hash_router_;
  std::vector<OwnedLearnedRouterWork> learned_router_;
  std::vector<OwnedDecodeAttentionWork> decode_attention_;
  std::vector<OwnedChunkAttentionWork> chunk_attention_;
  std::vector<OwnedDenseAttentionWork> dense_attention_;
  std::vector<OwnedMhcWork> mhc_attention_;
  std::vector<OwnedMhcWork> mhc_feed_forward_;
  std::vector<DeepSeekEndpointStageSequenceWork> endpoint_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<DeepSeekDsparkStageWork> dspark_;
  std::vector<DeepSeekBoundDsparkMtpStageWork> dspark_mtp_;
#endif
  std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>
      attention_transactions_;
  std::vector<std::shared_ptr<const void>> lifetime_backings_;
  bool finished_ = false;
};

}  // namespace pih
