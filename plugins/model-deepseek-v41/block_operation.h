#pragma once
#include "block_bindings.h"
#include "engram_pipeline.h"
#include "block_sequence.h"
#include "uploaded_bindings.h"
#include "sequence_cache.h"
#include "ffn_plan.h"
#include "attention_prepare_plan.h"
#include "attention_output_plan.h"
#include "engram_plan.h"
#include "compressor_plan.h"
#include "indexer_plan.h"

namespace pih::deepseek_v41 {
enum class BlockOperationState { kWaitingEngram, kWaitingSources, kWaitingBlock, kComplete, kFailed };
// Backbone layer from admitted residual/cache inputs and step phases to FFN residual.
// Hash admission and weight loading remain upstream responsibilities.
// All device resources/provider/sequence lifetimes borrowed.
class BlockOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  BlockOperation(const BlockOperation&) = delete;
  BlockOperation& operator=(const BlockOperation&) = delete;
  BlockOperation(BlockOperation&& other) noexcept;
  ~BlockOperation();
  BlockOperation& operator=(BlockOperation&&) = delete;
  static Result<BlockOperation> Start(const FlashConfig& config, BlockSequence& sequence,
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
      const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<BlockOperationState> Advance();
 private:
  BlockOperation() = default;
  Status StartSources();
  Result<BlockOperationState> AdvanceImpl();
  FlashConfig config_{};
  StepPhasesLaunch phases_{};
  IndexedSourcesLaunch source_launch_{};
  PreparedBlockLaunch block_launch_{};
  std::vector<ExpertWeights> weights_;
  ExpertWorkspaceOwner* workspace_ = nullptr;
  BlockSequence* sequence_ = nullptr;
  EngramDeviceRegion host_counts_{};
  std::uintptr_t communicator_ = 0;
  std::uint64_t reservation_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<IndexedSourcesOperation> sources_;
  std::optional<EngramTensorParallel> engram_;
  std::optional<PreparedBlockOperation> block_;
  BlockOperationState state_ = BlockOperationState::kFailed;
};
}  // namespace pih::deepseek_v41
