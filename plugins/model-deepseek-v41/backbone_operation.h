#pragma once
#include "block_operation.h"
#include <memory>

namespace pih::deepseek_v41 {
// Every reference/allocation is borrowed through completion or fault retirement.
// Two FFN arenas carry alternating residual/pre outputs. Index-owner outputs
// remain live for downstream sharing layers, so each owner has its own arena.
struct BackboneResources final {
  const StepPhasesPlan& phases;
  const EngramPlan& engram;
  const SequenceCachePlan& cache;
  const FfnPlan& ffn;
  const AttentionPreparePlan& attention;
  const AttentionOutputPlan& output;
  const CompressorPlan& compressor;
  const IndexerPlan& indexer;
  const BackboneWeightUpload& weights;
  ExpertWorkspaceOwner& workspace;
  EngramDeviceRegion phase_arena, engram_arena, cache_arena;
  std::array<EngramDeviceRegion, 2> ffn_arenas;
  EngramDeviceRegion attention_arena, output_arena, compressor_arena;
  // Layer order 2,8,14,20,24,28,32,36.
  std::array<EngramDeviceRegion, 8> indexer_arenas;
  EngramDeviceRegion host_counts;
  std::uintptr_t communicator = 0;
  EngramCompletionResources completion;
};
enum class BackboneState { kRunning, kComplete, kFailed };
// One complete ordinary-text backbone step after embedding, before head.
// Exclusive, single-threaded use of the sequence and borrowed resources.
class BackboneOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  BackboneOperation(const BackboneOperation&) = delete;
  BackboneOperation& operator=(const BackboneOperation&) = delete;
  ~BackboneOperation();
  static Result<std::unique_ptr<BackboneOperation>> Start(const FlashConfig& config,
      BlockSequence& sequence, const BackboneResources& resources, Clock::time_point deadline);
  Result<BackboneState> Advance();
  Result<SequenceStepOutput> Output() const;
  BackboneState state() const noexcept { return state_; }
 private:
  BackboneOperation(const FlashConfig& config, BlockSequence& sequence,
      const BackboneResources& resources, Clock::time_point deadline);
  Status StartLayer();
  Result<BackboneState> AdvanceImpl();
  FlashConfig config_;
  BlockSequence& sequence_;
  BackboneResources resources_;
  Clock::time_point deadline_;
  std::optional<BlockOperation> block_;
  std::uint32_t layer_ = 0, start_ = 0, tokens_ = 0;
  BackboneState state_ = BackboneState::kFailed;
};
}  // namespace pih::deepseek_v41
