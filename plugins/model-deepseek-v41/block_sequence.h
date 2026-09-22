#pragma once
#include "block_bindings.h"

namespace pih::deepseek_v41 {
class BlockOperation;
class HeadOperation;
class SamplingOperation;
class EmbeddingOperation;
class EngramHashState;
struct SequenceStepOutput final {
  EngramDeviceRegion residual, pre;
  std::uint32_t start = 0, tokens = 0;
};
// One ordinary autoregressive sequence. Nonmovable, borrowed by the active
// block; keep it alive until that operation is destroyed. Not thread safe.
// Completion is published only by BlockOperation, never by caller assertions.
class BlockSequence final {
 public:
  explicit BlockSequence(const FlashConfig& config) : config_(config) {}
  BlockSequence(const BlockSequence&) = delete;
  BlockSequence& operator=(const BlockSequence&) = delete;
  std::uint32_t next_layer() const noexcept { return layer_; }
  std::uint32_t next_position() const noexcept { return start_; }
  bool failed() const noexcept { return failed_; }
  Result<SequenceStepOutput> StepOutput() const;
 private:
  friend class BlockOperation;
  friend class BackboneOperation;
  friend class InferenceOperation;
  friend class RankWorkerLoop;
  friend class HeadOperation;
  friend class SamplingOperation;
  friend class EmbeddingOperation;
  struct Publication {
    EngramDeviceRegion ring, values, scores, compressed, key, selected, candidates;
  };
  Status Prepare(const FlashConfig& config, const StepPhasesLaunch& phases,
      IndexedSourcesLaunch& sources, PreparedBlockLaunch& block, std::uintptr_t communicator);
  Status Reserve(const IndexedSourcesLaunch& sources, const PreparedBlockLaunch& block, std::uintptr_t communicator);
  std::vector<EngramDeviceRegion> Retained(bool include_current = false, bool include_inputs = true) const;
  void Complete() noexcept;
  void Fail() noexcept { failed_ = true; active_ = false; }
  FlashConfig config_;
  std::array<Publication, FlashConfig::kMainLayers> published_{};
  Publication pending_{};
  EngramDeviceRegion residual_{}, pre_{}, pending_residual_{}, pending_pre_{}, error_{};
  std::uintptr_t stream_ = 0;
  std::uintptr_t communicator_ = 0;
  std::uint32_t layer_ = 0, start_ = 0, tokens_ = 0, pending_tokens_ = 0, world_ = 0, rank_ = 0;
  std::uint32_t completed_tokens_ = 0;
  std::uint32_t head_end_ = 0;
  std::uint32_t sample_end_ = 0;
  EngramDeviceRegion embedding_weight_{}, embedding_ids_{};
  EngramDeviceRegion weight_arena_{};
  EngramDeviceRegion cache_arena_{};
  std::uint32_t cache_capacity_ = 0;
  std::uint32_t input_tokens_ = 0;
  bool input_ready_ = false;
  EngramHashState* hash_state_ = nullptr;
  std::array<EngramDeviceRegion, 2> engram_ids_{};
  bool active_ = false, failed_ = false;
};
}  // namespace pih::deepseek_v41
