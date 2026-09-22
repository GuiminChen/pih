#pragma once

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_attention_state.h"
#include "pih/model/deepseek_dspark_rank_state_cut_publisher.h"

namespace pih {

class DeepSeekAttentionTerminalStateDrainer final
    : public DeepSeekDsparkTerminalStateDrainer {
 public:
  static Result<DeepSeekAttentionTerminalStateDrainer> Create(
      std::uint32_t sequence,
      DeepSeekAttentionStateReservation& state_reservation,
      DeepSeekRatio4PagePool& ratio4_pool,
      DeepSeekRatio128PagePool& ratio128_pool);
  Status drain_committed_sequence_state(
      const DeepSeekDsparkStateCutDecision& decision) override;
  [[nodiscard]] bool drained() const noexcept { return drained_; }

 private:
  std::uint32_t sequence_ = 0;
  DeepSeekAttentionStateReservation* state_reservation_ = nullptr;
  DeepSeekRatio4PagePool* ratio4_pool_ = nullptr;
  DeepSeekRatio128PagePool* ratio128_pool_ = nullptr;
  bool drained_ = false;
};

}  // namespace pih
#endif
