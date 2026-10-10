#pragma once
#include "packed_request.h"

namespace pih::deepseek_v41 {
// CPU admission state used by the actual rank worker. A plan is admitted as a
// whole, and only a safely retired GPU plan commits positions and RNG ordinals.
class PackedRequestLedger final {
 public:
  struct Slot final {
    SamplingIdentity identity{};
    SamplingParameters parameters{};
    std::uint64_t ordinal{};
    std::uint32_t position{};
  };
  PackedRequestLedger(std::uint64_t epoch, std::uint32_t slots) : epoch_(epoch), slots_(slots) {}
  const Slot& slot(std::uint32_t index) const { return slots_.at(index); }
  bool idle() const noexcept {
    return !pending_ && std::none_of(slots_.begin(), slots_.end(), [](const Slot& slot) { return slot.identity.sequence_generation != 0; });
  }
  Status Admit(const PackedRequest& plan) {
    if (pending_ || plan.spans().empty() || plan.spans().front().identity.plan_seq <= last_plan_)
      return Status::FailedPrecondition("Packed ledger plan pending or order regressed");
    for (unsigned i = 0; i < plan.spans().size(); ++i) {
      const auto& span = plan.spans()[i]; const auto& p = plan.parameters()[i];
      if (span.cache_slot >= slots_.size() || span.identity.epoch != epoch_)
        return Status::FailedPrecondition("Packed ledger slot or epoch differs from admission");
      const auto& slot = slots_[span.cache_slot];
      if (!slot.identity.sequence_generation) {
        if (span.finish || span.position || p.ordinal)
          return Status::FailedPrecondition("Packed fresh slot requires ordinary prefill and fresh RNG");
      } else if (slot.identity.sequence_generation != span.identity.sequence_generation ||
          slot.identity.sampling_config_id != span.identity.sampling_config_id || slot.ordinal != p.ordinal ||
          slot.position != span.position || p.temperature != slot.parameters.temperature ||
          p.top_p != slot.parameters.top_p || p.top_k != slot.parameters.top_k || p.seed != slot.parameters.seed ||
          p.logprobs != slot.parameters.logprobs || p.top_count != slot.parameters.top_count)
        return Status::FailedPrecondition("Packed ledger identity, position or sampling changed");
      if (!span.finish && p.temperature != 0 && p.ordinal == UINT64_MAX)
        return Status::ResourceExhausted("Packed member sampling ordinal exhausted");
    }
    pending_ = plan.spans().front().identity.plan_seq; return Status::Ok();
  }
  Status CommitRetired(const PackedRequest& plan) {
    if (!pending_ || plan.spans().empty() || plan.spans().front().identity.plan_seq != pending_)
      return Status::FailedPrecondition("Packed ledger retirement does not match admitted plan");
    for (unsigned i = 0; i < plan.spans().size(); ++i) {
      const auto& span = plan.spans()[i]; auto& slot = slots_[span.cache_slot];
      if (span.finish) { slot = {}; continue; }
      slot.identity = span.identity; slot.parameters = plan.parameters()[i];
      slot.ordinal = slot.parameters.ordinal + (slot.parameters.temperature != 0 ? 1 : 0);
      slot.position = span.position + span.token_count;
    }
    last_plan_ = pending_; pending_ = 0; return Status::Ok();
  }
 private:
  std::uint64_t epoch_{}, last_plan_{}, pending_{};
  std::vector<Slot> slots_;
};
}
