#include "pih/model/deepseek_rank_scm_rights_ledger.h"

#include <atomic>
#include <limits>
#include <utility>

namespace pih {

struct DeepSeekRankScmRightsInFlightLedger::State final {
  std::atomic<std::uint64_t> descriptor_count{0};
};

DeepSeekRankScmRightsInFlightLedger::Lease::Lease(
    std::shared_ptr<State> state,
    std::uint64_t descriptor_count) noexcept
    : state_(std::move(state)), descriptor_count_(descriptor_count) {}

DeepSeekRankScmRightsInFlightLedger::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)),
      descriptor_count_(std::exchange(other.descriptor_count_, 0)) {}

DeepSeekRankScmRightsInFlightLedger::Lease&
DeepSeekRankScmRightsInFlightLedger::Lease::operator=(
    Lease&& other) noexcept {
  if (this == &other) return *this;
  release();
  state_ = std::move(other.state_);
  descriptor_count_ = std::exchange(other.descriptor_count_, 0);
  return *this;
}

DeepSeekRankScmRightsInFlightLedger::Lease::~Lease() { release(); }

void DeepSeekRankScmRightsInFlightLedger::Lease::release() noexcept {
  if (state_ == nullptr) return;
  state_->descriptor_count.fetch_sub(
      descriptor_count_, std::memory_order_acq_rel);
  state_.reset();
  descriptor_count_ = 0;
}

DeepSeekRankScmRightsInFlightLedger::
    DeepSeekRankScmRightsInFlightLedger()
    : state_(std::make_shared<State>()) {}

Result<DeepSeekRankScmRightsInFlightLedger::Lease>
DeepSeekRankScmRightsInFlightLedger::acquire(
    std::uint64_t descriptor_count) {
  if (descriptor_count == 0) {
    return Status::InvalidArgument(
        "SCM_RIGHTS in-flight descriptor lease is empty");
  }
  auto current = state_->descriptor_count.load(std::memory_order_acquire);
  for (;;) {
    if (current > std::numeric_limits<std::uint64_t>::max() -
                      descriptor_count) {
      return Status::ResourceExhausted(
          "SCM_RIGHTS in-flight descriptor ledger overflowed");
    }
    if (state_->descriptor_count.compare_exchange_weak(
            current, current + descriptor_count,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return Lease(state_, descriptor_count);
    }
  }
}

Result<std::uint64_t>
DeepSeekRankScmRightsInFlightLedger::sample_inflight_fd_count() {
  return state_->descriptor_count.load(std::memory_order_acquire);
}

}  // namespace pih
