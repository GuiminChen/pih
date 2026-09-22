#include "pih/model/deepseek_expert_subwave_executor.h"

#include <new>

namespace pih {

Result<DeepSeekExpertSubwaveExecutor> DeepSeekExpertSubwaveExecutor::Create(
    std::uint16_t layer, const DeepSeekExpertSubwavePlan& plan,
    DeepSeekExpertPager& pager) {
  if (layer > 42 || plan.token_count() == 0 || pager.poisoned()) {
    return Status::InvalidArgument("DeepSeek expert executor inputs are invalid");
  }
  return DeepSeekExpertSubwaveExecutor(layer, plan, pager);
}

Status DeepSeekExpertSubwaveExecutor::poison(Status status) {
  state_ = DeepSeekExpertSubwaveExecutorState::kPoisoned;
  pager_->poison_epoch();
  return status.ok() ? Status::Internal("DeepSeek expert executor poisoned")
                     : status;
}

std::optional<std::uint16_t> DeepSeekExpertSubwaveExecutor::next_expert() {
  const auto& offsets = plan_->expert_offsets();
  while (scan_expert_ < DeepSeekExpertSubwavePlan::kExpertCount) {
    const auto expert = scan_expert_++;
    if (offsets[expert] != offsets[expert + 1]) return expert;
  }
  return std::nullopt;
}

Status DeepSeekExpertSubwaveExecutor::fill_window(
    DeepSeekExpertTransferDriver& transfer) {
  while (pending_.size() < pager_->transfer_reservation_window()) {
    const auto expert = next_expert();
    if (!expert.has_value()) break;
    const DeepSeekExpertIdentity identity{layer_, *expert};
    auto demand = pager_->demand(identity);
    if (!demand.ok() && demand.status().code() == StatusCode::kResourceExhausted &&
        !evictable_.empty()) {
      const auto victim = evictable_.front();
      evictable_.pop_front();
      const auto eviction =
          pager_->request_eviction(victim.identity, victim.generation);
      if (!eviction.ok()) return poison(eviction);
      demand = pager_->demand(identity);
    }
    if (!demand.ok()) {
      --scan_expert_;
      if (demand.status().code() == StatusCode::kResourceExhausted) {
        if (!pending_.empty()) break;
        return Status::Unavailable(
            "DeepSeek expert slots are temporarily unavailable");
      }
      return poison(demand.status());
    }
    Pending item{identity, demand->slot, demand->generation,
                 demand->disposition ==
                     DeepSeekExpertDemandDisposition::kResidentHit};
    try {
      pending_.push_back(item);
    } catch (const std::bad_alloc&) {
      --scan_expert_;
      if (demand->disposition == DeepSeekExpertDemandDisposition::kNewMiss) {
        const auto failed =
            pager_->fail_transfer(identity, demand->generation);
        return poison(failed.ok()
                          ? Status::ResourceExhausted(
                                "DeepSeek pending expert allocation failed")
                          : failed);
      }
      return Status::ResourceExhausted(
          "DeepSeek pending expert allocation failed");
    }
    if (demand->disposition == DeepSeekExpertDemandDisposition::kNewMiss) {
      const auto begin = pager_->begin_h2d(
          identity, demand->generation, DeepSeekExpertPager::kBundleBytes);
      if (!begin.ok()) {
        pending_.pop_back();
        return poison(begin);
      }
      const auto start = transfer.start(
          identity, demand->slot, demand->generation,
          DeepSeekExpertPager::kBundleBytes);
      if (!start.ok()) {
        pending_.pop_back();
        const auto failed = pager_->fail_transfer(identity, demand->generation);
        return poison(failed.ok() ? start : failed);
      }
    }
  }
  return Status::Ok();
}

Status DeepSeekExpertSubwaveExecutor::poll_transfers(
    DeepSeekExpertTransferDriver& transfer) {
  for (auto& item : pending_) {
    if (item.resident) continue;
    auto status = transfer.poll(item.identity, item.generation);
    if (!status.ok()) {
      const auto failed = pager_->fail_transfer(item.identity, item.generation);
      return poison(failed.ok() ? status.status() : failed);
    }
    if (*status == DeepSeekExpertAsyncStatus::kError) {
      const auto failed = pager_->fail_transfer(item.identity, item.generation);
      return poison(failed.ok() ? Status::Internal("DeepSeek expert H2D failed")
                                : failed);
    }
    if (*status == DeepSeekExpertAsyncStatus::kSuccess) {
      const auto complete =
          pager_->complete_h2d(item.identity, item.generation);
      if (!complete.ok()) return poison(complete);
      item.resident = true;
    } else if (*status != DeepSeekExpertAsyncStatus::kInProgress) {
      const auto failed = pager_->fail_transfer(item.identity, item.generation);
      return poison(failed.ok()
                        ? Status::Internal(
                              "DeepSeek expert H2D returned invalid state")
                        : failed);
    }
  }
  return Status::Ok();
}

Status DeepSeekExpertSubwaveExecutor::launch_front(
    DeepSeekExpertKernelDriver& kernel) {
  if (pending_.empty() || !pending_.front().resident) {
    return Status::Unavailable("DeepSeek expert front is paging");
  }
  const auto item = pending_.front();
  auto lease = pager_->acquire(item.identity, item.generation);
  if (!lease.ok()) return poison(lease.status());
  const auto& offsets = plan_->expert_offsets();
  const auto begin = offsets[item.identity.expert];
  const auto count = offsets[item.identity.expert + 1] - begin;
  const auto launch = kernel.launch(*lease, plan_->routes().data() + begin, count);
  if (!launch.ok()) {
    const auto released = pager_->release(*lease);
    return poison(released.ok() ? launch : released);
  }
  active_ = *lease;
  state_ = DeepSeekExpertSubwaveExecutorState::kCompute;
  return Status::Ok();
}

Status DeepSeekExpertSubwaveExecutor::finish_compute(
    DeepSeekExpertKernelDriver& kernel) {
  auto status = kernel.poll();
  if (!status.ok()) return poison(status.status());
  if (*status == DeepSeekExpertAsyncStatus::kError) {
    return poison(Status::Internal("DeepSeek expert kernel failed"));
  }
  if (*status == DeepSeekExpertAsyncStatus::kInProgress) {
    return Status::Unavailable("DeepSeek expert kernel remains in progress");
  }
  if (*status != DeepSeekExpertAsyncStatus::kSuccess) {
    return poison(
        Status::Internal("DeepSeek expert kernel returned invalid state"));
  }
  const auto release = pager_->release(*active_);
  if (!release.ok()) return poison(release);
  evictable_.push_back(*active_);
  active_.reset();
  pending_.pop_front();
  ++completed_experts_;
  state_ = DeepSeekExpertSubwaveExecutorState::kPaging;
  return Status::Ok();
}

Status DeepSeekExpertSubwaveExecutor::advance(
    DeepSeekExpertTransferDriver& transfer, DeepSeekExpertKernelDriver& kernel) {
  if (state_ == DeepSeekExpertSubwaveExecutorState::kComplete) return Status::Ok();
  if (state_ == DeepSeekExpertSubwaveExecutorState::kPoisoned ||
      pager_->poisoned()) {
    return Status::Unavailable("DeepSeek expert execution epoch is poisoned");
  }
  if (state_ == DeepSeekExpertSubwaveExecutorState::kCompute) {
    const auto finish = finish_compute(kernel);
    if (!finish.ok()) return finish;
  }
  const auto fill = fill_window(transfer);
  if (!fill.ok()) return fill;
  const auto poll = poll_transfers(transfer);
  if (!poll.ok()) return poll;
  if (pending_.empty()) {
    state_ = DeepSeekExpertSubwaveExecutorState::kComplete;
    return Status::Ok();
  }
  state_ = DeepSeekExpertSubwaveExecutorState::kPaging;
  return launch_front(kernel);
}

}  // namespace pih
