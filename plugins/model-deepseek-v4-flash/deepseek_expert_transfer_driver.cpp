#include "pih/model/deepseek_expert_transfer_driver.h"

namespace pih {

Result<DeepSeekExpertTransferStateDriver>
DeepSeekExpertTransferStateDriver::Create(
    DeepSeekExpertHostSource& source, DeepSeekH2dRuntime& runtime,
    std::vector<DeepSeekExpertDeviceSlot> slots, std::uintptr_t stream,
    std::uint32_t transfer_reservation_window) {
  if (transfer_reservation_window == 0 ||
      slots.size() < transfer_reservation_window ||
      stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek H2D driver window exceeds slot capacity");
  }
  for (const auto& slot : slots) {
    if (slot.device == 0 || slot.completion_event == 0) {
      return Status::InvalidArgument(
          "DeepSeek H2D slot requires device storage and completion event");
    }
  }
  return DeepSeekExpertTransferStateDriver(source, runtime, std::move(slots),
                                            stream,
                                            transfer_reservation_window);
}

Status DeepSeekExpertTransferStateDriver::poison(Status status) {
  poisoned_ = true;
  return status.ok() ? Status::Internal("DeepSeek H2D driver poisoned")
                     : status;
}

Status DeepSeekExpertTransferStateDriver::start(
    DeepSeekExpertIdentity identity, std::uint32_t slot,
    std::uint64_t generation, std::uint64_t payload_bytes) {
  if (poisoned_) return Status::Unavailable("DeepSeek H2D driver is poisoned");
  if (slot >= slots_.size() || generation == 0 ||
      payload_bytes != DeepSeekExpertPager::kBundleBytes ||
      inflight_[slot].has_value()) {
    return Status::FailedPrecondition(
        "DeepSeek H2D transfer identity or slot is invalid");
  }
  std::uint32_t inflight_count = 0;
  for (const auto& inflight : inflight_) {
    if (inflight.has_value()) ++inflight_count;
  }
  if (inflight_count >= transfer_reservation_window_) {
    return Status::ResourceExhausted(
        "DeepSeek H2D transfer reservation window is full");
  }
  auto source = source_->resolve(identity, payload_bytes);
  if (!source.ok()) return poison(source.status());
  if (source->address == 0 || source->bytes != payload_bytes ||
      source->registration_identity == 0) {
    auto release = source_->release(identity, *source);
    return poison(release.ok()
                      ? Status::Internal(
                            "DeepSeek host expert extent is not registered")
                      : release);
  }
  auto status = runtime_->copy_async(slots_[slot].device, source->address,
                                     payload_bytes, stream_);
  if (!status.ok()) {
    // A failed asynchronous submission can still leave device work with
    // indeterminate completion. Quarantine the leased extent with the poisoned
    // driver so its bytes cannot be overwritten by a later transfer.
    return poison(status);
  }
  status = runtime_->record_event(slots_[slot].completion_event, stream_);
  if (!status.ok()) {
    // The copy was accepted but has no observable completion frontier. Never
    // return its source extent to the staging pool.
    return poison(status);
  }
  inflight_[slot] = Inflight{identity, generation, *source};
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus> DeepSeekExpertTransferStateDriver::poll(
    DeepSeekExpertIdentity identity, std::uint64_t generation) {
  if (poisoned_) return Status::Unavailable("DeepSeek H2D driver is poisoned");
  for (std::size_t slot = 0; slot < inflight_.size(); ++slot) {
    if (!inflight_[slot].has_value() ||
        inflight_[slot]->identity != identity ||
        inflight_[slot]->generation != generation) {
      continue;
    }
    auto result = runtime_->query_event(slots_[slot].completion_event);
    if (!result.ok()) return poison(result.status());
    if (*result == DeepSeekTransferEventStatus::kNotReady) {
      return DeepSeekExpertAsyncStatus::kInProgress;
    }
    if (*result != DeepSeekTransferEventStatus::kSuccess) {
      return poison(Status::Internal(
          "DeepSeek H2D runtime returned invalid event state"));
    }
    auto release = source_->release(identity, inflight_[slot]->source);
    if (!release.ok()) return poison(release);
    inflight_[slot].reset();
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  return Status::FailedPrecondition("DeepSeek H2D generation is not inflight");
}

}  // namespace pih
