#include "pih/model/deepseek_expert_transfer_lane.h"

#include <utility>

#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

Result<DeepSeekExpertTransferLane> DeepSeekExpertTransferLane::Create(
    DeepSeekExpertHostSource& source, DeepSeekH2dRuntime& runtime,
    std::vector<std::uintptr_t> slot_bases,
    const CudaRuntimeResourceIdentity& runtime_identity,
    CudaRuntimeResourceDriver& resource_driver,
    std::uint32_t transfer_reservation_window,
    const pih_memory_host_spill_api_v1* host_spill_api) {
  if (transfer_reservation_window == 0 ||
      slot_bases.size() < transfer_reservation_window ||
      runtime_identity.context == 0 ||
      runtime_identity.deepseek_paging_stream == 0 ||
      (host_spill_api != nullptr &&
       (host_spill_api->struct_size != sizeof(*host_spill_api) ||
        host_spill_api->contract_version !=
            PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        host_spill_api->context == nullptr ||
        host_spill_api->create_completion_event == nullptr ||
        host_spill_api->destroy_completion_event == nullptr))) {
    return Status::InvalidArgument(
        "DeepSeek transfer lane resources are invalid");
  }
  std::vector<DriverEventHandle> events;
  events.reserve(slot_bases.size());
  DeepSeekExpertTransferLane owner(
      std::move(events), nullptr, resource_driver, host_spill_api);
  std::vector<DeepSeekExpertDeviceSlot> slots;
  slots.reserve(slot_bases.size());
  for (const auto base : slot_bases) {
    if (base == 0) {
      return Status::InvalidArgument("DeepSeek transfer slot base is null");
    }
    auto event = [&]() -> Result<DriverEventHandle> {
      if (host_spill_api == nullptr) {
        return resource_driver.create_disable_timing_event(
            runtime_identity.context);
      }
      DriverEventHandle value = 0;
      auto status = plugin_status(host_spill_api->create_completion_event(
          host_spill_api->context, runtime_identity.context, &value));
      if (!status.ok()) return status;
      return value;
    }();
    if (!event.ok()) return event.status();
    if (*event == 0) {
      return Status::FailedPrecondition(
          "DeepSeek transfer completion event is null");
    }
    owner.events_.push_back(*event);
    slots.push_back({base, *event});
  }
  auto transfer_value = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, std::move(slots),
      runtime_identity.deepseek_paging_stream,
      transfer_reservation_window);
  if (!transfer_value.ok()) return transfer_value.status();
  owner.transfer_ = std::make_unique<DeepSeekExpertTransferStateDriver>(
      std::move(*transfer_value));
  return owner;
}

DeepSeekExpertTransferLane::~DeepSeekExpertTransferLane() { reset(); }

DeepSeekExpertTransferLane::DeepSeekExpertTransferLane(
    DeepSeekExpertTransferLane&& other) noexcept
    : events_(std::move(other.events_)),
      transfer_(std::move(other.transfer_)),
      resource_driver_(std::exchange(other.resource_driver_, nullptr)),
      host_spill_api_(std::exchange(other.host_spill_api_, nullptr)) {
  other.events_.clear();
}

DeepSeekExpertTransferLane& DeepSeekExpertTransferLane::operator=(
    DeepSeekExpertTransferLane&& other) noexcept {
  if (this != &other) {
    reset();
    events_ = std::move(other.events_);
    transfer_ = std::move(other.transfer_);
    resource_driver_ = std::exchange(other.resource_driver_, nullptr);
    host_spill_api_ = std::exchange(other.host_spill_api_, nullptr);
    other.events_.clear();
  }
  return *this;
}

void DeepSeekExpertTransferLane::reset() noexcept {
  transfer_.reset();
  if (host_spill_api_ != nullptr) {
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
      (void)host_spill_api_->destroy_completion_event(
          host_spill_api_->context, *it);
    }
  } else if (resource_driver_ != nullptr) {
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
      resource_driver_->destroy_event(*it);
    }
  }
  events_.clear();
  resource_driver_ = nullptr;
  host_spill_api_ = nullptr;
}

}  // namespace pih
