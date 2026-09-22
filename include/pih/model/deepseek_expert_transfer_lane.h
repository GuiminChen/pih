#pragma once

#include <memory>

#include "pih/backend/cuda/cuda_runtime_resources.h"
#include "pih/contracts/memory_host_spill_v1.h"
#include "pih/model/deepseek_expert_lane_owner.h"
#include "pih/model/deepseek_expert_transfer_driver.h"

namespace pih {

class DeepSeekExpertTransferLane final : public DeepSeekExpertTransferLaneOwner {
 public:
  static Result<DeepSeekExpertTransferLane> Create(
      DeepSeekExpertHostSource& source, DeepSeekH2dRuntime& runtime,
      std::vector<std::uintptr_t> slot_bases,
      const CudaRuntimeResourceIdentity& runtime_identity,
      CudaRuntimeResourceDriver& resource_driver,
      std::uint32_t transfer_reservation_window =
          DeepSeekExpertPager::kTransferReservationWindow,
      const pih_memory_host_spill_api_v1* host_spill_api = nullptr);

  ~DeepSeekExpertTransferLane();
  DeepSeekExpertTransferLane(const DeepSeekExpertTransferLane&) = delete;
  DeepSeekExpertTransferLane& operator=(const DeepSeekExpertTransferLane&) = delete;
  DeepSeekExpertTransferLane(DeepSeekExpertTransferLane&& other) noexcept;
  DeepSeekExpertTransferLane& operator=(DeepSeekExpertTransferLane&& other) noexcept;

  [[nodiscard]] DeepSeekExpertTransferDriver& transfer() noexcept override {
    return *transfer_;
  }
  [[nodiscard]] std::size_t slot_count() const noexcept { return events_.size(); }

 private:
  DeepSeekExpertTransferLane(
      std::vector<DriverEventHandle> events,
      std::unique_ptr<DeepSeekExpertTransferStateDriver> transfer,
      CudaRuntimeResourceDriver& resource_driver,
      const pih_memory_host_spill_api_v1* host_spill_api)
      : events_(std::move(events)), transfer_(std::move(transfer)),
        resource_driver_(&resource_driver), host_spill_api_(host_spill_api) {}
  void reset() noexcept;

  std::vector<DriverEventHandle> events_;
  std::unique_ptr<DeepSeekExpertTransferStateDriver> transfer_;
  CudaRuntimeResourceDriver* resource_driver_ = nullptr;
  const pih_memory_host_spill_api_v1* host_spill_api_ = nullptr;
};

}  // namespace pih
