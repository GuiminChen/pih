#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/deepseek_expert_subwave_executor.h"

namespace pih {

enum class DeepSeekTransferEventStatus : std::uint8_t {
  kNotReady,
  kSuccess,
};

struct DeepSeekPinnedExpertExtent final {
  std::uintptr_t address = 0;
  std::uint64_t bytes = 0;
  std::uint64_t registration_identity = 0;
};

class DeepSeekExpertHostSource {
 public:
  virtual ~DeepSeekExpertHostSource() = default;
  virtual Result<DeepSeekPinnedExpertExtent> resolve(
      DeepSeekExpertIdentity identity, std::uint64_t bytes) = 0;
  virtual Status release(DeepSeekExpertIdentity,
                         DeepSeekPinnedExpertExtent) {
    return Status::Ok();
  }
};

class DeepSeekH2dRuntime {
 public:
  virtual ~DeepSeekH2dRuntime() = default;
  virtual Status copy_async(std::uintptr_t destination,
                            std::uintptr_t source, std::uint64_t bytes,
                            std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekTransferEventStatus> query_event(
      std::uintptr_t event) = 0;
};

struct DeepSeekExpertDeviceSlot final {
  std::uintptr_t device = 0;
  std::uintptr_t completion_event = 0;
};

class DeepSeekExpertTransferStateDriver final
    : public DeepSeekExpertTransferDriver {
 public:
  static Result<DeepSeekExpertTransferStateDriver> Create(
      DeepSeekExpertHostSource& source, DeepSeekH2dRuntime& runtime,
      std::vector<DeepSeekExpertDeviceSlot> slots, std::uintptr_t stream,
      std::uint32_t transfer_reservation_window =
          DeepSeekExpertPager::kTransferReservationWindow);

  Status start(DeepSeekExpertIdentity identity, std::uint32_t slot,
               std::uint64_t generation, std::uint64_t payload_bytes) override;
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity identity, std::uint64_t generation) override;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  struct Inflight final {
    DeepSeekExpertIdentity identity;
    std::uint64_t generation = 0;
    DeepSeekPinnedExpertExtent source;
  };
  DeepSeekExpertTransferStateDriver(
      DeepSeekExpertHostSource& source, DeepSeekH2dRuntime& runtime,
      std::vector<DeepSeekExpertDeviceSlot> slots, std::uintptr_t stream,
      std::uint32_t transfer_reservation_window)
      : source_(&source), runtime_(&runtime), slots_(std::move(slots)),
        stream_(stream), inflight_(slots_.size()),
        transfer_reservation_window_(transfer_reservation_window) {}
  Status poison(Status status);

  DeepSeekExpertHostSource* source_ = nullptr;
  DeepSeekH2dRuntime* runtime_ = nullptr;
  std::vector<DeepSeekExpertDeviceSlot> slots_;
  std::uintptr_t stream_ = 0;
  std::vector<std::optional<Inflight>> inflight_;
  std::uint32_t transfer_reservation_window_ = 0;
  bool poisoned_ = false;
};

}  // namespace pih
