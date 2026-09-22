#pragma once
#include "pih/core/result.h"
#include "pih/contracts/nvidia_cuda_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include <array>
#include <cstdint>

namespace pih::deepseek_v41 {
enum class WorkerHandlesState { kEmpty, kReady, kRetiring, kRetired, kReleased, kQuarantined };
struct WorkerHandleLedger final {
  std::uintptr_t context = 0, stream = 0;
  // computation/upload, inference retirement, prefill/decode expert retirement.
  std::array<std::uintptr_t, 4> events{};
  std::array<bool, 4> events_released{};
  bool stream_released = false, context_released = false;
};
// Public-provider-only resource ownership. Destructor never frees handles.
class WorkerHandles final {
 public:
  WorkerHandles(const pih_nvidia_cuda_api_v1& device, const pih_nvidia_cuda_resources_api_v1& resources,
      const pih_nvidia_cuda_async_api_v1& async, std::int32_t ordinal)
      : device_(device), resources_(resources), async_(async), ordinal_(ordinal) {}
  WorkerHandles(const WorkerHandles&) = delete;
  WorkerHandles& operator=(const WorkerHandles&) = delete;
  Status Create(std::uint32_t expected_sm_major, std::uint32_t expected_sm_minor);
  // Only after all submitters stop and NCCL has no pending enqueue. Memory and
  // communicator owners must release their resources before Release().
  Status BeginRetirement();
  Result<bool> PollRetirement();
  Status Release();
  WorkerHandlesState state() const noexcept { return state_; }
  std::int32_t device_ordinal() const noexcept { return ordinal_; }
  const WorkerHandleLedger& ledger() const noexcept { return ledger_; }
 private:
  const pih_nvidia_cuda_api_v1& device_;
  const pih_nvidia_cuda_resources_api_v1& resources_;
  const pih_nvidia_cuda_async_api_v1& async_;
  std::int32_t ordinal_;
  WorkerHandleLedger ledger_{};
  WorkerHandlesState state_ = WorkerHandlesState::kEmpty;
};
}  // namespace pih::deepseek_v41
