#pragma once
#include "pih/core/result.h"
#include "pih/contracts/transport_collective_v1.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

namespace pih::deepseek_v41 {
Status BindCollectiveTransport(const pih_transport_collective_api_v1& api);
void UnbindCollectiveTransport() noexcept;
const pih_transport_collective_api_v1* CollectiveTransportApi() noexcept;

enum class WorkerCommunicatorState {
  kEmpty, kInitializing, kReady, kFinalizing, kFinalized,
  kReleased, kAborted, kQuarantined
};
// Exclusive worker-thread owner. No implicit destroy, abort, synchronize or
// retry in the destructor. Bootstrap distribution/authentication is external.
class WorkerCommunicator final {
 public:
  using Clock = std::chrono::steady_clock;
  using BootstrapId = std::array<std::byte, 128>;
  WorkerCommunicator() = default;
  WorkerCommunicator(const WorkerCommunicator&) = delete;
  WorkerCommunicator& operator=(const WorkerCommunicator&) = delete;
  Status Start(std::span<const std::byte> id, std::uint32_t world,
      std::uint32_t rank, std::int32_t device, Clock::time_point deadline);
  Result<bool> Poll();
  // All collective submitters must have stopped; no pending NCCL enqueue.
  Status BeginFinalize(Clock::time_point deadline);
  Status Release();
  // Fault path only, after submitters stop. One attempt, even on failure.
  // Abort is a host NCCL call, not a bounded polling operation; the supervisor
  // must retain its independent process timeout/kill policy.
  Status Abort();
  Result<std::uintptr_t> Borrow() const;
  WorkerCommunicatorState state() const noexcept { return state_; }
  std::uintptr_t retained_handle() const noexcept {
    return reinterpret_cast<std::uintptr_t>(handle_);
  }
 private:
  Status Fail(Status status);
  pih_transport_communicator_v1* handle_ = nullptr;
  std::uint32_t world_ = 0, rank_ = 0;
  std::int32_t device_ = -1;
  Clock::time_point deadline_{};
  bool abort_attempted_ = false, destroy_attempted_ = false;
  WorkerCommunicatorState state_ = WorkerCommunicatorState::kEmpty;
};
}  // namespace pih::deepseek_v41
