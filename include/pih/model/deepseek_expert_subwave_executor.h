#pragma once

#include <cstdint>
#include <deque>
#include <optional>

#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_expert_pager.h"
#include "pih/model/deepseek_expert_subwave_plan.h"

namespace pih {

enum class DeepSeekExpertAsyncStatus : std::uint8_t {
  kInProgress,
  kSuccess,
  kError,
};

class DeepSeekExpertTransferDriver {
 public:
  virtual ~DeepSeekExpertTransferDriver() = default;
  virtual Status start(DeepSeekExpertIdentity identity, std::uint32_t slot,
                       std::uint64_t generation,
                       std::uint64_t payload_bytes) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity identity, std::uint64_t generation) = 0;
};

class DeepSeekExpertKernelDriver {
 public:
  virtual ~DeepSeekExpertKernelDriver() = default;
  virtual Status launch(const DeepSeekExpertLease& lease,
                        const DeepSeekExpertRoute* routes,
                        std::uint32_t route_count) = 0;
  virtual Status launch_resident(
      DeepSeekExpertIdentity identity, std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView& bundle,
      const DeepSeekExpertRoute* routes, std::uint32_t route_count) {
    return Status::FailedPrecondition(
        "DeepSeek expert kernel lane has no resident bundle support");
  }
  virtual Result<DeepSeekExpertAsyncStatus> poll() = 0;
};

enum class DeepSeekExpertSubwaveExecutorState : std::uint8_t {
  kReady,
  kPaging,
  kCompute,
  kComplete,
  kPoisoned,
};

class DeepSeekExpertSubwaveExecutor final {
 public:
  static Result<DeepSeekExpertSubwaveExecutor> Create(
      std::uint16_t layer, const DeepSeekExpertSubwavePlan& plan,
      DeepSeekExpertPager& pager);
  Status advance(DeepSeekExpertTransferDriver& transfer,
                 DeepSeekExpertKernelDriver& kernel);
  [[nodiscard]] DeepSeekExpertSubwaveExecutorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t completed_experts() const noexcept {
    return completed_experts_;
  }

 private:
  struct Pending final {
    DeepSeekExpertIdentity identity;
    std::uint32_t slot = 0;
    std::uint64_t generation = 0;
    bool resident = false;
  };
  DeepSeekExpertSubwaveExecutor(std::uint16_t layer,
                                const DeepSeekExpertSubwavePlan& plan,
                                DeepSeekExpertPager& pager)
      : layer_(layer), plan_(&plan), pager_(&pager) {}
  Status fill_window(DeepSeekExpertTransferDriver& transfer);
  Status poll_transfers(DeepSeekExpertTransferDriver& transfer);
  Status launch_front(DeepSeekExpertKernelDriver& kernel);
  Status finish_compute(DeepSeekExpertKernelDriver& kernel);
  Status poison(Status status);
  std::optional<std::uint16_t> next_expert();

  std::uint16_t layer_ = 0;
  const DeepSeekExpertSubwavePlan* plan_ = nullptr;
  DeepSeekExpertPager* pager_ = nullptr;
  std::uint16_t scan_expert_ = 0;
  std::deque<Pending> pending_;
  std::deque<DeepSeekExpertLease> evictable_;
  std::optional<DeepSeekExpertLease> active_;
  std::uint32_t completed_experts_ = 0;
  DeepSeekExpertSubwaveExecutorState state_ =
      DeepSeekExpertSubwaveExecutorState::kReady;
};

}  // namespace pih
