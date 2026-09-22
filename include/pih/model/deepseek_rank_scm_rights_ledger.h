#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "pih/model/deepseek_rank_post_exec_resource_collector.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankScmRightsLedgerAbi =
    "pih_deepseek_rank_scm_rights_ledger_v1";

class DeepSeekRankScmRightsInFlightLedger final
    : public DeepSeekRankScmRightsInFlightProbe {
 private:
  struct State;

 public:
  class Lease final {
   public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();

    [[nodiscard]] std::uint64_t descriptor_count() const noexcept {
      return descriptor_count_;
    }
    [[nodiscard]] bool active() const noexcept {
      return state_ != nullptr;
    }

   private:
    friend class DeepSeekRankScmRightsInFlightLedger;
    Lease(std::shared_ptr<State> state,
          std::uint64_t descriptor_count) noexcept;
    void release() noexcept;

    std::shared_ptr<State> state_;
    std::uint64_t descriptor_count_ = 0;
  };

  DeepSeekRankScmRightsInFlightLedger();
  DeepSeekRankScmRightsInFlightLedger(
      const DeepSeekRankScmRightsInFlightLedger&) = delete;
  DeepSeekRankScmRightsInFlightLedger& operator=(
      const DeepSeekRankScmRightsInFlightLedger&) = delete;
  DeepSeekRankScmRightsInFlightLedger(
      DeepSeekRankScmRightsInFlightLedger&&) = delete;
  DeepSeekRankScmRightsInFlightLedger& operator=(
      DeepSeekRankScmRightsInFlightLedger&&) = delete;
  Result<Lease> acquire(std::uint64_t descriptor_count);
  Result<std::uint64_t> sample_inflight_fd_count() override;

 private:
  std::shared_ptr<State> state_;
};

}  // namespace pih
