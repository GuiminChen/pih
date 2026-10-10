#pragma once
#include "packed_request.h"
#include "rank_collector.h"
#include "request_channel.h"
#include <deque>

namespace pih::deepseek_v41 {
class SupervisorRequest;
struct PackedGenerationInput final {
  TokenSamplingRequest sampling;
  std::span<const std::uint32_t> prompt;
  std::optional<std::uint32_t> pending_input;
  std::uint32_t cache_slot = 0;
  bool first = false, finish = false;
};
// Pure dispatch assembly shared by production and CPU behavioral tests.
Result<PackedRequest> BuildPackedGenerationPlan(std::span<const PackedGenerationInput> inputs,
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens);
struct PackedGenerationOutput final { SupervisorRequest* request; TokenOutputLease lease; };
struct PackedGenerationCompleted final { SupervisorRequest* request; Status status; };
// Serialized CPU execution owner. Requests, channels and process custody are borrowed;
// every request must outlive its completion and all transferred publication leases.
class PackedGenerationSession final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<PackedGenerationSession>> Create(
      std::span<RankRequestChannel* const> requests, std::span<RankReceiptChannel* const> receipts,
      RankProcessWatch& processes, std::uint64_t epoch, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, std::uint32_t maximum_pending,
      std::chrono::milliseconds retirement_timeout = std::chrono::minutes(5), std::uint64_t first_plan = 1);
  PackedGenerationSession(const PackedGenerationSession&) = delete;
  ~PackedGenerationSession();
  Status Admit(SupervisorRequest& request, Clock::time_point deadline);
  bool HasWork() const noexcept;
  Status Poll();
  Result<std::optional<PackedGenerationOutput>> TakeOutput();
  Result<std::optional<PackedGenerationCompleted>> TakeCompleted();
  Status Cancel(std::uint64_t sequence_generation);
  void Fail() noexcept;
  const Status& failure() const noexcept { return failure_; }
 private:
  struct Member;
  PackedGenerationSession() = default;
  Status PollImpl();
  Status PreparePlan();
  Status FinishPlan();
  std::array<RankRequestChannel*, 8> requests_{};
  std::array<RankReceiptChannel*, 8> receipts_{};
  RankProcessWatch* processes_ = nullptr;
  std::uint32_t world_ = 0, maximum_sequences_ = 0, maximum_tokens_ = 0, maximum_pending_ = 0;
  std::uint64_t epoch_ = 0, next_plan_ = 1;
  std::chrono::milliseconds timeout_{};
  Clock::time_point plan_deadline_{};
  std::vector<std::unique_ptr<Member>> members_;
  std::vector<Member*> selected_;
  std::deque<PackedGenerationOutput> outputs_;
  std::deque<PackedGenerationCompleted> completed_;
  enum class State { kIdle, kSending, kCollecting, kFailed };
  State state_ = State::kIdle;
  bool prefer_prefill_ = true;
  std::uint32_t sent_mask_ = 0, terminal_mask_ = 0;
  std::size_t collect_index_ = 0;
  Status failure_ = Status::Ok();
  Status stopped_status_ = Status::Unavailable("Packed generation session failed");
};
}
