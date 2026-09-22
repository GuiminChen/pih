#pragma once
#include "token_ledger.h"

namespace pih::deepseek_v41 {
// Serialized output custody shared by CLI and service adapters. No process,
// descriptor, signal or callback ownership. Queue and ledger outlive this object.
// A writer returns the consumed prefix length; zero means bounded backpressure.
// Writer callbacks must not reenter this object or mutate its borrowed queue.
// Finish checks delivery only; the caller must separately prove worker retirement.
// On failure the caller cancels/retires execution, then explicitly discards output.
class SupervisorOutput final {
 public:
  using Writer = Result<std::size_t> (*)(void*, std::string_view);
  explicit SupervisorOutput(TokenOutputQueue& queue) noexcept : queue_(&queue) {}
  SupervisorOutput(const SupervisorOutput&) = delete;
  SupervisorOutput& operator=(const SupervisorOutput&) = delete;
  Status Adopt(TokenOutputLease lease);
  Result<bool> Write(void* context, Writer writer);
  Status Discard();
  Status Finish(const TokenLedger& ledger) const;
  bool pending() const noexcept { return lease_.has_value(); }
  bool failed() const noexcept { return failed_; }
  std::uint64_t visible_bytes() const noexcept { return visible_bytes_; }
 private:
  Status Release(bool delivered);
  TokenOutputQueue* queue_;
  std::optional<TokenOutputLease> lease_;
  TokenPublication publication_{};
  std::size_t sent_ = 0;
  std::uint64_t released_ = 0, delivered_ = 0, visible_bytes_ = 0;
  bool failed_ = false;
};
}  // namespace pih::deepseek_v41
