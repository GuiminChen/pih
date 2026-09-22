#pragma once

#include <chrono>
#include <functional>
#include <span>
#include <thread>
#include <vector>
#include "pih/model/nvidia_qwen_packed_runtime.h"
#include "pih/scheduler/controller_request_arena.h"

namespace pih::qwen_plugin {
enum class TokenAction { kContinue, kCancel, kStop };
struct Generation final {
  std::vector<std::int64_t> tokens;
  ControllerFinishReason finish = ControllerFinishReason::kNone;
  bool cancelled = false;
  bool deadline_elapsed = false;
  bool callback_failed = false;
  bool stopped_by_callback = false;
};

// One serialized request per plugin. Both HTTP response modes must use this
// packed path: a native engine cannot switch runtime modes after activation.
// Callbacks never own device resources. Cancel/stop request cooperative retirement
// at a completed native step; it is not preemption of an in-flight CUDA kernel.
template<class Engine>
Result<Generation> Generate(Engine& engine, std::span<const std::int64_t> prompt,
    std::uint32_t maximum, const ControllerRequestSampling& sampling,
    std::chrono::steady_clock::time_point deadline,
    const std::function<TokenAction(std::int64_t)>& token_callback = {},
    const std::function<bool()>& cancelled = {}) {
  Generation result;
  constexpr auto cleanup_budget = std::chrono::minutes(5);
  if (deadline > std::chrono::steady_clock::time_point::max() - cleanup_budget)
    return Status::InvalidArgument("Qwen deadline leaves no bounded cleanup interval");
  auto observe_cancellation = [&] {
    if (std::chrono::steady_clock::now() >= deadline) {
      result.cancelled = true;
      result.deadline_elapsed = true;
    }
    if (!result.cancelled && cancelled) {
      try {
        if (cancelled()) result.cancelled = true;
      } catch (...) {
        result.callback_failed = true;
        result.cancelled = true;
      }
    }
  };
  observe_cancellation();
  if (result.cancelled) return result;  // No request has been submitted yet.
  result.tokens.reserve(maximum);
  auto submitted = engine.submit_packed(prompt, maximum, sampling);
  if (!submitted.ok()) return submitted.status();
  const auto generation = *submitted;
  bool cancel_sent = false, draining = false, drain_pending = false;
  auto cleanup_deadline = deadline + cleanup_budget;
  std::uint64_t pending_plan = 0;
  std::uint32_t pending_index = 0, pending_count = 0;
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= cleanup_deadline)
      return Status::DeadlineExceeded("Qwen generation cleanup deadline elapsed");
    observe_cancellation();
    if (!cancel_sent && !draining && (result.cancelled || result.stopped_by_callback)) {
      auto status = engine.cancel_packed(generation);
      if (!status.ok()) return status;
      cancel_sent = true;
      cleanup_deadline = now > std::chrono::steady_clock::time_point::max() - cleanup_budget
          ? std::chrono::steady_clock::time_point::max() : now + cleanup_budget;
    }
    if (drain_pending) {
      auto status = engine.drain_packed(generation);
      if (status.ok()) drain_pending = false;
      else if (status.code() != StatusCode::kResourceExhausted && status.code() != StatusCode::kUnavailable)
        return status;
    }
    auto step = engine.drive_packed();
    if (!step.ok()) return step.status();
    // A native step may complete after the deadline. Do not publish its token
    // as a successful response; drain an already-terminal sequence normally,
    // or submit cancellation on the next iteration if more work remains.
    observe_cancellation();
    bool had_event = false;
    for (;;) {
      auto received = engine.try_take_packed_event();
      if (!received.ok()) return received.status();
      if (!received->has_value()) break;
      had_event = true;
      const auto& event = (**received).event;
      if (event.request_generation != generation)
        return Status::Internal("Qwen event belongs to another generation");
      if (event.plan_sequence != 0) {
        if (pending_plan == 0) {
          pending_plan = event.plan_sequence;
          pending_count = event.plan_event_count;
          pending_index = 0;
        }
        if (event.plan_sequence != pending_plan || event.plan_event_count != pending_count ||
            pending_count == 0 || event.plan_event_index != pending_index)
          return Status::Internal("Qwen output plan event sequence invalid");
        if (++pending_index == pending_count) {
          auto status = engine.acknowledge_packed_output(pending_plan);
          if (!status.ok()) return status;
          pending_plan = 0;
        }
      } else if (pending_plan != 0) {
        return Status::Internal("Qwen output plan was interrupted");
      }
      switch (event.kind) {
        case ControllerOutputEventKind::kAdmitted: break;
        case ControllerOutputEventKind::kTokenCommitted:
          if (event.token_id >= 151936 || event.token_ordinal != result.tokens.size() + 1 ||
              result.tokens.size() >= maximum || draining)
            return Status::Internal("Qwen token frontier invalid");
          result.tokens.push_back(event.token_id);
          // A preceding output callback can consume the remaining deadline or
          // observe a disconnect within the same native output bundle.
          observe_cancellation();
          if (!result.cancelled && !result.stopped_by_callback && token_callback) {
            try {
              const auto action = token_callback(event.token_id);
              if (action == TokenAction::kCancel) result.cancelled = true;
              else if (action == TokenAction::kStop) result.stopped_by_callback = true;
              else if (action != TokenAction::kContinue) {
                result.callback_failed = true;
                result.cancelled = true;
              }
            } catch (...) {
              result.callback_failed = true;
              result.cancelled = true;
            }
          }
          break;
        case ControllerOutputEventKind::kDraining:
          if (draining || pending_plan != 0)
            return Status::Internal("Qwen drain frontier invalid");
          draining = true;
          drain_pending = true;
          result.finish = event.finish_reason;
          break;
        case ControllerOutputEventKind::kCompleted:
          if (!draining || drain_pending || pending_plan || result.tokens.empty() ||
              (result.finish != ControllerFinishReason::kLength && result.finish != ControllerFinishReason::kStop))
            return Status::Internal("Qwen completion frontier invalid");
          return result;
        case ControllerOutputEventKind::kCancelled:
          if (!cancel_sent || !draining || drain_pending || pending_plan)
            return Status::Internal("Qwen cancellation frontier invalid");
          if (!result.stopped_by_callback) result.cancelled = true;
          return result;
        case ControllerOutputEventKind::kFailed:
          return Status::Internal("Qwen packed request failed");
        default: return Status::Internal("Qwen event kind invalid");
      }
    }
    if (pending_plan != 0) return Status::Internal("Qwen output bundle incomplete");
    if (!had_event) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
}  // namespace pih::qwen_plugin
