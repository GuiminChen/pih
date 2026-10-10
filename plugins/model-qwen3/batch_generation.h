#pragma once
#include "generation.h"
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace pih::qwen_plugin {
// Only this owner touches the packed engine. Callers consume private queues and
// invoke their own output callbacks, so a stalled socket cannot stall CUDA.
template<class Engine> class BatchGeneration final {
  struct Request {
    std::vector<std::int64_t> prompt;
    std::uint32_t maximum;
    ControllerRequestSampling sampling;
    std::chrono::steady_clock::time_point deadline;
    std::chrono::steady_clock::time_point cleanup_deadline = std::chrono::steady_clock::time_point::max();
    Generation result;
    std::deque<std::int64_t> output;
    Status status = Status::Ok();
    std::uint64_t generation = 0;
    bool done = false, cancel = false, cancel_sent = false, drain = false, draining = false, admitted = false;
  };
 public:
  BatchGeneration(Engine& engine, std::uint32_t sequences, std::uint32_t pending)
      : engine_(engine), sequences_(validate(sequences, 32)), pending_(validate(pending, 4096)), worker_([this]{ run(); }) {}
  ~BatchGeneration() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    cv_.notify_all();
    worker_.join();
  }
  std::size_t pending_requests() {
    std::lock_guard lock(mutex_); return queue_.size();
  }
  Result<Generation> Generate(std::span<const std::int64_t> prompt,
      std::uint32_t maximum, const ControllerRequestSampling& sampling,
      std::chrono::steady_clock::time_point deadline,
      const std::function<TokenAction(std::int64_t)>& callback,
      const std::function<bool()>& cancelled) {
    if (deadline > std::chrono::steady_clock::time_point::max() - std::chrono::minutes(5))
      return Status::InvalidArgument("Qwen deadline leaves no cleanup interval");
    auto r = std::make_shared<Request>();
    r->prompt.assign(prompt.begin(), prompt.end()); r->maximum = maximum;
    r->sampling = sampling; r->deadline = deadline;
    std::unique_lock lock(mutex_);
    if (stopping_ || !failure_.ok()) return Status::Unavailable("Qwen batch owner stopped");
    if (queue_.size() + active_.size() >= sequences_ + pending_)
      return Status::ResourceExhausted("Qwen pending queue full");
    queue_.push_back(r); cv_.notify_all();
    for (;;) {
      lock.unlock();
      bool cancel = false, callback_failed = false, stop = false;
      try { cancel = cancelled && cancelled(); } catch (...) { cancel = callback_failed = true; }
      lock.lock();
      if (std::chrono::steady_clock::now() >= deadline) { r->result.deadline_elapsed = true; cancel = true; }
      if (cancel) { r->cancel = true; r->result.cancelled = true; }
      r->result.callback_failed |= callback_failed;
      if (!r->output.empty()) {
        const auto token = r->output.front(); r->output.pop_front();
        const bool publish = !r->cancel;
        lock.unlock();
        if (publish && callback) {
          try {
            const auto action = callback(token);
            stop = action == TokenAction::kStop;
            cancel = action != TokenAction::kContinue;
            callback_failed = action != TokenAction::kContinue && !stop && action != TokenAction::kCancel;
          } catch (...) { cancel = callback_failed = true; }
        }
        lock.lock();
        if (cancel) r->cancel = true;
        r->result.stopped_by_callback |= stop;
        r->result.callback_failed |= callback_failed;
        if (cancel && !stop) r->result.cancelled = true;
        cv_.notify_all();
        continue;
      }
      if (r->done) {
        if (!r->status.ok()) return r->status;
        return std::move(r->result);
      }
      cv_.wait_for(lock, std::chrono::milliseconds(1));
    }
  }
 private:
  static std::uint32_t validate(std::uint32_t value, std::uint32_t maximum) {
    if (!value || value > maximum) throw std::invalid_argument("Qwen batch owner capacity invalid");
    return value;
  }
  void fail(Status status) {
    failure_ = status;
    for (auto& [_, r] : active_) { r->status = status; r->done = true; }
    for (auto& r : queue_) { r->status = status; r->done = true; }
    active_.clear(); queue_.clear(); cv_.notify_all();
  }
  void run() noexcept {
    std::unique_lock lock(mutex_);
    try {
      if constexpr (requires { engine_.bind_packed_thread(); }) {
        auto status = engine_.bind_packed_thread();
        if (!status.ok()) { fail(status); return; }
      }
      for (;;) {
        while (!queue_.empty() && active_.size() < sequences_ && !stopping_) {
          auto r = queue_.front(); queue_.pop_front();
          if (r->cancel || std::chrono::steady_clock::now() >= r->deadline) {
            r->done = true; r->result.cancelled = true; cv_.notify_all(); continue;
          }
          auto submitted = engine_.submit_packed(r->prompt, r->maximum, r->sampling);
          if (!submitted.ok()) { r->status = submitted.status(); r->done = true; cv_.notify_all(); continue; }
          r->generation = *submitted; active_.emplace(*submitted, r);
        }
        if (stopping_) {
          for (auto& r : queue_) { r->result.cancelled = true; r->done = true; }
          queue_.clear();
        }
        if (active_.empty()) {
          cv_.notify_all(); if (stopping_) return;
          cv_.wait(lock); continue;
        }
        for (auto& [id, r] : active_) {
          if (std::chrono::steady_clock::now() >= r->deadline) {
            r->cancel = true; r->result.cancelled = true; r->result.deadline_elapsed = true;
          }
          if (std::chrono::steady_clock::now() > r->cleanup_deadline) {
            fail(Status::DeadlineExceeded("Qwen batch cleanup deadline elapsed")); return;
          }
          if ((stopping_ || r->cancel) && r->admitted && !r->cancel_sent && !r->draining) {
            auto s = engine_.cancel_packed(id); if (!s.ok()) { fail(s); return; }
            r->cancel_sent = true;
            r->cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
          }
          if (r->drain) {
            auto s = engine_.drain_packed(id);
            if (s.ok()) r->drain = false;
            else if (s.code() != StatusCode::kResourceExhausted && s.code() != StatusCode::kUnavailable) { fail(s); return; }
          }
        }
        lock.unlock(); auto step = engine_.drive_packed(); lock.lock();
        if (!step.ok()) { fail(step.status()); return; }
        std::uint64_t plan = 0; std::uint32_t index = 0, count = 0;
        for (;;) {
          auto received = engine_.try_take_packed_event();
          if (!received.ok()) { fail(received.status()); return; }
          if (!received->has_value()) break;
          const auto& e = (**received).event;
          auto found = active_.find(e.request_generation);
          if (found == active_.end()) { fail(Status::Internal("Qwen unknown request generation")); return; }
          auto r = found->second;
          if (e.plan_sequence) {
            if (!plan) { plan = e.plan_sequence; count = e.plan_event_count; index = 0; }
            if (plan != e.plan_sequence || !count || count != e.plan_event_count || e.plan_event_index != index++) {
              fail(Status::Internal("Qwen invalid batch output frontier")); return;
            }
            if (index == count) {
              auto s = engine_.acknowledge_packed_output(plan);
              if (!s.ok()) { fail(s); return; } plan = 0;
            }
          }
          if (!e.plan_sequence && plan) { fail(Status::Internal("Qwen interrupted batch output")); return; }
          switch (e.kind) {
            case ControllerOutputEventKind::kAdmitted:
              if (r->admitted) { fail(Status::Internal("Qwen duplicate admission")); return; }
              r->admitted = true; break;
            case ControllerOutputEventKind::kTokenCommitted:
              if (r->draining || e.token_id >= 151936 || e.token_ordinal != r->result.tokens.size() + 1 || r->result.tokens.size() >= r->maximum) {
                fail(Status::Internal("Qwen invalid token frontier")); return;
              }
              r->result.tokens.push_back(e.token_id);
              if (!r->cancel) {
                if (r->output.size() >= 256) { r->cancel = true; r->result.cancelled = true; }
                else r->output.push_back(e.token_id);
              }
              break;
            case ControllerOutputEventKind::kDraining:
              if (r->draining || plan) { fail(Status::Internal("Qwen invalid drain frontier")); return; }
              r->draining = true; r->drain = true; r->result.finish = e.finish_reason;
              r->cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5); break;
            case ControllerOutputEventKind::kCompleted:
            case ControllerOutputEventKind::kCancelled:
              if (!r->draining || r->drain || plan ||
                  (e.kind == ControllerOutputEventKind::kCancelled && !r->cancel_sent)) {
                fail(Status::Internal("Qwen invalid terminal frontier")); return;
              }
              if (e.kind == ControllerOutputEventKind::kCancelled && !r->result.stopped_by_callback) r->result.cancelled = true;
              r->done = true; active_.erase(found); break;
            case ControllerOutputEventKind::kFailed:
              if (r->admitted || plan) { fail(Status::Internal("Qwen packed request failed")); return; }
              r->status = Status::ResourceExhausted("Qwen KV admission capacity exhausted");
              r->done = true; active_.erase(found); break;
            default: fail(Status::Internal("Qwen packed request failed")); return;
          }
        }
        if (plan) { fail(Status::Internal("Qwen incomplete batch output")); return; }
        cv_.notify_all(); cv_.wait_for(lock, std::chrono::milliseconds(1));
      }
    } catch (...) {
      if (!lock.owns_lock()) lock.lock();
      fail(Status::Internal("Qwen batch owner exception"));
    }
  }
  Engine& engine_; std::uint32_t sequences_, pending_;
  std::mutex mutex_; std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> queue_;
  std::map<std::uint64_t, std::shared_ptr<Request>> active_;
  Status failure_ = Status::Ok(); bool stopping_ = false;
  std::thread worker_;
};
}
