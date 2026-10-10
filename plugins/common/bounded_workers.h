#pragma once
#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <stdexcept>
#include <thread>

namespace pih::plugin_runtime {
// Admission/join belong to the serving thread. Workers cannot mutate the list.
// Destruction joins before borrowed callback providers can be unloaded.
class BoundedWorkers final {
 public:
  explicit BoundedWorkers(std::size_t capacity) : capacity_(capacity) {
    if (!capacity || capacity > 8192) throw std::invalid_argument("worker capacity invalid");
  }
  BoundedWorkers(const BoundedWorkers&) = delete;
  ~BoundedWorkers() { Join(); }
  bool TryStart(std::function<void()> function) {
    Reap();
    if (workers_.size() == capacity_ || failed()) return false;
    workers_.emplace_back(); // Reserve ownership before starting the thread.
    auto& worker = workers_.back();
    try {
      auto done = worker.done;
      worker.thread = std::thread([this, done, function = std::move(function)] {
        try { function(); } catch (...) { failed_.store(true); }
        done->store(true, std::memory_order_release);
      });
    } catch (...) { workers_.pop_back(); throw; }
    return true;
  }
  void Join() noexcept {
    for (auto& worker : workers_) if (worker.thread.joinable()) worker.thread.join();
    workers_.clear();
  }
  bool failed() const noexcept { return failed_.load(); }
 private:
  void Reap() {
    for (auto it = workers_.begin(); it != workers_.end();) {
      if (!it->done->load(std::memory_order_acquire)) { ++it; continue; }
      it->thread.join();
      it = workers_.erase(it);
    }
  }
  struct Worker {
    std::shared_ptr<std::atomic<bool>> done = std::make_shared<std::atomic<bool>>(false);
    std::thread thread;
  };
  std::size_t capacity_;
  std::list<Worker> workers_;
  std::atomic<bool> failed_{false};
};
}
