#include "../../plugins/common/bounded_workers.h"
#include <atomic>
#include <stdexcept>
#include <chrono>
#include <future>
#include <thread>

void Require(bool condition) { if (!condition) throw std::runtime_error("bounded worker assertion failed"); }

int main() {
  using namespace std::chrono_literals;
  pih::plugin_runtime::BoundedWorkers pool(2);
  std::promise<void> released;
  auto ready = released.get_future().share();
  std::atomic<unsigned> running{0}, finished{0};
  auto task = [&] { ++running; ready.wait(); ++finished; };
  Require(pool.TryStart(task));
  Require(pool.TryStart(task));
  Require(!pool.TryStart(task));
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (running != 2 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  Require(running == 2); // Both connections execute concurrently.
  released.set_value();
  pool.Join();
  Require(finished == 2);
  Require(pool.TryStart([] { throw 42; }));
  pool.Join();
  Require(pool.failed());
}
