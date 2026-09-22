#pragma once

#include <optional>

#include "pih/model/deepseek_rank_serving_protocol.h"

namespace pih {

class DeepSeekRankServingWorkerTransport {
 public:
  virtual ~DeepSeekRankServingWorkerTransport() = default;
  virtual Result<std::optional<DeepSeekRankServingCommand>> receive() = 0;
  virtual Status send(const DeepSeekRankServingCompletion& completion) = 0;
};

class DeepSeekRankServingWorkerExecutor {
 public:
  virtual ~DeepSeekRankServingWorkerExecutor() = default;
  virtual Status start(const DeepSeekRankServingCommand& command) = 0;
  virtual Status cancel(const DeepSeekRankServingCommand& command) = 0;
  virtual Result<std::optional<DeepSeekRankServingCompletion>> poll() = 0;
};

class DeepSeekRankServingWorkerLoop final {
 public:
  static Result<DeepSeekRankServingWorkerLoop> Create(
      DeepSeekRankServingWorkerGate gate,
      DeepSeekRankServingWorkerTransport& transport,
      DeepSeekRankServingWorkerExecutor& executor);
  Status advance();
  [[nodiscard]] bool failed() const noexcept { return failed_; }

 private:
  DeepSeekRankServingWorkerLoop(DeepSeekRankServingWorkerGate gate,
      DeepSeekRankServingWorkerTransport& transport,
      DeepSeekRankServingWorkerExecutor& executor) noexcept
      : gate_(std::move(gate)), transport_(&transport), executor_(&executor) {}
  Status fail(Status cause) noexcept;
  DeepSeekRankServingWorkerGate gate_;
  DeepSeekRankServingWorkerTransport* transport_ = nullptr;
  DeepSeekRankServingWorkerExecutor* executor_ = nullptr;
  std::optional<DeepSeekRankServingCompletion> pending_completion_;
  bool failed_ = false;
};

}  // namespace pih
