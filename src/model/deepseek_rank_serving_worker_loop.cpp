#include "pih/model/deepseek_rank_serving_worker_loop.h"

namespace pih {
Result<DeepSeekRankServingWorkerLoop> DeepSeekRankServingWorkerLoop::Create(
    DeepSeekRankServingWorkerGate gate, DeepSeekRankServingWorkerTransport& transport,
    DeepSeekRankServingWorkerExecutor& executor) {
  return DeepSeekRankServingWorkerLoop(std::move(gate), transport, executor);
}
Status DeepSeekRankServingWorkerLoop::fail(Status cause) noexcept {
  failed_ = true;
  return cause.ok() ? Status::Internal("DeepSeek rank serving worker failed") : cause;
}
Status DeepSeekRankServingWorkerLoop::advance() {
  if (failed_) return Status::FailedPrecondition("DeepSeek rank serving worker is failed");
  if (pending_completion_) {
    auto status = transport_->send(*pending_completion_);
    if (status.code() == StatusCode::kUnavailable) return status;
    if (!status.ok()) return fail(status);
    pending_completion_.reset();
    return Status::Ok();
  }
  auto command = transport_->receive();
  if (!command.ok()) return fail(command.status());
  if (command->has_value()) {
    auto status = gate_.accept(**command);
    if (!status.ok()) return fail(status);
    status = (**command).kind == DeepSeekRankServingCommandKind::kExecute
                 ? executor_->start(**command) : executor_->cancel(**command);
    if (!status.ok()) return fail(status);
  }
  auto completion = executor_->poll();
  if (!completion.ok()) return fail(completion.status());
  if (!completion->has_value()) return Status::Ok();
  auto status = gate_.complete(**completion);
  if (!status.ok()) return fail(status);
  pending_completion_ = std::move(**completion);
  return Status::Ok();
}
}  // namespace pih
