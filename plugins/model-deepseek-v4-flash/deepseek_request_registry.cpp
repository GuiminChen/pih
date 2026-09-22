#include "pih/model/deepseek_request_registry.h"

#include <utility>

namespace pih {

Status DeepSeekRequestRegistry::submit(std::uint64_t request_id,
                                       std::uint64_t request_generation) {
  if (requests_.contains(request_id)) {
    return Status::FailedPrecondition("DeepSeek request ID is already live");
  }
  auto request = DeepSeekRequestLifecycle::Create(request_id,
                                                  request_generation);
  if (!request.ok()) return request.status();
  const auto status = request->admit();
  if (!status.ok()) return status;
  requests_.emplace(request_id, std::move(*request));
  return Status::Ok();
}

Status DeepSeekRequestRegistry::validate_prepare(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t plan_sequence) const {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->validate_prepare(plan_sequence)
                      : request.status();
}

Status DeepSeekRequestRegistry::prepare(std::uint64_t request_id,
                                        std::uint64_t request_generation,
                                        std::uint64_t plan_sequence) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->prepare(plan_sequence) : request.status();
}

Status DeepSeekRequestRegistry::validate_commit(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t plan_sequence) const {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->validate_commit(plan_sequence)
                      : request.status();
}

Status DeepSeekRequestRegistry::commit(std::uint64_t request_id,
                                       std::uint64_t request_generation,
                                       std::uint64_t plan_sequence) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->commit(plan_sequence) : request.status();
}

Status DeepSeekRequestRegistry::validate_abort_prepare(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t plan_sequence) const {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->validate_abort_prepare(plan_sequence)
                      : request.status();
}

Status DeepSeekRequestRegistry::abort_prepare(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t plan_sequence) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->abort_prepare(plan_sequence)
                      : request.status();
}

Status DeepSeekRequestRegistry::validate_cancel(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->validate_cancel(request_generation)
                      : request.status();
}

Status DeepSeekRequestRegistry::cancel(std::uint64_t request_id,
                                       std::uint64_t request_generation) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->cancel(request_generation) : request.status();
}

Status DeepSeekRequestRegistry::validate_backend_complete(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t plan_sequence) const {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->validate_backend_complete(plan_sequence)
                      : request.status();
}

Status DeepSeekRequestRegistry::backend_complete(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t plan_sequence, bool terminal) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->backend_complete(plan_sequence, terminal)
                      : request.status();
}

Status DeepSeekRequestRegistry::validate_retire(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  auto request = find(request_id, request_generation);
  if (!request.ok()) return request.status();
  if (!(*request)->user_terminal() || !(*request)->backend_drained()) {
    return Status::FailedPrecondition(
        "DeepSeek request cannot retire before terminal drain");
  }
  return Status::Ok();
}

Status DeepSeekRequestRegistry::retire(std::uint64_t request_id,
                                       std::uint64_t request_generation) {
  const auto status = validate_retire(request_id, request_generation);
  if (!status.ok()) return status;
  requests_.erase(request_id);
  return Status::Ok();
}

Status DeepSeekRequestRegistry::fail(std::uint64_t request_id,
                                     std::uint64_t request_generation) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->fail() : request.status();
}

Status DeepSeekRequestRegistry::finish(std::uint64_t request_id,
                                       std::uint64_t request_generation) {
  auto request = find(request_id, request_generation);
  return request.ok() ? (*request)->finish() : request.status();
}

void DeepSeekRequestRegistry::fail_all() noexcept {
  for (auto& [request_id, request] : requests_) {
    (void)request_id;
    if (!request.user_terminal()) (void)request.fail();
  }
}

void DeepSeekRequestRegistry::cancel_all() noexcept {
  for (auto& [request_id, request] : requests_) {
    (void)request_id;
    if (!request.user_terminal()) {
      (void)request.cancel(request.request_generation());
    }
  }
}

Result<DeepSeekRequestState> DeepSeekRequestRegistry::state(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  auto request = find(request_id, request_generation);
  if (!request.ok()) return request.status();
  return (*request)->state();
}

Result<DeepSeekRequestLifecycle*> DeepSeekRequestRegistry::find(
    std::uint64_t request_id, std::uint64_t request_generation) {
  const auto iterator = requests_.find(request_id);
  if (iterator == requests_.end() ||
      iterator->second.request_generation() != request_generation) {
    return Status::InvalidArgument("DeepSeek request identity is not live");
  }
  return &iterator->second;
}

Result<const DeepSeekRequestLifecycle*> DeepSeekRequestRegistry::find(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  const auto iterator = requests_.find(request_id);
  if (iterator == requests_.end() ||
      iterator->second.request_generation() != request_generation) {
    return Status::InvalidArgument("DeepSeek request identity is not live");
  }
  return &iterator->second;
}

}  // namespace pih
