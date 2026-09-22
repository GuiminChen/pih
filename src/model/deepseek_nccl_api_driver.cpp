#include "pih/model/deepseek_nccl_api_driver.h"

#include <utility>

namespace pih {

Result<DeepSeekNcclApiDriver> DeepSeekNcclApiDriver::Create(
    DeepSeekNcclCApi& api, DeepSeekNcclBootstrapLease& bootstrap,
    DeepSeekNcclReleaseConfig config, std::uint64_t engine_epoch,
    std::uint32_t edge_id, std::uint64_t bootstrap_lease_id,
    std::uint64_t device_identity, std::uintptr_t context_identity) {
  if (engine_epoch == 0 || bootstrap_lease_id == 0 || device_identity == 0 ||
      context_identity == 0 || config.nccl_release != 23102) {
    return Status::InvalidArgument("DeepSeek NCCL API driver identity is invalid");
  }
  auto runtime = api.runtime_version();
  if (!runtime.ok()) return runtime.status();
  if (*runtime != config.nccl_release) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL header and runtime release do not match 2.31.2");
  }
  auto borrowed = bootstrap.borrow(engine_epoch, edge_id, bootstrap_lease_id);
  if (!borrowed.ok()) return borrowed.status();
  return DeepSeekNcclApiDriver(api, bootstrap, std::move(config), engine_epoch,
                               edge_id, bootstrap_lease_id, device_identity,
                               context_identity);
}

DeepSeekNcclApiDriver::DeepSeekNcclApiDriver(
    DeepSeekNcclCApi& api, DeepSeekNcclBootstrapLease& bootstrap,
    DeepSeekNcclReleaseConfig config, std::uint64_t engine_epoch,
    std::uint32_t edge_id, std::uint64_t bootstrap_lease_id,
    std::uint64_t device_identity, std::uintptr_t context_identity) noexcept
    : api_(&api), bootstrap_(&bootstrap), config_(std::move(config)),
      engine_epoch_(engine_epoch), edge_id_(edge_id),
      bootstrap_lease_id_(bootstrap_lease_id), device_identity_(device_identity),
      context_identity_(context_identity) {}

Status DeepSeekNcclApiDriver::ensure_handle() const {
  if (communicator_ == nullptr) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator handle is absent");
  }
  return Status::Ok();
}

Result<DeepSeekNcclAsyncStatus> DeepSeekNcclApiDriver::init_rank_config(
    std::uint64_t bootstrap_lease_id, std::uint32_t nranks,
    std::uint32_t communicator_rank) {
  if (communicator_ != nullptr || bootstrap_lease_id != bootstrap_lease_id_ ||
      nranks != 2 || communicator_rank > 1) {
    return Status::FailedPrecondition("DeepSeek NCCL init arguments are invalid");
  }
  auto raw = bootstrap_->borrow(engine_epoch_, edge_id_, bootstrap_lease_id_);
  if (!raw.ok()) return raw.status();
  auto result = api_->init_rank_config(&communicator_, *raw, nranks,
                                       communicator_rank, config_);
  if (!result.ok()) {
    (void)bootstrap_->zeroize();
    return result.status();
  }
  if (*result != DeepSeekNcclAsyncStatus::kInProgress) {
    const auto scrubbed = bootstrap_->zeroize();
    if (!scrubbed.ok()) return scrubbed;
  }
  if (communicator_ == nullptr) {
    return Status::Internal("DeepSeek NCCL init returned no communicator handle");
  }
  return *result;
}

Result<DeepSeekNcclAsyncStatus> DeepSeekNcclApiDriver::async_status() {
  const auto present = ensure_handle();
  if (!present.ok()) return present;
  auto result = api_->async_status(communicator_);
  if (result.ok() && *result != DeepSeekNcclAsyncStatus::kInProgress &&
      !bootstrap_->zeroized()) {
    const auto scrubbed = bootstrap_->zeroize();
    if (!scrubbed.ok()) return scrubbed;
  }
  return result;
}

Result<std::uint32_t> DeepSeekNcclApiDriver::communicator_count() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  return api_->communicator_count(communicator_);
}

Result<std::uint32_t> DeepSeekNcclApiDriver::communicator_user_rank() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  return api_->communicator_user_rank(communicator_);
}

Result<std::uint64_t> DeepSeekNcclApiDriver::device_identity() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  return device_identity_;
}

Result<std::uintptr_t> DeepSeekNcclApiDriver::context_identity() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  return context_identity_;
}

Result<DeepSeekNcclAsyncStatus> DeepSeekNcclApiDriver::finalize() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  return api_->finalize(communicator_);
}

Status DeepSeekNcclApiDriver::destroy() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  const auto status = api_->destroy(communicator_);
  if (status.ok()) communicator_ = nullptr;
  return status;
}

Status DeepSeekNcclApiDriver::abort() {
  const auto present = ensure_handle(); if (!present.ok()) return present;
  const auto status = api_->abort(communicator_);
  if (status.ok()) communicator_ = nullptr;
  (void)bootstrap_->zeroize();
  return status;
}

Status DeepSeekNcclApiDriver::bind_p2p(
    DeepSeekNcclRole role, void* buffer, std::uint64_t buffer_bytes,
    std::uintptr_t stream) {
  const auto present = ensure_handle();
  if (!present.ok()) return present;
  if (buffer == nullptr || buffer_bytes == 0 || stream == 0 || group_started_ ||
      static_cast<std::uint8_t>(role) >
          static_cast<std::uint8_t>(DeepSeekNcclRole::kRecv)) {
    return Status::InvalidArgument("DeepSeek NCCL P2P binding is invalid");
  }
  bound_role_ = role;
  bound_buffer_ = buffer;
  bound_buffer_bytes_ = buffer_bytes;
  bound_stream_ = stream;
  operation_accepted_ = false;
  return Status::Ok();
}

Status DeepSeekNcclApiDriver::group_start() {
  if (!ensure_handle().ok() || bound_buffer_ == nullptr || group_started_) {
    return Status::FailedPrecondition("DeepSeek NCCL P2P group cannot start");
  }
  const auto status = api_->group_start();
  if (status.ok()) group_started_ = true;
  return status;
}

Status DeepSeekNcclApiDriver::send(std::uint64_t element_count,
                                   std::uint32_t peer) {
  if (!group_started_ || operation_accepted_ ||
      bound_role_ != DeepSeekNcclRole::kSend || element_count == 0 ||
      element_count > bound_buffer_bytes_ / 2U) {
    return Status::FailedPrecondition("DeepSeek NCCL send binding is invalid");
  }
  const auto status = api_->send(communicator_, bound_buffer_, element_count,
                                 peer, bound_stream_);
  if (status.ok()) operation_accepted_ = true;
  return status;
}

Status DeepSeekNcclApiDriver::recv(std::uint64_t element_count,
                                   std::uint32_t peer) {
  if (!group_started_ || operation_accepted_ ||
      bound_role_ != DeepSeekNcclRole::kRecv || element_count == 0 ||
      element_count > bound_buffer_bytes_ / 2U) {
    return Status::FailedPrecondition("DeepSeek NCCL recv binding is invalid");
  }
  const auto status = api_->recv(communicator_, bound_buffer_, element_count,
                                 peer, bound_stream_);
  if (status.ok()) operation_accepted_ = true;
  return status;
}

Result<DeepSeekNcclAsyncStatus> DeepSeekNcclApiDriver::group_end() {
  if (!group_started_ || !operation_accepted_) {
    return Status::FailedPrecondition("DeepSeek NCCL P2P group is incomplete");
  }
  auto result = api_->group_end();
  group_started_ = false;
  operation_accepted_ = false;
  bound_buffer_ = nullptr;
  bound_buffer_bytes_ = 0;
  bound_stream_ = 0;
  return result;
}

}  // namespace pih
