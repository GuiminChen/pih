#include "pih/model/deepseek_context_bound_nccl_api.h"

namespace pih {

Result<DeepSeekContextBoundNcclCApi> DeepSeekContextBoundNcclCApi::Create(
    std::uintptr_t context_identity, DeepSeekNcclCApi& api,
    DeepSeekNcclContextActivator& activator) {
  if (context_identity == 0) {
    return Status::InvalidArgument(
        "DeepSeek context-bound NCCL API context is null");
  }
  return DeepSeekContextBoundNcclCApi(context_identity, api, activator);
}

Status DeepSeekContextBoundNcclCApi::activate() {
  return activator_->activate(context_identity_);
}

Result<std::uint32_t> DeepSeekContextBoundNcclCApi::runtime_version() {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->runtime_version();
}

Result<DeepSeekNcclAsyncStatus>
DeepSeekContextBoundNcclCApi::init_rank_config(
    void** communicator, std::span<const std::byte> unique_id,
    std::uint32_t nranks, std::uint32_t rank,
    const DeepSeekNcclReleaseConfig& config) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->init_rank_config(communicator, unique_id, nranks, rank, config);
}

Result<DeepSeekNcclAsyncStatus> DeepSeekContextBoundNcclCApi::async_status(
    void* communicator) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->async_status(communicator);
}

Result<std::uint32_t> DeepSeekContextBoundNcclCApi::communicator_count(
    void* communicator) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->communicator_count(communicator);
}

Result<std::uint32_t> DeepSeekContextBoundNcclCApi::communicator_user_rank(
    void* communicator) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->communicator_user_rank(communicator);
}

Result<DeepSeekNcclAsyncStatus> DeepSeekContextBoundNcclCApi::finalize(
    void* communicator) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->finalize(communicator);
}

Status DeepSeekContextBoundNcclCApi::destroy(void* communicator) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->destroy(communicator);
}

Status DeepSeekContextBoundNcclCApi::abort(void* communicator) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->abort(communicator);
}

Status DeepSeekContextBoundNcclCApi::group_start() {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->group_start();
}

Status DeepSeekContextBoundNcclCApi::send(
    void* communicator, const void* buffer, std::uint64_t element_count,
    std::uint32_t peer, std::uintptr_t stream) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->send(communicator, buffer, element_count, peer, stream);
}

Status DeepSeekContextBoundNcclCApi::recv(
    void* communicator, void* buffer, std::uint64_t element_count,
    std::uint32_t peer, std::uintptr_t stream) {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->recv(communicator, buffer, element_count, peer, stream);
}

Result<DeepSeekNcclAsyncStatus> DeepSeekContextBoundNcclCApi::group_end() {
  auto status = activate();
  if (!status.ok()) return status;
  return api_->group_end();
}

}  // namespace pih
