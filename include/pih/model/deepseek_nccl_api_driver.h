#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/model/deepseek_nccl_bootstrap_lease.h"
#include "pih/model/deepseek_nccl_communicator.h"

namespace pih {

class DeepSeekNcclCApi {
 public:
  virtual ~DeepSeekNcclCApi() = default;
  virtual Result<std::uint32_t> runtime_version() = 0;
  virtual Result<DeepSeekNcclAsyncStatus> init_rank_config(
      void** communicator, std::span<const std::byte> unique_id,
      std::uint32_t nranks, std::uint32_t rank,
      const DeepSeekNcclReleaseConfig& config) = 0;
  virtual Result<DeepSeekNcclAsyncStatus> async_status(void* communicator) = 0;
  virtual Result<std::uint32_t> communicator_count(void* communicator) = 0;
  virtual Result<std::uint32_t> communicator_user_rank(void* communicator) = 0;
  virtual Result<DeepSeekNcclAsyncStatus> finalize(void* communicator) = 0;
  virtual Status destroy(void* communicator) = 0;
  virtual Status abort(void* communicator) = 0;
  virtual Status group_start() = 0;
  virtual Status send(void* communicator, const void* buffer,
                      std::uint64_t element_count, std::uint32_t peer,
                      std::uintptr_t stream) = 0;
  virtual Status recv(void* communicator, void* buffer,
                      std::uint64_t element_count, std::uint32_t peer,
                      std::uintptr_t stream) = 0;
  virtual Result<DeepSeekNcclAsyncStatus> group_end() = 0;
};

class DeepSeekNcclApiDriver final : public DeepSeekNcclCommunicatorDriver,
                                    public DeepSeekBoundaryTransportDriver {
 public:
  static Result<DeepSeekNcclApiDriver> Create(
      DeepSeekNcclCApi& api, DeepSeekNcclBootstrapLease& bootstrap,
      DeepSeekNcclReleaseConfig config, std::uint64_t engine_epoch,
      std::uint32_t edge_id, std::uint64_t bootstrap_lease_id,
      std::uint64_t device_identity, std::uintptr_t context_identity);

  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      std::uint64_t bootstrap_lease_id, std::uint32_t nranks,
      std::uint32_t communicator_rank) override;
  Result<DeepSeekNcclAsyncStatus> async_status() override;
  Result<std::uint32_t> communicator_count() override;
  Result<std::uint32_t> communicator_user_rank() override;
  Result<std::uint64_t> device_identity() override;
  Result<std::uintptr_t> context_identity() override;
  Result<DeepSeekNcclAsyncStatus> finalize() override;
  Status destroy() override;
  Status abort() override;
  Status bind_p2p(DeepSeekNcclRole role, void* buffer,
                  std::uint64_t buffer_bytes, std::uintptr_t stream) override;
  Status group_start() override;
  Status send(std::uint64_t element_count, std::uint32_t peer) override;
  Status recv(std::uint64_t element_count, std::uint32_t peer) override;
  Result<DeepSeekNcclAsyncStatus> group_end() override;

 private:
  DeepSeekNcclApiDriver(DeepSeekNcclCApi& api,
                        DeepSeekNcclBootstrapLease& bootstrap,
                        DeepSeekNcclReleaseConfig config,
                        std::uint64_t engine_epoch, std::uint32_t edge_id,
                        std::uint64_t bootstrap_lease_id,
                        std::uint64_t device_identity,
                        std::uintptr_t context_identity) noexcept;
  Status ensure_handle() const;
  DeepSeekNcclCApi* api_ = nullptr;
  DeepSeekNcclBootstrapLease* bootstrap_ = nullptr;
  DeepSeekNcclReleaseConfig config_;
  std::uint64_t engine_epoch_ = 0;
  std::uint32_t edge_id_ = 0;
  std::uint64_t bootstrap_lease_id_ = 0;
  std::uint64_t device_identity_ = 0;
  std::uintptr_t context_identity_ = 0;
  void* communicator_ = nullptr;
  DeepSeekNcclRole bound_role_ = DeepSeekNcclRole::kSend;
  void* bound_buffer_ = nullptr;
  std::uint64_t bound_buffer_bytes_ = 0;
  std::uintptr_t bound_stream_ = 0;
  bool group_started_ = false;
  bool operation_accepted_ = false;
};

}  // namespace pih
