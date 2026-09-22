#pragma once

#include "pih/model/deepseek_nccl_api_driver.h"

namespace pih {

class DeepSeekNcclContextActivator {
 public:
  virtual ~DeepSeekNcclContextActivator() = default;
  virtual Status activate(std::uintptr_t context_identity) = 0;
};

class DeepSeekContextBoundNcclCApi final : public DeepSeekNcclCApi {
 public:
  static Result<DeepSeekContextBoundNcclCApi> Create(
      std::uintptr_t context_identity, DeepSeekNcclCApi& api,
      DeepSeekNcclContextActivator& activator);

  Result<std::uint32_t> runtime_version() override;
  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      void** communicator, std::span<const std::byte> unique_id,
      std::uint32_t nranks, std::uint32_t rank,
      const DeepSeekNcclReleaseConfig& config) override;
  Result<DeepSeekNcclAsyncStatus> async_status(void* communicator) override;
  Result<std::uint32_t> communicator_count(void* communicator) override;
  Result<std::uint32_t> communicator_user_rank(void* communicator) override;
  Result<DeepSeekNcclAsyncStatus> finalize(void* communicator) override;
  Status destroy(void* communicator) override;
  Status abort(void* communicator) override;
  Status group_start() override;
  Status send(void* communicator, const void* buffer,
              std::uint64_t element_count, std::uint32_t peer,
              std::uintptr_t stream) override;
  Status recv(void* communicator, void* buffer,
              std::uint64_t element_count, std::uint32_t peer,
              std::uintptr_t stream) override;
  Result<DeepSeekNcclAsyncStatus> group_end() override;

 private:
  DeepSeekContextBoundNcclCApi(
      std::uintptr_t context_identity, DeepSeekNcclCApi& api,
      DeepSeekNcclContextActivator& activator) noexcept
      : context_identity_(context_identity), api_(&api),
        activator_(&activator) {}
  Status activate();

  std::uintptr_t context_identity_ = 0;
  DeepSeekNcclCApi* api_ = nullptr;
  DeepSeekNcclContextActivator* activator_ = nullptr;
};

}  // namespace pih
