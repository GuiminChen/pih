#include "pih/model/deepseek_nccl_api_driver.h"
#include "pih/model/deepseek_context_bound_nccl_api.h"

#include <gtest/gtest.h>

#include <array>
#include <new>

namespace pih {
namespace {

class ApiLeaseAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t, std::uint64_t) override {
    data = ::operator new(128, std::align_val_t{64});
    return Allocation{data, 128, 64, 1, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data, std::align_val_t{64}); data = nullptr;
  }
  void* data = nullptr;
};

struct FakeApi final : DeepSeekNcclCApi {
  Result<std::uint32_t> runtime_version() override { return version; }
  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      void** output, std::span<const std::byte> unique_id, std::uint32_t nranks,
      std::uint32_t rank, const DeepSeekNcclReleaseConfig& config) override {
    ++init_calls; id_bytes = unique_id.size(); observed_nranks = nranks;
    observed_rank = rank; observed_config = config; *output = &handle;
    return init_result;
  }
  Result<DeepSeekNcclAsyncStatus> async_status(void*) override { return async_result; }
  Result<std::uint32_t> communicator_count(void*) override { return 2U; }
  Result<std::uint32_t> communicator_user_rank(void*) override { return observed_rank; }
  Result<DeepSeekNcclAsyncStatus> finalize(void*) override { return finalize_result; }
  Status destroy(void*) override { ++destroy_calls; return Status::Ok(); }
  Status abort(void*) override { ++abort_calls; return Status::Ok(); }
  Status group_start() override { ++group_start_calls; return Status::Ok(); }
  Status send(void*, const void* buffer, std::uint64_t count,
              std::uint32_t peer, std::uintptr_t stream) override {
    ++send_calls; observed_buffer = buffer; observed_count = count;
    observed_peer = peer; observed_stream = stream; return Status::Ok();
  }
  Status recv(void*, void* buffer, std::uint64_t count,
              std::uint32_t peer, std::uintptr_t stream) override {
    ++recv_calls; observed_buffer = buffer; observed_count = count;
    observed_peer = peer; observed_stream = stream; return Status::Ok();
  }
  Result<DeepSeekNcclAsyncStatus> group_end() override {
    ++group_end_calls; return DeepSeekNcclAsyncStatus::kSuccess;
  }
  std::uint32_t version = 23102;
  DeepSeekNcclAsyncStatus init_result = DeepSeekNcclAsyncStatus::kSuccess;
  DeepSeekNcclAsyncStatus async_result = DeepSeekNcclAsyncStatus::kSuccess;
  DeepSeekNcclAsyncStatus finalize_result = DeepSeekNcclAsyncStatus::kSuccess;
  int handle = 0;
  int init_calls = 0;
  int destroy_calls = 0;
  int abort_calls = 0;
  int group_start_calls = 0;
  int send_calls = 0;
  int recv_calls = 0;
  int group_end_calls = 0;
  const void* observed_buffer = nullptr;
  std::uint64_t observed_count = 0;
  std::uint32_t observed_peer = 0;
  std::uintptr_t observed_stream = 0;
  std::size_t id_bytes = 0;
  std::uint32_t observed_nranks = 0;
  std::uint32_t observed_rank = 0;
  DeepSeekNcclReleaseConfig observed_config;
};

class FakeContextActivator final : public DeepSeekNcclContextActivator {
 public:
  Status activate(std::uintptr_t context_identity) override {
    contexts.push_back(context_identity);
    return status;
  }
  Status status = Status::Ok();
  std::vector<std::uintptr_t> contexts;
};

TEST(DeepSeekContextBoundNcclCApiTest,
     ActivatesDeclaredContextBeforeEveryDelegatedOperation) {
  FakeApi raw;
  FakeContextActivator activator;
  auto api = DeepSeekContextBoundNcclCApi::Create(77, raw, activator);
  ASSERT_TRUE(api.ok()) << api.status().message();
  EXPECT_EQ(api->runtime_version().value(), 23102U);
  std::array<std::byte, 128> unique_id{};
  void* communicator = nullptr;
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128).value();
  ASSERT_TRUE(api->init_rank_config(
      &communicator, unique_id, 2, 0, config).ok());
  ASSERT_TRUE(api->group_start().ok());
  std::array<std::byte, 2> buffer{};
  ASSERT_TRUE(api->send(communicator, buffer.data(), 1, 1, 99).ok());
  ASSERT_TRUE(api->group_end().ok());
  ASSERT_TRUE(api->finalize(communicator).ok());
  ASSERT_TRUE(api->destroy(communicator).ok());
  ASSERT_EQ(activator.contexts.size(), 7U);
  EXPECT_TRUE(std::ranges::all_of(
      activator.contexts, [](std::uintptr_t value) { return value == 77; }));
  EXPECT_EQ(raw.init_calls, 1);
  EXPECT_EQ(raw.send_calls, 1);
  EXPECT_EQ(raw.destroy_calls, 1);
}

TEST(DeepSeekContextBoundNcclCApiTest,
     SuppressesNcclCallWhenContextActivationFails) {
  FakeApi raw;
  FakeContextActivator activator;
  activator.status = Status::Internal("context activation failed");
  auto api = DeepSeekContextBoundNcclCApi::Create(77, raw, activator).value();
  EXPECT_FALSE(api.group_start().ok());
  EXPECT_EQ(raw.group_start_calls, 0);
  EXPECT_EQ(activator.contexts, (std::vector<std::uintptr_t>{77}));
}

TEST(DeepSeekNcclApiDriverTest, MapsPinnedConfigAndConsumesLeaseOnReady) {
  ApiLeaseAllocator allocator;
  std::array<std::byte, 128> raw{}; raw[0] = std::byte{7};
  auto lease = DeepSeekNcclBootstrapLease::Create(allocator, raw, 3, 1, 5);
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm90, 23102, 128);
  ASSERT_TRUE(lease.ok()); ASSERT_TRUE(config.ok());
  FakeApi api;
  auto driver = DeepSeekNcclApiDriver::Create(
      api, *lease, *config, 3, 1, 5, 7, 8);
  ASSERT_TRUE(driver.ok()) << driver.status().message();
  auto initialized = driver->init_rank_config(5, 2, 0);
  ASSERT_TRUE(initialized.ok());
  EXPECT_EQ(*initialized, DeepSeekNcclAsyncStatus::kSuccess);
  EXPECT_EQ(api.id_bytes, 128U);
  EXPECT_EQ(api.observed_config.host_cft_disabled, 1);
  EXPECT_TRUE(lease->zeroized());
  EXPECT_EQ(driver->communicator_count().value(), 2U);
  EXPECT_EQ(driver->device_identity().value(), 7U);
  EXPECT_EQ(driver->context_identity().value(), 8U);
}

TEST(DeepSeekNcclApiDriverTest, RetainsLeaseWhileInitInProgressThenZeroizes) {
  ApiLeaseAllocator allocator;
  std::array<std::byte, 128> raw{};
  auto lease = DeepSeekNcclBootstrapLease::Create(allocator, raw, 3, 1, 5);
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128);
  ASSERT_TRUE(lease.ok()); ASSERT_TRUE(config.ok());
  FakeApi api; api.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  auto driver = DeepSeekNcclApiDriver::Create(api, *lease, *config, 3, 1, 5, 7, 8);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->init_rank_config(5, 2, 0).ok());
  EXPECT_FALSE(lease->zeroized());
  api.async_result = DeepSeekNcclAsyncStatus::kSuccess;
  EXPECT_EQ(driver->async_status().value(), DeepSeekNcclAsyncStatus::kSuccess);
  EXPECT_TRUE(lease->zeroized());
}

TEST(DeepSeekNcclApiDriverTest, RejectsRuntimeAndLeaseMismatch) {
  ApiLeaseAllocator allocator;
  std::array<std::byte, 128> raw{};
  auto lease = DeepSeekNcclBootstrapLease::Create(allocator, raw, 3, 1, 5);
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128);
  ASSERT_TRUE(lease.ok()); ASSERT_TRUE(config.ok());
  FakeApi api; api.version = 23007;
  EXPECT_FALSE(DeepSeekNcclApiDriver::Create(
      api, *lease, *config, 3, 1, 5, 7, 8).ok());
  api.version = 23102;
  auto driver = DeepSeekNcclApiDriver::Create(api, *lease, *config, 3, 1, 5, 7, 8);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->init_rank_config(6, 2, 0).ok());
}

TEST(DeepSeekNcclApiDriverTest, BindsExactlyOneBf16P2pOperation) {
  ApiLeaseAllocator allocator;
  std::array<std::byte, 128> raw{};
  auto lease = DeepSeekNcclBootstrapLease::Create(allocator, raw, 3, 1, 5);
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128);
  ASSERT_TRUE(lease.ok()); ASSERT_TRUE(config.ok());
  FakeApi api;
  auto driver = DeepSeekNcclApiDriver::Create(
      api, *lease, *config, 3, 1, 5, 7, 8);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->init_rank_config(5, 2, 0).ok());
  std::array<std::byte, 65536> buffer{};
  ASSERT_TRUE(driver->bind_p2p(DeepSeekNcclRole::kSend, buffer.data(),
                               buffer.size(), 99).ok());
  ASSERT_TRUE(driver->group_start().ok());
  ASSERT_TRUE(driver->send(32768, 1).ok());
  EXPECT_FALSE(driver->send(32768, 1).ok());
  auto ended = driver->group_end();
  ASSERT_TRUE(ended.ok());
  EXPECT_EQ(api.group_start_calls, 1);
  EXPECT_EQ(api.send_calls, 1);
  EXPECT_EQ(api.group_end_calls, 1);
  EXPECT_EQ(api.observed_buffer, buffer.data());
  EXPECT_EQ(api.observed_count, 32768U);
  EXPECT_EQ(api.observed_peer, 1U);
  EXPECT_EQ(api.observed_stream, 99U);
}

TEST(DeepSeekNcclApiDriverTest, RejectsP2pBeyondBoundBufferBeforeApiCall) {
  ApiLeaseAllocator allocator;
  std::array<std::byte, 128> raw{};
  auto lease = DeepSeekNcclBootstrapLease::Create(allocator, raw, 3, 1, 5);
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128);
  ASSERT_TRUE(lease.ok()); ASSERT_TRUE(config.ok());
  FakeApi api;
  auto driver = DeepSeekNcclApiDriver::Create(
      api, *lease, *config, 3, 1, 5, 7, 8);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->init_rank_config(5, 2, 1).ok());
  std::array<std::byte, 2> buffer{};
  ASSERT_TRUE(driver->bind_p2p(DeepSeekNcclRole::kRecv, buffer.data(),
                               buffer.size(), 77).ok());
  ASSERT_TRUE(driver->group_start().ok());
  EXPECT_FALSE(driver->recv(2, 0).ok());
  EXPECT_EQ(api.recv_calls, 0);
  EXPECT_FALSE(driver->group_end().ok());
}

}  // namespace
}  // namespace pih
