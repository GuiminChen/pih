#include "pih/model/deepseek_nccl_edge_endpoint.h"

#include <array>
#include <new>

#include <gtest/gtest.h>

namespace pih {
namespace {

class EdgeAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t, std::uint64_t) override {
    data = ::operator new(128, std::align_val_t{64});
    return Allocation{data, 128, 64, 1, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data, std::align_val_t{64});
    data = nullptr;
  }
  void* data = nullptr;
};

class EdgeApi final : public DeepSeekNcclCApi {
 public:
  Result<std::uint32_t> runtime_version() override { return 23102U; }
  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      void** communicator, std::span<const std::byte>, std::uint32_t,
      std::uint32_t rank, const DeepSeekNcclReleaseConfig&) override {
    observed_rank = rank;
    *communicator = &handle;
    return init_result;
  }
  Result<DeepSeekNcclAsyncStatus> async_status(void*) override {
    return async_result;
  }
  Result<std::uint32_t> communicator_count(void*) override { return 2U; }
  Result<std::uint32_t> communicator_user_rank(void*) override {
    return observed_rank;
  }
  Result<DeepSeekNcclAsyncStatus> finalize(void*) override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  Status destroy(void*) override { ++destroy_calls; return Status::Ok(); }
  Status abort(void*) override { ++abort_calls; return Status::Ok(); }
  Status group_start() override { return Status::Ok(); }
  Status send(void*, const void*, std::uint64_t, std::uint32_t,
              std::uintptr_t) override { return Status::Ok(); }
  Status recv(void*, void*, std::uint64_t, std::uint32_t,
              std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekNcclAsyncStatus> group_end() override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }

  int handle = 0;
  std::uint32_t observed_rank = 0;
  DeepSeekNcclAsyncStatus init_result = DeepSeekNcclAsyncStatus::kSuccess;
  DeepSeekNcclAsyncStatus async_result = DeepSeekNcclAsyncStatus::kSuccess;
  int abort_calls = 0;
  int destroy_calls = 0;
};

DeepSeekNcclCommunicatorManifest endpoint_manifest() {
  return {.engine_epoch = 3,
          .communicator_generation = 4,
          .bootstrap_lease_id = 5,
          .bootstrap_commitment_id = 6,
          .edge_id = 1,
          .local_global_rank = 1,
          .peer_global_rank = 2,
          .communicator_local_rank = 0,
          .device_identity = 7,
          .context_identity = 8,
          .config_identity = 9};
}

Result<DeepSeekNcclEdgeEndpoint> make_endpoint(EdgeAllocator& allocator,
                                               EdgeApi& api) {
  std::array<std::byte, 128> raw{};
  raw[0] = std::byte{7};
  auto lease = DeepSeekNcclBootstrapLease::Create(
      allocator, raw, 3, 1, 5);
  if (!lease.ok()) return lease.status();
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128);
  if (!config.ok()) return config.status();
  return DeepSeekNcclEdgeEndpoint::Create(
      api, std::make_unique<DeepSeekNcclBootstrapLease>(std::move(*lease)),
      *config, endpoint_manifest());
}

TEST(DeepSeekNcclEdgeEndpointTest, ExposesTransportOnlyAfterSealed) {
  EdgeAllocator allocator;
  EdgeApi api;
  auto endpoint = make_endpoint(allocator, api);
  ASSERT_TRUE(endpoint.ok()) << endpoint.status().message();
  EXPECT_EQ(endpoint->transport(), nullptr);
  ASSERT_TRUE(endpoint->accept_bootstrap(true).ok());
  ASSERT_TRUE(endpoint->begin_init().ok());
  EXPECT_TRUE(endpoint->bootstrap_zeroized());
  ASSERT_TRUE(endpoint->reconcile().ok());
  ASSERT_TRUE(endpoint->mark_warmed(true, true).ok());
  ASSERT_TRUE(endpoint->seal().ok());
  EXPECT_NE(endpoint->transport(), nullptr);
  ASSERT_TRUE(endpoint->begin_finalize().ok());
  ASSERT_TRUE(endpoint->destroy().ok());
  EXPECT_EQ(api.destroy_calls, 1);
  EXPECT_EQ(api.abort_calls, 0);
}

TEST(DeepSeekNcclEdgeEndpointTest, RetainsLeaseDuringAsyncInit) {
  EdgeAllocator allocator;
  EdgeApi api;
  api.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  auto endpoint = make_endpoint(allocator, api);
  ASSERT_TRUE(endpoint.ok());
  ASSERT_TRUE(endpoint->accept_bootstrap(true).ok());
  ASSERT_TRUE(endpoint->begin_init().ok());
  EXPECT_FALSE(endpoint->bootstrap_zeroized());
  api.async_result = DeepSeekNcclAsyncStatus::kSuccess;
  ASSERT_TRUE(endpoint->poll_init().ok());
  EXPECT_TRUE(endpoint->bootstrap_zeroized());
}

TEST(DeepSeekNcclEdgeEndpointTest, DestructorAbortsLiveCommunicatorOnce) {
  EdgeAllocator allocator;
  EdgeApi api;
  {
    auto endpoint = make_endpoint(allocator, api);
    ASSERT_TRUE(endpoint.ok());
    ASSERT_TRUE(endpoint->accept_bootstrap(true).ok());
    ASSERT_TRUE(endpoint->begin_init().ok());
  }
  EXPECT_EQ(api.abort_calls, 1);
  EXPECT_EQ(allocator.data, nullptr);
}

}  // namespace
}  // namespace pih
