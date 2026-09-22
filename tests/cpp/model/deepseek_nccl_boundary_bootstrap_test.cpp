#include "pih/model/deepseek_nccl_boundary_bootstrap.h"
#include "pih/model/deepseek_nccl_generation_warmup_runner.h"

#include <array>
#include <new>

#include <gtest/gtest.h>

namespace pih {
namespace {

class BootstrapPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t{alignment});
    return Allocation{data, bytes, alignment, generation_++, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data,
                      std::align_val_t{allocation.alignment});
  }
 private:
  std::uint64_t generation_ = 1;
};

class BootstrapApi final : public DeepSeekNcclCApi,
                           public DeepSeekNcclUniqueIdSource {
 public:
  Result<std::array<std::byte, DeepSeekNcclBootstrapLease::kUniqueIdBytes>>
  get_unique_id() override {
    ++unique_id_calls;
    std::array<std::byte, DeepSeekNcclBootstrapLease::kUniqueIdBytes> id{};
    id[0] = static_cast<std::byte>(unique_id_calls);
    return id;
  }
  Result<std::uint32_t> runtime_version() override { return 23102U; }
  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      void** communicator, std::span<const std::byte>, std::uint32_t,
      std::uint32_t rank, const DeepSeekNcclReleaseConfig&) override {
    handles[next_handle] = rank;
    *communicator = &handles[next_handle++];
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  Result<DeepSeekNcclAsyncStatus> async_status(void*) override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  Result<std::uint32_t> communicator_count(void*) override { return 2U; }
  Result<std::uint32_t> communicator_user_rank(void* communicator) override {
    return *static_cast<std::uint32_t*>(communicator);
  }
  Result<DeepSeekNcclAsyncStatus> finalize(void*) override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  Status destroy(void*) override { return Status::Ok(); }
  Status abort(void*) override { ++abort_calls; return Status::Ok(); }
  Status group_start() override { return Status::Ok(); }
  Status send(void*, const void*, std::uint64_t, std::uint32_t,
              std::uintptr_t) override { return Status::Ok(); }
  Status recv(void*, void*, std::uint64_t, std::uint32_t,
              std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekNcclAsyncStatus> group_end() override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  int unique_id_calls = 0;
  int abort_calls = 0;
  std::array<std::uint32_t, 6> handles{};
  std::size_t next_handle = 0;
};

class BootstrapWarmup final : public DeepSeekNcclGenerationWarmup {
 public:
  Result<std::vector<DeepSeekNcclWarmupReceipt>> run(
      DeepSeekNcclCommunicatorGeneration& generation,
      const DeepSeekNcclEndpointManifestPlan&,
      std::uint32_t maximum_wire_tokens) override {
    observed_maximum = maximum_wire_tokens;
    std::vector<DeepSeekNcclWarmupReceipt> receipts;
    for (std::uint32_t edge = 0; edge + 1 < generation.identity().world_size;
         ++edge) {
      receipts.push_back({generation.identity().engine_epoch,
                          generation.identity().communicator_generation,
                          edge, true, maximum_passed});
    }
    return receipts;
  }
  std::uint32_t observed_maximum = 0;
  bool maximum_passed = true;
};

class BootstrapWarmupPayload final
    : public DeepSeekNcclWarmupPayloadOperations {
 public:
  Status prepare(DeepSeekNcclRole, void*, std::uint64_t,
                 DriverStreamHandle, std::uint64_t) override {
    return Status::Ok();
  }
  Result<Sha256Digest> digest(const void*, std::uint64_t bytes) override {
    Sha256Digest digest{};
    digest.bytes[0] = static_cast<std::byte>(bytes & 0xffU);
    digest.bytes[1] = static_cast<std::byte>((bytes >> 8U) & 0xffU);
    return digest;
  }
};

class BootstrapEventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
};

class BootstrapEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class BootstrapClock final : public DeepSeekBoundaryPreparationClock {
 public:
  Result<std::uint64_t> now_ns() override { return ++now_; }
 private:
  std::uint64_t now_ = 100;
};

class BootstrapDriverFactory final : public DeepSeekBoundaryDriverOwnerFactory {
 public:
  Result<std::unique_ptr<DeepSeekBoundaryDriverOwner>> create(
      DeepSeekBoundaryTransportDriver&) override {
    return Status::Internal("not exercised");
  }
};

class BootstrapDeviceAllocator final : public Allocator {
 public:
  explicit BootstrapDeviceAllocator(std::int32_t rank)
      : device_(Device::Create(DeviceType::kCuda, rank).value()) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    return Allocation{data, bytes, alignment, generation_++, device_};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  Device device_;
  std::uint64_t generation_ = 1;
};

class BootstrapRuntime final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,
                                                std::uint32_t) override {
    return 1;
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override { return 2; }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t) override { return next_event_++; }
  void destroy_event(DriverEventHandle) noexcept override {}
  void destroy_stream(DriverStreamHandle) noexcept override {}
  void release_primary_context(std::int32_t,
                               std::uintptr_t) noexcept override {}
 private:
  DriverEventHandle next_event_ = 10;
};

struct OwnedAssemblyInputs final {
  std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>> resources;
  std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks;
  std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>> factories;
  std::vector<DeepSeekRankBoundaryFactoryIdentity> identities;
};

OwnedAssemblyInputs assembly_inputs(
    std::array<BootstrapDeviceAllocator, 3>& allocators,
    BootstrapRuntime& runtime) {
  OwnedAssemblyInputs result;
  for (std::uint32_t rank = 0; rank < 3; ++rank) {
    result.resources.push_back(DeepSeekRankBoundaryDeviceResources::Allocate(
        rank, 3, 2, 100 + rank, allocators[rank], runtime).value());
    result.clocks.push_back(std::make_unique<BootstrapClock>());
    result.factories.push_back(std::make_unique<BootstrapDriverFactory>());
    result.identities.push_back(
        {7, rank, 3, 100 + rank, 200 + rank, 11, 1000,
         rank == 0 ? std::array<std::uint64_t, 2>{0, 0}
                   : std::array<std::uint64_t, 2>{300 + rank * 2,
                                                  301 + rank * 2}});
  }
  return result;
}

TEST(DeepSeekNcclBoundaryBootstrapTest,
     BuildsSealedAdjacentGenerationForThreeRanks) {
  BootstrapApi api;
  BootstrapPinnedAllocator pinned;
  BootstrapWarmupPayload payload;
  BootstrapEventDriver events;
  BootstrapEvidence evidence;
  std::array<BootstrapClock, 4> warmup_clocks;
  std::array<std::byte, 4> warmup_buffers{};
  std::vector<DeepSeekNcclGenerationWarmupEndpointResources>
      warmup_resources;
  for (std::uint32_t endpoint = 0; endpoint < 4; ++endpoint) {
    warmup_resources.push_back(
        {100 + endpoint * 2, 200 + endpoint, 1, &warmup_buffers[endpoint],
         65536, 300 + endpoint, {400 + endpoint * 2, 401 + endpoint * 2},
         100, 1000, &payload, &events, &evidence,
         &warmup_clocks[endpoint]});
  }
  auto warmup = DeepSeekNcclGenerationWarmupRunner::Create(
      std::move(warmup_resources)).value();
  BootstrapRuntime runtime;
  std::array<BootstrapDeviceAllocator, 3> allocators{{
      BootstrapDeviceAllocator(0), BootstrapDeviceAllocator(1),
      BootstrapDeviceAllocator(2)}};
  auto owned = assembly_inputs(allocators, runtime);
  const std::array<DeepSeekNcclRankEndpointIdentity, 3> endpoints{{
      {0, 10, 100}, {1, 11, 101}, {2, 12, 102}}};
  const std::array<DeepSeekNcclCApi*, 3> apis{{&api, &api, &api}};
  const auto release = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128).value();
  const std::array<DeepSeekNcclReleaseConfig, 3> releases{{
      release, release, release}};
  auto result = DeepSeekNcclBoundaryBootstrap::Build(
      {7, 11, 13, 3, 2, 17, 1000, 2000}, endpoints, apis, releases,
      api, pinned, warmup, std::move(owned.identities),
      std::move(owned.resources), std::move(owned.clocks),
      std::move(owned.factories));
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ((*result)->world_size(), 3U);
  EXPECT_EQ((*result)->generation_state(), DeepSeekNcclGenerationState::kSealed);
  EXPECT_EQ(api.unique_id_calls, 2);
}

TEST(DeepSeekNcclBoundaryBootstrapTest,
     RejectsNonContiguousRanksBeforeIssuingCapabilities) {
  BootstrapApi api;
  BootstrapPinnedAllocator pinned;
  BootstrapWarmup warmup;
  const std::array<DeepSeekNcclRankEndpointIdentity, 3> endpoints{{
      {0, 10, 100}, {2, 11, 101}, {1, 12, 102}}};
  const std::array<DeepSeekNcclCApi*, 3> apis{{&api, &api, &api}};
  const auto release = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128).value();
  const std::array<DeepSeekNcclReleaseConfig, 3> releases{{
      release, release, release}};
  auto result = DeepSeekNcclBoundaryBootstrap::Build(
      {7, 11, 13, 3, 2, 17, 1000, 2000}, endpoints, apis, releases,
      api, pinned, warmup, {}, {}, {}, {});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(api.unique_id_calls, 0);
}

TEST(DeepSeekNcclBoundaryBootstrapTest,
     AbortsAllEndpointsWhenMaximumWarmupEvidenceIsMissing) {
  BootstrapApi api;
  BootstrapPinnedAllocator pinned;
  BootstrapWarmup warmup;
  warmup.maximum_passed = false;
  BootstrapRuntime runtime;
  std::array<BootstrapDeviceAllocator, 3> allocators{{
      BootstrapDeviceAllocator(0), BootstrapDeviceAllocator(1),
      BootstrapDeviceAllocator(2)}};
  auto owned = assembly_inputs(allocators, runtime);
  const std::array<DeepSeekNcclRankEndpointIdentity, 3> endpoints{{
      {0, 10, 100}, {1, 11, 101}, {2, 12, 102}}};
  const std::array<DeepSeekNcclCApi*, 3> apis{{&api, &api, &api}};
  const auto release = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128).value();
  const std::array<DeepSeekNcclReleaseConfig, 3> releases{{
      release, release, release}};
  auto result = DeepSeekNcclBoundaryBootstrap::Build(
      {7, 11, 13, 3, 2, 17, 1000, 2000}, endpoints, apis, releases,
      api, pinned, warmup, std::move(owned.identities),
      std::move(owned.resources), std::move(owned.clocks),
      std::move(owned.factories));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInternal);
  EXPECT_EQ(api.abort_calls, 4);
}

TEST(DeepSeekNcclGenerationWarmupRunnerTest,
     RejectsOverlappingEndpointOperationOrdinals) {
  BootstrapWarmupPayload payload;
  BootstrapEventDriver events;
  BootstrapEvidence evidence;
  BootstrapClock clock;
  std::array<std::byte, 2> buffers{};
  std::vector<DeepSeekNcclGenerationWarmupEndpointResources> endpoints;
  endpoints.push_back(
      {100, 200, 1, &buffers[0], 32768, 300, {400, 401}, 100, 1000,
       &payload, &events, &evidence, &clock});
  endpoints.push_back(
      {101, 201, 1, &buffers[1], 32768, 301, {402, 403}, 100, 1000,
       &payload, &events, &evidence, &clock});
  auto result = DeepSeekNcclGenerationWarmupRunner::Create(
      std::move(endpoints));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
