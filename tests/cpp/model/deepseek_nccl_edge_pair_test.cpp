#include "pih/model/deepseek_nccl_edge_pair.h"
#include "pih/model/deepseek_nccl_communicator_generation.h"
#include "pih/model/deepseek_nccl_generation_boundary_assembly.h"

#include <array>
#include <new>

#include <gtest/gtest.h>

namespace pih {
namespace {

class PairAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t, std::uint64_t) override {
    auto* data = ::operator new(128, std::align_val_t{64});
    return Allocation{data, 128, 64, generation_++, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data, std::align_val_t{64});
  }
 private:
  std::uint64_t generation_ = 1;
};

class PairApi final : public DeepSeekNcclCApi {
 public:
  Result<std::uint32_t> runtime_version() override { return 23102U; }
  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      void** communicator, std::span<const std::byte>, std::uint32_t,
      std::uint32_t rank, const DeepSeekNcclReleaseConfig&) override {
    user_rank = rank;
    *communicator = &handle;
    return init_result;
  }
  Result<DeepSeekNcclAsyncStatus> async_status(void*) override {
    return async_result;
  }
  Result<std::uint32_t> communicator_count(void*) override { return 2U; }
  Result<std::uint32_t> communicator_user_rank(void*) override {
    return user_rank;
  }
  Result<DeepSeekNcclAsyncStatus> finalize(void*) override {
    ++finalize_calls;
    return finalize_result;
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
  std::uint32_t user_rank = 0;
  DeepSeekNcclAsyncStatus init_result = DeepSeekNcclAsyncStatus::kSuccess;
  DeepSeekNcclAsyncStatus async_result = DeepSeekNcclAsyncStatus::kSuccess;
  DeepSeekNcclAsyncStatus finalize_result =
      DeepSeekNcclAsyncStatus::kSuccess;
  int finalize_calls = 0;
  int destroy_calls = 0;
  int abort_calls = 0;
};

class AssemblyDeviceAllocator final : public Allocator {
 public:
  explicit AssemblyDeviceAllocator(std::int32_t rank)
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

class AssemblyRuntime final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,
                                                std::uint32_t) override {
    return 77;
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override { return 88; }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t) override { return next_event_++; }
  void destroy_event(DriverEventHandle) noexcept override {}
  void destroy_stream(DriverStreamHandle) noexcept override {}
  void release_primary_context(std::int32_t,
                               std::uintptr_t) noexcept override {}
 private:
  DriverEventHandle next_event_ = 100;
};

class AssemblyClock final : public DeepSeekBoundaryPreparationClock {
 public:
  Result<std::uint64_t> now_ns() override { return 100; }
};

class AssemblyDriverFactory final : public DeepSeekBoundaryDriverOwnerFactory {
 public:
  Result<std::unique_ptr<DeepSeekBoundaryDriverOwner>> create(
      DeepSeekBoundaryTransportDriver&) override {
    return Status::Internal("not exercised by assembly ownership test");
  }
};

class AssemblyDependencyOwner final
    : public DeepSeekNcclGenerationDependencyOwner {
 public:
  explicit AssemblyDependencyOwner(bool& destroyed) : destroyed_(&destroyed) {}
  ~AssemblyDependencyOwner() override { *destroyed_ = true; }
 private:
  bool* destroyed_;
};

DeepSeekNcclCommunicatorManifest manifest(std::uint32_t local_rank,
                                          std::uint32_t edge_id = 1) {
  const bool lower = local_rank == 0;
  return {.engine_epoch = 3,
          .communicator_generation = 4,
          .bootstrap_lease_id = 5 + edge_id,
          .bootstrap_commitment_id = 6,
          .edge_id = edge_id,
          .local_global_rank = lower ? edge_id : edge_id + 1,
          .peer_global_rank = lower ? edge_id + 1 : edge_id,
          .communicator_local_rank = local_rank,
          .device_identity = lower ? 7U : 8U,
          .context_identity = lower ? 17U : 18U,
          .config_identity = 9};
}

std::unique_ptr<DeepSeekNcclEdgeEndpoint> endpoint(
    PairAllocator& allocator, PairApi& api, std::uint32_t local_rank,
    std::uint64_t commitment = 6, std::uint32_t edge_id = 1) {
  std::array<std::byte, 128> raw{};
  raw[0] = std::byte{7};
  auto lease = DeepSeekNcclBootstrapLease::Create(
      allocator, raw, 3, edge_id, 5 + edge_id).value();
  auto config = DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm89, 23102, 128).value();
  auto identity = manifest(local_rank, edge_id);
  identity.bootstrap_commitment_id = commitment;
  auto value = DeepSeekNcclEdgeEndpoint::Create(
      api, std::make_unique<DeepSeekNcclBootstrapLease>(std::move(lease)),
      config, identity).value();
  return std::make_unique<DeepSeekNcclEdgeEndpoint>(std::move(value));
}

TEST(DeepSeekNcclEdgePairTest,
     ReconcilesSealsAndCleanlyDestroysBothEndpoints) {
  PairAllocator lower_allocator;
  PairAllocator upper_allocator;
  PairApi lower_api;
  PairApi upper_api;
  auto pair = DeepSeekNcclEdgePair::Create(
      endpoint(lower_allocator, lower_api, 0),
      endpoint(upper_allocator, upper_api, 1));
  ASSERT_TRUE(pair.ok()) << pair.status().message();
  ASSERT_TRUE(pair->begin_init().ok());
  EXPECT_EQ(pair->state(), DeepSeekNcclEdgePairState::kReconciled);
  ASSERT_TRUE(pair->seal({3, 4, 1, true, true}).ok());
  EXPECT_NE(pair->lower_transport(), nullptr);
  EXPECT_NE(pair->upper_transport(), nullptr);
  ASSERT_TRUE(pair->begin_finalize().ok());
  ASSERT_TRUE(pair->destroy().ok());
  EXPECT_EQ(pair->state(), DeepSeekNcclEdgePairState::kDestroyed);
  EXPECT_EQ(lower_api.destroy_calls, 1);
  EXPECT_EQ(upper_api.destroy_calls, 1);
  EXPECT_EQ(lower_api.abort_calls, 0);
  EXPECT_EQ(upper_api.abort_calls, 0);
}

TEST(DeepSeekNcclEdgePairTest,
     PollsBothAsyncEndpointsAndAbortsPairOnWarmupFailure) {
  PairAllocator lower_allocator;
  PairAllocator upper_allocator;
  PairApi lower_api;
  PairApi upper_api;
  lower_api.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  upper_api.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  lower_api.async_result = DeepSeekNcclAsyncStatus::kInProgress;
  upper_api.async_result = DeepSeekNcclAsyncStatus::kInProgress;
  auto pair = DeepSeekNcclEdgePair::Create(
      endpoint(lower_allocator, lower_api, 0),
      endpoint(upper_allocator, upper_api, 1)).value();
  EXPECT_EQ(pair.begin_init().code(), StatusCode::kUnavailable);
  lower_api.async_result = DeepSeekNcclAsyncStatus::kSuccess;
  upper_api.async_result = DeepSeekNcclAsyncStatus::kSuccess;
  ASSERT_TRUE(pair.poll_init().ok());
  EXPECT_FALSE(pair.seal({3, 4, 1, true, false}).ok());
  EXPECT_EQ(pair.state(), DeepSeekNcclEdgePairState::kAborted);
  EXPECT_EQ(lower_api.abort_calls, 1);
  EXPECT_EQ(upper_api.abort_calls, 1);
}

TEST(DeepSeekNcclEdgePairTest, RejectsMixedBootstrapCommitments) {
  PairAllocator lower_allocator;
  PairAllocator upper_allocator;
  PairApi lower_api;
  PairApi upper_api;
  auto pair = DeepSeekNcclEdgePair::Create(
      endpoint(lower_allocator, lower_api, 0, 6),
      endpoint(upper_allocator, upper_api, 1, 7));
  EXPECT_FALSE(pair.ok());
}

TEST(DeepSeekNcclEdgePairTest, SurvivesMoveWithoutAbortingLiveEndpoints) {
  PairAllocator lower_allocator;
  PairAllocator upper_allocator;
  PairApi lower_api;
  PairApi upper_api;
  auto created = DeepSeekNcclEdgePair::Create(
      endpoint(lower_allocator, lower_api, 0),
      endpoint(upper_allocator, upper_api, 1));
  ASSERT_TRUE(created.ok());
  DeepSeekNcclEdgePair moved(std::move(*created));
  EXPECT_EQ(lower_api.abort_calls, 0);
  EXPECT_EQ(upper_api.abort_calls, 0);
  ASSERT_TRUE(moved.begin_init().ok());
  ASSERT_TRUE(moved.seal({3, 4, 1, true, true}).ok());
}

TEST(DeepSeekNcclCommunicatorGenerationTest,
     SerializesEdgesPublishesTransportsAndDestroysInReverse) {
  PairAllocator a0_lower;
  PairAllocator a0_upper;
  PairAllocator a1_lower;
  PairAllocator a1_upper;
  PairApi api0_lower;
  PairApi api0_upper;
  PairApi api1_lower;
  PairApi api1_upper;
  std::vector<DeepSeekNcclEdgePair> edges;
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(a0_lower, api0_lower, 0, 6, 0),
      endpoint(a0_upper, api0_upper, 1, 6, 0)).value());
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(a1_lower, api1_lower, 0, 7, 1),
      endpoint(a1_upper, api1_upper, 1, 7, 1)).value());
  auto generation = DeepSeekNcclCommunicatorGeneration::Create(
      {3, 4, 3, 30, 60}, std::move(edges));
  ASSERT_TRUE(generation.ok()) << generation.status().message();
  ASSERT_TRUE(generation->begin_init(100).ok());
  EXPECT_EQ(generation->state(), DeepSeekNcclGenerationState::kReconciled);
  EXPECT_NE(generation->outgoing_warmup_transport(0), nullptr);
  EXPECT_NE(generation->incoming_warmup_transport(1), nullptr);
  EXPECT_EQ(generation->incoming_transport(1), nullptr);
  const std::array<DeepSeekNcclWarmupReceipt, 2> receipts{{
      {3, 4, 0, true, true}, {3, 4, 1, true, true}}};
  ASSERT_TRUE(generation->seal(receipts).ok());
  EXPECT_EQ(generation->outgoing_warmup_transport(0), nullptr);
  EXPECT_EQ(generation->incoming_warmup_transport(1), nullptr);
  EXPECT_NE(generation->outgoing_transport(0), nullptr);
  EXPECT_NE(generation->incoming_transport(1), nullptr);
  EXPECT_NE(generation->outgoing_transport(1), nullptr);
  EXPECT_NE(generation->incoming_transport(2), nullptr);
  EXPECT_EQ(generation->incoming_transport(0), nullptr);
  EXPECT_EQ(generation->outgoing_transport(2), nullptr);
  ASSERT_TRUE(generation->begin_teardown(200).ok());
  EXPECT_EQ(generation->state(), DeepSeekNcclGenerationState::kDestroyed);
  EXPECT_EQ(api0_lower.destroy_calls + api0_upper.destroy_calls, 2);
  EXPECT_EQ(api1_lower.destroy_calls + api1_upper.destroy_calls, 2);
}

TEST(DeepSeekNcclCommunicatorGenerationTest,
     AbsoluteEdgeDeadlineAbortsWholeGeneration) {
  PairAllocator lower_allocator;
  PairAllocator upper_allocator;
  PairApi lower_api;
  PairApi upper_api;
  lower_api.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  upper_api.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  lower_api.async_result = DeepSeekNcclAsyncStatus::kInProgress;
  upper_api.async_result = DeepSeekNcclAsyncStatus::kInProgress;
  std::vector<DeepSeekNcclEdgePair> edges;
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(lower_allocator, lower_api, 0, 6, 0),
      endpoint(upper_allocator, upper_api, 1, 6, 0)).value());
  auto generation = DeepSeekNcclCommunicatorGeneration::Create(
      {3, 4, 2, 30, 30}, std::move(edges)).value();
  EXPECT_EQ(generation.begin_init(100).code(), StatusCode::kUnavailable);
  EXPECT_EQ(generation.advance_init(130).code(),
            StatusCode::kDeadlineExceeded);
  EXPECT_EQ(generation.state(), DeepSeekNcclGenerationState::kAborted);
  EXPECT_EQ(generation.advance_init(131).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(lower_api.abort_calls, 1);
  EXPECT_EQ(upper_api.abort_calls, 1);
}

TEST(DeepSeekNcclCommunicatorGenerationTest,
     TeardownDeadlineIsTerminalRatherThanRetryable) {
  PairAllocator lower_allocator;
  PairAllocator upper_allocator;
  PairApi lower_api;
  PairApi upper_api;
  std::vector<DeepSeekNcclEdgePair> edges;
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(lower_allocator, lower_api, 0, 6, 0),
      endpoint(upper_allocator, upper_api, 1, 6, 0)).value());
  auto generation = DeepSeekNcclCommunicatorGeneration::Create(
      {3, 4, 2, 30, 30}, std::move(edges)).value();
  ASSERT_TRUE(generation.begin_init(100).ok());
  const std::array<DeepSeekNcclWarmupReceipt, 1> receipts{{
      {3, 4, 0, true, true}}};
  ASSERT_TRUE(generation.seal(receipts).ok());
  lower_api.finalize_result = DeepSeekNcclAsyncStatus::kInProgress;
  upper_api.finalize_result = DeepSeekNcclAsyncStatus::kInProgress;
  lower_api.async_result = DeepSeekNcclAsyncStatus::kInProgress;
  upper_api.async_result = DeepSeekNcclAsyncStatus::kInProgress;
  EXPECT_EQ(generation.begin_teardown(200).code(), StatusCode::kUnavailable);
  EXPECT_EQ(generation.advance_teardown(230).code(),
            StatusCode::kDeadlineExceeded);
  EXPECT_EQ(generation.state(), DeepSeekNcclGenerationState::kAborted);
  EXPECT_EQ(generation.advance_teardown(231).code(),
            StatusCode::kFailedPrecondition);
}

TEST(DeepSeekNcclCommunicatorGenerationTest,
     RejectsReusedBootstrapCapabilityAcrossEdges) {
  PairAllocator a0_lower;
  PairAllocator a0_upper;
  PairAllocator a1_lower;
  PairAllocator a1_upper;
  PairApi api0_lower;
  PairApi api0_upper;
  PairApi api1_lower;
  PairApi api1_upper;
  std::vector<DeepSeekNcclEdgePair> edges;
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(a0_lower, api0_lower, 0, 6, 0),
      endpoint(a0_upper, api0_upper, 1, 6, 0)).value());
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(a1_lower, api1_lower, 0, 6, 1),
      endpoint(a1_upper, api1_upper, 1, 6, 1)).value());
  auto generation = DeepSeekNcclCommunicatorGeneration::Create(
      {3, 4, 3, 30, 60}, std::move(edges));
  EXPECT_FALSE(generation.ok());
}

TEST(DeepSeekNcclGenerationBoundaryAssemblyTest,
     OwnsSealedGenerationBehindEveryRankFactory) {
  PairAllocator a0_lower;
  PairAllocator a0_upper;
  PairAllocator a1_lower;
  PairAllocator a1_upper;
  PairApi api0_lower;
  PairApi api0_upper;
  PairApi api1_lower;
  PairApi api1_upper;
  std::vector<DeepSeekNcclEdgePair> edges;
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(a0_lower, api0_lower, 0, 6, 0),
      endpoint(a0_upper, api0_upper, 1, 6, 0)).value());
  edges.push_back(DeepSeekNcclEdgePair::Create(
      endpoint(a1_lower, api1_lower, 0, 7, 1),
      endpoint(a1_upper, api1_upper, 1, 7, 1)).value());
  auto generation = DeepSeekNcclCommunicatorGeneration::Create(
      {3, 4, 3, 30, 60}, std::move(edges)).value();
  ASSERT_TRUE(generation.begin_init(100).ok());
  const std::array<DeepSeekNcclWarmupReceipt, 2> receipts{{
      {3, 4, 0, true, true}, {3, 4, 1, true, true}}};
  ASSERT_TRUE(generation.seal(receipts).ok());

  AssemblyRuntime runtime;
  std::array<AssemblyDeviceAllocator, 3> allocators{{
      AssemblyDeviceAllocator(0), AssemblyDeviceAllocator(1),
      AssemblyDeviceAllocator(2)}};
  std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>> resources;
  std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks;
  std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>> factories;
  std::vector<DeepSeekRankBoundaryFactoryIdentity> identities;
  for (std::uint32_t rank = 0; rank < 3; ++rank) {
    resources.push_back(DeepSeekRankBoundaryDeviceResources::Allocate(
        rank, 3, 2, 77, allocators[rank], runtime).value());
    clocks.push_back(std::make_unique<AssemblyClock>());
    factories.push_back(std::make_unique<AssemblyDriverFactory>());
    identities.push_back(
        {3, rank, 3, 77, 88, 4, 1000,
         rank == 0 ? std::array<std::uint64_t, 2>{0, 0}
                   : std::array<std::uint64_t, 2>{100 + rank * 2,
                                                  101 + rank * 2}});
  }
  bool dependencies_destroyed = false;
  auto assembly =
      DeepSeekNcclGenerationBoundaryAssembly::CreateWithDependencies(
      std::move(generation), std::move(identities), std::move(resources),
      std::move(clocks), std::move(factories),
      std::make_unique<AssemblyDependencyOwner>(dependencies_destroyed));
  ASSERT_TRUE(assembly.ok()) << assembly.status().message();
  EXPECT_NE((*assembly)->factory(0), nullptr);
  EXPECT_NE((*assembly)->factory(1), nullptr);
  EXPECT_NE((*assembly)->factory(2), nullptr);
  EXPECT_EQ((*assembly)->generation_state(),
            DeepSeekNcclGenerationState::kSealed);
  EXPECT_FALSE(dependencies_destroyed);
  ASSERT_TRUE((*assembly)->advance_clean_teardown().ok());
  EXPECT_EQ((*assembly)->generation_state(),
            DeepSeekNcclGenerationState::kDestroyed);
  assembly->reset();
  EXPECT_TRUE(dependencies_destroyed);
  EXPECT_EQ(api0_lower.abort_calls + api0_upper.abort_calls, 0);
  EXPECT_EQ(api1_lower.abort_calls + api1_upper.abort_calls, 0);
  EXPECT_EQ(api0_lower.destroy_calls + api0_upper.destroy_calls, 2);
  EXPECT_EQ(api1_lower.destroy_calls + api1_upper.destroy_calls, 2);
}

}  // namespace
}  // namespace pih
