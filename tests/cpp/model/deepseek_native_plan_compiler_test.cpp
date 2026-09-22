#include "pih/model/deepseek_native_plan_compiler.h"
#include "pih/model/deepseek_deferred_native_plan_compiler.h"
#include "pih/model/deepseek_production_rank_plan_input_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Operations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status gemm(DeepSeekRouterBf16GemmLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class DeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("device allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Transfer final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity, std::uint32_t, std::uint64_t,
               std::uint64_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity, std::uint64_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Kernel final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease&, const DeepSeekExpertRoute*,
                std::uint32_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

template <typename T>
T* fake(std::uintptr_t value) { return reinterpret_cast<T*>(value); }

DeepSeekRankPlanCompiler make_compiler(
    DeepSeekStagePlan stage, DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations) {
  std::vector<DeepSeekHashRouterLayerTable> tables;
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= std::min(2U, stage.layers.last_layer); ++layer) {
    tables.push_back({layer, std::vector<std::uint16_t>(
        DeepSeekExpertSubwavePlan::kRoutesPerToken)});
  }
  auto hash = DeepSeekHashRouterWorkFactory::Create(
      stage.layers, 1, 1, std::move(tables)).value();
  std::array<float, 256> bias{};
  std::vector<DeepSeekLearnedRouterLayerBias> biases;
  for (std::uint32_t layer = std::max(3U, stage.layers.first_layer);
       layer <= stage.layers.last_layer; ++layer) {
    biases.push_back({layer, bias});
  }
  auto learned = DeepSeekLearnedRouterWorkFactory::Create(
      stage.layers, 1, std::move(biases), store, operations).value();
  std::vector<DeepSeekAttentionLayerRuntimeResources> attention_resources;
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    attention_resources.push_back({
        layer, fake<DeepSeekRecentStateWriter>(0x10000 + layer * 0x100),
        fake<DeepSeekCompressedLayerUpdateCoordinator>(0x20000 + layer * 0x100),
        fake<DeepSeekAttentionLayerCoordinator>(0x30000 + layer * 0x100),
        fake<DeepSeekPrefillLayerCoordinator>(0x40000 + layer * 0x100)});
  }
  auto attention = DeepSeekAttentionWorkFactory::Create(
      stage.layers,
      std::move(attention_resources)).value();
  auto dense = DeepSeekDenseMhcWorkFactory::Create(
      stage.layers, 1, 1).value();
  auto endpoint = DeepSeekEndpointWorkFactory::Create(stage, 1).value();
  return DeepSeekRankPlanCompiler::Create(
      stage, std::move(hash), std::move(learned), std::move(attention),
      std::move(dense), std::move(endpoint)).value();
}

DeepSeekRankPlanCompilerInput make_input(
    PinnedAllocator& allocator, DeepSeekRankAttentionStatePool& state_pool,
    DeepSeekStagePlan stage = {0, {0, 42}, true, true, false}) {
  DeepSeekRankPlanCompilerInput input;
  input.token_ids = {0};
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= std::min(2U, stage.layers.last_layer); ++layer) {
    input.hash_router_scores.push_back({layer, std::vector<float>(256)});
  }
  for (std::uint32_t layer = std::max(3U, stage.layers.first_layer);
       layer <= stage.layers.last_layer; ++layer) {
    DeepSeekLearnedRouterSubmission submission;
    submission.layer = layer;
    submission.token_count = 1;
    submission.input_bf16 = 1;
    submission.weight_bf16 = 2;
    submission.scores_f32 = 6;
    submission.error_flag_u32 = 7;
    submission.stream = 8;
    submission.completion_event = 9;
    auto staging = DeepSeekLearnedRouterHostStaging::Allocate(1, allocator);
    input.learned_router.push_back({
        submission, std::make_shared<DeepSeekLearnedRouterHostStaging>(
                        std::move(*staging))});
  }
  std::vector<std::uint32_t> positions{0};
  std::vector<std::uint32_t> visible{0};
  auto* transaction = fake<DeepSeekAttentionSequenceTransaction>(0x50000);
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    DeepSeekDecodeAttentionLayerPlanInput attention;
    attention.layer = layer;
    attention.attention.query_positions = positions;
    attention.attention.selection.visible_slot_counts = visible;
    attention.transaction = transaction;
    input.decode_attention.push_back(attention);

    DeepSeekDenseAttentionStageSequenceWork dense;
    dense.input_coordinator = fake<DeepSeekAttentionProjectionCoordinator>(
        0x60000 + layer * 0x100);
    dense.output_coordinator =
        fake<DeepSeekAttentionOutputProjectionCoordinator>(
            0x70000 + layer * 0x100);
    dense.transaction = transaction;
    dense.input.input_quant.token_count = 1;
    dense.sparse_query_bf16 = 1;
    dense.sparse_kv_bf16 = 2;
    dense.sparse_output_bf16 = 3;
    input.dense_attention.push_back({layer, {dense}});

    DeepSeekMhcStageSequenceWork mhc;
    mhc.executor = fake<DeepSeekMhcSequenceExecutor>(0x80000 + layer * 0x100);
    mhc.transaction = transaction;
    mhc.submission.kind = DeepSeekMhcBranchKind::kAttention;
    mhc.submission.layer_id = layer;
    mhc.submission.token_count = 1;
    input.mhc_attention.push_back({layer, {mhc}});
    mhc.executor = fake<DeepSeekMhcSequenceExecutor>(0x90000 + layer * 0x100);
    mhc.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
    input.mhc_feed_forward.push_back({layer, {mhc}});
  }
  if (stage.owns_embedding || stage.owns_lm_head) {
    DeepSeekEndpointStageSequenceWork endpoint;
    endpoint.executor = fake<DeepSeekEndpointSequenceExecutor>(0xA0000);
    endpoint.transaction = transaction;
    input.endpoint.push_back(endpoint);
  }
  input.attention_state_pool = &state_pool;
  input.attention_bindings = {{91, 0}};
  return input;
}

class DeferredInputAssembler final
    : public DeepSeekDeferredRankPlanInputAssembler {
 public:
  explicit DeferredInputAssembler(
      DeepSeekRankPlanCompilerInput input,
      std::optional<DeepSeekBoundarySendSource> outgoing = std::nullopt)
      : input_(std::move(input)), outgoing_(std::move(outgoing)) {}
  Result<std::optional<DeepSeekBoundarySendSource>> outgoing_source()
      override {
    return outgoing_;
  }
  Result<DeepSeekRankPlanCompilerInput> assemble(
      std::uintptr_t incoming_activation_bf16) override {
    ++calls;
    incoming = incoming_activation_bf16;
    input_.outgoing_boundary = outgoing_;
    return std::move(input_);
  }
  int calls = 0;
  std::uintptr_t incoming = UINTPTR_MAX;
 private:
  DeepSeekRankPlanCompilerInput input_;
  std::optional<DeepSeekBoundarySendSource> outgoing_;
};

TEST(DeepSeekNativePlanCompilerTest,
     DeferredCompilerPropagatesCanonicalPpOneThroughFourIncomingAddresses) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto topology = DeepSeekPipelinePlan::Create(world_size, false).value();
    Operations operations;
    Transfer transfer;
    Kernel kernel;
    PinnedAllocator pinned_allocator;
    DeviceAllocator device_allocator;
    FixedOperations fixed_operations;
    std::vector<std::unique_ptr<DeepSeekExpertPager>> pagers;
    std::vector<std::unique_ptr<DeepSeekRankComputeBundle>> bundles;
    std::vector<std::unique_ptr<DeepSeekRankAttentionStatePool>> state_pools;
    std::vector<std::unique_ptr<Buffer>> boundary_buffers;
    std::vector<DeepSeekRankPlanCompiler> compilers;
    std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
        assemblers;
    std::vector<DeferredInputAssembler*> observed;
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      const auto stage = topology.rank(rank);
      auto pager = std::make_unique<DeepSeekExpertPager>(
          DeepSeekExpertPager::Create(stage.layers, 2, 2).value());
      auto bundle = std::make_unique<DeepSeekRankComputeBundle>(
          DeepSeekRankComputeBundle::Create(
              stage, 1, 1, *pager, transfer, kernel, 23).value());
      auto state_pool = std::make_unique<DeepSeekRankAttentionStatePool>(
          DeepSeekRankAttentionStatePool::Allocate(
              device_allocator, stage, 1, 1, 19, fixed_operations,
              17 + rank, 0).value());
      compilers.push_back(make_compiler(
          stage, bundle->expert_store(), operations));
      std::optional<DeepSeekBoundarySendSource> outgoing;
      if (rank + 1 < world_size) {
        auto buffer = std::make_unique<Buffer>(Buffer::Allocate(
            device_allocator, DeepSeekBoundarySendSource::kWireBytesPerToken,
            256).value());
        outgoing = DeepSeekBoundarySendSource::Create(
            *buffer, 0, DeepSeekBoundarySendSource::kWireBytesPerToken,
            1, 100 + rank, 19, 200 + rank).value();
        boundary_buffers.push_back(std::move(buffer));
      }
      auto assembler = std::make_unique<DeferredInputAssembler>(
          make_input(pinned_allocator, *state_pool, stage), outgoing);
      observed.push_back(assembler.get());
      assemblers.push_back(std::move(assembler));
      pagers.push_back(std::move(pager));
      bundles.push_back(std::move(bundle));
      state_pools.push_back(std::move(state_pool));
    }
    auto compiler = DeepSeekDeferredNativePlanCompiler::Create(
        topology, {11, world_size, DeepSeekPlanPhase::kDecode, 1, 1},
        std::move(compilers), std::move(assemblers)).value();
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      EXPECT_EQ(compiler.outgoing_source(rank).value().has_value(),
                rank + 1 < world_size);
      const auto incoming = rank == 0 ? 0U : 0xABC000U + rank * 0x10000U;
      auto work = compiler.compile(rank, incoming);
      ASSERT_TRUE(work.ok()) << "PP" << world_size << " rank " << rank
                             << ": " << work.status().message();
      EXPECT_EQ(observed[rank]->incoming, incoming);
      EXPECT_EQ(work->outgoing_boundary.has_value(),
                rank + 1 < world_size);
    }
  }
}

TEST(DeepSeekNativePlanCompilerTest,
     DeferredCompilerOwnsDescriptorAndCompilesRankOnlyOnce) {
  auto topology = DeepSeekPipelinePlan::Create(1, false).value();
  Operations operations;
  auto pager = DeepSeekExpertPager::Create(
      topology.rank(0).layers, 2, 2).value();
  Transfer transfer;
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::Create(
      topology.rank(0), 1, 1, pager, transfer, kernel, 23).value();
  PinnedAllocator allocator;
  DeviceAllocator device_allocator;
  FixedOperations fixed_operations;
  auto state_pool = DeepSeekRankAttentionStatePool::Allocate(
      device_allocator, topology.rank(0), 1, 1, 19, fixed_operations,
      17, 0).value();
  std::vector<DeepSeekRankPlanCompiler> ranks;
  ranks.push_back(make_compiler(
      topology.rank(0), bundle.expert_store(), operations));
  auto assembler = std::make_unique<DeferredInputAssembler>(
      make_input(allocator, state_pool));
  auto* observed = assembler.get();
  std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
      assemblers;
  assemblers.push_back(std::move(assembler));
  auto compiler = DeepSeekDeferredNativePlanCompiler::Create(
      topology, {11, 7, DeepSeekPlanPhase::kDecode, 1, 1},
      std::move(ranks), std::move(assemblers)).value();
  EXPECT_FALSE(compiler.outgoing_source(0).value().has_value());
  auto work = compiler.compile(0, 0);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_EQ(observed->incoming, 0U);
  EXPECT_EQ(work->decode_attention.size(), 43U);
  EXPECT_FALSE(compiler.compile(0, 0).ok());
  EXPECT_FALSE(compiler.compile(1, 0).ok());
}

TEST(DeepSeekNativePlanCompilerTest,
     DeferredRankFailurePoisonsWholeCompilerWithoutReconsumingAssemblers) {
  auto topology = DeepSeekPipelinePlan::Create(2, false).value();
  Operations operations;
  Transfer transfer;
  Kernel kernel;
  PinnedAllocator allocator;
  DeviceAllocator device_allocator;
  FixedOperations fixed_operations;
  std::vector<std::unique_ptr<DeepSeekExpertPager>> pagers;
  std::vector<std::unique_ptr<DeepSeekRankComputeBundle>> bundles;
  std::vector<std::unique_ptr<DeepSeekRankAttentionStatePool>> state_pools;
  std::vector<DeepSeekRankPlanCompiler> compilers;
  std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
      assemblers;
  std::vector<DeferredInputAssembler*> observed;
  for (std::uint32_t rank = 0; rank < 2; ++rank) {
    const auto stage = topology.rank(rank);
    auto pager = std::make_unique<DeepSeekExpertPager>(
        DeepSeekExpertPager::Create(stage.layers, 2, 2).value());
    auto bundle = std::make_unique<DeepSeekRankComputeBundle>(
        DeepSeekRankComputeBundle::Create(
            stage, 1, 1, *pager, transfer, kernel, 23).value());
    auto state_pool = std::make_unique<DeepSeekRankAttentionStatePool>(
        DeepSeekRankAttentionStatePool::Allocate(
            device_allocator, stage, 1, 1, 19, fixed_operations,
            17 + rank, 0).value());
    compilers.push_back(make_compiler(
        stage, bundle->expert_store(), operations));
    auto input = make_input(allocator, *state_pool, stage);
    if (rank == 0) input.token_ids.clear();
    auto assembler = std::make_unique<DeferredInputAssembler>(
        std::move(input));
    observed.push_back(assembler.get());
    assemblers.push_back(std::move(assembler));
    pagers.push_back(std::move(pager));
    bundles.push_back(std::move(bundle));
    state_pools.push_back(std::move(state_pool));
  }
  auto compiler = DeepSeekDeferredNativePlanCompiler::Create(
      topology, {11, 8, DeepSeekPlanPhase::kDecode, 1, 1},
      std::move(compilers), std::move(assemblers)).value();
  const auto first = compiler.compile(0, 0);
  ASSERT_FALSE(first.ok());
  const auto repeated = compiler.compile(0, 0);
  EXPECT_FALSE(repeated.ok());
  EXPECT_EQ(repeated.status().code(), first.status().code());
  EXPECT_EQ(repeated.status().message(), first.status().message());
  EXPECT_FALSE(compiler.compile(1, 0xABC000).ok());
  EXPECT_EQ(observed[0]->calls, 1);
  EXPECT_EQ(observed[1]->calls, 0);
}

TEST(DeepSeekNativePlanCompilerTest,
     CompilesCanonicalPp1FortyThreeLayerDecodePlan) {
  auto topology = DeepSeekPipelinePlan::Create(1, false).value();
  Operations operations;
  auto pager_result = DeepSeekExpertPager::Create(
      topology.rank(0).layers, 2, 2);
  ASSERT_TRUE(pager_result.ok()) << pager_result.status().message();
  auto pager = std::move(*pager_result);
  Transfer transfer;
  Kernel kernel;
  auto bundle_result = DeepSeekRankComputeBundle::Create(
      topology.rank(0), 1, 1, pager, transfer, kernel, 23);
  ASSERT_TRUE(bundle_result.ok()) << bundle_result.status().message();
  auto bundle = std::move(*bundle_result);
  std::vector<DeepSeekRankPlanCompiler> ranks;
  ranks.push_back(make_compiler(
      topology.rank(0), bundle.expert_store(), operations));
  auto compiler = DeepSeekNativePlanCompiler::Create(
      topology, std::move(ranks)).value();
  EXPECT_EQ(compiler.topology().world_size(), 1U);
  PinnedAllocator allocator;
  DeviceAllocator device_allocator;
  FixedOperations fixed_operations;
  auto state_pool = DeepSeekRankAttentionStatePool::Allocate(
      device_allocator, topology.rank(0), 1, 1, 19, fixed_operations,
      17, 0).value();
  std::vector<DeepSeekRankPlanCompilerInput> input;
  input.push_back(make_input(allocator, state_pool));
  auto plan = compiler.compile(
      {11, 1, DeepSeekPlanPhase::kDecode, 1, 1}, std::move(input));
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->rank_work().size(), 1U);
  EXPECT_EQ(plan->rank_work()[0].decode_attention.size(), 43U);
  EXPECT_NE(plan->rank_work()[0].lifetime_owner, nullptr);
  auto* owned_transaction =
      plan->rank_work()[0].decode_attention[0].work.transaction;
  EXPECT_NE(owned_transaction,
            fake<DeepSeekAttentionSequenceTransaction>(0x50000));
  EXPECT_EQ(plan->rank_work()[0].decode_attention[42].work.transaction,
            owned_transaction);
  EXPECT_EQ(plan->rank_work()[0].endpoint[0].transaction,
            owned_transaction);

  auto rank_work = std::move(*plan).release_rank_work();
  auto bind_status = bundle.bind_plan(
      {11, 1, DeepSeekPlanPhase::kDecode, 1, 1},
      std::move(rank_work[0]));
  ASSERT_TRUE(bind_status.ok()) << bind_status.message();
  EXPECT_TRUE(bundle.attention_transaction_driver_bound());
  EXPECT_EQ(bundle.attention_compute_stream(), 23U);
  EXPECT_EQ(owned_transaction->state(),
            DeepSeekAttentionSequenceTransactionState::kIdle);
}

TEST(DeepSeekNativePlanCompilerTest,
     ReleasesRankCompilersOnlyThroughRvalueOwnershipTransfer) {
  auto topology = DeepSeekPipelinePlan::Create(1, false).value();
  Operations operations;
  auto pager = DeepSeekExpertPager::Create(
      topology.rank(0).layers, 2, 2).value();
  Transfer transfer;
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::Create(
      topology.rank(0), 1, 1, pager, transfer, kernel, 23).value();
  std::vector<DeepSeekRankPlanCompiler> ranks;
  ranks.push_back(make_compiler(
      topology.rank(0), bundle.expert_store(), operations));
  auto compiler = DeepSeekNativePlanCompiler::Create(
      topology, std::move(ranks)).value();
  auto released = std::move(compiler).release_rank_compilers();
  ASSERT_EQ(released.size(), 1U);
  EXPECT_EQ(released[0].stage().rank, 0U);
}

TEST(DeepSeekNativePlanCompilerTest,
     PublishesResolvedProductionInputWithPinnedLeaseOwnership) {
  PinnedAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 256, 256).value();
  auto lease_owner = std::make_shared<Buffer>(std::move(buffer));
  DeepSeekProductionRankPlanSeed seed;
  seed.token_ids = {17};
  seed.positions = {9};
  seed.attention_binding = {55, 2};
  DeepSeekAttentionLayerPlanShapeSeed shape;
  shape.layer = 10;
  shape.ratio = 4;
  shape.positions = {9};
  shape.recent = {{10, 20, 12, 1, 9}};
  DeepSeekCompressedLayerUpdateSubmission update;
  update.ratio = 4;
  update.main_state.layer_id = 10;
  update.main_state.absolute_position = 9;
  update.main_state.stream = 12;
  shape.updates.push_back(update);
  shape.index_query_bf16 = 1;
  shape.index_kv_bf16 = 2;
  shape.index_head_weight_f32 = 3;
  shape.index_arena = {4, 5};
  shape.has_indexer_projection = true;
  shape.indexer_projection =
      {13, 14, 15, 23, 24, 25, 16, 17, 18, 1, 3, 19, 12, 1, 1048576};
  shape.sparse_query_bf16 = 6;
  shape.sparse_kv_bf16 = 7;
  shape.attention_sink_f32 = 8;
  shape.sparse_indices_i32 = 9;
  shape.sparse_page_slots_u32 = 21;
  shape.sparse_output_bf16 = 10;
  shape.sparse_error_u32 = 11;
  shape.sparse_kv_count = 4096;
  shape.compressed_physical_offset = 1024;
  shape.stream = 12;
  seed.attention_shape_seeds.push_back(std::move(shape));
  DeepSeekProductionRankResolvedInput resolved;
  resolved.stage = {1, {10, 20}, false, false, false};
  resolved.attention_state_pool =
      fake<DeepSeekRankAttentionStatePool>(0x123000);
  resolved.request_input_lease = {lease_owner, 1};
  DeepSeekDenseMhcLayerSubmissionInput dense;
  dense.layer = 10;
  resolved.dense_mhc.layers.push_back(dense);

  auto input = DeepSeekProductionRankPlanInputAssembler::PublishResolved(
      {11, 7, DeepSeekPlanPhase::kDecode, 1, 1}, std::move(seed),
      std::move(resolved));
  ASSERT_TRUE(input.ok()) << input.status().message();
  EXPECT_EQ(input->token_ids, std::vector<std::uint32_t>({17}));
  EXPECT_EQ(input->attention_bindings[0].sequence, 55U);
  EXPECT_EQ(input->attention_bindings[0].state_slot, 2U);
  EXPECT_EQ(input->attention_state_pool,
            fake<DeepSeekRankAttentionStatePool>(0x123000));
  EXPECT_EQ(input->decode_attention.size(), 1U);
  EXPECT_EQ(input->dense_mhc_layers.size(), 1U);
  ASSERT_EQ(input->lifetime_backings.size(), 2U);
  EXPECT_EQ(input->lifetime_backings[0].get(), lease_owner.get());
  EXPECT_EQ(input->decode_attention[0].attention.query_positions[0], 9U);
  EXPECT_EQ(
      input->decode_attention[0].attention.selection.visible_slot_counts[0],
      2U);
  EXPECT_FALSE(input->endpoint.size());
  EXPECT_FALSE(input->dspark.has_value());
  EXPECT_FALSE(input->outgoing_boundary.has_value());
}

TEST(DeepSeekNativePlanCompilerTest,
     RejectsRankCountBeforeCompilingPartialWork) {
  auto topology = DeepSeekPipelinePlan::Create(1, false).value();
  EXPECT_FALSE(DeepSeekNativePlanCompiler::Create(topology, {}).ok());
}

TEST(DeepSeekNativePlanCompilerTest,
     PrefillResolvedInputRequiresAndPublishesThreeDsparkMtpStages) {
  PinnedAllocator allocator;
  auto make = [&]() {
    auto buffer = Buffer::Allocate(allocator, 256, 256).value();
    DeepSeekProductionRankPlanSeed seed;
    seed.token_ids = {17};
    seed.positions = {9};
    seed.attention_binding = {55, 0};
    DeepSeekProductionRankResolvedInput resolved;
    resolved.stage = {0, {0, 42}, true, true, true};
    resolved.attention_state_pool =
        fake<DeepSeekRankAttentionStatePool>(0x123000);
    resolved.request_input_lease = {
        std::make_shared<Buffer>(std::move(buffer)), 1};
    resolved.dense_mhc.layers.push_back({.layer = 0});
    resolved.endpoint.emplace();
    resolved.dspark.emplace();
    return std::pair(std::move(seed), std::move(resolved));
  };

  auto missing = make();
  EXPECT_EQ(DeepSeekProductionRankPlanInputAssembler::PublishResolved(
                {11, 7, DeepSeekPlanPhase::kPrefill, 1, 1},
                std::move(missing.first), std::move(missing.second))
                .status().code(),
            StatusCode::kInvalidArgument);

  auto complete = make();
  complete.second.dspark_mtp.resize(kDeepSeekDsparkStageCount);
  auto input = DeepSeekProductionRankPlanInputAssembler::PublishResolved(
      {11, 8, DeepSeekPlanPhase::kPrefill, 1, 1},
      std::move(complete.first), std::move(complete.second));
  ASSERT_TRUE(input.ok()) << input.status().message();
  EXPECT_EQ(input->dspark_mtp.size(), kDeepSeekDsparkStageCount);
  EXPECT_TRUE(input->dspark.has_value());
}

TEST(DeepSeekNativePlanCompilerTest,
     VerifyOnDraftingStageDoesNotRequireOrPublishDsparkWork) {
  PinnedAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 256, 256).value();
  DeepSeekProductionRankPlanSeed seed;
  seed.token_ids = {17};
  seed.positions = {9};
  seed.attention_binding = {55, 2};
  DeepSeekProductionRankResolvedInput resolved;
  resolved.stage = {0, {0, 42}, true, true, true};
  resolved.attention_state_pool =
      fake<DeepSeekRankAttentionStatePool>(0x123000);
  resolved.request_input_lease = {
      std::make_shared<Buffer>(std::move(buffer)), 1};
  DeepSeekDenseMhcLayerSubmissionInput dense;
  dense.layer = 0;
  resolved.dense_mhc.layers.push_back(dense);
  resolved.endpoint.emplace();

  auto input = DeepSeekProductionRankPlanInputAssembler::PublishResolved(
      {11, 7, DeepSeekPlanPhase::kVerify, 1, 1}, std::move(seed),
      std::move(resolved));
  ASSERT_TRUE(input.ok()) << input.status().message();
  EXPECT_EQ(input->endpoint.size(), 1U);
  EXPECT_FALSE(input->dspark.has_value());
}

}  // namespace
}  // namespace pih
