#include "pih/model/deepseek_rank_engine_resources.h"
#include "pih/model/deepseek_rank_materialization_completion.h"
#include "pih/model/deepseek_engine_resources.h"
#include "pih/model/deepseek_engine.h"
#include "pih/model/deepseek_production_rank_plan_input_assembler.h"
#include "pih/model/deepseek_weight_binding_dry_run.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <new>

namespace pih { namespace {

std::vector<DeepSeekRankTensorRecord> engine_rank_records() {
  std::vector<DeepSeekRankTensorRecord> records;
  std::uint64_t offset = 0;
  for (std::uint32_t expert = 0; expert < 256; ++expert) {
    for (const auto* matrix : {"w1", "w2", "w3"}) {
      const bool w2 = std::string_view(matrix) == "w2";
      const auto prefix = "layers.4.ffn.experts." + std::to_string(expert) +
                          "." + matrix;
      records.push_back({prefix + ".weight", "a.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kInt8,
                         w2 ? std::vector<std::uint64_t>{4096, 1024}
                            : std::vector<std::uint64_t>{2048, 2048},
                         offset, offset + 4194304});
      offset += 4194304;
      records.push_back({prefix + ".scale", "a.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kFloat8E8M0,
                         w2 ? std::vector<std::uint64_t>{4096, 64}
                            : std::vector<std::uint64_t>{2048, 128},
                         offset, offset + 262144});
      offset += 262144;
    }
  }
  records.push_back({"layers.4.attn_norm.weight", "b.safetensors",
                     DeepSeekTensorRole::kMainLayer, DType::kBFloat16,
                     {4096}, 4096, 12288});
  return records;
}

class TestCudaAllocator final : public Allocator {
 public:
  TestCudaAllocator() = default;
  TestCudaAllocator(bool& runtime_alive, bool& released_after_runtime)
      : runtime_alive_(&runtime_alive),
        released_after_runtime_(&released_after_runtime) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("CUDA test allocation failed");
    return Allocation{data, bytes, alignment, 101,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    if (runtime_alive_ != nullptr && !*runtime_alive_) {
      *released_after_runtime_ = true;
    }
    _aligned_free(allocation.data);
  }
 private:
  bool* runtime_alive_ = nullptr;
  bool* released_after_runtime_ = nullptr;
};

class TestRankRuntimeOwner final : public DeepSeekRankRuntimeOwner {
 public:
  explicit TestRankRuntimeOwner(bool& alive, int& activations,
                                bool& activation_fail)
      : alive_(&alive), activations_(&activations),
        activation_fail_(&activation_fail) {
    *alive_ = true;
  }
  ~TestRankRuntimeOwner() override { *alive_ = false; }
  Status activate() override {
    ++*activations_;
    return *activation_fail_
               ? Status::FailedPrecondition(
                     "injected rank runtime activation failure")
               : Status::Ok();
  }
  DeepSeekLearnedRouterOperations* learned_router_operations()
      noexcept override {
    return reinterpret_cast<DeepSeekLearnedRouterOperations*>(0x1234);
  }
 private:
  bool* alive_;
  int* activations_;
  bool* activation_fail_;
};

class TestPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("pinned test allocation failed");
    return Allocation{data, bytes, alignment, 202, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
};

class TestCudaCopier final : public MemoryCopier {
 public:
  Status copy(void* destination, Device destination_device,
              const void* source, Device source_device,
              std::uint64_t bytes) override {
    if (destination_device.type() != DeviceType::kCuda ||
        source_device != Device::Cpu()) {
      return Status::InvalidArgument("rank engine copy direction is invalid");
    }
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return Status::Ok();
  }
};

class TestTransferDriver final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity, std::uint32_t, std::uint64_t,
               std::uint64_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity, std::uint64_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class TestKernelDriver final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease&, const DeepSeekExpertRoute*,
                std::uint32_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class TestTransferOwner final : public DeepSeekExpertTransferLaneOwner {
 public:
  DeepSeekExpertTransferDriver& transfer() noexcept override { return driver_; }
 private:
  TestTransferDriver driver_;
};

class TestKernelOwner final : public DeepSeekExpertKernelLaneOwner {
 public:
  DeepSeekExpertKernelDriver& kernel() noexcept override { return driver_; }
 private:
  TestKernelDriver driver_;
};

class ScriptedArtifactPoller final : public DeepSeekEngineArtifactPoller {
 public:
  std::uint32_t world_size() const noexcept override { return 1; }
  Status poll() override {
    ++calls;
    return fail ? Status::FailedPrecondition("injected artifact poll failure")
                : Status::Ok();
  }
  bool fail = false;
  int calls = 0;
};

class MismatchedDeferredCompiler final
    : public DeepSeekDeferredRankComputePlanCompiler {
 public:
  std::uint32_t world_size() const noexcept override { return 2; }
  Result<std::optional<DeepSeekBoundarySendSource>> outgoing_source(
      std::uint32_t) override {
    return std::optional<DeepSeekBoundarySendSource>{};
  }
  Result<DeepSeekRankComputePlanWork> compile(
      std::uint32_t, std::uintptr_t) override {
    return Status::Internal("mismatched compiler must not be called");
  }
};

void make_sparse_file(const std::filesystem::path& path,
                      std::uint64_t bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.seekp(static_cast<std::streamoff>(bytes - 1U));
  output.put('\0');
}

TEST(DeepSeekRankEngineResourcesTest,
     OwnsCompleteHostSpillRankLifetimeAndReceipt) {
  bool runtime_alive = false;
  bool released_after_runtime = false;
  int runtime_activations = 0;
  bool runtime_activation_fail = false;
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-rank-engine-resources-test";
  std::filesystem::create_directory(root);
  const auto expert_bytes =
      256ULL * DeepSeekExpertBundleLayout::kBundleBytes;
  make_sparse_file(root / "a.safetensors", expert_bytes);
  make_sparse_file(root / "b.safetensors", 12288);
  auto a = ControllerFileLease::OpenBeneath(
      root, "a.safetensors", expert_bytes,
      ArtifactImmutabilityMode::kUncalibrated).value();
  auto b = ControllerFileLease::OpenBeneath(
      root, "b.safetensors", 12288,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("a.safetensors", a.duplicate_for_worker().value());
  descriptors.emplace_back("b.safetensors", b.duplicate_for_worker().value());
  DeepSeekRankMappingPlan mapping;
  mapping.rank = 0;
  mapping.shards = {{"a.safetensors", expert_bytes},
                    {"b.safetensors", 12288}};
  mapping.intervals = {{"a.safetensors", 0, expert_bytes},
                       {"b.safetensors", 4096, 12288}};
  mapping.mapped_interval_bytes = expert_bytes + 8192;
  auto device = std::make_unique<TestCudaAllocator>(
      runtime_alive, released_after_runtime);
  auto pinned = std::make_unique<TestPinnedAllocator>();
  TestCudaCopier copier;
  DeepSeekWeightBindingDryRun dry_run;
  {
    auto resources =
        DeepSeekRankEngineResources::BuildForInProcessDevelopment(
        11, 1, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill,
        std::move(mapping), std::move(descriptors), engine_rank_records(),
        2, 2, DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false).value(),
        *device, *pinned, copier, dry_run);
    ASSERT_TRUE(resources.ok()) << resources.status().message();
    EXPECT_EQ(resources->rank(), 0U);
    EXPECT_EQ(resources->resident_weights().device().type(), DeviceType::kCuda);
    EXPECT_NE(resources->expert_source(), nullptr);
    ASSERT_NE(resources->expert_pager(), nullptr);
    EXPECT_EQ(resources->expert_pager()->slot_count(), 2U);
    EXPECT_FALSE(resources->compute_ready());
    EXPECT_TRUE(resources->verify_weight_seal().ok());
    EXPECT_EQ(compile_deepseek_rank_materialization_completion(
                  *resources,
                  DeepSeekRankMaterializationCompletionObservation{})
                  .status()
                  .code(),
              StatusCode::kFailedPrecondition);
    auto plan_resources = resources->prepare_plan_resources(
        {11, 1, DeepSeekPlanPhase::kDecode, 1, 1});
    ASSERT_TRUE(plan_resources.ok()) << plan_resources.status().message();
    EXPECT_TRUE(plan_resources->commit().ok());
    plan_resources->release();
    auto held_plan_resources = resources->prepare_plan_resources(
        {11, 99, DeepSeekPlanPhase::kDecode, 1, 1});
    ASSERT_TRUE(held_plan_resources.ok());
    std::vector<DeepSeekRankEngineResources> ranks;
    ranks.push_back(std::move(*resources));
    std::vector<std::unique_ptr<Allocator>> device_owners;
    device_owners.push_back(std::move(device));
    std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners;
    runtime_owners.push_back(
        std::make_unique<TestRankRuntimeOwner>(runtime_alive,
                                               runtime_activations,
                                               runtime_activation_fail));
    auto generation = DeepSeekEngineResources::CreateOwnedWithRuntimes(
        11, 1000, std::move(device_owners), std::move(pinned),
        std::move(runtime_owners),
        std::move(ranks));
    ASSERT_TRUE(generation.ok()) << generation.status().message();
    const auto boundary_snapshot = generation->boundary_lifecycle_snapshot();
    EXPECT_EQ(boundary_snapshot.world_size, 1U);
    EXPECT_EQ(boundary_snapshot.expected_edge_count, 0U);
    EXPECT_EQ(boundary_snapshot.sealed_edge_count, 0U);
    EXPECT_FALSE(boundary_snapshot.assembly_present);
    EXPECT_EQ(boundary_snapshot.generation_state,
              DeepSeekBoundaryGenerationState::kNotApplicable);
    EXPECT_EQ(generation->learned_router_operations(0).value(),
              reinterpret_cast<DeepSeekLearnedRouterOperations*>(0x1234));
    EXPECT_FALSE(generation->learned_router_operations(1).ok());
    EXPECT_FALSE(generation->admission_allowed());
    const DeepSeekStagePlan stage{0, {4, 4}, true, true, false};
    auto lanes = DeepSeekRankComputeLaneSet::Create(
        stage, 8, 8, *generation->rank(0).expert_pager(),
        std::make_unique<TestTransferOwner>(),
        std::make_unique<TestKernelOwner>());
    ASSERT_TRUE(lanes.ok()) << lanes.status().message();
    ASSERT_TRUE(generation->rank(0).attach_compute_lanes(
        std::make_unique<DeepSeekRankComputeLaneSet>(
            std::move(*lanes))).ok());
    EXPECT_TRUE(generation->admission_allowed());
    DeepSeekProductionRankPlanSeed invalid_seed;
    EXPECT_EQ(DeepSeekProductionRankPlanInputAssembler::Create(
                  *generation, 0,
                  {11, 1, DeepSeekPlanPhase::kDecode, 1, 1},
                  std::move(invalid_seed))
                  .status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(generation->rank_request_token_ids_u32(0).status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(generation->rank_request_token_ids_u32(1).status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(generation->rank_initial_residual_bf16(0).status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(generation->create_rank_outgoing_source(0, 1, 1, 1)
                  .status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_FALSE(generation->assemble_rank_router_factories(0, 8, 1).ok());
    EXPECT_FALSE(generation->assemble_rank_router_factories(1, 8, 1).ok());
    EXPECT_EQ(generation->assemble_native_plan_compiler(8, 8, 1)
                  .status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(generation->assemble_native_plan_compiler(0, 8, 1)
                  .status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(generation->assemble_production_deferred_plan_compiler(
                  {11, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {}, 8, 8, 1)
                  .status().code(),
              StatusCode::kInvalidArgument);
    const DeepSeekPipelinePlanDescriptor bound_drain{
        11, 88, DeepSeekPlanPhase::kDrain, 0, 1};
    EXPECT_FALSE(generation->bind_rank_compute_plans(
        bound_drain, {}).ok());
    ASSERT_TRUE(generation->bind_rank_compute_plans(
        bound_drain, {DeepSeekRankComputePlanWork{}}).ok());
    ASSERT_TRUE(generation->abort_rank_compute_plans(bound_drain).ok());
    auto poller = std::make_unique<ScriptedArtifactPoller>();
    auto* poller_state = poller.get();
    auto engine = DeepSeekEngine::Create(
        std::make_unique<DeepSeekEngineResources>(std::move(*generation)),
        std::move(poller),
        DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false).value());
    ASSERT_TRUE(engine.ok()) << engine.status().message();
    MismatchedDeferredCompiler mismatched_compiler;
    EXPECT_EQ(engine->prepare_deferred_native_pipeline(
                  {11, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {},
                  mismatched_compiler)
                  .code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(poller_state->calls, 1);
    EXPECT_TRUE(engine->admit_request().ok());
    EXPECT_TRUE(engine->submit_request(77, 2).ok());
    auto first_binding = engine->request_attention_binding(77, 2);
    ASSERT_TRUE(first_binding.ok()) << first_binding.status().message();
    EXPECT_EQ(first_binding->sequence, 1U);
    EXPECT_EQ(first_binding->state_slot, 0U);
    EXPECT_FALSE(engine->request_attention_binding(77, 3).ok());
    EXPECT_TRUE(engine->configure_ledger(77, 2, 1, 1, 0).ok());
    EXPECT_EQ(engine->prepare_production_pipeline(
                  {11, 1, DeepSeekPlanPhase::kDecode, 1, 1},
                  {{77, 2, false}}, {}, 8, 8, 129280)
                  .code(),
              StatusCode::kInvalidArgument);
    EXPECT_FALSE(engine->execution_live());
    const auto missing_owner = engine->prepare_bound_pipeline(
        {11, 1, DeepSeekPlanPhase::kDecode, 1, 1},
        {{77, 2, false}}, {DeepSeekRankComputePlanWork{}});
    EXPECT_FALSE(missing_owner.ok());
    EXPECT_NE(missing_owner.message().find("lifetime owner"),
              std::string::npos);
    EXPECT_FALSE(engine->execution_live());
    EXPECT_FALSE(engine->prepare_pipeline(
        {11, 1, DeepSeekPlanPhase::kDecode, 1, 1},
        {{77, 2, false}}).ok());
    EXPECT_EQ(engine->request_state(77, 2).value(),
              DeepSeekRequestState::kAdmitted);
    held_plan_resources->release();
    ASSERT_TRUE(engine->submit_request(70, 1).ok());
    auto second_binding = engine->request_attention_binding(70, 1);
    ASSERT_TRUE(second_binding.ok()) << second_binding.status().message();
    EXPECT_EQ(second_binding->sequence, 2U);
    EXPECT_EQ(second_binding->state_slot, 1U);
    ASSERT_TRUE(engine->configure_ledger(70, 1, 1, 1, 0).ok());
    ASSERT_TRUE(engine->prepare_bound_pipeline(
        {11, 1, DeepSeekPlanPhase::kDrain, 0, 1},
        {{70, 1, false}}, {DeepSeekRankComputePlanWork{}}).ok());
    ASSERT_TRUE(engine->cancel_request(70, 1).ok());
    EXPECT_EQ(runtime_activations, 2);
    EXPECT_FALSE(engine->execution_live());
    EXPECT_FALSE(engine->advance_bound_pipeline().ok());
    ASSERT_TRUE(engine->retire_request(70, 1).ok());
    EXPECT_FALSE(engine->request_attention_binding(70, 1).ok());
    ASSERT_TRUE(engine->submit_request(71, 1).ok());
    auto reused_binding = engine->request_attention_binding(71, 1);
    ASSERT_TRUE(reused_binding.ok()) << reused_binding.status().message();
    EXPECT_EQ(reused_binding->sequence, 3U);
    EXPECT_EQ(reused_binding->state_slot, 1U);
    ASSERT_TRUE(engine->cancel_request(71, 1).ok());
    ASSERT_TRUE(engine->retire_request(71, 1).ok());
    EXPECT_FALSE(engine->prepare_bound_drain_pipeline(0, {{77, 2, false}}).ok());
    runtime_activation_fail = true;
    auto activation_failure = engine->prepare_bound_drain_pipeline(
        2, {{77, 2, false}});
    EXPECT_FALSE(activation_failure.ok());
    EXPECT_NE(activation_failure.message().find("activation failure"),
              std::string::npos);
    EXPECT_FALSE(engine->execution_live());
    runtime_activation_fail = false;
    auto bound_prepare = engine->prepare_bound_drain_pipeline(
        2, {{77, 2, false}});
    ASSERT_TRUE(bound_prepare.ok()) << bound_prepare.message();
    EXPECT_FALSE(engine->pipeline_stage_ready(0).ok());
    ASSERT_TRUE(engine->commit_pipeline().ok());
    EXPECT_FALSE(engine->pipeline_stage_complete(0).ok());
    EXPECT_FALSE(engine->pipeline_stage_failed(
        0, Status::Internal("external bound failure")).ok());
    ASSERT_TRUE(engine->advance_bound_pipeline().ok());
    ASSERT_TRUE(engine->advance_bound_pipeline().ok());
    EXPECT_EQ(runtime_activations, 8);
    EXPECT_FALSE(engine->execution_live());
    ASSERT_TRUE(engine->acknowledge_output_plan(2).ok());
    ASSERT_TRUE(engine->prepare_pipeline(
        {11, 3, DeepSeekPlanPhase::kDecode, 1, 1},
        {{77, 2, false}}).ok());
    ASSERT_TRUE(engine->pipeline_stage_ready(0).ok());
    ASSERT_TRUE(engine->commit_pipeline().ok());
    ASSERT_TRUE(engine->stage_accepted_tokens(
        3, {{{42}, DeepSeekFinishReason::kLength}}).ok());
    ASSERT_TRUE(engine->pipeline_stage_complete(0).ok());
    auto accepted = engine->accepted_token_snapshot(77, 2);
    ASSERT_TRUE(accepted.ok());
    EXPECT_EQ(accepted->token_ids, std::vector<std::uint32_t>({42}));
    EXPECT_EQ(accepted->finish_reason, DeepSeekFinishReason::kLength);
    ASSERT_TRUE(engine->acknowledge_output_plan(3).ok());
    ASSERT_TRUE(engine->finish_request(77, 2).ok());
    EXPECT_EQ(engine->request_state(77, 2).value(),
              DeepSeekRequestState::kCompleted);
    EXPECT_EQ(engine->epoch(), 11U);
    EXPECT_EQ(engine->world_size(), 1U);
    EXPECT_TRUE(engine->poll_artifacts().ok());
    EXPECT_EQ(poller_state->calls, 2);
    poller_state->fail = true;
    EXPECT_FALSE(engine->poll_artifacts().ok());
    EXPECT_EQ(engine->state(), DeepSeekEngineState::kFailed);
    EXPECT_EQ(engine->failure(), DeepSeekEngineFailure::kArtifactIntegrity);
    EXPECT_EQ(engine->public_error(), "engine_artifact_integrity_failed");
    EXPECT_FALSE(engine->admit_request().ok());
    EXPECT_EQ(engine->request_state(77, 2).value(),
              DeepSeekRequestState::kCompleted);
    EXPECT_TRUE(engine->retire_request(77, 2).ok());
    EXPECT_TRUE(engine->begin_close().ok());
    EXPECT_EQ(engine->state(), DeepSeekEngineState::kClosing);
    EXPECT_FALSE(engine->admit_request().ok());
    EXPECT_TRUE(engine->advance_close().ok());
    EXPECT_TRUE(engine->advance_close().ok());
    EXPECT_EQ(engine->state(), DeepSeekEngineState::kClosed);
  }
  EXPECT_FALSE(runtime_alive);
  EXPECT_FALSE(released_after_runtime);
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankEngineResourcesTest, RejectsInvalidExpertTopologyBeforeIo) {
  TestCudaAllocator device;
  TestPinnedAllocator pinned;
  TestCudaCopier copier;
  DeepSeekWeightBindingDryRun dry_run;
  EXPECT_FALSE(DeepSeekRankEngineResources::BuildForInProcessDevelopment(
      1, 1, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, {}, {}, {},
      1, 1, DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false).value(),
      device, pinned, copier, dry_run).ok());
}

TEST(DeepSeekRankEngineResourcesTest,
     DsparkResourcesReachTheBoundInventoryChecks) {
  TestCudaAllocator device;
  TestPinnedAllocator pinned;
  TestCudaCopier copier;
  DeepSeekWeightBindingDryRun dry_run;
  DeepSeekRankMappingPlan mapping;
  mapping.rank = 0;
  const std::vector<DeepSeekRankTensorRecord> tensors{{
      "layers.4.attn_norm.weight", "missing.safetensors",
      DeepSeekTensorRole::kMainLayer, DType::kBFloat16, {4096}, 0, 8192}};

  auto rejected =
      DeepSeekRankEngineResources::BuildForInProcessDevelopment(
      1, 1, {4, 4}, DeepSeekRoutedExpertResidency::kFullResident,
      std::move(mapping), {}, tensors, 0, 0,
      DeepSeekPipelineCapacity::Create(1, 8, 8, 1, true).value(),
      device, pinned, copier, dry_run);

  ASSERT_FALSE(rejected.ok());
  EXPECT_NE(rejected.status().message(),
            "deepseek_dspark_3stage_runtime_open");
}

} }  // namespace pih
