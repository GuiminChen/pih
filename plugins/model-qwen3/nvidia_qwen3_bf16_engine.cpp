#include "pih/model/nvidia_qwen3_bf16_engine.h"
#include "execution_binding.h"
#include "controller_client.h"
#include "memory_binding.h"
#include "resource_binding.h"
#include "async_binding.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>

#include "runtime_resources.h"
#include "pih/backend/cuda/nvidia_kernel_module_driver.h"
#include "pih/backend/cuda/nvidia_linux_pinned_placement.h"
#include "pih/backend/cuda/nvidia_qwen_step_health_probe.h"
#include "pih/core/checked_math.h"
#include "pih/model/nvidia_qwen3_bf16_execution_driver.h"
#include "pih/model/nvidia_qwen3_bf16_kv_scrub_driver.h"
#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_bf16_engine_arenas.h"
#include "pih/model/qwen3_bf16_host_runtime.h"
#include "pih/model/qwen3_bf16_instrumented_backend.h"
#include "pih/model/qwen3_bf16_kernel_bundle.h"
#include "pih/model/qwen3_bf16_kv_startup.h"
#include "pih/model/qwen3_bf16_linear_plan.h"
#include "pih/model/qwen3_bf16_pinned_arenas.h"
#include "pih/model/qwen3_bf16_packed_synchronous_backend.h"
#include "pih/model/qwen3_bf16_request_gate.h"
#include "pih/model/qwen_engine_close_gate.h"
#include "pih/model/qwen3_bf16_semantic_control_executor.h"
#include "pih/model/qwen3_bf16_step_health.h"
#include "pih/model/qwen3_bf16_tap_fixture_sequence.h"
#include "pih/model/qwen3_bf16_tap_suite_executor.h"
#include "pih/model/qwen3_bf16_weight_loader.h"
#include "pih/model/qwen3_config.h"
#include "pih/model/qwen3_manifest.h"
#include "pih/model/qwen3_source_artifact.h"
#include "pih/model/qwen3_semantic_capacity_accounting.h"
#include "pih/model/qwen3_semantic_observation_finalizer.h"
#include "pih/model/qwen3_semantic_observation_identity.h"
#include "pih/model/safetensors_file.h"
#include "pih/scheduler/controller_packed_driver.h"
#include "pih/scheduler/controller_runtime.h"
#include "pih/scheduler/qwen_kv_admission_coordinator.h"

namespace pih {
namespace {

using qwen_plugin::kMaximumStepTokens;
using qwen_plugin::kMaximumSequenceTokens;
using qwen_plugin::kSlotCount;
using qwen_plugin::kMaximumBatchSequences;
constexpr std::uint64_t kLinearWorkspaceBudget = UINT64_C(64) << 20;
constexpr std::uint64_t kOperationTimeoutNs = UINT64_C(300) * 1000 * 1000 * 1000;
constexpr std::uint64_t kMaximumCubinBytes = UINT64_C(64) << 20;

CudaCopyEndpoint device_endpoint(const QwenBf16DeviceArenaOwner& owner,
                                 std::uint64_t owner_id,
                                 std::uint32_t rank) {
  return {owner.base, owner.bytes, 0, owner_id, owner.generation,
          CudaCopyMemoryType::kDevice, rank,
          static_cast<std::int32_t>(rank)};
}

Result<Sha256Digest> packed_resource_vector_hash(
    const QwenBf16EngineResourcePlan& plan) {
  const std::string identity =
      "qwen3-0.6b-bf16-packed-runtime-v1;step=" +
      std::to_string(plan.maximum_step_tokens()) + ";context=" +
      std::to_string(plan.maximum_sequence_tokens()) + ";slots=" +
      std::to_string(plan.slot_count()) + ";sequences=" +
      std::to_string(plan.maximum_batch_sequences()) + ";device_bytes=" +
      std::to_string(plan.total_device_bytes());
  return sha256(std::as_bytes(std::span(identity)));
}

Status claim_runtime_mode(std::atomic<std::uint8_t>& mode,
                          std::uint8_t requested) {
  std::uint8_t expected = 0;
  if (mode.compare_exchange_strong(expected, requested) ||
      expected == requested) {
    return Status::Ok();
  }
  return Status::FailedPrecondition(
      "Qwen engine execution mode is sealed for this epoch");
}

}  // namespace

struct NvidiaQwen3Bf16Engine::State final {
  const pih_nvidia_cuda_async_api_v1* async_api{};
  ~State() { if(runtime) qwen_plugin::RetireStreamsOrTerminate(*async_api, runtime->identity()); }
  std::unique_ptr<qwen_plugin::CapabilityResourceDriver> runtime_driver;
  std::optional<qwen_plugin::RuntimeResources> runtime;
  std::unique_ptr<qwen_plugin::CapabilityDeviceAllocator> allocator;
  std::unique_ptr<qwen_plugin::CapabilityMemoryCopier> copier;
  std::unique_ptr<qwen_plugin::CapabilityPinnedAllocator> pinned_allocator;
  NvidiaLinuxPinnedPlacementVerifier pinned_placement;
  std::int32_t numa_node = -1;
  std::optional<QwenBf16EngineResourcePlan> plan;
  std::optional<QwenBf16EngineDeviceArenas> arenas;
  std::optional<QwenBf16PinnedHostArenas> pinned;
  std::optional<ResidentWeightSet> resident_weights;
  std::optional<QwenBf16WeightResourceSet> weights;
  std::optional<NvidiaKernelModuleDriver> kernel_driver;
  std::optional<QwenBf16KernelBundle> kernels;
  std::optional<QwenBf16ExecutionSchedule> schedule;
  std::optional<QwenBf16WeightBindingPlan> weight_bindings;
  std::optional<QwenBf16CommandBuffer> commands;
  std::optional<qwen_plugin::CapabilityTypedCopyDriver> copy_driver;
  std::optional<qwen_plugin::CapabilityErrorClearDriver> clear_driver;
  std::optional<qwen_plugin::CapabilityEventDriver> event_driver;
  std::optional<NvidiaQwenStepHealthProbe> health_probe;
  std::atomic<bool> poisoned{false};
  std::atomic<std::uint8_t> runtime_mode{0};
  std::optional<QwenBf16AtomicStepHealthProvider> health;
  QwenBf16SteadyClock clock;
  QwenBf16YieldWaiter waiter;
  std::optional<NvidiaQwenBf16KvScrubDriver> scrub_driver;
  std::optional<QwenKvSlotPool> pool;
  std::optional<QwenBf16SynchronousKvRecycler> recycler;
  std::unique_ptr<QwenBf16LinearPlanSet> prefill_linear_plans;
  std::unique_ptr<QwenBf16LinearPlanSet> decode_linear_plans;
  std::vector<std::unique_ptr<QwenBf16LinearPlanSet>>
      packed_lm_head_plan_owners;
  std::vector<const QwenBf16LinearPlanSet*> packed_lm_head_plans;
  std::optional<NvidiaQwenBf16ExecutionDriver> packed_execution;
  std::optional<QwenBf16PackedSynchronousBackend> packed_backend;
  std::optional<qwen_plugin::ControllerClient> controller;
  std::optional<ControllerPackedDriver> controller_driver;
  std::optional<QwenKvAdmissionCoordinator> admission_coordinator;
  QwenBf16RequestGate request_gate;
  QwenEngineCloseGate close_gate;
  std::recursive_mutex operation_mutex;
  std::mutex packed_submit_mutex;
  Qwen3Config config{};
  std::int32_t rank = 0;
  std::array<std::uint64_t, 2> pinned_endpoint_owner_ids{201, 202};
  std::array<std::uint64_t, 3> device_endpoint_owner_ids{203, 204, 205};
  std::uint64_t next_evidence_suite_generation = 1;
  std::uint64_t next_evidence_fixture_generation = 1;
  std::uint64_t next_evidence_event_generation = 1;
  std::uint64_t next_evidence_plan_id = 1;
  std::uint64_t next_packed_request_generation = 1;

  QwenBf16EngineDeviceArenas& device_arenas() noexcept {
    return *arenas;
  }
  QwenBf16PinnedHostArenas& pinned_arenas() noexcept {
    return *pinned;
  }
};

Result<std::unique_ptr<NvidiaQwen3Bf16Engine>>
NvidiaQwen3Bf16Engine::LoadPinnedSnapshot(
    const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_resources_api_v1& resources,
    const pih_nvidia_cuda_async_api_v1& async,
    const std::filesystem::path& cubin_root,
    std::int32_t device_ordinal,
    std::span<const std::byte> artifact_bytes,
    std::string_view config_json,
    Sha256Digest expected_artifact_digest,
    std::uint32_t provider_prepared_sm) {
  if (artifact_bytes.empty() || config_json.empty() || expected_artifact_digest == Sha256Digest{} ||
      (provider_prepared_sm != 89 && provider_prepared_sm != 90))
    return Status::InvalidArgument("Qwen BF16 authenticated snapshot invalid");
  return LoadImpl(memory, resources, async, cubin_root,
                  device_ordinal, expected_artifact_digest,
                  artifact_bytes, config_json, provider_prepared_sm);
}

Result<std::unique_ptr<NvidiaQwen3Bf16Engine>>
NvidiaQwen3Bf16Engine::LoadImpl(
    const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_resources_api_v1& resources,
    const pih_nvidia_cuda_async_api_v1& async,
    const std::filesystem::path& cubin_root,
    std::int32_t device_ordinal,
    Sha256Digest expected_artifact_digest,
    std::span<const std::byte> artifact_bytes,
    std::string_view config_json,
    std::uint32_t provider_prepared_sm) {
  if (device_ordinal < 0 || cubin_root.empty() ||
      (provider_prepared_sm != 89 && provider_prepared_sm != 90)) {
    return Status::InvalidArgument("Qwen engine load identity is invalid");
  }
  if (!qwen_plugin::ValidMemoryApi(memory) || !qwen_plugin::ValidResourceApi(resources) || !qwen_plugin::ValidAsyncApi(async))
    return Status::FailedPrecondition("Qwen CUDA memory capability is required");
  auto state = std::make_unique<State>();
  state->async_api = &async;
  state->runtime_driver = std::make_unique<qwen_plugin::CapabilityResourceDriver>(resources);
  state->rank = device_ordinal;
  auto config = Qwen3Config::Parse(config_json);
  if (!config.ok()) return config.status();
  state->config = *config;

  auto runtime = qwen_plugin::RuntimeResources::Create(
      device_ordinal, static_cast<std::uint32_t>(device_ordinal), 1,
      PIH_CUDA_CONTEXT_SCHED_YIELD_V1, *state->runtime_driver);
  if (!runtime.ok()) return runtime.status();
  state->runtime.emplace(std::move(*runtime));
  state->clear_driver.emplace(async, state->runtime->identity().context, state->rank);
  const std::uint32_t sm = provider_prepared_sm;
  auto prefill_linear_plans = QwenBf16LinearPlanSet::Create(
      kMaximumStepTokens, kLinearWorkspaceBudget);
  if (!prefill_linear_plans.ok()) return prefill_linear_plans.status();
  state->prefill_linear_plans = std::move(*prefill_linear_plans);
  auto decode_linear_plans = QwenBf16LinearPlanSet::Create(
      1, kLinearWorkspaceBudget);
  if (!decode_linear_plans.ok()) return decode_linear_plans.status();
  state->decode_linear_plans = std::move(*decode_linear_plans);
  state->packed_lm_head_plan_owners.reserve(kMaximumBatchSequences - 1);
  state->packed_lm_head_plans.reserve(kMaximumBatchSequences);
  state->packed_lm_head_plans.push_back(state->decode_linear_plans.get());
  for (std::uint32_t rows = 2; rows <= kMaximumBatchSequences; ++rows) {
    auto packed_plan = QwenBf16LinearPlanSet::Create(
        rows, kLinearWorkspaceBudget);
    if (!packed_plan.ok()) return packed_plan.status();
    state->packed_lm_head_plan_owners.push_back(std::move(*packed_plan));
    state->packed_lm_head_plans.push_back(
        state->packed_lm_head_plan_owners.back().get());
  }

  state->allocator = std::make_unique<qwen_plugin::CapabilityDeviceAllocator>(memory, device_ordinal);
  state->copier = std::make_unique<qwen_plugin::CapabilityMemoryCopier>(memory, device_ordinal);
  state->pinned_allocator = std::make_unique<qwen_plugin::CapabilityPinnedAllocator>(memory, -1);
  auto numa_node = NvidiaLinuxPinnedPlacementVerifier::DeviceNumaNode(
      device_ordinal);
  if (!numa_node.ok()) return numa_node.status();
  state->numa_node = *numa_node;
  auto plan = QwenBf16EngineResourcePlan::Create(
      kMaximumStepTokens, kMaximumSequenceTokens, kSlotCount,
      kLinearWorkspaceBudget, kMaximumBatchSequences);
  if (!plan.ok()) return plan.status();
  state->plan.emplace(std::move(*plan));
  auto numa_policy = NvidiaLinuxNumaPolicyGuard::Bind(state->numa_node);
  if (!numa_policy.ok()) return numa_policy.status();
  auto arenas = QwenBf16EngineDeviceArenas::Allocate(
      *state->plan, *state->allocator, device_ordinal);
  if (!arenas.ok()) return arenas.status();
  state->arenas.emplace(std::move(*arenas));
  auto pinned = QwenBf16PinnedHostArenas::AllocateVerified(
      *state->plan, *state->pinned_allocator, state->pinned_placement,
      state->numa_node);
  if (!pinned.ok()) return pinned.status();
  state->pinned.emplace(std::move(*pinned));
  const Status restored = numa_policy->restore();
  if (!restored.ok()) return restored;

  constexpr std::uint64_t kMaximumWeightFileBytes =
      Qwen3Manifest::kOfficialSourcePayloadBytes +
      SafetensorsHeader::kMaxHeaderBytes + 8;
  if (artifact_bytes.size() > kMaximumWeightFileBytes)
    return Status::ResourceExhausted("Qwen BF16 snapshot exceeds weight budget");
  auto source = SafetensorsFile::Borrow(artifact_bytes);
  if (!source.ok()) return source.status();
  Status manifest = Qwen3Manifest::Validate(source->header());
  if (!manifest.ok()) return manifest;
  const auto authenticated = verify_qwen3_source_artifact(*source,
      {artifact_bytes.size(), Qwen3Manifest::kOfficialTensorCount,
       Qwen3Manifest::kOfficialSourcePayloadBytes, expected_artifact_digest});
  if (!authenticated.ok()) return authenticated.status();
  const auto requirements = QwenBf16WeightLoader::Requirements();
  auto resident = ResidentWeightSet::Load(
      *source, requirements, *state->allocator, *state->copier, 256);
  if (!resident.ok()) return resident.status();
  state->resident_weights.emplace(std::move(*resident));
  auto weights = QwenBf16WeightLoader::Bind(
      *state->resident_weights, device_ordinal);
  if (!weights.ok()) return weights.status();
  state->weights.emplace(std::move(*weights));

  auto kernel_driver = NvidiaKernelModuleDriver::Create();
  if (!kernel_driver.ok()) return kernel_driver.status();
  state->kernel_driver.emplace(std::move(*kernel_driver));
  const auto sm_directory = cubin_root / ("sm_" + std::to_string(sm));
  auto kernels = QwenBf16KernelBundle::Load(
      *state->kernel_driver,
      sm_directory / "qwen_bf16_primitives.cubin.json",
      sm_directory / "qwen_bf16_primitives.cubin",
      sm / 10, sm % 10, kMaximumCubinBytes);
  if (!kernels.ok()) return kernels.status();
  state->kernels.emplace(std::move(*kernels));
  auto schedule = QwenBf16ExecutionSchedule::Create(state->config);
  if (!schedule.ok()) return schedule.status();
  state->schedule.emplace(std::move(*schedule));
  auto bindings = QwenBf16WeightBindingPlan::Create(*state->schedule);
  if (!bindings.ok()) return bindings.status();
  state->weight_bindings.emplace(std::move(*bindings));
  auto commands = QwenBf16CommandBuffer::Create(
      *state->schedule, *state->weight_bindings);
  if (!commands.ok()) return commands.status();
  state->commands.emplace(std::move(*commands));

  state->copy_driver.emplace(async, state->runtime->identity().context);
  state->event_driver.emplace(async, state->runtime->identity().context);
  state->health_probe.emplace(*state->event_driver);
  auto health = QwenBf16AtomicStepHealthProvider::Create(
      *state->health_probe, state->poisoned);
  if (!health.ok()) return health.status();
  state->health.emplace(std::move(*health));
  auto packed_execution = NvidiaQwenBf16ExecutionDriver::Create(
      *state->prefill_linear_plans, *state->decode_linear_plans,
      state->packed_lm_head_plans, state->device_arenas().linear_workspace(),
      state->device_arenas().linear_workspace_bytes(), state->rank);
  if (!packed_execution.ok()) return packed_execution.status();
  state->packed_execution.emplace(std::move(*packed_execution));
  auto scrub_driver = NvidiaQwenBf16KvScrubDriver::Create(
      async, state->runtime->identity().context, kOperationTimeoutNs);
  if (!scrub_driver.ok()) return scrub_driver.status();
  state->scrub_driver.emplace(std::move(*scrub_driver));
  auto pool = QwenKvSlotPool::Create(
      kSlotCount, state->plan->kv_backing_bytes(),
      state->plan->kv_metadata_bytes());
  if (!pool.ok()) return pool.status();
  state->pool.emplace(std::move(*pool));
  const auto owners = state->device_arenas().step_owners();
  Status sanitized = QwenBf16KvStartup::SanitizeAndPublish(
      *state->pool, owners.kv_backing, owners.kv_slot_states,
      state->runtime->identity().scrub_stream,
      state->runtime->identity().scrub_event, *state->scrub_driver);
  if (!sanitized.ok()) return sanitized;
  auto recycler = QwenBf16SynchronousKvRecycler::Create(
      owners.kv_backing.base, owners.kv_backing.bytes, kSlotCount,
      state->runtime->identity().scrub_stream,
      state->runtime->identity().scrub_event, 1, *state->scrub_driver);
  if (!recycler.ok()) return recycler.status();
  state->recycler.emplace(std::move(*recycler));

  const auto& runtime_identity = state->runtime->identity();
  Status initial_event = state->event_driver->record(
      runtime_identity.event, runtime_identity.stream);
  if (initial_event.ok()) {
    // The event follows all baseline work on this stream. Synchronizing the
    // owning stream establishes completion without bypassing the async provider.
    initial_event = qwen_plugin::MemoryStatus(async.synchronize_stream(
        async.context, runtime_identity.context, runtime_identity.stream));
  }
  if (!initial_event.ok()) return initial_event;
  auto pinned_staging = state->pinned_arenas().staging_endpoint(
      state->pinned_endpoint_owner_ids[0],
      static_cast<std::uint32_t>(state->rank));
  if (!pinned_staging.ok()) return pinned_staging.status();
  auto pinned_result = state->pinned_arenas().result_endpoint(
      state->pinned_endpoint_owner_ids[1],
      static_cast<std::uint32_t>(state->rank));
  if (!pinned_result.ok()) return pinned_result.status();
  QwenBf16PackedBackendArenas packed_arenas{
      owners, *pinned_staging,
      device_endpoint(owners.step_staging,
                      state->device_endpoint_owner_ids[0], state->rank),
      device_endpoint(owners.sampled_token,
                      state->device_endpoint_owner_ids[1], state->rank),
      device_endpoint(owners.device_error,
                      state->device_endpoint_owner_ids[2], state->rank),
      *pinned_result, state->pinned_arenas().staging(),
      state->pinned_arenas().result()};
  QwenBf16PackedBackendIdentity packed_identity{
      1, 1, 1, 1, kOperationTimeoutNs, runtime_identity.context,
      runtime_identity.stream, runtime_identity.event, state->rank,
      kSlotCount, kMaximumBatchSequences,
      static_cast<float>(state->config.rms_norm_epsilon),
      1.0F / std::sqrt(static_cast<float>(state->config.head_dim))};
  QwenBf16PackedBackendDrivers packed_drivers{
      &*state->copy_driver, &*state->clear_driver,
      &*state->kernel_driver, &*state->packed_execution,
      &*state->event_driver, &*state->health, &state->clock,
      &state->waiter};
  auto packed_backend = QwenBf16PackedSynchronousBackend::Create(
      *state->commands, state->kernels->legacy_functions(),
      state->kernels->packed_functions(), *state->weights, packed_arenas,
      packed_identity, packed_drivers);
  if (!packed_backend.ok()) return packed_backend.status();
  state->packed_backend.emplace(std::move(*packed_backend));

  auto resource_hash = packed_resource_vector_hash(*state->plan);
  if (!resource_hash.ok()) return resource_hash.status();
  auto controller = qwen_plugin::ControllerClient::CreateBound(
      {1, kMaximumBatchSequences,
       {{64, 128},
        {kMaximumBatchSequences, kMaximumSequenceTokens,
         kMaximumSequenceTokens}},
       {kMaximumBatchSequences, kMaximumBatchSequences,
        kMaximumStepTokens, kMaximumBatchSequences, 8, 5'000'000},
       {kMaximumBatchSequences, kMaximumStepTokens},
       kMaximumStepTokens, true, "qwen3-bf16-packed-v1", *resource_hash,
       static_cast<std::uint32_t>(state->config.eos_token_id)});
  if (!controller.ok()) return controller.status();
  state->controller.emplace(std::move(*controller));
  auto controller_driver = ControllerPackedDriver::Create(
      *state->controller, *state->packed_backend, kMaximumBatchSequences,
      {kMaximumBatchSequences, kMaximumStepTokens, kSlotCount});
  if (!controller_driver.ok()) return controller_driver.status();
  state->controller_driver.emplace(std::move(*controller_driver));
  auto admission_coordinator = QwenKvAdmissionCoordinator::Create(
      *state->pool, *state->controller_driver, kMaximumBatchSequences,
      {runtime_identity.event, 1});
  if (!admission_coordinator.ok()) return admission_coordinator.status();
  state->admission_coordinator.emplace(std::move(*admission_coordinator));
  return std::unique_ptr<NvidiaQwen3Bf16Engine>(
      new NvidiaQwen3Bf16Engine(std::move(state)));
}

NvidiaQwen3Bf16Engine::NvidiaQwen3Bf16Engine(
    std::unique_ptr<State> state)
    : state_(std::move(state)) {}

NvidiaQwen3Bf16Engine::~NvidiaQwen3Bf16Engine() = default;

Result<std::uint64_t> NvidiaQwen3Bf16Engine::submit_packed(
    std::span<const std::int64_t> prompt,
    std::uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling) {
  if (!state_ || state_->poisoned.load()) {
    return Status::FailedPrecondition("packed Qwen runtime is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  if (prompt.empty() || prompt.size() > kMaximumSequenceTokens ||
      maximum_new_tokens == 0 ||
      maximum_new_tokens > kMaximumSequenceTokens ||
      prompt.size() > kMaximumSequenceTokens - maximum_new_tokens) {
    return Status::InvalidArgument("packed Qwen request bounds are invalid");
  }
  const Status mode = claim_runtime_mode(state_->runtime_mode, 2);
  if (!mode.ok()) return mode;
  std::vector<std::uint32_t> tokens;
  tokens.reserve(prompt.size());
  for (const auto token : prompt) {
    if (token < 0 || token >= QwenBf16PackedResultLayout::kVocabularySize) {
      return Status::InvalidArgument(
          "packed Qwen prompt contains an invalid token");
    }
    tokens.push_back(static_cast<std::uint32_t>(token));
  }
  std::lock_guard submit(state_->packed_submit_mutex);
  if (state_->next_packed_request_generation == UINT64_MAX) {
    state_->poisoned.store(true);
    return Status::ResourceExhausted(
        "packed Qwen request generation is exhausted");
  }
  const auto generation = state_->next_packed_request_generation;
  auto submitted = state_->controller->submit_admit(
      generation, tokens, maximum_new_tokens, sampling);
  if (!submitted.ok()) return submitted;
  ++state_->next_packed_request_generation;
  return generation;
}

Status NvidiaQwen3Bf16Engine::cancel_packed(
    std::uint64_t request_generation) {
  if (!state_ || state_->poisoned.load()) {
    return Status::FailedPrecondition("packed Qwen runtime is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  auto cancelled = state_->controller->submit_cancel(request_generation);
  return cancelled;
}

Result<NvidiaQwenPackedStep> NvidiaQwen3Bf16Engine::drive_packed() {
  if (!state_ || state_->poisoned.load()) {
    return Status::FailedPrecondition("packed Qwen runtime is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  std::unique_lock operation(state_->operation_mutex, std::try_to_lock);
  if (!operation.owns_lock()) {
    return Status::Unavailable("Qwen engine operation is already active");
  }
  auto now = state_->clock.now_ns();
  if (!now.ok() || *now > static_cast<std::uint64_t>(INT64_MAX)) {
    state_->poisoned.store(true);
    return now.ok() ? Status::Internal("packed Qwen clock overflowed")
                    : now.status();
  }
  if (state_->controller->queued_command_count() != 0) {
    auto stepped = state_->controller->step(
        static_cast<std::int64_t>(*now), *state_->admission_coordinator);
    if (!stepped.ok()) {
      if (stepped.status().code() != StatusCode::kResourceExhausted) {
        state_->poisoned.store(true);
      }
      return stepped.status();
    }
    if (*stepped == ControllerRuntimeStep::kOutputBackpressured) {
      return NvidiaQwenPackedStep::kOutputBackpressured;
    }
    if (*stepped == ControllerRuntimeStep::kIdle) {
      return NvidiaQwenPackedStep::kIdle;
    }
    return NvidiaQwenPackedStep::kCommandProcessed;
  }
  auto executed = state_->controller_driver->execute_next(
      static_cast<std::int64_t>(*now));
  if (!executed.ok()) {
    state_->poisoned.store(true);
    return executed.status();
  }
  if (*executed == ControllerPackedDriverStep::kIdle) {
    return NvidiaQwenPackedStep::kIdle;
  }
  if (*executed == ControllerPackedDriverStep::kOutputBackpressured) {
    return NvidiaQwenPackedStep::kOutputBackpressured;
  }
  return NvidiaQwenPackedStep::kBatchCompleted;
}

Result<std::optional<NvidiaQwenPackedOutputEvent>>
NvidiaQwen3Bf16Engine::try_take_packed_event() {
  if (!state_ || state_->poisoned.load()) {
    return Status::FailedPrecondition("packed Qwen runtime is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  auto event = state_->controller->try_take_event();
  if (!event.ok()) return event.status();
  if (!event->has_value())
    return std::optional<NvidiaQwenPackedOutputEvent>{};
  std::optional<ControllerPackedSamplingReceipt> receipt;
  if ((*event)->kind == ControllerOutputEventKind::kTokenCommitted) {
    auto taken = state_->controller_driver->try_take_sampling_receipt();
    if (!taken.ok()) return taken.status();
    if (!taken->has_value() ||
        (**taken).request_generation != (*event)->request_generation ||
        (**taken).token_ordinal != (*event)->token_ordinal ||
        (**taken).sample.token_id != (*event)->token_id) {
      state_->poisoned.store(true);
      return Status::Internal("packed Qwen sampling receipt identity drifted");
    }
    receipt = **taken;
  }
  return std::optional<NvidiaQwenPackedOutputEvent>{
      NvidiaQwenPackedOutputEvent{**event, std::move(receipt)}};
}

Status NvidiaQwen3Bf16Engine::acknowledge_packed_output(
    std::uint64_t plan_sequence) {
  if (!state_ || state_->poisoned.load()) {
    return Status::FailedPrecondition("packed Qwen runtime is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  return state_->controller->acknowledge_output_plan(plan_sequence);
}

Status NvidiaQwen3Bf16Engine::drain_packed(
    std::uint64_t request_generation) {
  if (!state_ || state_->poisoned.load()) {
    return Status::FailedPrecondition("packed Qwen runtime is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  std::unique_lock operation(state_->operation_mutex, std::try_to_lock);
  if (!operation.owns_lock()) {
    return Status::Unavailable("Qwen engine operation is already active");
  }
  const Status drained = state_->admission_coordinator->drain(
      request_generation, *state_->recycler);
  if (!drained.ok() && drained.code() != StatusCode::kResourceExhausted) {
    state_->poisoned.store(true);
  }
  return drained;
}

Result<QwenBf16TapSuiteRunReceipt>
NvidiaQwen3Bf16Engine::run_numerical_evidence(
    std::span<const std::int64_t> fixture_tokens) {
  auto result = run_instrumented_semantic_evidence(fixture_tokens);
  if (!result.ok()) return result.status();
  return std::move(result->tap_suite);
}

Result<QwenBf16InstrumentedSemanticRunReceipt>
NvidiaQwen3Bf16Engine::run_instrumented_semantic_evidence(
    std::span<const std::int64_t> fixture_tokens) {
  if (!state_) {
    return Status::FailedPrecondition("Qwen engine is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  const Status mode = claim_runtime_mode(state_->runtime_mode, 1);
  if (!mode.ok()) return mode;
  std::unique_lock operation(state_->operation_mutex, std::try_to_lock);
  if (!operation.owns_lock()) {
    return Status::Unavailable("Qwen engine operation is already active");
  }
  auto suite = QwenBf16TapSuitePlan::Create();
  if (!suite.ok()) return suite.status();
  auto sequence = QwenBf16TapFixtureSequence::Create(fixture_tokens, *suite);
  if (!sequence.ok()) return sequence.status();
  const Status admitted = state_->request_gate.begin();
  if (!admitted.ok()) return admitted;
  auto result = [&]() -> Result<QwenBf16InstrumentedSemanticRunReceipt> {
    if (state_->poisoned.load() ||
        state_->next_evidence_suite_generation == UINT64_MAX ||
        state_->next_evidence_fixture_generation > UINT64_MAX - 5 ||
        state_->next_evidence_event_generation > UINT64_MAX - 11 ||
        state_->next_evidence_plan_id > UINT64_MAX - 45) {
      return Status::FailedPrecondition(
          "Qwen numerical evidence identity space is unavailable");
    }
    auto sequence_generation = state_->pool->acquire_sequence_generation();
    if (!sequence_generation.ok()) return sequence_generation.status();
    auto execution = NvidiaQwenBf16ExecutionDriver::Create(
        *state_->prefill_linear_plans, *state_->decode_linear_plans,
        state_->arenas->linear_workspace(),
        state_->arenas->linear_workspace_bytes(), state_->rank);
    if (!execution.ok()) return execution.status();
    const auto owners = state_->arenas->step_owners();
    auto pinned_staging = state_->pinned->staging_endpoint(
        101, static_cast<std::uint32_t>(state_->rank));
    if (!pinned_staging.ok()) return pinned_staging.status();
    auto pinned_result = state_->pinned->result_endpoint(
        102, static_cast<std::uint32_t>(state_->rank));
    if (!pinned_result.ok()) return pinned_result.status();
    QwenBf16InstrumentedBackendArenas backend_arenas{
        owners, *pinned_staging,
        device_endpoint(owners.step_staging, 103, state_->rank),
        device_endpoint(owners.sampled_token, 104, state_->rank),
        device_endpoint(owners.device_error, 105, state_->rank),
        *pinned_result, state_->pinned->staging(), state_->pinned->result()};
    const auto& runtime = state_->runtime->identity();
    const QwenBf16InstrumentedBackendIdentity identity{
        1, state_->next_evidence_fixture_generation,
        state_->next_evidence_event_generation,
        state_->next_evidence_plan_id, kOperationTimeoutNs, runtime.context,
        runtime.stream, runtime.diagnostic_stream, runtime.diagnostic_event,
        state_->rank, state_->numa_node, kSlotCount,
        static_cast<float>(state_->config.rms_norm_epsilon),
        1.0F / std::sqrt(static_cast<float>(state_->config.head_dim)),
        1000, 2000, 3000};
    auto backend = QwenBf16InstrumentedBackend::Create(
        *state_->commands, state_->kernels->legacy_functions(), *state_->weights,
        backend_arenas, identity,
        {state_->allocator.get(), state_->pinned_allocator.get(),
         &state_->pinned_placement, &*state_->copy_driver, &*state_->clear_driver,
         &*state_->kernel_driver, &*execution, &*state_->event_driver,
         &*state_->health, &state_->clock, &state_->waiter});
    if (!backend.ok()) return backend.status();
    auto executor = QwenBf16TapSuiteExecutor::Admit(
        *state_->pool, 0, *sequence_generation, *suite, *sequence,
        state_->next_evidence_suite_generation,
        state_->next_evidence_fixture_generation, *backend);
    if (!executor.ok()) return executor.status();
    const auto handles = std::vector<QwenKvBlockHandle>(
        executor->reserved_handles().begin(), executor->reserved_handles().end());
    auto numa_policy = NvidiaLinuxNumaPolicyGuard::Bind(state_->numa_node);
    if (!numa_policy.ok()) return numa_policy.status();
    auto receipt = executor->run();
    if (!receipt.ok()) return receipt.status();
    QwenSemanticOutcomeRecorder recorder;
    Status semantic_status = recorder.record_dispatch(*state_->commands);
    if (semantic_status.ok()) {
      semantic_status = executor->publish_token_semantics(recorder);
    }
    if (!semantic_status.ok()) return semantic_status;
    auto kv_plan = executor->make_kv_observation_plan(
        state_->plan->kv_backing_bytes());
    if (!kv_plan.ok()) return kv_plan.status();
    auto device_baseline = checked_add_u64(
        state_->arenas->total_bytes(), state_->resident_weights->total_bytes());
    if (!device_baseline.ok()) return device_baseline.status();
    auto semantic_pinned_bytes = checked_add_u64(
        QwenSemanticObservationTransfer::kFinalLogitsBytes,
        kv_plan->payload_bytes());
    if (!semantic_pinned_bytes.ok()) return semantic_pinned_bytes.status();
    auto capacity = qwen_semantic_capacity_pair(
        *device_baseline,
        state_->pinned->total_bytes(), backend->tap_device_peak_bytes(),
        backend->tap_pinned_peak_bytes(),
        *semantic_pinned_bytes);
    if (!capacity.ok()) return capacity.status();
    auto evidence = QwenBf16TapSnapshotEvidenceProvider::Create(*state_->health);
    if (!evidence.ok()) return evidence.status();
    auto observation_identity = reserve_qwen_semantic_observation_identity(
        backend->next_plan_id(), backend->next_event_generation(),
        kv_plan->slices().size());
    if (!observation_identity.ok()) return observation_identity.status();
    auto semantic = QwenSemanticObservationFinalizer::Run(
        *kv_plan,
        {device_endpoint(owners.logits, 4001, state_->rank),
         device_endpoint(owners.kv_backing, 4002, state_->rank)},
        {1, observation_identity->first_copy_plan_id,
         observation_identity->frontier_plan_id,
         observation_identity->completion_event_generation,
         kOperationTimeoutNs,
         runtime.context, runtime.diagnostic_stream, runtime.diagnostic_event,
         static_cast<std::uint32_t>(state_->rank), state_->numa_node, 5001,
         5002},
        {state_->pinned_allocator.get(), &state_->pinned_placement,
         &*state_->copy_driver, &*state_->event_driver, &*evidence,
         &state_->clock, &state_->waiter},
        capacity->instrumented_device_peak_bytes,
        capacity->instrumented_pinned_peak_bytes, std::move(recorder));
    const Status restored = numa_policy->restore();
    if (!restored.ok()) return restored;
    if (!semantic.ok()) return semantic.status();
    const auto completion = executor->last_completion_event();
    const Status released = executor->release();
    if (!released.ok()) return released;
    const Status recycled = state_->recycler->recycle(
        *state_->pool, handles, completion);
    if (!recycled.ok()) return recycled;
    ++state_->next_evidence_suite_generation;
    state_->next_evidence_fixture_generation += 5;
    state_->next_evidence_event_generation =
        observation_identity->next_event_generation;
    state_->next_evidence_plan_id = observation_identity->next_plan_id;
    return QwenBf16InstrumentedSemanticRunReceipt{
        std::move(*receipt), std::move(*semantic)};
  }();
  const bool recoverable =
      !result.ok() &&
      result.status().code() == StatusCode::kResourceExhausted &&
      !state_->pool->failed();
  const bool healthy = result.ok() || recoverable;
  if (!healthy) state_->poisoned.store(true);
  const Status finished = state_->request_gate.finish(healthy);
  if (!finished.ok()) {
    state_->poisoned.store(true);
    return finished;
  }
  return result;
}

Result<QwenSemanticOutcome>
NvidiaQwen3Bf16Engine::run_semantic_control_evidence(
    std::span<const std::int64_t> fixture_tokens) {
  if (!state_) {
    return Status::FailedPrecondition("Qwen engine is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  const Status mode = claim_runtime_mode(state_->runtime_mode, 1);
  if (!mode.ok()) return mode;
  std::unique_lock operation(state_->operation_mutex, std::try_to_lock);
  if (!operation.owns_lock()) {
    return Status::Unavailable("Qwen engine operation is already active");
  }
  auto suite = QwenBf16TapSuitePlan::Create();
  if (!suite.ok()) return suite.status();
  auto sequence = QwenBf16TapFixtureSequence::Create(fixture_tokens, *suite);
  if (!sequence.ok()) return sequence.status();
  const Status admitted = state_->request_gate.begin();
  if (!admitted.ok()) return admitted;
  auto result = [&]() -> Result<QwenSemanticOutcome> {
    if (state_->poisoned.load() ||
        state_->next_evidence_fixture_generation > UINT64_MAX - 5 ||
        state_->next_evidence_event_generation > UINT64_MAX - 6 ||
        state_->next_evidence_plan_id > UINT64_MAX - 35) {
      return Status::FailedPrecondition(
          "Qwen semantic control identity space is unavailable");
    }
    auto sequence_generation = state_->pool->acquire_sequence_generation();
    if (!sequence_generation.ok()) return sequence_generation.status();
    auto execution = NvidiaQwenBf16ExecutionDriver::Create(
        *state_->prefill_linear_plans, *state_->decode_linear_plans,
        state_->arenas->linear_workspace(),
        state_->arenas->linear_workspace_bytes(), state_->rank);
    if (!execution.ok()) return execution.status();
    const auto owners = state_->arenas->step_owners();
    auto pinned_staging = state_->pinned->staging_endpoint(
        101, static_cast<std::uint32_t>(state_->rank));
    if (!pinned_staging.ok()) return pinned_staging.status();
    auto pinned_result = state_->pinned->result_endpoint(
        102, static_cast<std::uint32_t>(state_->rank));
    if (!pinned_result.ok()) return pinned_result.status();
    QwenBf16SynchronousBackendArenas backend_arenas{
        owners, *pinned_staging,
        device_endpoint(owners.step_staging, 103, state_->rank),
        device_endpoint(owners.sampled_token, 104, state_->rank),
        device_endpoint(owners.device_error, 105, state_->rank),
        *pinned_result, state_->pinned->staging(), state_->pinned->result()};
    const auto& runtime = state_->runtime->identity();
    QwenBf16SynchronousBackendIdentity backend_identity{
        1, state_->next_evidence_fixture_generation,
        state_->next_evidence_event_generation,
        state_->next_evidence_plan_id, kOperationTimeoutNs, runtime.context,
        runtime.stream, runtime.event, state_->rank, kSlotCount,
        static_cast<float>(state_->config.rms_norm_epsilon),
        1.0F / std::sqrt(static_cast<float>(state_->config.head_dim))};
    QwenBf16SynchronousBackendDrivers backend_drivers{
        &*state_->copy_driver, &*state_->clear_driver, &*state_->kernel_driver,
        &*execution, &*state_->event_driver, &*state_->health,
        &state_->clock, &state_->waiter};
    auto backend = QwenBf16SynchronousBackend::Create(
        *state_->commands, state_->kernels->legacy_functions(), *state_->weights,
        backend_arenas, backend_identity, backend_drivers);
    if (!backend.ok()) return backend.status();
    auto executor = QwenBf16SemanticControlExecutor::Admit(
        *state_->pool, 0, *sequence_generation, *sequence, *backend, *backend);
    if (!executor.ok()) return executor.status();
    const auto handles = std::vector<QwenKvBlockHandle>(
        executor->reserved_handles().begin(), executor->reserved_handles().end());
    auto numa_policy = NvidiaLinuxNumaPolicyGuard::Bind(state_->numa_node);
    if (!numa_policy.ok()) return numa_policy.status();
    Status status = executor->run();
    if (!status.ok()) return status;
    QwenSemanticOutcomeRecorder recorder;
    status = recorder.record_dispatch(*state_->commands);
    if (status.ok()) status = executor->publish_token_semantics(recorder);
    if (!status.ok()) return status;
    auto kv_plan = executor->make_kv_observation_plan(
        state_->plan->kv_backing_bytes());
    if (!kv_plan.ok()) return kv_plan.status();
    auto device_baseline = checked_add_u64(
        state_->arenas->total_bytes(), state_->resident_weights->total_bytes());
    if (!device_baseline.ok()) return device_baseline.status();
    auto semantic_pinned_bytes = checked_add_u64(
        QwenSemanticObservationTransfer::kFinalLogitsBytes,
        kv_plan->payload_bytes());
    if (!semantic_pinned_bytes.ok()) return semantic_pinned_bytes.status();
    auto control_pinned_peak = checked_add_u64(
        state_->pinned->total_bytes(), *semantic_pinned_bytes);
    if (!control_pinned_peak.ok()) return control_pinned_peak.status();
    auto evidence = QwenBf16TapSnapshotEvidenceProvider::Create(*state_->health);
    if (!evidence.ok()) return evidence.status();
    auto observation_identity = reserve_qwen_semantic_observation_identity(
        backend->next_plan_id(), backend->next_event_generation(),
        kv_plan->slices().size());
    if (!observation_identity.ok()) return observation_identity.status();
    auto semantic = QwenSemanticObservationFinalizer::Run(
        *kv_plan,
        {device_endpoint(owners.logits, 6001, state_->rank),
         device_endpoint(owners.kv_backing, 6002, state_->rank)},
        {1, observation_identity->first_copy_plan_id,
         observation_identity->frontier_plan_id,
         observation_identity->completion_event_generation,
         kOperationTimeoutNs, runtime.context, runtime.diagnostic_stream,
         runtime.diagnostic_event, static_cast<std::uint32_t>(state_->rank),
         state_->numa_node, 7001, 7002},
        {state_->pinned_allocator.get(), &state_->pinned_placement,
         &*state_->copy_driver, &*state_->event_driver, &*evidence,
         &state_->clock, &state_->waiter},
        *device_baseline, *control_pinned_peak, std::move(recorder));
    const Status restored = numa_policy->restore();
    if (!restored.ok()) return restored;
    if (!semantic.ok()) return semantic.status();
    const auto completion = executor->last_completion_event();
    status = executor->release();
    if (!status.ok()) return status;
    status = state_->recycler->recycle(*state_->pool, handles, completion);
    if (!status.ok()) return status;
    state_->next_evidence_fixture_generation =
        backend->next_request_generation();
    state_->next_evidence_event_generation =
        observation_identity->next_event_generation;
    state_->next_evidence_plan_id = observation_identity->next_plan_id;
    return semantic;
  }();
  const bool recoverable =
      !result.ok() &&
      result.status().code() == StatusCode::kResourceExhausted &&
      !state_->pool->failed();
  const bool healthy = result.ok() || recoverable;
  if (!healthy) state_->poisoned.store(true);
  const Status finished = state_->request_gate.finish(healthy);
  if (!finished.ok()) {
    state_->poisoned.store(true);
    return finished;
  }
  return result;
}

Result<QwenSemanticControlReceipt>
NvidiaQwen3Bf16Engine::run_paired_semantic_evidence(
    const QwenNumericalRunIdentity& identity,
    std::span<const std::int64_t> fixture_tokens) {
  if (!state_) {
    return Status::FailedPrecondition("Qwen engine is not ready");
  }
  auto lease = state_->close_gate.enter();
  if (!lease.ok()) return lease.status();
  std::unique_lock operation(state_->operation_mutex, std::try_to_lock);
  if (!operation.owns_lock()) {
    return Status::Unavailable("Qwen engine operation is already active");
  }
  class Driver final : public QwenSemanticPairedExecutionDriver {
   public:
    explicit Driver(NvidiaQwen3Bf16Engine& engine) : engine_(&engine) {}
    Result<QwenSemanticInstrumentedRun> run_instrumented(
        std::span<const std::int64_t> tokens) override {
      return engine_->run_instrumented_semantic_evidence(tokens);
    }
    Result<QwenSemanticOutcome> run_control(
        std::span<const std::int64_t> tokens) override {
      return engine_->run_semantic_control_evidence(tokens);
    }

   private:
    NvidiaQwen3Bf16Engine* engine_;
  } driver(*this);
  QwenSemanticPairedExecutor executor;
  return executor.run(identity, fixture_tokens, driver);
}

Status NvidiaQwen3Bf16Engine::close() {
  if (!state_) return Status::Ok();
  const bool owns_close = state_->close_gate.request_close();
  if (!owns_close) return state_->close_gate.wait_closed();
  state_->close_gate.wait_drained();
  std::scoped_lock lock(state_->operation_mutex, state_->packed_submit_mutex);
  Status closed = Status::Ok();
  if (state_->controller &&
      (state_->controller->active_sequence_count() != 0 ||
       state_->controller->queued_command_count() != 0)) {
    closed = Status::FailedPrecondition(
        "Qwen engine cannot close with undrained packed requests");
  } else {
    closed = state_->request_gate.close();
    if (closed.ok() && state_->controller)
      closed = state_->controller->close();
  }
  if (!closed.ok()) state_->poisoned.store(true);
  state_->close_gate.finish_close(closed);
  return closed;
}

}  // namespace pih
