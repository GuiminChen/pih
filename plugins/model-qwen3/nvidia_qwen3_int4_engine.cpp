#include "pih/model/nvidia_qwen3_int4_engine.h"
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

#include "runtime_resources.h"
#include "pih/backend/cuda/nvidia_kernel_module_driver.h"
#include "pih/backend/cuda/nvidia_linux_pinned_placement.h"
#include "pih/backend/cuda/nvidia_qwen_step_health_probe.h"
#include "pih/model/nvidia_qwen3_bf16_kv_scrub_driver.h"
#include "pih/model/nvidia_qwen3_int4_capacity_probe.h"
#include "pih/model/nvidia_qwen3_int4_execution_driver.h"
#include "pih/model/nvidia_qwen3_int4_kernel_assets.h"
#include "pih/model/qwen3_bf16_host_runtime.h"
#include "pih/model/qwen3_bf16_kv_startup.h"

#include "pih/model/qwen3_bf16_step_health.h"

#include "pih/model/qwen3_int4_engine_load_plan.h"
#include "pih/model/qwen3_int4_weight_startup.h"
#include "pih/model/qwen3_int4_engine_runtime_arenas.h"
#include "pih/model/qwen3_int4_pinned_artifact.h"
#include "pih/model/qwen3_int4_packed_synchronous_backend.h"
#include "pih/model/qwen_engine_close_gate.h"
#include "pih/scheduler/controller_runtime.h"
#include "pih/scheduler/qwen_kv_admission_coordinator.h"

namespace pih { namespace {
using qwen_plugin::kMaximumStepTokens;
using qwen_plugin::kMaximumSequenceTokens;
using qwen_plugin::kSlotCount;
using qwen_plugin::kMaximumBatchSequences;
constexpr std::uint64_t kLinearWorkspaceBytes=UINT64_C(64)<<20;
constexpr std::uint64_t kOperationTimeoutNs=UINT64_C(300)*1000*1000*1000;
constexpr std::uint64_t kMaximumCubinBytes=UINT64_C(64)<<20;
constexpr std::uint64_t kDeviceReserveBytes=UINT64_C(256)<<20;
Result<Sha256Digest> packed_resource_vector_hash(
    const QwenInt4EngineResourcePlan& plan) {
  const auto& runtime=plan.runtime();
  const std::string identity=
      "qwen3-0.6b-int4-fused-packed-runtime-v1;step="+
      std::to_string(runtime.maximum_step_tokens())+";context="+
      std::to_string(runtime.maximum_sequence_tokens())+";slots="+
      std::to_string(runtime.slot_count())+";sequences="+
      std::to_string(kMaximumBatchSequences)+";device_bytes="+
      std::to_string(plan.total_device_bytes());
  return sha256(std::as_bytes(std::span(identity)));
}
}

struct NvidiaQwen3Int4Engine::State final {
 const pih_nvidia_cuda_async_api_v1* async_api{};
 ~State() { if(runtime) qwen_plugin::RetireStreamsOrTerminate(*async_api,runtime->identity()); }
 std::unique_ptr<qwen_plugin::CapabilityResourceDriver> runtime_driver;
 std::optional<qwen_plugin::RuntimeResources> runtime;
 std::unique_ptr<qwen_plugin::CapabilityDeviceAllocator> allocator;
 std::unique_ptr<qwen_plugin::CapabilityPinnedAllocator> pinned_allocator;
 NvidiaLinuxPinnedPlacementVerifier placement;
 std::int32_t numa_node=-1,rank=0;
 std::optional<QwenInt4EngineBootstrapPlan> bootstrap;
 std::optional<QwenInt4EngineRuntimeArenas> arenas;
 std::optional<QwenInt4PinnedArtifact> artifact;
 std::optional<NvidiaKernelModuleDriver> kernel_driver;
 std::optional<QwenInt4KernelBundle> kernels;
 std::optional<QwenBf16KernelBundle> packed_kernels;
 std::optional<QwenBf16ExecutionSchedule> schedule;
 std::optional<QwenBf16WeightBindingPlan> weight_bindings;
 std::optional<QwenBf16CommandBuffer> commands;
 std::optional<qwen_plugin::CapabilityTypedCopyDriver> copy_driver;
 std::optional<qwen_plugin::CapabilityErrorClearDriver> clear_driver;
 std::optional<qwen_plugin::CapabilityEventDriver> event_driver;
 std::optional<NvidiaQwenStepHealthProbe> health_probe;
 std::atomic<bool> poisoned{false};
 QwenEngineCloseGate close_gate;
 std::optional<QwenBf16AtomicStepHealthProvider> health;
 std::optional<QwenBf16TapSnapshotEvidenceProvider> evidence;
 QwenBf16SteadyClock clock;
 QwenBf16YieldWaiter waiter;
 std::optional<QwenInt4ResidentWeights> resident;
 std::optional<NvidiaQwenInt4ExecutionDriver> execution;
 std::optional<NvidiaQwenBf16KvScrubDriver> scrub_driver;
 std::optional<QwenKvSlotPool> pool;
 std::optional<QwenBf16SynchronousKvRecycler> recycler;

 std::optional<QwenInt4PackedSynchronousBackend> packed_backend;
 std::optional<qwen_plugin::ControllerClient> controller;
 std::optional<ControllerPackedDriver> controller_driver;
 std::optional<QwenKvAdmissionCoordinator> admission_coordinator;

 std::recursive_mutex operation_mutex;
 std::mutex packed_submit_mutex;

 std::uint64_t next_packed_request_generation=1;
 std::uint64_t pinned_artifact_owner_id=1001;
 std::uint64_t resident_weight_owner_id=1002;
 QwenInt4BackendArenaOwnerIds backend_owner_ids{2001,2002,2003,2004,2005};
 QwenInt4EngineRuntimeArenas& runtime_arenas() noexcept {
  return *arenas;
 }
};

Result<std::unique_ptr<NvidiaQwen3Int4Engine>> NvidiaQwen3Int4Engine::LoadPinnedSnapshot(
    const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_resources_api_v1& resources,
    const pih_nvidia_cuda_async_api_v1& async,
    const std::filesystem::path& cubin_root,
    std::int32_t ordinal,
    std::span<const std::byte> artifact_bytes,
    std::string_view config_json,
    Sha256Digest expected_artifact_digest,
    std::uint32_t provider_prepared_sm) {
 if(artifact_bytes.empty()||config_json.empty()||expected_artifact_digest==Sha256Digest{}||
    (provider_prepared_sm!=89&&provider_prepared_sm!=90))
  return Status::InvalidArgument("Qwen INT4 authenticated snapshot invalid");
 return LoadImpl(memory,resources,async,cubin_root,
                 ordinal,expected_artifact_digest,artifact_bytes,config_json,provider_prepared_sm);
}

Result<std::unique_ptr<NvidiaQwen3Int4Engine>> NvidiaQwen3Int4Engine::LoadImpl(
    const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_resources_api_v1& resources,
    const pih_nvidia_cuda_async_api_v1& async,
 const std::filesystem::path& cubin_root,std::int32_t ordinal,
      Sha256Digest expected_artifact_digest,
      std::span<const std::byte> artifact_bytes,
      std::string_view config_json,
      std::uint32_t provider_prepared_sm) {
 if(provider_prepared_sm!=89&&provider_prepared_sm!=90)
  return Status::InvalidArgument("Qwen INT4 requires a provider-admitted SM89 or SM90 device");
 auto load=QwenInt4EngineLoadPlan::CreateFromJson(config_json,cubin_root,ordinal);
 if(!load.ok())return load.status();
 if(!qwen_plugin::ValidMemoryApi(memory) || !qwen_plugin::ValidResourceApi(resources) || !qwen_plugin::ValidAsyncApi(async))
  return Status::FailedPrecondition("Qwen CUDA memory capability is required");
 auto state=std::make_unique<State>();
 state->async_api=&async;
 state->runtime_driver=std::make_unique<qwen_plugin::CapabilityResourceDriver>(resources);
 state->rank=ordinal;
 auto runtime=qwen_plugin::RuntimeResources::Create(ordinal,static_cast<std::uint32_t>(ordinal),1,
  PIH_CUDA_CONTEXT_SCHED_YIELD_V1,*state->runtime_driver);if(!runtime.ok())return runtime.status();
 state->runtime.emplace(std::move(*runtime));
 state->clear_driver.emplace(async,state->runtime->identity().context,state->rank);
 state->allocator=std::make_unique<qwen_plugin::CapabilityDeviceAllocator>(memory,ordinal);
 state->pinned_allocator=std::make_unique<qwen_plugin::CapabilityPinnedAllocator>(memory,-1);
 auto numa=NvidiaLinuxPinnedPlacementVerifier::DeviceNumaNode(ordinal);
 if(!numa.ok())return numa.status();state->numa_node=*numa;
 auto capacity=QwenInt4EngineResourcePlan::Create(kMaximumStepTokens,
  kMaximumSequenceTokens,kSlotCount,kLinearWorkspaceBytes,
  kMaximumBatchSequences);
 if(!capacity.ok())return capacity.status();
 const auto cubin_copy=load->cubin_root();
 auto bootstrap=NvidiaQwenInt4CapacityProbe::Admit(std::move(*load).take_model(),ordinal,
  kMaximumStepTokens,kMaximumSequenceTokens,kSlotCount,kLinearWorkspaceBytes,
  capacity->startup_pinned_peak_bytes(),kDeviceReserveBytes,
  kMaximumBatchSequences);
 if(!bootstrap.ok())return bootstrap.status();state->bootstrap.emplace(std::move(*bootstrap));
 // All verified host arenas must be allocated under the GPU's NUMA policy,
 // including staging/readback, not just the later artifact buffer.
 auto policy=NvidiaLinuxNumaPolicyGuard::Bind(state->numa_node);if(!policy.ok())return policy.status();
 auto arenas=QwenInt4EngineRuntimeArenas::Allocate(*state->bootstrap,*state->allocator,
  *state->pinned_allocator,state->placement,state->numa_node,ordinal);
 if(!arenas.ok())return arenas.status();state->arenas.emplace(std::move(*arenas));
 auto artifact=QwenInt4PinnedArtifact::LoadVerifiedBytes(artifact_bytes,*state->pinned_allocator,
      state->placement,state->numa_node,state->pinned_artifact_owner_id,
      static_cast<std::uint32_t>(state->rank),expected_artifact_digest);
 const Status restored=policy->restore();if(!restored.ok())return restored;
 if(!artifact.ok())return artifact.status();state->artifact.emplace(std::move(*artifact));
 auto kernel_driver=NvidiaKernelModuleDriver::Create();if(!kernel_driver.ok())return kernel_driver.status();
 state->kernel_driver.emplace(std::move(*kernel_driver));
 auto kernels=NvidiaQwenInt4KernelAssets::Load(*state->kernel_driver,cubin_copy,ordinal,kMaximumCubinBytes);
 if(!kernels.ok())return kernels.status();state->kernels.emplace(std::move(*kernels));
 auto packed_kernels=NvidiaQwenInt4KernelAssets::LoadPacked(
  *state->kernel_driver,cubin_copy,ordinal,kMaximumCubinBytes);
 if(!packed_kernels.ok())return packed_kernels.status();
 state->packed_kernels.emplace(std::move(*packed_kernels));
 auto schedule=QwenBf16ExecutionSchedule::Create(
  state->bootstrap->model().config());
 if(!schedule.ok())return schedule.status();state->schedule.emplace(std::move(*schedule));
 auto bindings=QwenBf16WeightBindingPlan::Create(*state->schedule);
 if(!bindings.ok())return bindings.status();
 state->weight_bindings.emplace(std::move(*bindings));
 auto commands=QwenBf16CommandBuffer::Create(
  *state->schedule,*state->weight_bindings);
 if(!commands.ok())return commands.status();state->commands.emplace(std::move(*commands));
 state->copy_driver.emplace(async,state->runtime->identity().context);
 state->event_driver.emplace(async,state->runtime->identity().context);state->health_probe.emplace(*state->event_driver);
 auto health=QwenBf16AtomicStepHealthProvider::Create(*state->health_probe,state->poisoned);
 if(!health.ok())return health.status();state->health.emplace(std::move(*health));
 auto evidence=QwenBf16TapSnapshotEvidenceProvider::Create(*state->health);
 if(!evidence.ok())return evidence.status();state->evidence.emplace(std::move(*evidence));
 auto now=state->clock.now_ns();if(!now.ok()||*now>UINT64_MAX-kOperationTimeoutNs)
  return Status::Unavailable("Qwen INT4 startup clock is unavailable");
 const auto& identity=state->runtime->identity();
 auto resident=QwenInt4ResidentWeights::Create(state->bootstrap->model().layout(),
  state->bootstrap->model().ledger(),1,*state->allocator,state->artifact->endpoint(),state->rank,
  state->resident_weight_owner_id,identity.context,
  identity.stream,identity.event,1,1,*now,*now+kOperationTimeoutNs);
 if(!resident.ok())return resident.status();state->resident.emplace(std::move(*resident));
 // Transfer ownership before submitting any DMA. State rollback synchronizes
 // all lanes before the resident or its pinned source can be destroyed.
 const auto published=QwenInt4WeightStartup::Publish(*state->resident,
  *state->copy_driver,*state->event_driver,*state->evidence,state->clock,state->waiter);
 if(!published.ok())return published;
 // Publication is synchronous and frontier-verified. Drop the 540 MB canonical
 // staging owner before steady state so the startup peak cannot become a leak.
 state->artifact.reset();
 auto weights=state->resident->resources();if(!weights.ok())return weights.status();
 auto execution=NvidiaQwenInt4ExecutionDriver::Create(state->runtime_arenas().device().linear_workspace(),
  state->runtime_arenas().device().linear_workspace_bytes(),kLinearWorkspaceBytes,
  state->rank,kMaximumBatchSequences);
 if(!execution.ok())return execution.status();state->execution.emplace(std::move(*execution));
 auto scrub=NvidiaQwenBf16KvScrubDriver::Create(async,state->runtime->identity().context,kOperationTimeoutNs);
 if(!scrub.ok())return scrub.status();state->scrub_driver.emplace(std::move(*scrub));
 const auto owners=state->runtime_arenas().device().step_owners();
 auto pool=QwenKvSlotPool::Create(kSlotCount,state->bootstrap->resources().runtime().kv_backing_bytes(),
  state->bootstrap->resources().runtime().kv_metadata_bytes());
 if(!pool.ok())return pool.status();state->pool.emplace(std::move(*pool));
 Status sanitized=QwenBf16KvStartup::SanitizeAndPublish(*state->pool,owners.kv_backing,
  owners.kv_slot_states,identity.scrub_stream,identity.scrub_event,*state->scrub_driver);
 if(!sanitized.ok())return sanitized;
 auto recycler=QwenBf16SynchronousKvRecycler::Create(owners.kv_backing.base,
  owners.kv_backing.bytes,kSlotCount,identity.scrub_stream,identity.scrub_event,1,*state->scrub_driver);
 if(!recycler.ok())return recycler.status();state->recycler.emplace(std::move(*recycler));
 auto backend_arenas=state->runtime_arenas().backend_arenas(
  state->backend_owner_ids,state->rank);
 if(!backend_arenas.ok())return backend_arenas.status();
 Status initial_event=state->event_driver->record(identity.event,identity.stream);
 if(initial_event.ok())initial_event=qwen_plugin::MemoryStatus(
  async.synchronize_stream(async.context,identity.context,identity.stream));
 if(!initial_event.ok())return initial_event;
 QwenInt4PackedBackendArenas packed_arenas{
  backend_arenas->device,backend_arenas->pinned_staging,
  backend_arenas->device_staging,backend_arenas->sampled_token,
  backend_arenas->device_error,backend_arenas->pinned_result,
  backend_arenas->pinned_staging_backing,
  backend_arenas->pinned_result_backing};
 QwenInt4PackedBackendIdentity packed_identity{
  1,1,1,1,kOperationTimeoutNs,identity.context,identity.stream,
  identity.event,state->rank,kSlotCount,kMaximumBatchSequences,
  static_cast<float>(state->bootstrap->model().config().rms_norm_epsilon),
  1.0F/std::sqrt(static_cast<float>(
      state->bootstrap->model().config().head_dim))};
 QwenInt4PackedBackendDrivers packed_drivers{
  &*state->copy_driver,&*state->clear_driver,&*state->kernel_driver,
  &*state->execution,&*state->event_driver,&*state->health,
  &state->clock,&state->waiter};
 auto packed_backend=QwenInt4PackedSynchronousBackend::Create(
  *state->commands,state->kernels->functions(),
  state->packed_kernels->packed_functions(),**weights,packed_arenas,
  packed_identity,packed_drivers);
 if(!packed_backend.ok())return packed_backend.status();
 state->packed_backend.emplace(std::move(*packed_backend));
 auto resource_hash=packed_resource_vector_hash(*capacity);
 if(!resource_hash.ok())return resource_hash.status();
 auto controller=qwen_plugin::ControllerClient::CreateBound(
  {1,kMaximumBatchSequences,{{64,128},
    {kMaximumBatchSequences,kMaximumSequenceTokens,kMaximumSequenceTokens}},
   {kMaximumBatchSequences,kMaximumBatchSequences,kMaximumStepTokens,
    kMaximumBatchSequences,8,5'000'000},
   {kMaximumBatchSequences,kMaximumStepTokens},kMaximumStepTokens,true,
   "qwen3-int4-fused-packed-v1",*resource_hash,
   static_cast<std::uint32_t>(state->bootstrap->model().config().eos_token_id)});
 if(!controller.ok())return controller.status();
 state->controller.emplace(std::move(*controller));
 auto controller_driver=ControllerPackedDriver::Create(
  *state->controller,*state->packed_backend,kMaximumBatchSequences,
  {kMaximumBatchSequences,kMaximumStepTokens,kSlotCount});
 if(!controller_driver.ok())return controller_driver.status();
 state->controller_driver.emplace(std::move(*controller_driver));
 auto admission=QwenKvAdmissionCoordinator::Create(
  *state->pool,*state->controller_driver,kMaximumBatchSequences,
  {identity.event,1});
 if(!admission.ok())return admission.status();
 state->admission_coordinator.emplace(std::move(*admission));
 return std::unique_ptr<NvidiaQwen3Int4Engine>(new NvidiaQwen3Int4Engine(std::move(state)));
}

NvidiaQwen3Int4Engine::NvidiaQwen3Int4Engine(std::unique_ptr<State> state):state_(std::move(state)){}
NvidiaQwen3Int4Engine::~NvidiaQwen3Int4Engine()=default;
Result<std::uint64_t> NvidiaQwen3Int4Engine::submit_packed(
 std::span<const std::int64_t> prompt,std::uint32_t maximum_new_tokens,
 const ControllerRequestSampling& sampling) {
 if(!state_||state_->poisoned.load())
  return Status::FailedPrecondition("packed Qwen INT4 runtime is not ready");
 auto lease=state_->close_gate.enter();if(!lease.ok())return lease.status();
 if(prompt.empty()||
    prompt.size()>kMaximumSequenceTokens||maximum_new_tokens==0||
    maximum_new_tokens>kMaximumSequenceTokens||
    prompt.size()>kMaximumSequenceTokens-maximum_new_tokens)
  return Status::InvalidArgument("packed Qwen INT4 request bounds are invalid");
 std::vector<std::uint32_t> tokens;tokens.reserve(prompt.size());
 for(const auto token:prompt){
  if(token<0||token>=QwenBf16PackedResultLayout::kVocabularySize)
   return Status::InvalidArgument("packed Qwen INT4 prompt token is invalid");
  tokens.push_back(static_cast<std::uint32_t>(token));
 }
 std::lock_guard submit(state_->packed_submit_mutex);
 const Status still_open=state_->close_gate.require_open();
 if(!still_open.ok()||state_->poisoned.load())
  return Status::FailedPrecondition("packed Qwen INT4 runtime is not ready");
 if(state_->next_packed_request_generation==UINT64_MAX){
  state_->poisoned.store(true);
  return Status::ResourceExhausted("packed Qwen INT4 generation is exhausted");
 }
 const auto generation=state_->next_packed_request_generation;
 auto submitted=state_->controller->submit_admit(
  generation,tokens,maximum_new_tokens,sampling);
 if(!submitted.ok())return submitted;
 ++state_->next_packed_request_generation;return generation;
}
Status NvidiaQwen3Int4Engine::cancel_packed(std::uint64_t generation){
 if(!state_||state_->poisoned.load())
  return Status::FailedPrecondition("packed Qwen INT4 runtime is not ready");
 auto lease=state_->close_gate.enter();if(!lease.ok())return lease.status();
 auto result=state_->controller->submit_cancel(generation);
 return result;
}
Result<NvidiaQwenPackedStep> NvidiaQwen3Int4Engine::drive_packed(){
 if(!state_||state_->poisoned.load())
  return Status::FailedPrecondition("packed Qwen INT4 runtime is not ready");
 auto lease=state_->close_gate.enter();if(!lease.ok())return lease.status();
 std::unique_lock operation(state_->operation_mutex,std::try_to_lock);
 if(!operation.owns_lock())
  return Status::Unavailable("Qwen INT4 engine operation is already active");
 const Status still_open=state_->close_gate.require_open();
 if(!still_open.ok())return still_open;
 auto now=state_->clock.now_ns();
 if(!now.ok()||*now>static_cast<std::uint64_t>(INT64_MAX)){
  state_->poisoned.store(true);
  return now.ok()?Status::Internal("packed Qwen INT4 clock overflowed"):now.status();
 }
 if(state_->controller->queued_command_count()!=0){
  auto stepped=state_->controller->step(static_cast<std::int64_t>(*now),
                                         *state_->admission_coordinator);
  if(!stepped.ok()){
   if(stepped.status().code()!=StatusCode::kResourceExhausted)
    state_->poisoned.store(true);
   return stepped.status();
  }
  if(*stepped==ControllerRuntimeStep::kOutputBackpressured)
   return NvidiaQwenPackedStep::kOutputBackpressured;
  if(*stepped==ControllerRuntimeStep::kIdle)return NvidiaQwenPackedStep::kIdle;
  return NvidiaQwenPackedStep::kCommandProcessed;
 }
 auto executed=state_->controller_driver->execute_next(
  static_cast<std::int64_t>(*now));
 if(!executed.ok()){state_->poisoned.store(true);return executed.status();}
 if(*executed==ControllerPackedDriverStep::kIdle)
  return NvidiaQwenPackedStep::kIdle;
 if(*executed==ControllerPackedDriverStep::kOutputBackpressured)
  return NvidiaQwenPackedStep::kOutputBackpressured;
 return NvidiaQwenPackedStep::kBatchCompleted;
}
Result<std::optional<NvidiaQwenPackedOutputEvent>>
NvidiaQwen3Int4Engine::try_take_packed_event(){
 if(!state_||state_->poisoned.load())return Status::FailedPrecondition(
  "packed Qwen INT4 runtime is not ready");
 auto lease=state_->close_gate.enter();if(!lease.ok())return lease.status();
 auto event=state_->controller->try_take_event();if(!event.ok())return event.status();
 if(!event->has_value())return std::optional<NvidiaQwenPackedOutputEvent>{};
 std::optional<ControllerPackedSamplingReceipt> receipt;
 if((*event)->kind==ControllerOutputEventKind::kTokenCommitted){
  auto taken=state_->controller_driver->try_take_sampling_receipt();
  if(!taken.ok())return taken.status();
  if(!taken->has_value()||(**taken).request_generation!=(*event)->request_generation||
     (**taken).token_ordinal!=(*event)->token_ordinal||
     (**taken).sample.token_id!=(*event)->token_id){
   state_->poisoned.store(true);
   return Status::Internal("packed Qwen INT4 sampling receipt drifted");
  }
  receipt=**taken;
 }
 return std::optional<NvidiaQwenPackedOutputEvent>{
  NvidiaQwenPackedOutputEvent{**event,std::move(receipt)}};
}
Status NvidiaQwen3Int4Engine::acknowledge_packed_output(std::uint64_t plan){
 if(!state_||state_->poisoned.load())
  return Status::FailedPrecondition("packed Qwen INT4 runtime is not ready");
 auto lease=state_->close_gate.enter();if(!lease.ok())return lease.status();
 return state_->controller->acknowledge_output_plan(plan);
}
Status NvidiaQwen3Int4Engine::drain_packed(std::uint64_t generation){
 if(!state_||state_->poisoned.load())
  return Status::FailedPrecondition("packed Qwen INT4 runtime is not ready");
 auto lease=state_->close_gate.enter();if(!lease.ok())return lease.status();
 std::unique_lock operation(state_->operation_mutex,std::try_to_lock);
 if(!operation.owns_lock())
  return Status::Unavailable("Qwen INT4 engine operation is already active");
 const Status still_open=state_->close_gate.require_open();
 if(!still_open.ok())return still_open;
 const Status drained=state_->admission_coordinator->drain(
  generation,*state_->recycler);
 if(!drained.ok()&&drained.code()!=StatusCode::kResourceExhausted)
  state_->poisoned.store(true);
 return drained;
}
Status NvidiaQwen3Int4Engine::close(){
 if(!state_)return Status::Ok();
 const bool owns_close=state_->close_gate.request_close();
 if(!owns_close)return state_->close_gate.wait_closed();
 state_->close_gate.wait_drained();
 std::scoped_lock lock(state_->operation_mutex,state_->packed_submit_mutex);
 Status closed=Status::Ok();
 if(state_->controller&&
    (state_->controller->active_sequence_count()!=0||
     state_->controller->queued_command_count()!=0))
  closed=Status::FailedPrecondition(
   "Qwen INT4 engine cannot close with undrained packed requests");

 if(closed.ok()&&state_->controller)
  closed=state_->controller->close();
 if(!closed.ok())state_->poisoned.store(true);
 state_->close_gate.finish_close(closed);
 return closed;
}
}
