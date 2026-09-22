#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_linear_materializer.h"
#include "pih/model/qwen3_int4_packed_linear_materializer.h"
#include "pih/model/qwen3_int4_packed_lm_head_materializer.h"
#include "pih/model/qwen3_int4_packed_prepared_execution.h"
#include "pih/model/qwen3_bf16_kernel_materializer.h"
#include "pih/model/qwen3_int4_prepared_execution.h"
#include "pih/model/qwen3_int4_step_builder.h"
#include "pih/model/qwen3_int4_synchronous_backend.h"
#include "pih/model/qwen3_int4_engine_backend_factory.h"

namespace pih {
namespace {

TensorView view(std::uintptr_t address,DType dtype,
                std::span<const std::int64_t> shape,
                std::uint64_t generation=9) {
  return TensorView::Create(reinterpret_cast<void*>(address),dtype,shape,{},
      Device::Create(DeviceType::kCuda,0).value(),generation).value();
}

QwenBf16ResourceSet activations() {
  const std::array<std::int64_t,1> two{2},one{1},four{4},sixteen{16},thirty_two{32},kv_bytes{3'670'016};
  const std::array<std::int64_t,2> hidden{2,1024},mlp{2,3072},angles{2,64},logits{1,151936};
  const std::array<std::int64_t,3> query{2,16,128},kv{2,8,128};
  std::vector<QwenBf16SlotResource> e{
    {QwenBf16ActivationSlot::kTokenIds,view(0x100000000,DType::kInt64,two)},
    {QwenBf16ActivationSlot::kPositions,view(0x200000000,DType::kInt64,two)},
    {QwenBf16ActivationSlot::kHidden,view(0x300000000,DType::kBFloat16,hidden)},
    {QwenBf16ActivationSlot::kNormalized,view(0x400000000,DType::kBFloat16,hidden)},
    {QwenBf16ActivationSlot::kQuery,view(0x500000000,DType::kBFloat16,query)},
    {QwenBf16ActivationSlot::kKey,view(0x600000000,DType::kBFloat16,kv)},
    {QwenBf16ActivationSlot::kValue,view(0x700000000,DType::kBFloat16,kv)},
    {QwenBf16ActivationSlot::kAttention,view(0x800000000,DType::kBFloat16,query)},
    {QwenBf16ActivationSlot::kGate,view(0x900000000,DType::kBFloat16,mlp)},
    {QwenBf16ActivationSlot::kUp,view(0xa00000000,DType::kBFloat16,mlp)},
    {QwenBf16ActivationSlot::kRopeCosine,view(0xb00000000,DType::kFloat32,angles)},
    {QwenBf16ActivationSlot::kRopeSine,view(0xc00000000,DType::kFloat32,angles)},
    {QwenBf16ActivationSlot::kKvBacking,view(0xd00000000,DType::kUInt8,kv_bytes)},
    {QwenBf16ActivationSlot::kKvSlotStates,view(0xe00000000,DType::kUInt8,thirty_two)},
    {QwenBf16ActivationSlot::kKvAppendHandles,view(0xf00000000,DType::kUInt8,sixteen)},
    {QwenBf16ActivationSlot::kKvVisibleHandles,view(0x1000000000,DType::kUInt8,sixteen)},
    {QwenBf16ActivationSlot::kKvTokenOffsets,view(0x1100000000,DType::kUInt8,four)},
    {QwenBf16ActivationSlot::kDeviceError,view(0x1200000000,DType::kUInt8,four)},
    {QwenBf16ActivationSlot::kLogits,view(0x1300000000,DType::kFloat32,logits)},
    {QwenBf16ActivationSlot::kSampledToken,view(0x1400000000,DType::kInt64,one)}};
  return QwenBf16ResourceSet::Create(9,0,e).value();
}

QwenInt4WeightResourceSet weights() {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  const std::array<std::int64_t,1> shape{
      static_cast<std::int64_t>(layout.logical_payload_bytes())};
  auto backing=view(0x20000000000,DType::kUInt8,shape,17);
  return QwenInt4WeightResourceSet::Create(layout,ledger,backing,0).value();
}

QwenBf16PackedResourceSet packed_activations() {
  auto execution=QwenBf16ExecutionArenaLayout::Create(8,3).value();
  auto staging=QwenBf16PackedStepStagingLayout::CreateBounded(8,3,4).value();
  auto result=QwenBf16PackedResultLayout::Create(3).value();
  const auto owner=[](std::uintptr_t base,std::uint64_t bytes,
                      std::uint64_t generation) {
    return QwenBf16DeviceArenaOwner{base,bytes,generation};
  };
  const QwenBf16StepDeviceOwners owners{
      owner(0x100000,staging.total_bytes(),41),
      owner(0x200000,execution.activation_arena_bytes(),2),
      owner(0x300000,execution.mlp().arena_bytes(),3),
      owner(0x400000,execution.rope_workspace_bytes(),4),
      owner(0x500000,execution.logit_workspace_bytes()*2,5),
      owner(0x900000,result.device_error().offset_bytes,6),
      owner(0xa00000,4,41),
      owner(0x1000000,2*QwenKvSlotPool::kSlotPayloadBytes,8),
      owner(0x1500000,2*sizeof(QwenKvSlotState),9)};
  return QwenBf16PackedResourceFactory::Create(
      41,0,2,3,execution,staging,owners).value();
}

ResolvedKernelFunction function() {
  auto plan=QwenInt4GemmPlan::Create(QwenInt4LinearShapeFamily::kQProj,2).value();
  const std::string cubin(64,'c'); auto signature=plan.signature(cubin).value();
  return {std::string(plan.kernel_symbol()),std::string(plan.logical_id()),cubin,
          std::string(signature.parameter_abi_sha256()),71};
}

ResolvedKernelFunction bf16_function(QwenBf16Primitive primitive) {
  const std::string cubin(64,'d');
  auto manifest=qwen_bf16_kernel_manifest(primitive,cubin).value();
  return {std::string(qwen_bf16_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()),cubin,
          std::string(manifest.parameter_abi_sha256()),81};
}

ResolvedKernelFunction packed_function(QwenBf16PackedPrimitive primitive) {
  const std::string cubin(64,'e');
  auto manifest=qwen_bf16_packed_kernel_manifest(primitive,cubin).value();
  return {std::string(qwen_bf16_packed_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()),cubin,
          std::string(manifest.parameter_abi_sha256()),82};
}

std::vector<ResolvedKernelFunction> packed_functions() {
  std::vector<ResolvedKernelFunction> result;
  for(std::size_t i=0;i<QwenBf16KernelBundle::kPackedPrimitiveCount;++i)
    result.push_back(packed_function(
        static_cast<QwenBf16PackedPrimitive>(i+1)));
  return result;
}

QwenBf16CommandBuffer commands() {
  Qwen3Config config{1024,3072,28,16,8,128,151936,40960,1'000'000.0,
                     0.000001,151643,151645};
  auto schedule=QwenBf16ExecutionSchedule::Create(config).value();
  auto bindings=QwenBf16WeightBindingPlan::Create(schedule).value();
  return QwenBf16CommandBuffer::Create(schedule,bindings).value();
}

TEST(QwenInt4LinearMaterializerTest, MaterializesAll196QuantizedLinears) {
  auto a=activations(); auto w=weights(); auto f=function(); auto c=commands();
  std::size_t quantized=0,lm_head=0;
  for(const auto& command:c) {
    if(command.backend!=QwenBf16CommandBackend::kLinear)continue;
    auto plan=QwenInt4LinearMaterializer::Create(command,f,a,w,9);
    if(command.linear_kind==QwenBf16LinearKind::kLmHead) {
      EXPECT_FALSE(plan.ok()); ++lm_head;
    } else {
      ASSERT_TRUE(plan.ok())<<plan.status().message(); ++quantized;
      EXPECT_EQ(plan->geometry().block_x(),256U);
    }
  }
  EXPECT_EQ(quantized,196U); EXPECT_EQ(lm_head,1U);
}

TEST(QwenInt4LinearMaterializerTest, RejectsGenerationAndBackendDrift) {
  auto a=activations(); auto w=weights(); auto f=function(); auto c=commands();
  const auto linear=std::find_if(c.begin(),c.end(),[](const auto& command){
    return command.backend==QwenBf16CommandBackend::kLinear &&
           command.linear_kind!=QwenBf16LinearKind::kLmHead;
  });
  ASSERT_NE(linear,c.end());
  EXPECT_FALSE(QwenInt4LinearMaterializer::Create(*linear,f,a,w,8).ok());
  auto wrong=*linear; wrong.backend=QwenBf16CommandBackend::kKernel;
  EXPECT_FALSE(QwenInt4LinearMaterializer::Create(wrong,f,a,w,9).ok());
}

TEST(QwenInt4PackedLinearMaterializerTest, BindsTheWholeExecutionBucket) {
  auto resources=packed_activations(); auto w=weights(); auto f=function();
  auto c=commands();
  const auto query=std::find_if(c.begin(),c.end(),[](const auto& command) {
    return command.backend==QwenBf16CommandBackend::kLinear &&
           command.linear_kind==QwenBf16LinearKind::kQuery;
  });
  ASSERT_NE(query,c.end());
  auto plan=QwenInt4PackedLinearMaterializer::Create(
      *query,f,resources,w,41);
  ASSERT_TRUE(plan.ok())<<plan.status().message();
  std::uint64_t rows=0;
  std::memcpy(&rows,plan->arguments().argument_cell(5),sizeof(rows));
  EXPECT_EQ(rows,8U);
  EXPECT_EQ(plan->geometry().grid_x(),64U);
  EXPECT_FALSE(QwenInt4PackedLinearMaterializer::Create(
      *query,f,resources,w,42).ok());
}

TEST(QwenInt4PackedLinearMaterializerTest, KeepsBf16LmHeadOutOfW4Path) {
  auto resources=packed_activations(); auto w=weights(); auto f=function();
  auto c=commands();
  const auto lm_head=std::find_if(c.begin(),c.end(),[](const auto& command) {
    return command.backend==QwenBf16CommandBackend::kLinear &&
           command.linear_kind==QwenBf16LinearKind::kLmHead;
  });
  ASSERT_NE(lm_head,c.end());
  EXPECT_FALSE(QwenInt4PackedLinearMaterializer::Create(
      *lm_head,f,resources,w,41).ok());
}

TEST(QwenInt4PackedLmHeadMaterializerTest, ConsumesGatheredSamplePrefix) {
  auto resources=packed_activations(); auto w=weights(); auto c=commands();
  const auto lm_head=std::find_if(c.begin(),c.end(),[](const auto& command) {
    return command.backend==QwenBf16CommandBackend::kLinear &&
           command.linear_kind==QwenBf16LinearKind::kLmHead;
  });
  ASSERT_NE(lm_head,c.end());
  auto binding=QwenInt4PackedLmHeadMaterializer::Create(
      *lm_head,resources,w,41);
  ASSERT_TRUE(binding.ok())<<binding.status().message();
  auto hidden=resources.activations().view(QwenBf16ActivationSlot::kHidden);
  ASSERT_TRUE(hidden.ok());
  EXPECT_EQ(binding->input().data(),hidden->data());
  EXPECT_EQ(binding->input().dim(0),3);
  EXPECT_EQ(binding->input().dim(1),1024);
  EXPECT_EQ(binding->output().dim(0),3);
  EXPECT_EQ(binding->output().dim(1),151936);
  EXPECT_FALSE(QwenInt4PackedLmHeadMaterializer::Create(
      *lm_head,resources,w,42).ok());
}

TEST(QwenInt4KernelMaterializerTest, MaterializesAll312RetainedKernels) {
  auto a=activations(); auto w=weights(); auto c=commands();
  const QwenBf16KernelContext context{9,23,15,17,2,0.000001F,
                                      0.0883883476F};
  std::size_t kernels=0;
  for(const auto& command:c) {
    if(command.backend!=QwenBf16CommandBackend::kKernel)continue;
    auto plan=QwenBf16KernelMaterializer::Create(
        command,bf16_function(command.primitive),a,w,context);
    ASSERT_TRUE(plan.ok())<<plan.status().message(); ++kernels;
  }
  EXPECT_EQ(kernels,312U);
}

class ClearDriver final : public QwenBf16DeviceErrorClearDriver {
 public:
  Status clear_u32_async(const TensorView&,std::int32_t,
                         DriverStreamHandle) override { ++calls; return result; }
  int calls=0; Status result=Status::Ok();
};
class KernelDriver final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle,const KernelLaunchGeometry&,
                DriverStreamHandle,void**) override { ++calls; return result; }
  int calls=0; Status result=Status::Ok();
};
class LmDriver final : public QwenInt4LmHeadExecutionDriver {
 public:
  Status execute(const QwenInt4LmHeadBinding&,DriverStreamHandle) override {
    ++calls; return result;
  }
  int calls=0; Status result=Status::Ok();
};

std::vector<ResolvedKernelFunction> mixed_functions() {
  std::vector<ResolvedKernelFunction> result;
  for(std::size_t i=0;i<QwenInt4KernelBundle::kBf16PrimitiveCount;++i)
    result.push_back(bf16_function(static_cast<QwenBf16Primitive>(i)));
  result.push_back(function()); return result;
}

TEST(QwenInt4PreparedExecutionTest, RunsFrozen509CommandMixOnce) {
  auto a=activations(); auto w=weights(); auto c=commands();
  auto functions=mixed_functions();
  const QwenBf16KernelContext context{9,23,15,17,2,0.000001F,
                                      0.0883883476F};
  auto execution=QwenInt4PreparedExecution::Create(c,functions,a,w,context);
  ASSERT_TRUE(execution.ok())<<execution.status().message();
  EXPECT_EQ(execution->size(),509U);
  EXPECT_EQ(execution->bf16_kernel_count(),312U);
  EXPECT_EQ(execution->int4_linear_count(),196U);
  EXPECT_EQ(execution->lm_head_count(),1U);
  ClearDriver clear; KernelDriver kernels; LmDriver lm;
  ASSERT_TRUE(execution->run(clear,kernels,lm,41).ok());
  EXPECT_EQ(clear.calls,1); EXPECT_EQ(kernels.calls,508); EXPECT_EQ(lm.calls,1);
  EXPECT_EQ(execution->state(),QwenInt4PreparedExecutionState::kCompleted);
  EXPECT_FALSE(execution->run(clear,kernels,lm,41).ok());
}

TEST(QwenInt4PreparedExecutionTest, KernelFailurePoisonsWithoutLmHead) {
  auto a=activations(); auto w=weights(); auto c=commands();
  auto functions=mixed_functions();
  const QwenBf16KernelContext context{9,23,15,17,2,0.000001F,
                                      0.0883883476F};
  auto execution=QwenInt4PreparedExecution::Create(c,functions,a,w,context)
      .value();
  ClearDriver clear; KernelDriver kernels; LmDriver lm;
  kernels.result=Status::Internal("injected kernel failure");
  EXPECT_FALSE(execution.run(clear,kernels,lm,41).ok());
  EXPECT_EQ(execution.state(),QwenInt4PreparedExecutionState::kPoisoned);
  EXPECT_EQ(kernels.calls,1); EXPECT_EQ(lm.calls,0);
}

TEST(QwenInt4PackedPreparedExecutionTest, RunsOneFusedMixedBucket) {
  auto resources=packed_activations(); auto w=weights(); auto c=commands();
  auto functions=mixed_functions(); auto packed=packed_functions();
  const QwenBf16PackedKernelContext context{
      41,7,3,2,0.000001F,0.0883883476F};
  auto execution=QwenInt4PackedPreparedExecution::Create(
      c,functions,packed,resources,w,context);
  ASSERT_TRUE(execution.ok())<<execution.status().message();
  EXPECT_EQ(execution->size(),510U);
  EXPECT_EQ(execution->legacy_kernel_count(),253U);
  EXPECT_EQ(execution->packed_kernel_count(),60U);
  EXPECT_EQ(execution->int4_linear_count(),196U);
  EXPECT_EQ(execution->lm_head_count(),1U);
  ClearDriver clear; KernelDriver kernels; LmDriver lm;
  ASSERT_TRUE(execution->run(clear,kernels,lm,41).ok());
  EXPECT_EQ(clear.calls,1);
  EXPECT_EQ(kernels.calls,509);
  EXPECT_EQ(lm.calls,1);
  EXPECT_EQ(execution->state(),
            QwenInt4PackedPreparedExecutionState::kCompleted);
  EXPECT_FALSE(execution->run(clear,kernels,lm,41).ok());
}

TEST(QwenInt4PackedPreparedExecutionTest, KernelFailurePoisonsBeforeLmHead) {
  auto resources=packed_activations(); auto w=weights(); auto c=commands();
  auto functions=mixed_functions(); auto packed=packed_functions();
  const QwenBf16PackedKernelContext context{
      41,7,3,2,0.000001F,0.0883883476F};
  auto execution=QwenInt4PackedPreparedExecution::Create(
      c,functions,packed,resources,w,context).value();
  ClearDriver clear; KernelDriver kernels; LmDriver lm;
  kernels.result=Status::Internal("injected packed INT4 failure");
  EXPECT_FALSE(execution.run(clear,kernels,lm,41).ok());
  EXPECT_EQ(execution.state(),
            QwenInt4PackedPreparedExecutionState::kPoisoned);
  EXPECT_EQ(kernels.calls,1);
  EXPECT_EQ(lm.calls,0);
  EXPECT_FALSE(execution.run(clear,kernels,lm,41).ok());
}

TEST(QwenInt4StepBuilderTest, BuildsMixedStepFromSharedQwenArenas) {
  const QwenKvBlockHandle handles[]={{4,9},{7,3}};
  auto table=QwenKvBlockTable::Create(23,6,32,handles).value();
  auto append=table.prepare_append(2).value();
  const std::int64_t token_ids[]={4,5};
  auto input=QwenBf16StepInputPlan::Create(token_ids,0,table,append).value();
  auto staging=QwenBf16StepStagingLayout::Create(input).value();
  auto arena=QwenBf16ExecutionArenaLayout::Create(2,1).value();
  const QwenBf16StepDeviceOwners owners{
      {0x2100000000,staging.total_bytes(),1},
      {0x2200000000,arena.activation_arena_bytes(),2},
      {0x2300000000,arena.mlp().arena_bytes(),3},
      {0x2400000000,arena.rope_workspace_bytes(),4},
      {0x2500000000,arena.logit_workspace_bytes(),5},
      {0x2600000000,sizeof(std::int64_t),6},
      {0x2700000000,sizeof(std::uint32_t),9},
      {0x2800000000,2*QwenKvSlotPool::kSlotPayloadBytes,8},
      {0x2900000000,2*sizeof(QwenKvSlotState),9}};
  auto c=commands(); auto w=weights(); auto functions=mixed_functions();
  std::vector<std::byte> pinned(staging.total_bytes()+256);

  auto built=QwenInt4StepBuilder::Create(
      token_ids,0,table,append,9,0,2,0.000001F,0.0883883476F,
      c,functions,w,owners,pinned);
  ASSERT_TRUE(built.ok())<<built.status().message();
  EXPECT_EQ(built->staging_layout().token_count(),2U);
  EXPECT_EQ(built->execution_layout().tokens(),2U);
  EXPECT_EQ(built->resources().request_generation(),9U);
  EXPECT_EQ(built->execution().size(),509U);
  ClearDriver clear; KernelDriver kernels; LmDriver lm;
  ASSERT_TRUE(built->execution().run(clear,kernels,lm,41).ok());
  EXPECT_EQ(clear.calls,1); EXPECT_EQ(kernels.calls,508); EXPECT_EQ(lm.calls,1);
}

class BackendCopies final : public TypedCopyDriver {
 public:
  BackendCopies(std::uintptr_t sampled,std::uintptr_t error)
      :sampled_(sampled),error_(error){}
  std::uintptr_t context_identity() const noexcept override{return 17;}
  Status copy(CudaCopyKind kind,std::uintptr_t destination,
              std::uintptr_t source,std::uint64_t bytes,
              DriverStreamHandle) override {
    if(kind==CudaCopyKind::kDeviceToHost && source==sampled_ && bytes==8) {
      const std::int64_t token=42;std::memcpy(reinterpret_cast<void*>(destination),&token,8);
    }
    if(kind==CudaCopyKind::kDeviceToHost && source==error_ && bytes==4) {
      const std::uint32_t clean=0;std::memcpy(reinterpret_cast<void*>(destination),&clean,4);
    }
    return Status::Ok();
  }
 private: std::uintptr_t sampled_,error_;
};
class BackendEvents final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle,DriverStreamHandle) override{return Status::Ok();}
  Result<CudaEventQueryResult> query(DriverEventHandle) override{
    if(!query_failure.ok())return query_failure;
    return CudaEventQueryResult::kSuccess;
  }
  Status query_failure=Status::Ok();
};
class BackendHealth final : public QwenBf16StepHealthProvider {
 public: Result<QwenBf16StepHealth> collect() override{
    return QwenBf16StepHealth{true,false};
  }
};
class BackendClock final : public QwenBf16MonotonicClock {
 public: Result<std::uint64_t> now_ns() override{return now++;}
  std::uint64_t now=100;
};
class BackendWaiter final : public QwenBf16PollWaiter {
 public: Status wait() override{++calls;return Status::Ok();}
  int calls=0;
};

CudaCopyEndpoint backend_endpoint(std::uintptr_t base,std::uint64_t bytes,
                                  std::uint64_t owner,std::uint64_t generation,
                                  CudaCopyMemoryType type) {
  return {base,bytes,0,owner,generation,type,0,0};
}

TEST(QwenInt4SynchronousBackendTest, ExecutesAndPublishesMixedTokenStep) {
  const QwenKvBlockHandle handles[]={{4,9},{7,3}};
  auto table=QwenKvBlockTable::Create(23,6,32,handles).value();
  auto append=table.prepare_append(2).value();
  const std::int64_t token_ids[]={4,5};
  auto input=QwenBf16StepInputPlan::Create(token_ids,0,table,append).value();
  auto staging=QwenBf16StepStagingLayout::Create(input).value();
  auto arena=QwenBf16ExecutionArenaLayout::Create(2,1).value();
  const QwenBf16StepDeviceOwners owners{
      {0x2100000000,staging.total_bytes(),1},{0x2200000000,arena.activation_arena_bytes(),2},
      {0x2300000000,arena.mlp().arena_bytes(),3},{0x2400000000,arena.rope_workspace_bytes(),4},
      {0x2500000000,arena.logit_workspace_bytes(),5},{0x2600000000,8,6},
      {0x2700000000,4,9},{0x2800000000,2*QwenKvSlotPool::kSlotPayloadBytes,8},
      {0x2900000000,2*sizeof(QwenKvSlotState),9}};
  alignas(256) std::array<std::byte,2048> pinned_staging{};
  alignas(256) std::array<std::byte,QwenBf16StepResultLayout::kTotalBytes> pinned_result{};
  BackendCopies copies(owners.sampled_token.base,owners.device_error.base);
  ClearDriver clear;KernelDriver kernels;LmDriver lm;BackendEvents events;
  BackendHealth health;BackendClock clock;BackendWaiter waiter;
  QwenInt4SynchronousBackendArenas arenas{
      owners,
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_staging.data()),pinned_staging.size(),101,1,CudaCopyMemoryType::kRegisteredPinnedHost),
      backend_endpoint(owners.step_staging.base,owners.step_staging.bytes,102,1,CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.sampled_token.base,8,103,6,CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.device_error.base,4,104,9,CudaCopyMemoryType::kDevice),
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_result.data()),pinned_result.size(),105,1,CudaCopyMemoryType::kRegisteredPinnedHost),
      pinned_staging,pinned_result};
  QwenInt4SynchronousBackendIdentity identity{1,9,23,100,1000,17,19,31,0,2,
                                               0.000001F,0.0883883476F};
  QwenInt4SynchronousBackendDrivers drivers{
      &copies,&clear,&kernels,&lm,&events,&health,&clock,&waiter};
  const Qwen3Config engine_config{1024,3072,28,16,8,128,151936,40960,
      1'000'000.0,0.000001,151643,151645};
  auto bootstrap=QwenInt4EngineBootstrapPlan::Create(engine_config,4096,40960,2560,
      UINT64_C(64)<<20,UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_C(1)<<30).value();
  auto f=mixed_functions();auto w=weights();
  auto backend=QwenInt4EngineBackendFactory::Create(
      bootstrap,f,w,arenas,identity,drivers);
  ASSERT_TRUE(backend.ok())<<backend.status().message();
  auto token=backend->execute_and_read_token(token_ids,0,table,append);
  ASSERT_TRUE(token.ok())<<token.status().message();EXPECT_EQ(*token,42);
  EXPECT_EQ(backend->next_request_generation(),10U);
  EXPECT_EQ(backend->state(),QwenInt4SynchronousBackendState::kReady);
  EXPECT_EQ(waiter.calls,0);

  events.query_failure=Status::Internal(
      "injected asynchronous completion query failure");
  auto asynchronous=QwenInt4EngineBackendFactory::Create(
      bootstrap,f,w,arenas,identity,drivers);
  ASSERT_TRUE(asynchronous.ok())<<asynchronous.status().message();
  auto asynchronous_result=asynchronous->execute_and_read_token(
      token_ids,0,table,append);
  ASSERT_FALSE(asynchronous_result.ok());
  EXPECT_EQ(asynchronous_result.status().code(),StatusCode::kInternal);
  EXPECT_EQ(asynchronous->state(),QwenInt4SynchronousBackendState::kPoisoned);
  events.query_failure=Status::Ok();
  EXPECT_EQ(asynchronous->execute_and_read_token(token_ids,0,table,append)
                .status().code(),StatusCode::kFailedPrecondition);

  kernels.result=Status::Internal("injected composed INT4 kernel failure");
  const auto calls_before_failure=kernels.calls;
  auto failed=backend->execute_and_read_token(token_ids,0,table,append);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.status().code(),StatusCode::kInternal);
  EXPECT_EQ(kernels.calls,calls_before_failure+1);
  EXPECT_EQ(backend->state(),QwenInt4SynchronousBackendState::kPoisoned);
  kernels.result=Status::Ok();
  auto replay=backend->execute_and_read_token(token_ids,0,table,append);
  ASSERT_FALSE(replay.ok());
  EXPECT_EQ(replay.status().code(),StatusCode::kFailedPrecondition);
  EXPECT_EQ(kernels.calls,calls_before_failure+1);

  auto exhausted_identity=identity;
  exhausted_identity.first_request_generation=UINT64_MAX;
  auto exhausted=QwenInt4EngineBackendFactory::Create(
      bootstrap,f,w,arenas,exhausted_identity,drivers);
  ASSERT_TRUE(exhausted.ok())<<exhausted.status().message();
  auto exhausted_result=exhausted->execute_and_read_token(
      token_ids,0,table,append);
  ASSERT_FALSE(exhausted_result.ok());
  EXPECT_EQ(exhausted_result.status().code(),StatusCode::kResourceExhausted);
  EXPECT_EQ(exhausted->state(),QwenInt4SynchronousBackendState::kPoisoned);
  EXPECT_EQ(exhausted->execute_and_read_token(token_ids,0,table,append)
                .status().code(),StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
