#include "pih/model/qwen3_bf16_packed_linear_materializer.h"
#include "pih/model/qwen3_bf16_packed_kernel_materializer.h"
#include "pih/model/qwen3_bf16_packed_prepared_execution.h"
#include "pih/model/qwen3_bf16_packed_step_builder.h"
#include "pih/model/qwen3_bf16_packed_synchronous_backend.h"
#include "pih/model/qwen3_int4_packed_synchronous_backend.h"

#include <algorithm>
#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_execution_schedule.h"
#include "pih/model/qwen3_bf16_weight_binding.h"
#include "pih/model/qwen3_config.h"

namespace pih {
namespace {

TensorView packed_weight_view(std::uintptr_t address,
                              std::span<const std::int64_t> shape) {
  return TensorView::Create(reinterpret_cast<void*>(address),
                            DType::kBFloat16, shape, {},
                            Device::Create(DeviceType::kCuda, 0).value(), 1)
      .value();
}

QwenBf16WeightResourceSet packed_weights() {
  std::vector<TensorView> views;
  std::uintptr_t address = 0x10000000000;
  for (const auto& tensor : Qwen3Manifest::expected_tensors()) {
    std::vector<std::int64_t> shape;
    for (const auto extent : tensor.shape)
      shape.push_back(static_cast<std::int64_t>(extent));
    views.push_back(packed_weight_view(address, shape));
    address += 0x100000000;
  }
  return QwenBf16WeightResourceSet::Create(0, views).value();
}

QwenInt4WeightResourceSet packed_int4_weights() {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  const std::array<std::int64_t,1> shape{
      static_cast<std::int64_t>(layout.logical_payload_bytes())};
  auto backing=TensorView::Create(reinterpret_cast<void*>(0x20000000000),
      DType::kUInt8,shape,{},Device::Create(DeviceType::kCuda,0).value(),17)
      .value();
  return QwenInt4WeightResourceSet::Create(layout,ledger,backing,0).value();
}

QwenBf16CommandBuffer packed_commands() {
  const Qwen3Config config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                           1'000'000.0, 0.000001, 151643, 151645};
  auto schedule = QwenBf16ExecutionSchedule::Create(config).value();
  auto weights = QwenBf16WeightBindingPlan::Create(schedule).value();
  return QwenBf16CommandBuffer::Create(schedule, weights).value();
}

QwenBf16PackedResourceSet linear_resources() {
  auto execution = QwenBf16ExecutionArenaLayout::Create(8, 3).value();
  auto staging =
      QwenBf16PackedStepStagingLayout::CreateBounded(8, 3, 4).value();
  auto result = QwenBf16PackedResultLayout::Create(3).value();
  const auto owner = [](std::uintptr_t base, std::uint64_t bytes,
                        std::uint64_t generation) {
    return QwenBf16DeviceArenaOwner{base, bytes, generation};
  };
  const QwenBf16StepDeviceOwners owners{
      owner(0x100000, staging.total_bytes(), 41),
      owner(0x200000, execution.activation_arena_bytes(), 2),
      owner(0x300000, execution.mlp().arena_bytes(), 3),
      owner(0x400000, execution.rope_workspace_bytes(), 4),
      owner(0x500000, execution.logit_workspace_bytes() * 2, 5),
      owner(0x900000, result.device_error().offset_bytes, 6),
      owner(0xA00000, 4, 41),
      owner(0x1000000, 2 * QwenKvSlotPool::kSlotPayloadBytes, 8),
      owner(0x1500000, 2 * sizeof(QwenKvSlotState), 9)};
  return QwenBf16PackedResourceFactory::Create(
      41, 0, 2, 3, execution, staging, owners).value();
}

ResolvedKernelFunction packed_kernel_function(
    QwenBf16PackedPrimitive primitive) {
  const std::string digest(64, 'e');
  auto manifest = qwen_bf16_packed_kernel_manifest(primitive, digest).value();
  return {std::string(qwen_bf16_packed_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), 73};
}

ResolvedKernelFunction legacy_kernel_function(QwenBf16Primitive primitive) {
  const std::string digest(64, 'f');
  auto manifest = qwen_bf16_kernel_manifest(primitive, digest).value();
  return {std::string(qwen_bf16_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), 74};
}

QwenBf16PackedPrimitive packed_primitive_for(
    QwenBf16ExecutionOp operation) {
  switch (operation) {
    case QwenBf16ExecutionOp::kEmbedding:
      return QwenBf16PackedPrimitive::kEmbedding;
    case QwenBf16ExecutionOp::kPrepareRopeAngles:
      return QwenBf16PackedPrimitive::kRopeAngles;
    case QwenBf16ExecutionOp::kKvAppend:
      return QwenBf16PackedPrimitive::kKvAppend;
    case QwenBf16ExecutionOp::kPagedGqa:
      return QwenBf16PackedPrimitive::kPagedGqa;
    case QwenBf16ExecutionOp::kGreedyArgmax:
      return QwenBf16PackedPrimitive::kSampler;
    default:
      return static_cast<QwenBf16PackedPrimitive>(255);
  }
}

std::array<ResolvedKernelFunction, QwenBf16KernelBundle::kExecutionPrimitiveCount>
legacy_functions() {
  std::array<ResolvedKernelFunction, QwenBf16KernelBundle::kExecutionPrimitiveCount>
      functions;
  for (std::size_t index = 0; index < functions.size(); ++index) {
    functions[index] =
        legacy_kernel_function(static_cast<QwenBf16Primitive>(index));
  }
  return functions;
}

std::array<ResolvedKernelFunction,
           QwenBf16KernelBundle::kPackedPrimitiveCount>
packed_functions() {
  std::array<ResolvedKernelFunction,
             QwenBf16KernelBundle::kPackedPrimitiveCount>
      functions;
  for (std::size_t index = 0; index < functions.size(); ++index) {
    functions[index] = packed_kernel_function(
        static_cast<QwenBf16PackedPrimitive>(index + 1));
  }
  return functions;
}

std::vector<ResolvedKernelFunction> packed_int4_functions() {
  auto legacy=legacy_functions();
  std::vector<ResolvedKernelFunction> result(legacy.begin(),legacy.end());
  auto plan=QwenInt4GemmPlan::Create(
      QwenInt4LinearShapeFamily::kQProj,8).value();
  const std::string digest(64,'c');
  auto signature=plan.signature(digest).value();
  result.push_back({std::string(plan.kernel_symbol()),
      std::string(plan.logical_id()),digest,
      std::string(signature.parameter_abi_sha256()),75});
  return result;
}

Sha256Digest packed_builder_digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

PackedTokenPlan packed_builder_plan() {
  const std::array sequences{
      PackedSequenceInput{11, 4, 2, 31, packed_builder_digest("first")},
      PackedSequenceInput{12, 9, 1, 32, packed_builder_digest("second")}};
  return PackedTokenPlan::Create(
             1, 1, PackedTokenPhase::kPrefill, "packed-builder-r1",
             packed_builder_digest("resources"), sequences, 8,
             PackedTokenPlanLimits{2, 8, 8})
      .value();
}

struct PackedBuilderMetadata final {
  std::array<std::uint32_t, 8> tokens{7, 8, 9, 0, 0, 0, 0, 0};
  std::array<std::uint64_t, 8> positions{4, 5, 9, 0, 0, 0, 0, 0};
  std::array<std::uint32_t, 8> request_indices{
      0, 0, 1, kInvalidPackedRequestIndex, kInvalidPackedRequestIndex,
      kInvalidPackedRequestIndex, kInvalidPackedRequestIndex,
      kInvalidPackedRequestIndex};
  std::array<std::uint32_t, 3> query_offsets{0, 2, 3};
  std::array<std::uint32_t, 1> sample_rows{2};
  std::array<QwenKvBlockHandle, 8> append_handles{
      QwenKvBlockHandle{0, 1}, QwenKvBlockHandle{0, 1},
      QwenKvBlockHandle{1, 1}, kInvalidPackedKvBlockHandle,
      kInvalidPackedKvBlockHandle, kInvalidPackedKvBlockHandle,
      kInvalidPackedKvBlockHandle, kInvalidPackedKvBlockHandle};
  std::array<std::uint16_t, 8> token_offsets{4, 5, 9, 0, 0, 0, 0, 0};
  std::array<std::uint32_t, 3> visible_offsets{0, 1, 2};
  std::array<QwenKvBlockHandle, 2> visible_handles{
      QwenKvBlockHandle{0, 1}, QwenKvBlockHandle{1, 1}};
  std::array<std::uint32_t, 2> key_counts{6, 10};
  std::array<std::uint32_t, 2> owner_indices{0, 1};

  PackedTokenMetadataView tokens_view() {
    return {7, tokens, positions, request_indices, query_offsets, sample_rows,
            3};
  }
  QwenBf16PackedKvMetadataView kv_view() {
    return {7, append_handles, token_offsets, visible_offsets,
            visible_handles, key_counts, owner_indices};
  }
};

QwenBf16StepDeviceOwners packed_builder_owners(std::uint64_t staging_bytes) {
  auto execution = QwenBf16ExecutionArenaLayout::Create(8, 1).value();
  auto result = QwenBf16PackedResultLayout::Create(1).value();
  const auto owner = [](std::uintptr_t base, std::uint64_t bytes,
                        std::uint64_t generation) {
    return QwenBf16DeviceArenaOwner{base, bytes, generation};
  };
  return {owner(0x100000, staging_bytes, 41),
          owner(0x200000, execution.activation_arena_bytes(), 2),
          owner(0x300000, execution.mlp().arena_bytes(), 3),
          owner(0x400000, execution.rope_workspace_bytes(), 4),
          owner(0x500000, execution.logit_workspace_bytes() * 2, 5),
          owner(0x900000, result.device_error().offset_bytes, 6),
          owner(0xA00000, 4, 41),
          owner(0x1000000, 2 * QwenKvSlotPool::kSlotPayloadBytes, 8),
          owner(0x1500000, 2 * sizeof(QwenKvSlotState), 9)};
}

struct PackedExecutionTrace final {
  std::vector<char> events;
};

class PackedPreludeDriver final : public QwenBf16DeviceErrorClearDriver {
 public:
  explicit PackedPreludeDriver(PackedExecutionTrace& trace) : trace_(&trace) {}
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override {
    trace_->events.push_back('C');
    return Status::Ok();
  }

 private:
  PackedExecutionTrace* trace_;
};

class PackedKernelDriver final : public KernelLaunchDriver {
 public:
  explicit PackedKernelDriver(PackedExecutionTrace& trace) : trace_(&trace) {}
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override {
    trace_->events.push_back('K');
    ++calls;
    if (fail_on_call != 0 && calls == fail_on_call)
      return Status::Internal("injected packed kernel failure");
    return Status::Ok();
  }
  std::size_t calls = 0;
  std::size_t fail_on_call = 0;

 private:
  PackedExecutionTrace* trace_;
};

class PackedLinearDriver final : public QwenBf16LinearExecutionDriver {
 public:
  explicit PackedLinearDriver(PackedExecutionTrace& trace) : trace_(&trace) {}
  Status execute(const QwenBf16LinearBinding&,
                 DriverStreamHandle) override {
    trace_->events.push_back('L');
    ++calls;
    return Status::Ok();
  }
  std::size_t calls = 0;

 private:
  PackedExecutionTrace* trace_;
};

class PackedInt4LmDriver final : public QwenInt4LmHeadExecutionDriver {
 public:
  explicit PackedInt4LmDriver(PackedExecutionTrace& trace):trace_(&trace){}
  Status execute(const QwenInt4LmHeadBinding&,DriverStreamHandle) override{
    trace_->events.push_back('H');++calls;return Status::Ok();
  }
  std::size_t calls=0;
 private:PackedExecutionTrace* trace_;
};

class PackedCopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    kind == CudaCopyKind::kHostToDevice ? ++uploads : ++downloads;
    return Status::Ok();
  }
  std::size_t uploads = 0, downloads = 0;
};

class PackedEventDriver final : public CompletionEventDriver {
 public:
  PackedEventDriver(std::span<std::byte> result,
                    QwenBf16PackedResultLayout layout)
      : result_(result), layout_(layout) {}
  Status record(DriverEventHandle, DriverStreamHandle) override {
    ++records;
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    if (always_not_ready) {
      ++queries;
      return CudaEventQueryResult::kNotReady;
    }
    if (queries++ == 0) return CudaEventQueryResult::kNotReady;
    const std::uint32_t token = 42, error = 0, rng = 0, top_count = 1;
    const float logprob = -0.25F;
    std::memcpy(result_.data() + layout_.sampled_token_ids().offset_bytes,
                &token, sizeof(token));
    std::memcpy(result_.data() + layout_.selected_logprobs().offset_bytes,
                &logprob, sizeof(logprob));
    std::memcpy(result_.data() + layout_.rng_words().offset_bytes,
                &rng, sizeof(rng));
    std::memcpy(
        result_.data() + layout_.top_logprob_token_ids().offset_bytes,
        &token, sizeof(token));
    std::memcpy(result_.data() + layout_.top_logprobs().offset_bytes,
                &logprob, sizeof(logprob));
    std::memcpy(result_.data() + layout_.top_logprob_counts().offset_bytes,
                &top_count, sizeof(top_count));
    std::memcpy(result_.data() + layout_.device_error().offset_bytes, &error,
                sizeof(error));
    return CudaEventQueryResult::kSuccess;
  }
  int records = 0, queries = 0;
  bool always_not_ready = false;

 private:
  std::span<std::byte> result_;
  QwenBf16PackedResultLayout layout_;
};

class PackedHealth final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    return QwenBf16StepHealth{true, false};
  }
};

class PackedClock final : public QwenBf16MonotonicClock {
 public:
  Result<std::uint64_t> now_ns() override { return now++; }
  std::uint64_t now = 100;
};

class PackedWaiter final : public QwenBf16PollWaiter {
 public:
  Status wait() override { ++calls; return Status::Ok(); }
  int calls = 0;
};

CudaCopyEndpoint packed_endpoint(std::uintptr_t base, std::uint64_t bytes,
                                 std::uint64_t owner,
                                 std::uint64_t generation,
                                 CudaCopyMemoryType type) {
  return {base, bytes, 0, owner, generation, type, 0, 0};
}

TEST(QwenBf16PackedLinearMaterializerTest, LmHeadConsumesGatheredPrefix) {
  auto commands = packed_commands();
  auto resources = linear_resources();
  auto weights = packed_weights();
  const auto& lm_head = commands[507];
  ASSERT_EQ(lm_head.linear_kind, QwenBf16LinearKind::kLmHead);
  auto binding = QwenBf16PackedLinearMaterializer::Create(
      lm_head, resources, weights, 41);
  ASSERT_TRUE(binding.ok()) << binding.status().message();
  auto hidden = resources.activations().view(QwenBf16ActivationSlot::kHidden);
  ASSERT_TRUE(hidden.ok());
  EXPECT_EQ(binding->input().data(), hidden->data());
  EXPECT_EQ(binding->input().dim(0), 3);
  EXPECT_EQ(binding->input().dim(1), 1024);
  EXPECT_EQ(binding->output().dim(0), 3);
  EXPECT_EQ(binding->output().dim(1), 151936);
}

TEST(QwenBf16PackedLinearMaterializerTest, ReusesBucketRowsForLayerLinears) {
  auto commands = packed_commands();
  auto resources = linear_resources();
  auto weights = packed_weights();
  const auto& query = commands[3];
  ASSERT_EQ(query.linear_kind, QwenBf16LinearKind::kQuery);
  auto binding = QwenBf16PackedLinearMaterializer::Create(
      query, resources, weights, 41);
  ASSERT_TRUE(binding.ok()) << binding.status().message();
  EXPECT_EQ(binding->input().dim(0), 8);
  EXPECT_EQ(binding->output().dim(0), 8);
  EXPECT_FALSE(QwenBf16PackedLinearMaterializer::Create(
      query, resources, weights, 42).ok());
}

TEST(QwenBf16PackedKernelMaterializerTest, CoversFrozenKernelCommandTable) {
  auto commands = packed_commands();
  auto resources = linear_resources();
  auto weights = packed_weights();
  const QwenBf16PackedKernelContext context{
      41, 5, 3, 2, 0.000001F, 0.0883883476F};
  std::size_t packed_count = 0;
  std::size_t legacy_count = 0;
  for (const auto& command : commands) {
    if (command.backend != QwenBf16CommandBackend::kKernel) continue;
    if (QwenBf16PackedKernelMaterializer::uses_packed_primitive(command)) {
      const auto primitive =
          packed_primitive_for(command.execution_step.operation);
      auto plan = QwenBf16PackedKernelMaterializer::CreatePacked(
          command, packed_kernel_function(primitive), resources, weights,
          context);
      ASSERT_TRUE(plan.ok())
          << static_cast<int>(command.execution_step.operation) << ": "
          << plan.status().message();
      ++packed_count;
    } else {
      auto plan = QwenBf16PackedKernelMaterializer::CreateLegacy(
          command, legacy_kernel_function(command.primitive), resources,
          weights, context);
      ASSERT_TRUE(plan.ok())
          << static_cast<int>(command.execution_step.operation) << ": "
          << plan.status().message();
      ++legacy_count;
    }
  }
  EXPECT_EQ(packed_count, 59U);
  EXPECT_EQ(legacy_count, 253U);
  auto gather = QwenBf16PackedKernelMaterializer::CreateSampleHidden(
      packed_kernel_function(QwenBf16PackedPrimitive::kSampleHidden),
      resources, context);
  ASSERT_TRUE(gather.ok()) << gather.status().message();
  EXPECT_EQ(gather->primitive(), QwenBf16PackedPrimitive::kSampleHidden);
}

TEST(QwenBf16PackedPreparedExecutionTest, RunsCompletePackedCommandTableOnce) {
  auto commands = packed_commands();
  auto resources = linear_resources();
  auto weights = packed_weights();
  auto legacy = legacy_functions();
  auto packed = packed_functions();
  const QwenBf16PackedKernelContext context{
      41, 5, 3, 2, 0.000001F, 0.0883883476F};
  auto execution = QwenBf16PackedPreparedExecution::Create(
      commands, legacy, packed, resources, weights, context);
  ASSERT_TRUE(execution.ok()) << execution.status().message();
  EXPECT_EQ(execution->size(), 510U);
  EXPECT_EQ(execution->next_command(), 0U);

  PackedExecutionTrace trace;
  PackedPreludeDriver prelude(trace);
  PackedKernelDriver kernels(trace);
  PackedLinearDriver linears(trace);
  ASSERT_TRUE(execution->submit(prelude, kernels, linears, 0x55).ok());
  EXPECT_EQ(trace.events.front(), 'C');
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'K'), 313);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'L'), 197);
  EXPECT_EQ(execution->next_command(), 510U);
  EXPECT_EQ(execution->state(),
            QwenBf16PackedPreparedExecutionState::kCompleted);

  const auto event_count = trace.events.size();
  EXPECT_FALSE(execution->submit(prelude, kernels, linears, 0x55).ok());
  EXPECT_EQ(trace.events.size(), event_count);
}

TEST(QwenBf16PackedPreparedExecutionTest, FailurePoisonsWithoutReplay) {
  auto commands = packed_commands();
  auto resources = linear_resources();
  auto weights = packed_weights();
  auto legacy = legacy_functions();
  auto packed = packed_functions();
  const QwenBf16PackedKernelContext context{
      41, 5, 3, 2, 0.000001F, 0.0883883476F};
  auto execution = QwenBf16PackedPreparedExecution::Create(
      commands, legacy, packed, resources, weights, context);
  ASSERT_TRUE(execution.ok()) << execution.status().message();

  PackedExecutionTrace trace;
  PackedPreludeDriver prelude(trace);
  PackedKernelDriver kernels(trace);
  PackedLinearDriver linears(trace);
  kernels.fail_on_call = 2;
  EXPECT_FALSE(execution->submit(prelude, kernels, linears, 7).ok());
  const auto stopped = execution->next_command();
  const auto event_count = trace.events.size();
  EXPECT_EQ(execution->state(),
            QwenBf16PackedPreparedExecutionState::kPoisoned);
  EXPECT_FALSE(execution->submit(prelude, kernels, linears, 7).ok());
  EXPECT_EQ(execution->next_command(), stopped);
  EXPECT_EQ(trace.events.size(), event_count);
}

TEST(QwenBf16PackedStepBuilderTest, BuildsPreparedStepFromOnePackedIdentity) {
  auto plan = packed_builder_plan();
  PackedBuilderMetadata metadata;
  auto layout = QwenBf16PackedStepStagingLayout::Create(
                    metadata.tokens_view(), metadata.kv_view())
                    .value();
  std::vector<std::byte> backing(layout.total_bytes());
  auto commands = packed_commands();
  auto weights = packed_weights();
  auto legacy = legacy_functions();
  auto packed = packed_functions();
  auto owners = packed_builder_owners(layout.total_bytes());
  auto built = QwenBf16PackedStepBuilder::Create(
      plan, metadata.tokens_view(), metadata.kv_view(), 41, 0, 2,
      0.000001F, 0.0883883476F, commands, legacy, packed, weights, owners,
      backing);
  ASSERT_TRUE(built.ok()) << built.status().message();
  EXPECT_EQ(built->staging_layout().execution_bucket_tokens(), 8U);
  EXPECT_EQ(built->execution_layout().active_logit_rows(), 1U);
  EXPECT_EQ(built->resources().sample_count(), 1U);
  EXPECT_EQ(built->compute().size(), 510U);

  PackedExecutionTrace trace;
  PackedPreludeDriver prelude(trace);
  PackedKernelDriver kernels(trace);
  PackedLinearDriver linears(trace);
  EXPECT_TRUE(built->compute().submit(prelude, kernels, linears, 9).ok());
}

TEST(QwenBf16PackedStepBuilderTest, RejectsSplitMetadataGenerations) {
  auto plan = packed_builder_plan();
  PackedBuilderMetadata metadata;
  auto token_view = metadata.tokens_view();
  auto kv_view = metadata.kv_view();
  kv_view.generation = token_view.generation + 1;
  auto layout = QwenBf16PackedStepStagingLayout::Create(
                    token_view, metadata.kv_view())
                    .value();
  std::vector<std::byte> backing(layout.total_bytes());
  auto commands = packed_commands();
  auto weights = packed_weights();
  auto legacy = legacy_functions();
  auto packed = packed_functions();
  auto owners = packed_builder_owners(layout.total_bytes());
  EXPECT_FALSE(QwenBf16PackedStepBuilder::Create(
                   plan, token_view, kv_view, 41, 0, 2, 0.000001F,
                   0.0883883476F, commands, legacy, packed, weights, owners,
                   backing)
                   .ok());
}

TEST(QwenBf16PackedStepBuilderTest,
     IntermediatePrefillSkipsSamplingCommands) {
  auto plan = packed_builder_plan();
  PackedBuilderMetadata metadata;
  auto tokens = metadata.tokens_view();
  tokens.sample_row_index = {};
  auto layout = QwenBf16PackedStepStagingLayout::Create(
                    tokens, metadata.kv_view()).value();
  std::vector<std::byte> backing(layout.total_bytes());
  auto commands = packed_commands();
  auto weights = packed_weights();
  auto legacy = legacy_functions();
  auto packed = packed_functions();
  auto owners = packed_builder_owners(layout.total_bytes());
  auto built = QwenBf16PackedStepBuilder::Create(
      plan, tokens, metadata.kv_view(), 41, 0, 2, 0.000001F,
      0.0883883476F, commands, legacy, packed, weights, owners, backing);
  ASSERT_TRUE(built.ok()) << built.status().message();
  EXPECT_EQ(built->resources().sample_count(), 0U);
  EXPECT_EQ(built->execution_layout().active_logit_rows(), 1U);
  EXPECT_EQ(built->compute().size(), 507U);
  PackedExecutionTrace trace;
  PackedPreludeDriver prelude(trace);
  PackedKernelDriver kernels(trace);
  PackedLinearDriver linears(trace);
  ASSERT_TRUE(built->compute().submit(prelude, kernels, linears, 9).ok());
  EXPECT_EQ(kernels.calls, 311U);
  EXPECT_EQ(linears.calls, 196U);
}

TEST(QwenBf16PackedSynchronousBackendTest,
     CoversPublicationComputeFailureAndDeadlinePoisoning) {
  auto plan = packed_builder_plan();
  PackedBuilderMetadata metadata;
  auto staging_layout = QwenBf16PackedStepStagingLayout::Create(
                            metadata.tokens_view(), metadata.kv_view())
                            .value();
  auto result_layout = QwenBf16PackedResultLayout::Create(1).value();
  alignas(256) std::array<std::byte, 4096> staging_storage{};
  alignas(256) std::array<std::byte, 2048> result_storage{};
  ASSERT_LE(staging_layout.total_bytes(), staging_storage.size());
  ASSERT_LE(result_layout.total_bytes(), result_storage.size());
  auto staging = std::span(staging_storage);
  auto result = std::span(result_storage).first(result_layout.total_bytes());
  auto owners = packed_builder_owners(staging_layout.total_bytes());
  auto commands = packed_commands();
  auto weights = packed_weights();
  auto legacy = legacy_functions();
  auto packed = packed_functions();
  PackedExecutionTrace trace;
  PackedCopyDriver copies;
  PackedPreludeDriver clear(trace);
  PackedKernelDriver kernels(trace);
  PackedLinearDriver linears(trace);
  PackedEventDriver events(result, result_layout);
  PackedHealth health;
  PackedClock clock;
  PackedWaiter waiter;
  const QwenBf16PackedBackendArenas arenas{
      owners,
      packed_endpoint(reinterpret_cast<std::uintptr_t>(staging.data()),
                      staging.size(), 101, 1,
                      CudaCopyMemoryType::kRegisteredPinnedHost),
      packed_endpoint(owners.step_staging.base, owners.step_staging.bytes,
                      102, owners.step_staging.generation,
                      CudaCopyMemoryType::kDevice),
      packed_endpoint(owners.sampled_token.base, owners.sampled_token.bytes,
                      103, owners.sampled_token.generation,
                      CudaCopyMemoryType::kDevice),
      packed_endpoint(owners.device_error.base, owners.device_error.bytes,
                      104, owners.device_error.generation,
                      CudaCopyMemoryType::kDevice),
      packed_endpoint(reinterpret_cast<std::uintptr_t>(result.data()),
                      result.size(), 105, 1,
                      CudaCopyMemoryType::kRegisteredPinnedHost),
      staging, result};
  const QwenBf16PackedBackendIdentity identity{
      1, 41, 23, 100, 100, 17, 19, 31, 0, 2, 1,
      0.000001F, 0.0883883476F};
  const QwenBf16PackedBackendDrivers drivers{
      &copies, &clear, &kernels, &linears, &events, &health, &clock, &waiter};
  auto backend = QwenBf16PackedSynchronousBackend::Create(
      commands, legacy, packed, weights, arenas, identity, drivers);
  ASSERT_TRUE(backend.ok()) << backend.status().message();
  const std::array<QwenKvBlockHandle, 1> first_handles{{0, 1}};
  const std::array<QwenKvBlockHandle, 1> second_handles{{1, 1}};
  auto first_table = QwenKvBlockTable::Create(0, 4, 16, first_handles).value();
  auto second_table = QwenKvBlockTable::Create(1, 9, 16, second_handles).value();
  const std::array<QwenBf16PackedKvBinding, 2> kv_bindings{{
      {11, &first_table}, {12, &second_table}}};
  const std::array append_plans{
      first_table.prepare_append(6).value(),
      second_table.prepare_append(10).value()};
  const std::array<Qwen3SamplingDescriptor, 2> sampling{};
  auto execution = backend->execute_packed(
      plan, metadata.tokens_view(), metadata.kv_view(), kv_bindings,
      append_plans, sampling);
  ASSERT_TRUE(execution.ok()) << execution.status().message();
  ASSERT_EQ(execution->sampled_token_ids.size(), 1U);
  EXPECT_EQ(execution->sampled_token_ids.front(), 42U);
  ASSERT_EQ(execution->sampling_receipts.size(), 1U);
  EXPECT_EQ(execution->sampling_receipts.front().token_id, 42U);
  EXPECT_FLOAT_EQ(execution->sampling_receipts.front().selected_logprob,
                  -0.25F);
  EXPECT_EQ(execution->sampling_receipts.front().rng_word, 0U);
  EXPECT_EQ(execution->sampling_receipts.front().top_logprob_count, 1U);
  EXPECT_EQ(execution->sampling_receipts.front().top_token_ids[0], 42U);
  EXPECT_FLOAT_EQ(execution->sampling_receipts.front().top_logprobs[0],
                  -0.25F);
  EXPECT_EQ(execution->completion_event.handle, 31U);
  EXPECT_EQ(execution->completion_event.generation, 23U);
  EXPECT_EQ(copies.uploads, 14U);
  EXPECT_EQ(copies.downloads, 7U);
  EXPECT_EQ(kernels.calls, 313U);
  EXPECT_EQ(linears.calls, 197U);
  EXPECT_EQ(events.records, 1);
  EXPECT_EQ(waiter.calls, 1);
  EXPECT_EQ(backend->state(), QwenBf16PackedBackendState::kReady);

  auto zero_sample_backend = QwenBf16PackedSynchronousBackend::Create(
      commands, legacy, packed, weights, arenas, identity, drivers);
  ASSERT_TRUE(zero_sample_backend.ok());
  auto intermediate_metadata = metadata.tokens_view();
  intermediate_metadata.sample_row_index = {};
  auto intermediate = zero_sample_backend->execute_packed(
      plan, intermediate_metadata, metadata.kv_view(), kv_bindings,
      append_plans, sampling);
  ASSERT_TRUE(intermediate.ok()) << intermediate.status().message();
  EXPECT_TRUE(intermediate->sampled_token_ids.empty());
  EXPECT_EQ(copies.uploads, 28U);
  EXPECT_EQ(copies.downloads, 8U);
  EXPECT_EQ(kernels.calls, 624U);
  EXPECT_EQ(linears.calls, 393U);

  auto failed_backend = QwenBf16PackedSynchronousBackend::Create(
      commands, legacy, packed, weights, arenas, identity, drivers);
  ASSERT_TRUE(failed_backend.ok());
  kernels.fail_on_call = kernels.calls + 2;
  auto failed = failed_backend->execute_packed(
      plan, metadata.tokens_view(), metadata.kv_view(), kv_bindings,
      append_plans, sampling);
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(failed_backend->state(), QwenBf16PackedBackendState::kPoisoned);

  kernels.fail_on_call = 0;
  events.always_not_ready = true;
  auto timeout_identity = identity;
  timeout_identity.timeout_ns = 2;
  auto timeout_backend = QwenBf16PackedSynchronousBackend::Create(
      commands, legacy, packed, weights, arenas, timeout_identity, drivers);
  ASSERT_TRUE(timeout_backend.ok());
  auto timed_out = timeout_backend->execute_packed(
      plan, metadata.tokens_view(), metadata.kv_view(), kv_bindings,
      append_plans, sampling);
  EXPECT_FALSE(timed_out.ok());
  EXPECT_EQ(timeout_backend->state(), QwenBf16PackedBackendState::kPoisoned);
}

TEST(QwenInt4PackedSynchronousBackendTest,
     ExecutesOneFusedBucketAndPublishesSamplingEvidence) {
  auto plan=packed_builder_plan();PackedBuilderMetadata metadata;
  auto staging_layout=QwenBf16PackedStepStagingLayout::Create(
      metadata.tokens_view(),metadata.kv_view()).value();
  auto result_layout=QwenBf16PackedResultLayout::Create(1).value();
  alignas(256) std::array<std::byte,4096> staging_storage{};
  alignas(256) std::array<std::byte,2048> result_storage{};
  auto staging=std::span(staging_storage);
  auto result=std::span(result_storage).first(result_layout.total_bytes());
  auto owners=packed_builder_owners(staging_layout.total_bytes());
  auto commands=packed_commands();auto weights=packed_int4_weights();
  auto int4=packed_int4_functions();auto packed=packed_functions();
  PackedExecutionTrace trace;PackedCopyDriver copies;
  PackedPreludeDriver clear(trace);PackedKernelDriver kernels(trace);
  PackedInt4LmDriver lm(trace);PackedEventDriver events(result,result_layout);
  PackedHealth health;PackedClock clock;PackedWaiter waiter;
  const QwenInt4PackedBackendArenas arenas{
      owners,
      packed_endpoint(reinterpret_cast<std::uintptr_t>(staging.data()),
          staging.size(),101,1,CudaCopyMemoryType::kRegisteredPinnedHost),
      packed_endpoint(owners.step_staging.base,owners.step_staging.bytes,
          102,owners.step_staging.generation,CudaCopyMemoryType::kDevice),
      packed_endpoint(owners.sampled_token.base,owners.sampled_token.bytes,
          103,owners.sampled_token.generation,CudaCopyMemoryType::kDevice),
      packed_endpoint(owners.device_error.base,owners.device_error.bytes,
          104,owners.device_error.generation,CudaCopyMemoryType::kDevice),
      packed_endpoint(reinterpret_cast<std::uintptr_t>(result.data()),
          result.size(),105,1,CudaCopyMemoryType::kRegisteredPinnedHost),
      staging,result};
  const QwenInt4PackedBackendIdentity identity{
      1,41,23,100,100,17,19,31,0,2,1,0.000001F,0.0883883476F};
  const QwenInt4PackedBackendDrivers drivers{
      &copies,&clear,&kernels,&lm,&events,&health,&clock,&waiter};
  auto backend=QwenInt4PackedSynchronousBackend::Create(
      commands,int4,packed,weights,arenas,identity,drivers);
  ASSERT_TRUE(backend.ok())<<backend.status().message();
  const std::array<QwenKvBlockHandle,1> first_handles{{0,1}};
  const std::array<QwenKvBlockHandle,1> second_handles{{1,1}};
  auto first=QwenKvBlockTable::Create(0,4,16,first_handles).value();
  auto second=QwenKvBlockTable::Create(1,9,16,second_handles).value();
  const std::array<QwenBf16PackedKvBinding,2> bindings{{
      {11,&first},{12,&second}}};
  const std::array appends{first.prepare_append(6).value(),
                           second.prepare_append(10).value()};
  const std::array<Qwen3SamplingDescriptor,2> sampling{};
  auto execution=backend->execute_packed(plan,metadata.tokens_view(),
      metadata.kv_view(),bindings,appends,sampling);
  ASSERT_TRUE(execution.ok())<<execution.status().message();
  ASSERT_EQ(execution->sampling_receipts.size(),1U);
  EXPECT_EQ(execution->sampling_receipts.front().token_id,42U);
  EXPECT_EQ(execution->completion_event.generation,23U);
  EXPECT_EQ(kernels.calls,509U);
  EXPECT_EQ(lm.calls,1U);
  EXPECT_EQ(copies.uploads,14U);
  EXPECT_EQ(copies.downloads,7U);
  EXPECT_EQ(waiter.calls,1);
  EXPECT_EQ(backend->state(),QwenInt4PackedBackendState::kReady);
}

}  // namespace
}  // namespace pih
