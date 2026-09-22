#include "pih/model/qwen3_bf16_kernel_materializer.h"
#include "pih/model/qwen3_bf16_linear_materializer.h"
#include "pih/model/qwen3_bf16_prepared_execution.h"
#include "pih/model/qwen3_bf16_step_builder.h"
#include "pih/model/qwen3_bf16_instrumented_backend.h"
#include "pih/model/qwen3_bf16_synchronous_backend.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Qwen3Config official_config() {
  return Qwen3Config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                     1'000'000.0, 0.000001, 151643, 151645};
}

TensorView view(std::uintptr_t address, DType dtype,
                std::span<const std::int64_t> shape,
                std::uint64_t generation = 1) {
  auto device = Device::Create(DeviceType::kCuda, 0);
  auto result = TensorView::Create(reinterpret_cast<void*>(address), dtype,
                                   shape, {}, *device, generation);
  EXPECT_TRUE(result.ok()) << result.status().message();
  return std::move(result).value();
}

QwenBf16ResourceSet resources() {
  const std::array<std::int64_t, 1> two{2};
  const std::array<std::int64_t, 1> one{1};
  const std::array<std::int64_t, 1> four{4};
  const std::array<std::int64_t, 1> sixteen{16};
  const std::array<std::int64_t, 1> thirty_two{32};
  const std::array<std::int64_t, 1> kv_bytes{3'670'016};
  const std::array<std::int64_t, 2> hidden{2, 1024};
  const std::array<std::int64_t, 2> mlp{2, 3072};
  const std::array<std::int64_t, 2> angles{2, 64};
  const std::array<std::int64_t, 2> logits{1, 151936};
  const std::array<std::int64_t, 3> query{2, 16, 128};
  const std::array<std::int64_t, 3> kv{2, 8, 128};
  std::vector<QwenBf16SlotResource> entries{
      {QwenBf16ActivationSlot::kTokenIds,
       view(0x100000000, DType::kInt64, two)},
      {QwenBf16ActivationSlot::kPositions,
       view(0x200000000, DType::kInt64, two)},
      {QwenBf16ActivationSlot::kHidden,
       view(0x300000000, DType::kBFloat16, hidden)},
      {QwenBf16ActivationSlot::kNormalized,
       view(0x400000000, DType::kBFloat16, hidden)},
      {QwenBf16ActivationSlot::kQuery,
       view(0x500000000, DType::kBFloat16, query)},
      {QwenBf16ActivationSlot::kKey,
       view(0x600000000, DType::kBFloat16, kv)},
      {QwenBf16ActivationSlot::kValue,
       view(0x700000000, DType::kBFloat16, kv)},
      {QwenBf16ActivationSlot::kAttention,
       view(0x800000000, DType::kBFloat16, query)},
      {QwenBf16ActivationSlot::kGate,
       view(0x900000000, DType::kBFloat16, mlp)},
      {QwenBf16ActivationSlot::kUp,
       view(0xA00000000, DType::kBFloat16, mlp)},
      {QwenBf16ActivationSlot::kRopeCosine,
       view(0xB00000000, DType::kFloat32, angles)},
      {QwenBf16ActivationSlot::kRopeSine,
       view(0xC00000000, DType::kFloat32, angles)},
      {QwenBf16ActivationSlot::kKvBacking,
       view(0xD00000000, DType::kUInt8, kv_bytes)},
      {QwenBf16ActivationSlot::kKvSlotStates,
       view(0xE00000000, DType::kUInt8, thirty_two)},
      {QwenBf16ActivationSlot::kKvAppendHandles,
       view(0xF00000000, DType::kUInt8, sixteen)},
      {QwenBf16ActivationSlot::kKvVisibleHandles,
       view(0x1000000000, DType::kUInt8, sixteen)},
      {QwenBf16ActivationSlot::kKvTokenOffsets,
       view(0x1100000000, DType::kUInt8, four)},
      {QwenBf16ActivationSlot::kDeviceError,
       view(0x1200000000, DType::kUInt8, four, 9)},
      {QwenBf16ActivationSlot::kLogits,
       view(0x1300000000, DType::kFloat32, logits)},
      {QwenBf16ActivationSlot::kSampledToken,
       view(0x1400000000, DType::kInt64, one)}};
  auto set = QwenBf16ResourceSet::Create(9, 0, entries);
  EXPECT_TRUE(set.ok()) << set.status().message();
  return std::move(set).value();
}

QwenBf16WeightResourceSet weights() {
  std::vector<TensorView> entries;
  std::uintptr_t address = 0x10000000000;
  for (const auto& expected : Qwen3Manifest::expected_tensors()) {
    std::vector<std::int64_t> shape;
    for (const auto extent : expected.shape) {
      shape.push_back(static_cast<std::int64_t>(extent));
    }
    entries.push_back(view(address, DType::kBFloat16, shape));
    address += 0x100000000;
  }
  auto set = QwenBf16WeightResourceSet::Create(0, entries);
  EXPECT_TRUE(set.ok()) << set.status().message();
  return std::move(set).value();
}

ResolvedKernelFunction function(QwenBf16Primitive primitive) {
  const std::string digest(64, 'a');
  auto manifest = qwen_bf16_kernel_manifest(primitive, digest);
  auto symbol = qwen_bf16_kernel_symbol(primitive);
  EXPECT_TRUE(manifest.ok());
  EXPECT_TRUE(symbol.ok());
  return {std::string(*symbol), std::string(manifest->logical_id()), digest,
          std::string(manifest->parameter_abi_sha256()), 0x1234};
}

std::array<ResolvedKernelFunction, QwenBf16KernelBundle::kExecutionPrimitiveCount>
functions() {
  std::array<ResolvedKernelFunction, QwenBf16KernelBundle::kExecutionPrimitiveCount>
      result;
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = function(static_cast<QwenBf16Primitive>(index));
  }
  return result;
}

TEST(QwenBf16KernelMaterializerTest, MaterializesAllKernelCommands) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();
  const QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                      0.0883883476F};

  std::size_t materialized = 0;
  for (const auto& command : *commands) {
    if (command.backend == QwenBf16CommandBackend::kLinear) continue;
    auto plan = QwenBf16KernelMaterializer::Create(
        command, function(command.primitive), request_resources,
        weight_resources, context);
    ASSERT_TRUE(plan.ok()) << static_cast<int>(command.execution_step.operation)
                           << ": " << plan.status().message();
    EXPECT_EQ(plan->primitive(), command.primitive);
    ++materialized;
  }
  EXPECT_EQ(materialized, QwenBf16CommandBuffer::kKernelCommandCount);
}

TEST(QwenBf16KernelMaterializerTest, RejectsLinearGenerationAndSubcommandDrift) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();
  QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                0.0883883476F};
  EXPECT_FALSE(QwenBf16KernelMaterializer::Create(
                   (*commands)[3], function(QwenBf16Primitive::kEmbedding),
                   request_resources, weight_resources, context)
                   .ok());
  context.request_generation = 10;
  EXPECT_FALSE(QwenBf16KernelMaterializer::Create(
                   (*commands)[0], function(QwenBf16Primitive::kEmbedding),
                   request_resources, weight_resources, context)
                   .ok());
  auto rope = (*commands)[8];
  rope.subcommand = 2;
  context.request_generation = 9;
  EXPECT_FALSE(QwenBf16KernelMaterializer::Create(
                   rope, function(QwenBf16Primitive::kRope),
                   request_resources, weight_resources, context)
                   .ok());
}

TEST(QwenBf16LinearMaterializerTest, MaterializesAllLinearCommands) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();

  std::size_t materialized = 0;
  for (const auto& command : *commands) {
    if (command.backend == QwenBf16CommandBackend::kKernel) continue;
    auto binding = QwenBf16LinearMaterializer::Create(
        command, request_resources, weight_resources, 9);
    ASSERT_TRUE(binding.ok())
        << static_cast<int>(command.execution_step.operation) << ": "
        << binding.status().message();
    EXPECT_EQ(binding->kind(), command.linear_kind);
    ++materialized;
  }
  EXPECT_EQ(materialized, QwenBf16CommandBuffer::kLinearCommandCount);
}

TEST(QwenBf16LinearMaterializerTest, LmHeadBindsOnlyLastTokenRow) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();
  const auto& command = (*commands)[507];
  auto binding = QwenBf16LinearMaterializer::Create(
      command, request_resources, weight_resources, 9);
  ASSERT_TRUE(binding.ok()) << binding.status().message();
  auto normalized = request_resources.view(QwenBf16ActivationSlot::kNormalized);
  ASSERT_TRUE(normalized.ok());
  EXPECT_EQ(binding->input().dim(0), 1);
  EXPECT_EQ(binding->input().dim(1), 1024);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(binding->input().data()),
            reinterpret_cast<std::uintptr_t>(normalized->data()) + 2048);
  EXPECT_EQ(binding->output().dtype(), DType::kFloat32);
  EXPECT_EQ(binding->output().dim(0), 1);
  EXPECT_EQ(binding->output().dim(1), 151936);
}

TEST(QwenBf16LinearMaterializerTest, RejectsKernelAndGenerationDrift) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();
  EXPECT_FALSE(QwenBf16LinearMaterializer::Create(
                   (*commands)[0], request_resources, weight_resources, 9)
                   .ok());
  EXPECT_FALSE(QwenBf16LinearMaterializer::Create(
                   (*commands)[3], request_resources, weight_resources, 10)
                   .ok());
}

struct ExecutionTrace final {
  std::vector<char> events;
};

class PreludeDriver final : public QwenBf16DeviceErrorClearDriver {
 public:
  explicit PreludeDriver(ExecutionTrace& trace) : trace_(&trace) {}
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override {
    trace_->events.push_back('C');
    return Status::Ok();
  }
 private:
  ExecutionTrace* trace_;
};

class KernelDriver final : public KernelLaunchDriver {
 public:
  explicit KernelDriver(ExecutionTrace& trace) : trace_(&trace) {}
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override {
    trace_->events.push_back('K');
    return Status::Ok();
  }
 private:
  ExecutionTrace* trace_;
};

class LinearDriver final : public QwenBf16LinearExecutionDriver {
 public:
  explicit LinearDriver(ExecutionTrace& trace) : trace_(&trace) {}
  Status execute(const QwenBf16LinearBinding&,
                 DriverStreamHandle) override {
    trace_->events.push_back('L');
    ++calls;
    if (fail_on_call != 0 && calls == fail_on_call) {
      return Status::Internal("injected linear failure");
    }
    return Status::Ok();
  }
  std::size_t calls = 0;
  std::size_t fail_on_call = 0;
 private:
  ExecutionTrace* trace_;
};

class SnapshotDriver final : public QwenBf16TapSnapshotDriver {
 public:
  explicit SnapshotDriver(ExecutionTrace& trace) : trace_(&trace) {}
  Status snapshot(const QwenBf16TapBinding& binding,
                  DriverStreamHandle stream) override {
    trace_->events.push_back('S');
    command_indices.push_back(binding.producer_command_index);
    streams.push_back(stream);
    ++calls;
    if (fail_on_call != 0 && calls == fail_on_call) {
      return Status::Internal("injected snapshot failure");
    }
    return Status::Ok();
  }
  std::vector<std::size_t> command_indices;
  std::vector<DriverStreamHandle> streams;
  std::size_t calls = 0;
  std::size_t fail_on_call = 0;
 private:
  ExecutionTrace* trace_;
};

class SnapshotRecorder final : public QwenBf16TapSnapshotFrontierRecorder {
 public:
  explicit SnapshotRecorder(ExecutionTrace& trace) : trace_(&trace) {}
  Status record_snapshot(CompletionEventDriver&, std::uint64_t submit_ns,
                         std::uint64_t deadline_ns) override {
    trace_->events.push_back('E');
    observed_submit_ns = submit_ns;
    observed_deadline_ns = deadline_ns;
    return result;
  }
  std::uint64_t observed_submit_ns = 0;
  std::uint64_t observed_deadline_ns = 0;
  Status result = Status::Ok();
 private:
  ExecutionTrace* trace_;
};

class BackendCopyDriver final : public TypedCopyDriver {
 public:
  explicit BackendCopyDriver(ExecutionTrace& trace) : trace_(&trace) {}
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    trace_->events.push_back(kind == CudaCopyKind::kHostToDevice ? 'U' : 'D');
    return Status::Ok();
  }
 private:
  ExecutionTrace* trace_;
};

class BackendEventDriver final : public CompletionEventDriver {
 public:
  BackendEventDriver(ExecutionTrace& trace, std::span<std::byte> result)
      : trace_(&trace), result_(result) {}
  Status record(DriverEventHandle, DriverStreamHandle) override {
    trace_->events.push_back('E');
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    if (queries++ == 0) return CudaEventQueryResult::kNotReady;
    const std::int64_t token = 42;
    const std::uint32_t error = 0;
    std::memcpy(result_.data() +
                    QwenBf16StepResultLayout::sampled_token().offset_bytes,
                &token, sizeof(token));
    std::memcpy(result_.data() +
                    QwenBf16StepResultLayout::device_error().offset_bytes,
                &error, sizeof(error));
    return CudaEventQueryResult::kSuccess;
  }
  int queries = 0;
 private:
  ExecutionTrace* trace_;
  std::span<std::byte> result_;
};

class BackendHealth final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    return QwenBf16StepHealth{true, false};
  }
};

class BackendClock final : public QwenBf16MonotonicClock {
 public:
  Result<std::uint64_t> now_ns() override {
    const auto result = now;
    now += increment;
    return result;
  }
  std::uint64_t now = 100;
  std::uint64_t increment = 1;
};

class BackendWaiter final : public QwenBf16PollWaiter {
 public:
  Status wait() override {
    ++calls;
    return Status::Ok();
  }
  int calls = 0;
};

class BackendDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t, std::uint64_t) override {
    ++calls;
    return Status::Internal("unexpected allocation");
  }
  void deallocate(Allocation) noexcept override {}
  int calls = 0;
};

class BackendPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t, std::uint64_t) override {
    ++calls;
    return Status::Internal("unexpected allocation");
  }
  void deallocate(Allocation) noexcept override {}
  int calls = 0;
};

class BackendPlacement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void*, std::uint64_t, std::int32_t) override {
    ++calls;
    return Status::Ok();
  }
  int calls = 0;
};

CudaCopyEndpoint backend_endpoint(std::uintptr_t base, std::uint64_t bytes,
                                  std::uint64_t owner,
                                  std::uint64_t generation,
                                  CudaCopyMemoryType type) {
  return {base, bytes, 0, owner, generation, type, 0, 0};
}

TEST(QwenBf16PreparedExecutionTest, RunsPreludeThenAllCommandsInFrozenOrder) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                      0.0883883476F};
  auto execution = QwenBf16PreparedExecution::Create(
      *commands, kernel_functions, request_resources, weight_resources,
      context);
  ASSERT_TRUE(execution.ok()) << execution.status().message();
  EXPECT_EQ(execution->size(), 509);
  EXPECT_EQ(execution->state(), QwenBf16PreparedExecutionState::kPrepared);

  ExecutionTrace trace;
  PreludeDriver prelude(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  ASSERT_TRUE(execution->run(prelude, kernels, linears, 0x55).ok());
  ASSERT_EQ(trace.events.size(), 510);
  EXPECT_EQ(trace.events.front(), 'C');
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'K'), 312);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'L'), 197);
  EXPECT_EQ(execution->next_command(), 509);
  EXPECT_EQ(execution->state(), QwenBf16PreparedExecutionState::kCompleted);
}

TEST(QwenBf16PreparedExecutionTest, FailurePoisonsWithoutReplay) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto request_resources = resources();
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                      0.0883883476F};
  auto execution = QwenBf16PreparedExecution::Create(
      *commands, kernel_functions, request_resources, weight_resources,
      context);
  ASSERT_TRUE(execution.ok());
  ExecutionTrace trace;
  PreludeDriver prelude(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  linears.fail_on_call = 2;
  EXPECT_FALSE(execution->run(prelude, kernels, linears, 1).ok());
  const auto stopped = execution->next_command();
  EXPECT_EQ(execution->state(), QwenBf16PreparedExecutionState::kPoisoned);
  EXPECT_FALSE(execution->run(prelude, kernels, linears, 1).ok());
  EXPECT_EQ(execution->next_command(), stopped);
}

TEST(QwenBf16PreparedExecutionTest,
     InstrumentedRunSnapshotsImmediatelyAfterProducerOnSameStream) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto weight_bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weight_bindings);
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kQueryAfterRope, 0, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 0, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20);
  auto tap_bindings = QwenBf16TapBindingPlan::Create(*taps, *commands);
  auto request_resources = resources();
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                      0.0883883476F};
  auto execution = QwenBf16PreparedExecution::Create(
      *commands, kernel_functions, request_resources, weight_resources,
      context);
  ASSERT_TRUE(execution.ok() && tap_bindings.ok());

  ExecutionTrace trace;
  PreludeDriver prelude(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  SnapshotDriver snapshots(trace);
  constexpr DriverStreamHandle kStream = 0x77;
  ASSERT_TRUE(execution->run_instrumented(prelude, kernels, linears, snapshots,
                                         *tap_bindings, kStream).ok());
  ASSERT_EQ(snapshots.calls, 3);
  EXPECT_EQ(snapshots.streams,
            std::vector<DriverStreamHandle>({kStream, kStream, kStream}));
  EXPECT_TRUE(std::is_sorted(snapshots.command_indices.begin(),
                             snapshots.command_indices.end()));
  std::size_t prior_snapshots = 0;
  for (const auto command_index : snapshots.command_indices) {
    const auto event_index = 1 + command_index + prior_snapshots;
    ASSERT_LT(event_index + 1, trace.events.size());
    EXPECT_TRUE(trace.events[event_index] == 'K' ||
                trace.events[event_index] == 'L');
    EXPECT_EQ(trace.events[event_index + 1], 'S');
    ++prior_snapshots;
  }
}

TEST(QwenBf16PreparedExecutionTest, SnapshotFailurePoisonsAtProducerCommand) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto weight_bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weight_bindings);
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kQueryAfterRope, 0, 0, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20);
  auto tap_bindings = QwenBf16TapBindingPlan::Create(*taps, *commands);
  auto request_resources = resources();
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                      0.0883883476F};
  auto execution = QwenBf16PreparedExecution::Create(
      *commands, kernel_functions, request_resources, weight_resources,
      context);
  ASSERT_TRUE(execution.ok() && tap_bindings.ok());
  ExecutionTrace trace;
  PreludeDriver prelude(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  SnapshotDriver snapshots(trace);
  snapshots.fail_on_call = 1;
  EXPECT_FALSE(execution->run_instrumented(prelude, kernels, linears,
                                           snapshots, *tap_bindings, 0x77)
                   .ok());
  EXPECT_EQ(execution->state(), QwenBf16PreparedExecutionState::kPoisoned);
  EXPECT_EQ(execution->next_command(),
            (*tap_bindings)[0].producer_command_index);
  EXPECT_FALSE(execution->run(prelude, kernels, linears, 0x77).ok());
}

TEST(QwenBf16PreparedExecutionTest,
     InstrumentedSubmissionRecordsFrontierAfterAllCommands) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto weight_bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weight_bindings);
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 0, 1}};
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20);
  auto tap_bindings = QwenBf16TapBindingPlan::Create(*taps, *commands);
  auto request_resources = resources();
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16KernelContext context{9, 23, 15, 17, 2, 0.000001F,
                                      0.0883883476F};
  auto execution = QwenBf16PreparedExecution::Create(
      *commands, kernel_functions, request_resources, weight_resources,
      context);
  ASSERT_TRUE(execution.ok() && tap_bindings.ok());
  ExecutionTrace trace;
  PreludeDriver prelude(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  SnapshotDriver snapshots(trace);
  SnapshotRecorder recorder(trace);
  BackendEventDriver events(trace, {});

  QwenBf16PreparedStepCompute compute(std::move(*execution));
  ASSERT_TRUE(compute.submit_instrumented_and_record(
                  prelude, kernels, linears, snapshots, *tap_bindings,
                  recorder, events, 0x77, 100, 200)
                  .ok());
  ASSERT_EQ(trace.events.size(), 512);
  EXPECT_EQ(trace.events[509], 'S');
  EXPECT_EQ(trace.events[510], 'K');
  EXPECT_EQ(trace.events[511], 'E');
  EXPECT_EQ(recorder.observed_submit_ns, 100);
  EXPECT_EQ(recorder.observed_deadline_ns, 200);
}

TEST(QwenBf16StepBuilderTest, BuildsAndRunsRealPreparedStepFromArenaOwners) {
  const QwenKvBlockHandle handles[] = {{4, 9}, {7, 3}};
  auto table = QwenKvBlockTable::Create(23, 6, 32, handles).value();
  auto append = table.prepare_append(2).value();
  const std::int64_t token_ids[] = {4, 5};
  auto input = QwenBf16StepInputPlan::Create(token_ids, 0, table, append);
  auto staging = QwenBf16StepStagingLayout::Create(*input);
  auto arena = QwenBf16ExecutionArenaLayout::Create(2, 1);
  ASSERT_TRUE(staging.ok() && arena.ok());
  const QwenBf16StepDeviceOwners owners{
      {0x2100000000, staging->total_bytes(), 1},
      {0x2200000000, arena->activation_arena_bytes(), 2},
      {0x2300000000, arena->mlp().arena_bytes(), 3},
      {0x2400000000, arena->rope_workspace_bytes(), 4},
      {0x2500000000, arena->logit_workspace_bytes(), 5},
      {0x2600000000, sizeof(std::int64_t), 6},
      {0x2700000000, sizeof(std::uint32_t), 9},
      {0x2800000000, 2 * QwenKvSlotPool::kSlotPayloadBytes, 8},
      {0x2900000000, 2 * sizeof(QwenKvSlotState), 9}};
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto weight_resources = weights();
  auto kernel_functions = functions();
  std::vector<std::byte> pinned(staging->total_bytes() + 256);
  auto built = QwenBf16StepBuilder::Create(
      token_ids, 0, table, append, 9, 0, 2, 0.000001F, 0.0883883476F,
      *commands, kernel_functions, weight_resources, owners, pinned);
  ASSERT_TRUE(built.ok()) << built.status().message();
  EXPECT_EQ(built->staging_layout().token_count(), 2);
  EXPECT_EQ(built->execution_layout().tokens(), 2);
  EXPECT_EQ(built->resources().request_generation(), 9);
  EXPECT_EQ(built->resources().owning_rank(), 0);
  auto kv_resource = built->resources().view(
      QwenBf16ActivationSlot::kKvBacking);
  ASSERT_TRUE(kv_resource.ok());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(kv_resource->data()),
            owners.kv_backing.base);

  ExecutionTrace trace;
  PreludeDriver prelude(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  ASSERT_TRUE(built->compute().submit(prelude, kernels, linears, 0x55).ok());
  EXPECT_EQ(trace.events.size(), 510);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'K'), 312);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'L'), 197);
}

TEST(QwenBf16SynchronousBackendTest, RunsCompleteStepThroughPublication) {
  const QwenKvBlockHandle handles[] = {{4, 9}, {7, 3}};
  auto table = QwenKvBlockTable::Create(23, 6, 32, handles).value();
  auto append = table.prepare_append(2).value();
  const std::int64_t token_ids[] = {4, 5};
  auto input = QwenBf16StepInputPlan::Create(token_ids, 0, table, append);
  auto staging = QwenBf16StepStagingLayout::Create(*input).value();
  auto arena = QwenBf16ExecutionArenaLayout::Create(2, 1).value();
  const QwenBf16StepDeviceOwners owners{
      {0x2100000000, staging.total_bytes(), 1},
      {0x2200000000, arena.activation_arena_bytes(), 2},
      {0x2300000000, arena.mlp().arena_bytes(), 3},
      {0x2400000000, arena.rope_workspace_bytes(), 4},
      {0x2500000000, arena.logit_workspace_bytes(), 5},
      {0x2600000000, sizeof(std::int64_t), 6},
      {0x2700000000, sizeof(std::uint32_t), 9},
      {0x2800000000, 2 * QwenKvSlotPool::kSlotPayloadBytes, 8},
      {0x2900000000, 2 * sizeof(QwenKvSlotState), 9}};
  alignas(256) std::array<std::byte, 2048> pinned_staging{};
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      pinned_result{};
  ExecutionTrace trace;
  BackendCopyDriver copies(trace);
  PreludeDriver clear(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  BackendEventDriver events(trace, pinned_result);
  BackendHealth health;
  BackendClock clock;
  BackendWaiter waiter;
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto weight_resources = weights();
  auto kernel_functions = functions();
  QwenBf16SynchronousBackendArenas arenas{
      owners,
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_staging.data()),
                       pinned_staging.size(), 101, 1,
                       CudaCopyMemoryType::kRegisteredPinnedHost),
      backend_endpoint(owners.step_staging.base, owners.step_staging.bytes,
                       102, 1, CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.sampled_token.base, owners.sampled_token.bytes,
                       103, 6, CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.device_error.base, owners.device_error.bytes,
                       104, 9, CudaCopyMemoryType::kDevice),
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_result.data()),
                       pinned_result.size(), 105, 1,
                       CudaCopyMemoryType::kRegisteredPinnedHost),
      pinned_staging,
      pinned_result};
  const QwenBf16SynchronousBackendIdentity identity{
      1, 9, 23, 100, 100, 17, 19, 31, 0, 2,
      0.000001F, 0.0883883476F};
  const QwenBf16SynchronousBackendDrivers drivers{
      &copies, &clear, &kernels, &linears, &events, &health, &clock, &waiter};
  auto backend = QwenBf16SynchronousBackend::Create(
      *commands, kernel_functions, weight_resources, arenas, identity, drivers);
  ASSERT_TRUE(backend.ok()) << backend.status().message();
  auto token = backend->execute_and_read_token(token_ids, 0, table, append);
  ASSERT_TRUE(token.ok()) << token.status().message();
  EXPECT_EQ(*token, 42);
  auto completion = backend->last_completion_event();
  ASSERT_TRUE(completion.ok());
  EXPECT_EQ(completion->handle, 31U);
  EXPECT_EQ(completion->generation, 23U);
  EXPECT_EQ(backend->next_request_generation(), 10U);
  EXPECT_EQ(backend->next_event_generation(), 24U);
  EXPECT_EQ(backend->next_plan_id(), 107U);
  EXPECT_EQ(waiter.calls, 1);
  EXPECT_EQ(trace.events.size(), 518);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'U'), 5);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'K'), 312);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'L'), 197);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'D'), 2);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'E'), 1);
  EXPECT_EQ(backend->state(), QwenBf16SynchronousBackendState::kReady);
}

TEST(QwenBf16SynchronousBackendTest, DeadlinePreventsLateEventPublication) {
  const QwenKvBlockHandle handles[] = {{4, 9}};
  auto table = QwenKvBlockTable::Create(23, 6, 16, handles).value();
  auto append = table.prepare_append(1).value();
  const std::int64_t token_ids[] = {4};
  auto input = QwenBf16StepInputPlan::Create(token_ids, 0, table, append);
  auto staging = QwenBf16StepStagingLayout::Create(*input).value();
  auto arena = QwenBf16ExecutionArenaLayout::Create(1, 1).value();
  const QwenBf16StepDeviceOwners owners{
      {0x3100000000, staging.total_bytes(), 1},
      {0x3200000000, arena.activation_arena_bytes(), 2},
      {0x3300000000, arena.mlp().arena_bytes(), 3},
      {0x3400000000, arena.rope_workspace_bytes(), 4},
      {0x3500000000, arena.logit_workspace_bytes(), 5},
      {0x3600000000, sizeof(std::int64_t), 6},
      {0x3700000000, sizeof(std::uint32_t), 9},
      {0x3800000000, QwenKvSlotPool::kSlotPayloadBytes, 8},
      {0x3900000000, sizeof(QwenKvSlotState), 9}};
  alignas(256) std::array<std::byte, 2048> pinned_staging{};
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      pinned_result{};
  ExecutionTrace trace;
  BackendCopyDriver copies(trace);
  PreludeDriver clear(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  BackendEventDriver events(trace, pinned_result);
  BackendHealth health;
  BackendClock clock;
  clock.increment = 2;
  BackendWaiter waiter;
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *bindings);
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16SynchronousBackendArenas arenas{
      owners,
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_staging.data()),
                       pinned_staging.size(), 201, 1,
                       CudaCopyMemoryType::kRegisteredPinnedHost),
      backend_endpoint(owners.step_staging.base, owners.step_staging.bytes,
                       202, 1, CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.sampled_token.base, owners.sampled_token.bytes,
                       203, 6, CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.device_error.base, owners.device_error.bytes,
                       204, 9, CudaCopyMemoryType::kDevice),
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_result.data()),
                       pinned_result.size(), 205, 1,
                       CudaCopyMemoryType::kRegisteredPinnedHost),
      pinned_staging,
      pinned_result};
  const QwenBf16SynchronousBackendIdentity identity{
      1, 9, 23, 100, 3, 17, 19, 31, 0, 1,
      0.000001F, 0.0883883476F};
  const QwenBf16SynchronousBackendDrivers drivers{
      &copies, &clear, &kernels, &linears, &events, &health, &clock, &waiter};
  auto backend = QwenBf16SynchronousBackend::Create(
      *commands, kernel_functions, weight_resources, arenas, identity, drivers)
                     .value();
  auto token = backend.execute_and_read_token(token_ids, 0, table, append);
  EXPECT_FALSE(token.ok());
  EXPECT_EQ(events.queries, 1);
  EXPECT_EQ(backend.state(), QwenBf16SynchronousBackendState::kPoisoned);
}

TEST(QwenBf16InstrumentedBackendTest,
     GenerationDriftPoisonsBeforeAnyResourceAction) {
  ExecutionTrace trace;
  BackendCopyDriver copies(trace);
  PreludeDriver clear(trace);
  KernelDriver kernels(trace);
  LinearDriver linears(trace);
  alignas(256) std::array<std::byte, 2048> pinned_staging{};
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      pinned_result{};
  BackendEventDriver events(trace, pinned_result);
  BackendHealth health;
  BackendClock clock;
  BackendWaiter waiter;
  BackendDeviceAllocator device_allocator;
  BackendPinnedAllocator pinned_allocator;
  BackendPlacement placement;
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config()).value();
  auto weight_bindings = QwenBf16WeightBindingPlan::Create(schedule).value();
  auto commands = QwenBf16CommandBuffer::Create(
      schedule, weight_bindings).value();
  auto weight_resources = weights();
  auto kernel_functions = functions();
  const QwenBf16StepDeviceOwners owners{
      {0x2100000000, pinned_staging.size(), 1},
      {0x2200000000, 64ULL << 20, 2},
      {0x2300000000, 64ULL << 20, 3},
      {0x2400000000, 1ULL << 20, 4},
      {0x2500000000, 1ULL << 20, 5},
      {0x2600000000, sizeof(std::int64_t), 6},
      {0x2700000000, sizeof(std::uint32_t), 7},
      {0x2800000000, QwenKvSlotPool::kSlotPayloadBytes, 8},
      {0x2900000000, sizeof(QwenKvSlotState), 9}};
  QwenBf16InstrumentedBackendArenas arenas{
      owners,
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_staging.data()),
                       pinned_staging.size(), 11, 1,
                       CudaCopyMemoryType::kRegisteredPinnedHost),
      backend_endpoint(owners.step_staging.base, owners.step_staging.bytes,
                       12, 1, CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.sampled_token.base, owners.sampled_token.bytes,
                       13, 6, CudaCopyMemoryType::kDevice),
      backend_endpoint(owners.device_error.base, owners.device_error.bytes,
                       14, 7, CudaCopyMemoryType::kDevice),
      backend_endpoint(reinterpret_cast<std::uintptr_t>(pinned_result.data()),
                       pinned_result.size(), 15, 1,
                       CudaCopyMemoryType::kRegisteredPinnedHost),
      pinned_staging, pinned_result};
  const QwenBf16InstrumentedBackendIdentity identity{
      1, 100, 200, 300, 1000, 17, 41, 42, 43, 0, 0, 1,
      0.000001F, 0.0883883476F, 500, 600, 700};
  auto backend = QwenBf16InstrumentedBackend::Create(
      commands, kernel_functions, weight_resources, arenas, identity,
      {&device_allocator, &pinned_allocator, &placement, &copies, &clear,
       &kernels, &linears, &events, &health, &clock, &waiter}).value();
  const std::array<QwenKvBlockHandle, 1> handles{{{0, 1}}};
  auto table = QwenKvBlockTable::Create(5, 1, 1, handles).value();
  auto append = table.prepare_append(1).value();
  const std::array<std::int64_t, 1> token{{7}};
  const QwenBf16TapFixtureInput input{
      QwenBf16TapFixtureKind::kPositionZeroActivations, 0, 0, token};
  const std::array request{QwenNumericalTapRequest{
      QwenNumericalTapPoint::kLayerHidden, 0, 0, 1}};
  auto taps = QwenNumericalTapPlan::Create(request, 1ULL << 20).value();
  const std::array<QwenKvSlotState, 1> projected{{
      {1, 5, 1, QwenKvSlotLifecycle::kOwned, 0, 0}}};

  EXPECT_FALSE(backend.execute_and_capture(
      input, taps, table, append, projected, 101).ok());
  EXPECT_EQ(backend.state(), QwenBf16InstrumentedBackendState::kPoisoned);
  EXPECT_TRUE(trace.events.empty());
  EXPECT_EQ(device_allocator.calls, 0);
  EXPECT_EQ(pinned_allocator.calls, 0);
  EXPECT_EQ(placement.calls, 0);
  EXPECT_EQ(backend.tap_device_peak_bytes(), 0);
  EXPECT_EQ(backend.tap_pinned_peak_bytes(), 0);
}

}  // namespace
}  // namespace pih
