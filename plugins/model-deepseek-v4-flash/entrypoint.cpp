#include "pih/contracts/engine_v1.h"
#include "pih/contracts/token_generation_v1.h"
#include "generation.h"
#include "text_completion.h"
#include "pih/core/bounded_json.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/execution_default_v1.h"
#include "pih/contracts/memory_host_spill_v1.h"
#include "pih/contracts/nvidia_cuda_v1.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/contracts/verified_artifact_v1.h"
#include "pih/backend/cuda/nvidia_deepseek_engine_bootstrap.h"
#include "pih/backend/cuda/nvidia_deepseek_rank_runtime.h"
#include "pih/model/deepseek_engine.h"
#include "pih/model/deepseek_expert_pager.h"
#include "pih/plugin_sdk/abi.h"
#include "pih/plugin_sdk/kernel_pack.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr char kModelId[] = "deepseek-v4-flash";
static_assert(sizeof(kModelId) <= PIH_ENGINE_MODEL_ID_CAPACITY_V1);
constexpr std::size_t kMaximumConfigurationBytes = 16U * 1024U;
constexpr std::size_t kMaximumDeploymentRootBytes = 4096U;

bool ExactCString(const char* value, std::string_view expected) noexcept {
  if (value == nullptr) return false;
  const auto* terminator = static_cast<const char*>(
      std::memchr(value, '\0', expected.size() + 1));
  return terminator == value + expected.size() &&
         std::memcmp(value, expected.data(), expected.size()) == 0;
}

bool ExactEngineModelId(
    const char (&value)[PIH_ENGINE_MODEL_ID_CAPACITY_V1]) noexcept {
  return pih_engine_model_id_is_valid_v1(value) &&
         ExactCString(value, kModelId);
}

bool BoundedCString(const char* value, std::size_t maximum) noexcept {
  return value != nullptr && std::memchr(value, '\0', maximum + 1) != nullptr;
}

struct State final {
  std::mutex text_mutex;
  std::unique_ptr<pih::plugin_text::DeepSeekSemanticArtifacts> text_semantic;
  pih_engine_handle_v1 text_handle{sizeof(pih_engine_handle_v1), PIH_ENGINE_ABI_VERSION_V1};
  bool text_failed{};
  uint32_t text_generation_timeout_ms{};
  std::mutex engine_mutex;
  std::atomic<uint32_t> phase{};
  const pih_host_api_v1* host{};
  const pih_verified_artifact_api_v1* artifact{};
  const pih_nvidia_cuda_api_v1* cuda{};
  const pih_nvidia_cuda_memory_api_v1* cuda_memory{};
  const pih_nvidia_cuda_resources_api_v1* cuda_resources{};
  const pih_nvidia_cuda_async_api_v1* cuda_async{};
  const pih_execution_default_api_v1* execution{};
  const pih_memory_host_spill_api_v1* host_spill{};
  const pih_deepseek_kernels_api_v1* kernels{};
  uint64_t next_request_id{1};
  uint64_t next_request_generation{1};
  uint64_t next_plan_sequence{1};
  uint64_t next_engine_generation{1};
  uint32_t context_capacity{};
  uint32_t prefill_chunk{};
  bool engine_generation_issued{};
  pih::DeepSeekEngine* engine_instance{};
  void* engine_handle_identity{};
  uint64_t engine_activation_epoch{};
  uint64_t engine_generation{};
  bool engine_host_spill_enabled{};
  bool engine_snapshot_pending{};
  bool engine_cleanup_failed{};
};

void ClearResolvedCapabilities(State& state) noexcept {
  state.artifact = nullptr;
  state.cuda = nullptr;
  state.cuda_memory = nullptr;
  state.cuda_resources = nullptr;
  state.cuda_async = nullptr;
  state.execution = nullptr;
  state.host_spill = nullptr;
  state.kernels = nullptr;
}

class CapabilityConfigurationTransaction final {
 public:
  explicit CapabilityConfigurationTransaction(State& state) noexcept
      : state_(state) {}
  CapabilityConfigurationTransaction(
      const CapabilityConfigurationTransaction&) = delete;
  CapabilityConfigurationTransaction& operator=(
      const CapabilityConfigurationTransaction&) = delete;
  ~CapabilityConfigurationTransaction() {
    if (!committed_) ClearResolvedCapabilities(state_);
  }
  void Commit() noexcept { committed_ = true; }

 private:
  State& state_;
  bool committed_ = false;
};


class BoundRankRuntimeFactory final
    : public pih::NvidiaDeepSeekRankRuntimeFactory {
 public:
  explicit BoundRankRuntimeFactory(
      const pih_deepseek_kernels_api_v1& kernels,
      const pih_nvidia_cuda_memory_api_v1& memory,
      const pih_nvidia_cuda_resources_api_v1& resources,
      const pih_nvidia_cuda_async_api_v1& async,
      const pih_memory_host_spill_api_v1* host_spill) noexcept
      : kernels_(kernels), memory_(memory), resources_(resources),
        async_(async), host_spill_(host_spill) {}

  pih::Result<std::unique_ptr<pih::NvidiaDeepSeekRankRuntimeView>> Create(
      std::int32_t device_ordinal, std::uint32_t rank,
      std::uint64_t worker_generation,
      const pih::DeepSeekOptimizationPolicy& optimization_policy) override {
    if (kernels_.identity == nullptr ||
        kernels_.identity->abi_version != PIH_KERNEL_PACK_ABI_VERSION_V1 ||
        (!ExactCString(kernels_.identity->architecture, "sm89") &&
         !ExactCString(kernels_.identity->architecture, "sm90"))) {
      return pih::Status::FailedPrecondition(
          "DeepSeek runtime factory is not bound to an admitted Kernel Pack");
    }
    auto runtime = pih::NvidiaDeepSeekRankRuntime::Create(
        device_ordinal, rank, worker_generation, optimization_policy,
        memory_, resources_, async_, kernels_, host_spill_);
    if (!runtime.ok()) return runtime.status();
    return std::unique_ptr<pih::NvidiaDeepSeekRankRuntimeView>(
        std::move(*runtime));
  }

 private:
  const pih_deepseek_kernels_api_v1& kernels_;
  const pih_nvidia_cuda_memory_api_v1& memory_;
  const pih_nvidia_cuda_resources_api_v1& resources_;
  const pih_nvidia_cuda_async_api_v1& async_;
  const pih_memory_host_spill_api_v1* host_spill_;
};

struct DeepSeekPp1Request final {
  uint32_t struct_size{};
  uint64_t activation_epoch{};
  int32_t device_ordinal{};
  uint32_t host_spill_enabled{};
  uint32_t expert_slot_count{};
  uint32_t staging_extent_count{};
  uint32_t artifact_poll_interval_ms{};
  uint32_t attention_reserved_tokens_per_sequence{};
  uint32_t maximum_prefill_chunk_tokens{};
  uint32_t maximum_decode_sequences{};
  uint32_t maximum_verify_sequences{};
  uint64_t maximum_shard_bytes{};
  const char* target_generation_root{};
  const char* artifact_root_sha256_hex{};
};

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

pih_status_v1 CheckedCapabilityStatus(pih_status_v1 status,
                                      const char* invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  auto& state = *static_cast<State*>(context);
  if (!state.phase.compare_exchange_strong(expected, expected + 1)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 EngineStatus(const pih::Status& status) {
  if (status.ok()) return Status(PIH_STATUS_OK_V1);
  uint32_t code = PIH_STATUS_INTERNAL_V1;
  switch (status.code()) {
    case pih::StatusCode::kOk:
      code = PIH_STATUS_OK_V1;
      break;
    case pih::StatusCode::kInvalidArgument:
      code = PIH_STATUS_INVALID_ARGUMENT_V1;
      break;
    case pih::StatusCode::kResourceExhausted:
      code = PIH_STATUS_RESOURCE_EXHAUSTED_V1;
      break;
    case pih::StatusCode::kFailedPrecondition:
      code = PIH_STATUS_FAILED_PRECONDITION_V1;
      break;
    case pih::StatusCode::kInternal:
      code = PIH_STATUS_INTERNAL_V1;
      break;
    case pih::StatusCode::kUnavailable:
      code = PIH_STATUS_UNAVAILABLE_V1;
      break;
    case pih::StatusCode::kDeadlineExceeded:
      code = PIH_STATUS_DEADLINE_EXCEEDED_V1;
      break;
  }
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  const auto message = status.message();
  const auto bytes = std::min(message.size(), sizeof(value.message) - 1);
  std::memcpy(value.message, message.data(), bytes);
  return value;
}

void Expect(const std::string& bytes, std::size_t& cursor,
            const char* literal) {
  const std::string expected(literal);
  if (bytes.compare(cursor, expected.size(), expected) != 0) {
    throw std::invalid_argument("deepseek_configuration_invalid");
  }
  cursor += expected.size();
}

uint64_t ReadUint64(const std::string& bytes, std::size_t& cursor) {
  if (cursor >= bytes.size() || bytes[cursor] < '0' || bytes[cursor] > '9') {
    throw std::invalid_argument("deepseek_configuration_integer_invalid");
  }
  uint64_t value = 0;
  while (cursor < bytes.size() && bytes[cursor] >= '0' &&
         bytes[cursor] <= '9') {
    const auto digit = static_cast<uint64_t>(bytes[cursor] - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
      throw std::invalid_argument("deepseek_configuration_integer_overflow");
    }
    value = value * 10 + digit;
    ++cursor;
  }
  return value;
}

uint32_t ReadUint32(const std::string& bytes, std::size_t& cursor) {
  const auto value = ReadUint64(bytes, cursor);
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("deepseek_configuration_integer_overflow");
  }
  return static_cast<uint32_t>(value);
}

std::string ReadString(const std::string& bytes, std::size_t& cursor) {
  const auto end = bytes.find('"', cursor);
  if (end == std::string::npos) {
    throw std::invalid_argument("deepseek_configuration_string_invalid");
  }
  auto value = bytes.substr(cursor, end - cursor);
  cursor = end + 1;
  return value;
}

bool SafeRelativePath(const std::string& value) {
  if (value.empty() || value.size() > 4096 || value.back() == '/' ||
      value.find('\\') != std::string::npos) {
    return false;
  }
  for (const char byte : value) {
    if (!((byte >= 'A' && byte <= 'Z') ||
          (byte >= 'a' && byte <= 'z') ||
          (byte >= '0' && byte <= '9') || byte == '.' || byte == '-' ||
          byte == '_' || byte == '/')) {
      return false;
    }
  }
  const std::filesystem::path path(value);
  if (path.is_absolute() || path.has_root_path() ||
      path.lexically_normal().generic_string() != value) {
    return false;
  }
  bool first = true;
  bool has_generation_component = false;
  for (const auto& component : path) {
    if (component.empty() || component == "." || component == ".." ||
        (first && component != "artifacts")) {
      return false;
    }
    if (!first) has_generation_component = true;
    first = false;
  }
  return has_generation_component;
}

bool CanonicalNonzeroSha256Hex(std::string_view value) {
  return value.size() == 64 &&
         std::ranges::all_of(value, [](char byte) {
           return (byte >= '0' && byte <= '9') ||
                  (byte >= 'a' && byte <= 'f');
         }) &&
         std::ranges::any_of(value, [](char byte) { return byte != '0'; });
}

struct ParsedPp1Config final {
  int32_t device_ordinal{};
  bool host_spill_enabled{};
  uint32_t expert_slot_count{};
  uint32_t staging_extent_count{};
  uint32_t artifact_poll_interval_ms{};
  uint32_t attention_reserved_tokens_per_sequence{};
  uint32_t maximum_prefill_chunk_tokens{};
  uint32_t maximum_decode_sequences{};
  uint32_t maximum_verify_sequences{};
  uint64_t maximum_shard_bytes{};
  std::string target_generation_root;
  std::string artifact_root_sha256_hex;
};

ParsedPp1Config ParsePp1Configuration(const pih_engine_create_request_v1& request) {
  if (!BoundedCString(request.deployment_root, kMaximumDeploymentRootBytes) ||
      request.configuration_json == nullptr || request.configuration_size == 0 ||
      request.configuration_size > kMaximumConfigurationBytes ||
      request.configuration_size > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("deepseek_configuration_missing");
  }
  const std::string bytes(request.configuration_json,
                          static_cast<std::size_t>(request.configuration_size));
  std::size_t cursor = 0;
  Expect(bytes, cursor,
         "{\"schema\":\"pih.deepseek-v4-flash.pp1.v1\","
         "\"device_ordinal\":");
  const auto device_ordinal = ReadUint64(bytes, cursor);
  if (device_ordinal > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("deepseek_configuration_integer_overflow");
  }
  ParsedPp1Config config;
  config.device_ordinal = static_cast<int32_t>(device_ordinal);
  Expect(bytes, cursor, ",\"host_spill_enabled\":");
  const auto spill = ReadUint64(bytes, cursor);
  if (spill > 1) throw std::invalid_argument("deepseek_configuration_boolean_invalid");
  config.host_spill_enabled = spill != 0;
  Expect(bytes, cursor, ",\"expert_slot_count\":");
  config.expert_slot_count = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"staging_extent_count\":");
  config.staging_extent_count = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"artifact_poll_interval_ms\":");
  config.artifact_poll_interval_ms = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"attention_reserved_tokens_per_sequence\":");
  config.attention_reserved_tokens_per_sequence = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"maximum_prefill_chunk_tokens\":");
  config.maximum_prefill_chunk_tokens = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"maximum_decode_sequences\":");
  config.maximum_decode_sequences = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"maximum_verify_sequences\":");
  config.maximum_verify_sequences = ReadUint32(bytes, cursor);
  Expect(bytes, cursor, ",\"maximum_shard_bytes\":");
  config.maximum_shard_bytes = ReadUint64(bytes, cursor);
  Expect(bytes, cursor, ",\"target_generation_root\":\"");
  auto relative_root = ReadString(bytes, cursor);
  Expect(bytes, cursor, ",\"artifact_root_sha256_hex\":\"");
  config.artifact_root_sha256_hex = ReadString(bytes, cursor);
  Expect(bytes, cursor, "}");
  if (cursor != bytes.size() || !SafeRelativePath(relative_root) ||
      !CanonicalNonzeroSha256Hex(config.artifact_root_sha256_hex)) {
    throw std::invalid_argument("deepseek_configuration_not_canonical");
  }
  const std::filesystem::path deployment_root(request.deployment_root);
  if (!deployment_root.is_absolute() || !deployment_root.has_root_path()) {
    throw std::invalid_argument("deepseek_deployment_root_not_absolute");
  }
  config.target_generation_root =
      (deployment_root / relative_root)
          .lexically_normal()
          .string();
  if (config.target_generation_root.size() > kMaximumDeploymentRootBytes) {
    throw std::invalid_argument("deepseek_generation_root_too_long");
  }
  return config;
}

pih_status_v1 CreatePp1(
    void* context, const DeepSeekPp1Request* request,
    pih_engine_handle_v1* engine) {
  if (context == nullptr || request == nullptr || engine == nullptr ||
      request->struct_size != sizeof(*request) ||
      !pih_engine_handle_is_empty_v1(engine) ||
      request->activation_epoch == 0 || request->device_ordinal < 0 ||
      request->maximum_prefill_chunk_tokens == 0 ||
      request->maximum_decode_sequences == 0 ||
      request->maximum_verify_sequences == 0 ||
      request->maximum_shard_bytes == 0 ||
      request->target_generation_root == nullptr ||
      request->artifact_root_sha256_hex == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "pp1_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.phase.load() != 4 || state.artifact == nullptr || state.cuda == nullptr ||
      state.cuda_memory == nullptr || state.execution == nullptr ||
      state.cuda_resources == nullptr || state.cuda_async == nullptr ||
      state.kernels == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "pp1_capabilities_not_bound");
  }
  auto digest = pih::Sha256Digest::ParseHex(
      request->artifact_root_sha256_hex);
  if (!digest.ok()) return EngineStatus(digest.status());
  pih_verified_artifact_generation_request_v1 artifact_request{};
  artifact_request.struct_size = sizeof(artifact_request);
  artifact_request.abi_version = PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1;
  artifact_request.generation_root = request->target_generation_root;
  artifact_request.root_sha256_hex = request->artifact_root_sha256_hex;
  artifact_request.maximum_shard_bytes = request->maximum_shard_bytes;
  const auto artifact_status = CheckedCapabilityStatus(
      state.artifact->validate_generation(state.artifact->context,
                                          &artifact_request),
      "artifact_provider_status_invalid");
  if (!pih_status_is_ok_v1(&artifact_status)) return artifact_status;
  auto artifact_pipeline = pih::DeepSeekPipelinePlan::Create(1, false);
  if (!artifact_pipeline.ok()) return EngineStatus(artifact_pipeline.status());
  // Seal the complete artifact authority before any CUDA initialization. A
  // malformed generation must fail without acquiring device-scoped state.
  auto artifact_catalog =
      pih::DeepSeekCapabilityArtifactCatalog::OpenFlash0731TargetGeneration(
          *state.artifact,
          std::filesystem::path(request->target_generation_root), *digest,
          *artifact_pipeline, request->maximum_shard_bytes);
  if (!artifact_catalog.ok()) return EngineStatus(artifact_catalog.status());
  std::uint32_t bound_expert_slot_count = request->expert_slot_count;
  std::uint32_t bound_staging_extent_count = request->staging_extent_count;
  std::uint64_t bound_host_spill_pinned_bytes = 0;
  std::uint64_t bound_host_spill_device_bytes = 0;
  std::uint32_t bound_transfer_reservation_window = 0;
  if (request->host_spill_enabled != 0) {
    if (state.host_spill == nullptr ||
        state.host_spill->struct_size != sizeof(*state.host_spill) ||
        state.host_spill->contract_version !=
            PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        state.host_spill->context == nullptr ||
        state.host_spill->compile_plan == nullptr ||
        state.host_spill->allocate_pinned == nullptr ||
        state.host_spill->deallocate_pinned == nullptr ||
        state.host_spill->copy_h2d_async == nullptr ||
        state.host_spill->record_event == nullptr ||
        state.host_spill->query_event == nullptr ||
        state.host_spill->create_completion_event == nullptr ||
        state.host_spill->destroy_completion_event == nullptr ||
        state.host_spill->allocate_device_slot == nullptr ||
        state.host_spill->deallocate_device_slot == nullptr ||
        state.host_spill->inspect == nullptr) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "host_spill_capability_invalid");
    }
    pih_memory_host_spill_plan_v1 spill_plan{};
    spill_plan.struct_size = sizeof(spill_plan);
    spill_plan.contract_version = PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1;
    const auto spill_status = CheckedCapabilityStatus(
        state.host_spill->compile_plan(
            state.host_spill->context, 1, request->expert_slot_count,
            request->staging_extent_count,
            pih::DeepSeekExpertPager::kBundleBytes, &spill_plan),
        "host_spill_provider_status_invalid");
    if (!pih_status_is_ok_v1(&spill_status)) return spill_status;
    if (spill_plan.struct_size != sizeof(spill_plan) ||
        spill_plan.contract_version != PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        spill_plan.enabled != 1 ||
        spill_plan.expert_slot_count != request->expert_slot_count ||
        spill_plan.staging_extent_count != request->staging_extent_count ||
        spill_plan.transfer_reservation_window !=
            pih::DeepSeekExpertPager::kTransferReservationWindow ||
        spill_plan.transfer_reservation_window >
            spill_plan.expert_slot_count ||
        spill_plan.transfer_reservation_window >
            spill_plan.staging_extent_count ||
        spill_plan.reserved != 0 ||
        spill_plan.expert_bundle_bytes !=
            pih::DeepSeekExpertPager::kBundleBytes ||
        spill_plan.pinned_bytes !=
            pih::DeepSeekExpertPager::kBundleBytes *
                request->staging_extent_count ||
        spill_plan.device_bytes !=
            pih::DeepSeekExpertPager::kBundleBytes *
                request->expert_slot_count) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "host_spill_plan_contract_mismatch");
    }
    bound_expert_slot_count = spill_plan.expert_slot_count;
    bound_staging_extent_count = spill_plan.staging_extent_count;
    bound_host_spill_pinned_bytes = spill_plan.pinned_bytes;
    bound_host_spill_device_bytes = spill_plan.device_bytes;
    bound_transfer_reservation_window =
        spill_plan.transfer_reservation_window;
  } else if (request->expert_slot_count != 0 ||
             request->staging_extent_count != 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "resident_plan_has_host_spill_capacity");
  }
  pih_execution_capacity_request_v1 capacity_request{};
  capacity_request.struct_size = sizeof(capacity_request);
  capacity_request.abi_version = PIH_EXECUTION_DEFAULT_ABI_VERSION_V1;
  capacity_request.maximum_prefill_chunk_tokens =
      request->maximum_prefill_chunk_tokens;
  capacity_request.maximum_decode_sequences =
      request->maximum_decode_sequences;
  capacity_request.maximum_verify_sequences =
      request->maximum_verify_sequences;
  capacity_request.speculative_tokens_per_sequence = 0;
  pih_execution_capacity_v1 bound_capacity{};
  bound_capacity.struct_size = sizeof(bound_capacity);
  bound_capacity.abi_version = PIH_EXECUTION_DEFAULT_ABI_VERSION_V1;
  const auto execution_status = CheckedCapabilityStatus(
      state.execution->compile_capacity(
          state.execution->context, &capacity_request, &bound_capacity),
      "execution_provider_status_invalid");
  if (!pih_status_is_ok_v1(&execution_status)) return execution_status;
  if (bound_capacity.struct_size != sizeof(bound_capacity) ||
      bound_capacity.abi_version != PIH_EXECUTION_DEFAULT_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "execution_capacity_result_invalid");
  }
  auto capacity = pih::DeepSeekPipelineCapacity::Create(
      1, request->maximum_prefill_chunk_tokens,
      request->maximum_decode_sequences, request->maximum_verify_sequences,
      false);
  if (!capacity.ok()) return EngineStatus(capacity.status());
  if (capacity->max_pipeline_tokens !=
          bound_capacity.max_pipeline_tokens ||
      capacity->max_sequences != bound_capacity.maximum_sequences ||
      capacity->expert_tokens != bound_capacity.expert_tokens) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "execution_capacity_contract_mismatch");
  }
  const auto residency = request->host_spill_enabled != 0
      ? pih::DeepSeekRoutedExpertResidency::kHostSpill
      : pih::DeepSeekRoutedExpertResidency::kFullResident;
  auto config = pih::NvidiaDeepSeekBootstrapConfig::Create(
      request->activation_epoch, 1, {request->device_ordinal}, residency,
      bound_expert_slot_count, bound_staging_extent_count,
      request->artifact_poll_interval_ms,
      request->attention_reserved_tokens_per_sequence,
      pih::DeepSeekExecutionTopology::kInProcessRankSetDevelopment,
      bound_host_spill_pinned_bytes, bound_host_spill_device_bytes,
      bound_transfer_reservation_window);
  if (!config.ok()) return EngineStatus(config.status());
  // All filesystem and pure planning failures have been closed above; device
  // preparation is the first hardware-side effect in engine creation.
  const auto device_status = CheckedCapabilityStatus(
      state.cuda->prepare_device(state.cuda->context,
                                 request->device_ordinal, 8, 9),
      "cuda_provider_status_invalid");
  if (!pih_status_is_ok_v1(&device_status)) return device_status;
  BoundRankRuntimeFactory runtime_factory(*state.kernels,
                                          *state.cuda_memory,
                                          *state.cuda_resources,
                                          *state.cuda_async,
                                          request->host_spill_enabled != 0
                                              ? state.host_spill
                                              : nullptr);
  const auto retain_unpublished_cleanup =
      [&](pih_status_v1 failure) -> pih_status_v1 {
    engine->activation_epoch = request->activation_epoch;
    engine->instance = &state;
    std::memcpy(engine->model_id, kModelId, sizeof(kModelId));
    return failure;
  };
  // Normalize exceptions at the first hardware-owning construction boundary.
  // Stack unwinding releases partially built runtime resources before the
  // common failed-build path inspects host-spill and, if necessary, publishes
  // a retryable cleanup-only handle.
  auto built = [&]() -> pih::Result<std::unique_ptr<pih::DeepSeekEngine>> {
    try {
      return pih::NvidiaDeepSeekEngineBootstrap::
          BuildDevelopmentSm89Pp1FromArtifactCatalogWithRuntimeFactory(
              std::move(*artifact_catalog), std::move(*capacity), *config,
              runtime_factory);
    } catch (const std::bad_alloc&) {
      return pih::Status::ResourceExhausted(
          "DeepSeek engine construction allocation failed");
    } catch (...) {
      return pih::Status::Internal("DeepSeek engine construction failed");
    }
  }();
  if (!built.ok()) {
    if (request->host_spill_enabled != 0) {
      pih_memory_host_spill_snapshot_v1 failed_snapshot{};
      failed_snapshot.struct_size = sizeof(failed_snapshot);
      failed_snapshot.contract_version =
          PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1;
      const auto failed_snapshot_status = CheckedCapabilityStatus(
          state.host_spill->inspect(state.host_spill->context,
                                    &failed_snapshot),
          "host_spill_provider_status_invalid");
      if (!pih_status_is_ok_v1(&failed_snapshot_status)) {
        return retain_unpublished_cleanup(failed_snapshot_status);
      }
      if (failed_snapshot.struct_size != sizeof(failed_snapshot) ||
          failed_snapshot.contract_version !=
              PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
          failed_snapshot.enabled != 1 ||
          failed_snapshot.device_ordinal != -1 ||
          failed_snapshot.live_expert_slot_count != 0 ||
          failed_snapshot.live_staging_extent_count != 0 ||
          failed_snapshot.completion_event_count != 0 ||
          failed_snapshot.pinned_live_bytes != 0 ||
          failed_snapshot.device_live_bytes != 0) {
        return retain_unpublished_cleanup(
            Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                   "host_spill_creation_failure_resources_live"));
      }
    }
    return EngineStatus(built.status());
  }
  const auto release_unpublished_engine =
      [&](pih_status_v1 failure) -> pih_status_v1 {
    built->reset();
    if (request->host_spill_enabled == 0) return failure;
    pih_memory_host_spill_snapshot_v1 released{};
    released.struct_size = sizeof(released);
    released.contract_version = PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1;
    const auto released_status = CheckedCapabilityStatus(
        state.host_spill->inspect(state.host_spill->context, &released),
        "host_spill_provider_status_invalid");
    if (!pih_status_is_ok_v1(&released_status)) {
      return retain_unpublished_cleanup(released_status);
    }
    if (released.struct_size != sizeof(released) ||
        released.contract_version != PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        released.enabled != 1 || released.device_ordinal != -1 ||
        released.live_expert_slot_count != 0 ||
        released.live_staging_extent_count != 0 ||
        released.completion_event_count != 0 ||
        released.pinned_live_bytes != 0 || released.device_live_bytes != 0) {
      return retain_unpublished_cleanup(
          Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                 "host_spill_unpublished_engine_resources_live"));
    }
    return failure;
  };
  if (built->get() == nullptr) {
    return release_unpublished_engine(
        Status(PIH_STATUS_INTERNAL_V1, "deepseek_engine_instance_missing"));
  }
  if (request->host_spill_enabled != 0) {
    pih_memory_host_spill_snapshot_v1 snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.contract_version = PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1;
    const auto snapshot_status = CheckedCapabilityStatus(
        state.host_spill->inspect(state.host_spill->context, &snapshot),
        "host_spill_provider_status_invalid");
    if (!pih_status_is_ok_v1(&snapshot_status)) {
      return release_unpublished_engine(snapshot_status);
    }
    if (snapshot.struct_size != sizeof(snapshot) ||
        snapshot.contract_version != PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        snapshot.enabled != 1 ||
        snapshot.device_ordinal != request->device_ordinal ||
        snapshot.expert_slot_count != bound_expert_slot_count ||
        snapshot.live_expert_slot_count != bound_expert_slot_count ||
        snapshot.staging_extent_count != bound_staging_extent_count ||
        snapshot.live_staging_extent_count != bound_staging_extent_count ||
        snapshot.completion_event_count !=
            bound_transfer_reservation_window ||
        snapshot.pinned_budget_bytes != bound_host_spill_pinned_bytes ||
        snapshot.pinned_live_bytes != bound_host_spill_pinned_bytes ||
        snapshot.device_budget_bytes != bound_host_spill_device_bytes ||
        snapshot.device_live_bytes != bound_host_spill_device_bytes) {
      return release_unpublished_engine(
          Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                 "host_spill_runtime_snapshot_mismatch"));
    }
  }
  engine->activation_epoch = request->activation_epoch;
  engine->instance = built->release();
  std::memcpy(engine->model_id, kModelId, sizeof(kModelId));
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 CreateCore(void* context,
                         const pih_engine_create_request_v1* request,
                         pih_engine_handle_v1* engine) {
  if (context == nullptr || request == nullptr ||
      request->struct_size != sizeof(*request) ||
      request->abi_version != PIH_ENGINE_ABI_VERSION_V1 ||
      request->activation_epoch == 0 ||
      !pih_engine_handle_is_empty_v1(engine)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "engine_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  if (state.host == nullptr ||
      state.host->activation_epoch != request->activation_epoch ||
      state.phase.load() != 4 || state.engine_instance != nullptr ||
      state.engine_handle_identity != nullptr ||
      state.engine_generation_issued) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_not_creatable");
  }
  if (state.next_engine_generation == std::numeric_limits<uint64_t>::max()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_engine_generation_exhausted");
  }
  try {
    const auto config = ParsePp1Configuration(*request);
    DeepSeekPp1Request pp1{};
    pp1.struct_size = sizeof(pp1);
    pp1.activation_epoch = request->activation_epoch;
    pp1.device_ordinal = config.device_ordinal;
    pp1.host_spill_enabled = config.host_spill_enabled ? 1U : 0U;
    pp1.expert_slot_count = config.expert_slot_count;
    pp1.staging_extent_count = config.staging_extent_count;
    pp1.artifact_poll_interval_ms = config.artifact_poll_interval_ms;
    pp1.attention_reserved_tokens_per_sequence =
        config.attention_reserved_tokens_per_sequence;
    pp1.maximum_prefill_chunk_tokens = config.maximum_prefill_chunk_tokens;
    pp1.maximum_decode_sequences = config.maximum_decode_sequences;
    pp1.maximum_verify_sequences = config.maximum_verify_sequences;
    pp1.maximum_shard_bytes = config.maximum_shard_bytes;
    pp1.target_generation_root = config.target_generation_root.c_str();
    pp1.artifact_root_sha256_hex = config.artifact_root_sha256_hex.c_str();
    const auto status = CreatePp1(context, &pp1, engine);
    if (pih_status_is_ok_v1(&status)) {
      engine->generation = state.next_engine_generation++;
      state.engine_instance =
          static_cast<pih::DeepSeekEngine*>(engine->instance);
      state.engine_handle_identity = engine->instance;
      state.engine_activation_epoch = engine->activation_epoch;
      state.engine_generation = engine->generation;
      state.engine_host_spill_enabled = config.host_spill_enabled;
      state.engine_snapshot_pending = false;
      state.engine_cleanup_failed = false;
      state.engine_generation_issued = true;
      state.context_capacity = config.attention_reserved_tokens_per_sequence;
      state.prefill_chunk = config.maximum_prefill_chunk_tokens;
    } else if (engine->instance != nullptr) {
      if (engine->instance != &state ||
          engine->activation_epoch != request->activation_epoch ||
          !ExactEngineModelId(engine->model_id)) {
        return Status(PIH_STATUS_INTERNAL_V1,
                      "deepseek_cleanup_handle_invalid");
      }
      // Build rollback has already destroyed every engine-owned resource, but
      // a failed final host-spill inspection still needs a live ABI identity
      // so the worker can retry that proof through destroy().
      engine->generation = state.next_engine_generation++;
      state.engine_instance = nullptr;
      state.engine_handle_identity = engine->instance;
      state.engine_activation_epoch = engine->activation_epoch;
      state.engine_generation = engine->generation;
      state.engine_host_spill_enabled = config.host_spill_enabled;
      state.engine_snapshot_pending = true;
      state.engine_cleanup_failed = true;
      state.engine_generation_issued = true;
    }
    return status;
  } catch (const std::invalid_argument&) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "deepseek_configuration_invalid");
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_engine_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "deepseek_engine_creation_failed");
  }
}

pih_status_v1 Create(void* context, const pih_engine_create_request_v1* request,
                     pih_engine_handle_v1* engine) {
  try {
    return CreateCore(context, request, engine);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_engine_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "deepseek_engine_creation_failed");
  }
}

pih_status_v1 AdmitCore(void* context, const pih_engine_handle_v1* engine) {
  if (context == nullptr || !pih_engine_handle_is_live_v1(engine) ||
      !ExactEngineModelId(engine->model_id)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "engine_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  if (state.phase.load() != 4 || state.engine_instance == nullptr ||
      engine->instance != state.engine_handle_identity ||
      engine->activation_epoch != state.engine_activation_epoch ||
      engine->generation != state.engine_generation) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_handle_stale");
  }
  return EngineStatus(state.engine_instance->admit_request());
}

pih_status_v1 Admit(void* context, const pih_engine_handle_v1* engine) {
  try {
    return AdmitCore(context, engine);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_admission_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "deepseek_admission_failed");
  }
}

pih_status_v1 ExecuteSmokeRequestCore(
    void* context, const pih_engine_handle_v1* engine,
    const pih_engine_smoke_request_v1* request,
    pih_engine_smoke_result_v1* result) {
  if (result == nullptr || result->struct_size != sizeof(*result) ||
      result->abi_version != PIH_ENGINE_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "engine_smoke_request_invalid");
  }
  *result = {sizeof(*result), PIH_ENGINE_ABI_VERSION_V1};
  if (context == nullptr || !pih_engine_handle_is_live_v1(engine) ||
      !ExactEngineModelId(engine->model_id)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "engine_handle_invalid");
  }
  if (request == nullptr ||
      request->struct_size != sizeof(*request) ||
      request->abi_version != PIH_ENGINE_ABI_VERSION_V1 ||
      request->deadline_monotonic_ns == 0 ||
      request->deadline_monotonic_ns >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      request->prompt_count != 1 || request->prompt_bytes == nullptr ||
      request->prompt_size != 3 ||
      std::string_view(request->prompt_bytes,
                       static_cast<std::size_t>(request->prompt_size)) !=
          "pih" ||
      request->maximum_completion_tokens != 1 ||
      request->temperature != 0.0F || request->top_p != 1.0F ||
      request->seed_present != 1 || request->seed != 7 ||
      request->logprobs_enabled != 1 ||
      request->top_logprobs_count !=
          PIH_ENGINE_SMOKE_TOP_LOGPROBS_MAX_V1 ||
      request->stop_count != 1 || request->stop_bytes == nullptr ||
      request->stop_size != 3 ||
      std::string_view(request->stop_bytes,
                       static_cast<std::size_t>(request->stop_size)) !=
          "END") {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "engine_smoke_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  if (state.phase.load() != 4 || state.engine_instance == nullptr ||
      engine->instance != state.engine_handle_identity ||
      engine->activation_epoch != state.engine_activation_epoch ||
      engine->generation != state.engine_generation) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_handle_stale");
  }
  auto& instance = *state.engine_instance;
  const auto execution_deadline = std::chrono::steady_clock::time_point(
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::nanoseconds(request->deadline_monotonic_ns)));
  if (std::chrono::steady_clock::now() >= execution_deadline) {
    return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                  "deepseek_smoke_deadline_elapsed");
  }
  if (state.next_request_id == std::numeric_limits<uint64_t>::max() ||
      state.next_request_generation ==
          std::numeric_limits<uint64_t>::max() ||
      state.next_plan_sequence == std::numeric_limits<uint64_t>::max()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_request_identity_exhausted");
  }
  pih::DeepSeekRequestSamplingConfig sampling{};
  sampling.mode = pih::DeepSeekSamplingMode::kGreedy;
  sampling.effective_seed = request->seed;
  sampling.temperature = request->temperature;
  sampling.top_p = request->top_p;
  sampling.logprobs_enabled = true;
  sampling.top_logprobs_count = request->top_logprobs_count;
  const uint32_t smoke_token = 0;
  auto accepted = pih::deepseek_plugin::Generate(instance, {&smoke_token, 1},
      1, 0, sampling, state.prefill_chunk, state.context_capacity,
      state.next_request_id++, state.next_request_generation++,
      state.next_plan_sequence, execution_deadline);
  if (!accepted.ok()) return EngineStatus(accepted.status());
  if (accepted->token_ids.size() != 1 ||
      accepted->accepted_completion_count != 1 ||
      accepted->finish_reason != pih::DeepSeekFinishReason::kLength ||
      accepted->selected_logprobs.size() != 1 ||
      accepted->top_logprobs.size() != 1 ||
      accepted->top_logprobs.front().size() != PIH_ENGINE_SMOKE_TOP_LOGPROBS_MAX_V1)
    return Status(PIH_STATUS_INTERNAL_V1, "deepseek_smoke_output_invalid");
  pih_engine_smoke_result_v1 completed_result{
      sizeof(completed_result), PIH_ENGINE_ABI_VERSION_V1};
  completed_result.token_id = accepted->token_ids.front();
  completed_result.selected_logprob = accepted->selected_logprobs.front();
  completed_result.top_logprobs_count =
      static_cast<uint32_t>(accepted->top_logprobs.front().size());
  for (uint32_t index = 0; index < completed_result.top_logprobs_count;
       ++index) {
    completed_result.top_logprobs[index].token_id =
        accepted->top_logprobs.front()[index].token_id;
    completed_result.top_logprobs[index].logprob =
        accepted->top_logprobs.front()[index].logprob;
  }
  completed_result.finish_reason_length = 1;
  *result = completed_result;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ExecuteSmokeRequest(void* context,
                                  const pih_engine_handle_v1* engine,
                                  const pih_engine_smoke_request_v1* request,
                                  pih_engine_smoke_result_v1* result) {
  try {
    return ExecuteSmokeRequestCore(context, engine, request, result);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_request_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "deepseek_request_execution_failed");
  }
}

pih_status_v1 GenerateTokens(void* context, const pih_engine_handle_v1* engine,
    const pih_token_generation_request_v1* request,
    pih_token_generation_result_v1* result) noexcept {
  try {
    if (!result || result->struct_size != sizeof(*result) ||
        result->contract_version != PIH_TOKEN_GENERATION_ABI_V1)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "token_result_invalid");
    result->token_count = 0;
    result->finish_reason = 0;
    result->prompt_token_count = result->completion_token_count = 0;
    if (!context || !pih_engine_handle_is_live_v1(engine) ||
        !ExactEngineModelId(engine->model_id) || !request ||
        request->struct_size != sizeof(*request) ||
        request->contract_version != PIH_TOKEN_GENERATION_ABI_V1 ||
        !request->prompt_tokens || !result->tokens ||
        request->prompt_token_count == 0 || request->maximum_completion_tokens == 0 ||
        result->token_capacity < request->maximum_completion_tokens ||
        request->minimum_completion_tokens > request->maximum_completion_tokens ||
        request->stop_token_count > 16 ||
        (request->stop_token_count != 0 && !request->stop_tokens) ||
        !std::isfinite(request->temperature) || request->temperature < 0 || request->temperature > 2 ||
        !std::isfinite(request->top_p) || request->top_p <= 0 || request->top_p > 1 ||
        (request->temperature == 0 && request->top_p != 1) ||
        request->deadline_monotonic_ns == 0 ||
        request->deadline_monotonic_ns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "token_request_invalid");
    pih::DeepSeekRequestSamplingConfig sampling{};
    sampling.mode = request->temperature == 0 ? pih::DeepSeekSamplingMode::kGreedy
                                             : pih::DeepSeekSamplingMode::kStochastic;
    sampling.temperature = request->temperature;
    sampling.top_p = request->top_p;
    sampling.effective_seed = request->seed;
    sampling.stop_token_count = request->stop_token_count;
    for (uint32_t index = 0; index < request->stop_token_count; ++index) {
      const auto token = request->stop_tokens[index];
      if (token >= 129280 || std::find(request->stop_tokens, request->stop_tokens + index, token) != request->stop_tokens + index)
        return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "stop_token_invalid");
      sampling.stop_token_ids[index] = token;
    }
    auto& state = *static_cast<State*>(context);
    std::lock_guard transaction(state.engine_mutex);
    if (state.phase.load() != 4 || !state.engine_instance ||
        engine->instance != state.engine_handle_identity ||
        engine->activation_epoch != state.engine_activation_epoch ||
        engine->generation != state.engine_generation)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "deepseek_engine_handle_stale");
    if (state.next_request_id == std::numeric_limits<uint64_t>::max() ||
        state.next_request_generation == std::numeric_limits<uint64_t>::max())
      return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "deepseek_request_identity_exhausted");
    const auto deadline = std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(request->deadline_monotonic_ns)));
    auto generated = pih::deepseek_plugin::Generate(*state.engine_instance,
        {request->prompt_tokens, request->prompt_token_count},
        request->maximum_completion_tokens, request->minimum_completion_tokens,
        sampling, state.prefill_chunk, state.context_capacity,
        state.next_request_id++, state.next_request_generation++, state.next_plan_sequence, deadline);
    if (!generated.ok()) return EngineStatus(generated.status());
    if (generated->token_ids.size() > result->token_capacity ||
        (generated->finish_reason != pih::DeepSeekFinishReason::kStop &&
         generated->finish_reason != pih::DeepSeekFinishReason::kLength))
      return Status(PIH_STATUS_INTERNAL_V1, "generated_result_invalid");
    std::copy(generated->token_ids.begin(), generated->token_ids.end(), result->tokens);
    result->token_count = static_cast<uint32_t>(generated->token_ids.size());
    result->finish_reason = generated->finish_reason == pih::DeepSeekFinishReason::kStop
        ? PIH_TOKEN_FINISH_STOP_V1 : PIH_TOKEN_FINISH_LENGTH_V1;
    result->prompt_token_count = request->prompt_token_count;
    result->completion_token_count = generated->accepted_completion_count;
    return Status(PIH_STATUS_OK_V1);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "generation_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, "generation_failed");
  }
}

pih_status_v1 DestroyCore(void* context, pih_engine_handle_v1* engine) {
  if (context == nullptr || !pih_engine_handle_is_live_v1(engine) ||
      !ExactEngineModelId(engine->model_id)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "engine_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  if (state.phase.load() != 4 ||
      engine->instance != state.engine_handle_identity ||
      engine->activation_epoch != state.engine_activation_epoch ||
      engine->generation != state.engine_generation) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_handle_stale");
  }
  auto* instance = state.engine_instance;
  if (!state.engine_snapshot_pending) {
    const auto close = instance->close();
    if (!close.ok()) {
      // A failed close can mean that submitted device work has no trustworthy
      // completion frontier. Keep the entire engine ownership graph alive and
      // leave the caller's handle bound so no CUDA or host-spill allocation is
      // released or reused before this fail-stop worker exits.
      state.engine_cleanup_failed = true;
      return EngineStatus(close);
    }
    delete instance;
    state.engine_instance = nullptr;
    state.engine_handle_identity = &state;
    engine->instance = &state;
    // A stable state-owned token replaces the deleted engine pointer. The
    // pending flag prevents every later retry from dereferencing or deleting
    // it. Keeping the ABI handle non-empty prevents the worker from treating
    // a transient provider-snapshot failure as completed cleanup.
    state.engine_snapshot_pending = true;
  }
  const bool inspect_host_spill = state.engine_host_spill_enabled;
  // A prior close attempt may have been transient (for example, asynchronous
  // boundary teardown still in progress). A successful close supersedes that
  // attempt. Retain the caller-visible cleanup token until the final provider
  // snapshot proves that every host-spill allocation was released, so a
  // transient snapshot failure remains retryable.
  state.engine_cleanup_failed = inspect_host_spill;
  pih_status_v1 resource_release = Status(PIH_STATUS_OK_V1);
  if (inspect_host_spill) {
    pih_memory_host_spill_snapshot_v1 snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.contract_version = PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1;
    resource_release = CheckedCapabilityStatus(
        state.host_spill->inspect(state.host_spill->context, &snapshot),
        "host_spill_provider_status_invalid");
    if (pih_status_is_ok_v1(&resource_release) &&
        (snapshot.struct_size != sizeof(snapshot) ||
         snapshot.contract_version != PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
         snapshot.enabled != 1 || snapshot.device_ordinal != -1 ||
         snapshot.live_expert_slot_count != 0 ||
         snapshot.live_staging_extent_count != 0 ||
         snapshot.completion_event_count != 0 ||
         snapshot.pinned_live_bytes != 0 ||
         snapshot.device_live_bytes != 0)) {
      resource_release = Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                                "host_spill_resources_not_released");
    }
  }
  state.engine_cleanup_failed = !pih_status_is_ok_v1(&resource_release);
  if (!pih_status_is_ok_v1(&resource_release)) return resource_release;
  state.engine_instance = nullptr;
  state.engine_handle_identity = nullptr;
  state.engine_activation_epoch = 0;
  state.engine_generation = 0;
  state.engine_host_spill_enabled = false;
  state.engine_snapshot_pending = false;
  engine->instance = nullptr;
  engine->activation_epoch = 0;
  engine->generation = 0;
  std::memset(engine->model_id, 0, sizeof(engine->model_id));
  return resource_release;
}

pih_status_v1 Destroy(void* context, pih_engine_handle_v1* engine) {
  try {
    return DestroyCore(context, engine);
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "deepseek_engine_destruction_failed");
  }
}

pih_status_v1 LoadText(void* context, const char* configuration, uint64_t bytes) noexcept {
  if (!context || !configuration || bytes == 0 || bytes > 64 * 1024)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "text_configuration_invalid");
  auto& state = *static_cast<State*>(context);
  try {
    std::unique_lock text_lock(state.text_mutex, std::try_to_lock);
    if (!text_lock.owns_lock()) return Status(PIH_STATUS_UNAVAILABLE_V1, "text_engine_busy");
    if (state.text_semantic || !pih_engine_handle_is_empty_v1(&state.text_handle) ||
        state.phase.load() != 4 || !state.host)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "text_engine_not_loadable");
    auto parsed = pih::JsonValue::Parse({configuration, static_cast<size_t>(bytes)}, {64 * 1024, 8, 128, 32 * 1024});
    if (!parsed.ok()) return EngineStatus(parsed.status());
    const auto& value = *parsed;
    auto field = [&](const char* name) -> const pih::JsonValue& {
      const auto* result = value.at(name);
      if (!result) throw std::invalid_argument("text configuration field missing");
      return *result;
    };
    if (value.object().size() != 5 || field("schema").string() != "pih.deepseek-v4-flash.text.v1")
      throw std::invalid_argument("text configuration schema invalid");
    const auto root = field("deployment_root").string();
    const auto semantic_root = field("semantic_snapshot").string();
    const auto native_configuration = field("engine_configuration").string();
    const auto timeout = field("generation_timeout_ms").integer();
    if (timeout < 1 || timeout > 86400000)
      throw std::invalid_argument("text generation timeout invalid");
    for (const auto* path : {&root, &semantic_root})
      if (path->empty() || path->size() > 4096 || path->find('\0') != std::string::npos ||
          !std::filesystem::path(*path).is_absolute())
        throw std::invalid_argument("text configuration paths must be absolute");
    auto semantic = std::make_unique<pih::plugin_text::DeepSeekSemanticArtifacts>(
        pih::plugin_text::DeepSeekSemanticArtifacts::Load(semantic_root));
    pih_engine_create_request_v1 request{};
    request.struct_size = sizeof(request); request.abi_version = PIH_ENGINE_ABI_VERSION_V1;
    request.activation_epoch = state.host->activation_epoch;
    request.deployment_root = root.c_str(); request.configuration_json = native_configuration.data();
    request.configuration_size = native_configuration.size();
    const auto created = Create(context, &request, &state.text_handle);
    if (!pih_status_is_ok_v1(&created)) return created; // close retains any rollback handle
    state.text_semantic = std::move(semantic); state.text_failed = false;
    state.text_generation_timeout_ms = static_cast<uint32_t>(timeout);
    return Status(PIH_STATUS_OK_V1);
  } catch (const std::invalid_argument&) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "deepseek_text_configuration_or_semantics_invalid");
  } catch (const std::bad_variant_access&) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "deepseek_text_configuration_type_invalid");
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "deepseek_text_allocation_failed");
  } catch (...) { return Status(PIH_STATUS_INTERNAL_V1, "deepseek_text_load_failed"); }
}

pih_status_v1 CompleteText(void* context, uint32_t chat, const char* request, uint64_t bytes,
    char* response, uint64_t capacity, uint64_t* written, const pih_text_output_sink_v2* sink) noexcept {
  if (written) *written = 0;
  if (!context || chat > 1 || !request || !bytes || bytes > 1024 * 1024 || !response || !capacity || !written ||
      !sink || sink->struct_size != sizeof(*sink) || sink->contract_version != PIH_TEXT_INFERENCE_ABI_V2 ||
      !sink->start || !sink->write || !sink->cancelled)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "deepseek_text_request_invalid");
  auto& state = *static_cast<State*>(context);
  try {
    std::unique_lock text_lock(state.text_mutex, std::try_to_lock);
    if (!text_lock.owns_lock()) return Status(PIH_STATUS_UNAVAILABLE_V1, "text_engine_busy");
    std::lock_guard engine_lock(state.engine_mutex);
    if (!state.text_semantic || state.text_failed || !state.engine_instance || state.engine_snapshot_pending ||
        state.phase.load() != 4 || !pih_engine_handle_is_live_v1(&state.text_handle) ||
        state.text_handle.instance != state.engine_handle_identity)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "text_engine_not_ready");
    try {
      auto result = pih::deepseek_plugin::CompleteText(*state.engine_instance, *state.text_semantic,
          state.context_capacity, state.prefill_chunk, state.text_generation_timeout_ms,
          state.next_request_id, state.next_request_generation,
          state.next_plan_sequence, chat != 0, {request, static_cast<size_t>(bytes)}, *sink);
      if (!result.ok()) {
        if (result.status().code() != pih::StatusCode::kInvalidArgument &&
            result.status().code() != pih::StatusCode::kUnavailable &&
            result.status().code() != pih::StatusCode::kDeadlineExceeded) state.text_failed = true;
        return EngineStatus(result.status());
      }
      if (result->size() > capacity) return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "text_response_buffer_too_small");
      std::memcpy(response, result->data(), result->size()); *written = result->size();
      return Status(PIH_STATUS_OK_V1);
    } catch (const std::invalid_argument&) {
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "deepseek_text_parameters_invalid");
    } catch (const std::bad_variant_access&) {
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "deepseek_text_parameter_type_invalid");
    } catch (...) { state.text_failed = true; throw; }
  } catch (...) { return Status(PIH_STATUS_INTERNAL_V1, "deepseek_text_execution_failed"); }
}

pih_status_v1 CloseText(void* context) noexcept {
  if (!context) return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "text_context_invalid");
  auto& state = *static_cast<State*>(context);
  try {
    std::unique_lock text_lock(state.text_mutex, std::try_to_lock);
    if (!text_lock.owns_lock()) return Status(PIH_STATUS_UNAVAILABLE_V1, "text_engine_busy");
    if (!pih_engine_handle_is_empty_v1(&state.text_handle)) {
      const auto status = Destroy(context, &state.text_handle);
      if (!pih_status_is_ok_v1(&status)) return status;
    }
    state.text_semantic.reset(); state.text_failed = false;
    return Status(PIH_STATUS_OK_V1);
  } catch (...) { return Status(PIH_STATUS_INTERNAL_V1, "deepseek_text_close_failed"); }
}

pih_status_v1 RegisterCore(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.capability_id = "engine.primary.v1";
  capability.contract_id = "pih.engine.v1";
  static pih_engine_factory_api_v1 engine_factory_api{
      sizeof(pih_engine_factory_api_v1), PIH_ENGINE_ABI_VERSION_V1,
      PIH_ENGINE_HANDLE_GENERATION_BOUND_READ_ONLY_V1, &state, &Create,
      &Admit, &ExecuteSmokeRequest, &Destroy};
  capability.api = &engine_factory_api;
  capability.threading_model = PIH_CAPABILITY_THREADING_SERIALIZED_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  static pih_token_generation_api_v1 generation_api{
      sizeof(pih_token_generation_api_v1), PIH_TOKEN_GENERATION_ABI_V1,
      &state, &GenerateTokens};
  capability.capability_id = "inference.tokens.v1";
  capability.contract_id = "pih.inference.tokens.v1";
  capability.api = &generation_api;
  const auto generation_status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&generation_status)) return generation_status;
  static pih_text_inference_api_v2 text_api{
      sizeof(pih_text_inference_api_v2), PIH_TEXT_INFERENCE_ABI_V2, &state,
      "deepseek-ai/DeepSeek-V4-Flash-0731", &LoadText, &CompleteText, &CloseText};
  capability.capability_id = "inference.text.v2";
  capability.contract_id = "pih.inference.text.v2";
  capability.api = &text_api;
  const auto text_status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability), "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&text_status)) return text_status;
  return Advance(context, 0);
}

pih_status_v1 Resolve(State& state, const char* capability_id,
                      const char* contract_id, uint32_t required_scope,
                      uint32_t required_cardinality, const void** api) {
  if (state.host->resolve_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_resolver_missing");
  }
  return CheckedCapabilityStatus(
      state.host->resolve_capability(
          state.host->context, capability_id, contract_id, required_scope,
          required_cardinality, api),
      "capability_resolver_status_invalid");
}

pih_status_v1 ConfigureCore(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.phase.load() != 1) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  CapabilityConfigurationTransaction transaction(state);
  const void* api = nullptr;
  auto status = Resolve(state, "artifact.verified-reader.v1",
                        "pih.artifact.verified-reader.v1",
                        PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                        PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.artifact = static_cast<const pih_verified_artifact_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "device.cuda-runtime.v1",
                   "pih.device.cuda-runtime.v1", PIH_CAPABILITY_SCOPE_PROCESS_V1,
                   PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.cuda = static_cast<const pih_nvidia_cuda_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "device.cuda-memory.v1",
                   "pih.device.cuda-memory.v1", PIH_CAPABILITY_SCOPE_PROCESS_V1,
                   PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.cuda_memory =
      static_cast<const pih_nvidia_cuda_memory_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "device.cuda-resources.v1",
                   "pih.device.cuda-resources.v1",
                   PIH_CAPABILITY_SCOPE_PROCESS_V1,
                   PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.cuda_resources =
      static_cast<const pih_nvidia_cuda_resources_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "device.cuda-async.v1",
                   "pih.device.cuda-async.v1", PIH_CAPABILITY_SCOPE_PROCESS_V1,
                   PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.cuda_async = static_cast<const pih_nvidia_cuda_async_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "execution.default.v1", "pih.execution.default.v1",
                   PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                   PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.execution = static_cast<const pih_execution_default_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "memory.host-spill.v1",
                   "pih.memory.host-spill.v1",
                   PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                   PIH_CAPABILITY_CARDINALITY_ZERO_OR_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.host_spill = static_cast<const pih_memory_host_spill_api_v1*>(api);
  api = nullptr;
  status = Resolve(state, "pih.kernels.deepseek-v4.sm89",
                   "pih.deepseek-sm89-kernel-pack.v1",
                   PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                   PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
  if (!pih_status_is_ok_v1(&status)) return status;
  state.kernels =
      static_cast<const pih_deepseek_kernels_api_v1*>(api);
  if (state.artifact == nullptr || state.cuda == nullptr ||
      state.cuda_memory == nullptr || state.cuda_resources == nullptr ||
      state.cuda_async == nullptr || state.execution == nullptr ||
      state.kernels == nullptr ||
      state.artifact->struct_size != sizeof(*state.artifact) ||
      state.artifact->contract_version !=
          PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      state.artifact->context == nullptr ||
      state.artifact->validate_generation == nullptr ||
      state.artifact->open_development_lease == nullptr ||
      state.artifact->read_lease == nullptr ||
      state.artifact->poll_lease == nullptr ||
      state.artifact->release_lease == nullptr ||
      state.artifact->map_lease == nullptr ||
      state.artifact->release_mapping == nullptr ||
      state.cuda->struct_size != sizeof(*state.cuda) ||
      state.cuda->contract_version != PIH_NVIDIA_CUDA_ABI_VERSION_V1 ||
      state.cuda->context == nullptr ||
      state.cuda->prepare_device == nullptr ||
      state.cuda_memory->struct_size != sizeof(*state.cuda_memory) ||
      state.cuda_memory->contract_version !=
          PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      state.cuda_memory->context == nullptr ||
      state.cuda_memory->allocate_device == nullptr ||
      state.cuda_memory->deallocate_device == nullptr ||
      state.cuda_memory->allocate_pinned_host == nullptr ||
      state.cuda_memory->deallocate_pinned_host == nullptr ||
      state.cuda_memory->copy_h2d == nullptr ||
      (state.host_spill != nullptr &&
       (state.host_spill->struct_size != sizeof(*state.host_spill) ||
        state.host_spill->contract_version !=
            PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        state.host_spill->context == nullptr ||
        state.host_spill->compile_plan == nullptr ||
        state.host_spill->allocate_pinned == nullptr ||
        state.host_spill->deallocate_pinned == nullptr ||
        state.host_spill->copy_h2d_async == nullptr ||
        state.host_spill->record_event == nullptr ||
        state.host_spill->query_event == nullptr ||
        state.host_spill->create_completion_event == nullptr ||
        state.host_spill->destroy_completion_event == nullptr ||
        state.host_spill->allocate_device_slot == nullptr ||
        state.host_spill->deallocate_device_slot == nullptr ||
        state.host_spill->inspect == nullptr)) ||
      state.cuda_resources->struct_size != sizeof(*state.cuda_resources) ||
      state.cuda_resources->contract_version !=
          PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 ||
      state.cuda_resources->context == nullptr ||
      state.cuda_resources->retain_primary_context == nullptr ||
      state.cuda_resources->bind_runtime == nullptr ||
      state.cuda_resources->create_nonblocking_stream == nullptr ||
      state.cuda_resources->create_disable_timing_event == nullptr ||
      state.cuda_resources->destroy_event == nullptr ||
      state.cuda_resources->destroy_stream == nullptr ||
      state.cuda_resources->release_primary_context == nullptr ||
      state.cuda_async->struct_size != sizeof(*state.cuda_async) ||
      state.cuda_async->contract_version !=
          PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      state.cuda_async->context == nullptr ||
      state.cuda_async->activate_context == nullptr ||
      state.cuda_async->validate_pinned_host == nullptr ||
      state.cuda_async->copy_async == nullptr ||
      state.cuda_async->memset_async == nullptr ||
      state.cuda_async->record_event == nullptr ||
      state.cuda_async->query_event == nullptr ||
      state.cuda_async->synchronize_stream == nullptr ||
      state.execution->struct_size != sizeof(*state.execution) ||
      state.execution->contract_version !=
          PIH_EXECUTION_DEFAULT_ABI_VERSION_V1 ||
      state.execution->context == nullptr ||
      state.execution->compile_capacity == nullptr ||
      state.kernels->struct_size != sizeof(*state.kernels) ||
      state.kernels->contract_version !=
          PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      state.kernels->identity == nullptr ||
      state.kernels->launch_rope_table == nullptr ||
      state.kernels->launch_rms_norm == nullptr ||
      state.kernels->launch_fp8_activation_quant == nullptr ||
      state.kernels->launch_fp8_gemm == nullptr ||
      state.kernels->launch_head_rms == nullptr ||
      state.kernels->launch_rotary == nullptr ||
      state.kernels->launch_kv_fp8_simulate == nullptr ||
      state.kernels->launch_grouped_fp8_gemm == nullptr ||
      state.kernels->launch_route_gather == nullptr ||
      state.kernels->launch_fp4_gemm == nullptr ||
      state.kernels->launch_expert_swiglu == nullptr ||
      state.kernels->launch_expert_accumulate == nullptr ||
      state.kernels->launch_embedding == nullptr ||
      state.kernels->launch_hc_head == nullptr ||
      state.kernels->launch_lm_head == nullptr ||
      state.kernels->launch_argmax == nullptr ||
      state.kernels->launch_stochastic_sample == nullptr ||
      state.kernels->launch_compressor_pooling == nullptr ||
      state.kernels->launch_compressor_projection == nullptr ||
      state.kernels->launch_compressor_store == nullptr ||
      state.kernels->launch_indexer_projection == nullptr ||
      state.kernels->launch_index_score == nullptr ||
      state.kernels->launch_sparse_attention == nullptr ||
      state.kernels->launch_router_gemm == nullptr ||
      state.kernels->launch_mhc_pre == nullptr ||
      state.kernels->launch_mhc_post == nullptr ||
      state.kernels->launch_mhc_target_tap == nullptr ||
      state.kernels->launch_shared_swiglu == nullptr ||
      state.kernels->launch_expert_finalize == nullptr ||
      state.kernels->identity->struct_size !=
          sizeof(*state.kernels->identity) ||
      state.kernels->identity->abi_version !=
          PIH_KERNEL_PACK_ABI_VERSION_V1 ||
      state.kernels->identity->pack_id == nullptr ||
      state.kernels->identity->pack_version == nullptr ||
      state.kernels->identity->pack_abi == nullptr ||
      state.kernels->identity->architecture == nullptr ||
      !ExactCString(state.kernels->identity->pack_id,
                    "pih.kernels.deepseek-v4.sm89") ||
      !ExactCString(state.kernels->identity->pack_version, "1.0.0") ||
      !ExactCString(state.kernels->identity->pack_abi,
                    "pih.deepseek-sm89-kernel-pack.v1") ||
      !ExactCString(state.kernels->identity->architecture, "sm89")) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "pp1_capability_contract_invalid");
  }
  const auto advanced = Advance(context, 1);
  if (pih_status_is_ok_v1(&advanced)) transaction.Commit();
  return advanced;
}
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) { return Advance(context, 3); }
pih_status_v1 DrainCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  if (state.engine_cleanup_failed) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_cleanup_failed");
  }
  if (state.engine_instance != nullptr ||
      state.engine_handle_identity != nullptr ||
      state.engine_activation_epoch != 0 ||
      state.engine_generation != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_still_live");
  }
  return Advance(context, 4);
}
pih_status_v1 StopCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  if (state.phase.load() == 3) {
    state.phase.store(6);
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard transaction(state.engine_mutex);
  const auto phase = state.phase.load();
  if (phase != 1 && phase != 2 && phase != 6) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  if (state.engine_cleanup_failed) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_cleanup_failed");
  }
  if (state.engine_instance != nullptr ||
      state.engine_handle_identity != nullptr ||
      state.engine_activation_epoch != 0 ||
      state.engine_generation != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "deepseek_engine_still_live");
  }
  ClearResolvedCapabilities(state);
  state.next_request_id = 1;
  state.next_request_generation = 1;
  state.next_plan_sequence = 1;
  state.next_engine_generation = 1;
  state.engine_instance = nullptr;
  state.engine_handle_identity = nullptr;
  state.engine_generation = 0;
  state.engine_host_spill_enabled = false;
  state.engine_snapshot_pending = false;
  state.engine_cleanup_failed = false;
  state.host = nullptr;
  state.phase.store(7);
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ContainLifecycle(void* context,
                               pih_lifecycle_callback_v1 callback,
                               const char* failure) noexcept {
  if (context == nullptr || callback == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "deepseek_lifecycle_context_invalid");
  }
  try {
    return callback(context);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "deepseek_lifecycle_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Register(void* context) noexcept {
  return ContainLifecycle(context, &RegisterCore,
                          "deepseek_registration_failed");
}

pih_status_v1 Configure(void* context) noexcept {
  return ContainLifecycle(context, &ConfigureCore,
                          "deepseek_configuration_failed");
}

pih_status_v1 Start(void* context) noexcept {
  return ContainLifecycle(context, &StartCore, "deepseek_start_failed");
}

pih_status_v1 Ready(void* context) noexcept {
  return ContainLifecycle(context, &ReadyCore, "deepseek_ready_failed");
}

pih_status_v1 Drain(void* context) noexcept {
  return ContainLifecycle(context, &DrainCore, "deepseek_drain_failed");
}

pih_status_v1 Stop(void* context) noexcept {
  return ContainLifecycle(context, &StopCore, "deepseek_stop_failed");
}

pih_status_v1 Dispose(void* context) noexcept {
  return ContainLifecycle(context, &DisposeCore,
                          "deepseek_dispose_failed");
}

State state;

}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "host_api_invalid");
  }
  if (!pih_plugin_api_accepts_v1(plugin)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "plugin_api_invalid");
  }
  if (state.host != nullptr || state.phase.load() != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "plugin_entry_already_bound");
  }
  state.host = host;
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.model.deepseek-v4-flash";
  plugin->plugin_version = "1.0.0";
  plugin->context = &state;
  plugin->lifecycle.struct_size = sizeof(pih_plugin_lifecycle_v1);
  plugin->lifecycle.abi_version = PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1;
  plugin->lifecycle.register_plugin = &Register;
  plugin->lifecycle.configure = &Configure;
  plugin->lifecycle.start = &Start;
  plugin->lifecycle.ready = &Ready;
  plugin->lifecycle.drain = &Drain;
  plugin->lifecycle.stop = &Stop;
  plugin->lifecycle.dispose = &Dispose;
  return Status(PIH_STATUS_OK_V1);
}
