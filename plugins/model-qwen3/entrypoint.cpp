#include "pih/plugin_sdk/abi.h"
#include "pih/contracts/text_inference_v2.h"
#include "pih/contracts/qwen_kernels_v1.h"
#include "pih/contracts/artifact_snapshot_v1.h"
#include "pih/core/bounded_json.h"
#include "pih/model/nvidia_qwen3_bf16_engine.h"
#include "pih/model/nvidia_qwen3_int4_engine.h"
#include "tokenizer.h"
#include "generation.h"
#include "execution_binding.h"
#include "controller_client.h"
#include "memory_binding.h"
#include "resource_binding.h"
#include "async_binding.h"
#include "pih/contracts/nvidia_cuda_v1.h"
#include "chat_tokens.h"
#include "request_sampling.h"
#include "text_stop.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace {
using pih::JsonValue;
struct ArtifactSnapshot;
pih_status_v1 Status(uint32_t code, std::string_view text = "");
struct State {
  ~State();
  const pih_host_api_v1* host{};
  uint32_t phase{};
  bool failed{};
  uint32_t maximum_context{};
  uint32_t generation_timeout_ms{};
  uint64_t next_response_id{1};
  const pih_qwen_kernels_api_v1* kernels{};
  const pih_execution_default_api_v1* execution{};
  const pih_artifact_snapshot_api_v1* artifact{};
  const pih_nvidia_cuda_api_v1* cuda{};
  const pih_nvidia_cuda_memory_api_v1* memory{};
  const pih_nvidia_cuda_resources_api_v1* resources{};
  const pih_nvidia_cuda_async_api_v1* async{};
  std::mutex mutex;
  std::unique_ptr<pih::NvidiaQwen3Bf16Engine> bf16;
  std::unique_ptr<pih::NvidiaQwen3Int4Engine> int4;
  std::unique_ptr<pih::qwen_plugin::Tokenizer> tokenizer;
  std::unique_ptr<ArtifactSnapshot> artifact_snapshot;
} state;

struct ArtifactSnapshot final {
  const pih_artifact_snapshot_api_v1& api;
  pih_artifact_snapshot_v1 snapshot{sizeof(snapshot), PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1};
  explicit ArtifactSnapshot(const pih_artifact_snapshot_api_v1& provider) : api(provider) {}
  ArtifactSnapshot(const ArtifactSnapshot&) = delete;
  ArtifactSnapshot& operator=(const ArtifactSnapshot&) = delete;
  // The activation retains every failed-release handle. Close releases it only
  // after engine rollback/retirement, and cannot unload the provider on failure.
  pih_status_v1 Open(const std::filesystem::path& root, const char* name,
                     std::string_view digest, uint64_t maximum_bytes) {
    const auto root_string = root.string();
    const auto digest_string = std::string(digest);
    auto status = api.open_snapshot(api.context, root_string.c_str(), name,
                                     digest_string.c_str(), maximum_bytes, &snapshot);
    if (!pih_status_is_valid_v1(&status) || !pih_status_is_ok_v1(&status) ||
        snapshot.struct_size != sizeof(snapshot) ||
        snapshot.abi_version != PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1 ||
        !snapshot.handle || !snapshot.address || !snapshot.bytes ||
        snapshot.bytes > maximum_bytes || snapshot.bytes > SIZE_MAX ||
        snapshot.bytes - 1 > UINTPTR_MAX - snapshot.address)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_artifact_snapshot_invalid");
    return Status(PIH_STATUS_OK_V1);
  }
  std::span<const std::byte> bytes() const {
    return {reinterpret_cast<const std::byte*>(snapshot.address),
            static_cast<std::size_t>(snapshot.bytes)};
  }
  pih_status_v1 Release() noexcept {
    if (snapshot.handle) {
      auto status = api.release_snapshot(api.context, &snapshot);
      if (!pih_status_is_valid_v1(&status) || !pih_status_is_ok_v1(&status))
        return Status(PIH_STATUS_INTERNAL_V1, "qwen_artifact_snapshot_release_failed");
      snapshot = {sizeof(snapshot), PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1};
    }
    return Status(PIH_STATUS_OK_V1);
  }
};
State::~State() = default;

pih_status_v1 Status(uint32_t code, std::string_view text) {
  pih_status_v1 result{}; result.struct_size = sizeof(result);
  result.abi_version = PIH_STATUS_ABI_VERSION_V1; result.code = code;
  std::memcpy(result.message, text.data(), std::min(text.size(), sizeof(result.message) - 1)); return result;
}
template<class F> pih_status_v1 Guard(F f) noexcept {
  try { return f(); }
  catch (const std::bad_alloc&) { return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "allocation_failed"); }
  catch (const std::invalid_argument& e) { return Status(PIH_STATUS_INVALID_ARGUMENT_V1, e.what()); }
  catch (const std::bad_variant_access&) { return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "JSON field type invalid"); }
  catch (...) { return Status(PIH_STATUS_INTERNAL_V1, "qwen_plugin_operation_failed"); }
}
const JsonValue& Field(const JsonValue& object, const char* key) {
  auto* value = object.at(key);
  if (!value) throw std::invalid_argument("required field missing");
  return *value;
}
JsonValue Parse(const char* bytes, uint64_t size, uint64_t limit) {
  if (!bytes || !size || size > limit) throw std::invalid_argument("JSON size invalid");
  auto result = JsonValue::Parse(std::string_view(bytes, size));
  if (!result.ok() || !result->is_object()) throw std::invalid_argument("invalid JSON object");
  return std::move(*result);
}
std::string Quote(std::string_view value) {
  std::string result = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
    else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
    else result += static_cast<char>(c);
  }
  return result + '"';
}
pih_status_v1 Load(void* context, const char* configuration, uint64_t size) noexcept {
  return Guard([&] {
    if (context != &state) return Status(PIH_STATUS_INVALID_ARGUMENT_V1);
    std::lock_guard lock(state.mutex);
    if (state.phase != 4 || state.tokenizer || state.bf16 || state.int4 || state.failed)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
    const auto config = Parse(configuration, size, 16384);
    if (config.object().size() != 6) throw std::invalid_argument("unexpected model configuration fields");
    const auto model = std::filesystem::path(Field(config, "model_directory").string());
    const char* kernel_root = nullptr;
    if (!state.kernels) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "kernel_pack_not_bound");
    const auto root_status = state.kernels->root(&kernel_root);
    if (!pih_status_is_valid_v1(&root_status) || !pih_status_is_ok_v1(&root_status) || !kernel_root)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "kernel_pack_root_invalid");
    const auto cubin = std::filesystem::path(kernel_root);
    const auto precision = Field(config, "precision").string();
    auto expected_artifact_digest = pih::Sha256Digest::ParseHex(
        Field(config, "artifact_sha256").string());
    if (!expected_artifact_digest.ok() || *expected_artifact_digest == pih::Sha256Digest{})
      throw std::invalid_argument("Qwen artifact digest must be a nonzero SHA-256");
    auto expected_config_digest = pih::Sha256Digest::ParseHex(
        Field(config, "config_sha256").string());
    if (!expected_config_digest.ok() || *expected_config_digest == pih::Sha256Digest{})
      throw std::invalid_argument("Qwen config digest must be a nonzero SHA-256");
    const auto maximum = Field(config, "maximum_context_tokens").integer();
    const auto timeout = Field(config, "generation_timeout_ms").integer();
    if (!model.is_absolute() || !cubin.is_absolute() || maximum < 2 || maximum > 40960 ||
        timeout < 1 || timeout > 86400000 ||
        (precision != "bf16" && precision != "int4")) throw std::invalid_argument("invalid Qwen configuration");
    // Admit the device before snapshot/tokenizer allocation. Engine loading
    // binds its NUMA policy before allocating and first-touching pinned arenas.
    state.failed = true;
    if (!state.cuda || !state.memory || !state.resources || !state.async)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_cuda_capabilities_missing");
    const auto prepared = state.cuda->prepare_device(
        state.cuda->context, 0, state.kernels->target_sm / 10, state.kernels->target_sm % 10);
    if (!pih_status_is_valid_v1(&prepared))
      return Status(PIH_STATUS_INTERNAL_V1, "qwen_cuda_prepare_status_invalid");
    if (!pih_status_is_ok_v1(&prepared)) return prepared;
    const auto execution_status = pih::qwen_plugin::CompileExecutionCapacity(state.execution);
    if (!pih_status_is_ok_v1(&execution_status)) return execution_status;
    if (!state.artifact || !state.artifact->open_snapshot || !state.artifact->release_snapshot)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_artifact_provider_missing");
    state.artifact_snapshot = std::make_unique<ArtifactSnapshot>(*state.artifact);
    auto& tokenizer_source = *state.artifact_snapshot;
    const auto tokenizer_opened = tokenizer_source.Open(model, "tokenizer.json",
        pih::qwen_plugin::kTokenizerSha256, 16ULL * 1024 * 1024);
    if (!pih_status_is_ok_v1(&tokenizer_opened)) return tokenizer_opened;
    auto tokenizer = std::make_unique<pih::qwen_plugin::Tokenizer>(
        pih::qwen_plugin::LoadPinnedTokenizer(tokenizer_source.bytes()));
    const auto tokenizer_released = tokenizer_source.Release();
    if (!pih_status_is_ok_v1(&tokenizer_released)) return tokenizer_released;
    state.artifact_snapshot = std::make_unique<ArtifactSnapshot>(*state.artifact);
    auto& config_source = *state.artifact_snapshot;
    const auto config_opened = config_source.Open(model, "config.json",
        expected_config_digest->hex(), 1024 * 1024);
    if (!pih_status_is_ok_v1(&config_opened)) return config_opened;
    const auto config_bytes = config_source.bytes();
    const std::string config_json(reinterpret_cast<const char*>(config_bytes.data()), config_bytes.size());
    const auto config_released = config_source.Release();
    if (!pih_status_is_ok_v1(&config_released)) return config_released;
    state.artifact_snapshot.reset();
    state.artifact_snapshot = std::make_unique<ArtifactSnapshot>(*state.artifact);
    auto& snapshot = *state.artifact_snapshot;
    const auto opened = snapshot.Open(model, precision == "bf16" ? "model.safetensors" : "model.xing-int4",
                                      expected_artifact_digest->hex(), 2ULL * 1024 * 1024 * 1024);
    if (!pih_status_is_ok_v1(&opened)) return opened;
    if (precision == "bf16") {
      auto loaded = pih::NvidiaQwen3Bf16Engine::LoadPinnedSnapshot(*state.memory, *state.resources, *state.async, cubin, 0, snapshot.bytes(), config_json, *expected_artifact_digest, state.kernels->target_sm);
      if (!loaded.ok()) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, loaded.status().message());
      state.bf16 = std::move(*loaded);
    } else {
      auto loaded = pih::NvidiaQwen3Int4Engine::LoadPinnedSnapshot(*state.memory, *state.resources, *state.async, cubin, 0, snapshot.bytes(), config_json, *expected_artifact_digest, state.kernels->target_sm);
      if (!loaded.ok()) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, loaded.status().message());
      state.int4 = std::move(*loaded);
    }
    const auto released = snapshot.Release();
    if (!pih_status_is_ok_v1(&released)) return released;
    state.artifact_snapshot.reset();
    state.tokenizer = std::move(tokenizer); state.maximum_context = static_cast<uint32_t>(maximum);
    state.generation_timeout_ms = static_cast<uint32_t>(timeout);
    state.failed = false;
    return Status(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 Complete(void* context, uint32_t chat, const char* request, uint64_t size,
    char* response, uint64_t capacity, uint64_t* written,
    const pih_text_output_sink_v2* sink) noexcept {
  if (written) *written = 0;
  return Guard([&] {
    if (context != &state || chat > 1 || !response || !written || capacity < 1024 ||
        !sink || sink->struct_size != sizeof(*sink) || sink->contract_version != PIH_TEXT_INFERENCE_ABI_V2 ||
        !sink->context || !sink->start || !sink->write || !sink->cancelled)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1);
    std::unique_lock lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return Status(PIH_STATUS_UNAVAILABLE_V1, "engine_busy");
    if (!state.tokenizer || state.failed || state.phase != 4) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "engine_not_ready");
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(state.generation_timeout_ms);
    auto body = Parse(request, size, 1024 * 1024);
    for (const auto& [key, _] : body.object()) {
      if (key != "model" && key != "max_tokens" && key != "temperature" && key != "top_p" &&
          key != "stream" && key != "n" && key != "seed" && key != "stop" && key != (chat ? "messages" : "prompt"))
        throw std::invalid_argument("unsupported request field");
    }
    if (Field(body, "model").string() != "Qwen/Qwen3-0.6B") throw std::invalid_argument("model identity mismatch");
    const auto* stream_field = body.at("stream");
    if (stream_field && !stream_field->is_boolean()) throw std::invalid_argument("stream must be boolean");
    const bool streaming = stream_field && stream_field->boolean();
    const auto sampling = pih::qwen_plugin::RequestSampling(body);
    pih::qwen_plugin::TextStop stop_filter(body.at("stop"));
    const auto* requested_maximum = body.at("max_tokens");
    const auto maximum = requested_maximum ? requested_maximum->integer() : 128;
    if (maximum < 1 || maximum >= state.maximum_context) throw std::invalid_argument("invalid max_tokens");
    std::vector<int64_t> tokens;
    auto append = [&](std::string_view text) {
      auto part = state.tokenizer->Encode(text); tokens.insert(tokens.end(), part.begin(), part.end());
    };
    if (chat) {
      tokens = pih::qwen_plugin::PlainChatTokens(
          Field(body, "messages"), *state.tokenizer, state.maximum_context - maximum);
    } else append(Field(body, "prompt").string());
    if (tokens.empty() || tokens.size() + maximum > state.maximum_context)
      throw std::invalid_argument("prompt and output exceed context limit");
    if (state.next_response_id == std::numeric_limits<uint64_t>::max())
      return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "response_identity_exhausted");
    const auto created = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    const auto identity = "{\"id\":\"pih-qwen-" + std::to_string(state.next_response_id++) +
        "\",\"created\":" + std::to_string(created) + ",\"model\":\"Qwen/Qwen3-0.6B\",\"object\":";
    const auto chunk = [&](std::string_view text, const char* finish, std::string_view usage = "") {
      const auto json = identity + Quote(chat ? "chat.completion.chunk" : "text_completion") +
          ",\"choices\":[{\"index\":0,\"finish_reason\":" + (finish ? Quote(finish) : "null") +
          (chat ? ",\"delta\":{\"content\":" + Quote(text) + "}" : ",\"text\":" + Quote(text)) +
          "}]" + std::string(usage) + "}";
      return sink->write(sink->context, json.data(), json.size()) == 1;
    };
    if (std::chrono::steady_clock::now() >= deadline)
      return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1, "request_preparation_deadline_elapsed");
    if (sink->cancelled(sink->context) || sink->start(sink->context, streaming ? 1U : 0U) != 1)
      return Status(PIH_STATUS_UNAVAILABLE_V1, "connection_cancelled_before_generation");
    if (streaming && chat) {
      const auto first = identity + Quote("chat.completion.chunk") +
          ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}";
      if (sink->write(sink->context, first.data(), first.size()) != 1)
        return Status(PIH_STATUS_UNAVAILABLE_V1, "connection_cancelled_before_generation");
    }
    std::string pending_utf8, collected_text;
    constexpr size_t kMaximumTextBytes = 8U << 20;
    size_t published_text_bytes = 0;
    bool output_exceeded = false;
    auto publish_text = [&](std::string_view text) {
      if (text.empty()) return true;
      if (text.size() > kMaximumTextBytes - published_text_bytes) {
        output_exceeded = true;
        return false;
      }
      published_text_bytes += text.size();
      if (streaming) return chunk(text, nullptr);
      collected_text.append(text);
      return true;
    };
    const std::function<pih::qwen_plugin::TokenAction(int64_t)> on_token = [&](int64_t token) {
      const auto delta = state.tokenizer->DecodeIncremental({&token, 1}, pending_utf8);
      if (!publish_text(stop_filter.Feed(delta))) return pih::qwen_plugin::TokenAction::kCancel;
      return stop_filter.matched() ? pih::qwen_plugin::TokenAction::kStop
                                   : pih::qwen_plugin::TokenAction::kContinue;
    };
    const std::function<bool()> cancelled = [&] { return sink->cancelled(sink->context) != 0; };
    auto generated = [&] {
      try {
        return state.bf16
            ? pih::qwen_plugin::Generate(*state.bf16, tokens, static_cast<uint32_t>(maximum), sampling, deadline, on_token, cancelled)
            : pih::qwen_plugin::Generate(*state.int4, tokens, static_cast<uint32_t>(maximum), sampling, deadline, on_token, cancelled);
      } catch (...) {
        state.failed = true;
        throw std::runtime_error("native_generation_exception");
      }
    }();
    if (!generated.ok()) { state.failed = true; return Status(PIH_STATUS_INTERNAL_V1, "inference_failed_restart_required"); }
    if (output_exceeded)
      return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "completion_exceeds_text_byte_budget");
    if (generated->callback_failed)
      return Status(PIH_STATUS_INTERNAL_V1, "output_callback_failed_no_live_request");
    if (generated->deadline_elapsed) return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1, "generation_deadline_elapsed");
    if (generated->cancelled) return Status(PIH_STATUS_UNAVAILABLE_V1, "generation_cancelled");
    const auto& ids = generated->tokens;
    const auto tail = state.tokenizer->DecodeIncremental({}, pending_utf8, true);
    if (!publish_text(stop_filter.Feed(tail)) || !publish_text(stop_filter.Finish()))
      return output_exceeded
          ? Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "completion_exceeds_text_byte_budget")
          : Status(PIH_STATUS_UNAVAILABLE_V1, "connection_cancelled_after_generation");
    const auto stopped = generated->finish == pih::ControllerFinishReason::kStop ||
                         generated->stopped_by_callback || stop_filter.matched();
    const auto usage = ",\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) + ",\"completion_tokens\":" +
      std::to_string(ids.size()) + ",\"total_tokens\":" + std::to_string(tokens.size() + ids.size()) + "}";
    if (streaming) {
      if (!chunk("", stopped ? "stop" : "length", usage))
        return Status(PIH_STATUS_UNAVAILABLE_V1, "connection_cancelled_after_generation");
      return Status(PIH_STATUS_OK_V1);
    }
    const auto& text = collected_text;
    std::string result = identity + Quote(chat ? "chat.completion" : "text_completion") +
      ",\"choices\":[{\"index\":0,\"finish_reason\":" + Quote(stopped ? "stop" : "length") +
      (chat ? ",\"message\":{\"role\":\"assistant\",\"content\":" + Quote(text) + "}" : ",\"text\":" + Quote(text)) +
      "}]" + usage + "}";
    if (result.size() > capacity) return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "response_buffer_too_small");
    std::memcpy(response, result.data(), result.size()); *written = result.size(); return Status(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 Close(void* context) noexcept {
  return Guard([&] {
    if (context != &state) return Status(PIH_STATUS_INVALID_ARGUMENT_V1);
    std::unique_lock lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return Status(PIH_STATUS_UNAVAILABLE_V1, "engine_operation_active");
    const auto retire = [&](auto& engine) {
      if (!engine) return Status(PIH_STATUS_OK_V1);
      const auto result = engine->close();
      if (!result.ok()) {
        // Both native engines finish their close gate with this result. It is
        // cached, not a pending operation that another close can advance.
        // Retain ownership and fail-stop instead of retrying for five minutes.
        state.failed = true;
        return Status(PIH_STATUS_INTERNAL_V1, result.message());
      }
      engine.reset();
      return Status(PIH_STATUS_OK_V1);
    };
    auto result = retire(state.bf16);
    if (!pih_status_is_ok_v1(&result)) return result;
    result = retire(state.int4);
    if (!pih_status_is_ok_v1(&result)) return result;
    // Engine destruction/failed-load rollback retires all GPU borrowers first.
    // A failed snapshot release retains its owner for a subsequent close.
    if (state.artifact_snapshot) {
      result = state.artifact_snapshot->Release();
      if (!pih_status_is_ok_v1(&result)) return result;
      state.artifact_snapshot.reset();
    }
    state.tokenizer.reset(); return Status(PIH_STATUS_OK_V1);
  });
}
pih_text_inference_api_v2 api{sizeof(api), PIH_TEXT_INFERENCE_ABI_V2, &state, "Qwen/Qwen3-0.6B", Load, Complete, Close};
pih_status_v1 Advance(void* context, uint32_t expected) noexcept {
  if (context != &state || state.phase != expected) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "lifecycle_order_invalid");
  ++state.phase; return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Register(void* context) noexcept {
  if (context != &state || state.phase != 0 || !state.host) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  pih_capability_v1 capability{sizeof(capability), PIH_CAPABILITY_ABI_VERSION_V1,
      "inference.text.v2", "pih.inference.text.v2", &api, PIH_CAPABILITY_THREADING_SERIALIZED_V1,
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
  const auto result = state.host->register_capability(state.host->context, &capability);
  if (!pih_status_is_valid_v1(&result)) return Status(PIH_STATUS_INTERNAL_V1);
  return pih_status_is_ok_v1(&result) ? Advance(context, 0) : result;
}
pih_status_v1 Configure(void* c) noexcept {
  if (c != &state || state.phase != 1) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  const pih_qwen_kernels_api_v1* selected = nullptr;
  for (auto id : {"pih.kernels.qwen3.sm89", "pih.kernels.qwen3.sm90"}) {
    const void* resolved = nullptr;
    auto status = state.host->resolve_capability(state.host->context, id, "pih.qwen-kernels.v1",
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_ZERO_OR_ONE_V1, &resolved);
    if (!pih_status_is_ok_v1(&status)) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "kernel_resolution_failed");
    if (!resolved) continue;
    const auto* pack = static_cast<const pih_qwen_kernels_api_v1*>(resolved);
    const uint32_t expected_sm = std::string_view(id) == "pih.kernels.qwen3.sm89" ? 89U : 90U;
    if (selected || pack->struct_size != sizeof(*pack) || pack->contract_version != 1 || !pack->root ||
        pack->target_sm != expected_sm) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "select_exactly_one_matching_qwen_kernel_pack");
    selected = pack;
  }
  if (!selected) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_kernel_pack_missing");
  const void* execution = nullptr;
  const auto execution_status = state.host->resolve_capability(
      state.host->context, "execution.default.v1", "pih.execution.default.v1",
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1,
      &execution);
  if (!pih_status_is_valid_v1(&execution_status))
    return Status(PIH_STATUS_INTERNAL_V1, "execution_resolution_status_invalid");
  if (!pih_status_is_ok_v1(&execution_status)) return execution_status;
  const auto* execution_api = static_cast<const pih_execution_default_api_v1*>(execution);
  if (!pih::qwen_plugin::ValidExecutionApi(execution_api))
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_execution_provider_invalid");
  const void* controller = nullptr;
  const auto controller_status = state.host->resolve_capability(
      state.host->context, "execution.controller.v1", "pih.execution.controller.v1",
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1,
      &controller);
  if (!pih_status_is_valid_v1(&controller_status))
    return Status(PIH_STATUS_INTERNAL_V1, "qwen_controller_resolution_status_invalid");
  if (!pih_status_is_ok_v1(&controller_status)) return controller_status;
  const auto* controller_api = static_cast<const pih_execution_controller_api_v1*>(controller);
  const void* artifact = nullptr;
  const auto artifact_status = state.host->resolve_capability(
      state.host->context, "artifact.authenticated-snapshot.v1", "pih.artifact.authenticated-snapshot.v1",
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1,
      &artifact);
  if (!pih_status_is_valid_v1(&artifact_status))
    return Status(PIH_STATUS_INTERNAL_V1, "qwen_artifact_resolution_status_invalid");
  if (!pih_status_is_ok_v1(&artifact_status)) return artifact_status;
  const auto* artifact_api = static_cast<const pih_artifact_snapshot_api_v1*>(artifact);
  if (!artifact_api || artifact_api->struct_size != sizeof(*artifact_api) ||
      artifact_api->contract_version != PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1 ||
      !artifact_api->context || !artifact_api->open_snapshot ||
      !artifact_api->release_snapshot)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_artifact_provider_invalid");
  const void* cuda = nullptr;
  const void* memory = nullptr;
  const void* resources = nullptr;
  const void* async = nullptr;
  struct Binding { const char* id; const char* contract; const void** output; };
  for (const auto& binding : {
      Binding{"device.cuda-runtime.v1", "pih.device.cuda-runtime.v1", &cuda},
      Binding{"device.cuda-memory.v1", "pih.device.cuda-memory.v1", &memory},
      Binding{"device.cuda-resources.v1", "pih.device.cuda-resources.v1", &resources},
      Binding{"device.cuda-async.v1", "pih.device.cuda-async.v1", &async}}) {
    const auto status = state.host->resolve_capability(
        state.host->context, binding.id, binding.contract, PIH_CAPABILITY_SCOPE_PROCESS_V1,
        PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, binding.output);
    if (!pih_status_is_valid_v1(&status))
      return Status(PIH_STATUS_INTERNAL_V1, "qwen_cuda_resolution_status_invalid");
    if (!pih_status_is_ok_v1(&status)) return status;
  }
  const auto* cuda_api = static_cast<const pih_nvidia_cuda_api_v1*>(cuda);
  const auto* memory_api = static_cast<const pih_nvidia_cuda_memory_api_v1*>(memory);
  const auto* resources_api = static_cast<const pih_nvidia_cuda_resources_api_v1*>(resources);
  const auto* async_api = static_cast<const pih_nvidia_cuda_async_api_v1*>(async);
  if (!cuda_api || cuda_api->struct_size != sizeof(*cuda_api) ||
      cuda_api->contract_version != PIH_NVIDIA_CUDA_ABI_VERSION_V1 ||
      !cuda_api->context || !cuda_api->prepare_device ||
      !memory_api || !pih::qwen_plugin::ValidMemoryApi(*memory_api) ||
      !resources_api || !pih::qwen_plugin::ValidResourceApi(*resources_api) ||
      !async_api || !pih::qwen_plugin::ValidAsyncApi(*async_api))
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "qwen_cuda_capabilities_invalid");
  const auto bound_controller = pih::qwen_plugin::ControllerClient::Bind(controller_api);
  if (!bound_controller.ok())
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, bound_controller.message());
  state.cuda = cuda_api;
  state.memory = memory_api;
  state.resources = resources_api;
  state.async = async_api;
  state.execution = execution_api;
  state.artifact = artifact_api;
  state.kernels = selected; return Advance(c, 1);
}
pih_status_v1 Start(void* c) noexcept { return Advance(c, 2); }
pih_status_v1 Ready(void* c) noexcept { return Advance(c, 3); }
pih_status_v1 Drain(void* c) noexcept { return Advance(c, 4); }
pih_status_v1 Stop(void* c) noexcept {
  auto result = Close(c); if (!pih_status_is_ok_v1(&result)) return result;
  if (state.phase == 3) { state.phase = 6; return result; } return Advance(c, 5);
}
pih_status_v1 Dispose(void* c) noexcept {
  if (c != &state || (state.phase != 1 && state.phase != 2 && state.phase != 6) ||
      state.tokenizer || state.bf16 || state.int4 || state.artifact_snapshot)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  state.execution = nullptr; state.artifact = nullptr; state.kernels = nullptr;
  state.cuda = nullptr; state.memory = nullptr;
  state.resources = nullptr;
  state.async = nullptr;
  pih::qwen_plugin::ControllerClient::Unbind();
  state.host = nullptr; state.phase = 7; return Status(PIH_STATUS_OK_V1);
}
}
extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host) || !pih_plugin_api_accepts_v1(plugin)) return Status(PIH_STATUS_INVALID_ARGUMENT_V1);
  if (state.host || state.phase) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  state.host = host; plugin->plugin_id = "pih.model.qwen3"; plugin->plugin_version = "1.0.0";
  plugin->context = &state;
  plugin->lifecycle = {sizeof(pih_plugin_lifecycle_v1), PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1,
    Register, Configure, Start, Ready, Drain, Stop, Dispose};
  return Status(PIH_STATUS_OK_V1);
}

