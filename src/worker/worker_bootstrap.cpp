#include "worker/worker_bootstrap.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "microkernel/activation_runtime.h"
#include "microkernel/capability_binding.h"
#include "microkernel/kernel_pack_loader.h"
#include "microkernel/plugin_loader.h"
#include "microkernel/plugin_registry.h"
#include "pih/contracts/engine_v1.h"
#include "pih/contracts/text_inference_v2.h"
#include "pih/contracts/token_generation_v1.h"
#include "pih/contracts/health_v1.h"
#include "pih/contracts/openai_http_v1.h"
#include "pih/plugin_sdk/abi.h"
#include "pih/plugin_sdk/kernel_pack.h"
#include "worker/deployment_lock.h"

namespace pih::worker {
namespace {

[[noreturn]] void FailStopRetirement(const char* reason) noexcept {
  // A live device operation may still borrow plugin/host storage. Throwing
  // would unwind host contexts; returning from main would run DSO globals'
  // destructors and attempt another unsafe model cleanup. Only the OS may
  // reclaim this process once retirement has become terminal.
  try { std::cerr << "pih_worker_fail_stop: " << reason << '\n' << std::flush; }
  catch (...) { /* Diagnostics must not reintroduce stack unwinding. */ }
  std::_Exit(EXIT_FAILURE);
}

// Declare after host storage, so failed loading/activation cannot unwind that
// storage while a mapped plugin still holds callbacks or asynchronous work.
struct PluginRetirementGuard final {
  bool settled = false;
  ~PluginRetirementGuard() {
    if (!settled) FailStopRetirement("plugin_stack_retirement_unproven");
  }
};

#if defined(__linux__)
bool DescriptorStillOpen(int descriptor) noexcept {
  int flags = -1;
  do {
    flags = ::fcntl(descriptor, F_GETFD);
  } while (flags == -1 && errno == EINTR);
  return flags != -1;
}
#endif

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

bool StatusOk(const pih_status_v1& status) {
  return pih_status_is_ok_v1(&status);
}

std::string StatusMessage(const pih_status_v1& status) {
  if (!pih_status_is_valid_v1(&status)) {
    return "status_abi_invalid";
  }
  const auto* end = std::find(std::begin(status.message),
                              std::end(status.message), '\0');
  if (end == std::begin(status.message)) return "plugin_operation_failed";
  return std::string(std::begin(status.message), end);
}

const char* ThreadingModelName(uint32_t value) {
  switch (value) {
    case PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1:
      return "single_threaded";
    case PIH_CAPABILITY_THREADING_SERIALIZED_V1: return "serialized";
    case PIH_CAPABILITY_THREADING_CONCURRENT_V1: return "concurrent";
    default: throw std::logic_error("capability_threading_model_invalid");
  }
}

const char* CapabilityScopeName(uint32_t value) {
  switch (value) {
    case PIH_CAPABILITY_SCOPE_PROCESS_V1: return "process";
    case PIH_CAPABILITY_SCOPE_ACTIVATION_V1: return "activation";
    case PIH_CAPABILITY_SCOPE_ENGINE_V1: return "engine";
    case PIH_CAPABILITY_SCOPE_REQUEST_V1: return "request";
    default: throw std::logic_error("capability_scope_invalid");
  }
}

const char* CapabilityCardinalityName(uint32_t value) {
  if (value == PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1) {
    return "exactly_one";
  }
  throw std::logic_error("capability_cardinality_invalid");
}

struct HostContext final {
  microkernel::CapabilityBindings* bindings{};
  std::mutex* authority_mutex{};
  std::string provider_id;
  std::atomic_bool registration_enabled{};
  std::atomic_bool resolution_enabled{};
  pih_host_api_v1 api{};
};

struct RegistrationBarrierContext final {
  microkernel::CapabilityBindings* bindings{};
  std::mutex* authority_mutex{};
  std::vector<std::unique_ptr<HostContext>>* hosts{};
  const std::vector<LockedCapability>* locked_capabilities{};
};

bool ValidBoundedCString(const char* value, std::size_t maximum_chars) {
  return value != nullptr && value[0] != '\0' &&
         std::memchr(value, '\0', maximum_chars + 1) != nullptr;
}

std::string BoundedDiagnosticString(const char* value,
                                    std::size_t maximum_bytes) {
  if (value == nullptr) return "invalid";
  const auto* end = static_cast<const char*>(
      std::memchr(value, '\0', maximum_bytes + 1));
  if (end == nullptr) return "truncated";
  std::string escaped;
  escaped.reserve(static_cast<std::size_t>(end - value));
  for (auto cursor = value; cursor != end; ++cursor) {
    const auto byte = static_cast<unsigned char>(*cursor);
    if (byte == '"' || byte == '\\') {
      escaped.push_back('\\');
      escaped.push_back(static_cast<char>(byte));
    } else if (byte >= 0x20 && byte <= 0x7e) {
      escaped.push_back(static_cast<char>(byte));
    } else {
      escaped.push_back('?');
    }
  }
  return escaped;
}

void Log(void* context, uint32_t severity, const char* code,
         const char* message) noexcept {
  try {
    static std::mutex output_mutex;
    const auto* host = static_cast<const HostContext*>(context);
    const auto plugin_id = host == nullptr ? std::string("unknown")
                                           : host->provider_id;
    const auto activation_epoch =
        host == nullptr ? 0 : host->api.activation_epoch;
    const auto safe_code = BoundedDiagnosticString(code, 128);
    const auto safe_message = BoundedDiagnosticString(message, 512);
    std::lock_guard lock(output_mutex);
    std::cerr << "{\"activation_epoch\":" << activation_epoch
              << ",\"code\":\"" << safe_code << "\",\"message\":\""
              << safe_message << "\",\"plugin_id\":\"" << plugin_id
              << "\",\"severity\":" << severity << "}\n";
  } catch (...) {
    // Logging cannot change plugin or activation outcomes.
  }
}

pih_status_v1 RegisterCapability(
    void* context, const pih_capability_v1* capability) noexcept {
  if (context == nullptr || capability == nullptr ||
      capability->struct_size != sizeof(pih_capability_v1) ||
      capability->abi_version != PIH_CAPABILITY_ABI_VERSION_V1 ||
      !ValidBoundedCString(capability->capability_id, 256) ||
      !ValidBoundedCString(capability->contract_id, 256) ||
      capability->api == nullptr ||
      capability->threading_model <
          PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1 ||
      capability->threading_model >
          PIH_CAPABILITY_THREADING_CONCURRENT_V1 ||
      capability->scope < PIH_CAPABILITY_SCOPE_PROCESS_V1 ||
      capability->scope > PIH_CAPABILITY_SCOPE_REQUEST_V1 ||
      capability->cardinality !=
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "capability_invalid");
  }
  auto& host = *static_cast<HostContext*>(context);
  std::lock_guard authority_lock(*host.authority_mutex);
  if (!host.registration_enabled.load(std::memory_order_acquire)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registration_phase_invalid");
  }
  try {
    host.bindings->Bind(std::string(capability->capability_id),
                        host.provider_id,
                        std::string(capability->contract_id), capability->api,
                        capability->threading_model,
                        capability->scope, capability->cardinality);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "capability_registration_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registration_rejected");
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ResolveCapability(void* context, const char* capability_id,
                                const char* contract_id,
                                uint32_t required_scope,
                                uint32_t required_cardinality,
                                const void** api) noexcept {
  if (api != nullptr) *api = nullptr;
  if (context == nullptr ||
      !ValidBoundedCString(capability_id, 256) ||
      !ValidBoundedCString(contract_id, 256) || api == nullptr ||
      required_scope < PIH_CAPABILITY_SCOPE_PROCESS_V1 ||
      required_scope > PIH_CAPABILITY_SCOPE_REQUEST_V1 ||
      (required_cardinality !=
           PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1 &&
       required_cardinality !=
           PIH_CAPABILITY_CARDINALITY_ZERO_OR_ONE_V1)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "capability_query_invalid");
  }
  auto& host = *static_cast<HostContext*>(context);
  std::lock_guard authority_lock(*host.authority_mutex);
  if (!host.resolution_enabled.load(std::memory_order_acquire)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_resolution_phase_invalid");
  }
  try {
    *api = host.bindings->ResolveApi(
        std::string(capability_id), std::string(contract_id), required_scope,
        required_cardinality);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "capability_resolution_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_requirement_unsatisfied");
  }
  return Status(PIH_STATUS_OK_V1);
}

void InitializeHostContext(HostContext& context,
                           microkernel::CapabilityBindings& bindings,
                           std::mutex& authority_mutex) {
  context.bindings = &bindings;
  context.authority_mutex = &authority_mutex;
  context.api.struct_size = sizeof(context.api);
  context.api.abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  context.api.context = &context;
  context.api.log = &Log;
  context.api.register_capability = &RegisterCapability;
  context.api.resolve_capability = &ResolveCapability;
  context.api.activation_epoch = bindings.activation_epoch();
}

void CloseRegistrationAndSealBindings(void* context,
                                      bool registration_complete) {
  auto& barrier = *static_cast<RegistrationBarrierContext*>(context);
  std::lock_guard authority_lock(*barrier.authority_mutex);
  for (auto& host : *barrier.hosts) {
    host->registration_enabled.store(false, std::memory_order_release);
  }
  if (!barrier.bindings->sealed()) barrier.bindings->Seal();
  if (registration_complete) {
    const auto actual = barrier.bindings->Describe();
    if (barrier.locked_capabilities == nullptr ||
        actual.size() != barrier.locked_capabilities->size()) {
      throw std::runtime_error("locked_capability_graph_mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto& observed = actual[index];
      const auto& expected = (*barrier.locked_capabilities)[index];
      if (observed.capability_id != expected.capability_id ||
          observed.provider_id != expected.provider_id ||
          observed.contract_id != expected.contract_id ||
          observed.threading_model != expected.threading_model ||
          observed.scope != expected.scope ||
          observed.cardinality != expected.cardinality) {
        throw std::runtime_error("locked_capability_graph_mismatch");
      }
    }
    for (auto& host : *barrier.hosts) {
      host->resolution_enabled.store(true, std::memory_order_release);
    }
  } else {
    for (auto& host : *barrier.hosts) {
      host->resolution_enabled.store(false, std::memory_order_release);
    }
  }
}

void CloseCapabilityResolution(
    std::vector<std::unique_ptr<HostContext>>& contexts,
    std::mutex& authority_mutex) noexcept {
  std::lock_guard authority_lock(authority_mutex);
  for (auto& context : contexts) {
    context->resolution_enabled.store(false, std::memory_order_release);
  }
}

}  // namespace

struct WorkerPluginStack::Impl {
  enum class State { kEmpty, kStarting, kReady, kShuttingDown, kStopped, kFailed };
  State state = State::kEmpty;
  DevelopmentLock lock;
  std::unique_ptr<microkernel::CapabilityBindings> bindings;
  microkernel::PluginRegistry registry;
  microkernel::PluginLoader loader;
  microkernel::KernelPackLoader kernel_loader;
  std::mutex mutex;
  std::vector<microkernel::KernelPackInstance> packs;
  std::vector<std::unique_ptr<HostContext>> contexts;
  std::vector<microkernel::PluginInstance> plugins;
};
WorkerPluginStack::WorkerPluginStack() : impl_(std::make_unique<Impl>()) {}
WorkerPluginStack::~WorkerPluginStack() {
  // Leaking host storage alone is insufficient: normal process exit still
  // runs provider DSO globals' destructors. As with the service worker, only
  // the OS may reclaim an activation whose retirement was not proven.
  if (impl_ && impl_->state != Impl::State::kEmpty && impl_->state != Impl::State::kStopped)
    FailStopRetirement("rank_plugin_stack_retirement_unproven");
}
void WorkerPluginStack::Start(const std::string& snapshot, const std::string& path,
    const Sha256Digest& expected, std::uint64_t epoch) {
  auto& s = *impl_;
  if (s.state != Impl::State::kEmpty) throw std::logic_error("rank_plugin_stack_single_use");
  if (!epoch || expected == Sha256Digest{} || snapshot.empty() || snapshot.size() > (1U << 20))
    throw std::invalid_argument("rank_plugin_stack_identity_invalid");
  const auto digest = sha256(std::as_bytes(std::span(snapshot.data(), snapshot.size())));
  if (!digest.ok() || *digest != expected) throw std::invalid_argument("rank_plugin_lock_digest_mismatch");
  s.lock = ParseDevelopmentLock(snapshot, path);
  if (s.lock.has_engine) throw std::invalid_argument("rank_plugin_lock_must_be_backend_only");
  for (const auto& plugin : s.lock.plugins)
    if (plugin.entrypoint_sha256_hex.empty()) throw std::invalid_argument("rank_plugin_binary_digest_required");
  for (const auto& pack : s.lock.kernel_packs)
    if (pack.binary_sha256_hex.empty()) throw std::invalid_argument("rank_kernel_pack_digest_required");
  s.bindings = std::make_unique<microkernel::CapabilityBindings>(epoch);
  s.packs.reserve(s.lock.kernel_packs.size()); s.contexts.reserve(s.lock.plugins.size()); s.plugins.reserve(s.lock.plugins.size());
  s.state = Impl::State::kStarting;
  try {
    for (const auto& locked : s.lock.kernel_packs) {
      auto pack = s.kernel_loader.Load(locked.binary, locked.binary_sha256_hex);
      if (pack.id() != locked.pack_id || pack.version() != locked.pack_version ||
          pack.pack_abi() != locked.pack_abi || pack.architecture() != locked.architecture)
        throw std::runtime_error("locked_kernel_pack_identity_mismatch");
      s.packs.push_back(std::move(pack));
    }
    s.kernel_loader.Seal();
    for (const auto& pack : s.packs)
      s.bindings->Bind(pack.id(), pack.id(), pack.pack_abi(), pack.contract_api(),
          PIH_CAPABILITY_THREADING_CONCURRENT_V1, PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1);
    for (const auto& locked : s.lock.plugins) {
      auto host = std::make_unique<HostContext>();
      InitializeHostContext(*host, *s.bindings, s.mutex);
      host->provider_id = locked.plugin_id;
      // Retain the host BEFORE invoking an external plugin entry point.
      s.contexts.push_back(std::move(host));
      auto plugin = s.loader.Load(locked.entrypoint, s.contexts.back()->api, locked.entrypoint_sha256_hex);
      if (plugin.id() != locked.plugin_id || plugin.version() != locked.plugin_version)
        throw std::runtime_error("locked_plugin_identity_mismatch");
      s.registry.Add(plugin.id()); s.plugins.push_back(std::move(plugin));
    }
    s.registry.Seal(); s.loader.Seal();
    for (auto& context : s.contexts) context->registration_enabled.store(true, std::memory_order_release);
    RegistrationBarrierContext barrier{ s.bindings.get(), &s.mutex, &s.contexts, &s.lock.capabilities };
    microkernel::ActivatePluginStack(s.plugins, &CloseRegistrationAndSealBindings, &barrier);
    s.state = Impl::State::kReady;
  } catch (...) {
    s.state = Impl::State::kFailed;
    CloseCapabilityResolution(s.contexts, s.mutex);
    for (auto& context : s.contexts) context->registration_enabled.store(false, std::memory_order_release);
    if (s.bindings->sealed() && !s.bindings->revoked()) s.bindings->Revoke();
    throw;
  }
}
const void* WorkerPluginStack::Resolve(const std::string& contract, std::uint32_t scope, std::uint32_t cardinality) const {
  const auto& s = *impl_;
  if (s.state != Impl::State::kReady) throw std::logic_error("rank_plugin_stack_not_ready");
  const LockedCapability* selected = nullptr;
  for (const auto& capability : s.lock.capabilities) {
    if (capability.contract_id != contract) continue;
    if (selected) throw std::runtime_error("rank_plugin_contract_ambiguous");
    selected = &capability;
  }
  if (!selected) throw std::runtime_error("rank_plugin_contract_missing");
  return s.bindings->ResolveApi(selected->capability_id, contract, scope, cardinality);
}
void WorkerPluginStack::Shutdown() {
  auto& s = *impl_;
  if (s.state == Impl::State::kStopped) return;
  if (s.state != Impl::State::kReady) throw std::logic_error("rank_plugin_shutdown_not_ready");
  s.state = Impl::State::kShuttingDown;
  try {
    CloseCapabilityResolution(s.contexts, s.mutex);
    microkernel::ShutdownPluginStackChecked(s.plugins);
    s.bindings->Revoke();
    s.state = Impl::State::kStopped;
  } catch (...) { s.state = Impl::State::kFailed; throw; }
}

int RunDevelopmentLock(const std::string& lock_path, std::uint16_t serve_port,
    const std::vector<std::uint32_t>& prompt_tokens,
    std::uint32_t maximum_completion_tokens) {
  auto lock = LoadDevelopmentLock(lock_path);
  const bool text_service = lock.has_engine && lock.engine.contract_id == "pih.inference.text.v2";
  // The lock selects a provider, not an arbitrary callback-table layout.
  if (lock.has_engine && !text_service && lock.engine.contract_id != "pih.engine.v1")
    throw std::invalid_argument("worker_engine_contract_unsupported");
  if ((serve_port != 0) != text_service)
    throw std::invalid_argument("serve_requires_native_text_inference_lock");
  if (!prompt_tokens.empty() && (!lock.has_engine || text_service || serve_port ||
      maximum_completion_tokens == 0 || maximum_completion_tokens > 65536 || prompt_tokens.size() > 65536))
    throw std::invalid_argument("token_generation_requires_engine_lock");
  microkernel::CapabilityBindings bindings(
      lock.has_engine ? lock.engine.activation_epoch : 1);
  microkernel::PluginRegistry registry;
  microkernel::PluginLoader loader;
  microkernel::KernelPackLoader kernel_pack_loader;
  std::mutex authority_mutex;
  std::vector<microkernel::KernelPackInstance> kernel_packs;
  std::vector<std::unique_ptr<HostContext>> contexts;
  std::vector<microkernel::PluginInstance> plugins;
  kernel_packs.reserve(lock.kernel_packs.size());
  contexts.reserve(lock.plugins.size());
  plugins.reserve(lock.plugins.size());
  PluginRetirementGuard retirement;
  for (const auto& locked : lock.kernel_packs) {
    auto pack = kernel_pack_loader.Load(locked.binary,
                                        locked.binary_sha256_hex);
    if (pack.id() != locked.pack_id || pack.version() != locked.pack_version ||
        pack.pack_abi() != locked.pack_abi ||
        pack.architecture() != locked.architecture) {
      throw std::runtime_error("locked_kernel_pack_identity_mismatch");
    }
    kernel_packs.push_back(std::move(pack));
  }
  kernel_pack_loader.Seal();
  for (const auto& pack : kernel_packs) {
    bindings.Bind(pack.id(), pack.id(), pack.pack_abi(),
                  pack.contract_api(), PIH_CAPABILITY_THREADING_CONCURRENT_V1,
                  PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                  PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1);
  }
  for (const auto& locked : lock.plugins) {
    auto context = std::unique_ptr<HostContext>(new HostContext{});
    InitializeHostContext(*context, bindings, authority_mutex);
    context->provider_id = locked.plugin_id;
    contexts.push_back(std::move(context));
    auto plugin = loader.Load(locked.entrypoint, contexts.back()->api,
                              locked.entrypoint_sha256_hex);
    if (plugin.id() != locked.plugin_id) {
      throw std::runtime_error("locked_plugin_identity_mismatch");
    }
    if (plugin.version() != locked.plugin_version) {
      throw std::runtime_error("locked_plugin_version_mismatch");
    }
    registry.Add(plugin.id());
    plugins.push_back(std::move(plugin));
  }
  registry.Seal();
  loader.Seal();
  for (auto& context : contexts) {
    context->registration_enabled.store(true, std::memory_order_release);
  }
  RegistrationBarrierContext registration_barrier{
      &bindings, &authority_mutex, &contexts, &lock.capabilities};
  try {
    microkernel::ActivatePluginStack(plugins,
                                     &CloseRegistrationAndSealBindings,
                                     &registration_barrier);
  } catch (...) {
    const auto activation_failure = std::current_exception();
    try {
      if (bindings.sealed() && !bindings.revoked()) bindings.Revoke();
    } catch (...) {
      // Activation rollback already closed every Host registration and
      // resolution gate. Preserve the phase failure if authority revocation
      // itself detects corrupted microkernel state; this process exits.
    }
    std::rethrow_exception(activation_failure);
  }
  if (!prompt_tokens.empty()) {
    const pih_engine_factory_api_v1* factory = nullptr;
    pih_engine_handle_v1 handle{};
    handle.struct_size = sizeof(handle);
    handle.abi_version = PIH_ENGINE_ABI_VERSION_V1;
    std::vector<std::uint32_t> output(maximum_completion_tokens);
    pih_token_generation_result_v1 result{};
    result.struct_size = sizeof(result);
    result.contract_version = PIH_TOKEN_GENERATION_ABI_V1;
    result.tokens = output.data();
    result.token_capacity = maximum_completion_tokens;
    std::exception_ptr failure;
    try {
      factory = static_cast<const pih_engine_factory_api_v1*>(bindings.ResolveApi(
          lock.engine.capability_id, lock.engine.contract_id,
          PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (!factory || factory->struct_size != sizeof(*factory) ||
          factory->contract_version != PIH_ENGINE_ABI_VERSION_V1 ||
          factory->handle_generation_semantics != PIH_ENGINE_HANDLE_GENERATION_BOUND_READ_ONLY_V1 ||
          !factory->context || !factory->create || !factory->destroy) {
        factory = nullptr;
        throw std::runtime_error("token_engine_factory_invalid");
      }
      const auto* generator = static_cast<const pih_token_generation_api_v1*>(bindings.ResolveApi(
          "inference.tokens.v1", "pih.inference.tokens.v1",
          PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (!generator || generator->struct_size != sizeof(*generator) ||
          generator->contract_version != PIH_TOKEN_GENERATION_ABI_V1 ||
          !generator->context || !generator->generate)
        throw std::runtime_error("token_generation_contract_invalid");
      CloseCapabilityResolution(contexts, authority_mutex);
      const pih_engine_create_request_v1 creation{sizeof(creation), PIH_ENGINE_ABI_VERSION_V1,
          lock.engine.activation_epoch, lock.deployment_root.c_str(),
          lock.engine.configuration_json.data(), lock.engine.configuration_json.size()};
      auto status = factory->create(factory->context, &creation, &handle);
      if (!StatusOk(status)) throw std::runtime_error(StatusMessage(status));
      if (!pih_engine_handle_is_live_v1(&handle) || handle.activation_epoch != lock.engine.activation_epoch)
        throw std::runtime_error("token_engine_handle_invalid");
      pih_token_generation_request_v1 request{};
      request.struct_size = sizeof(request);
      request.contract_version = PIH_TOKEN_GENERATION_ABI_V1;
      request.prompt_tokens = prompt_tokens.data();
      request.prompt_token_count = static_cast<std::uint32_t>(prompt_tokens.size());
      request.maximum_completion_tokens = maximum_completion_tokens;
      request.top_p = 1;
      request.seed = 7;
      request.deadline_monotonic_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              (std::chrono::steady_clock::now() + std::chrono::minutes(30)).time_since_epoch()).count());
      status = generator->generate(generator->context, &handle, &request, &result);
      if (!StatusOk(status)) throw std::runtime_error(StatusMessage(status));
      if (result.struct_size != sizeof(result) || result.contract_version != PIH_TOKEN_GENERATION_ABI_V1 ||
          result.tokens != output.data() || result.token_capacity != maximum_completion_tokens ||
          result.token_count == 0 || result.token_count > output.size() ||
          result.prompt_token_count != prompt_tokens.size() || result.completion_token_count != result.token_count ||
          (result.finish_reason != PIH_TOKEN_FINISH_STOP_V1 && result.finish_reason != PIH_TOKEN_FINISH_LENGTH_V1))
        throw std::runtime_error("token_generation_result_invalid");
    } catch (...) { failure = std::current_exception(); }
    if (factory && handle.instance) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
      for (;;) {
        const auto status = factory->destroy(factory->context, &handle);
        if (StatusOk(status)) {
          if (!pih_engine_handle_is_empty_v1(&handle)) throw std::runtime_error("token_engine_cleanup_handle_invalid");
          break;
        }
        if (!pih_status_is_valid_v1(&status) || status.code != PIH_STATUS_UNAVAILABLE_V1 ||
            !handle.instance || std::chrono::steady_clock::now() >= deadline)
          FailStopRetirement("token_engine_cleanup_failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    CloseCapabilityResolution(contexts, authority_mutex);
    microkernel::ShutdownPluginStackChecked(plugins);
    retirement.settled = true;
    bindings.Revoke();
    if (failure) std::rethrow_exception(failure);
    std::cout << "{\"schema\":\"pih.token-generation-result.v1\",\"token_ids\":[";
    for (std::uint32_t index = 0; index < result.token_count; ++index) {
      if (index) std::cout << ',';
      std::cout << output[index];
    }
    std::cout << "],\"finish_reason\":\"" << (result.finish_reason == PIH_TOKEN_FINISH_STOP_V1 ? "stop" : "length")
              << "\",\"prompt_tokens\":" << result.prompt_token_count
              << ",\"completion_tokens\":" << result.completion_token_count << "}\n";
    return 0;
  }
  if (text_service) {
    const pih_text_inference_api_v2* engine = nullptr;
    std::exception_ptr failure;
    try {
      const auto* candidate = static_cast<const pih_text_inference_api_v2*>(bindings.ResolveApi(
          lock.engine.capability_id, lock.engine.contract_id, PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (!candidate || candidate->struct_size != sizeof(*candidate) ||
          candidate->contract_version != PIH_TEXT_INFERENCE_ABI_V2 ||
          !candidate->context || !candidate->load || !candidate->complete || !candidate->close) {
        throw std::runtime_error("native_text_engine_contract_invalid");
      }
      // Only a validated callback table may enter the cleanup path. Surface
      // resolution can throw, including before any load has been attempted.
      engine = candidate;
      const auto* service = static_cast<const pih_text_service_api_v2*>(bindings.ResolveApi(
          "surface.text-http.v2", "pih.surface.text-http.v2", PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (!service || service->struct_size != sizeof(*service) || service->contract_version != PIH_TEXT_INFERENCE_ABI_V2 ||
          !service->context || !service->serve) throw std::runtime_error("native_text_surface_contract_invalid");
      CloseCapabilityResolution(contexts, authority_mutex);
      auto status = engine->load(engine->context, lock.engine.configuration_json.data(), lock.engine.configuration_json.size());
      if (!StatusOk(status)) throw std::runtime_error(StatusMessage(status));
      status = service->serve(service->context, serve_port);
      if (!StatusOk(status)) throw std::runtime_error(StatusMessage(status));
    } catch (...) { failure = std::current_exception(); }
    // Covers resolution/validation failures as well as load/serve failures;
    // cleanup must not gain access to new capabilities while retiring state.
    CloseCapabilityResolution(contexts, authority_mutex);
    if (engine) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
      for (;;) {
        const auto status = engine->close(engine->context);
        if (StatusOk(status)) break;
        if (!pih_status_is_valid_v1(&status) || status.code != PIH_STATUS_UNAVAILABLE_V1 ||
            std::chrono::steady_clock::now() >= deadline)
          FailStopRetirement("native_text_engine_cleanup_failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    CloseCapabilityResolution(contexts, authority_mutex);
    microkernel::ShutdownPluginStackChecked(plugins);
    retirement.settled = true;
    bindings.Revoke();
    if (failure) std::rethrow_exception(failure);
    return 0;
  }
  const pih_health_api_v1* active_health = nullptr;
  const pih_openai_http_api_v1* active_surface = nullptr;
  const pih_engine_factory_api_v1* active_factory = nullptr;
  pih_engine_handle_v1 active_engine{};
  active_engine.struct_size = sizeof(active_engine);
  active_engine.abi_version = PIH_ENGINE_ABI_VERSION_V1;
  bool engine_bound = false;
  bool readiness_published = false;
  bool engine_cleanup_complete = false;
  bool engine_cleanup_terminal = false;
  pih_status_v1 engine_cleanup_terminal_status = Status(PIH_STATUS_OK_V1);
  bool engine_cleanup_deadline_started = false;
  std::chrono::steady_clock::time_point engine_cleanup_deadline{};
  const auto cleanup_engine = [&]() -> pih_status_v1 {
    if (engine_cleanup_complete) return Status(PIH_STATUS_OK_V1);
    if (engine_cleanup_terminal) return engine_cleanup_terminal_status;
    pih_status_v1 first_error = Status(PIH_STATUS_OK_V1);
    bool engine_may_be_destroyed = true;
    if (readiness_published && active_health != nullptr) {
      const auto revoke =
          active_health->publish_readiness(
              active_health->context, active_engine.generation, 0);
      if (!StatusOk(revoke)) first_error = revoke;
      if (StatusOk(revoke)) {
        readiness_published = false;
      } else {
        engine_may_be_destroyed = false;
      }
    }
    if (engine_bound && active_surface != nullptr) {
      const auto unbind =
          active_surface->unbind_engine(active_surface->context);
      if (StatusOk(first_error) && !StatusOk(unbind)) first_error = unbind;
      if (StatusOk(unbind)) {
        engine_bound = false;
      } else {
        engine_may_be_destroyed = false;
      }
    }
    if (engine_may_be_destroyed && active_engine.instance != nullptr &&
        active_factory != nullptr) {
      const auto destroy =
          active_factory->destroy(active_factory->context, &active_engine);
      if (StatusOk(first_error) && !StatusOk(destroy)) first_error = destroy;
      if (!StatusOk(destroy) &&
          pih_engine_handle_is_empty_v1(&active_engine)) {
        engine_cleanup_terminal_status = Status(
            PIH_STATUS_INTERNAL_V1,
            "engine_destroy_failed_after_clearing_handle");
        engine_cleanup_terminal = true;
        first_error = engine_cleanup_terminal_status;
      } else if (!StatusOk(destroy) &&
                 (!pih_status_is_valid_v1(&destroy) ||
                  destroy.code != PIH_STATUS_UNAVAILABLE_V1)) {
        engine_cleanup_terminal_status = pih_status_is_valid_v1(&destroy)
            ? destroy
            : Status(PIH_STATUS_INTERNAL_V1,
                     "engine_destroy_status_invalid");
        engine_cleanup_terminal = true;
        first_error = engine_cleanup_terminal_status;
      } else if (StatusOk(destroy) &&
                 !pih_engine_handle_is_empty_v1(&active_engine)) {
        engine_cleanup_terminal_status = Status(
            PIH_STATUS_INTERNAL_V1, "engine_destroy_did_not_clear_handle");
        engine_cleanup_terminal = true;
        first_error = engine_cleanup_terminal_status;
      }
    }
    if (StatusOk(first_error) && !readiness_published && !engine_bound &&
        pih_engine_handle_is_empty_v1(&active_engine)) {
      engine_cleanup_complete = true;
    }
    if (!engine_cleanup_terminal && !StatusOk(first_error) &&
        (!pih_status_is_valid_v1(&first_error) ||
         first_error.code != PIH_STATUS_UNAVAILABLE_V1)) {
      engine_cleanup_terminal_status = pih_status_is_valid_v1(&first_error)
          ? first_error
          : Status(PIH_STATUS_INTERNAL_V1,
                   "engine_cleanup_status_invalid");
      engine_cleanup_terminal = true;
      first_error = engine_cleanup_terminal_status;
    }
    return first_error;
  };
  const auto cleanup_engine_until_settled = [&]() -> pih_status_v1 {
    constexpr auto kEngineCleanupTimeout = std::chrono::minutes(5);
    if (!engine_cleanup_deadline_started) {
      engine_cleanup_deadline = std::chrono::steady_clock::now() +
                                kEngineCleanupTimeout;
      engine_cleanup_deadline_started = true;
    }
    for (;;) {
      const auto status = cleanup_engine();
      if (StatusOk(status) || !pih_status_is_valid_v1(&status) ||
          status.code != PIH_STATUS_UNAVAILABLE_V1) {
        return status;
      }
      if (std::chrono::steady_clock::now() >= engine_cleanup_deadline) {
        return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                      "engine_cleanup_deadline_elapsed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  try {
    if (lock.has_engine) {
      const auto* health = static_cast<const pih_health_api_v1*>(
          bindings.ResolveApi(
              "surface.health.v1", "pih.surface.health.v1",
              PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
              PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (health == nullptr || health->struct_size != sizeof(*health) ||
          health->contract_version != PIH_HEALTH_ABI_VERSION_V1 ||
          health->readiness_semantics !=
              PIH_HEALTH_READINESS_ENGINE_GENERATION_V1 ||
          health->accepted_socket_ownership !=
              PIH_HEALTH_ACCEPTED_SOCKET_CALLER_OWNED_V1 ||
          health->context == nullptr ||
          health->snapshot == nullptr ||
          health->bind_activation == nullptr ||
          health->publish_capabilities == nullptr ||
          health->publish_readiness == nullptr ||
          health->execute_http_exchange == nullptr ||
          health->execute_http_connection == nullptr) {
        throw std::runtime_error("health_surface_invalid");
      }
      active_health = health;
      const auto health_bind_status =
          health->bind_activation(health->context,
                                  lock.engine.activation_epoch);
      if (!StatusOk(health_bind_status)) {
        throw std::runtime_error(StatusMessage(health_bind_status));
      }
      std::string capability_projection = "[";
      const auto capability_descriptions = bindings.Describe();
      for (std::size_t index = 0; index < capability_descriptions.size();
           ++index) {
        const auto& description = capability_descriptions[index];
        if (index != 0) capability_projection += ',';
        capability_projection += "{\"capability_id\":\"";
        capability_projection += description.capability_id;
        capability_projection += "\",\"cardinality\":\"";
        capability_projection +=
            CapabilityCardinalityName(description.cardinality);
        capability_projection += "\",\"contract_id\":\"";
        capability_projection += description.contract_id;
        capability_projection += "\",\"provider_id\":\"";
        capability_projection += description.provider_id;
        capability_projection += "\",\"scope\":\"";
        capability_projection += CapabilityScopeName(description.scope);
        capability_projection += "\",\"threading_model\":\"";
        capability_projection +=
            ThreadingModelName(description.threading_model);
        capability_projection += "\"}";
      }
      capability_projection += ']';
      const auto capability_status = health->publish_capabilities(
          health->context, capability_projection.data(),
          capability_projection.size());
      if (!StatusOk(capability_status)) {
        throw std::runtime_error(StatusMessage(capability_status));
      }
      const auto health_http_status = [&](std::string_view request,
                                          std::string_view expected_prefix,
                                          std::string_view expected_body) {
        char response[8192]{};
        bool response_complete = false;
#if defined(__linux__)
        int sockets[2]{-1, -1};
        if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
          return false;
        }
        std::size_t sent = 0;
        while (sent < request.size()) {
          const auto result = ::send(sockets[0], request.data() + sent,
                                     request.size() - sent, MSG_NOSIGNAL);
          if (result < 0 && errno == EINTR) continue;
          if (result <= 0) break;
          sent += static_cast<std::size_t>(result);
        }
        const auto health_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        const auto health_deadline_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                health_deadline.time_since_epoch()).count();
        pih_status_v1 status = sent == request.size()
            ? health->execute_http_connection(
                  health->context, sockets[1],
                  static_cast<uint64_t>(health_deadline_ns))
            : Status(PIH_STATUS_INTERNAL_V1,
                     "health_http_socket_request_failed");
        if (!DescriptorStillOpen(sockets[1])) {
          status = Status(PIH_STATUS_INTERNAL_V1,
                          "health_http_closed_caller_socket");
          sockets[1] = -1;
        }
        std::size_t response_size = 0;
        if (StatusOk(status)) {
          for (;;) {
            if (response_size == sizeof(response)) break;
            const auto received = ::recv(
                sockets[0], response + response_size,
                sizeof(response) - response_size, 0);
            if (received < 0 && errno == EINTR) continue;
            if (received < 0) break;
            if (received == 0) {
              response_complete = true;
              break;
            }
            response_size += static_cast<std::size_t>(received);
          }
        }
        ::close(sockets[0]);
        if (sockets[1] >= 0) ::close(sockets[1]);
        const std::string_view wire(response, response_size);
#else
        pih_health_http_exchange_v1 exchange{
            sizeof(exchange), PIH_HEALTH_ABI_VERSION_V1, request.data(),
            request.size(), response, sizeof(response), 0};
        const auto status =
            health->execute_http_exchange(health->context, &exchange);
        const auto response_size =
            exchange.struct_size == sizeof(exchange) &&
                    exchange.abi_version == PIH_HEALTH_ABI_VERSION_V1 &&
                    exchange.response_size <= exchange.response_capacity
                ? exchange.response_size
                : 0;
        response_complete = response_size != 0;
        const std::string_view wire(response, response_size);
#endif
        return StatusOk(status) && response_complete && !wire.empty() &&
               wire.starts_with(expected_prefix) &&
               wire.find(expected_body) != std::string_view::npos;
      };
      constexpr std::string_view livez_request =
          "GET /livez HTTP/1.1\r\nHost: local\r\n\r\n";
      constexpr std::string_view readyz_request =
          "GET /readyz HTTP/1.1\r\nHost: local\r\n\r\n";
      constexpr std::string_view capabilities_request =
          "GET /pih/v1/capabilities HTTP/1.1\r\nHost: local\r\n\r\n";
      constexpr std::string_view operational_status_request =
          "GET /pih/v1/status HTTP/1.1\r\nHost: local\r\n\r\n";
      pih_health_snapshot_v1 health_snapshot{};
      health_snapshot.struct_size = sizeof(health_snapshot);
      health_snapshot.abi_version = PIH_HEALTH_ABI_VERSION_V1;
      const auto health_status =
          health->snapshot(health->context, &health_snapshot);
      if (!StatusOk(health_status) ||
          health_snapshot.struct_size != sizeof(health_snapshot) ||
          health_snapshot.abi_version != PIH_HEALTH_ABI_VERSION_V1 ||
          health_snapshot.ready != 0 || health_snapshot.live != 1 ||
          health_snapshot.reason != PIH_HEALTH_REASON_ENGINE_STARTING_V1 ||
          health_snapshot.activation_epoch != lock.engine.activation_epoch ||
          health_snapshot.generation != 0 ||
          health_snapshot.revision != 1 ||
          !health_http_status(livez_request, "HTTP/1.1 200 OK\r\n",
                              R"("live":true})") ||
          !health_http_status(
              readyz_request,
              "HTTP/1.1 503 Service Unavailable\r\n",
              R"("reason":"engine_starting"})") ||
          !health_http_status(
              capabilities_request, "HTTP/1.1 200 OK\r\n",
              R"("capability_id":"engine.primary.v1")") ||
          !health_http_status(
              operational_status_request, "HTTP/1.1 200 OK\r\n",
              R"("ready":false,"reason":"engine_starting")")) {
        throw std::runtime_error("health_surface_premature_ready");
      }
      const auto* factory = static_cast<const pih_engine_factory_api_v1*>(
          bindings.ResolveApi(lock.engine.capability_id,
                              lock.engine.contract_id,
                              PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                              PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (factory == nullptr || factory->struct_size != sizeof(*factory) ||
          factory->contract_version != PIH_ENGINE_ABI_VERSION_V1 ||
          factory->handle_generation_semantics !=
              PIH_ENGINE_HANDLE_GENERATION_BOUND_READ_ONLY_V1 ||
          factory->context == nullptr ||
          factory->create == nullptr ||
          factory->admit_request == nullptr ||
          factory->execute_smoke_request == nullptr ||
          factory->destroy == nullptr) {
        throw std::runtime_error("engine_factory_invalid");
      }
      active_factory = factory;
      const auto* surface = static_cast<const pih_openai_http_api_v1*>(
          bindings.ResolveApi("surface.openai-http.v1",
                              "pih.surface.openai-http.v1",
                              PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                              PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
      if (surface == nullptr || surface->struct_size != sizeof(*surface) ||
          surface->contract_version != PIH_OPENAI_HTTP_ABI_VERSION_V1 ||
          surface->accepted_socket_ownership !=
              PIH_OPENAI_HTTP_ACCEPTED_SOCKET_CALLER_OWNED_V1 ||
          surface->context == nullptr ||
          surface->bind_engine == nullptr ||
          surface->unbind_engine == nullptr ||
          surface->execute_smoke_exchange == nullptr ||
          surface->execute_smoke_connection == nullptr) {
        throw std::runtime_error("openai_http_surface_invalid");
      }
      active_surface = surface;
      pih_engine_create_request_v1 request{};
      request.struct_size = sizeof(request);
      request.abi_version = PIH_ENGINE_ABI_VERSION_V1;
      request.activation_epoch = lock.engine.activation_epoch;
      request.deployment_root = lock.deployment_root.c_str();
      request.configuration_json = lock.engine.configuration_json.c_str();
      request.configuration_size = lock.engine.configuration_json.size();
      auto& engine = active_engine;
      auto status = factory->create(factory->context, &request, &engine);
      if (!StatusOk(status)) {
        const auto create_message = StatusMessage(status);
        if (engine.instance != nullptr) {
          const auto cleanup = cleanup_engine_until_settled();
          if (!StatusOk(cleanup)) {
            throw std::runtime_error(StatusMessage(cleanup));
          }
        }
        throw std::runtime_error(create_message);
      }
      if (!pih_engine_handle_is_live_v1(&engine) ||
          engine.activation_epoch != request.activation_epoch) {
        if (engine.instance != nullptr) {
          const auto cleanup = cleanup_engine_until_settled();
          if (!StatusOk(cleanup)) {
            throw std::runtime_error(StatusMessage(cleanup));
          }
        }
        throw std::runtime_error("engine_factory_returned_invalid_handle");
      }
      const auto engine_generation = engine.generation;
      status = factory->admit_request(factory->context, &engine);
      if (!StatusOk(status)) {
        const auto admission_message = StatusMessage(status);
        const auto cleanup = cleanup_engine_until_settled();
        if (!StatusOk(cleanup)) {
          throw std::runtime_error(StatusMessage(cleanup));
        }
        throw std::runtime_error(admission_message);
      }
      CloseCapabilityResolution(contexts, authority_mutex);
      status = surface->bind_engine(surface->context, &engine);
      if (!StatusOk(status)) {
        const auto bind_message = StatusMessage(status);
        const auto cleanup = cleanup_engine_until_settled();
        if (!StatusOk(cleanup)) {
          throw std::runtime_error(StatusMessage(cleanup));
        }
        throw std::runtime_error(bind_message);
      }
      engine_bound = true;
      const std::string smoke_request =
          "POST /v1/completions HTTP/1.1\r\n"
          "Host: local\r\n"
          "Content-Type: application/json\r\n"
          "Content-Length: " +
          std::to_string(lock.engine.smoke_http_body_json.size()) +
          "\r\n\r\n" + lock.engine.smoke_http_body_json;
      char smoke_response[4096]{};
      pih_openai_http_exchange_v1 exchange{
          sizeof(exchange), PIH_OPENAI_HTTP_ABI_VERSION_V1,
          smoke_request.data(), smoke_request.size(), smoke_response,
          sizeof(smoke_response), 0};
#if defined(__linux__)
      int sockets[2]{-1, -1};
      if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
        status = Status(PIH_STATUS_INTERNAL_V1,
                        "openai_http_socket_pair_failed");
      } else {
        bool peer_closed = false;
        std::size_t sent = 0;
        while (sent < smoke_request.size()) {
          const auto result = ::send(sockets[0], smoke_request.data() + sent,
                                     smoke_request.size() - sent,
                                     MSG_NOSIGNAL);
          if (result < 0 && errno == EINTR) continue;
          if (result <= 0) {
            status = Status(PIH_STATUS_INTERNAL_V1,
                            "openai_http_socket_request_failed");
            break;
          }
          sent += static_cast<std::size_t>(result);
        }
        if (StatusOk(status)) {
          const auto connection_deadline =
              std::chrono::steady_clock::now() + std::chrono::minutes(6);
          const auto connection_deadline_ns =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  connection_deadline.time_since_epoch()).count();
          status = surface->execute_smoke_connection(surface->context,
              sockets[1], static_cast<uint64_t>(connection_deadline_ns));
          if (!DescriptorStillOpen(sockets[1])) {
            status = Status(PIH_STATUS_INTERNAL_V1,
                            "openai_http_closed_caller_socket");
            sockets[1] = -1;
          }
        }
        if (StatusOk(status)) {
          while (exchange.response_size < exchange.response_capacity) {
            const auto received = ::recv(
                sockets[0], smoke_response + exchange.response_size,
                exchange.response_capacity - exchange.response_size, 0);
            if (received < 0 && errno == EINTR) continue;
            if (received < 0) {
              status = Status(PIH_STATUS_INTERNAL_V1,
                              "openai_http_socket_response_failed");
            } else if (received == 0) {
              peer_closed = true;
              break;
            } else {
              exchange.response_size += static_cast<std::size_t>(received);
            }
            if (!StatusOk(status)) break;
          }
          if (StatusOk(status) && exchange.response_size == 0) {
            status = Status(PIH_STATUS_INTERNAL_V1,
                            "openai_http_socket_response_empty");
          } else if (StatusOk(status) && !peer_closed) {
            status = Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                            "openai_http_socket_response_capacity_exceeded");
          }
        }
        ::close(sockets[0]);
        if (sockets[1] >= 0) ::close(sockets[1]);
      }
#else
      status = surface->execute_smoke_exchange(surface->context, &exchange);
#endif
      if (StatusOk(status) &&
          (exchange.struct_size != sizeof(exchange) ||
           exchange.abi_version != PIH_OPENAI_HTTP_ABI_VERSION_V1 ||
           exchange.response_size == 0 ||
           exchange.response_size > exchange.response_capacity ||
           std::string_view(smoke_response, exchange.response_size)
                   .find("HTTP/1.1 200 OK\r\n") != 0 ||
           std::string_view(smoke_response, exchange.response_size)
                   .find(R"("finish_reason":"length")") ==
               std::string_view::npos ||
           std::string_view(smoke_response, exchange.response_size)
                   .find(R"("completion_tokens":1)") ==
               std::string_view::npos ||
           std::string_view(smoke_response, exchange.response_size)
                   .find(R"("logprobs":{"token_id":)") ==
               std::string_view::npos ||
           std::string_view(smoke_response, exchange.response_size)
                   .find(R"("top_logprobs":[{)") ==
               std::string_view::npos)) {
        status = Status(PIH_STATUS_INTERNAL_V1,
                        "openai_http_exchange_response_invalid");
      }
      const auto smoke_ok = StatusOk(status);
      const auto smoke_message = StatusMessage(status);
      if (smoke_ok) {
        auto readiness_status =
            health->publish_readiness(health->context, engine_generation, 1);
        readiness_published = StatusOk(readiness_status);
        if (StatusOk(readiness_status)) {
          health_snapshot.ready = 0;
          readiness_status =
              health->snapshot(health->context, &health_snapshot);
        }
        if (!StatusOk(readiness_status) ||
            health_snapshot.struct_size != sizeof(health_snapshot) ||
            health_snapshot.abi_version != PIH_HEALTH_ABI_VERSION_V1 ||
            health_snapshot.ready != 1 || health_snapshot.live != 1 ||
            health_snapshot.reason != PIH_HEALTH_REASON_READY_V1 ||
            health_snapshot.activation_epoch != lock.engine.activation_epoch ||
            health_snapshot.generation != engine_generation ||
            health_snapshot.revision != 2 ||
            !health_http_status(livez_request, "HTTP/1.1 200 OK\r\n",
                                R"("live":true})") ||
            !health_http_status(readyz_request, "HTTP/1.1 200 OK\r\n",
                                R"("reason":"ready"})") ||
            !health_http_status(
                operational_status_request, "HTTP/1.1 200 OK\r\n",
                                R"("ready":true,"reason":"ready")")) {
          const auto cleanup = cleanup_engine_until_settled();
          if (!StatusOk(cleanup)) {
            throw std::runtime_error(StatusMessage(cleanup));
          }
          throw std::runtime_error("engine_readiness_publish_failed");
        }
        std::cout << "{\"plugin_count\":" << plugins.size()
                  << ",\"state\":\"ready\"}\n";
      }
      const auto cleanup_status = cleanup_engine_until_settled();
      if (!smoke_ok) {
        throw std::runtime_error(smoke_message);
      }
      if (!StatusOk(cleanup_status)) {
        throw std::runtime_error(StatusMessage(cleanup_status));
      }
      health_snapshot = {};
      health_snapshot.struct_size = sizeof(health_snapshot);
      health_snapshot.abi_version = PIH_HEALTH_ABI_VERSION_V1;
      const auto drained_health_status =
          health->snapshot(health->context, &health_snapshot);
      if (!StatusOk(drained_health_status) ||
          health_snapshot.struct_size != sizeof(health_snapshot) ||
          health_snapshot.abi_version != PIH_HEALTH_ABI_VERSION_V1 ||
          health_snapshot.ready != 0 ||
          health_snapshot.live != 1 ||
          health_snapshot.reason != PIH_HEALTH_REASON_DRAINING_V1 ||
          health_snapshot.activation_epoch != lock.engine.activation_epoch ||
          health_snapshot.generation != engine_generation ||
          health_snapshot.revision != 3 ||
          !health_http_status(livez_request, "HTTP/1.1 200 OK\r\n",
                              R"("live":true})") ||
          !health_http_status(
              readyz_request,
              "HTTP/1.1 503 Service Unavailable\r\n",
              R"("reason":"draining"})") ||
          !health_http_status(
              operational_status_request, "HTTP/1.1 200 OK\r\n",
              R"("ready":false,"reason":"draining")")) {
        throw std::runtime_error("health_readiness_revoke_failed");
      }
      std::cout << "{\"engine_capability\":\""
                << lock.engine.capability_id
                << "\",\"state\":\"locked-request-ok\"}\n";
    } else {
      CloseCapabilityResolution(contexts, authority_mutex);
      std::cout << "{\"plugin_count\":" << plugins.size()
                << ",\"state\":\"ready\"}\n";
    }
  } catch (...) {
    const auto operation_failure = std::current_exception();
    (void)cleanup_engine_until_settled();
    CloseCapabilityResolution(contexts, authority_mutex);
    try {
      microkernel::ShutdownPluginStackChecked(plugins);
      retirement.settled = true;
    } catch (...) {
      // The process exits after the original operation failure. A checked
      // shutdown failure must stop in-process teardown rather than expose a
      // later provider to a still-live consumer.
    }
    try {
      if (bindings.sealed() && !bindings.revoked()) bindings.Revoke();
    } catch (...) {
      // Preserve the original engine/surface failure. Host resolution is
      // already closed and this fail-stop worker cannot start another epoch.
    }
    std::rethrow_exception(operation_failure);
  }
  CloseCapabilityResolution(contexts, authority_mutex);
  microkernel::ShutdownPluginStackChecked(plugins);
  retirement.settled = true;
  bindings.Revoke();
  std::cout << "{\"plugin_count\":" << plugins.size()
            << ",\"state\":\"stopped\"}\n";
  return 0;
}

}  // namespace pih::worker
