#include "pih/contracts/health_v1.h"
#include "pih/plugin_sdk/abi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <string_view>

#if defined(__linux__)
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>
#endif

namespace {
struct State final {
  std::mutex mutex;
  std::atomic<uint32_t> phase{};
  const pih_host_api_v1* host{};
  std::atomic<bool> engine_ready{};
  uint64_t activation_epoch{};
  uint64_t generation{};
  uint64_t revision{};
  std::string capabilities_json;
  bool activation_bound{};
  bool accepting{};
  std::size_t in_flight{};
};
State state;

bool BeginOperation(State& current) {
  std::lock_guard lock(current.mutex);
  if (current.phase.load() != 4 || !current.accepting ||
      current.in_flight == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  ++current.in_flight;
  return true;
}

struct OperationGuard final {
  State& state;
  ~OperationGuard() {
    std::lock_guard lock(state.mutex);
    --state.in_flight;
  }
};

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

pih_status_v1 CheckedHostStatus(pih_status_v1 status,
                                const char* invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}

template <typename Operation>
pih_status_v1 ContainAbi(Operation operation, const char* failure) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "health_surface_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}
#if defined(__linux__)
bool WaitSocketUntil(int socket, short events,
                     std::chrono::steady_clock::time_point deadline) {
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return false;
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now + std::chrono::milliseconds(1));
    const auto timeout = static_cast<int>(std::min<int64_t>(
        remaining.count(), std::numeric_limits<int>::max()));
    pollfd descriptor{socket, events, 0};
    const auto result = ::poll(&descriptor, 1, timeout);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0 || (descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
      return false;
    }
    return (descriptor.revents & (events | POLLHUP)) != 0;
  }
}
#endif
pih_status_v1 Advance(void* context, uint32_t expected) {
  auto& current = *static_cast<State*>(context);
  if (!current.phase.compare_exchange_strong(expected, expected + 1)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "lifecycle_order_invalid");
  }
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 SnapshotCore(void* context, pih_health_snapshot_v1* snapshot) {
  if (snapshot == nullptr || snapshot->struct_size != sizeof(*snapshot) ||
      snapshot->abi_version != PIH_HEALTH_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "health_snapshot_invalid");
  }
  *snapshot = {sizeof(*snapshot), PIH_HEALTH_ABI_VERSION_V1};
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "health_snapshot_invalid");
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (!current.activation_bound) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_activation_unbound");
  }
  const auto phase = current.phase.load();
  const auto ready = phase == 4 && current.engine_ready.load();
  snapshot->ready = ready ? 1U : 0U;
  snapshot->live = phase >= 4 && phase < 7 ? 1U : 0U;
  snapshot->reason = ready
      ? PIH_HEALTH_REASON_READY_V1
      : (current.generation == 0
            ? PIH_HEALTH_REASON_ENGINE_STARTING_V1
            : PIH_HEALTH_REASON_DRAINING_V1);
  snapshot->activation_epoch = current.activation_epoch;
  snapshot->generation = current.generation;
  snapshot->revision = current.revision;
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 BindActivationCore(void* context, uint64_t activation_epoch) {
  if (context == nullptr || activation_epoch == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_activation_invalid");
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (current.host == nullptr ||
      current.host->activation_epoch != activation_epoch ||
      current.phase.load() != 4 || current.activation_bound ||
      current.engine_ready.load()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_activation_not_bindable");
  }
  current.activation_epoch = activation_epoch;
  current.generation = 0;
  current.revision = 1;
  current.activation_bound = true;
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 PublishCapabilitiesCore(void* context,
                                      const char* capabilities_json,
                                      size_t capabilities_size) {
  if (context == nullptr || capabilities_json == nullptr ||
      capabilities_size < 2 || capabilities_size > 4096 ||
      capabilities_json[0] != '[' ||
      capabilities_json[capabilities_size - 1] != ']') {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_capabilities_invalid");
  }
  for (size_t index = 0; index < capabilities_size; ++index) {
    const auto byte = static_cast<unsigned char>(capabilities_json[index]);
    if (byte < 0x20 || byte > 0x7e) {
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                    "health_capabilities_invalid");
    }
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (current.phase.load() != 4 || !current.activation_bound ||
      !current.capabilities_json.empty() || current.engine_ready.load()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_capabilities_not_publishable");
  }
  try {
    current.capabilities_json.assign(capabilities_json, capabilities_size);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "health_capabilities_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "health_capabilities_publish_failed");
  }
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 PublishReadinessCore(void* context,
                                   uint64_t engine_generation,
                                   uint32_t ready) {
  if (context == nullptr || engine_generation == 0 || ready > 1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_readiness_invalid");
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (current.phase.load() != 4 || !current.activation_bound) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_surface_not_ready");
  }
  if (current.revision == std::numeric_limits<uint64_t>::max()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "health_revision_exhausted");
  }
  const auto requested_ready = ready != 0;
  if ((requested_ready &&
       (!current.accepting || current.generation != 0 ||
        current.engine_ready.load())) ||
      (!requested_ready &&
       (current.generation != engine_generation ||
        !current.engine_ready.load()))) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_readiness_transition_invalid");
  }
  current.generation = engine_generation;
  current.engine_ready.store(requested_ready);
  ++current.revision;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ExecuteHttpExchangeAdmitted(
    void* context, pih_health_http_exchange_v1* exchange) {
  if (context == nullptr || exchange == nullptr ||
      exchange->struct_size != sizeof(*exchange) ||
      exchange->abi_version != PIH_HEALTH_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_http_exchange_invalid");
  }
  // A failed exchange must not leave a previously published response visible
  // to the caller.  Publish the response length only after the complete HTTP
  // envelope has been built and copied below.
  exchange->response_size = 0;
  if (exchange->request_bytes == nullptr || exchange->request_size == 0 ||
      exchange->request_size > 4096 || exchange->response_bytes == nullptr ||
      exchange->response_capacity == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_http_exchange_invalid");
  }
  const std::string_view request(exchange->request_bytes,
                                 exchange->request_size);
  const auto boundary = request.find("\r\n\r\n");
  const auto request_line_end = request.find("\r\n");
  const auto request_line = request.substr(0, request_line_end);
  const auto livez = request_line == "GET /livez HTTP/1.1";
  const auto readyz = request_line == "GET /readyz HTTP/1.1";
  const auto capabilities =
      request_line == "GET /pih/v1/capabilities HTTP/1.1";
  const auto operational_status =
      request_line == "GET /pih/v1/status HTTP/1.1";
  if (boundary == std::string_view::npos || boundary + 4 != request.size() ||
      request_line_end == std::string_view::npos ||
      (!livez && !readyz && !capabilities && !operational_status)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_http_request_invalid");
  }
  bool host_seen = false;
  auto cursor = request_line_end + 2;
  while (cursor < boundary) {
    const auto line_end = request.find("\r\n", cursor);
    if (line_end == std::string_view::npos || line_end > boundary) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "health_http_request_invalid");
    }
    const auto line = request.substr(cursor, line_end - cursor);
    const auto colon = line.find(':');
    if (line.empty() || line.front() == ' ' || line.front() == '\t' ||
        colon == std::string_view::npos || colon == 0) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "health_http_request_invalid");
    }
    auto value = line.substr(colon + 1);
    if (!value.empty() && value.front() == ' ') value.remove_prefix(1);
    const auto name = line.substr(0, colon);
    const auto host_header = name.size() == 4 &&
        (name[0] == 'H' || name[0] == 'h') &&
        (name[1] == 'O' || name[1] == 'o') &&
        (name[2] == 'S' || name[2] == 's') &&
        (name[3] == 'T' || name[3] == 't');
    if (host_header) {
      if (host_seen || value.empty()) {
        return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                      "health_http_host_invalid");
      }
      host_seen = true;
    }
    cursor = line_end + 2;
  }
  if (!host_seen) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_http_host_missing");
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (!current.activation_bound) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_activation_unbound");
  }
  const auto phase = current.phase.load();
  const auto live = phase >= 4 && phase < 7;
  const auto ready = phase == 4 && current.engine_ready.load();
  const auto reason = ready ? "ready" : (current.generation == 0
      ? "engine_starting"
      : "draining");
  char body[4608]{};
  const auto body_bytes = operational_status
      ? std::snprintf(
            body, sizeof(body),
            "{\"activation_epoch\":%llu,\"generation\":%llu,\"revision\":%llu,\"live\":%s,\"ready\":%s,\"reason\":\"%s\"}",
            static_cast<unsigned long long>(current.activation_epoch),
            static_cast<unsigned long long>(current.generation),
            static_cast<unsigned long long>(current.revision),
            live ? "true" : "false", ready ? "true" : "false", reason)
      : capabilities
      ? (current.capabilities_json.empty()
            ? -1
            : std::snprintf(
                  body, sizeof(body),
                  "{\"activation_epoch\":%llu,\"capabilities\":%.*s}",
                  static_cast<unsigned long long>(current.activation_epoch),
                  static_cast<int>(current.capabilities_json.size()),
                  current.capabilities_json.data()))
      : livez
      ? std::snprintf(
            body, sizeof(body),
            "{\"activation_epoch\":%llu,\"generation\":%llu,\"revision\":%llu,\"live\":%s}",
            static_cast<unsigned long long>(current.activation_epoch),
            static_cast<unsigned long long>(current.generation),
            static_cast<unsigned long long>(current.revision),
            live ? "true" : "false")
      : std::snprintf(
            body, sizeof(body),
            "{\"activation_epoch\":%llu,\"generation\":%llu,\"revision\":%llu,\"ready\":%s,\"reason\":\"%s\"}",
            static_cast<unsigned long long>(current.activation_epoch),
            static_cast<unsigned long long>(current.generation),
            static_cast<unsigned long long>(current.revision),
            ready ? "true" : "false", reason);
  if (body_bytes <= 0 || static_cast<std::size_t>(body_bytes) >= sizeof(body)) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "health_http_body_serialization_failed");
  }
  const auto status_line =
      ((capabilities || operational_status) ? live : (livez ? live : ready))
      ? "HTTP/1.1 200 OK\r\n"
      : "HTTP/1.1 503 Service Unavailable\r\n";
  char response[8192]{};
  const auto response_bytes = std::snprintf(
      response, sizeof(response),
      "%sConnection: close\r\nContent-Length: %d\r\nContent-Type: application/json\r\n\r\n%.*s",
      status_line, body_bytes, body_bytes, body);
  if (response_bytes <= 0 ||
      static_cast<std::size_t>(response_bytes) >= sizeof(response) ||
      static_cast<std::size_t>(response_bytes) > exchange->response_capacity) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "health_http_response_capacity_exceeded");
  }
  std::memcpy(exchange->response_bytes, response,
              static_cast<std::size_t>(response_bytes));
  exchange->response_size = static_cast<std::size_t>(response_bytes);
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ExecuteHttpExchangeCore(
    void* context, pih_health_http_exchange_v1* exchange) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_http_exchange_invalid");
  }
  auto& current = *static_cast<State*>(context);
  if (!BeginOperation(current)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_surface_not_accepting");
  }
  OperationGuard operation{current};
  return ExecuteHttpExchangeAdmitted(context, exchange);
}

pih_status_v1 ExecuteHttpConnectionCore(void* context,
                                        intptr_t accepted_socket,
                                        uint64_t deadline_monotonic_ns) {
#if defined(__linux__)
  if (context == nullptr || accepted_socket < 0 ||
      accepted_socket > std::numeric_limits<int>::max() ||
      deadline_monotonic_ns == 0 ||
      deadline_monotonic_ns >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_http_connection_invalid");
  }
  const auto deadline = std::chrono::steady_clock::time_point(
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::nanoseconds(deadline_monotonic_ns)));
  if (std::chrono::steady_clock::now() >= deadline) {
    return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                  "health_http_socket_deadline_elapsed");
  }
  auto& current = *static_cast<State*>(context);
  if (!BeginOperation(current)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_surface_not_accepting");
  }
  OperationGuard operation{current};
  char request[4096]{};
  std::size_t request_size = 0;
  for (;;) {
    if (request_size == sizeof(request)) {
      return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                    "health_http_request_capacity_exceeded");
    }
    if (!WaitSocketUntil(static_cast<int>(accepted_socket), POLLIN, deadline)) {
      return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                    "health_http_socket_receive_deadline");
    }
    const auto received = ::recv(
        static_cast<int>(accepted_socket), request + request_size,
        sizeof(request) - request_size, MSG_DONTWAIT);
    if (received < 0 && (errno == EINTR || errno == EAGAIN ||
                         errno == EWOULDBLOCK)) continue;
    if (received <= 0) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "health_http_socket_receive_failed");
    }
    request_size += static_cast<std::size_t>(received);
    const std::string_view wire(request, request_size);
    const auto boundary = wire.find("\r\n\r\n");
    if (boundary == std::string_view::npos) continue;
    if (boundary + 4 != request_size) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "health_http_request_invalid");
    }
    break;
  }
  char response[8192]{};
  pih_health_http_exchange_v1 exchange{
      sizeof(exchange), PIH_HEALTH_ABI_VERSION_V1, request, request_size,
      response, sizeof(response), 0};
  auto status = ExecuteHttpExchangeAdmitted(context, &exchange);
  if (!pih_status_is_ok_v1(&status)) return status;
  if (exchange.struct_size != sizeof(exchange) ||
      exchange.abi_version != PIH_HEALTH_ABI_VERSION_V1 ||
      exchange.response_size == 0 ||
      exchange.response_size > exchange.response_capacity) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "health_http_socket_response_invalid");
  }
  std::size_t sent = 0;
  while (sent < exchange.response_size) {
    if (!WaitSocketUntil(static_cast<int>(accepted_socket), POLLOUT,
                         deadline)) {
      return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                    "health_http_socket_send_deadline");
    }
    const auto result = ::send(
        static_cast<int>(accepted_socket), response + sent,
        exchange.response_size - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (result < 0 && (errno == EINTR || errno == EAGAIN ||
                       errno == EWOULDBLOCK)) continue;
    if (result <= 0) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "health_http_socket_send_failed");
    }
    sent += static_cast<std::size_t>(result);
  }
  if (::shutdown(static_cast<int>(accepted_socket), SHUT_WR) != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_http_socket_shutdown_failed");
  }
  return Status(PIH_STATUS_OK_V1);
#else
  (void)context;
  (void)accepted_socket;
  (void)deadline_monotonic_ns;
  return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                "health_http_socket_platform_unsupported");
#endif
}

pih_status_v1 Snapshot(void* context,
                       pih_health_snapshot_v1* snapshot) noexcept {
  return ContainAbi([&] { return SnapshotCore(context, snapshot); },
                    "health_snapshot_failed");
}

pih_status_v1 BindActivation(void* context,
                             uint64_t activation_epoch) noexcept {
  return ContainAbi(
      [&] { return BindActivationCore(context, activation_epoch); },
      "health_activation_bind_failed");
}

pih_status_v1 PublishCapabilities(void* context,
                                  const char* capabilities_json,
                                  size_t capabilities_size) noexcept {
  return ContainAbi(
      [&] {
        return PublishCapabilitiesCore(context, capabilities_json,
                                       capabilities_size);
      },
      "health_capabilities_publish_failed");
}

pih_status_v1 PublishReadiness(void* context, uint64_t engine_generation,
                               uint32_t ready) noexcept {
  return ContainAbi(
      [&] {
        return PublishReadinessCore(context, engine_generation, ready);
      },
      "health_readiness_publish_failed");
}

pih_status_v1 ExecuteHttpExchange(
    void* context, pih_health_http_exchange_v1* exchange) noexcept {
  return ContainAbi(
      [&] { return ExecuteHttpExchangeCore(context, exchange); },
      "health_http_exchange_failed");
}

pih_status_v1 ExecuteHttpConnection(
    void* context, intptr_t accepted_socket,
    uint64_t deadline_monotonic_ns) noexcept {
  return ContainAbi(
      [&] {
        return ExecuteHttpConnectionCore(context, accepted_socket,
                                         deadline_monotonic_ns);
      },
      "health_http_connection_failed");
}

pih_health_api_v1 api{sizeof(pih_health_api_v1), PIH_HEALTH_ABI_VERSION_V1,
                      PIH_HEALTH_READINESS_ENGINE_GENERATION_V1,
                      PIH_HEALTH_ACCEPTED_SOCKET_CALLER_OWNED_V1, &state,
                      &Snapshot,
                      &BindActivation, &PublishCapabilities, &PublishReadiness,
                      &ExecuteHttpExchange, &ExecuteHttpConnection};
pih_status_v1 RegisterCore(void* context) {
  auto& current = *static_cast<State*>(context);
  pih_capability_v1 capability{sizeof(pih_capability_v1),
                               PIH_CAPABILITY_ABI_VERSION_V1,
                               "surface.health.v1",
                               "pih.surface.health.v1", &api,
                               PIH_CAPABILITY_THREADING_CONCURRENT_V1,
                               PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                               PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
  const auto status = CheckedHostStatus(
      current.host->register_capability(current.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  return Advance(context, 0);
}
pih_status_v1 ConfigureCore(void* context) { return Advance(context, 1); }
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  const auto status = Advance(context, 3);
  if (pih_status_is_ok_v1(&status)) state.accepting = true;
  return status;
}
pih_status_v1 DrainCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase.load() != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  state.accepting = false;
  if (state.engine_ready.load()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_readiness_still_published");
  }
  if (state.in_flight != 0) {
    return Status(PIH_STATUS_UNAVAILABLE_V1,
                  "health_surface_connections_draining");
  }
  return Advance(context, 4);
}
pih_status_v1 StopCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase.load() == 3) {
    state.phase.store(6);
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  const auto phase = state.phase.load();
  if ((phase != 1 && phase != 2 && phase != 6) ||
      state.engine_ready.load() ||
      state.accepting || state.in_flight != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "health_surface_resources_still_live");
  }
  state.activation_epoch = 0;
  state.generation = 0;
  state.revision = 0;
  state.capabilities_json.clear();
  state.activation_bound = false;
  state.host = nullptr;
  state.phase.store(7);
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 Register(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return RegisterCore(context); },
                    "health_registration_failed");
}
pih_status_v1 Configure(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return ConfigureCore(context); },
                    "health_configuration_failed");
}
pih_status_v1 Start(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return StartCore(context); },
                    "health_start_failed");
}
pih_status_v1 Ready(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return ReadyCore(context); },
                    "health_ready_failed");
}
pih_status_v1 Drain(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return DrainCore(context); },
                    "health_drain_failed");
}
pih_status_v1 Stop(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return StopCore(context); },
                    "health_stop_failed");
}
pih_status_v1 Dispose(void* context) noexcept {
  if (context == nullptr)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "health_lifecycle_context_invalid");
  return ContainAbi([&] { return DisposeCore(context); },
                    "health_dispose_failed");
}
}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "host_api_invalid");
  }
  if (!pih_plugin_api_accepts_v1(plugin)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "plugin_api_invalid");
  }
  {
    std::lock_guard lock(state.mutex);
    if (state.host != nullptr || state.phase.load() != 0) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "plugin_entry_already_bound");
    }
    state.host = host;
  }
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.surface.health";
  plugin->plugin_version = "1.0.0";
  plugin->context = &state;
  plugin->lifecycle = {sizeof(pih_plugin_lifecycle_v1),
                       PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1,
                       &Register, &Configure, &Start, &Ready, &Drain, &Stop,
                       &Dispose};
  return Status(PIH_STATUS_OK_V1);
}
