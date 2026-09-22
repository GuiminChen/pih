#include "pih/contracts/openai_http_v1.h"
#include "pih/plugin_sdk/abi.h"

#include "completion_request.h"
#include "http_request_framing.h"
#include "http_response_framing.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>
#endif

namespace {
struct State final {
  std::mutex mutex;
  uint32_t phase{};
  const pih_host_api_v1* host{};
  const pih_engine_factory_api_v1* factory{};
  pih_engine_handle_v1 engine{};
  std::string model_id;
  bool engine_bound{};
  bool engine_binding_issued{};
  bool accepting{};
  std::size_t in_flight{};
};
State state;

bool AppendFloat(std::string& output, float value) {
  if (!std::isfinite(value)) return false;
  char bytes[64]{};
  const auto converted = std::to_chars(
      std::begin(bytes), std::end(bytes), value,
      std::chars_format::general, std::numeric_limits<float>::max_digits10);
  if (converted.ec != std::errc{}) return false;
  output.append(bytes, converted.ptr);
  return true;
}

bool BeginOperation(State& current) {
  std::lock_guard lock(current.mutex);
  if (current.phase != 4 || current.factory == nullptr ||
      !current.engine_bound || !current.accepting ||
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

pih_status_v1 Status(uint32_t code, std::string_view message = {}) {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  const auto bytes = std::min(message.size(), sizeof(value.message) - 1);
  if (bytes != 0) std::memcpy(value.message, message.data(), bytes);
  return value;
}

pih_status_v1 CheckedHostStatus(pih_status_v1 status,
                                std::string_view invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}
#if defined(__linux__)
pih::surface_openai_http::JsonResponseStatus HttpStatusFor(
    pih::surface_openai_http::ParseError error) noexcept {
  using HttpStatus = pih::surface_openai_http::JsonResponseStatus;
  using ParseError = pih::surface_openai_http::ParseError;
  switch (error) {
    case ParseError::kEndpointNotFound: return HttpStatus::kNotFound;
    case ParseError::kMethodNotAllowed: return HttpStatus::kMethodNotAllowed;
    case ParseError::kHeaderTooLarge:
    case ParseError::kHeadersTooLarge:
      return HttpStatus::kRequestHeaderFieldsTooLarge;
    case ParseError::kBodyTooLarge: return HttpStatus::kPayloadTooLarge;
    case ParseError::kUnsupportedMediaType:
      return HttpStatus::kUnsupportedMediaType;
    default: return HttpStatus::kBadRequest;
  }
}

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

pih_status_v1 SendSocketResponse(
    int socket, std::string_view response,
    std::chrono::steady_clock::time_point deadline) {
  std::size_t sent = 0;
  while (sent < response.size()) {
    if (!WaitSocketUntil(socket, POLLOUT, deadline)) {
      return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                    "openai_http_socket_send_deadline");
    }
    const auto result = ::send(socket, response.data() + sent,
                               response.size() - sent,
                               MSG_NOSIGNAL | MSG_DONTWAIT);
    if (result < 0 && (errno == EINTR || errno == EAGAIN ||
                       errno == EWOULDBLOCK)) {
      continue;
    }
    if (result <= 0) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "openai_http_socket_send_failed");
    }
    sent += static_cast<std::size_t>(result);
  }
  if (::shutdown(socket, SHUT_WR) != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_http_socket_shutdown_failed");
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 SendParseError(int socket,
                             pih::surface_openai_http::ParseError error,
                             std::chrono::steady_clock::time_point deadline) {
  const auto code = pih::surface_openai_http::ParseErrorMessage(error);
  std::string body = R"({"error":{"code":")";
  body += code;
  body += R"(","message":"Invalid HTTP request.","type":"invalid_request_error"}})";
  auto response = pih::surface_openai_http::SerializeJsonResponse(
      HttpStatusFor(error), body, false, 4096, 4096);
  if (!response.ok()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_http_error_response_failed");
  }
  return SendSocketResponse(socket, *response, deadline);
}

pih_status_v1 SendOperationError(
    int socket, const pih_status_v1& status,
    std::chrono::steady_clock::time_point deadline) {
  using HttpStatus = pih::surface_openai_http::JsonResponseStatus;
  const auto client_error = status.code == PIH_STATUS_INVALID_ARGUMENT_V1;
  const auto http_status = client_error ? HttpStatus::kBadRequest
                                        : HttpStatus::kServiceUnavailable;
  const auto body = client_error
      ? R"({"error":{"code":"openai_request_rejected","message":"The request was rejected.","type":"invalid_request_error"}})"
      : R"({"error":{"code":"openai_service_unavailable","message":"The service is unavailable.","type":"server_error"}})";
  auto response = pih::surface_openai_http::SerializeJsonResponse(
      http_status, body, false, 4096, 4096);
  if (!response.ok()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_http_error_response_failed");
  }
  return SendSocketResponse(socket, *response, deadline);
}
#endif

pih_status_v1 Advance(void* context, uint32_t expected) {
  auto& current = *static_cast<State*>(context);
  if (current.phase != expected) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "lifecycle_order_invalid");
  }
  ++current.phase;
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 BindEngineCore(
    void* context, const pih_engine_handle_v1* engine) {
  if (context == nullptr || !pih_engine_handle_is_live_v1(engine)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "engine_handle_invalid");
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (current.host == nullptr ||
      current.host->activation_epoch != engine->activation_epoch ||
      current.phase != 4 || current.factory == nullptr ||
      current.engine_bound || current.engine_binding_issued ||
      current.accepting || current.in_flight != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_not_bindable");
  }
  std::string model_id(engine->model_id);
  current.engine = *engine;
  current.model_id = std::move(model_id);
  current.engine_bound = true;
  current.engine_binding_issued = true;
  current.accepting = true;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 BindEngine(
    void* context, const pih_engine_handle_v1* engine) {
  try {
    return BindEngineCore(context, engine);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_surface_binding_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_surface_binding_failed");
  }
}
pih_status_v1 ExecuteSmokeExchangeAdmitted(
    void* context, pih_openai_http_exchange_v1* exchange,
    uint64_t deadline_monotonic_ns) {
  if (context == nullptr || exchange == nullptr ||
      exchange->struct_size != sizeof(*exchange) ||
      exchange->abi_version != PIH_OPENAI_HTTP_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "surface_context_invalid");
  }
  // Treat the response as an output transaction.  Once the ABI envelope is
  // known to be writable, revoke any caller-visible result before validating
  // the request or invoking the engine.  Only the final successful commit
  // below publishes a non-zero size.
  exchange->response_size = 0;
  if (exchange->request_bytes == nullptr || exchange->request_size == 0 ||
      exchange->response_bytes == nullptr || exchange->response_capacity == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "surface_context_invalid");
  }
  auto& current = *static_cast<State*>(context);
  const std::string_view wire(exchange->request_bytes, exchange->request_size);
  const auto parsed = pih::surface_openai_http::ParseRequest(wire);
  if (!parsed.ok()) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  pih::surface_openai_http::ParseErrorMessage(parsed.error));
  }
  if (parsed.request.endpoint !=
          pih::surface_openai_http::Endpoint::kCompletions ||
      parsed.request.body.empty()) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "openai_http_smoke_request_drifted");
  }
  pih_engine_smoke_request_v1 engine_request{};
  engine_request.struct_size = sizeof(engine_request);
  engine_request.abi_version = PIH_ENGINE_ABI_VERSION_V1;
  engine_request.deadline_monotonic_ns = deadline_monotonic_ns;
  std::string prompt_bytes;
  std::string stop_bytes;
  try {
    auto completion = pih::surface_openai_http::ParseCompletionRequest(
        parsed.request.body, current.model_id, 1);
    if (!completion.ok()) {
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                    completion.status().message());
    }
    if (completion->prompts.size() != 1 ||
        completion->maximum_completion_tokens != 1 || completion->stream ||
        completion->temperature != 0.0 || completion->top_p != 1.0 ||
        completion->stop != std::vector<std::string>{"END"} ||
        completion->seed != 7 || !completion->logprobs ||
        completion->top_logprobs != 2 || completion->include_usage ||
        completion->user != "smoke") {
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                    "openai_completion_smoke_request_drifted");
    }
    engine_request.prompt_count =
        static_cast<uint32_t>(completion->prompts.size());
    prompt_bytes = completion->prompts.front();
    engine_request.prompt_bytes = prompt_bytes.data();
    engine_request.prompt_size = prompt_bytes.size();
    engine_request.maximum_completion_tokens =
        completion->maximum_completion_tokens;
    engine_request.temperature = static_cast<float>(completion->temperature);
    engine_request.top_p = static_cast<float>(completion->top_p);
    engine_request.seed = *completion->seed;
    engine_request.seed_present = 1;
    engine_request.logprobs_enabled = completion->logprobs ? 1U : 0U;
    engine_request.top_logprobs_count = completion->top_logprobs;
    engine_request.stop_count = static_cast<uint32_t>(completion->stop.size());
    stop_bytes = completion->stop.front();
    engine_request.stop_bytes = stop_bytes.data();
    engine_request.stop_size = stop_bytes.size();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_completion_normalization_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_completion_normalization_failed");
  }
  pih_engine_smoke_result_v1 engine_result{};
  engine_result.struct_size = sizeof(engine_result);
  engine_result.abi_version = PIH_ENGINE_ABI_VERSION_V1;
  const auto engine_status = current.factory->execute_smoke_request(
      current.factory->context, &current.engine, &engine_request,
      &engine_result);
  if (!pih_status_is_valid_v1(&engine_status)) {
    return Status(PIH_STATUS_INTERNAL_V1, "engine_status_abi_invalid");
  }
  if (!pih_status_is_ok_v1(&engine_status)) return engine_status;
  if (engine_result.struct_size != sizeof(engine_result) ||
      engine_result.abi_version != PIH_ENGINE_ABI_VERSION_V1 ||
      engine_result.top_logprobs_count !=
          PIH_ENGINE_SMOKE_TOP_LOGPROBS_MAX_V1 ||
      engine_result.finish_reason_length != 1 ||
      !std::isfinite(engine_result.selected_logprob)) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_surface_engine_result_invalid");
  }
  bool selected_token_present = false;
  for (uint32_t index = 0; index < engine_result.top_logprobs_count; ++index) {
    const auto& candidate = engine_result.top_logprobs[index];
    if (!std::isfinite(candidate.logprob) ||
        (index != 0 &&
         engine_result.top_logprobs[index - 1].logprob < candidate.logprob)) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    "openai_surface_engine_result_invalid");
    }
    for (uint32_t prior = 0; prior < index; ++prior) {
      if (engine_result.top_logprobs[prior].token_id == candidate.token_id) {
        return Status(PIH_STATUS_INTERNAL_V1,
                      "openai_surface_engine_result_invalid");
      }
    }
    if (candidate.token_id == engine_result.token_id) {
      if (candidate.logprob != engine_result.selected_logprob) {
        return Status(PIH_STATUS_INTERNAL_V1,
                      "openai_surface_engine_result_invalid");
      }
      selected_token_present = true;
    }
  }
  if (!selected_token_present) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_surface_engine_result_invalid");
  }
  try {
    std::string response_body =
        R"({"choices":[{"finish_reason":"length","index":0,"logprobs":{"token_id":)";
    response_body += std::to_string(engine_result.token_id);
    response_body += R"(,"token_logprob":)";
    if (!AppendFloat(response_body, engine_result.selected_logprob)) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    "openai_surface_engine_result_invalid");
    }
    response_body += R"(,"top_logprobs":[)";
    for (uint32_t index = 0; index < engine_result.top_logprobs_count;
         ++index) {
      if (index != 0) response_body += ',';
      response_body += R"({"token_id":)";
      response_body +=
          std::to_string(engine_result.top_logprobs[index].token_id);
      response_body += R"(,"logprob":)";
      if (!AppendFloat(response_body,
                       engine_result.top_logprobs[index].logprob)) {
        return Status(PIH_STATUS_INTERNAL_V1,
                      "openai_surface_engine_result_invalid");
      }
      response_body += '}';
    }
    response_body +=
        R"(]},"text":""}],"created":0,"id":"pih-smoke","model":")";
    response_body += current.model_id;
    response_body +=
        R"(","object":"text_completion","usage":{"completion_tokens":1,"prompt_tokens":1,"total_tokens":2}})";
    auto response = pih::surface_openai_http::SerializeJsonResponse(
        response_body, false, 4096, 1024 * 1024);
    if (!response.ok()) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    response.status().message());
    }
    constexpr std::string_view expected_prefix =
        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: ";
    if (!response->starts_with(expected_prefix) ||
        !response->ends_with(response_body) ||
        response->size() > exchange->response_capacity) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    "openai_http_smoke_response_drifted");
    }
    std::string stream_event =
        R"({"choices":[{"finish_reason":"length","index":0,"text":""}],"id":"pih-smoke","model":")";
    stream_event += current.model_id;
    stream_event += R"(","object":"text_completion.chunk"})";
    auto stream_response = pih::surface_openai_http::SerializeSseResponse(
        stream_event, 4096, 1024 * 1024);
    if (!stream_response.ok()) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    stream_response.status().message());
    }
    if (!stream_response->starts_with("HTTP/1.1 200 OK\r\n") ||
        !stream_response->ends_with("data: [DONE]\r\n\r\n") ||
        stream_response->find("Content-Type: text/event-stream\r\n") ==
            std::string::npos) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    "openai_sse_smoke_response_drifted");
    }
    std::memcpy(exchange->response_bytes, response->data(), response->size());
    exchange->response_size = response->size();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_http_response_serialization_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_http_response_serialization_failed");
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ExecuteSmokeExchangeCore(
    void* context, pih_openai_http_exchange_v1* exchange) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "surface_context_invalid");
  }
  auto& current = *static_cast<State*>(context);
  if (!BeginOperation(current)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_engine_unbound");
  }
  OperationGuard operation{current};
  constexpr auto kSmokeExecutionTimeout = std::chrono::minutes(5);
  const auto deadline = std::chrono::steady_clock::now() +
                        kSmokeExecutionTimeout;
  const auto deadline_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               deadline.time_since_epoch())
                               .count();
  if (deadline_ns <= 0) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_smoke_deadline_invalid");
  }
  return ExecuteSmokeExchangeAdmitted(
      context, exchange, static_cast<uint64_t>(deadline_ns));
}

pih_status_v1 ExecuteSmokeExchange(void* context,
                                   pih_openai_http_exchange_v1* exchange) {
  try {
    return ExecuteSmokeExchangeCore(context, exchange);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_http_exchange_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_http_exchange_failed");
  }
}

pih_status_v1 ExecuteSmokeConnection(void* context,
                                     intptr_t accepted_socket,
                                     uint64_t deadline_monotonic_ns) {
#if defined(__linux__)
  if (context == nullptr || accepted_socket < 0 ||
      accepted_socket > std::numeric_limits<int>::max() ||
      deadline_monotonic_ns == 0 ||
      deadline_monotonic_ns >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "openai_http_socket_invalid");
  }
  auto& current = *static_cast<State*>(context);
  if (!BeginOperation(current)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_not_accepting");
  }
  OperationGuard connection{current};
  const auto deadline = std::chrono::steady_clock::time_point(
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::nanoseconds(deadline_monotonic_ns)));
  if (std::chrono::steady_clock::now() >= deadline) {
    return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                  "openai_http_socket_deadline_elapsed");
  }
  constexpr std::size_t kMaximumWireBytes = 16 * 1024 + 4 * 1024 * 1024;
  constexpr std::size_t kReadChunkBytes = 16 * 1024;
  std::vector<char> request;
  try {
    request.reserve(kReadChunkBytes);
    for (;;) {
      if (request.size() == kMaximumWireBytes) {
        return SendParseError(
            static_cast<int>(accepted_socket),
            pih::surface_openai_http::ParseError::kBodyTooLarge, deadline);
      }
      const auto previous = request.size();
      const auto available = std::min(kReadChunkBytes,
                                      kMaximumWireBytes - previous);
      request.resize(previous + available);
      if (!WaitSocketUntil(static_cast<int>(accepted_socket), POLLIN,
                           deadline)) {
        request.resize(previous);
        return Status(PIH_STATUS_DEADLINE_EXCEEDED_V1,
                      "openai_http_socket_receive_deadline");
      }
      const auto received = ::recv(static_cast<int>(accepted_socket),
                                   request.data() + previous, available,
                                   MSG_DONTWAIT);
      if (received < 0 && (errno == EINTR || errno == EAGAIN ||
                           errno == EWOULDBLOCK)) {
        request.resize(previous);
        continue;
      }
      if (received <= 0) {
        return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                      "openai_http_socket_receive_failed");
      }
      request.resize(previous + static_cast<std::size_t>(received));
      const std::string_view wire(request.data(), request.size());
      const auto boundary = wire.find("\r\n\r\n");
      if (boundary == std::string_view::npos) {
        if (wire.size() > 16 * 1024) {
          return SendParseError(
              static_cast<int>(accepted_socket),
              pih::surface_openai_http::ParseError::kHeaderTooLarge,
              deadline);
        }
        continue;
      }
      const auto parsed = pih::surface_openai_http::ParseRequest(wire);
      if (!parsed.ok() &&
          parsed.error ==
              pih::surface_openai_http::ParseError::kBodyLengthMismatch &&
          wire.size() < parsed.expected_wire_size) {
        continue;
      }
      if (!parsed.ok()) {
        return SendParseError(static_cast<int>(accepted_socket), parsed.error,
                              deadline);
      }
      break;
    }
    std::vector<char> response(1024 * 1024);
    pih_openai_http_exchange_v1 exchange{
        sizeof(exchange), PIH_OPENAI_HTTP_ABI_VERSION_V1, request.data(),
        request.size(), response.data(), response.size(), 0};
    auto status = ExecuteSmokeExchangeAdmitted(
        context, &exchange, deadline_monotonic_ns);
    if (!pih_status_is_ok_v1(&status)) {
      return SendOperationError(static_cast<int>(accepted_socket), status,
                                deadline);
    }
    if (exchange.struct_size != sizeof(exchange) ||
        exchange.abi_version != PIH_OPENAI_HTTP_ABI_VERSION_V1 ||
        exchange.response_size == 0 ||
        exchange.response_size > exchange.response_capacity) {
      return Status(PIH_STATUS_INTERNAL_V1,
                    "openai_http_socket_response_invalid");
    }
    return SendSocketResponse(
        static_cast<int>(accepted_socket),
        std::string_view(response.data(), exchange.response_size), deadline);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_http_socket_buffer_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_http_socket_execution_failed");
  }
#else
  (void)context;
  (void)accepted_socket;
  (void)deadline_monotonic_ns;
  return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                "openai_http_socket_platform_unsupported");
#endif
}
pih_status_v1 UnbindEngineCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "surface_context_invalid");
  }
  auto& current = *static_cast<State*>(context);
  std::lock_guard lock(current.mutex);
  if (current.phase != 4 || !current.engine_bound) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_engine_unbound");
  }
  current.accepting = false;
  if (current.in_flight != 0) {
    return Status(PIH_STATUS_UNAVAILABLE_V1,
                  "openai_surface_connections_draining");
  }
  current.engine = {};
  current.model_id.clear();
  current.engine_bound = false;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 UnbindEngine(void* context) noexcept {
  try {
    return UnbindEngineCore(context);
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "openai_surface_unbind_failed");
  }
}
pih_openai_http_api_v1 api{sizeof(pih_openai_http_api_v1),
                           PIH_OPENAI_HTTP_ABI_VERSION_V1,
                           PIH_OPENAI_HTTP_ACCEPTED_SOCKET_CALLER_OWNED_V1,
                           &state,
                           &BindEngine, &UnbindEngine,
                           &ExecuteSmokeExchange, &ExecuteSmokeConnection};
pih_status_v1 RegisterCore(void* context) {
  auto& current = *static_cast<State*>(context);
  pih_capability_v1 capability{sizeof(pih_capability_v1),
                               PIH_CAPABILITY_ABI_VERSION_V1,
                               "surface.openai-http.v1",
                               "pih.surface.openai-http.v1", &api,
                               PIH_CAPABILITY_THREADING_CONCURRENT_V1,
                               PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
                               PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
  const auto status = CheckedHostStatus(
      current.host->register_capability(current.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  return Advance(context, 0);
}
pih_status_v1 ConfigureCore(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.phase != 1) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  if (state.host->resolve_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_resolver_missing");
  }
  const void* api = nullptr;
  const auto resolved = CheckedHostStatus(
      state.host->resolve_capability(
          state.host->context, "engine.primary.v1", "pih.engine.v1",
          PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api),
      "capability_resolver_status_invalid");
  if (!pih_status_is_ok_v1(&resolved)) return resolved;
  const auto* factory =
      static_cast<const pih_engine_factory_api_v1*>(api);
  if (factory == nullptr || factory->struct_size != sizeof(*factory) ||
      factory->contract_version != PIH_ENGINE_ABI_VERSION_V1 ||
      factory->handle_generation_semantics !=
          PIH_ENGINE_HANDLE_GENERATION_BOUND_READ_ONLY_V1 ||
      factory->context == nullptr ||
      factory->create == nullptr ||
      factory->admit_request == nullptr ||
      factory->execute_smoke_request == nullptr ||
      factory->destroy == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_engine_factory_invalid");
  }
  const auto advanced = Advance(context, 1);
  if (!pih_status_is_ok_v1(&advanced)) return advanced;
  state.factory = factory;
  return advanced;
}
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) { return Advance(context, 3); }
pih_status_v1 DrainCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.engine_bound || state.engine.instance != nullptr ||
      state.engine.activation_epoch != 0 || state.engine.generation != 0 ||
      state.accepting ||
      state.in_flight != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_engine_still_bound");
  }
  return Advance(context, 4);
}
pih_status_v1 StopCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if ((state.phase != 1 && state.phase != 2 && state.phase != 6) ||
      state.engine_bound ||
      state.engine.instance != nullptr || state.engine.activation_epoch != 0 ||
      state.engine.generation != 0 ||
      state.accepting || state.in_flight != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "openai_surface_resources_still_live");
  }
  state.factory = nullptr;
  state.engine = {};
  state.engine_bound = false;
  state.engine_binding_issued = false;
  state.accepting = false;
  state.host = nullptr;
  state.phase = 7;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ContainLifecycle(void* context,
                               pih_lifecycle_callback_v1 callback,
                               const char* failure) noexcept {
  if (context == nullptr || callback == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "openai_lifecycle_context_invalid");
  }
  try {
    return callback(context);
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "openai_lifecycle_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Register(void* context) noexcept {
  return ContainLifecycle(context, &RegisterCore,
                          "openai_registration_failed");
}
pih_status_v1 Configure(void* context) noexcept {
  return ContainLifecycle(context, &ConfigureCore,
                          "openai_configuration_failed");
}
pih_status_v1 Start(void* context) noexcept {
  return ContainLifecycle(context, &StartCore, "openai_start_failed");
}
pih_status_v1 Ready(void* context) noexcept {
  return ContainLifecycle(context, &ReadyCore, "openai_ready_failed");
}
pih_status_v1 Drain(void* context) noexcept {
  return ContainLifecycle(context, &DrainCore, "openai_drain_failed");
}
pih_status_v1 Stop(void* context) noexcept {
  return ContainLifecycle(context, &StopCore, "openai_stop_failed");
}
pih_status_v1 Dispose(void* context) noexcept {
  return ContainLifecycle(context, &DisposeCore,
                          "openai_dispose_failed");
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
    if (state.host != nullptr || state.phase != 0) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "plugin_entry_already_bound");
    }
    state.host = host;
  }
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.surface.openai-http";
  plugin->plugin_version = "1.0.0";
  plugin->context = &state;
  plugin->lifecycle = {sizeof(pih_plugin_lifecycle_v1),
                       PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1,
                       &Register, &Configure, &Start, &Ready, &Drain, &Stop,
                       &Dispose};
  return Status(PIH_STATUS_OK_V1);
}
