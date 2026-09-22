#include "pih/plugin_sdk/abi.h"
#include "pih/contracts/text_inference_v2.h"
#include "pih/core/bounded_json.h"
#include "../surface-openai-http/http_request_framing.h"
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace {
namespace http = pih::surface_openai_http;
volatile std::sig_atomic_t stopping = 0;
void StopSignal(int) { stopping = 1; }
struct Descriptor {
  int value{-1};
  explicit Descriptor(int fd) : value(fd) {}
  Descriptor(const Descriptor&) = delete;
  ~Descriptor() { if (value >= 0) ::close(value); }
};
struct State { const pih_host_api_v1* host{}; const pih_text_inference_api_v2* engine{}; uint32_t phase{}; } state;
pih_status_v1 Status(uint32_t code, const char* text = "") {
  pih_status_v1 value{}; value.struct_size = sizeof(value); value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code; std::strncpy(value.message, text, sizeof(value.message) - 1); return value;
}
enum class WaitResult { kReady, kTimeout, kStopped, kClosed, kError };
WaitResult Wait(int fd, short events, std::chrono::steady_clock::time_point until) noexcept {
  while (!stopping) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= until) return WaitResult::kTimeout;
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(until - now).count();
    const auto timeout = static_cast<int>(std::min<decltype(remaining)>(remaining, 250));
    pollfd item{fd, events, 0}; const auto ready = ::poll(&item, 1, timeout);
    if (ready > 0) {
      if (item.revents & POLLNVAL) return WaitResult::kError;
      // Read buffered input before observing an accompanying hangup. A write
      // attempt still reports peer closure through send/MSG_NOSIGNAL.
      if (item.revents & events) return WaitResult::kReady;
      if (item.revents & POLLERR) return WaitResult::kError;
      if (item.revents & POLLHUP) return WaitResult::kClosed;
      return WaitResult::kError;
    }
    if (ready < 0 && errno != EINTR) return WaitResult::kError;
  }
  return WaitResult::kStopped;
}
bool Send(int fd, std::string_view wire) {
  size_t offset = 0; const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (offset < wire.size() && Wait(fd, POLLOUT, deadline) == WaitResult::kReady) {
    auto written = ::send(fd, wire.data() + offset, wire.size() - offset, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (written < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (written <= 0) break;
    offset += static_cast<size_t>(written);
  }
  return offset == wire.size();
}
void Reply(int fd, int code, std::string_view body, const char* allowed_method = nullptr) {
  const auto wire = "HTTP/1.1 " + std::to_string(code) + (code == 200 ? " OK\r\n" : " Error\r\n") +
      (allowed_method ? std::string("Allow: ") + allowed_method + "\r\n" : std::string{}) +
      "Content-Type: application/json; charset=utf-8\r\nConnection: close\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
  (void)Send(fd, wire);
}
bool ValidJsonObject(std::string_view bytes) {
  auto parsed = pih::JsonValue::Parse(bytes,
      {8U * 1024 * 1024, 32, 262144, 8U * 1024 * 1024});
  return parsed.ok() && parsed->is_object();
}
struct Output {
  int fd;
  bool started = false;
  bool streaming = false;
  bool disconnected = false;
  bool contract_failed = false;
};
thread_local Output* active_output = nullptr;
struct OutputScope {
  explicit OutputScope(Output& output) {
    if (active_output) throw std::runtime_error("nested_output_contract");
    active_output = &output;
  }
  OutputScope(const OutputScope&) = delete;
  ~OutputScope() { active_output = nullptr; }
};
bool ValidOutputContext(void* context) noexcept {
  if (active_output && context == active_output) return true;
  if (active_output) active_output->contract_failed = true;
  return false;
}
uint32_t Cancelled(void* context) noexcept {
  if (!ValidOutputContext(context)) return 1;
  auto& output = *static_cast<Output*>(context);
  if (output.disconnected || output.contract_failed || stopping) return 1;
  char byte;
  const auto count = ::recv(output.fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
  // This close-delimited service does not accept pipelining or input half-close.
  // Socket EOF is treated as cancellation even if the peer retains a read half.
  if (count >= 0 || (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK))
    output.disconnected = true;
  return output.disconnected ? 1U : 0U;
}
uint32_t StartOutput(void* context, uint32_t streaming) noexcept {
  if (!ValidOutputContext(context)) return 0;
  auto& output = *static_cast<Output*>(context);
  if (output.started || streaming > 1) {
    output.contract_failed = true;
    return 0;
  }
  if (Cancelled(context)) return 0;
  output.started = true;
  output.streaming = streaming != 0;
  if (!output.streaming) return 1;
  output.disconnected = !Send(output.fd,
      "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
      "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\nConnection: close\r\n\r\n");
  return output.disconnected ? 0U : 1U;
}
uint32_t WriteOutput(void* context, const char* json, uint64_t bytes) noexcept {
  if (!ValidOutputContext(context)) return 0;
  auto& output = *static_cast<Output*>(context);
  if (!output.started || !output.streaming || !json || !bytes || bytes > 8 * 1024 * 1024) {
    output.contract_failed = true;
    return 0;
  }
  if (Cancelled(context)) return 0;
  try {
    const std::string_view value(json, static_cast<size_t>(bytes));
    if (value.find_first_of("\r\n") != std::string_view::npos) {
      output.contract_failed = true;
      return 0;
    }
    if (!ValidJsonObject(value)) {
      output.contract_failed = true;
      return 0;
    }
    const auto frame = "data: " + std::string(value) + "\n\n";
    output.disconnected = !Send(output.fd, frame);
    return output.disconnected ? 0U : 1U;
  } catch (...) { output.disconnected = true; return 0; }
}
void Connection(int fd) {
  std::string wire;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  http::ParseResult request{};
  for (;;) {
    const auto ready = Wait(fd, POLLIN, deadline);
    if (ready != WaitResult::kReady) {
      if (ready == WaitResult::kTimeout && !wire.empty())
        Reply(fd, 408, "{\"error\":\"request_read_deadline_elapsed\"}");
      return;
    }
    char buffer[8192]; auto count = ::recv(fd, buffer, sizeof(buffer), MSG_DONTWAIT);
    if (count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (count <= 0) return;
    wire.append(buffer, static_cast<size_t>(count));
    if (wire.size() > 1024 * 1024 + 16384) { Reply(fd, 413, "{\"error\":\"request_too_large\"}"); return; }
    auto boundary = wire.find("\r\n\r\n");
    if (boundary == std::string::npos) {
      if (wire.size() > 16384) { Reply(fd, 431, "{\"error\":\"headers_too_large\"}"); return; }
      continue;
    }
    if (boundary + 4 > 16384) { Reply(fd, 431, "{\"error\":\"headers_too_large\"}"); return; }
    request = http::ParseRequest(wire, true);
    // The text service has a tighter bound than the shared framing parser.
    // Reject the declared size immediately rather than waiting for its payload.
    if (request.expected_wire_size > boundary + 4 + 1024 * 1024) {
      Reply(fd, 413, "{\"error\":\"request_too_large\"}"); return;
    }
    if (request.error == http::ParseError::kBodyLengthMismatch && request.expected_wire_size > wire.size()) continue;
    if (!request.ok()) {
      const bool discovery = request.request.endpoint == http::Endpoint::kHealth ||
          request.request.endpoint == http::Endpoint::kReady || request.request.endpoint == http::Endpoint::kModels;
      const char* allowed = request.error == http::ParseError::kMethodNotAllowed ?
          (discovery ? "GET" : "POST") : nullptr;
      Reply(fd, http::ParseErrorHttpStatus(request.error),
          std::string("{\"error\":\"") + http::ParseErrorMessage(request.error) + "\"}", allowed);
      return;
    }
    if (request.request.endpoint == http::Endpoint::kHealth || request.request.endpoint == http::Endpoint::kReady) {
      Reply(fd, 200, "{\"ready\":true,\"backend\":\"native-plugin\",\"support_status\":\"unqualified\"}"); return;
    }
    if (request.request.endpoint == http::Endpoint::kModels) {
      Reply(fd, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"" + std::string(state.engine->model_id) + "\",\"object\":\"model\"}]}"); return;
    }
    break;
  }
  std::vector<char> response(8 * 1024 * 1024); uint64_t size = 0;
  Output output{fd};
  OutputScope output_scope(output);
  const pih_text_output_sink_v2 sink{sizeof(sink), PIH_TEXT_INFERENCE_ABI_V2,
      &output, StartOutput, WriteOutput, Cancelled};
  const auto status = state.engine->complete(state.engine->context,
      request.request.endpoint == http::Endpoint::kChatCompletions ? 1 : 0,
      request.request.body.data(), request.request.body.size(), response.data(), response.size(), &size, &sink);
  if (!pih_status_is_valid_v1(&status) || size > response.size()) throw std::runtime_error("engine_contract_invalid");
  // A callback contract failure is terminal even if the provider ignores its
  // cancellation result and returns OK. Never certify that stream with DONE.
  if (output.contract_failed) throw std::runtime_error("engine_output_contract_invalid");
  if (pih_status_is_ok_v1(&status)) {
    if (!output.started || (output.streaming && size != 0) ||
        (!output.streaming && size == 0))
      throw std::runtime_error("engine_output_contract_invalid");
    if (output.streaming) { if (!output.disconnected) (void)Send(fd, "data: [DONE]\n\n"); }
    else {
      const std::string_view body(response.data(), size);
      if (!ValidJsonObject(body)) throw std::runtime_error("engine_response_json_invalid");
      Reply(fd, 200, body);
    }
    return;
  }
  state.host->log(state.host->context, 3, "native_inference_request_failed", status.message);
  const auto error_reply = [&](int code, std::string_view body) {
    if (output.disconnected) return;
    if (output.streaming) {
      (void)WriteOutput(&output, body.data(), body.size());
      // An error terminates the stream without a successful finish or [DONE].
    } else Reply(fd, code, body);
  };
  if (status.code == PIH_STATUS_DEADLINE_EXCEEDED_V1) {
    error_reply(504, "{\"error\":\"generation_deadline_elapsed\"}"); return;
  }
  if (status.code == PIH_STATUS_UNAVAILABLE_V1) {
    error_reply(503, "{\"error\":\"generation_unavailable\"}"); return;
  }
  if (status.code == PIH_STATUS_INVALID_ARGUMENT_V1) {
    error_reply(400, "{\"error\":{\"message\":\"invalid or unsupported request; check the selected model's documented fields and sampling limits\"}}"); return;
  }
  error_reply(500, "{\"error\":\"inference_failed_restart_required\"}");
  throw std::runtime_error("engine_execution_failed");
}
struct SignalScope {
  struct sigaction previous_int{}, previous_term{};
  bool installed_int{}, installed_term{};
  SignalScope() {
    struct sigaction action{}; action.sa_handler = StopSignal; ::sigemptyset(&action.sa_mask);
    installed_int = ::sigaction(SIGINT, &action, &previous_int) == 0;
    if (installed_int) installed_term = ::sigaction(SIGTERM, &action, &previous_term) == 0;
    if (!installed_int || !installed_term) {
      if (installed_int) ::sigaction(SIGINT, &previous_int, nullptr);
      throw std::runtime_error("signal_install_failed");
    }
  }
  ~SignalScope() { ::sigaction(SIGINT, &previous_int, nullptr); ::sigaction(SIGTERM, &previous_term, nullptr); }
};
pih_status_v1 Serve(void* context, uint16_t port) noexcept {
  if (context != &state || state.phase != 4 || !state.engine || !port) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  try {
    Descriptor socket(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (socket.value < 0) throw std::runtime_error("listen_socket_failed");
    int reuse = 1; ::setsockopt(socket.value, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(socket.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(socket.value, 16) != 0)
      throw std::runtime_error("listen_bind_failed");
    stopping = 0; SignalScope signals;
    std::cout << "{\"state\":\"serving\",\"host\":\"127.0.0.1\",\"port\":" << port << ",\"backend\":\"native-plugin\"}" << std::endl;
    while (!stopping) {
      const auto ready = Wait(socket.value, POLLIN,
          std::chrono::steady_clock::now() + std::chrono::seconds(1));
      if (ready == WaitResult::kStopped) break;
      if (ready == WaitResult::kTimeout) continue;
      if (ready != WaitResult::kReady) throw std::runtime_error("listen_poll_failed");
      Descriptor connection(::accept4(socket.value, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK));
      if (connection.value < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
        throw std::runtime_error("accept_failed");
      }
      Connection(connection.value);
    }
    return Status(PIH_STATUS_OK_V1);
  } catch (const std::exception& error) { return Status(PIH_STATUS_INTERNAL_V1, error.what()); }
  catch (...) { return Status(PIH_STATUS_INTERNAL_V1, "native_service_failed"); }
}
pih_text_service_api_v2 api{sizeof(api), PIH_TEXT_INFERENCE_ABI_V2, &state, Serve};
pih_status_v1 Advance(void* c, uint32_t expected) noexcept {
  if (c != &state || state.phase != expected) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  ++state.phase; return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Register(void* c) noexcept {
  if (c != &state || state.phase != 0) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  pih_capability_v1 capability{sizeof(capability), PIH_CAPABILITY_ABI_VERSION_V1, "surface.text-http.v2",
    "pih.surface.text-http.v2", &api, PIH_CAPABILITY_THREADING_SERIALIZED_V1, PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
    PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
  auto result = state.host->register_capability(state.host->context, &capability);
  if (!pih_status_is_valid_v1(&result)) return Status(PIH_STATUS_INTERNAL_V1);
  return pih_status_is_ok_v1(&result) ? Advance(c, 0) : result;
}
pih_status_v1 Configure(void* c) noexcept {
  if (c != &state || state.phase != 1) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  const void* resolved = nullptr;
  const auto status = state.host->resolve_capability(state.host->context, "inference.text.v2", "pih.inference.text.v2",
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1, PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &resolved);
  if (!pih_status_is_ok_v1(&status) || !resolved) return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "text_engine_missing");
  auto* engine = static_cast<const pih_text_inference_api_v2*>(resolved);
  if (engine->struct_size != sizeof(*engine) || engine->contract_version != PIH_TEXT_INFERENCE_ABI_V2 ||
      !engine->context || !engine->model_id || !engine->complete || !engine->load || !engine->close)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "text_engine_contract_invalid");
  // Only printable model IDs may be included in HTTP JSON without escaping.
  size_t count = 0;
  for (; count < 129 && engine->model_id[count]; ++count) {
    const char byte = engine->model_id[count];
    if (!((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9') ||
          byte == '/' || byte == '-' || byte == '_' || byte == '.')) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  }
  if (count == 0 || count == 129) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  state.engine = engine; return Advance(c, 1);
}
pih_status_v1 Start(void* c) noexcept { return Advance(c, 2); }
pih_status_v1 Ready(void* c) noexcept { return Advance(c, 3); }
pih_status_v1 Drain(void* c) noexcept { return Advance(c, 4); }
pih_status_v1 Stop(void* c) noexcept {
  if (c == &state && state.phase == 3) { state.phase = 6; return Status(PIH_STATUS_OK_V1); }
  return Advance(c, 5);
}
pih_status_v1 Dispose(void* c) noexcept {
  if (c != &state || (state.phase != 1 && state.phase != 2 && state.phase != 6)) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  state.engine = nullptr; state.host = nullptr; state.phase = 7; return Status(PIH_STATUS_OK_V1);
}
}
extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host) || !pih_plugin_api_accepts_v1(plugin)) return Status(PIH_STATUS_INVALID_ARGUMENT_V1);
  if (state.host || state.phase) return Status(PIH_STATUS_FAILED_PRECONDITION_V1);
  state.host = host; plugin->plugin_id = "pih.surface.text-http"; plugin->plugin_version = "1.0.0"; plugin->context = &state;
  plugin->lifecycle = {sizeof(pih_plugin_lifecycle_v1), PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1,
    Register, Configure, Start, Ready, Drain, Stop, Dispose};
  return Status(PIH_STATUS_OK_V1);
}
