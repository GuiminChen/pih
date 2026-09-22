#include "http_response_framing.h"

#include <charconv>
#include <system_error>

namespace pih::surface_openai_http {
namespace {

std::string_view ReasonPhrase(JsonResponseStatus status) noexcept {
  switch (status) {
    case JsonResponseStatus::kOk: return "OK";
    case JsonResponseStatus::kBadRequest: return "Bad Request";
    case JsonResponseStatus::kNotFound: return "Not Found";
    case JsonResponseStatus::kMethodNotAllowed: return "Method Not Allowed";
    case JsonResponseStatus::kPayloadTooLarge: return "Payload Too Large";
    case JsonResponseStatus::kUnsupportedMediaType:
      return "Unsupported Media Type";
    case JsonResponseStatus::kRequestHeaderFieldsTooLarge:
      return "Request Header Fields Too Large";
    case JsonResponseStatus::kServiceUnavailable:
      return "Service Unavailable";
  }
  return {};
}

}  // namespace

Result<std::string> SerializeJsonResponse(JsonResponseStatus status,
                                          std::string_view body,
                                          bool keep_alive,
                                          std::size_t maximum_head_bytes,
                                          std::size_t maximum_body_bytes) {
  if (maximum_head_bytes == 0 || maximum_head_bytes > 1024 * 1024 ||
      maximum_body_bytes == 0 || maximum_body_bytes > 64 * 1024 * 1024) {
    return Status::InvalidArgument("openai_http_response_limits_invalid");
  }
  if (body.empty() || body.size() > maximum_body_bytes) {
    return Status::ResourceExhausted("openai_http_response_body_too_large");
  }
  const auto reason = ReasonPhrase(status);
  if (reason.empty()) {
    return Status::InvalidArgument("openai_http_response_status_invalid");
  }
  char length[32]{};
  const auto converted = std::to_chars(length, length + sizeof(length),
                                       body.size());
  if (converted.ec != std::errc{}) {
    return Status::Internal("openai_http_content_length_failed");
  }
  char status_code[8]{};
  const auto status_converted = std::to_chars(
      status_code, status_code + sizeof(status_code),
      static_cast<unsigned short>(status));
  if (status_converted.ec != std::errc{}) {
    return Status::Internal("openai_http_status_code_failed");
  }
  std::string head = "HTTP/1.1 ";
  head.append(status_code, status_converted.ptr);
  head += " ";
  head += reason;
  head += "\r\nConnection: ";
  head += keep_alive ? "keep-alive" : "close";
  head += "\r\nContent-Length: ";
  head.append(length, converted.ptr);
  head += "\r\nContent-Type: application/json\r\n\r\n";
  if (head.size() > maximum_head_bytes) {
    return Status::ResourceExhausted("openai_http_response_head_too_large");
  }
  if (body.size() > head.max_size() - head.size()) {
    return Status::ResourceExhausted("openai_http_response_size_overflow");
  }
  head.append(body);
  return head;
}

Result<std::string> SerializeJsonResponse(std::string_view body,
                                          bool keep_alive,
                                          std::size_t maximum_head_bytes,
                                          std::size_t maximum_body_bytes) {
  return SerializeJsonResponse(JsonResponseStatus::kOk, body, keep_alive,
                               maximum_head_bytes, maximum_body_bytes);
}

Result<std::string> SerializeSseResponse(std::string_view event_json,
                                         std::size_t maximum_head_bytes,
                                         std::size_t maximum_body_bytes) {
  if (maximum_head_bytes == 0 || maximum_head_bytes > 1024 * 1024 ||
      maximum_body_bytes == 0 || maximum_body_bytes > 64 * 1024 * 1024) {
    return Status::InvalidArgument("openai_sse_response_limits_invalid");
  }
  if (event_json.empty() || event_json.size() > maximum_body_bytes ||
      event_json.find('\r') != std::string_view::npos ||
      event_json.find('\n') != std::string_view::npos) {
    return Status::InvalidArgument("openai_sse_event_invalid");
  }
  std::string response =
      "HTTP/1.1 200 OK\r\n"
      "Cache-Control: no-store\r\n"
      "Connection: close\r\n"
      "Content-Type: text/event-stream\r\n\r\n";
  if (response.size() > maximum_head_bytes) {
    return Status::ResourceExhausted("openai_sse_response_head_too_large");
  }
  constexpr std::string_view prefix = "data: ";
  constexpr std::string_view separator = "\r\n\r\n";
  constexpr std::string_view terminal = "data: [DONE]\r\n\r\n";
  const auto body_bytes = prefix.size() + event_json.size() +
                          separator.size() + terminal.size();
  if (body_bytes > maximum_body_bytes ||
      body_bytes > response.max_size() - response.size()) {
    return Status::ResourceExhausted("openai_sse_response_body_too_large");
  }
  response.append(prefix);
  response.append(event_json);
  response.append(separator);
  response.append(terminal);
  return response;
}

}  // namespace pih::surface_openai_http
