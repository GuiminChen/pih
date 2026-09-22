#include "http_request_framing.h"

#include <array>
#include <limits>

namespace pih::surface_openai_http {
namespace {

constexpr std::size_t kMaximumHeaderBytes = 16 * 1024;
constexpr std::size_t kMaximumHeaderCount = 32;
constexpr std::size_t kMaximumBodyBytes = 4 * 1024 * 1024;

bool ascii_iequal(std::string_view left, std::string_view right) noexcept {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i) {
    auto a = left[i];
    auto b = right[i];
    if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
    if (a != b) return false;
  }
  return true;
}

bool token_name(std::string_view value) noexcept {
  if (value.empty()) return false;
  constexpr std::string_view punctuation = "!#$%&'*+-.^_`|~";
  for (const char byte : value) {
    if ((byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') ||
        (byte >= 'a' && byte <= 'z') ||
        punctuation.find(byte) != std::string_view::npos) {
      continue;
    }
    return false;
  }
  return true;
}

ParseResult fail(ParseError error, std::size_t expected_wire_size = 0) noexcept {
  return {{}, error, expected_wire_size};
}

}  // namespace

ParseResult ParseRequest(std::string_view wire, bool allow_discovery) noexcept {
  const auto boundary = wire.find("\r\n\r\n");
  if (boundary == std::string_view::npos) return fail(ParseError::kInvalidSyntax);
  if (boundary + 4 > kMaximumHeaderBytes) {
    return fail(ParseError::kHeaderTooLarge);
  }
  const auto head = wire.substr(0, boundary);
  for (std::size_t i = 0; i < head.size(); ++i) {
    const auto byte = static_cast<unsigned char>(head[i]);
    if (byte == 0 || byte == 127 || byte > 127 ||
        (byte == '\n' && (i == 0 || head[i - 1] != '\r')) ||
        (byte == '\r' && (i + 1 == head.size() || head[i + 1] != '\n'))) {
      return fail(ParseError::kInvalidSyntax);
    }
  }
  const auto request_line_end = head.find("\r\n");
  const auto request_line = head.substr(0, request_line_end);
  const auto first_space = request_line.find(' ');
  const auto second_space = first_space == std::string_view::npos
      ? std::string_view::npos
      : request_line.find(' ', first_space + 1);
  if (first_space == std::string_view::npos ||
      second_space == std::string_view::npos || first_space == 0 ||
      second_space == first_space + 1 || second_space + 1 >= request_line.size() ||
      request_line.find(' ', second_space + 1) != std::string_view::npos) {
    return fail(ParseError::kInvalidRequestLine);
  }
  const auto method = request_line.substr(0, first_space);
  const auto target = request_line.substr(first_space + 1,
                                          second_space - first_space - 1);
  const auto version = request_line.substr(second_space + 1);
  if (version != "HTTP/1.1") return fail(ParseError::kUnsupportedVersion);
  Endpoint endpoint;
  if (target == "/v1/completions") {
    endpoint = Endpoint::kCompletions;
  } else if (target == "/v1/chat/completions") {
    endpoint = Endpoint::kChatCompletions;
  } else if (allow_discovery && target == "/health") {
    endpoint = Endpoint::kHealth;
  } else if (allow_discovery && target == "/readyz") {
    endpoint = Endpoint::kReady;
  } else if (allow_discovery && target == "/v1/models") {
    endpoint = Endpoint::kModels;
  } else {
    return fail(ParseError::kEndpointNotFound);
  }
  const bool discovery = endpoint == Endpoint::kHealth ||
      endpoint == Endpoint::kReady || endpoint == Endpoint::kModels;
  if (method != (discovery ? "GET" : "POST"))
    return {{endpoint, {}}, ParseError::kMethodNotAllowed, 0};

  std::array<std::string_view, kMaximumHeaderCount> names{};
  std::size_t header_count = 0;
  bool host = false;
  bool content_length_seen = false;
  bool content_type = false;
  std::uint64_t content_length = 0;
  std::size_t cursor = request_line_end == std::string_view::npos
      ? head.size()
      : request_line_end + 2;
  while (cursor < head.size()) {
    const auto end = head.find("\r\n", cursor);
    const auto line_end = end == std::string_view::npos ? head.size() : end;
    const auto line = head.substr(cursor, line_end - cursor);
    if (header_count == names.size()) return fail(ParseError::kHeadersTooLarge);
    const auto colon = line.find(':');
    if (line.empty() || line.front() == ' ' || line.front() == '\t' ||
        colon == std::string_view::npos || !token_name(line.substr(0, colon))) {
      return fail(ParseError::kInvalidSyntax);
    }
    const auto name = line.substr(0, colon);
    for (std::size_t i = 0; i < header_count; ++i) {
      if (ascii_iequal(names[i], name)) return fail(ParseError::kDuplicateHeader);
    }
    names[header_count++] = name;
    auto value = line.substr(colon + 1);
    if (!value.empty() && value.front() == ' ') value.remove_prefix(1);
    if ((!value.empty() && (value.front() == ' ' || value.front() == '\t' ||
                            value.back() == ' ' || value.back() == '\t'))) {
      return fail(ParseError::kInvalidSyntax);
    }
    for (const char byte : value) {
      if (static_cast<unsigned char>(byte) < 32 || byte == 127) {
        return fail(ParseError::kInvalidSyntax);
      }
    }
    if (ascii_iequal(name, "host")) {
      if (value.empty()) return fail(ParseError::kMissingHost);
      host = true;
    } else if (ascii_iequal(name, "transfer-encoding")) {
      return fail(ParseError::kTransferEncodingUnsupported);
    } else if (ascii_iequal(name, "expect")) {
      // No interim responses are supported. Reject before waiting for a body
      // whose sender may be waiting for 100 Continue from this server.
      return fail(ParseError::kExpectationUnsupported);
    } else if (ascii_iequal(name, "content-type")) {
      content_type = ascii_iequal(value, "application/json") ||
                     ascii_iequal(value, "application/json; charset=utf-8");
    } else if (ascii_iequal(name, "content-length")) {
      if (value.empty() || (value.size() > 1 && value.front() == '0')) {
        return fail(ParseError::kInvalidContentLength);
      }
      for (const char digit : value) {
        if (digit < '0' || digit > '9' ||
            content_length > (std::numeric_limits<std::uint64_t>::max() -
                              static_cast<std::uint64_t>(digit - '0')) / 10) {
          return fail(ParseError::kInvalidContentLength);
        }
        content_length = content_length * 10 +
                         static_cast<std::uint64_t>(digit - '0');
      }
      content_length_seen = true;
    }
    cursor = line_end == head.size() ? head.size() : line_end + 2;
  }
  if (!host) return fail(ParseError::kMissingHost);
  if (discovery) {
    // No payload, transfer coding, or pipelined bytes on discovery requests.
    // Header validation above is identical to the inference path.
    if (content_length != 0) return fail(ParseError::kInvalidContentLength);
    if (wire.size() != boundary + 4) return fail(ParseError::kBodyLengthMismatch, boundary + 4);
    return {{endpoint, {}}, ParseError::kNone, boundary + 4};
  }
  if (!content_length_seen) return fail(ParseError::kMissingContentLength);
  if (content_length > kMaximumBodyBytes) return fail(ParseError::kBodyTooLarge);
  if (!content_type) return fail(ParseError::kUnsupportedMediaType);
  const auto body = wire.substr(boundary + 4);
  const auto expected_wire_size =
      boundary + 4 + static_cast<std::size_t>(content_length);
  if (body.size() != content_length) {
    return fail(ParseError::kBodyLengthMismatch, expected_wire_size);
  }
  return {{endpoint, body}, ParseError::kNone, expected_wire_size};
}

unsigned ParseErrorHttpStatus(ParseError error) noexcept {
  switch (error) {
    case ParseError::kNone: return 200;
    case ParseError::kEndpointNotFound: return 404;
    case ParseError::kMethodNotAllowed: return 405;
    case ParseError::kHeaderTooLarge:
    case ParseError::kHeadersTooLarge: return 431;
    case ParseError::kBodyTooLarge: return 413;
    case ParseError::kUnsupportedMediaType: return 415;
    case ParseError::kExpectationUnsupported: return 417;
    case ParseError::kMissingContentLength: return 411;
    case ParseError::kUnsupportedVersion: return 505;
    default: return 400;
  }
}

const char* ParseErrorMessage(ParseError error) noexcept {
  switch (error) {
    case ParseError::kNone: return "";
    case ParseError::kHeaderTooLarge: return "openai_http_headers_too_large";
    case ParseError::kInvalidSyntax: return "openai_http_invalid_syntax";
    case ParseError::kInvalidRequestLine: return "openai_http_invalid_request_line";
    case ParseError::kUnsupportedVersion: return "openai_http_version_unsupported";
    case ParseError::kEndpointNotFound: return "openai_http_endpoint_not_found";
    case ParseError::kMethodNotAllowed: return "openai_http_method_not_allowed";
    case ParseError::kHeadersTooLarge: return "openai_http_header_count_exceeded";
    case ParseError::kDuplicateHeader: return "openai_http_duplicate_header";
    case ParseError::kMissingHost: return "openai_http_host_missing";
    case ParseError::kTransferEncodingUnsupported: return "openai_http_transfer_encoding_unsupported";
    case ParseError::kMissingContentLength: return "openai_http_content_length_missing";
    case ParseError::kInvalidContentLength: return "openai_http_content_length_invalid";
    case ParseError::kBodyTooLarge: return "openai_http_body_too_large";
    case ParseError::kUnsupportedMediaType: return "openai_http_media_type_unsupported";
    case ParseError::kBodyLengthMismatch: return "openai_http_body_length_mismatch";
    case ParseError::kExpectationUnsupported: return "openai_http_expectation_unsupported";
  }
  return "openai_http_parse_error";
}

}  // namespace pih::surface_openai_http
