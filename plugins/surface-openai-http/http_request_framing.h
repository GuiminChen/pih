#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace pih::surface_openai_http {

enum class Endpoint : std::uint8_t { kCompletions, kChatCompletions, kHealth, kReady, kModels };

struct ParsedRequest final {
  Endpoint endpoint{};
  std::string_view body;
};

enum class ParseError : std::uint8_t {
  kNone,
  kHeaderTooLarge,
  kInvalidSyntax,
  kInvalidRequestLine,
  kUnsupportedVersion,
  kEndpointNotFound,
  kMethodNotAllowed,
  kHeadersTooLarge,
  kDuplicateHeader,
  kMissingHost,
  kTransferEncodingUnsupported,
  kMissingContentLength,
  kInvalidContentLength,
  kBodyTooLarge,
  kUnsupportedMediaType,
  kBodyLengthMismatch,
  kExpectationUnsupported,
};

struct ParseResult final {
  ParsedRequest request{};
  ParseError error{ParseError::kNone};
  std::size_t expected_wire_size{};

  [[nodiscard]] bool ok() const noexcept { return error == ParseError::kNone; }
};

// Discovery routes are opt-in: existing inference-only consumers cannot
// accidentally dispatch a GET to their completion implementation.
ParseResult ParseRequest(std::string_view wire, bool allow_discovery = false) noexcept;
const char* ParseErrorMessage(ParseError error) noexcept;
unsigned ParseErrorHttpStatus(ParseError error) noexcept;

}  // namespace pih::surface_openai_http
