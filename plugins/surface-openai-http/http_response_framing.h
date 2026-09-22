#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "pih/core/result.h"

namespace pih::surface_openai_http {

enum class JsonResponseStatus : unsigned short {
  kOk = 200,
  kBadRequest = 400,
  kNotFound = 404,
  kMethodNotAllowed = 405,
  kPayloadTooLarge = 413,
  kUnsupportedMediaType = 415,
  kRequestHeaderFieldsTooLarge = 431,
  kServiceUnavailable = 503,
};

Result<std::string> SerializeJsonResponse(JsonResponseStatus status,
                                          std::string_view body,
                                          bool keep_alive,
                                          std::size_t maximum_head_bytes,
                                          std::size_t maximum_body_bytes);
Result<std::string> SerializeJsonResponse(std::string_view body,
                                          bool keep_alive,
                                          std::size_t maximum_head_bytes,
                                          std::size_t maximum_body_bytes);
Result<std::string> SerializeSseResponse(std::string_view event_json,
                                         std::size_t maximum_head_bytes,
                                         std::size_t maximum_body_bytes);

}  // namespace pih::surface_openai_http
