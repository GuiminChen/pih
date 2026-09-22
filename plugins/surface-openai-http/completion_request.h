#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/result.h"

namespace pih::surface_openai_http {

struct CompletionRequest final {
  std::string model;
  std::vector<std::string> prompts;
  std::uint32_t maximum_completion_tokens{};
  double temperature{1.0};
  double top_p{1.0};
  std::vector<std::string> stop;
  std::optional<std::uint64_t> seed;
  bool logprobs{};
  std::uint32_t top_logprobs{};
  bool stream{};
  bool include_usage{};
  std::optional<std::string> user;
};

Result<CompletionRequest> ParseCompletionRequest(
    std::string_view body, std::string_view expected_model,
    std::uint32_t default_maximum_completion_tokens);

}  // namespace pih::surface_openai_http
