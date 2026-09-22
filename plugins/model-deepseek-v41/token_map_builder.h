#pragma once
#include "config.h"
#include <vector>

namespace pih::deepseek_v41 {
// Offline generation only. Authenticates the frozen tokenizer JSON and requires
// the generated map to match an independently trusted reference map digest.
// Does not qualify ICU/Rust tokenizer equivalence by vocabulary count alone.
Result<std::vector<std::byte>> BuildEngramTokenMap(std::string_view tokenizer_json,
    const Sha256Digest& expected_map_digest);
}  // namespace pih::deepseek_v41
