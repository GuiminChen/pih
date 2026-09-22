#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "pih/core/bounded_json.h"

namespace pih {

// Serializes the bounded JSON tree using the same ASCII-only canonical form as
// Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True).
// Floating-point JSON numbers and non-ASCII strings are deliberately rejected:
// signed inference authorities use integer geometry and canonical ASCII names.
Result<std::string> canonical_ascii_json(
    const JsonValue& value, std::size_t maximum_bytes,
    std::span<const std::string_view> omitted_root_fields = {});

}  // namespace pih
