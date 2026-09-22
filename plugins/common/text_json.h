#pragma once
#include "pih/core/bounded_json.h"

namespace pih::plugin_text {
// Insertion-order JSON with Python json.dumps(ensure_ascii=False) separators.
// Not authority/canonical JSON. Rejects non-finite numbers and invalid UTF-8.
std::string TextJson(const JsonValue& value, size_t maximum_bytes = 8 * 1024 * 1024);
std::string QuoteText(std::string_view text, size_t maximum_bytes = 8 * 1024 * 1024);
void RequireUtf8(std::string_view text);
}
