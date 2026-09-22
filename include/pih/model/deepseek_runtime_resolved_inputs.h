#pragma once

#include "pih/core/sha256.h"

namespace pih {

// Non-release profile inputs which a domain evidence object may resolve
// without creating evidence -> profile -> evidence dependency cycles.
struct DeepSeekRuntimeResolvedInputRoots final {
  Sha256Digest runtime_semantic_root{};
  Sha256Digest capacity_template_root{};
  Sha256Digest feature_selection_root{};
  Sha256Digest hardware_identity_root{};
  Sha256Digest kernel_closure_root{};
  Sha256Digest security_runtime_root{};
};

}  // namespace pih
