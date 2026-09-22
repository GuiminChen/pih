#pragma once
#include <string>
#include <vector>
#include <span>
#include <string_view>
#include "pih/core/dtype.h"
#include "pih/core/result.h"

namespace pih::offline_deepseek {
// Frozen V4 Flash 0731 grammar. Source checkpoints include three MTP stages;
// the currently deployed PP1 target excludes them. This verifies names only.
Result<std::vector<std::string>> ExpectedTensorNames(bool include_mtp);
// Only routed main-layer experts are checked here; other tensor classes are
// outside this helper's scope. Exact name inventory must be checked separately.
Status ValidateRoutedExpertGeometry(std::string_view name, DType dtype,
                                   std::span<const std::uint64_t> shape);
// Validates every non-routed PP1 tensor. Use with exact name inventory and the
// routed-expert helper; runtime fused names and MTP tensors are not accepted.
Status ValidatePp1DenseGeometry(std::string_view name, DType dtype,
                              std::span<const std::uint64_t> shape);
// Complete source geometry, including MTP common tensors and stage-specific
// boundary/head tensors. Storage validation only, not DSpark execution support.
Status ValidateSourceTensorGeometry(std::string_view name, DType dtype,
                                    std::span<const std::uint64_t> shape);
}  // namespace pih::offline_deepseek
