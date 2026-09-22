#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "pih/core/result.h"
#include "pih/core/status.h"

namespace pih {

// Hardware-test-only fixture for the deployed canonical W4A16 entry. It owns
// temporary allocations and therefore must never be called by an engine epoch.
Status cuda_verify_qwen_int4_gemm_fixture();

// Deterministic hardware qualification matrix spanning all five official
// linear shape families and non-uniform token-row boundaries.
Status cuda_verify_qwen_int4_gemm_shape_matrix();

// Executes the deployed teacher-forced metric reducer over the official Qwen
// vocabulary and compares its compact outputs with the independent CPU oracle.
Status cuda_verify_qwen_teacher_forced_metric_fixture();

struct QwenInt4ShapeCaseReceipt final {
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
  std::uint64_t pattern_seed = 0;
  std::string output_sha256;
};

struct QwenInt4ShapeMatrixReceipt final {
  std::array<QwenInt4ShapeCaseReceipt, 5> cases;
  std::array<std::uint32_t, 3> negative_launch_invariants;
};

// Returns digests of the bytes actually copied back from each deployed-kernel
// execution. The verifier wrapper below consumes this same implementation.
Result<QwenInt4ShapeMatrixReceipt> cuda_collect_qwen_int4_gemm_shape_matrix();

}  // namespace pih
