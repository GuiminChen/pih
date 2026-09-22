#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include "pih/core/tensor_view.h"

namespace pih {

class GemmPlan final {
 public:
  static Result<std::unique_ptr<GemmPlan>> Create(
      std::uint64_t m, std::uint64_t n, std::uint64_t k,
      std::uint64_t max_workspace_bytes,
      DType output_dtype = DType::kBFloat16);
  ~GemmPlan();

  GemmPlan(const GemmPlan&) = delete;
  GemmPlan& operator=(const GemmPlan&) = delete;

  Status execute(const TensorView& a, const TensorView& b, const TensorView& c,
                 void* workspace, std::uint64_t workspace_bytes,
                 cudaStream_t stream) const;

  [[nodiscard]] std::uint64_t m() const noexcept { return m_; }
  [[nodiscard]] std::uint64_t n() const noexcept { return n_; }
  [[nodiscard]] std::uint64_t k() const noexcept { return k_; }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept {
    return workspace_bytes_;
  }
  [[nodiscard]] std::int32_t algorithm_id() const noexcept { return algorithm_id_; }
  [[nodiscard]] DType output_dtype() const noexcept { return output_dtype_; }

 private:
  struct Impl;
  GemmPlan(std::uint64_t m, std::uint64_t n, std::uint64_t k,
           std::uint64_t workspace_bytes, std::int32_t algorithm_id,
           DType output_dtype,
           std::unique_ptr<Impl> impl);

  std::uint64_t m_;
  std::uint64_t n_;
  std::uint64_t k_;
  std::uint64_t workspace_bytes_;
  std::int32_t algorithm_id_;
  DType output_dtype_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pih
