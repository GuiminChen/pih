#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/tensor_view.h"
#include "pih/model/qwen3_int4_artifact_layout.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"

namespace pih {

struct QwenInt4LinearResources final {
  QwenInt4LinearShapeFamily family;
  TensorView packed;
  TensorView scales;
};

class QwenInt4WeightResourceSet final {
 public:
  static Result<QwenInt4WeightResourceSet> Create(
      const QwenInt4ArtifactLayout& layout,
      const QwenInt4LinearShapeLedger& ledger,
      const TensorView& canonical_pool, std::int32_t owning_rank);

  Result<TensorView> view(std::string_view identity) const;
  Result<QwenInt4LinearResources> linear(std::string_view source_name) const;
  [[nodiscard]] std::size_t physical_payload_count() const noexcept {
    return physical_payload_count_;
  }
  [[nodiscard]] std::size_t logical_record_count() const noexcept {
    return views_.size();
  }
  [[nodiscard]] std::uint64_t resident_bytes() const noexcept {
    return resident_bytes_;
  }
  [[nodiscard]] std::int32_t owning_rank() const noexcept {
    return owning_rank_;
  }
  [[nodiscard]] std::int32_t device_index() const noexcept {
    return device_index_;
  }

 private:
  struct NamedView final { std::string name; TensorView view; };
  struct NamedLinear final {
    std::string source_name;
    QwenInt4LinearResources resources;
  };
  QwenInt4WeightResourceSet(std::vector<NamedView> views,
                            std::vector<NamedLinear> linears,
                            std::size_t physical_payload_count,
                            std::uint64_t resident_bytes,
                            std::int32_t owning_rank,
                            std::int32_t device_index)
      : views_(std::move(views)), linears_(std::move(linears)),
        physical_payload_count_(physical_payload_count),
        resident_bytes_(resident_bytes), owning_rank_(owning_rank),
        device_index_(device_index) {}

  std::vector<NamedView> views_;
  std::vector<NamedLinear> linears_;
  std::size_t physical_payload_count_;
  std::uint64_t resident_bytes_;
  std::int32_t owning_rank_;
  std::int32_t device_index_;
};

}  // namespace pih
