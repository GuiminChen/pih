#pragma once
#include "weight_dequantize.h"
#include "pih/model/safetensors_header.h"

namespace pih::deepseek_v41 {
// Metadata binding only. Headers must come from independently authenticated,
// stable source files. This object cannot authenticate a file or checkpoint.
class WoASourceBinding final {
 public:
  static Result<WoASourceBinding> Create(std::uint32_t layer,
      const SafetensorsHeader& weight_header, std::string_view weight_name,
      const SafetensorsHeader& scale_header, std::string_view scale_name);
  const std::string& canonical_name() const noexcept { return canonical_name_; }
  const WoASourceGeometry& geometry() const noexcept { return geometry_; }
  // Readers now receive absolute offsets within their respective source files,
  // unlike DequantizeWoATensor's tensor-relative callback contract. Writer still
  // receives canonical tensor-relative offsets. Source files may be distinct.
  Status Convert(const WoATensorReader& read_weight_file,
      const WoATensorReader& read_scale_file, const WoATensorWriter& write) const;
 private:
  WoASourceBinding() = default;
  std::string canonical_name_;
  WoASourceGeometry geometry_;
  std::uint64_t weight_begin_ = 0, scale_begin_ = 0;
};
}  // namespace pih::deepseek_v41
