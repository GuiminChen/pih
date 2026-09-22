#pragma once
// Private to native offline Qwen conversion; not a model/runtime SDK interface.

#include <span>

#include "qwen3_int4_conversion_stream.h"
#include "pih/model/qwen3_int4_source_binding.h"
#include "pih/model/safetensors_file.h"

namespace pih {

Result<std::span<const std::byte>> verify_qwen_int4_bound_tensor(
    const ImmutableTensorBytes& tensor, const SafetensorRecord& physical,
    const QwenInt4ObservedSourceRecord& binding);

class QwenInt4SafetensorsSource final
    : public QwenInt4ConversionTensorSource {
 public:
  static Result<QwenInt4SafetensorsSource> Create(
      const SafetensorsFile& source, const QwenInt4SourceBinding& binding);
  Result<std::span<const std::byte>> tensor(
      std::string_view source_name) override;

 private:
  QwenInt4SafetensorsSource(const SafetensorsFile* source,
                            const QwenInt4SourceBinding* binding)
      : source_(source), binding_(binding) {}
  const SafetensorsFile* source_;
  const QwenInt4SourceBinding* binding_;
};

}  // namespace pih
