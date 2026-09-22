#pragma once
// Private to native offline Qwen conversion; not a model/runtime SDK interface.

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "pih/io/exclusive_file_publisher.h"
#include "pih/model/qwen3_int4_artifact_io.h"
#include "pih/model/qwen3_int4_source_binding.h"
#include "pih/model/qwen3_source_artifact.h"
#include "pih/model/safetensors_file.h"

namespace pih {

struct QwenInt4ConversionReceipt final {
  ExclusiveFilePublicationReceipt publication;
  QwenInt4ArtifactRoots roots;
  std::size_t source_record_count;
  std::size_t target_record_count;
  std::size_t payload_record_count;
  std::uint64_t quantization_workspace_peak_bytes;
  std::size_t writer_workspace_bytes;
};

Result<QwenInt4ConversionReceipt> convert_qwen3_official_to_int4(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactReceipt& source_receipt,
    const QwenInt4SourceBinding& source_binding,
    const std::filesystem::path& staging_path,
    const std::filesystem::path& published_path,
    std::size_t writer_workspace_bytes);

}  // namespace pih
