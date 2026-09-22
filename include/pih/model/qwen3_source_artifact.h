#pragma once

#include <cstddef>
#include <cstdint>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

class SafetensorsFile;

struct Qwen3SourceArtifactExpectation final {
  std::uint64_t file_bytes;
  std::size_t tensor_count;
  std::uint64_t data_bytes;
  Sha256Digest file_sha256;
};

struct Qwen3SourceArtifactReceipt final {
  std::uint64_t file_bytes;
  std::size_t tensor_count;
  std::uint64_t data_bytes;
  Sha256Digest file_sha256;
  std::uint64_t tied_weight_bytes;
  std::uint64_t embedding_file_begin;
  std::uint64_t embedding_file_end;
  Sha256Digest embedding_sha256;
  std::uint64_t lm_head_file_begin;
  std::uint64_t lm_head_file_end;
  Sha256Digest lm_head_sha256;
};

Result<Qwen3SourceArtifactReceipt> verify_qwen3_source_artifact(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactExpectation& expectation);
Status revalidate_qwen3_source_artifact_receipt(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactReceipt& receipt);

}  // namespace pih
