#include "pih/model/qwen3_source_artifact.h"

#include <algorithm>

#include "pih/model/safetensors_file.h"

namespace pih {
namespace {

constexpr std::string_view kEmbeddingName = "model.embed_tokens.weight";
constexpr std::string_view kLmHeadName = "lm_head.weight";

Status metadata_mismatch(std::string_view field) {
  return Status::FailedPrecondition("Qwen source artifact " +
                                    std::string(field) +
                                    " does not match sealed manifest");
}

}  // namespace

Result<Qwen3SourceArtifactReceipt> verify_qwen3_source_artifact(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactExpectation& expectation) {
  if (source.size_bytes() != expectation.file_bytes) {
    return metadata_mismatch("file length");
  }
  if (source.header().tensors().size() != expectation.tensor_count) {
    return metadata_mismatch("tensor count");
  }
  if (source.header().data_bytes() != expectation.data_bytes) {
    return metadata_mismatch("data length");
  }

  auto observed_digest = sha256(source.file_bytes());
  if (!observed_digest.ok()) return observed_digest.status();
  if (*observed_digest != expectation.file_sha256) {
    return metadata_mismatch("SHA-256");
  }

  auto embedding = source.tensor(kEmbeddingName);
  if (!embedding.ok()) {
    return Status::FailedPrecondition("Qwen source artifact lacks embedding weight");
  }
  auto lm_head = source.tensor(kLmHeadName);
  if (!lm_head.ok()) {
    return Status::FailedPrecondition("Qwen source artifact lacks LM-head weight");
  }
  if (embedding->dtype != lm_head->dtype ||
      !std::ranges::equal(embedding->shape, lm_head->shape) ||
      !std::ranges::equal(embedding->bytes, lm_head->bytes)) {
    return Status::FailedPrecondition(
        "Qwen source embedding and LM-head weights are not byte-identical");
  }

  const auto* embedding_record = source.header().tensor(kEmbeddingName);
  const auto* lm_head_record = source.header().tensor(kLmHeadName);
  if (embedding_record == nullptr || lm_head_record == nullptr ||
      embedding_record->file_begin == lm_head_record->file_begin ||
      embedding_record->file_end == lm_head_record->file_end) {
    return Status::FailedPrecondition(
        "Qwen source tied weights lack distinct physical ranges");
  }
  auto embedding_digest = sha256(embedding->bytes);
  if (!embedding_digest.ok()) return embedding_digest.status();
  auto lm_head_digest = sha256(lm_head->bytes);
  if (!lm_head_digest.ok()) return lm_head_digest.status();
  if (*embedding_digest != *lm_head_digest) {
    return Status::FailedPrecondition(
        "Qwen source tied weight digests do not match");
  }

  return Qwen3SourceArtifactReceipt{
      source.size_bytes(), source.header().tensors().size(),
      source.header().data_bytes(), *observed_digest,
      static_cast<std::uint64_t>(embedding->bytes.size()),
      embedding_record->file_begin, embedding_record->file_end,
      *embedding_digest, lm_head_record->file_begin, lm_head_record->file_end,
      *lm_head_digest};
}

Status revalidate_qwen3_source_artifact_receipt(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactReceipt& receipt) {
  if (source.size_bytes() != receipt.file_bytes ||
      source.header().tensors().size() != receipt.tensor_count ||
      source.header().data_bytes() != receipt.data_bytes) {
    return metadata_mismatch("sealed receipt geometry");
  }
  auto file_digest = sha256(source.file_bytes());
  if (!file_digest.ok()) return file_digest.status();
  if (*file_digest != receipt.file_sha256) {
    return metadata_mismatch("sealed receipt SHA-256");
  }
  const auto* embedding_record = source.header().tensor(kEmbeddingName);
  const auto* head_record = source.header().tensor(kLmHeadName);
  auto embedding = source.tensor(kEmbeddingName);
  auto head = source.tensor(kLmHeadName);
  if (embedding_record == nullptr || head_record == nullptr ||
      !embedding.ok() || !head.ok()) {
    return Status::FailedPrecondition(
        "Qwen source sealed tied records are missing");
  }
  auto embedding_digest = sha256(embedding->bytes);
  auto head_digest = sha256(head->bytes);
  if (!embedding_digest.ok()) return embedding_digest.status();
  if (!head_digest.ok()) return head_digest.status();
  if (embedding_record->file_begin != receipt.embedding_file_begin ||
      embedding_record->file_end != receipt.embedding_file_end ||
      head_record->file_begin != receipt.lm_head_file_begin ||
      head_record->file_end != receipt.lm_head_file_end ||
      embedding->bytes.size() != receipt.tied_weight_bytes ||
      *embedding_digest != receipt.embedding_sha256 ||
      *head_digest != receipt.lm_head_sha256 ||
      *embedding_digest != *head_digest) {
    return Status::FailedPrecondition(
        "Qwen source sealed tied evidence changed");
  }
  return Status::Ok();
}

}  // namespace pih
