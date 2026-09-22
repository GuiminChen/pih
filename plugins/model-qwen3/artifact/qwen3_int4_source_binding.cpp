#include "pih/model/qwen3_int4_source_binding.h"

#include <algorithm>
#include <array>
#include <span>
#include <string_view>

#include "pih/model/qwen3_manifest.h"
#include "pih/model/safetensors_file.h"

namespace pih {
namespace {

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t index = 0; index < wire.size(); ++index) {
    wire[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return digest.update(wire);
}

Status update_string(Sha256& digest, std::string_view value) {
  Status status = update_u64(digest, value.size());
  if (!status.ok()) return status;
  return digest.update(std::as_bytes(std::span(value)));
}

const QwenInt4ObservedSourceRecord* find_record(
    const std::vector<QwenInt4ObservedSourceRecord>& records,
    std::string_view name) {
  const auto found = std::lower_bound(
      records.begin(), records.end(), name,
      [](const auto& record, std::string_view key) { return record.name < key; });
  return found == records.end() || found->name != name ? nullptr : &*found;
}

}  // namespace

Result<QwenInt4SourceBinding> QwenInt4SourceBinding::Verify(
    const QwenInt4DispositionPlan& plan,
    std::vector<QwenInt4ObservedSourceRecord> observed,
    const Qwen3SourceArtifactReceipt& receipt) {
  std::sort(observed.begin(), observed.end(), [](const auto& left,
                                                  const auto& right) {
    return left.name < right.name;
  });
  if (observed.size() != plan.records().size() ||
      observed.size() != QwenInt4DispositionPlan::kOfficialSourceCount) {
    return Status::FailedPrecondition("Qwen INT4 source binding count mismatch");
  }
  for (std::size_t index = 0; index < observed.size(); ++index) {
    const auto& actual = observed[index];
    const auto& expected = plan.records()[index];
    if (index != 0 && observed[index - 1].name == actual.name) {
      return Status::FailedPrecondition("Qwen INT4 source binding duplicate");
    }
    if (actual.name != expected.source_name || actual.dtype != DType::kBFloat16 ||
        actual.shape != expected.source_shape ||
        actual.file_end < actual.file_begin ||
        actual.file_end - actual.file_begin != expected.source_bytes ||
        actual.payload_sha256 == Sha256Digest{}) {
      return Status::FailedPrecondition("Qwen INT4 source binding geometry mismatch");
    }
  }
  const auto* embedding = find_record(observed, "model.embed_tokens.weight");
  const auto* head = find_record(observed, "lm_head.weight");
  if (embedding == nullptr || head == nullptr ||
      embedding->file_begin != receipt.embedding_file_begin ||
      embedding->file_end != receipt.embedding_file_end ||
      embedding->payload_sha256 != receipt.embedding_sha256 ||
      head->file_begin != receipt.lm_head_file_begin ||
      head->file_end != receipt.lm_head_file_end ||
      head->payload_sha256 != receipt.lm_head_sha256 ||
      embedding->payload_sha256 != head->payload_sha256 ||
      embedding->file_begin == head->file_begin) {
    return Status::FailedPrecondition("Qwen INT4 tied-source evidence mismatch");
  }
  return QwenInt4SourceBinding(std::move(observed));
}

Result<QwenInt4SourceBinding> QwenInt4SourceBinding::ObserveAndVerifyOfficial(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactReceipt& receipt) {
  auto plan = QwenInt4DispositionPlan::CreateOfficialPureW4();
  if (!plan.ok()) return plan.status();
  std::vector<QwenInt4ObservedSourceRecord> observed;
  observed.reserve(source.header().tensors().size());
  for (const auto& record : source.header().tensors()) {
    auto tensor = source.tensor(record.name);
    if (!tensor.ok()) return tensor.status();
    auto digest = sha256(tensor->bytes);
    if (!digest.ok()) return digest.status();
    observed.push_back({record.name, record.dtype, record.shape,
                        record.file_begin, record.file_end, *digest});
  }
  return Verify(*plan, std::move(observed), receipt);
}

const QwenInt4ObservedSourceRecord* QwenInt4SourceBinding::find(
    std::string_view name) const noexcept {
  const auto found = std::lower_bound(
      records_.begin(), records_.end(), name,
      [](const auto& record, std::string_view key) {
        return record.name < key;
      });
  return found == records_.end() || found->name != name ? nullptr : &*found;
}

Result<Sha256Digest> QwenInt4SourceBinding::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain = "pih.qwen3_int4_source_binding.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = update_u64(digest, records_.size());
  for (const auto& record : records_) {
    if (status.ok()) status = update_string(digest, record.name);
    if (status.ok()) status = update_u64(digest, static_cast<std::uint8_t>(record.dtype));
    if (status.ok()) status = update_u64(digest, record.shape.size());
    for (const auto extent : record.shape) {
      if (status.ok()) status = update_u64(digest, extent);
    }
    if (status.ok()) status = update_u64(digest, record.file_begin);
    if (status.ok()) status = update_u64(digest, record.file_end);
    if (status.ok()) status = digest.update(record.payload_sha256.bytes);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

}  // namespace pih
