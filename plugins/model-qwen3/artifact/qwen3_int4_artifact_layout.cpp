#include "pih/model/qwen3_int4_artifact_layout.h"

#include <algorithm>
#include <array>
#include <span>
#include <string_view>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"

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

void append_norm(std::vector<QwenInt4ArtifactRecordPlan>& records,
                 std::string name, std::uint64_t elements) {
  records.push_back({name, QwenInt4ArtifactRecordKind::kBf16Norm,
                     std::move(name), {}, elements * 2, 0, 0});
}

}  // namespace

Result<QwenInt4ArtifactLayout>
QwenInt4ArtifactLayout::CreateOfficialPureW4() {
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  if (!ledger.ok()) return ledger.status();

  std::vector<QwenInt4ArtifactRecordPlan> records;
  records.reserve(kOfficialRecordCount);
  records.push_back({"model.embed_tokens.weight",
                     QwenInt4ArtifactRecordKind::kBf16Embedding,
                     "model.embed_tokens.weight", {}, 311'164'928, 0, 0});
  records.push_back({"lm_head.weight", QwenInt4ArtifactRecordKind::kAlias,
                     "lm_head.weight", "model.embed_tokens.weight", 0, 0, 0});
  for (const auto& linear : ledger->records()) {
    records.push_back({linear.values_name,
                       QwenInt4ArtifactRecordKind::kPackedValues,
                       linear.source_name, {}, linear.packed_bytes, 0, 0});
    records.push_back({linear.scales_name,
                       QwenInt4ArtifactRecordKind::kFp16Scales,
                       linear.source_name, {}, linear.scale_bytes, 0, 0});
  }
  for (std::uint64_t layer = 0; layer < 28; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    append_norm(records, prefix + "input_layernorm.weight", 1024);
    append_norm(records, prefix + "post_attention_layernorm.weight", 1024);
    append_norm(records, prefix + "self_attn.q_norm.weight", 128);
    append_norm(records, prefix + "self_attn.k_norm.weight", 128);
  }
  append_norm(records, "model.norm.weight", 1024);
  std::sort(records.begin(), records.end(), [](const auto& left,
                                                const auto& right) {
    return left.identity < right.identity;
  });

  std::uint64_t logical_bytes = 0;
  std::uint64_t next_offset = kMetadataBytes;
  std::size_t payload_count = 0;
  for (std::size_t index = 0; index < records.size(); ++index) {
    auto& record = records[index];
    if (index != 0 && records[index - 1].identity == record.identity) {
      return Status::Internal("Qwen INT4 artifact identity is duplicated");
    }
    if (record.kind == QwenInt4ArtifactRecordKind::kAlias) continue;
    auto extent = checked_align_up_u64(record.logical_bytes, kExtentAlignment);
    if (!extent.ok()) return extent.status();
    auto logical_sum = checked_add_u64(logical_bytes, record.logical_bytes);
    auto offset_sum = checked_add_u64(next_offset, *extent);
    if (!logical_sum.ok()) return logical_sum.status();
    if (!offset_sum.ok()) return offset_sum.status();
    record.file_offset = next_offset;
    record.extent_bytes = *extent;
    logical_bytes = *logical_sum;
    next_offset = *offset_sum;
    ++payload_count;
  }
  const std::uint64_t padded_data_bytes = next_offset - kMetadataBytes;
  if (records.size() != kOfficialRecordCount ||
      payload_count != kOfficialPayloadRecordCount ||
      logical_bytes != kOfficialLogicalPayloadBytes ||
      padded_data_bytes != kOfficialPaddedDataBytes ||
      next_offset != kOfficialFileBytes) {
    return Status::Internal("Qwen INT4 canonical artifact totals drifted");
  }
  return QwenInt4ArtifactLayout(std::move(records), payload_count,
                                logical_bytes, padded_data_bytes, next_offset);
}

Result<Sha256Digest> QwenInt4ArtifactLayout::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain =
      "pih.qwen3_int4_artifact_layout.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = update_u64(digest, records_.size());
  if (status.ok()) status = update_u64(digest, kMetadataBytes);
  for (const auto& record : records_) {
    if (status.ok()) status = update_string(digest, record.identity);
    if (status.ok()) status = update_u64(digest, static_cast<std::uint8_t>(record.kind));
    if (status.ok()) status = update_string(digest, record.source_name);
    if (status.ok()) status = update_string(digest, record.alias_owner);
    if (status.ok()) status = update_u64(digest, record.logical_bytes);
    if (status.ok()) status = update_u64(digest, record.file_offset);
    if (status.ok()) status = update_u64(digest, record.extent_bytes);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

}  // namespace pih
