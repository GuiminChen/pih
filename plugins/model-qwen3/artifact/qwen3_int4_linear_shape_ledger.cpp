#include "pih/model/qwen3_int4_linear_shape_ledger.h"

#include <algorithm>
#include <array>
#include <span>
#include <string_view>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

struct FamilyGeometry final {
  QwenInt4LinearShapeFamily family;
  std::string_view suffix;
  std::uint64_t rows;
  std::uint64_t columns;
};

constexpr std::array kPerLayerGeometry{
    FamilyGeometry{QwenInt4LinearShapeFamily::kQProj,
                   "self_attn.q_proj.weight", 2048, 1024},
    FamilyGeometry{QwenInt4LinearShapeFamily::kKvProj,
                   "self_attn.k_proj.weight", 1024, 1024},
    FamilyGeometry{QwenInt4LinearShapeFamily::kKvProj,
                   "self_attn.v_proj.weight", 1024, 1024},
    FamilyGeometry{QwenInt4LinearShapeFamily::kOProj,
                   "self_attn.o_proj.weight", 1024, 2048},
    FamilyGeometry{QwenInt4LinearShapeFamily::kGateUpProj,
                   "mlp.gate_proj.weight", 3072, 1024},
    FamilyGeometry{QwenInt4LinearShapeFamily::kGateUpProj,
                   "mlp.up_proj.weight", 3072, 1024},
    FamilyGeometry{QwenInt4LinearShapeFamily::kDownProj,
                   "mlp.down_proj.weight", 1024, 3072},
};

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

Result<QwenInt4LinearRecordPlan> make_record(
    std::uint64_t layer, const FamilyGeometry& geometry) {
  if (geometry.columns % 128 != 0) {
    return Status::InvalidArgument("Qwen INT4 K must be divisible by 128");
  }
  auto elements = checked_mul_u64(geometry.rows, geometry.columns);
  if (!elements.ok()) return elements.status();
  auto scale_elements = checked_mul_u64(geometry.rows, geometry.columns / 128);
  if (!scale_elements.ok()) return scale_elements.status();
  auto scale_bytes = checked_mul_u64(*scale_elements, std::uint64_t{2});
  if (!scale_bytes.ok()) return scale_bytes.status();
  auto bf16_bytes = checked_mul_u64(*elements, std::uint64_t{2});
  if (!bf16_bytes.ok()) return bf16_bytes.status();
  const std::string source = "model.layers." + std::to_string(layer) + "." +
                             std::string(geometry.suffix);
  return QwenInt4LinearRecordPlan{
      geometry.family,
      source,
      source + ".packed_values",
      source + ".scales",
      geometry.rows,
      geometry.columns,
      geometry.columns / 128,
      *elements / 2,
      *scale_bytes,
      *bf16_bytes,
      *bf16_bytes - (*elements / 2 + *scale_bytes)};
}

}  // namespace

Result<QwenInt4LinearShapeLedger>
QwenInt4LinearShapeLedger::CreateOfficial() {
  std::vector<QwenInt4LinearRecordPlan> records;
  records.reserve(kOfficialRecordCount);
  for (std::uint64_t layer = 0; layer < 28; ++layer) {
    for (const auto& geometry : kPerLayerGeometry) {
      auto record = make_record(layer, geometry);
      if (!record.ok()) return record.status();
      records.push_back(std::move(*record));
    }
  }
  std::sort(records.begin(), records.end(), [](const auto& left,
                                                const auto& right) {
    return left.source_name < right.source_name;
  });
  if (records.size() != kOfficialRecordCount) {
    return Status::Internal("Qwen INT4 official linear record count drifted");
  }
  std::uint64_t packed_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint64_t bf16_bytes = 0;
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (index != 0 &&
        records[index - 1].source_name == records[index].source_name) {
      return Status::Internal("Qwen INT4 official linear name is duplicated");
    }
    auto packed_sum = checked_add_u64(packed_bytes, records[index].packed_bytes);
    auto scale_sum = checked_add_u64(scale_bytes, records[index].scale_bytes);
    auto bf16_sum = checked_add_u64(bf16_bytes, records[index].bf16_bytes);
    if (!packed_sum.ok()) return packed_sum.status();
    if (!scale_sum.ok()) return scale_sum.status();
    if (!bf16_sum.ok()) return bf16_sum.status();
    packed_bytes = *packed_sum;
    scale_bytes = *scale_sum;
    bf16_bytes = *bf16_sum;
  }
  if (packed_bytes != kOfficialPackedBytes ||
      scale_bytes != kOfficialScaleBytes || bf16_bytes != kOfficialBf16Bytes) {
    return Status::Internal("Qwen INT4 official linear byte totals drifted");
  }
  return QwenInt4LinearShapeLedger(std::move(records));
}

const QwenInt4LinearRecordPlan* QwenInt4LinearShapeLedger::find_first(
    QwenInt4LinearShapeFamily family) const noexcept {
  const auto found = std::find_if(records_.begin(), records_.end(),
                                  [family](const auto& record) {
                                    return record.family == family;
                                  });
  return found == records_.end() ? nullptr : &*found;
}

Result<Sha256Digest> QwenInt4LinearShapeLedger::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain =
      "pih.qwen3_int4_linear_shape_ledger.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = update_u64(digest, records_.size());
  for (const auto& record : records_) {
    if (status.ok()) status = update_u64(digest, static_cast<std::uint8_t>(record.family));
    if (status.ok()) status = update_string(digest, record.source_name);
    if (status.ok()) status = update_string(digest, record.values_name);
    if (status.ok()) status = update_string(digest, record.scales_name);
    for (const auto value : {record.rows, record.columns, record.groups_per_row,
                             record.packed_bytes, record.scale_bytes,
                             record.bf16_bytes, record.override_added_bytes}) {
      if (status.ok()) status = update_u64(digest, value);
    }
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

}  // namespace pih
