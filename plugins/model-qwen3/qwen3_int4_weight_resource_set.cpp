#include "pih/model/qwen3_int4_weight_resource_set.h"

#include <algorithm>

#include <array>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

const QwenInt4LinearRecordPlan* find_linear_record(
    const QwenInt4LinearShapeLedger& ledger, std::string_view identity) {
  for (const auto& record : ledger.records()) {
    if (record.values_name == identity || record.scales_name == identity)
      return &record;
  }
  return nullptr;
}

Result<TensorView> make_view(const QwenInt4ArtifactRecordPlan& record,
                            const QwenInt4LinearRecordPlan* linear,
                            std::uintptr_t address, Device device,
                            std::uint64_t generation) {
  DType dtype = DType::kUInt8;
  std::array<std::int64_t,2> shape{};
  std::size_t rank = 0;
  switch (record.kind) {
    case QwenInt4ArtifactRecordKind::kPackedValues:
      if (linear == nullptr) break;
      dtype = DType::kUInt8;
      shape = {static_cast<std::int64_t>(linear->rows),
               static_cast<std::int64_t>(linear->columns / 2)};
      rank = 2;
      break;
    case QwenInt4ArtifactRecordKind::kFp16Scales:
      if (linear == nullptr) break;
      dtype = DType::kFloat16;
      shape = {static_cast<std::int64_t>(linear->rows),
               static_cast<std::int64_t>(linear->groups_per_row)};
      rank = 2;
      break;
    case QwenInt4ArtifactRecordKind::kBf16Embedding:
      dtype = DType::kBFloat16;
      shape = {151936,1024};
      rank = 2;
      break;
    case QwenInt4ArtifactRecordKind::kBf16Norm:
      dtype = DType::kBFloat16;
      shape = {static_cast<std::int64_t>(record.logical_bytes / 2),0};
      rank = 1;
      break;
    case QwenInt4ArtifactRecordKind::kAlias:
      break;
  }
  if (rank == 0) {
    return Status::InvalidArgument("Qwen INT4 record has no runtime shape");
  }
  return TensorView::Create(reinterpret_cast<void*>(address), dtype,
                            std::span<const std::int64_t>(shape.data(),rank),
                            {}, device, generation);
}

}  // namespace

Result<QwenInt4WeightResourceSet> QwenInt4WeightResourceSet::Create(
    const QwenInt4ArtifactLayout& layout,
    const QwenInt4LinearShapeLedger& ledger,
    const TensorView& canonical_pool, std::int32_t owning_rank) {
  if (owning_rank < 0 || canonical_pool.dtype() != DType::kUInt8 ||
      canonical_pool.device().type() != DeviceType::kCuda ||
      canonical_pool.device().index() != owning_rank || canonical_pool.generation() == 0 ||
      canonical_pool.rank() != 1 || canonical_pool.stride(0) != 1 ||
      canonical_pool.byte_span() != layout.logical_payload_bytes() ||
      layout.logical_payload_bytes() !=
          QwenInt4ArtifactLayout::kOfficialLogicalPayloadBytes ||
      layout.records().size() != QwenInt4ArtifactLayout::kOfficialRecordCount ||
      ledger.records().size() != QwenInt4LinearShapeLedger::kOfficialRecordCount) {
    return Status::InvalidArgument("Qwen INT4 canonical pool contract mismatch");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(canonical_pool.data());
  if (base == 0 || base % 256 != 0) {
    return Status::InvalidArgument("Qwen INT4 canonical pool is misaligned");
  }
  std::vector<NamedView> views;
  views.reserve(layout.records().size());
  std::uint64_t compact_offset = 0;
  std::size_t payload_count = 0;
  for (const auto& record : layout.records()) {
    if (record.kind == QwenInt4ArtifactRecordKind::kAlias) continue;
    if (record.logical_bytes == 0 || record.logical_bytes % 256 != 0 ||
        compact_offset > UINTPTR_MAX - base) {
      return Status::InvalidArgument("Qwen INT4 compact extent is invalid");
    }
    const auto* linear = find_linear_record(ledger, record.identity);
    auto tensor = make_view(record, linear, base + compact_offset,
                            canonical_pool.device(),
                            canonical_pool.generation());
    if (!tensor.ok() || tensor->byte_span() != record.logical_bytes) {
      return Status::InvalidArgument("Qwen INT4 runtime tensor shape drifted");
    }
    views.push_back({record.identity,*tensor});
    auto next = checked_add_u64(compact_offset,record.logical_bytes);
    if (!next.ok()) return next.status();
    compact_offset = *next;
    ++payload_count;
  }
  for (const auto& record : layout.records()) {
    if (record.kind != QwenInt4ArtifactRecordKind::kAlias) continue;
    const auto owner = std::find_if(views.begin(),views.end(),[&](const auto& v){
      return v.name == record.alias_owner;
    });
    if (owner == views.end()) {
      return Status::InvalidArgument("Qwen INT4 alias owner is missing");
    }
    views.push_back({record.identity,owner->view});
  }
  if (compact_offset != layout.logical_payload_bytes() ||
      payload_count != layout.payload_record_count() ||
      views.size() != layout.records().size()) {
    return Status::InvalidArgument("Qwen INT4 compact pool totals drifted");
  }
  std::vector<NamedLinear> linears;
  linears.reserve(ledger.records().size());
  for (const auto& record : ledger.records()) {
    auto packed = std::find_if(views.begin(),views.end(),[&](const auto& v){
      return v.name == record.values_name;
    });
    auto scales = std::find_if(views.begin(),views.end(),[&](const auto& v){
      return v.name == record.scales_name;
    });
    if (packed == views.end() || scales == views.end()) {
      return Status::InvalidArgument("Qwen INT4 linear pair is incomplete");
    }
    linears.push_back({record.source_name,
                       {record.family,packed->view,scales->view}});
  }
  return QwenInt4WeightResourceSet(
      std::move(views),std::move(linears),payload_count,compact_offset,
      owning_rank,canonical_pool.device().index());
}

Result<TensorView> QwenInt4WeightResourceSet::view(
    std::string_view identity) const {
  for (const auto& entry : views_) if (entry.name == identity) return entry.view;
  return Status::InvalidArgument("Qwen INT4 weight identity is not bound");
}

Result<QwenInt4LinearResources> QwenInt4WeightResourceSet::linear(
    std::string_view source_name) const {
  for (const auto& entry : linears_)
    if (entry.source_name == source_name) return entry.resources;
  return Status::InvalidArgument("Qwen INT4 linear source is not bound");
}

}  // namespace pih
