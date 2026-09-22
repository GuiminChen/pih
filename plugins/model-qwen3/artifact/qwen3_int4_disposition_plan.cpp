#include "pih/model/qwen3_int4_disposition_plan.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <span>
#include <string_view>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_int4_artifact_layout.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"
#include "pih/model/qwen3_manifest.h"

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

Result<std::uint64_t> bf16_bytes(const std::vector<std::uint64_t>& shape) {
  std::uint64_t elements = 1;
  for (const auto extent : shape) {
    auto product = checked_mul_u64(elements, extent);
    if (!product.ok()) return product.status();
    elements = *product;
  }
  return checked_mul_u64(elements, std::uint64_t{2});
}

}  // namespace

Result<QwenInt4DispositionPlan>
QwenInt4DispositionPlan::CreateOfficialPureW4() {
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  if (!ledger.ok()) return ledger.status();
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  if (!layout.ok()) return layout.status();

  std::map<std::string, const QwenInt4LinearRecordPlan*> linears;
  for (const auto& record : ledger->records()) {
    linears.emplace(record.source_name, &record);
  }
  std::vector<QwenInt4SourceDisposition> records;
  records.reserve(kOfficialSourceCount);
  std::uint64_t source_total = 0;
  for (const auto& expected : Qwen3Manifest::expected_tensors()) {
    auto bytes = bf16_bytes(expected.shape);
    if (!bytes.ok()) return bytes.status();
    auto total = checked_add_u64(source_total, *bytes);
    if (!total.ok()) return total.status();
    source_total = *total;

    if (const auto found = linears.find(expected.name); found != linears.end()) {
      records.push_back({expected.name, expected.shape, *bytes,
                         QwenInt4DispositionKind::kQuantizeW4A16G128,
                         {found->second->values_name,
                          found->second->scales_name},
                         {}});
    } else if (expected.name == "lm_head.weight") {
      records.push_back({expected.name, expected.shape, *bytes,
                         QwenInt4DispositionKind::kEqualSourceDedupToAlias,
                         {"lm_head.weight"}, "model.embed_tokens.weight"});
    } else {
      records.push_back({expected.name, expected.shape, *bytes,
                         QwenInt4DispositionKind::kIdentityCopyBf16,
                         {expected.name}, {}});
    }
  }
  std::sort(records.begin(), records.end(), [](const auto& left,
                                                const auto& right) {
    return left.source_name < right.source_name;
  });

  std::set<std::string> targets;
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (index != 0 && records[index - 1].source_name == records[index].source_name) {
      return Status::Internal("Qwen INT4 source disposition is duplicated");
    }
    for (const auto& target : records[index].target_identities) {
      if (!targets.insert(target).second) {
        return Status::Internal("Qwen INT4 target derivation is duplicated");
      }
    }
  }
  std::set<std::string> layout_targets;
  for (const auto& record : layout->records()) layout_targets.insert(record.identity);
  if (records.size() != kOfficialSourceCount ||
      targets.size() != kOfficialTargetCount || targets != layout_targets ||
      source_total != Qwen3Manifest::kOfficialSourcePayloadBytes) {
    return Status::Internal("Qwen INT4 disposition coverage drifted");
  }
  return QwenInt4DispositionPlan(std::move(records));
}

const QwenInt4SourceDisposition* QwenInt4DispositionPlan::find(
    std::string_view source_name) const noexcept {
  const auto found = std::lower_bound(
      records_.begin(), records_.end(), source_name,
      [](const auto& record, std::string_view name) {
        return record.source_name < name;
      });
  return found == records_.end() || found->source_name != source_name
             ? nullptr
             : &*found;
}

Result<Sha256Digest> QwenInt4DispositionPlan::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain =
      "pih.qwen3_int4_disposition_plan.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = update_u64(digest, records_.size());
  for (const auto& record : records_) {
    if (status.ok()) status = update_string(digest, record.source_name);
    if (status.ok()) status = update_u64(digest, record.source_shape.size());
    for (const auto extent : record.source_shape) {
      if (status.ok()) status = update_u64(digest, extent);
    }
    if (status.ok()) status = update_u64(digest, record.source_bytes);
    if (status.ok()) status = update_u64(digest, static_cast<std::uint8_t>(record.kind));
    if (status.ok()) status = update_u64(digest, record.target_identities.size());
    for (const auto& target : record.target_identities) {
      if (status.ok()) status = update_string(digest, target);
    }
    if (status.ok()) status = update_string(digest, record.equality_owner_source);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

}  // namespace pih
