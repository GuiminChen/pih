#include "disposition_records.h"
#include "tensor_inventory.h"
#include "pih/core/canonical_hash.h"
#include <map>

namespace pih::offline_deepseek {
namespace {
Status Text(CanonicalHashBuilder& hash, std::uint16_t id, std::string_view value) {
  return hash.add_bytes(id, std::as_bytes(std::span(value)));
}
Result<std::string_view> DTypeText(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return std::string_view("F32");
    case DType::kFloat16: return std::string_view("F16");
    case DType::kBFloat16: return std::string_view("BF16");
    case DType::kInt8: return std::string_view("I8");
    case DType::kUInt8: return std::string_view("U8");
    case DType::kInt32: return std::string_view("I32");
    case DType::kInt64: return std::string_view("I64");
    case DType::kBool: return std::string_view("BOOL");
    case DType::kFloat8E4M3: return std::string_view("F8_E4M3");
    case DType::kFloat8E8M0: return std::string_view("F8_E8M0");
    default: return Status::InvalidArgument("disposition checkpoint dtype unsupported");
  }
}
}  // namespace
Result<Pp1DispositionRecords> BuildPp1DispositionRecords(std::span<const ObservedSourceTensor> source) {
  auto names = ExpectedTensorNames(true);
  if (!names.ok()) return names.status();
  if (source.size() != names->size())
    return Status::InvalidArgument("disposition requires complete sorted source inventory");
  Pp1DispositionRecords result;
  std::uint64_t selected_bytes = 0, excluded_bytes = 0;
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto& item = source[i];
    const auto& tensor = item.tensor;
    if (tensor.name != (*names)[i] || item.source_record_root == Sha256Digest{} ||
        item.payload_record_root == Sha256Digest{} || tensor.source.payload_sha256 == Sha256Digest{} ||
        tensor.source.source_fd < 0 || tensor.shape.empty() || tensor.shape.size() > 8)
      return Status::InvalidArgument("disposition source identity or roots invalid");
    auto dtype = DTypeText(tensor.dtype);
    if (!dtype.ok()) return dtype.status();
    auto geometry = ValidateSourceTensorGeometry(tensor.name, tensor.dtype, tensor.shape);
    if (!geometry.ok()) return geometry;
    auto bytes = dtype_size(tensor.dtype);
    if (!bytes.ok()) return bytes.status();
    for (auto dimension : tensor.shape) {
      if (dimension == 0 || dimension > 166'878'536'440ULL / *bytes)
        return Status::InvalidArgument("disposition shape outside byte budget");
      *bytes *= dimension;
    }
    if (*bytes != tensor.source.bytes)
      return Status::InvalidArgument("disposition source byte size differs from shape");
    const bool excluded = tensor.name.starts_with("mtp.");
    const auto space = (excluded || tensor.name.starts_with("layers."))
        ? tensor.name.substr(0, tensor.name.find('.', tensor.name.find('.') + 1))
        : std::string("endpoint");
    Sha256Digest logical;
    if (!excluded) {
      std::vector<std::byte> packed(tensor.shape.size() * 8);
      for (std::size_t d = 0; d < tensor.shape.size(); ++d)
        for (unsigned b = 0; b < 8; ++b)
          packed[d * 8 + b] = static_cast<std::byte>((tensor.shape[d] >> (b * 8)) & 255);
      auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-logical-record:v1", 10);
      if (!hash.ok()) return hash.status();
      auto status = hash->add_hash(1, item.source_record_root);
      if (status.ok()) status = hash->add_hash(2, item.payload_record_root);
      if (status.ok()) status = hash->add_hash(3, tensor.source.payload_sha256);
      if (status.ok()) status = Text(*hash, 4, tensor.name);
      if (status.ok()) status = Text(*hash, 5, *dtype);
      if (status.ok()) status = hash->add_u32(6, static_cast<std::uint32_t>(tensor.shape.size()));
      if (status.ok()) status = hash->add_bytes(7, packed);
      if (status.ok()) status = Text(*hash, 8, space);
      if (status.ok()) status = hash->add_u64(9, *bytes);
      if (status.ok()) status = Text(*hash, 10, "identity_copy_v1");
      if (!status.ok()) return status;
      auto root = hash->finalize();
      if (!root.ok()) return root.status();
      logical = *root;
    }
    auto hash = CanonicalHashBuilder::Create(excluded
        ? "pih:deepseek-v4-flash-0731-excluded-disposition:v1"
        : "pih:deepseek-v4-flash-0731-selected-disposition:v1", excluded ? 8 : 9);
    if (!hash.ok()) return hash.status();
    auto status = hash->add_hash(1, item.source_record_root);
    if (status.ok()) status = hash->add_hash(2, item.payload_record_root);
    if (status.ok()) status = hash->add_hash(3, excluded ? tensor.source.payload_sha256 : logical);
    if (status.ok()) status = Text(*hash, 4, tensor.name);
    if (status.ok()) status = Text(*hash, 5, space);
    if (status.ok()) status = hash->add_u64(6, *bytes);
    if (status.ok()) status = Text(*hash, 7, excluded ? "explicit_excluded" : "identity_copy");
    if (status.ok()) status = Text(*hash, 8, excluded ? "dspark_disabled" : "identity_copy_v1");
    if (status.ok() && !excluded) status = hash->add_u32(9, 0);
    if (!status.ok()) return status;
    auto disposition = hash->finalize();
    if (!disposition.ok()) return disposition.status();
    if (excluded) {
      excluded_bytes += *bytes;
      result.excluded_dispositions.push_back({tensor.name, *disposition});
    } else {
      selected_bytes += *bytes;
      result.selected.push_back(tensor);
      result.layout_authorities.push_back({tensor.name, item.payload_record_root, logical});
      result.selected_dispositions.push_back({tensor.name, *disposition});
    }
  }
  if (selected_bytes != 156'015'698'140ULL || excluded_bytes != 10'862'838'300ULL ||
      result.selected.size() != 67'612 || result.excluded_dispositions.size() != 4'705)
    return Status::InvalidArgument("disposition selected/excluded ledgers differ from frozen PP1");
  auto layout = BuildGenerationLayout(result.selected);
  if (!layout.ok()) return layout.status();
  return result;
}
Result<Pp1Disposition> BuildPp1Disposition(std::span<const ObservedSourceTensor> source,
    const Sha256Digest& inventory_root, const Sha256Digest& payload_root) {
  if (inventory_root == Sha256Digest{} || payload_root == Sha256Digest{})
    return Status::InvalidArgument("disposition aggregate antecedents missing");
  auto records = BuildPp1DispositionRecords(source);
  if (!records.ok()) return records.status();
  auto set_root = [](std::string_view domain, const std::vector<Sha256Digest>& roots)
      -> Result<Sha256Digest> {
    if (roots.size() > 4096) return Status::InvalidArgument("disposition root set exceeds namespace bound");
    auto hash = CanonicalHashBuilder::Create(domain, static_cast<std::uint32_t>(roots.size() + 1));
    if (!hash.ok()) return hash.status();
    auto status = hash->add_u32(1, static_cast<std::uint32_t>(roots.size()));
    for (std::size_t i = 0; status.ok() && i < roots.size(); ++i)
      status = hash->add_hash(static_cast<std::uint16_t>(100 + i), roots[i]);
    if (!status.ok()) return status;
    return hash->finalize();
  };
  struct Group {
    std::vector<Sha256Digest> all, target, excluded, owner;
    std::uint64_t selected_bytes = 0, excluded_bytes = 0;
  };
  std::map<std::string, Group> groups;
  std::size_t selected = 0, excluded = 0;
  for (const auto& item : source) {
    const auto& name = item.tensor.name;
    const bool omit = name.starts_with("mtp.");
    const auto space = (omit || name.starts_with("layers."))
        ? name.substr(0, name.find('.', name.find('.') + 1)) : std::string("endpoint");
    auto& group = groups[space];
    if (omit) {
      const auto root = records->excluded_dispositions[excluded++].disposition_record_root;
      group.all.push_back(root); group.excluded.push_back(root);
      group.excluded_bytes += item.tensor.source.bytes;
    } else {
      const auto root = records->selected_dispositions[selected].disposition_record_root;
      group.all.push_back(root); group.owner.push_back(root);
      group.target.push_back(records->layout_authorities[selected++].target_logical_root);
      group.selected_bytes += item.tensor.source.bytes;
    }
  }
  std::vector<std::string> namespaces{"endpoint"};
  for (unsigned i = 0; i < 43; ++i) namespaces.push_back("layers." + std::to_string(i));
  for (unsigned i = 0; i < 3; ++i) namespaces.push_back("mtp." + std::to_string(i));
  Pp1Disposition result;
  std::vector<Sha256Digest> owner_namespaces;
  for (const auto& space : namespaces) {
    const auto& group = groups.at(space);
    auto all = set_root("pih:deepseek-v4-flash-0731-source-disposition-set:v1", group.all);
    auto target = set_root("pih:deepseek-v4-flash-0731-target-record-set:v1", group.target);
    auto omitted = set_root("pih:deepseek-v4-flash-0731-excluded-record-set:v1", group.excluded);
    auto owned = set_root("pih:deepseek-v4-flash-0731-owner-record-set:v1", group.owner);
    if (!all.ok()) return all.status();
    if (!target.ok()) return target.status();
    if (!omitted.ok()) return omitted.status();
    if (!owned.ok()) return owned.status();
    auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-disposition-namespace:v1", 10);
    if (!hash.ok()) return hash.status();
    auto status = Text(*hash, 1, space);
    if (status.ok()) status = hash->add_u32(2, static_cast<std::uint32_t>(group.all.size()));
    if (status.ok()) status = hash->add_u64(3, group.selected_bytes + group.excluded_bytes);
    if (status.ok()) status = hash->add_u32(4, static_cast<std::uint32_t>(group.target.size()));
    if (status.ok()) status = hash->add_u64(5, group.selected_bytes);
    if (status.ok()) status = hash->add_u32(6, static_cast<std::uint32_t>(group.excluded.size()));
    if (status.ok()) status = hash->add_u64(7, group.excluded_bytes);
    if (status.ok()) status = hash->add_hash(8, *all);
    if (status.ok()) status = hash->add_hash(9, *target);
    if (status.ok()) status = hash->add_hash(10, *omitted);
    if (!status.ok()) return status;
    auto root = hash->finalize();
    if (!root.ok()) return root.status();
    result.namespace_roots.push_back(*root);
    hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-owner-namespace:v1", 5);
    if (!hash.ok()) return hash.status();
    status = hash->add_u32(1, 0);
    if (status.ok()) status = Text(*hash, 2, space);
    if (status.ok()) status = hash->add_u32(3, static_cast<std::uint32_t>(group.owner.size()));
    if (status.ok()) status = hash->add_u64(4, group.selected_bytes);
    if (status.ok()) status = hash->add_hash(5, *owned);
    if (!status.ok()) return status;
    root = hash->finalize();
    if (!root.ok()) return root.status();
    owner_namespaces.push_back(*root);
  }
  auto summary = set_root("pih:deepseek-v4-flash-0731-owner-namespace-summary:v1", owner_namespaces);
  if (!summary.ok()) return summary.status();
  auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-owner:v1", 9);
  if (!hash.ok()) return hash.status();
  auto status = hash->add_u32(1, 0);
  if (status.ok()) status = hash->add_u32(2, 0);
  if (status.ok()) status = hash->add_u32(3, 42);
  if (status.ok()) status = hash->add_u32(4, 1);
  if (status.ok()) status = hash->add_u32(5, 1);
  if (status.ok()) status = hash->add_u32(6, 0);
  if (status.ok()) status = hash->add_u32(7, 67'612);
  if (status.ok()) status = hash->add_u64(8, 156'015'698'140ULL);
  if (status.ok()) status = hash->add_hash(9, *summary);
  if (!status.ok()) return status;
  auto owner = hash->finalize();
  if (!owner.ok()) return owner.status();
  hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-disposition:v1", 64);
  if (!hash.ok()) return hash.status();
  status = Text(*hash, 1, "deepseek_v4_flash_0731_target_disposition_v1");
  if (status.ok()) status = Text(*hash, 2, "deepseek_v4_flash_0731");
  if (status.ok()) status = hash->add_hash(3, inventory_root);
  if (status.ok()) status = hash->add_hash(4, payload_root);
  if (status.ok()) status = hash->add_u32(5, 0);
  if (status.ok()) status = hash->add_u32(6, 1);
  if (status.ok()) status = hash->add_u32(7, 72'317);
  if (status.ok()) status = hash->add_u64(8, 166'878'536'440ULL);
  if (status.ok()) status = hash->add_u32(9, 67'612);
  if (status.ok()) status = hash->add_u64(10, 156'015'698'140ULL);
  if (status.ok()) status = hash->add_u32(11, 4'705);
  if (status.ok()) status = hash->add_u64(12, 10'862'838'300ULL);
  if (status.ok()) status = hash->add_u32(13, 47);
  if (status.ok()) status = hash->add_u32(14, 1);
  if (status.ok()) status = Text(*hash, 15, "total_source_to_target_logical_disposition_non_authorizing");
  if (status.ok()) status = Text(*hash, 16, "hardware_evidence_open");
  for (std::size_t i = 0; status.ok() && i < result.namespace_roots.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), result.namespace_roots[i]);
  if (status.ok()) status = hash->add_hash(200, *owner);
  if (!status.ok()) return status;
  auto disposition = hash->finalize();
  if (!disposition.ok()) return disposition.status();
  result.layout_authority = {inventory_root, payload_root, *disposition, *owner};
  result.records = std::move(*records);
  return result;
}
}  // namespace pih::offline_deepseek
