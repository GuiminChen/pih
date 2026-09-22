#include "source_inventory.h"
#include "pih/core/canonical_hash.h"
#include <algorithm>
#include <map>

namespace pih::offline_deepseek {
namespace {
Status Text(CanonicalHashBuilder& hash, std::uint16_t id, std::string_view value) {
  return hash.add_bytes(id, std::as_bytes(std::span(value)));
}
Result<std::string_view> DTypeText(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return std::string_view("F32");
    case DType::kBFloat16: return std::string_view("BF16");
    case DType::kInt8: return std::string_view("I8");
    case DType::kInt64: return std::string_view("I64");
    case DType::kFloat8E4M3: return std::string_view("F8_E4M3");
    case DType::kFloat8E8M0: return std::string_view("F8_E8M0");
    default: return Status::InvalidArgument("source inventory dtype outside frozen geometry");
  }
}
}  // namespace
Result<SourceInventoryShards> CompileSourceInventoryShards(const SourceArtifact& source) {
  auto semantics = CompileSourceSemantics(source);
  if (!semantics.ok()) return semantics.status();
  SourceInventoryShards result;
  result.semantic_root = semantics->semantic_root;
  result.shards.reserve(48);
  for (std::size_t i = 0; i < semantics->shards.size(); ++i) {
    const auto& semantic = semantics->shards[i];
    SourcePayloadShardInput shard;
    shard.descriptor = source.descriptors()[i + 2];
    shard.name = semantic.name;
    shard.artifact_object_root = source.objects()[i + 2].object_root;
    shard.tensor_bytes = semantic.data_bytes;
    shard.tensors.reserve(semantic.tensors.size());
    auto set = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-shard-record-set:v1",
        static_cast<std::uint32_t>(semantic.tensors.size() + 1));
    if (!set.ok()) return set.status();
    auto status = set->add_u32(1, static_cast<std::uint32_t>(semantic.tensors.size()));
    if (!status.ok()) return status;
    for (std::size_t t = 0; t < semantic.tensors.size(); ++t) {
      const auto& item = semantic.tensors[t];
      const auto& tensor = item.tensor;
      const bool indexed = tensor.name.starts_with("layers.") || tensor.name.starts_with("mtp.");
      const auto space = indexed ? tensor.name.substr(0, tensor.name.find('.', tensor.name.find('.') + 1))
                                 : std::string("endpoint");
      const auto suffix = indexed ? std::string_view(tensor.name).substr(space.size() + 1)
                                  : std::string_view(tensor.name);
      auto dtype = DTypeText(tensor.dtype);
      if (!dtype.ok()) return dtype.status();
      std::vector<std::byte> shape(tensor.shape.size() * 8);
      for (std::size_t d = 0; d < tensor.shape.size(); ++d)
        for (unsigned b = 0; b < 8; ++b)
          shape[d * 8 + b] = static_cast<std::byte>((tensor.shape[d] >> (b * 8)) & 255);
      auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-tensor:v1", 12);
      if (!hash.ok()) return hash.status();
      status = Text(*hash, 1, tensor.name);
      if (status.ok()) status = Text(*hash, 2, shard.name);
      if (status.ok()) status = Text(*hash, 3, *dtype);
      if (status.ok()) status = hash->add_u32(4, static_cast<std::uint32_t>(tensor.shape.size()));
      if (status.ok()) status = hash->add_bytes(5, shape);
      if (status.ok()) status = hash->add_u64(6, tensor.file_begin - semantic.header_bytes - 8);
      if (status.ok()) status = hash->add_u64(7, tensor.file_end - semantic.header_bytes - 8);
      if (status.ok()) status = hash->add_hash(8, item.semantic_root);
      if (status.ok()) status = Text(*hash, 9, space);
      if (status.ok()) status = Text(*hash, 10, suffix);
      if (status.ok()) status = hash->add_u64(11, tensor.file_begin);
      if (status.ok()) status = hash->add_u64(12, tensor.file_end);
      if (!status.ok()) return status;
      auto root = hash->finalize();
      if (!root.ok()) return root.status();
      status = set->add_hash(static_cast<std::uint16_t>(100 + t), *root);
      if (!status.ok()) return status;
      shard.tensors.push_back({tensor.name, *root, tensor.dtype, tensor.shape, tensor.file_begin, tensor.file_end});
    }
    auto record_set = set->finalize();
    if (!record_set.ok()) return record_set.status();
    auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-shard-inventory:v1", 8);
    if (!hash.ok()) return hash.status();
    status = Text(*hash, 1, shard.name);
    if (status.ok()) status = hash->add_hash(2, semantic.shard_root);
    if (status.ok()) status = hash->add_u32(3, static_cast<std::uint32_t>(shard.tensors.size()));
    if (status.ok()) status = hash->add_u64(4, shard.tensor_bytes);
    if (status.ok()) status = hash->add_hash(5, *record_set);
    if (status.ok()) status = hash->add_u64(6, source.objects()[i + 2].bytes);
    if (status.ok()) status = hash->add_u64(7, semantic.header_bytes);
    if (status.ok()) status = hash->add_u64(8, semantic.data_bytes);
    if (!status.ok()) return status;
    auto root = hash->finalize();
    if (!root.ok()) return root.status();
    shard.inventory_root = *root;
    result.shards.push_back(std::move(shard));
  }
  auto status = source.Revalidate();
  if (!status.ok()) return status;
  return result;
}
Result<SourceInventory> CompileSourceInventory(const SourceArtifact& source) {
  auto plan = CompileSourceInventoryShards(source);
  if (!plan.ok()) return plan.status();
  struct Namespace {
    std::string name;
    std::size_t count;
    std::uint64_t bytes;
  };
  std::vector<Namespace> namespaces{{"endpoint", 6, 2'118'393'876ULL}};
  for (unsigned i = 0; i < 43; ++i) {
    const std::size_t count = i <= 1 ? 1565 : i % 2 ? 1569 : 1576;
    const std::uint64_t bytes = i <= 1 ? 3'566'148'952ULL : i == 2 ? 3'596'055'640ULL :
        i % 2 ? 3'568'596'312ULL : 3'589'851'224ULL;
    namespaces.push_back({"layers." + std::to_string(i), count, bytes});
  }
  namespaces.push_back({"mtp.0", 1568, 3'610'287'448ULL});
  namespaces.push_back({"mtp.1", 1565, 3'559'944'536ULL});
  namespaces.push_back({"mtp.2", 1572, 3'692'606'316ULL});
  std::map<std::string, std::vector<const SourceTensorAuthority*>> grouped;
  for (const auto& shard : plan->shards) {
    for (const auto& tensor : shard.tensors) {
      const auto& name = tensor.name;
      const bool indexed = name.starts_with("layers.") || name.starts_with("mtp.");
      const auto space = indexed ? name.substr(0, name.find('.', name.find('.') + 1)) : std::string("endpoint");
      grouped[space].push_back(&tensor);
    }
  }
  auto grammar = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-name-grammar:v1", 50);
  auto summary = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-namespace-set:v1", 48);
  if (!grammar.ok()) return grammar.status();
  if (!summary.ok()) return summary.status();
  auto status = Text(*grammar, 1, "deepseek_v4_flash_0731_source_inventory_v1");
  if (status.ok()) status = Text(*grammar, 2, "deepseek_v4_flash_0731");
  if (status.ok()) status = grammar->add_u32(3, 47);
  if (status.ok()) status = summary->add_u32(1, 47);
  if (!status.ok()) return status;
  for (std::size_t i = 0; i < namespaces.size(); ++i) {
    const auto& expected = namespaces[i];
    auto& records = grouped[expected.name];
    std::sort(records.begin(), records.end(), [](auto* a, auto* b) { return a->name < b->name; });
    std::uint64_t bytes = 0;
    for (const auto* record : records) bytes += record->file_end - record->file_begin;
    if (records.size() != expected.count || bytes != expected.bytes)
      return Status::InvalidArgument("source inventory namespace differs from frozen geometry");
    auto names = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-name-group:v1",
        static_cast<std::uint32_t>(2 + records.size()));
    if (!names.ok()) return names.status();
    status = Text(*names, 1, expected.name);
    if (status.ok()) status = names->add_u32(2, static_cast<std::uint32_t>(records.size()));
    if (!status.ok()) return status;
    for (std::size_t j = 0; j < records.size(); ++j) {
      auto name = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-name:v1", 1);
      if (!name.ok()) return name.status();
      status = Text(*name, 1, records[j]->name);
      if (!status.ok()) return status;
      auto root = name->finalize();
      if (!root.ok()) return root.status();
      status = names->add_hash(static_cast<std::uint16_t>(100 + j), *root);
      if (!status.ok()) return status;
    }
    auto name_root = names->finalize();
    if (!name_root.ok()) return name_root.status();
    status = grammar->add_hash(static_cast<std::uint16_t>(100 + i), *name_root);
    if (!status.ok()) return status;
    auto space = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-namespace:v1",
        static_cast<std::uint32_t>(4 + records.size()));
    if (!space.ok()) return space.status();
    status = Text(*space, 1, expected.name);
    if (status.ok()) status = space->add_u32(2, static_cast<std::uint32_t>(records.size()));
    if (status.ok()) status = space->add_u64(3, bytes);
    if (status.ok()) status = space->add_hash(4, *name_root);
    for (std::size_t j = 0; status.ok() && j < records.size(); ++j)
      status = space->add_hash(static_cast<std::uint16_t>(100 + j), records[j]->source_record_root);
    if (!status.ok()) return status;
    auto root = space->finalize();
    if (!root.ok()) return root.status();
    status = summary->add_hash(static_cast<std::uint16_t>(100 + i), *root);
    if (!status.ok()) return status;
  }
  auto grammar_root = grammar->finalize();
  auto namespace_root = summary->finalize();
  if (!grammar_root.ok()) return grammar_root.status();
  if (!namespace_root.ok()) return namespace_root.status();
  auto pinned_grammar = Sha256Digest::ParseHex("cebbc15b457d8302b6bd5a39abf72ee71d897f52eb394b5994cf6daf5fb9364f");
  if (!pinned_grammar.ok()) return pinned_grammar.status();
  if (*grammar_root != *pinned_grammar)
    return Status::FailedPrecondition("source name grammar differs from pinned root");
  auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-inventory:v1", 65);
  if (!hash.ok()) return hash.status();
  status = Text(*hash, 1, "deepseek_v4_flash_0731_source_inventory_v1");
  if (status.ok()) status = Text(*hash, 2, "deepseek_v4_flash_0731");
  if (status.ok()) status = hash->add_hash(3, source.model_digest());
  if (status.ok()) status = hash->add_hash(4, plan->semantic_root);
  if (status.ok()) status = hash->add_hash(5, *grammar_root);
  if (status.ok()) status = hash->add_hash(6, *namespace_root);
  if (status.ok()) status = hash->add_u32(7, 48);
  if (status.ok()) status = hash->add_u64(8, 72'317);
  if (status.ok()) status = hash->add_u64(9, 166'878'536'440ULL);
  if (status.ok()) status = hash->add_u32(10, 6);
  if (status.ok()) status = hash->add_u64(11, 2'118'393'876ULL);
  if (status.ok()) status = hash->add_u64(12, 67'606);
  if (status.ok()) status = hash->add_u64(13, 153'897'304'264ULL);
  if (status.ok()) status = hash->add_u32(14, 4'705);
  if (status.ok()) status = hash->add_u64(15, 10'862'838'300ULL);
  if (status.ok()) status = Text(*hash, 16, "exact_pinned_source_name_set_and_header_semantics_non_authorizing");
  if (status.ok()) status = Text(*hash, 17, "hardware_evidence_open");
  for (std::size_t i = 0; status.ok() && i < plan->shards.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), plan->shards[i].inventory_root);
  if (!status.ok()) return status;
  auto root = hash->finalize();
  if (!root.ok()) return root.status();
  status = source.Revalidate();
  if (!status.ok()) return status;
  return SourceInventory{std::move(*plan), *grammar_root, *namespace_root, *root};
}
}  // namespace pih::offline_deepseek
