#include "shard_layout.h"
#include <algorithm>
#include <array>
#include <charconv>
#include "pih/core/canonical_json.h"
#include "pih/core/canonical_hash.h"
#include "pih/model/safetensors_header.h"

namespace pih::offline_deepseek {
namespace {
Result<std::string> DTypeName(DType type) {
  switch (type) {
    case DType::kFloat32: return std::string("F32");
    case DType::kFloat16: return std::string("F16");
    case DType::kBFloat16: return std::string("BF16");
    case DType::kInt8: return std::string("I8");
    case DType::kUInt8: return std::string("U8");
    case DType::kInt32: return std::string("I32");
    case DType::kInt64: return std::string("I64");
    case DType::kBool: return std::string("BOOL");
    case DType::kFloat8E4M3: return std::string("F8_E4M3");
    case DType::kFloat8E8M0: return std::string("F8_E8M0");
    default: return Status::InvalidArgument("unsupported identity checkpoint dtype");
  }
}
bool Namespace(std::string_view value) {
  if (value == "endpoint") return true;
  if (!value.starts_with("layers.")) return false;
  auto suffix = value.substr(7);
  if (suffix.empty() || (suffix.size() > 1 && suffix.front() == '0')) return false;
  unsigned layer = 43;
  const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
  return parsed.ec == std::errc{} && parsed.ptr == suffix.data() + suffix.size() && layer < 43;
}
bool InNamespace(std::string_view name, std::string_view space) {
  if (space != "endpoint") return name.starts_with(std::string(space) + ".");
  constexpr std::array<std::string_view, 6> endpoint{
      "embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_scale", "hc_head_base"};
  return std::ranges::find(endpoint, name) != endpoint.end();
}
}  // namespace
Result<ShardLayout> BuildShardLayout(std::string_view name_space,
                                    std::span<const IdentityTensor> tensors) {
  if (!Namespace(name_space) || tensors.empty() || tensors.size() > SafetensorsHeader::kMaxTensorCount)
    return Status::InvalidArgument("identity shard namespace or tensor count invalid");
  JsonValue::Object header;
  header.emplace_back("__metadata__", JsonValue(JsonValue::Object{
      {"format", JsonValue(std::string("pt"))}}));
  ShardLayout result;
  auto suffix = std::string(name_space);
  std::replace(suffix.begin(), suffix.end(), '.', '-');
  result.member_name = "model-" + suffix + ".safetensors";
  std::uint64_t cursor = 0;
  std::string_view previous;
  constexpr std::uint64_t maximum = 512ULL << 30;
  for (const auto& tensor : tensors) {
    if (tensor.name.empty() || tensor.name.size() > 512 || tensor.name <= previous ||
        !InNamespace(tensor.name, name_space) || tensor.shape.empty() || tensor.shape.size() > 8 ||
        tensor.source.source_fd < 0 || tensor.source.payload_sha256 == Sha256Digest{})
      return Status::InvalidArgument("identity tensor order, namespace or source invalid");
    previous = tensor.name;
    auto type = DTypeName(tensor.dtype);
    auto element_bytes = dtype_size(tensor.dtype);
    if (!type.ok()) return type.status();
    if (!element_bytes.ok()) return element_bytes.status();
    auto bytes = *element_bytes;
    JsonValue::Array shape;
    for (auto dimension : tensor.shape) {
      if (dimension == 0 || dimension > maximum / bytes)
        return Status::InvalidArgument("identity tensor shape overflows its shard budget");
      bytes *= dimension;
      shape.emplace_back(static_cast<std::int64_t>(dimension));
    }
    if (bytes != tensor.source.bytes || bytes > maximum - cursor)
      return Status::InvalidArgument("identity tensor payload size differs from shape");
    const auto end = cursor + bytes;
    header.emplace_back(tensor.name, JsonValue(JsonValue::Object{
        {"dtype", JsonValue(std::move(*type))}, {"shape", JsonValue(std::move(shape))},
        {"data_offsets", JsonValue(JsonValue::Array{
            JsonValue(static_cast<std::int64_t>(cursor)), JsonValue(static_cast<std::int64_t>(end))})}}));
    result.file_ranges.emplace_back(cursor, end);
    cursor = end;
  }
  auto json = canonical_ascii_json(JsonValue(std::move(header)), SafetensorsHeader::kMaxHeaderBytes);
  if (!json.ok()) return json.status();
  const auto padded = (json->size() + 7) & ~std::size_t{7};
  if (padded > SafetensorsHeader::kMaxHeaderBytes || cursor > maximum - padded - 8)
    return Status::InvalidArgument("identity header exceeds its shard budget");
  result.header_prefix.resize(padded + 8, std::byte{' '});
  for (unsigned i = 0; i < 8; ++i)
    result.header_prefix[i] = static_cast<std::byte>((static_cast<std::uint64_t>(padded) >> (8 * i)) & 255);
  for (std::size_t i = 0; i < json->size(); ++i)
    result.header_prefix[i + 8] = static_cast<std::byte>((*json)[i]);
  result.file_bytes = cursor + result.header_prefix.size();
  for (auto& [begin, end] : result.file_ranges) {
    begin += result.header_prefix.size(); end += result.header_prefix.size();
  }
  auto parsed = SafetensorsHeader::ParsePrefix(result.header_prefix, result.file_bytes);
  if (!parsed.ok()) return parsed.status();
  auto digest = sha256(result.header_prefix);
  if (!digest.ok()) return digest.status();
  result.header_sha256 = *digest;
  return result;
}
Result<CopyReceipt> CopyPlannedIdentityShard(int staging_directory_fd,
    std::string_view name_space, std::span<const IdentityTensor> tensors) {
  auto layout = BuildShardLayout(name_space, tensors);
  if (!layout.ok()) return layout.status();
  std::vector<CopyRange> ranges;
  ranges.reserve(tensors.size());
  for (const auto& tensor : tensors) ranges.push_back(tensor.source);
  return CopyIdentityShard(staging_directory_fd, layout->member_name, layout->header_prefix, ranges);
}
namespace {
Status HashText(CanonicalHashBuilder& hash, std::uint16_t id, std::string_view text) {
  return hash.add_bytes(id, std::as_bytes(std::span(text)));
}
Result<Sha256Digest> HashSet(std::string_view domain, std::span<const Sha256Digest> roots) {
  if (roots.empty() || roots.size() > 4096)
    return Status::InvalidArgument("shard authority root count invalid");
  auto hash = CanonicalHashBuilder::Create(domain, static_cast<std::uint32_t>(roots.size() + 1));
  if (!hash.ok()) return hash.status();
  auto status = hash->add_u32(1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t i = 0; status.ok() && i < roots.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), roots[i]);
  if (!status.ok()) return status;
  return hash->finalize();
}
}  // namespace

Result<BoundShardLayout> BindShardLayout(std::string_view name_space,
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> authorities) {
  if (authorities.size() != tensors.size() || tensors.empty() || tensors.size() > 4096)
    return Status::InvalidArgument("shard authority cardinality invalid");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (authorities[i].name != tensors[i].name ||
        authorities[i].source_payload_record_root == Sha256Digest{} ||
        authorities[i].target_logical_root == Sha256Digest{})
      return Status::InvalidArgument("shard authority name join or root invalid");
  }
  auto layout = BuildShardLayout(name_space, tensors);
  if (!layout.ok()) return layout.status();
  BoundShardLayout result;
  result.layout = std::move(*layout);
  const auto& bytes = result.layout;
  const auto payload_begin = static_cast<std::uint64_t>(bytes.header_prefix.size());
  std::vector<Sha256Digest> weight_roots;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const auto& authority = authorities[i];
    auto dtype = DTypeName(tensor.dtype);
    if (!dtype.ok()) return dtype.status();
    std::vector<std::byte> shape(tensor.shape.size() * 8);
    for (std::size_t dim = 0; dim < tensor.shape.size(); ++dim)
      for (unsigned b = 0; b < 8; ++b)
        shape[dim * 8 + b] = static_cast<std::byte>((tensor.shape[dim] >> (b * 8)) & 255);
    const auto [begin, end] = bytes.file_ranges[i];
    auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-layout-record:v1", 15);
    if (!hash.ok()) return hash.status();
    auto status = hash->add_hash(1, authority.source_payload_record_root);
    if (status.ok()) status = hash->add_hash(2, authority.target_logical_root);
    if (status.ok()) status = hash->add_hash(3, tensor.source.payload_sha256);
    if (status.ok()) status = HashText(*hash, 4, tensor.name);
    if (status.ok()) status = HashText(*hash, 5, *dtype);
    if (status.ok()) status = hash->add_u32(6, static_cast<std::uint32_t>(tensor.shape.size()));
    if (status.ok()) status = hash->add_bytes(7, shape);
    if (status.ok()) status = HashText(*hash, 8, name_space);
    if (status.ok()) status = HashText(*hash, 9, bytes.member_name);
    if (status.ok()) status = hash->add_u64(10, begin - payload_begin);
    if (status.ok()) status = hash->add_u64(11, end - payload_begin);
    if (status.ok()) status = hash->add_u64(12, begin);
    if (status.ok()) status = hash->add_u64(13, end);
    if (status.ok()) status = hash->add_u64(14, tensor.source.bytes);
    if (status.ok()) status = hash->add_hash(15, bytes.header_sha256);
    if (!status.ok()) return status;
    auto record = hash->finalize();
    if (!record.ok()) return record.status();
    result.layout_record_roots.push_back(*record);
    hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-weight-map-entry:v1", 3);
    if (!hash.ok()) return hash.status();
    status = HashText(*hash, 1, tensor.name);
    if (status.ok()) status = HashText(*hash, 2, bytes.member_name);
    if (status.ok()) status = hash->add_hash(3, authority.target_logical_root);
    if (!status.ok()) return status;
    auto entry = hash->finalize();
    if (!entry.ok()) return entry.status();
    weight_roots.push_back(*entry);
  }
  auto records = HashSet("pih:deepseek-v4-flash-0731-target-layout-record-set:v1", result.layout_record_roots);
  auto weights = HashSet("pih:deepseek-v4-flash-0731-target-weight-map-shard:v1", weight_roots);
  if (!records.ok()) return records.status();
  if (!weights.ok()) return weights.status();
  result.record_layout_set_root = *records;
  result.weight_map_set_root = *weights;
  auto json_end = bytes.header_prefix.size();
  while (json_end > 8 && bytes.header_prefix[json_end - 1] == std::byte{' '}) --json_end;
  const auto padding = static_cast<std::uint32_t>(bytes.header_prefix.size() - json_end);
  auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-shard-layout:v1", 12);
  if (!hash.ok()) return hash.status();
  auto status = HashText(*hash, 1, name_space);
  if (status.ok()) status = HashText(*hash, 2, bytes.member_name);
  if (status.ok()) status = hash->add_u32(3, static_cast<std::uint32_t>(tensors.size()));
  if (status.ok()) status = hash->add_u64(4, bytes.file_bytes - payload_begin);
  if (status.ok()) status = hash->add_u64(5, json_end - 8);
  if (status.ok()) status = hash->add_u32(6, padding);
  if (status.ok()) status = hash->add_u64(7, payload_begin - 8);
  if (status.ok()) status = hash->add_u64(8, payload_begin);
  if (status.ok()) status = hash->add_u64(9, bytes.file_bytes);
  if (status.ok()) status = hash->add_hash(10, bytes.header_sha256);
  if (status.ok()) status = hash->add_hash(11, *records);
  if (status.ok()) status = hash->add_hash(12, *weights);
  if (!status.ok()) return status;
  auto shard = hash->finalize();
  if (!shard.ok()) return shard.status();
  result.shard_layout_root = *shard;
  return result;
}
}  // namespace pih::offline_deepseek
