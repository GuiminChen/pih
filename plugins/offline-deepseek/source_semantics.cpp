#include "source_semantics.h"
#include "pih/core/canonical_hash.h"
#include <algorithm>
#include <cerrno>
#include <map>
#include <unistd.h>

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
    default: return Status::InvalidArgument("source semantic dtype outside frozen inventory");
  }
}
}  // namespace
Result<SourceSemantics> CompileSourceSemantics(const SourceArtifact& source) {
  auto headers = source.ReadShardHeaders();
  if (!headers.ok()) return headers.status();
  SourceSemantics result;
  std::map<std::string, std::uint64_t> dtype_bytes;
  std::uint64_t total_header_bytes = 0;
  for (std::size_t i = 0; i < headers->size(); ++i) {
    const auto& header = (*headers)[i];
    const auto& object = source.objects()[i + 2];
    SourceSemanticShard shard;
    shard.name = object.name;
    shard.header_bytes = header.header_bytes();
    shard.data_bytes = header.data_bytes();
    total_header_bytes += shard.header_bytes;
    std::vector<std::byte> prefix(static_cast<std::size_t>(shard.header_bytes + 8));
    for (std::size_t offset = 0; offset < prefix.size();) {
      const auto count = ::pread(source.descriptors()[i + 2], prefix.data() + offset,
                                prefix.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return Status::Unavailable("source semantic header read failed");
      offset += static_cast<std::size_t>(count);
    }
    auto prefix_hash = sha256(prefix);
    if (!prefix_hash.ok()) return prefix_hash.status();
    shard.header_prefix_sha256 = *prefix_hash;
    auto tensors = header.tensors();
    std::sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    auto set = CanonicalHashBuilder::Create("pih:deepseek-safetensors-tensor-set:v1",
        static_cast<std::uint32_t>(tensors.size() + 1));
    if (!set.ok()) return set.status();
    auto status = set->add_u32(1, static_cast<std::uint32_t>(tensors.size()));
    if (!status.ok()) return status;
    for (std::size_t t = 0; t < tensors.size(); ++t) {
      auto& tensor = tensors[t];
      auto dtype = DTypeText(tensor.dtype);
      if (!dtype.ok()) return dtype.status();
      dtype_bytes[std::string(*dtype)] += tensor.file_end - tensor.file_begin;
      std::vector<std::byte> shape(tensor.shape.size() * 8);
      for (std::size_t d = 0; d < tensor.shape.size(); ++d)
        for (unsigned b = 0; b < 8; ++b)
          shape[d * 8 + b] = static_cast<std::byte>((tensor.shape[d] >> (8 * b)) & 255);
      auto hash = CanonicalHashBuilder::Create("pih:deepseek-safetensors-tensor:v1", 7);
      if (!hash.ok()) return hash.status();
      status = Text(*hash, 1, tensor.name);
      if (status.ok()) status = Text(*hash, 2, shard.name);
      if (status.ok()) status = Text(*hash, 3, *dtype);
      if (status.ok()) status = hash->add_u32(4, static_cast<std::uint32_t>(tensor.shape.size()));
      if (status.ok()) status = hash->add_bytes(5, shape);
      if (status.ok()) status = hash->add_u64(6, tensor.file_begin - prefix.size());
      if (status.ok()) status = hash->add_u64(7, tensor.file_end - prefix.size());
      if (!status.ok()) return status;
      auto root = hash->finalize();
      if (!root.ok()) return root.status();
      status = set->add_hash(static_cast<std::uint16_t>(100 + t), *root);
      if (!status.ok()) return status;
      shard.tensors.push_back({std::move(tensor), *root});
    }
    auto set_root = set->finalize();
    if (!set_root.ok()) return set_root.status();
    shard.tensor_set_root = *set_root;
    auto hash = CanonicalHashBuilder::Create("pih:deepseek-safetensors-shard:v1", 8);
    if (!hash.ok()) return hash.status();
    status = hash->add_hash(1, object.object_root);
    if (status.ok()) status = Text(*hash, 2, shard.name);
    if (status.ok()) status = hash->add_u64(3, object.bytes);
    if (status.ok()) status = hash->add_u64(4, shard.header_bytes);
    if (status.ok()) status = hash->add_u64(5, shard.data_bytes);
    if (status.ok()) status = hash->add_hash(6, *prefix_hash);
    if (status.ok()) status = hash->add_u32(7, static_cast<std::uint32_t>(shard.tensors.size()));
    if (status.ok()) status = hash->add_hash(8, *set_root);
    if (!status.ok()) return status;
    auto root = hash->finalize();
    if (!root.ok()) return root.status();
    shard.shard_root = *root;
    result.shards.push_back(std::move(shard));
  }
  const std::map<std::string, std::uint64_t> expected{
      {"BF16", 2'967'134'976ULL}, {"I64", 18'616'320ULL}, {"F32", 150'966'520ULL},
      {"F8_E8M0", 9'261'408'000ULL}, {"F8_E4M3", 6'304'038'912ULL}, {"I8", 148'176'371'712ULL}};
  if (dtype_bytes != expected) return Status::InvalidArgument("source semantic dtype-byte ledger differs");
  auto types = CanonicalHashBuilder::Create("pih:deepseek-safetensors-dtypes:v1", 7);
  if (!types.ok()) return types.status();
  auto status = types->add_u32(1, 6);
  std::uint16_t field = 100;
  for (const auto& [dtype, bytes] : dtype_bytes) {
    std::vector<std::byte> packed(dtype.size() + 1 + 8);
    for (std::size_t i = 0; i < dtype.size(); ++i) packed[i] = static_cast<std::byte>(dtype[i]);
    for (unsigned b = 0; b < 8; ++b) packed[dtype.size() + 1 + b] = static_cast<std::byte>((bytes >> (8 * b)) & 255);
    if (status.ok()) status = types->add_bytes(field++, packed);
  }
  if (!status.ok()) return status;
  auto dtype_root = types->finalize();
  if (!dtype_root.ok()) return dtype_root.status();
  result.dtype_summary_root = *dtype_root;
  auto hash = CanonicalHashBuilder::Create("pih:deepseek-safetensors-semantic-closure:v1", 57);
  if (!hash.ok()) return hash.status();
  status = Text(*hash, 1, "deepseek_safetensors_semantic_closure_v1");
  if (status.ok()) status = Text(*hash, 2, "deepseek_v4_flash_0731");
  if (status.ok()) status = hash->add_hash(3, source.model_digest());
  if (status.ok()) status = hash->add_hash(4, source.objects()[1].object_root);
  if (status.ok()) status = hash->add_u32(5, 48);
  if (status.ok()) status = hash->add_u64(6, 72'317);
  if (status.ok()) status = hash->add_u64(7, 166'878'536'440ULL);
  if (status.ok()) status = hash->add_u64(8, total_header_bytes);
  if (status.ok()) status = hash->add_hash(9, *dtype_root);
  for (std::size_t i = 0; status.ok() && i < result.shards.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), result.shards[i].shard_root);
  if (!status.ok()) return status;
  auto root = hash->finalize();
  if (!root.ok()) return root.status();
  result.semantic_root = *root;
  status = source.Revalidate();
  if (!status.ok()) return status;
  return result;
}
}  // namespace pih::offline_deepseek
