#include "source_payload.h"
#include "tensor_inventory.h"
#include "pih/core/canonical_hash.h"
#include "pih/model/safetensors_header.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::offline_deepseek {
namespace {
Status Read(int fd, std::uint64_t offset, std::span<std::byte> bytes) {
  while (!bytes.empty()) {
    const auto count = ::pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("source payload read failed or truncated");
    offset += static_cast<std::size_t>(count);
    bytes = bytes.subspan(static_cast<std::size_t>(count));
  }
  return Status::Ok();
}
Status Text(CanonicalHashBuilder& hash, std::uint16_t id, std::string_view value) {
  return hash.add_bytes(id, std::as_bytes(std::span(value)));
}
}  // namespace
Result<std::vector<ObservedSourceTensor>> ObserveSourcePayloads(int source_fd,
    std::string_view shard_name, const Sha256Digest& artifact_object_root,
    std::span<const SourceTensorAuthority> authorities) {
  bool known_shard = false;
  for (unsigned i = 1; i <= 48; ++i) {
    const auto digits = std::to_string(i);
    if (shard_name == "model-" + std::string(5 - digits.size(), '0') + digits + "-of-00048.safetensors")
      known_shard = true;
  }
  if (!known_shard || artifact_object_root == Sha256Digest{} || authorities.empty() ||
      authorities.size() > SafetensorsHeader::kMaxTensorCount)
    return Status::InvalidArgument("source shard identity or authority count invalid");
  struct stat initial{};
  if (source_fd < 0 || ::fstat(source_fd, &initial) != 0 || !S_ISREG(initial.st_mode) ||
      initial.st_size < 8 || static_cast<std::uint64_t>(initial.st_size) > (512ULL << 30))
    return Status::InvalidArgument("source shard descriptor outside bounds");
  std::array<std::byte, 8> length{};
  auto status = Read(source_fd, 0, length);
  if (!status.ok()) return status;
  std::uint64_t header_bytes = 0;
  for (unsigned i = 0; i < 8; ++i)
    header_bytes |= static_cast<std::uint64_t>(std::to_integer<unsigned>(length[i])) << (8 * i);
  if (header_bytes == 0 || header_bytes > SafetensorsHeader::kMaxHeaderBytes ||
      header_bytes > static_cast<std::uint64_t>(initial.st_size) - 8)
    return Status::InvalidArgument("source shard header length invalid");
  std::vector<std::byte> prefix(static_cast<std::size_t>(header_bytes + 8));
  status = Read(source_fd, 0, prefix);
  if (!status.ok()) return status;
  auto header = SafetensorsHeader::ParsePrefix(prefix, static_cast<std::uint64_t>(initial.st_size));
  if (!header.ok()) return header.status();
  if (header->tensors().size() != authorities.size())
    return Status::InvalidArgument("source header and authority cardinality differ");
  std::vector<const SafetensorRecord*> ordered;
  for (const auto& tensor : header->tensors()) ordered.push_back(&tensor);
  std::sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) {
    return a->name < b->name;
  });
  for (std::size_t i = 0; i < authorities.size(); ++i) {
    if (authorities[i].name != ordered[i]->name ||
        authorities[i].dtype != ordered[i]->dtype || authorities[i].shape != ordered[i]->shape ||
        authorities[i].file_begin != ordered[i]->file_begin || authorities[i].file_end != ordered[i]->file_end ||
        (i != 0 && authorities[i - 1].name >= authorities[i].name) ||
        ordered[i]->file_begin == ordered[i]->file_end || ordered[i]->shape.empty() ||
        authorities[i].source_record_root == Sha256Digest{})
      return Status::InvalidArgument("source authority order or name join invalid");
    for (const unsigned char ch : authorities[i].name)
      if (ch < 0x20 || ch > 0x7e) return Status::InvalidArgument("source tensor name must be printable ASCII");
    status = ValidateSourceTensorGeometry(ordered[i]->name, ordered[i]->dtype, ordered[i]->shape);
    if (!status.ok()) return status;
  }
  std::vector<std::byte> buffer(1U << 20);
  std::vector<ObservedSourceTensor> result;
  result.reserve(authorities.size());
  for (std::size_t i = 0; i < authorities.size(); ++i) {
    const auto& tensor = *ordered[i];
    Sha256 payload;
    for (auto offset = tensor.file_begin; offset < tensor.file_end;) {
      auto chunk = std::span(buffer).first(static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), tensor.file_end - offset)));
      status = Read(source_fd, offset, chunk);
      if (!status.ok()) return status;
      status = payload.update(chunk);
      if (!status.ok()) return status;
      offset += chunk.size();
    }
    auto digest = payload.finalize();
    if (!digest.ok()) return digest.status();
    auto root = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-payload-record:v1", 9);
    if (!root.ok()) return root.status();
    status = root->add_hash(1, authorities[i].source_record_root);
    if (status.ok()) status = root->add_hash(2, artifact_object_root);
    if (status.ok()) status = Text(*root, 3, tensor.name);
    if (status.ok()) status = Text(*root, 4, shard_name);
    if (status.ok()) status = root->add_u64(5, tensor.file_begin - prefix.size());
    if (status.ok()) status = root->add_u64(6, tensor.file_end - prefix.size());
    if (status.ok()) status = root->add_u64(7, tensor.file_begin);
    if (status.ok()) status = root->add_u64(8, tensor.file_end);
    if (status.ok()) status = root->add_hash(9, *digest);
    if (!status.ok()) return status;
    auto record_root = root->finalize();
    if (!record_root.ok()) return record_root.status();
    result.push_back({{tensor.name, tensor.dtype, tensor.shape,
        {source_fd, tensor.file_begin, tensor.file_end - tensor.file_begin, *digest}},
        authorities[i].source_record_root, *record_root});
  }
  struct stat observed{};
  if (::fstat(source_fd, &observed) != 0 || initial.st_dev != observed.st_dev ||
      initial.st_ino != observed.st_ino || initial.st_mode != observed.st_mode ||
      initial.st_size != observed.st_size || initial.st_mtim.tv_sec != observed.st_mtim.tv_sec ||
      initial.st_mtim.tv_nsec != observed.st_mtim.tv_nsec || initial.st_ctim.tv_sec != observed.st_ctim.tv_sec ||
      initial.st_ctim.tv_nsec != observed.st_ctim.tv_nsec)
    return Status::FailedPrecondition("source shard changed during payload observation");
  return result;
}
Result<SourcePayloadClosure> ObserveSourcePayloadClosure(
    std::span<const SourcePayloadShardInput> shards,
    const Sha256Digest& artifact_model_digest, const Sha256Digest& semantic_root,
    const Sha256Digest& source_inventory_root) {
  constexpr std::uint64_t kBytes = 166'878'536'440ULL;
  constexpr std::size_t kCount = 72'317;
  if (shards.size() != 48 || artifact_model_digest == Sha256Digest{} ||
      semantic_root == Sha256Digest{} || source_inventory_root == Sha256Digest{})
    return Status::InvalidArgument("source payload closure antecedents invalid");
  std::vector<struct stat> initial(shards.size());
  std::set<std::pair<dev_t, ino_t>> identities;
  std::uint64_t total_bytes = 0;
  std::size_t total_count = 0;
  for (std::size_t i = 0; i < shards.size(); ++i) {
    const auto& shard = shards[i];
    const auto number = std::to_string(i + 1);
    const auto name = "model-" + std::string(5 - number.size(), '0') + number + "-of-00048.safetensors";
    if (shard.name != name || shard.inventory_root == Sha256Digest{} ||
        shard.artifact_object_root == Sha256Digest{} || shard.tensors.empty() ||
        shard.tensors.size() > SafetensorsHeader::kMaxTensorCount ||
        shard.tensor_bytes == 0 || shard.tensor_bytes > kBytes - total_bytes ||
        shard.tensors.size() > kCount - total_count || shard.descriptor < 0 ||
        ::fstat(shard.descriptor, &initial[i]) != 0 || !S_ISREG(initial[i].st_mode) ||
        !identities.emplace(initial[i].st_dev, initial[i].st_ino).second)
      return Status::InvalidArgument("source closure shard order, geometry or descriptor identity invalid");
    total_bytes += shard.tensor_bytes;
    total_count += shard.tensors.size();
  }
  if (total_bytes != kBytes || total_count != kCount)
    return Status::InvalidArgument("source closure differs from frozen count/byte ledger");
  auto names = ExpectedTensorNames(true);
  if (!names.ok()) return names.status();
  std::vector<std::string_view> supplied_names;
  supplied_names.reserve(kCount);
  for (const auto& shard : shards)
    for (const auto& tensor : shard.tensors) supplied_names.push_back(tensor.name);
  std::sort(supplied_names.begin(), supplied_names.end());
  for (std::size_t i = 0; i < kCount; ++i)
    if (supplied_names[i] != (*names)[i])
      return Status::InvalidArgument("source authority names differ from frozen inventory");
  SourcePayloadClosure result;
  result.tensors.reserve(kCount);
  for (const auto& shard : shards) {
    auto tensors = ObserveSourcePayloads(shard.descriptor, shard.name,
                                        shard.artifact_object_root, shard.tensors);
    if (!tensors.ok()) return tensors.status();
    auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-payload-record-set:v1",
        static_cast<std::uint32_t>(tensors->size() + 1));
    if (!hash.ok()) return hash.status();
    auto status = hash->add_u32(1, static_cast<std::uint32_t>(tensors->size()));
    std::uint64_t bytes = 0;
    for (std::size_t i = 0; i < tensors->size(); ++i) {
      if ((*tensors)[i].tensor.source.bytes > shard.tensor_bytes - bytes)
        return Status::InvalidArgument("observed source shard payload exceeds admitted ledger");
      bytes += (*tensors)[i].tensor.source.bytes;
      if (status.ok()) status = hash->add_hash(static_cast<std::uint16_t>(100 + i), (*tensors)[i].payload_record_root);
    }
    if (!status.ok()) return status;
    if (bytes != shard.tensor_bytes)
      return Status::InvalidArgument("observed source shard payload differs from admitted ledger");
    auto record_set = hash->finalize();
    if (!record_set.ok()) return record_set.status();
    hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-payload-shard:v1", 6);
    if (!hash.ok()) return hash.status();
    status = Text(*hash, 1, shard.name);
    if (status.ok()) status = hash->add_hash(2, shard.artifact_object_root);
    if (status.ok()) status = hash->add_hash(3, shard.inventory_root);
    if (status.ok()) status = hash->add_u32(4, static_cast<std::uint32_t>(tensors->size()));
    if (status.ok()) status = hash->add_u64(5, bytes);
    if (status.ok()) status = hash->add_hash(6, *record_set);
    if (!status.ok()) return status;
    auto shard_root = hash->finalize();
    if (!shard_root.ok()) return shard_root.status();
    result.shard_roots.push_back(*shard_root);
    for (auto& tensor : *tensors) result.tensors.push_back(std::move(tensor));
  }
  std::sort(result.tensors.begin(), result.tensors.end(), [](const auto& a, const auto& b) {
    return a.tensor.name < b.tensor.name;
  });
  for (std::size_t i = 0; i < kCount; ++i)
    if (result.tensors[i].tensor.name != (*names)[i])
      return Status::InvalidArgument("source payload inventory has absent, duplicate or foreign names");
  for (std::size_t i = 0; i < shards.size(); ++i) {
    struct stat observed{};
    const auto& before = initial[i];
    if (::fstat(shards[i].descriptor, &observed) != 0 || before.st_dev != observed.st_dev ||
        before.st_ino != observed.st_ino || before.st_mode != observed.st_mode ||
        before.st_size != observed.st_size || before.st_mtim.tv_sec != observed.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != observed.st_mtim.tv_nsec || before.st_ctim.tv_sec != observed.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != observed.st_ctim.tv_nsec)
      return Status::FailedPrecondition("source shard changed across complete payload scan");
  }
  auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-source-payload-closure:v1", 58);
  if (!hash.ok()) return hash.status();
  auto status = Text(*hash, 1, "deepseek_v4_flash_0731_source_payload_closure_v1");
  if (status.ok()) status = Text(*hash, 2, "deepseek_v4_flash_0731");
  if (status.ok()) status = hash->add_hash(3, artifact_model_digest);
  if (status.ok()) status = hash->add_hash(4, semantic_root);
  if (status.ok()) status = hash->add_hash(5, source_inventory_root);
  if (status.ok()) status = hash->add_u32(6, 48);
  if (status.ok()) status = hash->add_u64(7, kCount);
  if (status.ok()) status = hash->add_u64(8, kBytes);
  if (status.ok()) status = Text(*hash, 9, "exact_source_tensor_payload_bytes_non_authorizing");
  if (status.ok()) status = Text(*hash, 10, "hardware_evidence_open");
  for (std::size_t i = 0; status.ok() && i < result.shard_roots.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), result.shard_roots[i]);
  if (!status.ok()) return status;
  auto closure = hash->finalize();
  if (!closure.ok()) return closure.status();
  result.closure_root = *closure;
  return result;
}
}  // namespace pih::offline_deepseek
