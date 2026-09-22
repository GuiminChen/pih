#include "weight_source_checkpoint.h"
#include "weight_catalog.h"
#include "weight_scalar_convert.h"
#include "weight_expert_convert.h"
#include <algorithm>
#include <charconv>
#include "pih/core/bounded_json.h"
#include "pih/model/safetensors_shard_index.h"
#include <cerrno>
#include <fcntl.h>
#include <new>
#include <optional>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
struct Expected final { std::string name; std::uint64_t bytes; Sha256Digest digest; };
Result<Expected> ParseExpected(const JsonValue* item, std::uint64_t maximum) {
  if (!item || !item->is_object() || item->object().size() != 3)
    return Status::InvalidArgument("V4.1 source expectation member schema invalid");
  const auto* name = item->at("name"); const auto* bytes = item->at("bytes"); const auto* hash = item->at("sha256");
  if (!name || !name->is_string() || !bytes || !bytes->is_integer() || bytes->integer() <= 0 ||
      static_cast<std::uint64_t>(bytes->integer()) > maximum || !hash || !hash->is_string())
    return Status::InvalidArgument("V4.1 source expectation identity/extent invalid");
  auto digest = Sha256Digest::ParseHex(hash->string()); if (!digest.ok()) return digest.status();
  if (*digest == Sha256Digest{} || digest->hex() != hash->string())
    return Status::InvalidArgument("V4.1 source expectation requires nonzero lowercase digest");
  return Expected{name->string(), static_cast<std::uint64_t>(bytes->integer()), *digest};
}
struct Metadata final {
  int fd = -1;
  std::string name, contents;
  struct stat identity{};
  ~Metadata() { if (fd >= 0) ::close(fd); }
};
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_size == b.st_size && a.st_nlink == b.st_nlink && a.st_uid == b.st_uid && a.st_gid == b.st_gid &&
      a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
      a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
Status CheckMetadata(int root, const Metadata& metadata) {
  struct stat held{}, named{};
  if (::fstat(metadata.fd, &held) || ::fstatat(root, metadata.name.c_str(), &named, AT_SYMLINK_NOFOLLOW) ||
      !Same(metadata.identity, held) || !Same(held, named))
    return Status::FailedPrecondition("V4.1 source metadata changed or was replaced");
  return Status::Ok();
}
Status OpenMetadata(int root, const Expected& expected, Metadata& result) {
  result.name = expected.name;
  result.fd = ::openat(root, expected.name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (result.fd < 0 || ::fstat(result.fd, &result.identity) || !S_ISREG(result.identity.st_mode) ||
      result.identity.st_nlink != 1 || (result.identity.st_mode & 0222) || result.identity.st_size < 0 ||
      static_cast<std::uint64_t>(result.identity.st_size) != expected.bytes)
    return Status::FailedPrecondition("V4.1 source metadata must be exact-size read-only single-link regular storage");
  result.contents.resize(static_cast<std::size_t>(expected.bytes));
  for (std::size_t offset = 0; offset < result.contents.size();) {
    const auto count = ::pread(result.fd, result.contents.data() + offset,
        result.contents.size() - offset, static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("V4.1 source metadata read failed");
    offset += static_cast<std::size_t>(count);
  }
  auto digest = sha256(std::as_bytes(std::span(result.contents.data(), result.contents.size())));
  if (!digest.ok()) return digest.status();
  if (*digest != expected.digest) return Status::FailedPrecondition("V4.1 source metadata differs from trusted expectation");
  return CheckMetadata(root, result);
}
}
struct WeightSourceCheckpoint::Impl final {
  struct SharedExpert final {
    const SourceNameBinding* weight_binding;
    const SourceNameBinding* scale_binding;
    const WeightSourceFile* weight_file;
    const WeightSourceFile* scale_file;
    const SafetensorRecord* weight;
    const SafetensorRecord* scale;
    std::uint64_t rows, columns;
    bool output_scale;
    Status Convert(const WoATensorWriter& write) const {
      const WoATensorWriter discard = [](std::uint64_t, std::span<const std::byte>) { return Status::Ok(); };
      auto status = ConvertExpertFp4Tensor(rows, columns,
          [&](std::uint64_t offset, std::span<std::byte> bytes) { return weight_file->Read(weight->file_begin + offset, bytes); },
          [&](std::uint64_t offset, std::span<std::byte> bytes) { return scale_file->Read(scale->file_begin + offset, bytes); },
          output_scale ? discard : write, output_scale ? write : discard);
      if (!status.ok()) return status;
      status = weight_file->Revalidate(); if (!status.ok()) return status;
      return scale_file->Revalidate();
    }
  };
  Sha256Digest expectation_digest;
  int directory = -1;
  Metadata configuration, index;
  std::optional<FlashConfig> config;
  std::optional<WeightSourceInventory> inventory;
  std::optional<BackboneWeightInventory> targets;
  std::vector<std::unique_ptr<WeightSourceFile>> files;
  Result<std::optional<SharedExpert>> SharedExpertFor(std::string_view name) const {
    if (name.find(".ffn.shared_experts.") == name.npos ||
        (!name.ends_with(".weight") && !name.ends_with(".scale"))) return std::optional<SharedExpert>{};
    try {
      const auto prefix = std::string(name.substr(0, name.rfind('.') + 1));
      const auto* input = inventory->Find(prefix + "weight");
      if (!input) return Status::InvalidArgument("V4.1 shared expert weight missing");
      const auto& file = *files[input->shard];
      const auto* weight = file.header().tensor(input->source_name);
      if (!weight) return Status::Internal("V4.1 shared expert source record missing");
      if (weight->dtype != DType::kInt8 && weight->dtype != DType::kUInt8) return std::optional<SharedExpert>{};
      const auto* scale_input = inventory->Find(prefix + "scale");
      if (!scale_input) return Status::InvalidArgument("V4.1 packed shared expert scale missing");
      const auto& scale_file = *files[scale_input->shard];
      const auto* scale = scale_file.header().tensor(scale_input->source_name);
      const auto all = targets->weights();
      const auto found = std::lower_bound(all.begin(), all.end(), prefix + "weight",
          [](const auto& target, auto key) { return target.tensor.name < key; });
      if (found == all.end() || found->tensor.name != prefix + "weight" ||
          found->tensor.storage != WeightStorage::kE4M3FN)
        return Status::InvalidArgument("V4.1 shared expert FP8 target missing");
      const auto rows = found->tensor.rows, columns = found->tensor.columns;
      const auto target_scale = std::lower_bound(all.begin(), all.end(), prefix + "scale",
          [](const auto& target, auto key) { return target.tensor.name < key; });
      if (found->tensor.dimensions != 2 || found->tensor.bytes != rows * columns ||
          target_scale == all.end() || target_scale->tensor.name != prefix + "scale" ||
          target_scale->tensor.storage != WeightStorage::kE8M0 || target_scale->tensor.dimensions != 2 ||
          target_scale->tensor.rows != rows / 32 || target_scale->tensor.columns != columns / 32 ||
          target_scale->tensor.bytes != (rows / 32) * (columns / 32))
        return Status::InvalidArgument("V4.1 shared expert output scale geometry differs from converter");
      if (!((rows == 2304 && columns == 5120) || (rows == 5120 && columns == 2304)) ||
          weight->shape.size() != 2 || weight->shape[0] != rows || weight->shape[1] != columns / 2 ||
          weight->file_end < weight->file_begin || weight->file_end - weight->file_begin != rows * columns / 2 ||
          !scale || scale->dtype != DType::kFloat8E8M0 || scale->shape.size() != 2 ||
          scale->shape[0] != rows || scale->shape[1] != columns / 32 ||
          scale->file_end < scale->file_begin || scale->file_end - scale->file_begin != rows * columns / 32)
        return Status::InvalidArgument("V4.1 packed shared expert source dtype/shape/extent invalid");
      return std::optional<SharedExpert>(SharedExpert{input, scale_input, &file, &scale_file,
          weight, scale, rows, columns, name.ends_with(".scale")});
    } catch (const std::bad_alloc&) {
      return Status::ResourceExhausted("V4.1 shared expert binding allocation failed");
    }
  }
  ~Impl() { if (directory >= 0) ::close(directory); }
};
WeightSourceCheckpoint::WeightSourceCheckpoint(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
WeightSourceCheckpoint::~WeightSourceCheckpoint() = default;
const FlashConfig& WeightSourceCheckpoint::config() const noexcept { return *impl_->config; }
std::string_view WeightSourceCheckpoint::config_json() const noexcept { return impl_->configuration.contents; }
const Sha256Digest& WeightSourceCheckpoint::expectation_digest() const noexcept { return impl_->expectation_digest; }
const WeightSourceInventory& WeightSourceCheckpoint::inventory() const noexcept { return *impl_->inventory; }
Status WeightSourceCheckpoint::Revalidate() const {
  auto status = CheckMetadata(impl_->directory, impl_->configuration); if (!status.ok()) return status;
  status = CheckMetadata(impl_->directory, impl_->index); if (!status.ok()) return status;
  for (const auto& file : impl_->files) { status = file->Revalidate(); if (!status.ok()) return status; }
  return Status::Ok();
}
Result<std::unique_ptr<WeightSourceCheckpoint>> WeightSourceCheckpoint::Open(
    int directory, std::string_view json, const Sha256Digest& expected) {
  if (directory < 0 || json.empty() || json.size() > kMaximumExpectationBytes || expected == Sha256Digest{})
    return Status::InvalidArgument("V4.1 checkpoint expectations require bounded JSON and independent digest");
  try {
    auto digest = sha256(std::as_bytes(std::span(json.data(), json.size())));
    if (!digest.ok()) return digest.status();
    if (*digest != expected) return Status::FailedPrecondition("V4.1 source expectations differ from trusted digest");
    auto document = JsonValue::Parse(json, {kMaximumExpectationBytes, 4, 2048, 256});
    if (!document.ok()) return document.status();
    if (!document->is_object() || document->object().size() != 5)
      return Status::InvalidArgument("V4.1 source expectation root schema invalid");
    const auto* schema = document->at("schema"); const auto* revision = document->at("reference_revision");
    const auto* shards = document->at("shards");
    if (!schema || !schema->is_string() || schema->string() != "pih.deepseek-v41.source-expectations.v1" ||
        !revision || !revision->is_string() || revision->string() != FlashConfig::kReferenceRevision ||
        !shards || !shards->is_array() || shards->array().empty() || shards->array().size() > 256)
      return Status::InvalidArgument("V4.1 source expectation version/revision/shard count invalid");
    auto config = ParseExpected(document->at("config"), FlashConfig::kMaximumBytes);
    auto index = ParseExpected(document->at("index"), SafetensorsShardIndex::kMaximumIndexBytes);
    if (!config.ok()) return config.status(); if (!index.ok()) return index.status();
    if (config->name != "config.json" || index->name != "model.safetensors.index.json")
      return Status::InvalidArgument("V4.1 source config/index names must be fixed members");
    std::vector<Expected> members;
    for (const auto& shard : shards->array()) {
      auto member = ParseExpected(&shard, 512ULL << 30); if (!member.ok()) return member.status();
      if (!IsWeightShardMember(member->name) || member->bytes < 10)
        return Status::InvalidArgument("V4.1 source expectation shard member invalid");
      for (const auto& previous : members) if (previous.name == member->name)
        return Status::InvalidArgument("V4.1 source expectation has duplicate shard");
      members.push_back(std::move(*member));
    }
    auto impl = std::make_unique<Impl>();
    impl->expectation_digest = expected;
    impl->directory = ::fcntl(directory, F_DUPFD_CLOEXEC, 3);
    struct stat root{};
    if (impl->directory < 0 || ::fstat(impl->directory, &root) || !S_ISDIR(root.st_mode))
      return Status::FailedPrecondition("V4.1 checkpoint source directory inaccessible");
    auto status = OpenMetadata(impl->directory, *config, impl->configuration); if (!status.ok()) return status;
    auto parsed_config = FlashConfig::Parse(impl->configuration.contents); if (!parsed_config.ok()) return parsed_config.status();
    impl->config.emplace(std::move(*parsed_config));
    auto targets = BackboneWeightInventory::Create(*impl->config, 1, 0);
    if (!targets.ok()) return targets.status();
    impl->targets.emplace(std::move(*targets));
    status = OpenMetadata(impl->directory, *index, impl->index); if (!status.ok()) return status;
    std::vector<const WeightSourceFile*> views;
    for (const auto& member : members) {
      auto file = WeightSourceFile::Open(impl->directory, member.name, member.bytes, member.digest);
      if (!file.ok()) return file.status();
      views.push_back(file->get()); impl->files.push_back(std::move(*file));
    }
    auto inventory = WeightSourceInventory::Create(impl->index.contents, index->digest, views);
    if (!inventory.ok()) return inventory.status();
    impl->inventory.emplace(std::move(*inventory));
    // Release the index text; retain the bounded configuration snapshot for
    // byte-identical runtime artifact assembly without reopening its path.
    std::string{}.swap(impl->index.contents);
    auto result = std::unique_ptr<WeightSourceCheckpoint>(new WeightSourceCheckpoint(std::move(impl)));
    status = result->Revalidate(); if (!status.ok()) return status;
    return result;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 source checkpoint admission allocation failed");
  }
}
Status WeightSourceCheckpoint::ConvertWoA(std::uint32_t layer, const WoATensorWriter& write) const {
  if (layer >= FlashConfig::kMainLayers || !write)
    return Status::InvalidArgument("V4.1 source conversion requires main layer and writer");
  try {
    const auto prefix = "layers." + std::to_string(layer) + ".attn.wo_a.";
    const auto* weight = inventory().Find(prefix + "weight");
    const auto* scale = inventory().Find(prefix + "scale");
    if (!weight || !scale) return Status::InvalidArgument("V4.1 checkpoint missing wo_a weight or scale");
    auto status = Revalidate(); if (!status.ok()) return status;
    status = ConvertWoASourceFiles(layer, *impl_->files[weight->shard], weight->source_name,
        *impl_->files[scale->shard], scale->source_name, write);
    if (!status.ok()) return status;
    return Revalidate();
  } catch (...) {
    return Status::Internal("V4.1 checkpoint conversion failed; discard unpublished output");
  }
}
Status WeightSourceCheckpoint::ConvertBackboneTensor(std::string_view name, const WoATensorWriter& write) const {
  if (!write) return Status::InvalidArgument("V4.1 tensor conversion requires writer");
  const auto targets = impl_->targets->weights();
  const auto target = std::lower_bound(targets.begin(), targets.end(), name,
      [](const auto& weight, auto key) { return weight.tensor.name < key; });
  const auto* binding = inventory().Find(name);
  if (target == targets.end() || target->tensor.name != name || !binding)
    return Status::InvalidArgument("V4.1 tensor absent from canonical backbone or admitted source");
  const auto& file = *impl_->files[binding->shard];
  const auto* source = file.header().tensor(binding->source_name);
  if (!source) return Status::Internal("V4.1 source inventory lost tensor binding");
  auto shared = impl_->SharedExpertFor(name);
  if (!shared.ok()) return shared.status();
  if (shared->has_value()) {
    try { return shared->value().Convert(write); }
    catch (...) { return Status::Internal("V4.1 shared expert conversion failed; discard unpublished output"); }
  }
  if (name.ends_with(".attn.wo_a.weight") && source->dtype == DType::kFloat8E4M3) {
    const auto layer_text = name.substr(7, name.find('.', 7) - 7);
    std::uint32_t layer = 0;
    const auto parsed = std::from_chars(layer_text.data(), layer_text.data() + layer_text.size(), layer);
    if (parsed.ec != std::errc{} || parsed.ptr != layer_text.data() + layer_text.size())
      return Status::Internal("V4.1 canonical wo_a layer invalid");
    return ConvertWoA(layer, write);
  }
  try {
    auto status = file.Revalidate(); if (!status.ok()) return status;
    status = ConvertScalarWeight(target->tensor, *source,
        [&](std::uint64_t offset, std::span<std::byte> bytes) {
          if (offset > source->file_end - source->file_begin || bytes.size() > source->file_end - source->file_begin - offset)
            return Status::Internal("V4.1 scalar read exceeds source tensor");
          return file.Read(source->file_begin + offset, bytes);
        }, write);
    if (!status.ok()) return status;
    return file.Revalidate();
  } catch (...) {
    return Status::Internal("V4.1 backbone conversion failed; discard unpublished output");
  }
}
Status WeightSourceCheckpoint::ValidateBackboneConversion(std::span<const std::string> excluded) const {
  const auto bindings = inventory().bindings();
  if (excluded.size() > bindings.size())
    return Status::InvalidArgument("V4.1 exclusion count exceeds source inventory");
  try {
    std::vector<bool> consumed(bindings.size(), false);
    for (const auto& target : impl_->targets->weights()) {
      const auto* input = inventory().Find(target.tensor.name);
      if (!input) return Status::InvalidArgument("V4.1 conversion is missing a required backbone tensor");
      const auto& file = *impl_->files[input->shard];
      const auto* record = file.header().tensor(input->source_name);
      if (!record) return Status::Internal("V4.1 conversion lost source header member");
      consumed[static_cast<std::size_t>(input - bindings.data())] = true;
      auto shared = impl_->SharedExpertFor(target.tensor.name);
      if (!shared.ok()) return shared.status();
      if (shared->has_value()) {
        consumed[static_cast<std::size_t>(shared->value().weight_binding - bindings.data())] = true;
        consumed[static_cast<std::size_t>(shared->value().scale_binding - bindings.data())] = true;
        continue;
      }
      if (target.tensor.name.ends_with(".attn.wo_a.weight") && record->dtype == DType::kFloat8E4M3) {
        const auto& name = target.tensor.name;
        const auto* scale = inventory().Find(name.substr(0, name.size() - 6) + "scale");
        if (!scale) return Status::InvalidArgument("V4.1 conversion is missing wo_a scale");
        const auto layer_text = std::string_view(name).substr(7, name.find('.', 7) - 7);
        std::uint32_t layer = 0;
        const auto parsed = std::from_chars(layer_text.data(), layer_text.data() + layer_text.size(), layer);
        if (parsed.ec != std::errc{} || parsed.ptr != layer_text.data() + layer_text.size())
          return Status::Internal("V4.1 conversion inventory layer invalid");
        auto bound = WoASourceBinding::Create(layer, file.header(), input->source_name,
            impl_->files[scale->shard]->header(), scale->source_name);
        if (!bound.ok()) return bound.status();
        consumed[static_cast<std::size_t>(scale - bindings.data())] = true;
      } else {
        auto status = ValidateScalarWeight(target.tensor, *record);
        if (!status.ok()) return status;
      }
    }
    for (const auto& name : excluded) {
      if (name.empty() || name.size() > 512)
        return Status::InvalidArgument("V4.1 exclusion name invalid");
      const auto* input = inventory().Find(name);
      if (!input) return Status::InvalidArgument("V4.1 exclusion names unknown source tensor");
      const auto index = static_cast<std::size_t>(input - bindings.data());
      if (consumed[index]) return Status::InvalidArgument("V4.1 exclusion duplicates or discards a required input");
      consumed[index] = true;
    }
    if (std::find(consumed.begin(), consumed.end(), false) != consumed.end())
      return Status::InvalidArgument("V4.1 conversion has unclassified source tensors; explicit exclusions required");
    return Revalidate();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 conversion disposition allocation failed");
  }
}
Status WeightSourceCheckpoint::ConvertBackbone(std::span<const std::string> excluded,
    const BackboneWriter& write) const {
  if (!write) return Status::InvalidArgument("V4.1 backbone conversion writer absent");
  auto status = ValidateBackboneConversion(excluded); if (!status.ok()) return status;
  try {
    for (const auto& target : impl_->targets->weights()) {
      status = ConvertBackboneTensor(target.tensor.name,
          [&](std::uint64_t offset, std::span<const std::byte> bytes) { return write(target.tensor, offset, bytes); });
      if (!status.ok()) return status;
    }
    return Revalidate();
  } catch (...) {
    return Status::Internal("V4.1 backbone conversion failed; discard all unpublished output");
  }
}
}  // namespace pih::deepseek_v41
