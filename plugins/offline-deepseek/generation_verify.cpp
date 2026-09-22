// Offline observation of an externally rooted PP1 generation, not a storage
// capability or immutable admission receipt. No model execution or publication.
#include "pih/model/deepseek_runtime_artifact_manifest.h"
#include "pih/model/safetensors_header.h"
#include "pih/model/safetensors_shard_index.h"
#include "tensor_inventory.h"
#include "generation_verify.h"
#include "generation_receipt.h"
#include "prepare_generation.h"
#include "source_inventory.h"
#include "disposition_records.h"
#include "pih/core/bounded_json.h"
#include "pih/core/canonical_json.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
template<class T> T Require(pih::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
void Require(pih::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
class Descriptor final {
 public:
  explicit Descriptor(int value) : value_(value) {
    Check(value >= 0, "cannot open artifact path without following symlinks");
  }
  ~Descriptor() { if (value_ >= 0) ::close(value_); }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  Descriptor(Descriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) ::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  int get() const { return value_; }
 private:
  int value_;
};
Descriptor OpenRoot(const std::filesystem::path& path) {
  Check(path.is_absolute(), "generation root must be absolute");
  Descriptor root(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  for (const auto& component : path.relative_path()) {
    const auto name = component.string();
    Check(!name.empty() && name != "." && name != ".." && name.find('\0') == name.npos,
          "noncanonical generation path");
    root = Descriptor(::openat(root.get(), name.c_str(),
        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  }
  return root;
}
class File final {
 public:
  File(int root, std::string name, std::uint64_t maximum)
      : name_(std::move(name)), fd_(Open(root, name_)) {
    Check(::fstat(fd_.get(), &initial_) == 0 && S_ISREG(initial_.st_mode) &&
        initial_.st_size > 0 && static_cast<std::uint64_t>(initial_.st_size) <= maximum,
        "artifact member is not a regular file within its byte limit");
  }
  std::uint64_t size() const { return static_cast<std::uint64_t>(initial_.st_size); }
  void Sealed(dev_t device) const {
    Check(initial_.st_uid == ::geteuid() && initial_.st_nlink == 1 && initial_.st_dev == device &&
        (initial_.st_mode & 07777) == 0444, "receipt-bound member is not sealed on the generation filesystem");
  }
  void Stable(int root) const {
    struct stat observed{}, named{};
    Check(::fstat(fd_.get(), &observed) == 0 &&
        ::fstatat(root, name_.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Same(initial_, observed) && Same(initial_, named),
        "artifact member changed or was replaced during observation");
  }
  void Read(std::uint64_t offset, std::span<std::byte> output) const {
    Check(offset <= size() && output.size() <= size() - offset, "artifact read outside member");
    while (!output.empty()) {
      const auto count = ::pread(fd_.get(), output.data(), output.size(), static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Check(count > 0, "artifact read failed or file became shorter");
      offset += static_cast<std::size_t>(count);
      output = output.subspan(static_cast<std::size_t>(count));
    }
  }
  std::string Text() const {
    std::string value(static_cast<std::size_t>(size()), '\0');
    Read(0, std::as_writable_bytes(std::span(value)));
    return value;
  }
  void Hash(std::uint64_t expected_bytes, const pih::Sha256Digest& expected) const {
    Check(size() == expected_bytes, "artifact member size differs from manifest");
    HashRange(0, expected_bytes, expected);
  }
  void HashRange(std::uint64_t begin, std::uint64_t bytes,
                 const pih::Sha256Digest& expected) const {
    Check(begin <= size() && bytes <= size() - begin, "payload hash range outside member");
    std::vector<std::byte> buffer(1U << 20);
    pih::Sha256 hash;
    for (std::uint64_t offset = 0; offset < bytes;) {
      const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), bytes - offset));
      auto chunk = std::span(buffer).first(count);
      Read(begin + offset, chunk);
      Require(hash.update(chunk));
      offset += count;
    }
    Check(Require(hash.finalize()) == expected, "artifact byte range SHA-256 differs from expected digest");
  }
  pih::SafetensorsHeader Header() const {
    std::array<std::byte, 8> length{};
    Read(0, length);
    std::uint64_t bytes = 0;
    for (unsigned i = 0; i < 8; ++i)
      bytes |= static_cast<std::uint64_t>(std::to_integer<unsigned>(length[i])) << (i * 8);
    Check(bytes > 0 && bytes <= pih::SafetensorsHeader::kMaxHeaderBytes &&
        size() >= 8 && bytes <= size() - 8, "safetensors header length invalid");
    std::vector<std::byte> prefix(static_cast<std::size_t>(bytes + 8));
    Read(0, prefix);
    return Require(pih::SafetensorsHeader::ParsePrefix(prefix, size()));
  }
 private:
  static int Open(int root, const std::string& name) {
    Check(!name.empty() && name != "." && name != ".." &&
        name.find_first_of("/\\") == name.npos && name.find('\0') == name.npos,
        "artifact member must be a single path component");
    return ::openat(root, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  }
  static bool Same(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
        a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
        a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
        a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
  }
  std::string name_;
  Descriptor fd_;
  struct stat initial_{};
};
struct SourceExpectation final {
  const pih::offline_deepseek::GenerationMetadata& metadata;
  const pih::offline_deepseek::Pp1Disposition& disposition;
  const pih::Sha256Digest& converter_identity_root;
};
// Called only for a manifest already matched to independently rebuilt source
// metadata, after actual target hashes and source-payload comparisons succeed.
std::string VerificationProjection(std::string_view manifest_text) {
  using pih::JsonValue;
  const auto manifest = Require(JsonValue::Parse(manifest_text));
  const auto field = [](const JsonValue& object, std::string_view name) -> const JsonValue& {
    const auto* value = object.at(name);
    Check(value != nullptr, "verified manifest projection field missing");
    return *value;
  };
  const auto text = [](std::string_view value) { return JsonValue(std::string(value)); };
  JsonValue::Object fields{
      {"schema", text("pih.deepseek_v4_flash_0731_runtime_artifact_verification.v1")},
      {"verification_scope", text("converted_bytes_and_source_payload_non_authorizing")},
      {"manifest_bytes", JsonValue(static_cast<std::int64_t>(manifest_text.size()))},
      {"manifest_sha256", text(Require(pih::sha256(std::as_bytes(std::span(manifest_text)))).hex())}};
  for (const auto* name : {"artifact_abi", "converter_abi", "model_family", "artifact_root",
      "conversion_root", "layout_root", "disposition_root", "converter_identity_root",
      "manifest_body_sha256", "dspark_enabled", "world_size", "shards", "support_state"})
    fields.emplace_back(name, field(manifest, name));
  fields.emplace_back("copied_tensor_count", field(manifest, "tensor_count"));
  fields.emplace_back("copied_tensor_bytes", field(manifest, "tensor_bytes"));
  const auto& index = field(manifest, "index");
  fields.emplace_back("index_root", field(index, "index_root"));
  fields.emplace_back("index_object_root", field(index, "index_object_root"));
  fields.emplace_back("index_sha256", field(index, "object_sha256"));
  fields.emplace_back("index_bytes", field(index, "file_bytes"));
  const auto& records = field(manifest, "runtime_records");
  fields.emplace_back("runtime_records_root", field(records, "runtime_records_root"));
  fields.emplace_back("runtime_record_set_root", field(records, "record_set_root"));
  fields.emplace_back("runtime_records_object_root", field(records, "runtime_records_object_root"));
  fields.emplace_back("runtime_records_body_sha256", field(records, "body_sha256"));
  fields.emplace_back("runtime_records_sha256", field(records, "object_sha256"));
  fields.emplace_back("runtime_records_bytes", field(records, "file_bytes"));
  const auto& resources = field(manifest, "resources");
  for (const auto* name : {"maximum_copy_chunk_bytes", "source_descriptor_count", "maximum_output_descriptors"})
    fields.emplace_back(name, field(resources, name));
  return Require(pih::canonical_ascii_json(JsonValue(std::move(fields)), 1U << 20));
}
void ExactMembers(int root, const pih::DeepSeekRuntimeArtifactManifest& manifest) {
  std::set<std::string> expected{"pih.manifest.json", "pih.runtime-records.json", "model.safetensors.index.json"};
  for (const auto& shard : manifest.shards()) expected.insert(shard.shard_name);
  const auto fd = ::openat(root, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  Check(fd >= 0, "cannot open generation enumeration");
  auto* stream = ::fdopendir(fd);
  if (!stream) { ::close(fd); throw std::runtime_error("cannot enumerate generation members"); }
  std::unique_ptr<DIR, decltype(&::closedir)> owned(stream, &::closedir);
  errno = 0;
  while (const auto* entry = ::readdir(stream)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..")
      Check(expected.erase(std::string(name)) == 1, "generation contains an unexpected member");
    errno = 0;
  }
  Check(errno == 0 && expected.empty(), "generation member enumeration failed or incomplete");
}
pih::offline_deepseek::GenerationObservation Verify(
    const std::filesystem::path& directory, const pih::Sha256Digest& root_digest,
    const pih::offline_deepseek::VerificationProgress& progress,
    const SourceExpectation* source = nullptr,
    const pih::offline_deepseek::GenerationReceipt* receipt = nullptr) {
  Check(root_digest != pih::Sha256Digest{}, "zero artifact root is forbidden");
  auto root = OpenRoot(directory);
  struct stat root_identity{};
  Check(::fstat(root.get(), &root_identity) == 0, "cannot inspect generation directory");
  if (receipt) Check(root_identity.st_uid == ::geteuid() && (root_identity.st_mode & 07777) == 0555,
      "receipt-bound generation directory is not sealed");
  std::vector<std::unique_ptr<File>> files;
  const auto open = [&](const std::string& name, std::uint64_t limit) -> const File& {
    files.push_back(std::make_unique<File>(root.get(), name, limit));
    if (receipt) files.back()->Sealed(root_identity.st_dev);
    return *files.back();
  };
  const auto& manifest_file = open("pih.manifest.json", pih::DeepSeekRuntimeArtifactManifest::kMaximumBytes);
  const auto manifest_text = manifest_file.Text();
  const auto manifest = Require(pih::DeepSeekRuntimeArtifactManifest::Parse(manifest_text, root_digest));
  Require(manifest.validate_flash_0731_geometry());
  Check(manifest.world_size() == 1 && !manifest.dspark_enabled(), "only native PP1 artifacts are accepted");
  if (source || receipt) ExactMembers(root.get(), manifest);
  if (receipt) {
    Check(receipt->artifact_root == root_digest && receipt->objects.size() == 47,
        "receipt artifact root or member count differs");
    manifest_file.Hash(receipt->objects[46].bytes, receipt->objects[46].sha256);
    const auto document = Require(pih::JsonValue::Parse(manifest_text));
    const auto* converter = document.at("converter_identity_root");
    Check(manifest.conversion_root() == receipt->conversion_root && manifest.layout_root() == receipt->layout_root &&
        manifest.disposition_root() == receipt->disposition_root && converter && converter->is_string() &&
        converter->string() == receipt->converter_identity_root.hex(), "receipt conversion authority differs from manifest");
    const auto match = [&](std::size_t ordinal, std::string_view name, std::uint64_t bytes, const pih::Sha256Digest& digest) {
      const auto& object = receipt->objects.at(ordinal);
      Check(object.name == name && object.bytes == bytes && object.sha256 == digest,
          "receipt object ledger differs from artifact manifest");
    };
    match(44, manifest.index().name, manifest.index().file_bytes, manifest.index().object_sha256);
    match(45, "pih.runtime-records.json", manifest.runtime_records().object_bytes, manifest.runtime_records().object_sha256);
    for (std::size_t i = 0; i < manifest.shards().size(); ++i) {
      const auto& shard = manifest.shards()[i];
      match(i, shard.shard_name, shard.file_bytes, shard.object_sha256);
    }
  }
  const auto pipeline = Require(pih::DeepSeekPipelinePlan::Create(1, false));
  const auto& index_file = open(manifest.index().name, pih::SafetensorsShardIndex::kMaximumIndexBytes);
  index_file.Hash(manifest.index().file_bytes, manifest.index().object_sha256);
  const auto index = Require(pih::SafetensorsShardIndex::Parse(index_file.Text()));
  const auto& records_file = open("pih.runtime-records.json", pih::DeepSeekRuntimeRecordsManifest::kMaximumBytes);
  records_file.Hash(manifest.runtime_records().object_bytes, manifest.runtime_records().object_sha256);
  const auto records = Require(pih::DeepSeekRuntimeRecordsManifest::Parse(
      records_file.Text(), manifest.runtime_records(), pipeline));
  if (source) {
    const auto& metadata = source->metadata;
    const auto& bound = metadata.bound_layout;
    Check(index_file.Text() == bound.layout.index_json,
        "target index differs from independently rebuilt source layout");
    Check(records_file.Text() == metadata.runtime_records.json,
        "target runtime records differ from independently rebuilt source records");
    pih::DeepSeekRuntimeArtifactManifest::EncodeInput input;
    input.logical_layout_root = bound.logical_layout_root;
    input.source_inventory_root = source->disposition.layout_authority.source_inventory_root;
    input.source_payload_closure_root = source->disposition.layout_authority.source_payload_closure_root;
    input.converter_identity_root = source->converter_identity_root;
    input.records = metadata.runtime_records.authority;
    input.maximum_copy_chunk_bytes = 1U << 20;
    input.source_descriptor_count = 50;
    input.maximum_output_descriptors = 1;
    input.index = {"model.safetensors.index.json", bound.layout.index_json.size(),
        bound.layout.index_sha256, bound.index_root, {}};
    Check(manifest.shards().size() == bound.layout.shards.size(), "source/target shard count differs");
    for (std::size_t i = 0; i < bound.layout.shards.size(); ++i) {
      const auto& shard = bound.layout.shards[i];
      input.shards.push_back({shard.name_space, shard.layout.member_name, shard.layout.file_bytes,
          static_cast<std::uint32_t>(shard.tensor_indices.size()),
          shard.layout.file_bytes - shard.layout.header_prefix.size(),
          manifest.shards()[i].object_sha256, bound.shard_layout_roots[i], {}});
    }
    const auto expected = Require(pih::DeepSeekRuntimeArtifactManifest::Encode(input));
    Check(expected.artifact_root == root_digest && expected.json == manifest_text,
        "target manifest differs from independently rebuilt source conversion");
  }
  const auto expected_names = Require(pih::offline_deepseek::ExpectedTensorNames(false));
  Check(records.records().size() == expected_names.size(), "runtime record inventory count differs");
  for (std::size_t i = 0; i < expected_names.size(); ++i)
    Check(records.records()[i].tensor_name == expected_names[i], "runtime record differs from frozen tensor inventory");
  for (const auto& record : records.records())
    Require(pih::offline_deepseek::ValidateRoutedExpertGeometry(record.tensor_name, record.dtype, record.shape));
  for (const auto& record : records.records())
    Require(pih::offline_deepseek::ValidatePp1DenseGeometry(record.tensor_name, record.dtype, record.shape));
  const auto ownership = Require(pih::DeepSeekTensorOwnershipPlan::BindTargetManifest(pipeline, records));
  Check(ownership.owned_count(0) == manifest.tensor_count(), "PP1 ownership closure differs from manifest");
  Check(index.total_size() == manifest.tensor_bytes() && index.bindings().size() == manifest.tensor_count() &&
      records.records().size() == index.bindings().size(), "index and record ledgers differ from manifest");
  for (std::size_t i = 0; i < index.bindings().size(); ++i)
    Check(index.bindings()[i].tensor_name == records.records()[i].tensor_name &&
        index.bindings()[i].shard_name == records.records()[i].shard_name,
        "index binding differs from runtime record");
  std::vector<std::string> names;
  std::map<std::string, pih::SafetensorsHeader, std::less<>> headers;
  std::map<std::string, std::string, std::less<>> namespaces;
  std::map<std::string, const File*, std::less<>> shard_files;
  for (const auto& shard : manifest.shards()) names.push_back(shard.shard_name);
  std::ranges::sort(names);
  Check(names == index.shard_names(), "manifest and index shard sets differ");
  for (const auto& shard : manifest.shards()) {
    if (progress) progress(shard.shard_name, shard.file_bytes);
    const auto& file = open(shard.shard_name, 512ULL << 30);
    shard_files.emplace(shard.shard_name, &file);
    file.Hash(shard.file_bytes, shard.object_sha256);
    auto header = file.Header();
    Check(header.tensors().size() == shard.tensor_count && header.data_bytes() == shard.payload_bytes,
        "shard header ledger differs from manifest");
    Check(headers.emplace(shard.shard_name, std::move(header)).second, "duplicate shard");
    namespaces.emplace(shard.shard_name, shard.name_space);
  }
  std::map<std::string, std::uint32_t, std::less<>> counts;
  for (const auto& record : records.records()) {
    const auto header = headers.find(record.shard_name);
    Check(header != headers.end(), "runtime record references absent shard");
    Check(namespaces.at(record.shard_name) == record.name_space, "runtime record namespace differs from shard");
    const auto* tensor = header->second.tensor(record.tensor_name);
    Check(tensor && tensor->dtype == record.dtype && tensor->shape == record.shape &&
        tensor->file_begin == record.file_begin && tensor->file_end == record.file_end &&
        record.tensor_bytes == record.file_end - record.file_begin,
        "runtime record differs from actual safetensors header");
    ++counts[record.shard_name];
  }
  for (const auto& shard : manifest.shards())
    Check(counts[shard.shard_name] == shard.tensor_count, "runtime record closure incomplete");
  if (source) {
    const auto& selected = source->disposition.records.selected;
    Check(selected.size() == records.records().size(), "source/target tensor count differs");
    for (const auto& shard : source->metadata.bound_layout.layout.shards) {
      if (progress) progress("source payload comparison: " + shard.layout.member_name, shard.layout.file_bytes);
      std::vector<std::byte> actual(shard.layout.header_prefix.size());
      shard_files.at(shard.layout.member_name)->Read(0, actual);
      Check(actual == shard.layout.header_prefix, "target header differs from independently rebuilt source layout");
      for (const auto i : shard.tensor_indices) {
        const auto& record = records.records().at(i);
        const auto& tensor = selected.at(i);
        Check(tensor.name == record.tensor_name && tensor.source.bytes == record.tensor_bytes &&
            record.shard_name == shard.layout.member_name, "source payload and target record join differs");
        shard_files.at(record.shard_name)->HashRange(record.file_begin, record.tensor_bytes,
            tensor.source.payload_sha256);
      }
    }
    ExactMembers(root.get(), manifest);
  }
  if (receipt) ExactMembers(root.get(), manifest);
  for (const auto& file : files) file->Stable(root.get());
  auto current_root = OpenRoot(directory);
  struct stat current_identity{};
  Check(::fstat(current_root.get(), &current_identity) == 0 &&
      current_identity.st_dev == root_identity.st_dev && current_identity.st_ino == root_identity.st_ino,
      "generation root was replaced during observation");
  pih::offline_deepseek::GenerationObservation result{root_digest, manifest.tensor_count(), manifest.tensor_bytes(),
      static_cast<std::uint32_t>(manifest.shards().size()),
      static_cast<std::uint64_t>(root_identity.st_dev),
      static_cast<std::uint64_t>(root_identity.st_ino)};
  if (source) {
    result.verification_projection_json = VerificationProjection(manifest_text);
    result.verification_projection_sha256 = Require(pih::sha256(
        std::as_bytes(std::span(result.verification_projection_json))));
  }
  result.receipt_binding_verified = receipt != nullptr;
  return result;
}
}  // namespace
namespace pih::offline_deepseek {
Result<GenerationObservation> VerifyGeneration(const std::filesystem::path& directory,
    const Sha256Digest& expected_root, const VerificationProgress& progress) {
  try { return Verify(directory, expected_root, progress); }
  catch (const std::exception& error) {
    return Status::FailedPrecondition(error.what());
  }
}
Result<GenerationObservation> VerifyReceiptBoundGeneration(const std::filesystem::path& directory,
    std::string_view receipt_json, const Sha256Digest& artifact_root, const Sha256Digest& receipt_root,
    const VerificationProgress& progress) {
  try {
    const auto receipt = Require(ParseGenerationReceipt(receipt_json, artifact_root, receipt_root));
    return Verify(directory, artifact_root, progress, nullptr, &receipt);
  } catch (const std::exception& error) { return Status::FailedPrecondition(error.what()); }
}
Result<GenerationObservation> VerifySourceBoundGeneration(const SourceArtifact& source,
    const SourcePreparationAuthority& authority, const std::filesystem::path& directory,
    const Sha256Digest& expected_root, const VerificationProgress& progress) {
  try {
    Check(authority.model_digest == source.model_digest() && authority.model_digest != Sha256Digest{} &&
        authority.semantic_root != Sha256Digest{} && authority.inventory_root != Sha256Digest{} &&
        authority.expected_payload_root != Sha256Digest{} && authority.converter_identity_root != Sha256Digest{} &&
        expected_root != Sha256Digest{}, "source-bound verification authority invalid");
    Require(source.Revalidate());
    const auto inventory = Require(CompileSourceInventory(source));
    Check(inventory.plan.semantic_root == authority.semantic_root && inventory.inventory_root == authority.inventory_root,
        "source-bound verification semantic/inventory authority mismatch");
    const auto payload = Require(ObserveSourcePayloadClosure(inventory.plan.shards, authority.model_digest,
        authority.semantic_root, authority.inventory_root));
    Check(payload.closure_root == authority.expected_payload_root, "source-bound verification payload authority mismatch");
    const auto disposition = Require(BuildPp1Disposition(payload.tensors, inventory.inventory_root, payload.closure_root));
    const auto metadata = Require(BuildGenerationMetadata(disposition.records.selected,
        disposition.records.layout_authorities, disposition.records.selected_dispositions, disposition.layout_authority));
    const SourceExpectation expectation{metadata, disposition, authority.converter_identity_root};
    auto observed = Verify(directory, expected_root, progress, &expectation);
    Require(source.Revalidate());
    observed.source_payload_equivalence = true;
    return observed;
  } catch (const std::exception& error) {
    return Status::FailedPrecondition(error.what());
  }
}
}  // namespace pih::offline_deepseek
