#include "generation_pointer.h"
#include "generation_store.h"
#include "pih/core/bounded_json.h"
#include "pih/core/canonical_hash.h"
#include "pih/core/canonical_json.h"
#include <limits>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <utility>

namespace pih::offline_deepseek {
namespace {
constexpr std::string_view kAbi = "content_addressed_directory_generation_v1";
constexpr std::string_view kSchema = "pih.deepseek_v4_flash_0731_generation_pointer.v1";
constexpr std::string_view kSupport = "hardware_evidence_open";
constexpr std::size_t kMaximumBytes = 64U << 10;
constexpr auto kMaximumOrdinal = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
template<class T> T Require(Result<T> value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.status().message()));
  return std::move(*value);
}
void Require(Status value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.message()));
}
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
const JsonValue& Field(const JsonValue& object, std::string_view name) {
  const auto* value = object.at(name);
  Check(value != nullptr, "generation pointer field missing");
  return *value;
}
std::string Text(const JsonValue& object, std::string_view name) {
  const auto& value = Field(object, name);
  Check(value.is_string(), "generation pointer string field invalid");
  return value.string();
}
Sha256Digest Digest(const JsonValue& object, std::string_view name) {
  const auto text = Text(object, name);
  const auto digest = Require(Sha256Digest::ParseHex(text));
  Check(digest != Sha256Digest{} && digest.hex() == text, "generation pointer digest invalid");
  return digest;
}
JsonValue String(std::string_view value) { return JsonValue(std::string(value)); }
}  // namespace
Result<GenerationPointer> EncodeGenerationPointer(const GenerationPointerInput& input) {
  try {
    Check(input.artifact_root != Sha256Digest{} && input.receipt_root != Sha256Digest{} &&
        input.catalog_root != Sha256Digest{} && input.activation_ordinal > 0 &&
        input.activation_ordinal <= kMaximumOrdinal &&
        (input.previous_pointer_root.has_value() == (input.activation_ordinal > 1)) &&
        (!input.previous_pointer_root || *input.previous_pointer_root != Sha256Digest{}),
        "generation pointer authority or predecessor geometry invalid");
    auto hash = Require(CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-generation-pointer:v1", 7));
    Require(hash.add_bytes(1, std::as_bytes(std::span(kAbi))));
    Require(hash.add_hash(2, input.artifact_root));
    Require(hash.add_hash(3, input.receipt_root));
    Require(hash.add_hash(4, input.catalog_root));
    Require(hash.add_u64(5, input.activation_ordinal));
    if (input.previous_pointer_root) Require(hash.add_hash(6, *input.previous_pointer_root));
    else Require(hash.add_bytes(6, {}));
    Require(hash.add_bytes(7, std::as_bytes(std::span(kSupport))));
    const auto root = Require(hash.finalize());
    JsonValue::Object fields{{"schema", String(kSchema)}, {"publication_abi", String(kAbi)},
        {"support_state", String(kSupport)}, {"artifact_root", String(input.artifact_root.hex())},
        {"generation_receipt_root", String(input.receipt_root.hex())}, {"catalog_root", String(input.catalog_root.hex())},
        {"activation_ordinal", JsonValue(static_cast<std::int64_t>(input.activation_ordinal))},
        {"previous_pointer_root", input.previous_pointer_root ? String(input.previous_pointer_root->hex()) : JsonValue(nullptr)},
        {"pointer_root", String(root.hex())}};
    auto json = Require(canonical_ascii_json(JsonValue(std::move(fields)), kMaximumBytes));
    return GenerationPointer{input, root, std::move(json)};
  } catch (const std::exception& error) { return Status::InvalidArgument(error.what()); }
}
Result<GenerationPointer> ParseGenerationPointer(std::string_view json) {
  try {
    JsonLimits limits;
    limits.max_input_bytes = kMaximumBytes;
    limits.max_nodes = 32;
    limits.max_depth = 2;
    limits.max_string_bytes = 1024;
    const auto document = Require(JsonValue::Parse(json, limits));
    Check(document.is_object() && document.object().size() == 9 &&
        Text(document, "schema") == kSchema && Text(document, "publication_abi") == kAbi &&
        Text(document, "support_state") == kSupport, "generation pointer schema/field set invalid");
    const auto& ordinal = Field(document, "activation_ordinal");
    Check(ordinal.is_integer() && ordinal.integer() > 0, "generation pointer ordinal invalid");
    GenerationPointerInput input{Digest(document, "artifact_root"), Digest(document, "generation_receipt_root"),
        Digest(document, "catalog_root"), static_cast<std::uint64_t>(ordinal.integer()), {}};
    if (!Field(document, "previous_pointer_root").is_null()) input.previous_pointer_root = Digest(document, "previous_pointer_root");
    auto expected = Require(EncodeGenerationPointer(input));
    Check(expected.pointer_root == Digest(document, "pointer_root") && expected.json == json,
        "generation pointer root or canonical bytes differ");
    return expected;
  } catch (const std::exception& error) { return Status::InvalidArgument(error.what()); }
}
Result<GenerationPointer> NextGenerationPointer(const Sha256Digest& artifact_root,
    const Sha256Digest& receipt_root, const Sha256Digest& catalog_root, std::uint64_t ordinal,
    const std::optional<GenerationPointer>& previous) {
  GenerationPointerInput input{artifact_root, receipt_root, catalog_root, ordinal, {}};
  if (previous) {
    auto parsed = ParseGenerationPointer(previous->json);
    if (!parsed.ok()) return parsed.status();
    auto rebuilt = EncodeGenerationPointer(previous->input);
    if (!rebuilt.ok()) return rebuilt.status();
    if (parsed->pointer_root != previous->pointer_root || rebuilt->json != previous->json ||
        parsed->input.activation_ordinal == kMaximumOrdinal || ordinal != parsed->input.activation_ordinal + 1)
      return Status::FailedPrecondition("activation does not extend the supplied current pointer by one");
    input.previous_pointer_root = parsed->pointer_root;
  } else if (ordinal != 1) {
    return Status::FailedPrecondition("initial activation ordinal must be one");
  }
  return EncodeGenerationPointer(input);
}
Result<std::optional<GenerationPointer>> ReadCurrentGenerationPointer(const GenerationStore& store) {
  try {
    Require(store.Revalidate());
    const auto parent = store.pointers_descriptor();
    const auto fd = ::openat(parent, "current.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
      Check(errno == ENOENT, "cannot open current generation pointer");
      Require(store.Revalidate());
      struct stat named{};
      Check(::fstatat(parent, "current.json", &named, AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT,
          "current pointer appeared during read");
      return std::optional<GenerationPointer>{};
    }
    struct OwnedFd final {
      int value;
      ~OwnedFd() { ::close(value); }
    } owned{fd};
    struct stat before{}, directory{};
    Check(::fstat(fd, &before) == 0 && ::fstat(parent, &directory) == 0 && S_ISREG(before.st_mode) &&
        before.st_uid == ::geteuid() && before.st_nlink == 1 && before.st_dev == directory.st_dev &&
        (before.st_mode & (S_IWGRP | S_IWOTH)) == 0 && before.st_size > 0 &&
        static_cast<std::uint64_t>(before.st_size) <= kMaximumBytes,
        "current generation pointer identity or bounds invalid");
    std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto count = ::pread(fd, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Check(count > 0, "current generation pointer read failed");
      offset += static_cast<std::size_t>(count);
    }
    auto parsed = Require(ParseGenerationPointer(bytes));
    Require(store.Revalidate());
    struct stat after{}, named{};
    const auto same = [](const struct stat& a, const struct stat& b) {
      return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
          a.st_uid == b.st_uid && a.st_nlink == b.st_nlink && a.st_size == b.st_size &&
          a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
          a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
    };
    Check(::fstat(fd, &after) == 0 && ::fstatat(parent, "current.json", &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        same(before, after) && same(before, named), "current generation pointer changed or was replaced");
    return std::optional<GenerationPointer>{std::move(parsed)};
  } catch (const std::exception& error) { return Status::FailedPrecondition(error.what()); }
}
}  // namespace pih::offline_deepseek
