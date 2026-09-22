#include "pih/backend/cuda/cubin_artifact_manifest.h"

#include <array>
#include <limits>

#include "pih/core/bounded_json.h"
#include "pih/io/mapped_file.h"

namespace pih {
namespace {

constexpr std::string_view kSchema = "pih.cubin_artifact.v1";
constexpr std::array<std::string_view, 6> kFields{
    "schema",          "target_sm",    "producer_toolkit",
    "producer_flags_sha256", "cubin_sha256", "cubin_bytes"};

bool known_field(std::string_view name) {
  for (const auto field : kFields) {
    if (field == name) return true;
  }
  return false;
}

Result<std::string> required_string(const JsonValue& root,
                                    std::string_view name) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_string() || value->string().empty()) {
    return Status::InvalidArgument(std::string(name) +
                                   " must be a nonempty string");
  }
  return value->string();
}

}  // namespace

Result<CubinArtifactManifest> CubinArtifactManifest::Parse(
    std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaximumManifestBytes;
  limits.max_depth = 4;
  limits.max_nodes = 16;
  limits.max_string_bytes = 128;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!root->is_object() || root->object().size() != kFields.size()) {
    return Status::InvalidArgument(
        "cubin artifact manifest must contain exactly the v1 fields");
  }
  for (const auto& [name, value] : root->object()) {
    (void)value;
    if (!known_field(name)) {
      return Status::InvalidArgument("unknown cubin artifact manifest field");
    }
  }

  auto schema = required_string(*root, "schema");
  auto target = required_string(*root, "target_sm");
  auto toolkit = required_string(*root, "producer_toolkit");
  auto flags = required_string(*root, "producer_flags_sha256");
  auto cubin = required_string(*root, "cubin_sha256");
  if (!schema.ok()) return schema.status();
  if (!target.ok()) return target.status();
  if (!toolkit.ok()) return toolkit.status();
  if (!flags.ok()) return flags.status();
  if (!cubin.ok()) return cubin.status();
  if (schema.value() != kSchema) {
    return Status::InvalidArgument("unsupported cubin artifact schema");
  }
  std::uint32_t target_sm = 0;
  if (target.value() == "sm_89") {
    target_sm = 89;
  } else if (target.value() == "sm_90") {
    target_sm = 90;
  } else {
    return Status::InvalidArgument("cubin target must be sm_89 or sm_90");
  }
  if (toolkit->size() > 64) {
    return Status::InvalidArgument("CUDA toolkit identity is too long");
  }
  auto flags_digest = Sha256Digest::ParseHex(flags.value());
  auto cubin_digest = Sha256Digest::ParseHex(cubin.value());
  if (!flags_digest.ok()) return flags_digest.status();
  if (!cubin_digest.ok()) return cubin_digest.status();
  const auto* bytes = root->at("cubin_bytes");
  if (bytes == nullptr || !bytes->is_integer() || bytes->integer() < 64) {
    return Status::InvalidArgument(
        "cubin_bytes must include at least one ELF64 header");
  }
  return CubinArtifactManifest(
      target_sm, std::move(toolkit).value(), flags_digest.value(),
      cubin_digest.value(), static_cast<std::uint64_t>(bytes->integer()));
}

Result<CubinArtifactManifest> CubinArtifactManifest::Load(
    const std::filesystem::path& path, std::uint64_t maximum_bytes) {
  if (maximum_bytes == 0 || maximum_bytes > kMaximumManifestBytes) {
    return Status::InvalidArgument("cubin manifest byte budget is invalid");
  }
  auto mapped = MappedFile::OpenReadOnly(path, maximum_bytes);
  if (!mapped.ok()) return mapped.status();
  if (mapped->size_bytes() >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("cubin manifest exceeds address space");
  }
  const auto* text = reinterpret_cast<const char*>(mapped->data());
  return Parse(std::string_view(text, static_cast<std::size_t>(mapped->size_bytes())));
}

Result<VerifiedCubin> CubinArtifactManifest::load_cubin(
    const std::filesystem::path& path,
    std::uint64_t configured_maximum_bytes) const {
  if (cubin_bytes_ > configured_maximum_bytes) {
    return Status::ResourceExhausted(
        "manifest cubin size exceeds configured byte budget");
  }
  auto verified = VerifiedCubin::Load(path, cubin_bytes_, cubin_digest_);
  if (!verified.ok()) return verified.status();
  if (verified->bytes().size() != cubin_bytes_) {
    return Status::InvalidArgument("cubin size does not match manifest");
  }
  return std::move(verified).value();
}

}  // namespace pih
