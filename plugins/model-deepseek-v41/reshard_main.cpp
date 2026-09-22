#include "weight_materialize.h"
#include "engram_hash.h"
#include "pih/core/canonical_json.h"
#include <charconv>
#include <cerrno>
#include <cstdio>
#include <csignal>
#include <fcntl.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
using namespace pih;
using namespace pih::deepseek_v41;
#ifdef PIH_V41_CONVERT_COMMAND
constexpr bool kConvert = true;
#else
constexpr bool kConvert = false;
#endif
struct Fd { int value = -1; ~Fd() { if (value >= 0) ::close(value); } };
void Require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void Check(const Status& status) { if (!status.ok()) throw std::runtime_error(std::string(status.message())); }
template<class T> T Take(Result<T> result) { Check(result.status()); return std::move(*result); }
std::uint64_t Number(std::string_view text) {
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  Require(!text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "invalid unsigned decimal argument");
  return result;
}
Sha256Digest Digest(std::string_view text) {
  auto digest = Take(Sha256Digest::ParseHex(text));
  Require(digest != Sha256Digest{} && digest.hex() == text, "expected nonzero lowercase trusted SHA-256");
  return digest;
}
int Directory(const std::filesystem::path& path) {
  Require(path.is_absolute() && path.string().size() <= 4096, "directory must be a bounded absolute path");
  Fd current{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  Require(current.value >= 0, "cannot open filesystem root");
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    Require(!name.empty() && name != "." && name != "..", "directory path must be canonical");
    Fd next{::openat(current.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    Require(next.value >= 0, "cannot traverse directory without symlinks");
    std::swap(next.value, current.value);
  }
  const int result = current.value; current.value = -1; return result;
}
void Absent(int parent, const std::string& name) {
  struct stat st{};
  const int result = ::fstatat(parent, name.c_str(), &st, AT_SYMLINK_NOFOLLOW);
  Require(result < 0 && errno == ENOENT, "output or staging already exists, or cannot be inspected");
}
void Usage() {
  if (kConvert) {
    std::fputs("Usage: pih-v41-convert SOURCE_DIRECTORY EXPECTATIONS_JSON EXPECTATIONS_SHA256 EXCLUSIONS_JSON EXCLUSIONS_SHA256 DEVICE_BUDGET_BYTES OUTPUT_DIRECTORY\n"
      "Converts admitted text-backbone types to canonical TP1; no model execution.\n"
        "Optional suffix: --runtime-map MAP_FILE TRUSTED_MAP_SHA256 (assemble worker metadata).\n"
        "All paths absolute. Trusted digests required. Output and OUTPUT_DIRECTORY.staging must not exist.\n", stderr);
    return;
  }
  std::fputs("Usage: pih-v41-reshard CONFIG CONFIG_SHA256 SOURCE_DIRECTORY SOURCE_MANIFEST_SHA256 WORLD RANK DEVICE_BUDGET_BYTES OUTPUT_DIRECTORY\n"
      "Canonical TP1 text weights only; not raw checkpoint conversion. Linux CPU-only.\n"
      "Optional suffix: --runtime-map MAP_FILE TRUSTED_MAP_SHA256 (assemble worker metadata).\n"
      "All paths absolute. Trusted digests required. Output and OUTPUT_DIRECTORY.staging must not exist.\n", stderr);
}
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") { Usage(); return 0; }
  const int base_argc = kConvert ? 8 : 9;
  if (argc != base_argc && argc != base_argc + 3) { Usage(); return 2; }
  const bool runtime_assets = argc == base_argc + 3;
  bool published = false;
  try {
    Require(std::signal(SIGPIPE, SIG_IGN) != SIG_ERR, "cannot configure receipt pipe error handling");
    Sha256Digest map_digest;
    std::vector<std::byte> map_bytes;
    if (runtime_assets) {
      Require(std::string_view(argv[base_argc]) == "--runtime-map", "unknown offline command suffix");
      map_digest = Digest(argv[base_argc + 2]);
      map_bytes = Take(ReadAuthenticatedMetadata(argv[base_argc + 1], map_digest, FlashConfig::kVocabularySize * 4ULL));
    }
    Sha256Digest config_digest, exclusions_digest;
    const auto source_digest = Digest(argv[kConvert ? 3 : 4]);
    const auto world = kConvert ? std::uint64_t{1} : Number(argv[5]);
    const auto rank = kConvert ? std::uint64_t{0} : Number(argv[6]);
    const auto budget = Number(argv[kConvert ? 6 : 7]);
    Require((world == 1 || world == 2 || world == 4 || world == 8) && rank < world && budget > 0,
        "require TP1/2/4/8, rank below world and positive device budget");
    const std::filesystem::path output(argv[kConvert ? 7 : 8]);
    Require(output.is_absolute() && output.string().size() <= 4096, "output must be a bounded absolute path");
    const auto name = output.filename().string();
    Require(!name.empty() && name != "." && name != ".." && name.size() <= 200,
        "output must name a directory member, not a root or trailing slash");
    const auto staging = name + ".staging";
    Fd parent{Directory(output.parent_path())};
    struct stat parent_state{};
    Require(::fstat(parent.value, &parent_state) == 0 && parent_state.st_uid == ::geteuid() &&
        !(parent_state.st_mode & 0022), "output parent must be owned and not group/other writable");
    Absent(parent.value, name); Absent(parent.value, staging);
    std::optional<FlashConfig> config;
    std::string config_text;
    std::unique_ptr<BackboneWeightFiles> source;
    std::unique_ptr<WeightSourceCheckpoint> raw_source;
    std::vector<std::string> exclusions;
    if (kConvert) {
      exclusions_digest = Digest(argv[5]);
      auto exclusions_bytes = Take(ReadAuthenticatedMetadata(argv[4], exclusions_digest, 16 * 1024 * 1024));
      auto exclusion_json = Take(JsonValue::Parse(std::string_view(
          reinterpret_cast<const char*>(exclusions_bytes.data()), exclusions_bytes.size()),
          {16 * 1024 * 1024, 2, 200004, 512}));
      const auto* schema = exclusion_json.at("schema");
      const auto* names = exclusion_json.at("excluded_source_names");
      Require(exclusion_json.is_object() && exclusion_json.object().size() == 2 &&
          schema && schema->is_string() && schema->string() == "pih.deepseek-v41.conversion-exclusions.v1" &&
          names && names->is_array() && names->array().size() <= 200000,
          "exclusion document schema/count invalid");
      for (const auto& name : names->array()) {
        Require(name.is_string() && !name.string().empty(), "exclusion must be a nonempty normalized tensor name");
        exclusions.push_back(name.string());
      }
      auto expectation_bytes = Take(ReadAuthenticatedMetadata(argv[2], source_digest,
          WeightSourceCheckpoint::kMaximumExpectationBytes));
      Fd root{Directory(argv[1])};
      raw_source = Take(WeightSourceCheckpoint::Open(root.value, std::string_view(
          reinterpret_cast<const char*>(expectation_bytes.data()), expectation_bytes.size()), source_digest));
      config_digest = raw_source->config().config_sha256();
      config.emplace(raw_source->config());
      if (runtime_assets) config_text = raw_source->config_json();
      Check(raw_source->ValidateBackboneConversion(exclusions));
    } else {
      config_digest = Digest(argv[2]);
      auto config_bytes = Take(ReadAuthenticatedMetadata(argv[1], config_digest, FlashConfig::kMaximumBytes));
      config.emplace(Take(FlashConfig::Parse(std::string_view(
          reinterpret_cast<const char*>(config_bytes.data()), config_bytes.size()))));
      if (runtime_assets) config_text.assign(reinterpret_cast<const char*>(config_bytes.data()), config_bytes.size());
      source = Take(BackboneWeightFiles::OpenManifest(argv[3], source_digest, *config, 1, 0,
          std::numeric_limits<std::uint64_t>::max()));
    }
    // Source authentication completes before creating any destination files.
    if (runtime_assets) {
      auto hashes = EngramHashState::Create(*config, map_bytes, map_digest);
      Check(hashes.status());
    }
    Require(::mkdirat(parent.value, staging.c_str(), 0700) == 0, "cannot exclusively create staging directory");
    Fd destination{::openat(parent.value, staging.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    Require(destination.value >= 0, "cannot open new staging directory; inspect it before retrying");
    auto receipt = kConvert ? Take(MaterializeSourceBackbone(destination.value, *raw_source, exclusions, budget)) :
        Take(MaterializeCanonicalWeightRank(destination.value, *source, *config,
            static_cast<std::uint32_t>(world), static_cast<std::uint32_t>(rank), budget));
    std::uint64_t runtime_metadata_bytes = 0;
    if (runtime_assets) {
      runtime_metadata_bytes = Take(MaterializeRuntimeMetadata(destination.value, config_text,
          config_digest, map_bytes, map_digest));
      receipt.file_bytes += runtime_metadata_bytes;
    }
    Require(::fchmod(destination.value, 0500) == 0 && ::fsync(destination.value) == 0,
        "cannot sync read-only staging directory");
    struct stat held{}, named{};
    Require(::fstat(destination.value, &held) == 0 &&
        ::fstatat(parent.value, staging.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        held.st_dev == named.st_dev && held.st_ino == named.st_ino, "staging directory identity changed");
    Fd current_parent{Directory(output.parent_path())};
    struct stat current_state{};
    Require(::fstat(current_parent.value, &current_state) == 0 &&
        current_state.st_dev == parent_state.st_dev && current_state.st_ino == parent_state.st_ino,
        "output parent identity changed");
    // No replace, no cross-filesystem copy and no fallback to overwrite-capable rename.
    Require(::syscall(SYS_renameat2, parent.value, staging.c_str(), parent.value, name.c_str(), 1U) == 0,
        "no-replace directory publication failed; inspect retained staging");
    published = true;
    Require(::fsync(parent.value) == 0, "directory renamed but parent sync failed; durability unresolved");
    JsonValue::Object fields{
        {"schema", JsonValue(std::string(kConvert ? "pih.deepseek-v41.conversion-receipt.v1" : "pih.deepseek-v41.canonical-reshard-receipt.v1"))},
        {"config_sha256", JsonValue(config_digest.hex())},
        {kConvert ? "source_expectations_sha256" : "source_manifest_sha256", JsonValue(source_digest.hex())},
        {"manifest_sha256", JsonValue(receipt.manifest_sha256.hex())},
        {"world_size", JsonValue(static_cast<std::int64_t>(world))},
        {"rank", JsonValue(static_cast<std::int64_t>(rank))},
        {"file_bytes", JsonValue(static_cast<std::int64_t>(receipt.file_bytes))},
        {"shard_count", JsonValue(static_cast<std::int64_t>(receipt.shard_count))}};
    if (kConvert) {
      fields.emplace_back("exclusions_sha256", JsonValue(exclusions_digest.hex()));
      fields.emplace_back("provenance_sha256", JsonValue(receipt.provenance_sha256.hex()));
    }
    if (runtime_assets) {
      fields.emplace_back("map_sha256", JsonValue(map_digest.hex()));
      fields.emplace_back("runtime_metadata_bytes", JsonValue(static_cast<std::int64_t>(runtime_metadata_bytes)));
    }
    auto json = Take(canonical_ascii_json(JsonValue(std::move(fields)), 4096));
    Require(std::fwrite(json.data(), 1, json.size(), stdout) == json.size() &&
        std::fputc('\n', stdout) != EOF && std::fflush(stdout) == 0,
        "published output exists but receipt delivery failed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "V4.1 offline weights: %s. %s\n", error.what(), published ?
        "Output was renamed; inspect it and do not blindly rerun" :
        "No success receipt; inspect retained staging before retrying (nothing is automatically deleted)");
    return published ? 3 : 2;
  }
}
