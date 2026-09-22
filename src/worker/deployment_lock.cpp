#include "worker/deployment_lock.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "pih/core/sha256.h"
#include "pih/plugin_sdk/capability.h"

namespace pih::worker {
namespace {

constexpr std::uintmax_t kMaximumDevelopmentLockBytes = 1024 * 1024;
constexpr std::size_t kMaximumIdentifierBytes = 256;
constexpr std::size_t kMaximumVersionBytes = 64;
constexpr std::size_t kMaximumArchitectureBytes = 64;
constexpr std::size_t kMaximumEngineConfigurationBytes = 16 * 1024;

bool IsAbsolute(const std::string& path) {
#if defined(_WIN32)
  return path.size() >= 3 &&
         ((path[0] >= 'A' && path[0] <= 'Z') ||
          (path[0] >= 'a' && path[0] <= 'z')) &&
         path[1] == ':' && (path[2] == '/' || path[2] == '\\');
#else
  return !path.empty() && path.front() == '/';
#endif
}

bool SafeId(const std::string& value, std::size_t maximum_bytes) {
  if (value.empty() || value.size() > maximum_bytes || value.front() == '.' ||
      value.back() == '.' ||
      value.find("..") != std::string::npos) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= 'a' && character <= 'z') ||
          (character >= '0' && character <= '9') || character == '.' ||
          character == '-')) {
      return false;
    }
  }
  return true;
}

bool SafePluginEntrypoint(const std::string& value) {
  if (value.empty() || value.size() > 4096 || value.back() == '/' ||
      value.find('\\') != std::string::npos || IsAbsolute(value)) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= 'A' && character <= 'Z') ||
          (character >= 'a' && character <= 'z') ||
          (character >= '0' && character <= '9') || character == '.' ||
          character == '-' || character == '_' || character == '/')) {
      return false;
    }
  }
  const std::filesystem::path path(value);
  if (path.has_root_path() || path.lexically_normal().generic_string() != value) {
    return false;
  }
  auto component = path.begin();
  if (component == path.end() || *component != "lib") return false;
  ++component;
  if (component == path.end() || component->empty() || *component == "." ||
      *component == "..") {
    return false;
  }
  ++component;
  return component == path.end();
}

bool CanonicalNonzeroSha256Hex(std::string_view value) {
  return value.size() == 64 &&
         std::ranges::all_of(value, [](char byte) {
           return (byte >= '0' && byte <= '9') ||
                  (byte >= 'a' && byte <= 'f');
         }) &&
         std::ranges::any_of(value, [](char byte) { return byte != '0'; });
}

void VerifyArtifactDigest(const std::string& path,
                          const std::string& expected_hex,
                          const char* failure) {
  if (expected_hex.empty()) return;
  const auto expected = pih::Sha256Digest::ParseHex(expected_hex);
  if (!expected.ok()) throw std::invalid_argument(failure);
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error(failure);
  pih::Sha256 digest;
  std::array<char, 1024 * 1024> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = stream.gcount();
    if (count > 0) {
      const auto status = digest.update(std::as_bytes(std::span(
          buffer.data(), static_cast<std::size_t>(count))));
      if (!status.ok()) throw std::runtime_error(failure);
    }
  }
  if (!stream.eof()) throw std::runtime_error(failure);
  const auto observed = digest.finalize();
  if (!observed.ok() || *observed != *expected) {
    throw std::runtime_error(failure);
  }
}

void Expect(const std::string& bytes, std::size_t& cursor,
            const char* literal) {
  const std::string expected(literal);
  if (bytes.compare(cursor, expected.size(), expected) != 0) {
    throw std::invalid_argument("development_lock_schema_invalid");
  }
  cursor += expected.size();
}

std::string ReadString(const std::string& bytes, std::size_t& cursor) {
  const auto end = bytes.find('"', cursor);
  if (end == std::string::npos) {
    throw std::invalid_argument("development_lock_string_invalid");
  }
  auto value = bytes.substr(cursor, end - cursor);
  cursor = end + 1;
  return value;
}

std::string ReadObject(const std::string& bytes, std::size_t& cursor) {
  if (cursor >= bytes.size() || bytes[cursor] != '{') {
    throw std::invalid_argument("development_lock_object_invalid");
  }
  const auto begin = cursor;
  std::size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (; cursor < bytes.size(); ++cursor) {
    const char byte = bytes[cursor];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (byte == '\\') {
        escaped = true;
      } else if (byte == '"') {
        in_string = false;
      }
      continue;
    }
    if (byte == '"') {
      in_string = true;
    } else if (byte == '{') {
      ++depth;
    } else if (byte == '}') {
      if (--depth == 0) {
        ++cursor;
        return bytes.substr(begin, cursor - begin);
      }
    }
  }
  throw std::invalid_argument("development_lock_object_invalid");
}

uint64_t ReadUint64(const std::string& bytes, std::size_t& cursor) {
  if (cursor >= bytes.size() || bytes[cursor] < '0' || bytes[cursor] > '9') {
    throw std::invalid_argument("development_lock_integer_invalid");
  }
  uint64_t value = 0;
  while (cursor < bytes.size() && bytes[cursor] >= '0' &&
         bytes[cursor] <= '9') {
    const auto digit = static_cast<uint64_t>(bytes[cursor] - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
      throw std::invalid_argument("development_lock_integer_overflow");
    }
    value = value * 10 + digit;
    ++cursor;
  }
  return value;
}

std::string DirectoryOf(const std::string& path) {
  const std::filesystem::path filesystem_path(path);
  const auto directory = filesystem_path.parent_path();
  if (directory.empty() || !directory.is_absolute()) {
    throw std::invalid_argument("development_lock_path_invalid");
  }
  return directory.lexically_normal().string();
}

std::string ResolveArtifactPath(const std::string& directory,
                                const std::string& relative_path) {
  const auto resolved =
      (std::filesystem::path(directory) / relative_path).lexically_normal();
  if (!resolved.is_absolute()) {
    throw std::invalid_argument("development_lock_artifact_path_invalid");
  }
  return resolved.string();
}

}  // namespace

DevelopmentLock LoadDevelopmentLock(const std::string& absolute_path) {
  if (!IsAbsolute(absolute_path)) {
    throw std::invalid_argument("development_lock_path_must_be_absolute");
  }
  std::error_code lock_error;
  const std::filesystem::path lock_path(absolute_path);
  const auto lock_status = std::filesystem::symlink_status(lock_path,
                                                           lock_error);
  if (lock_error || !std::filesystem::is_regular_file(lock_status)) {
    throw std::runtime_error("development_lock_not_regular_file");
  }
  const auto lock_bytes = std::filesystem::file_size(lock_path, lock_error);
  if (lock_error || lock_bytes == 0 ||
      lock_bytes > kMaximumDevelopmentLockBytes) {
    throw std::runtime_error("development_lock_size_invalid");
  }
  std::ifstream stream(absolute_path, std::ios::binary);
  if (!stream) throw std::runtime_error("development_lock_open_failed");
  const std::string bytes((std::istreambuf_iterator<char>(stream)),
                          std::istreambuf_iterator<char>());
  return ParseDevelopmentLock(bytes, absolute_path);
}

DevelopmentLock ParseDevelopmentLock(const std::string& bytes, const std::string& absolute_path) {
  if (!IsAbsolute(absolute_path)) throw std::invalid_argument("development_lock_path_must_be_absolute");
  if (bytes.empty() || bytes.size() > kMaximumDevelopmentLockBytes)
    throw std::invalid_argument("development_lock_size_invalid");
  std::size_t cursor = 0;
  Expect(bytes, cursor,
         "{\"schema\":\"pih.development-lock.v1\","
         "\"support_status\":\"unsupported\",\"kernel_packs\":[");

  DevelopmentLock lock;
  const auto directory = DirectoryOf(absolute_path);
  lock.deployment_root = directory;
  std::unordered_set<std::string> artifact_ids;
  std::unordered_set<std::string> artifact_paths;
  while (cursor < bytes.size() && bytes[cursor] != ']') {
    if (!lock.kernel_packs.empty()) Expect(bytes, cursor, ",");
    Expect(bytes, cursor, "{\"pack_id\":\"");
    auto pack_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"pack_version\":\"");
    auto pack_version = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"pack_abi\":\"");
    auto pack_abi = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"architecture\":\"");
    auto architecture = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"binary\":\"");
    auto binary = ReadString(bytes, cursor);
    std::string binary_sha256_hex;
    constexpr std::string_view kBinaryDigestField =
        ",\"binary_sha256_hex\":\"";
    if (bytes.compare(cursor, kBinaryDigestField.size(),
                      kBinaryDigestField) == 0) {
      Expect(bytes, cursor, ",\"binary_sha256_hex\":\"");
      binary_sha256_hex = ReadString(bytes, cursor);
    }
    Expect(bytes, cursor, "}");
    if (!SafeId(pack_id, kMaximumIdentifierBytes) ||
        !SafeId(pack_version, kMaximumVersionBytes) ||
        !SafeId(pack_abi, kMaximumIdentifierBytes) ||
        !SafeId(architecture, kMaximumArchitectureBytes) ||
        !SafePluginEntrypoint(binary) ||
        (!binary_sha256_hex.empty() &&
         !CanonicalNonzeroSha256Hex(binary_sha256_hex))) {
      throw std::invalid_argument("development_lock_kernel_pack_invalid");
    }
    binary = ResolveArtifactPath(directory, binary);
    if (!artifact_ids.insert(pack_id).second) {
      throw std::invalid_argument("development_lock_artifact_id_duplicated");
    }
    if (!artifact_paths.insert(binary).second) {
      throw std::invalid_argument("development_lock_artifact_path_duplicated");
    }
    lock.kernel_packs.push_back(
        {std::move(pack_id), std::move(pack_version), std::move(pack_abi),
         std::move(architecture), std::move(binary),
         std::move(binary_sha256_hex)});
  }
  Expect(bytes, cursor, "],\"plugins\":[");
  std::unordered_set<std::string> plugin_ids;
  std::unordered_set<std::string> plugin_entrypoints;
  while (cursor < bytes.size() && bytes[cursor] != ']') {
    if (!lock.plugins.empty()) Expect(bytes, cursor, ",");
    Expect(bytes, cursor, "{\"plugin_id\":\"");
    auto plugin_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"plugin_version\":\"");
    auto plugin_version = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"entrypoint\":\"");
    auto entrypoint = ReadString(bytes, cursor);
    std::string entrypoint_sha256_hex;
    constexpr std::string_view kEntrypointDigestField =
        ",\"entrypoint_sha256_hex\":\"";
    if (bytes.compare(cursor, kEntrypointDigestField.size(),
                      kEntrypointDigestField) == 0) {
      Expect(bytes, cursor, ",\"entrypoint_sha256_hex\":\"");
      entrypoint_sha256_hex = ReadString(bytes, cursor);
    }
    Expect(bytes, cursor, "}");
    if (!SafeId(plugin_id, kMaximumIdentifierBytes) ||
        !SafeId(plugin_version, kMaximumVersionBytes) ||
        !SafePluginEntrypoint(entrypoint) ||
        (!entrypoint_sha256_hex.empty() &&
         !CanonicalNonzeroSha256Hex(entrypoint_sha256_hex))) {
      throw std::invalid_argument("development_lock_value_invalid");
    }
    entrypoint = ResolveArtifactPath(directory, entrypoint);
    if (!artifact_ids.insert(plugin_id).second) {
      throw std::invalid_argument("development_lock_artifact_id_duplicated");
    }
    if (!artifact_paths.insert(entrypoint).second) {
      throw std::invalid_argument("development_lock_artifact_path_duplicated");
    }
    if (!plugin_ids.insert(plugin_id).second) {
      throw std::invalid_argument("development_lock_plugin_id_duplicated");
    }
    if (!plugin_entrypoints.insert(entrypoint).second) {
      throw std::invalid_argument(
          "development_lock_plugin_entrypoint_duplicated");
    }
    lock.plugins.push_back({std::move(plugin_id), std::move(plugin_version),
                            std::move(entrypoint),
                            std::move(entrypoint_sha256_hex)});
  }
  Expect(bytes, cursor, "],\"capabilities\":[");
  std::unordered_set<std::string> capability_ids;
  while (cursor < bytes.size() && bytes[cursor] != ']') {
    if (!lock.capabilities.empty()) Expect(bytes, cursor, ",");
    Expect(bytes, cursor, "{\"capability_id\":\"");
    auto capability_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"provider_id\":\"");
    auto provider_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"contract_id\":\"");
    auto contract_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"threading_model\":");
    const auto threading_model = ReadUint64(bytes, cursor);
    Expect(bytes, cursor, ",\"scope\":");
    const auto scope = ReadUint64(bytes, cursor);
    Expect(bytes, cursor, ",\"cardinality\":");
    const auto cardinality = ReadUint64(bytes, cursor);
    Expect(bytes, cursor, "}");
    if (!SafeId(capability_id, kMaximumIdentifierBytes) ||
        !SafeId(provider_id, kMaximumIdentifierBytes) ||
        !SafeId(contract_id, kMaximumIdentifierBytes) ||
        !artifact_ids.contains(provider_id) ||
        threading_model < PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1 ||
        threading_model > PIH_CAPABILITY_THREADING_CONCURRENT_V1 ||
        scope < PIH_CAPABILITY_SCOPE_PROCESS_V1 ||
        scope > PIH_CAPABILITY_SCOPE_REQUEST_V1 ||
        cardinality != PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1 ||
        !capability_ids.insert(capability_id).second ||
        (!lock.capabilities.empty() &&
         lock.capabilities.back().capability_id >= capability_id)) {
      throw std::invalid_argument("development_lock_capability_invalid");
    }
    lock.capabilities.push_back(
        {std::move(capability_id), std::move(provider_id),
         std::move(contract_id), static_cast<std::uint32_t>(threading_model),
         static_cast<std::uint32_t>(scope),
         static_cast<std::uint32_t>(cardinality)});
  }
  Expect(bytes, cursor, "]");
  if (cursor < bytes.size() && bytes[cursor] == '}') {
    ++cursor;
  } else {
    Expect(bytes, cursor, ",\"engine\":{\"capability_id\":\"");
    lock.engine.capability_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"contract_id\":\"");
    lock.engine.contract_id = ReadString(bytes, cursor);
    Expect(bytes, cursor, ",\"activation_epoch\":");
    lock.engine.activation_epoch = ReadUint64(bytes, cursor);
    Expect(bytes, cursor, ",\"configuration\":");
    lock.engine.configuration_json = ReadObject(bytes, cursor);
    Expect(bytes, cursor, ",\"smoke_http_body\":");
    lock.engine.smoke_http_body_json = ReadObject(bytes, cursor);
    Expect(bytes, cursor, "}}");
    if (!SafeId(lock.engine.capability_id, kMaximumIdentifierBytes) ||
        !SafeId(lock.engine.contract_id, kMaximumIdentifierBytes) ||
        lock.engine.activation_epoch == 0 ||
        lock.engine.configuration_json.size() >
            kMaximumEngineConfigurationBytes ||
        lock.engine.smoke_http_body_json.size() > 64 * 1024) {
      throw std::invalid_argument("development_lock_engine_invalid");
    }
    const auto engine_capability = std::find_if(
        lock.capabilities.begin(), lock.capabilities.end(),
        [&](const LockedCapability& capability) {
          return capability.capability_id == lock.engine.capability_id &&
                 capability.contract_id == lock.engine.contract_id &&
                 capability.scope <= PIH_CAPABILITY_SCOPE_ACTIVATION_V1 &&
                 capability.cardinality ==
                     PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
        });
    if (engine_capability == lock.capabilities.end()) {
      throw std::invalid_argument(
          "development_lock_engine_capability_missing");
    }
    lock.has_engine = true;
  }
  if (cursor + 1 == bytes.size() && bytes[cursor] == '\n') {
    ++cursor;
  } else if (cursor + 2 == bytes.size() && bytes[cursor] == '\r' &&
             bytes[cursor + 1] == '\n') {
    cursor += 2;
  }
  if (cursor != bytes.size() || lock.plugins.empty()) {
    throw std::invalid_argument("development_lock_not_canonical");
  }
  if (lock.has_engine &&
      (std::ranges::any_of(lock.kernel_packs, [](const auto& pack) {
         return pack.binary_sha256_hex.empty();
       }) ||
       std::ranges::any_of(lock.plugins, [](const auto& plugin) {
         return plugin.entrypoint_sha256_hex.empty();
       }))) {
    throw std::invalid_argument("development_lock_engine_artifact_unsealed");
  }
  for (const auto& pack : lock.kernel_packs) {
    std::error_code error;
    const auto file_status = std::filesystem::symlink_status(pack.binary, error);
    if (error || !std::filesystem::is_regular_file(file_status)) {
      throw std::runtime_error("development_lock_kernel_pack_missing");
    }
    VerifyArtifactDigest(pack.binary, pack.binary_sha256_hex,
                         "development_lock_kernel_pack_digest_mismatch");
  }
  for (const auto& plugin : lock.plugins) {
    std::error_code error;
    const auto file_status =
        std::filesystem::symlink_status(plugin.entrypoint, error);
    if (error || !std::filesystem::is_regular_file(file_status)) {
      throw std::runtime_error("development_lock_plugin_entrypoint_missing");
    }
    VerifyArtifactDigest(plugin.entrypoint, plugin.entrypoint_sha256_hex,
                         "development_lock_plugin_digest_mismatch");
  }
  return lock;
}

}  // namespace pih::worker
