#pragma once

#include "pih/core/bounded_json.h"
#include "pih/model/qwen3_manifest.h"
#include "pih/model/qwen3_source_artifact.h"
#include "pih/model/safetensors_file.h"
#include <fstream>
#include <stdexcept>

namespace pih::qwen_offline {
inline constexpr uint64_t kSourceByteLimit = Qwen3Manifest::kOfficialSourcePayloadBytes +
    SafetensorsHeader::kMaxHeaderBytes + 8;

inline Qwen3SourceArtifactExpectation ReadSourceExpectation(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  if (!size || size > 2048) throw std::invalid_argument("expectation size invalid");
  std::ifstream input(path, std::ios::binary);
  std::string bytes(static_cast<size_t>(size), '\0');
  if (!input.read(bytes.data(), bytes.size()) || input.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("expectation read failed or size changed");
  auto value = JsonValue::Parse(bytes, {2048, 2, 16, 128});
  if (!value.ok() || !value->is_object() || value->object().size() != 5)
    throw std::invalid_argument("expectation fields invalid");
  auto field = [&](const char* name) -> const JsonValue& {
    const auto* member = value->at(name);
    if (!member) throw std::invalid_argument("expectation field missing");
    return *member;
  };
  const auto& schema = field("schema");
  if (!schema.is_string() || schema.string() != "pih.qwen3_source_artifact_expectation.v1")
    throw std::invalid_argument("expectation schema invalid");
  auto integer = [&](const char* name, uint64_t maximum) -> uint64_t {
    const auto& member = field(name);
    if (!member.is_integer() || member.integer() <= 0 ||
        static_cast<uint64_t>(member.integer()) > maximum)
      throw std::invalid_argument("expectation integer invalid");
    return static_cast<uint64_t>(member.integer());
  };
  const auto& hash = field("file_sha256");
  if (!hash.is_string()) throw std::invalid_argument("expectation digest must be a string");
  const auto digest = Sha256Digest::ParseHex(hash.string());
  if (!digest.ok() || *digest == Sha256Digest{} || digest->hex() != hash.string())
    throw std::invalid_argument("expectation digest must be nonzero lowercase SHA-256");
  return {integer("file_bytes", kSourceByteLimit), static_cast<size_t>(integer("tensor_count", 4096)),
          integer("data_bytes", Qwen3Manifest::kOfficialSourcePayloadBytes), *digest};
}
}  // namespace pih::qwen_offline
