#include "weight_manifest.h"
#include "pih/core/bounded_json.h"
#include <new>

namespace pih::deepseek_v41 {
Result<std::vector<ExpectedWeightShard>> ParseWeightManifest(std::string_view json,
    const Sha256Digest& expected, const FlashConfig& config, std::uint32_t world, std::uint32_t rank) {
  if (json.empty() || json.size() > kWeightManifestMaximumBytes || expected == Sha256Digest{} ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Weight manifest extent, digest or rank invalid");
  const auto digest = sha256(std::as_bytes(std::span(json.data(), json.size())));
  if (!digest.ok()) return digest.status();
  if (*digest != expected) return Status::FailedPrecondition("Weight manifest does not match admitted digest");
  try {
    const auto parsed = JsonValue::Parse(json, {kWeightManifestMaximumBytes, 4, 2048, 256});
    if (!parsed.ok()) return parsed.status();
    if (!parsed->is_object() || parsed->object().size() != 5)
      return Status::InvalidArgument("Weight manifest root schema invalid");
    const auto* schema = parsed->at("schema"); const auto* hash = parsed->at("config_sha256");
    const auto* size = parsed->at("world_size"); const auto* index = parsed->at("rank");
    const auto* shards = parsed->at("shards");
    if (!schema || !schema->is_string() || schema->string() != "pih.deepseek-v41.weights.v1" ||
        !hash || !hash->is_string() || !size || !size->is_integer() || size->integer() != world ||
        !index || !index->is_integer() || index->integer() != rank || !shards || !shards->is_array() ||
        shards->array().empty() || shards->array().size() > BackboneWeightCatalog::kMaximumShards)
      return Status::InvalidArgument("Weight manifest version, config or topology fields invalid");
    auto configuration = Sha256Digest::ParseHex(hash->string());
    if (!configuration.ok()) return configuration.status();
    if (*configuration != config.config_sha256())
      return Status::FailedPrecondition("Weight manifest belongs to another model configuration");
    std::vector<ExpectedWeightShard> output;
    output.reserve(shards->array().size());
    for (const auto& shard : shards->array()) {
      if (!shard.is_object() || shard.object().size() != 3)
        return Status::InvalidArgument("Weight manifest shard schema invalid");
      const auto* name = shard.at("name"); const auto* bytes = shard.at("bytes"); const auto* hash = shard.at("sha256");
      if (!name || !name->is_string() || !IsWeightShardMember(name->string()) ||
          !bytes || !bytes->is_integer() || bytes->integer() < 8 || !hash || !hash->is_string())
        return Status::InvalidArgument("Weight manifest shard identity or size invalid");
      auto digest = Sha256Digest::ParseHex(hash->string()); if (!digest.ok()) return digest.status();
      if (*digest == Sha256Digest{}) return Status::InvalidArgument("Weight manifest shard digest absent");
      for (const auto& prior : output) if (prior.name == name->string())
        return Status::InvalidArgument("Weight manifest contains duplicate shard names");
      output.push_back({name->string(), static_cast<std::uint64_t>(bytes->integer()), *digest});
    }
    return output;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Weight manifest allocation failed"); }
}
}  // namespace pih::deepseek_v41
