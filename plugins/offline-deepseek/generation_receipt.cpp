#include "generation_receipt.h"
#include "pih/core/bounded_json.h"
#include "pih/core/canonical_hash.h"
#include "pih/core/canonical_json.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace pih::offline_deepseek {
namespace {
constexpr std::string_view kAbi = "content_addressed_directory_generation_v1";
constexpr std::string_view kSupport = "hardware_evidence_open";
constexpr std::size_t kMaximumBytes = 2U << 20;
template<class T> T Require(Result<T> value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.status().message()));
  return std::move(*value);
}
void Require(Status value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.message()));
}
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
const JsonValue& Field(const JsonValue& value, std::string_view name) {
  const auto* result = value.at(name);
  Check(result != nullptr, "receipt projection field absent");
  return *result;
}
std::string Text(const JsonValue& value, std::string_view name) {
  const auto& field = Field(value, name);
  Check(field.is_string(), "receipt projection text field malformed");
  return field.string();
}
std::uint64_t Number(const JsonValue& value, std::string_view name, std::uint64_t maximum) {
  const auto& field = Field(value, name);
  Check(field.is_integer() && field.integer() > 0 &&
      static_cast<std::uint64_t>(field.integer()) <= maximum, "receipt projection integer outside bounds");
  return static_cast<std::uint64_t>(field.integer());
}
Sha256Digest Digest(const JsonValue& value, std::string_view name) {
  const auto text = Text(value, name);
  const auto digest = Require(Sha256Digest::ParseHex(text));
  Check(digest != Sha256Digest{} && digest.hex() == text, "receipt projection digest invalid");
  return digest;
}
JsonValue String(std::string_view value) { return JsonValue(std::string(value)); }
JsonValue Integer(std::uint64_t value) { return JsonValue(static_cast<std::int64_t>(value)); }
}  // namespace

Result<EncodedGenerationReceipt> EncodeGenerationReceipt(const GenerationObservation& observed) {
  try {
    Check(observed.source_payload_equivalence && observed.tensor_count == 67612 &&
        observed.tensor_bytes == 156015698140ULL && observed.shard_count == 44 &&
        observed.artifact_root != Sha256Digest{} && !observed.verification_projection_json.empty(),
        "receipt requires a complete source-bound PP1 observation");
    const auto& bytes = observed.verification_projection_json;
    JsonLimits limits;
    limits.max_input_bytes = 1U << 20;
    const auto projection = Require(JsonValue::Parse(bytes, limits));
    Check(projection.is_object() && projection.object().size() == 32 &&
        Require(canonical_ascii_json(projection, 1U << 20)) == bytes &&
        Require(sha256(std::as_bytes(std::span(bytes)))) == observed.verification_projection_sha256,
        "receipt verification projection is not canonical or its digest differs");
    Check(Text(projection, "schema") == "pih.deepseek_v4_flash_0731_runtime_artifact_verification.v1" &&
        Text(projection, "artifact_abi") == "deepseek_v4_runtime_artifact_v1" &&
        Text(projection, "converter_abi") == "deepseek_v4_streaming_identity_converter_v1" &&
        Text(projection, "model_family") == "deepseek_v4_flash_0731" &&
        Text(projection, "support_state") == kSupport &&
        Text(projection, "verification_scope") == "converted_bytes_and_source_payload_non_authorizing",
        "receipt projection schema or scope differs");
    Check(Digest(projection, "artifact_root") == observed.artifact_root &&
        Number(projection, "copied_tensor_count", 67612) == observed.tensor_count &&
        Number(projection, "copied_tensor_bytes", 156015698140ULL) == observed.tensor_bytes &&
        Number(projection, "world_size", 1) == 1 &&
        Field(projection, "dspark_enabled").is_boolean() && !Field(projection, "dspark_enabled").boolean() &&
        Number(projection, "maximum_copy_chunk_bytes", 1U << 20) == (1U << 20) &&
        Number(projection, "source_descriptor_count", 50) == 50 &&
        Number(projection, "maximum_output_descriptors", 1) == 1, "receipt projection PP1 geometry differs");
    for (const auto* name : {"manifest_body_sha256", "index_root", "index_object_root",
        "runtime_records_root", "runtime_record_set_root", "runtime_records_object_root", "runtime_records_body_sha256"})
      (void)Digest(projection, name);
    const auto& shards = Field(projection, "shards");
    Check(shards.is_array() && shards.array().size() == 44, "receipt projection shard set differs");
    JsonValue::Array objects;
    std::uint64_t total = 0, tensor_count = 0, tensor_bytes = 0;
    const auto append = [&](std::string name, std::uint64_t size, const Sha256Digest& digest) {
      Check(size <= (256ULL << 30) - total, "receipt aggregate object bytes exceed budget");
      total += size;
      objects.emplace_back(JsonValue::Object{{"name", String(name)}, {"bytes", Integer(size)},
          {"sha256", String(digest.hex())}});
    };
    for (std::size_t i = 0; i < shards.array().size(); ++i) {
      const auto& shard = shards.array()[i];
      const auto space = i == 0 ? std::string("endpoint") : "layers." + std::to_string(i - 1);
      auto suffix = space;
      std::replace(suffix.begin(), suffix.end(), '.', '-');
      const auto name = "model-" + suffix + ".safetensors";
      Check(shard.is_object() && shard.object().size() == 8 && Text(shard, "namespace") == space &&
          Text(shard, "name") == name, "receipt shard order/name invalid");
      const auto size = Number(shard, "file_bytes", 512ULL << 30);
      const auto payload = Number(shard, "payload_bytes", observed.tensor_bytes);
      Check(payload < size && size - payload <= (16ULL << 20) + 8, "receipt shard header size invalid");
      tensor_count += Number(shard, "tensor_count", 4096);
      tensor_bytes += payload;
      (void)Digest(shard, "shard_layout_root");
      (void)Digest(shard, "converted_shard_root");
      append(name, size, Digest(shard, "object_sha256"));
    }
    Check(tensor_count == observed.tensor_count && tensor_bytes == observed.tensor_bytes,
        "receipt shard aggregate tensor ledger differs");
    append("model.safetensors.index.json", Number(projection, "index_bytes", 64U << 20), Digest(projection, "index_sha256"));
    append("pih.runtime-records.json", Number(projection, "runtime_records_bytes", 128U << 20), Digest(projection, "runtime_records_sha256"));
    append("pih.manifest.json", Number(projection, "manifest_bytes", 16U << 20), Digest(projection, "manifest_sha256"));
    JsonValue::Object body{{"schema", String("pih.deepseek_v4_flash_0731_generation_receipt.v1")},
        {"publication_abi", String(kAbi)}, {"support_state", String(kSupport)},
        {"artifact_root", String(observed.artifact_root.hex())},
        {"generation_name", String("sha256-" + observed.artifact_root.hex())},
        {"verification_projection_sha256", String(observed.verification_projection_sha256.hex())},
        {"dspark_enabled", JsonValue(false)}, {"world_size", Integer(1)},
        {"object_count", Integer(objects.size())}, {"object_bytes", Integer(total)},
        {"objects", JsonValue(std::move(objects))}};
    for (const auto* name : {"conversion_root", "layout_root", "disposition_root", "converter_identity_root"})
      body.emplace_back(name, String(Digest(projection, name).hex()));
    const auto body_json = Require(canonical_ascii_json(JsonValue(body), kMaximumBytes));
    const auto body_digest = Require(sha256(std::as_bytes(std::span(body_json))));
    auto hash = Require(CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-generation-receipt:v1", 11));
    Require(hash.add_bytes(1, std::as_bytes(std::span(kAbi))));
    Require(hash.add_hash(2, observed.artifact_root));
    Require(hash.add_hash(3, observed.verification_projection_sha256));
    Require(hash.add_hash(4, Digest(projection, "conversion_root")));
    Require(hash.add_hash(5, Digest(projection, "layout_root")));
    Require(hash.add_hash(6, Digest(projection, "disposition_root")));
    Require(hash.add_hash(7, Digest(projection, "converter_identity_root")));
    Require(hash.add_u32(8, 47));
    Require(hash.add_u64(9, total));
    Require(hash.add_hash(10, body_digest));
    Require(hash.add_bytes(11, std::as_bytes(std::span(kSupport))));
    const auto root = Require(hash.finalize());
    body.emplace_back("receipt_body_sha256", String(body_digest.hex()));
    body.emplace_back("generation_receipt_root", String(root.hex()));
    auto json = Require(canonical_ascii_json(JsonValue(std::move(body)), kMaximumBytes));
    const auto object_digest = Require(sha256(std::as_bytes(std::span(json))));
    (void)Require(ParseGenerationReceipt(json, observed.artifact_root, root));
    return EncodedGenerationReceipt{std::move(json), root, body_digest, object_digest, total};
  } catch (const std::exception& error) { return Status::InvalidArgument(error.what()); }
}
Status ValidateGenerationReceipt(std::string_view json, const GenerationObservation& observation) {
  if (json.empty() || json.size() > kMaximumBytes) return Status::InvalidArgument("receipt size invalid");
  auto expected = EncodeGenerationReceipt(observation);
  if (!expected.ok()) return expected.status();
  if (json != expected->json) return Status::FailedPrecondition("receipt differs from verified source-bound observation");
  return Status::Ok();
}
Result<GenerationReceipt> ParseGenerationReceipt(std::string_view json,
    const Sha256Digest& expected_artifact_root, const Sha256Digest& expected_receipt_root) {
  try {
    Check(expected_artifact_root != Sha256Digest{} && expected_receipt_root != Sha256Digest{},
        "receipt parser requires independently expected nonzero roots");
    JsonLimits limits;
    limits.max_input_bytes = kMaximumBytes;
    limits.max_nodes = 1024;
    limits.max_depth = 6;
    limits.max_string_bytes = 4096;
    const auto document = Require(JsonValue::Parse(json, limits));
    Check(document.is_object() && document.object().size() == 17 &&
        Text(document, "schema") == "pih.deepseek_v4_flash_0731_generation_receipt.v1" &&
        Text(document, "publication_abi") == kAbi && Text(document, "support_state") == kSupport &&
        Require(canonical_ascii_json(document, kMaximumBytes)) == json,
        "stored receipt field set/schema/canonical bytes invalid");
    GenerationReceipt result;
    result.artifact_root = Digest(document, "artifact_root");
    result.receipt_root = Digest(document, "generation_receipt_root");
    result.verification_projection_sha256 = Digest(document, "verification_projection_sha256");
    result.conversion_root = Digest(document, "conversion_root");
    result.layout_root = Digest(document, "layout_root");
    result.disposition_root = Digest(document, "disposition_root");
    result.converter_identity_root = Digest(document, "converter_identity_root");
    Check(result.artifact_root == expected_artifact_root && result.receipt_root == expected_receipt_root &&
        Text(document, "generation_name") == "sha256-" + expected_artifact_root.hex() &&
        Field(document, "dspark_enabled").is_boolean() && !Field(document, "dspark_enabled").boolean() &&
        Number(document, "world_size", 1) == 1 && Number(document, "object_count", 47) == 47,
        "stored receipt expected roots or PP1 geometry differ");
    const auto& objects = Field(document, "objects");
    Check(objects.is_array() && objects.array().size() == 47, "stored receipt object table invalid");
    for (std::size_t i = 0; i < objects.array().size(); ++i) {
      std::string name;
      std::uint64_t maximum;
      if (i == 0) { name = "model-endpoint.safetensors"; maximum = 512ULL << 30; }
      else if (i < 44) { name = "model-layers-" + std::to_string(i - 1) + ".safetensors"; maximum = 512ULL << 30; }
      else if (i == 44) { name = "model.safetensors.index.json"; maximum = 64U << 20; }
      else if (i == 45) { name = "pih.runtime-records.json"; maximum = 128U << 20; }
      else { name = "pih.manifest.json"; maximum = 16U << 20; }
      const auto& row = objects.array()[i];
      Check(row.is_object() && row.object().size() == 3 && Text(row, "name") == name,
          "stored receipt member set/order differs from PP1");
      const auto bytes = Number(row, "bytes", maximum);
      Check(bytes <= (256ULL << 30) - result.object_bytes, "stored receipt aggregate bytes exceed limit");
      result.object_bytes += bytes;
      result.objects.push_back({std::move(name), bytes, Digest(row, "sha256")});
    }
    Check(Number(document, "object_bytes", 256ULL << 30) == result.object_bytes,
        "stored receipt aggregate object bytes differ");
    constexpr std::array<std::string_view, 2> omitted{"receipt_body_sha256", "generation_receipt_root"};
    const auto body_json = Require(canonical_ascii_json(document, kMaximumBytes, omitted));
    const auto body_hash = Require(sha256(std::as_bytes(std::span(body_json))));
    Check(body_hash == Digest(document, "receipt_body_sha256"), "stored receipt body digest differs");
    auto hash = Require(CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-generation-receipt:v1", 11));
    Require(hash.add_bytes(1, std::as_bytes(std::span(kAbi))));
    Require(hash.add_hash(2, result.artifact_root));
    Require(hash.add_hash(3, result.verification_projection_sha256));
    Require(hash.add_hash(4, Digest(document, "conversion_root")));
    Require(hash.add_hash(5, Digest(document, "layout_root")));
    Require(hash.add_hash(6, Digest(document, "disposition_root")));
    Require(hash.add_hash(7, Digest(document, "converter_identity_root")));
    Require(hash.add_u32(8, 47));
    Require(hash.add_u64(9, result.object_bytes));
    Require(hash.add_hash(10, body_hash));
    Require(hash.add_bytes(11, std::as_bytes(std::span(kSupport))));
    Check(Require(hash.finalize()) == result.receipt_root, "stored receipt typed root differs");
    return result;
  } catch (const std::exception& error) { return Status::InvalidArgument(error.what()); }
}
}  // namespace pih::offline_deepseek
