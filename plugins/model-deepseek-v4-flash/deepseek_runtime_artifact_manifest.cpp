#include "pih/model/deepseek_runtime_artifact_manifest.h"

#include <algorithm>
#include <array>
#include <limits>
#include <set>

#include "pih/core/bounded_json.h"
#include "pih/core/canonical_hash.h"
#include "pih/core/canonical_json.h"
#include "pih/core/checked_math.h"
#include "pih/model/safetensors_shard_index.h"

namespace pih {
namespace {

constexpr std::string_view kSchema =
    "pih.deepseek_v4_flash_0731_runtime_artifact.v1";
constexpr std::string_view kArtifactAbi = "deepseek_v4_runtime_artifact_v1";
constexpr std::string_view kConverterAbi =
    "deepseek_v4_streaming_identity_converter_v1";
constexpr std::string_view kRecordsAbi = "deepseek_v4_runtime_records_v1";
constexpr std::string_view kModelFamily = "deepseek_v4_flash_0731";
constexpr std::string_view kSupportState = "hardware_evidence_open";
constexpr std::array<std::string_view, 23> kFields{
    "schema",                     "artifact_abi",
    "converter_abi",              "model_family",
    "support_state",              "layout_root",
    "logical_layout_root",        "disposition_root",
    "source_inventory_root",      "source_payload_closure_root",
    "converter_identity_root",    "conversion_root",
    "dspark_enabled",             "world_size",
    "tensor_count",               "tensor_bytes",
    "shard_count",                "shards",
    "index",                      "runtime_records",
    "resources",                  "artifact_root",
    "manifest_body_sha256"};
constexpr std::array<std::string_view, 8> kShardFields{
    "namespace",       "name",              "file_bytes",
    "tensor_count",    "payload_bytes",     "object_sha256",
    "shard_layout_root", "converted_shard_root"};
constexpr std::array<std::string_view, 5> kIndexFields{
    "name", "file_bytes", "object_sha256", "index_root",
    "index_object_root"};
constexpr std::array<std::string_view, 10> kRecordsFields{
    "name",          "abi",          "file_bytes",
    "object_sha256", "body_sha256",  "record_count",
    "record_bytes",  "record_set_root", "runtime_records_root",
    "runtime_records_object_root"};
constexpr std::array<std::string_view, 3> kResourceFields{
    "maximum_copy_chunk_bytes", "source_descriptor_count",
    "maximum_output_descriptors"};

template <std::size_t N>
bool exact_fields(const JsonValue& value,
                  const std::array<std::string_view, N>& fields) {
  if (!value.is_object() || value.object().size() != fields.size()) return false;
  for (const auto& [name, member] : value.object()) {
    (void)member;
    if (std::ranges::find(fields, name) == fields.end()) return false;
  }
  return true;
}

Result<std::string> text(const JsonValue& object, std::string_view field,
                         std::size_t maximum_bytes = 255) {
  const auto* value = object.at(field);
  if (value == nullptr || !value->is_string() || value->string().empty() ||
      value->string().size() > maximum_bytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact text field is invalid");
  }
  for (const unsigned char byte : value->string()) {
    if (byte < 0x20 || byte > 0x7e) {
      return Status::InvalidArgument(
          "DeepSeek runtime artifact text must be printable ASCII");
    }
  }
  return value->string();
}

Result<std::uint64_t> u64(const JsonValue& object, std::string_view field,
                          bool positive = true) {
  const auto* value = object.at(field);
  if (value == nullptr || !value->is_integer() || value->integer() < 0 ||
      (positive && value->integer() == 0)) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact integer field is invalid");
  }
  return static_cast<std::uint64_t>(value->integer());
}

Result<std::uint32_t> u32(const JsonValue& object, std::string_view field,
                          bool positive = true) {
  auto value = u64(object, field, positive);
  if (!value.ok()) return value.status();
  if (*value > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact integer exceeds uint32");
  }
  return static_cast<std::uint32_t>(*value);
}

Result<Sha256Digest> digest(const JsonValue& object,
                            std::string_view field) {
  auto source = text(object, field, 64);
  if (!source.ok()) return source.status();
  auto parsed = Sha256Digest::ParseHex(*source);
  if (!parsed.ok()) return parsed.status();
  bool zero = true;
  for (const auto byte : parsed->bytes) zero &= byte == std::byte{0};
  if (zero) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact digest is zero");
  }
  return *parsed;
}

Status add_text(CanonicalHashBuilder& builder, std::uint16_t field,
                std::string_view value) {
  return builder.add_bytes(field, std::as_bytes(std::span(value)));
}

Result<Sha256Digest> converted_shard_root(
    const DeepSeekRuntimeArtifactShard& shard) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-converted-shard:v1", 5);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_hash(1, shard.shard_layout_root);
  if (status.ok()) status = builder->add_hash(2, shard.object_sha256);
  if (status.ok()) status = builder->add_u64(3, shard.file_bytes);
  if (status.ok()) status = builder->add_u32(4, shard.tensor_count);
  if (status.ok()) status = builder->add_u64(5, shard.payload_bytes);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> index_object_root(
    const DeepSeekRuntimeArtifactIndex& index) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-converted-index:v1", 3);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_hash(1, index.index_root);
  if (status.ok()) status = builder->add_hash(2, index.object_sha256);
  if (status.ok()) status = builder->add_u64(3, index.file_bytes);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> runtime_records_object_root(
    const DeepSeekRuntimeRecordsAuthority& records) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-runtime-records-object:v1", 3);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_hash(1, records.runtime_records_root);
  if (status.ok()) status = builder->add_hash(2, records.object_sha256);
  if (status.ok()) status = builder->add_u64(3, records.object_bytes);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> calculate_conversion_root(
    const Sha256Digest& layout_root,
    const Sha256Digest& converter_identity_root,
    std::span<const DeepSeekRuntimeArtifactShard> shards,
    const Sha256Digest& index_root,
    const Sha256Digest& records_root,
    std::uint32_t tensor_count, std::uint64_t tensor_bytes) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-identity-conversion:v1",
      static_cast<std::uint32_t>(11 + shards.size()));
  if (!builder.ok()) return builder.status();
  auto status = add_text(*builder, 1, kConverterAbi);
  if (status.ok()) status = builder->add_hash(2, layout_root);
  if (status.ok()) status = builder->add_hash(3, converter_identity_root);
  if (status.ok()) status = builder->add_u32(4, static_cast<std::uint32_t>(shards.size()));
  if (status.ok()) status = builder->add_hash(5, index_root);
  if (status.ok()) status = builder->add_u32(6, tensor_count);
  if (status.ok()) status = builder->add_u64(7, tensor_bytes);
  if (status.ok()) status = builder->add_u32(8, 1U << 20);
  if (status.ok()) status = builder->add_u32(9, 50);
  if (status.ok()) status = builder->add_u32(10, 1);
  if (status.ok()) status = builder->add_hash(11, records_root);
  for (std::size_t index = 0; status.ok() && index < shards.size(); ++index) {
    status = builder->add_hash(static_cast<std::uint16_t>(100 + index),
                               shards[index].converted_shard_root);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> calculate_artifact_root(
    const Sha256Digest& conversion, const Sha256Digest& layout,
    const Sha256Digest& body) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-runtime-artifact:v1", 5);
  if (!builder.ok()) return builder.status();
  auto status = add_text(*builder, 1, kArtifactAbi);
  if (status.ok()) status = builder->add_hash(2, conversion);
  if (status.ok()) status = builder->add_hash(3, layout);
  if (status.ok()) status = builder->add_hash(4, body);
  if (status.ok()) status = add_text(*builder, 5, kSupportState);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<DeepSeekRuntimeArtifactShard> parse_shard(const JsonValue& row) {
  if (!exact_fields(row, kShardFields)) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact shard field set is invalid");
  }
  auto name_space = text(row, "namespace", 32);
  auto name = text(row, "name", 255);
  auto file_bytes = u64(row, "file_bytes");
  auto count = u32(row, "tensor_count");
  auto payload = u64(row, "payload_bytes");
  auto object = digest(row, "object_sha256");
  auto layout = digest(row, "shard_layout_root");
  auto converted = digest(row, "converted_shard_root");
  if (!name_space.ok()) return name_space.status();
  if (!name.ok()) return name.status();
  if (!file_bytes.ok()) return file_bytes.status();
  if (!count.ok()) return count.status();
  if (!payload.ok()) return payload.status();
  if (!object.ok()) return object.status();
  if (!layout.ok()) return layout.status();
  if (!converted.ok()) return converted.status();
  std::string canonical_name = "model-";
  for (const char byte : *name_space) {
    canonical_name.push_back(byte == '.' ? '-' : byte);
  }
  canonical_name += ".safetensors";
  if (*name != canonical_name || *file_bytes <= *payload) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact shard geometry is invalid");
  }
  DeepSeekRuntimeArtifactShard result{
      std::move(*name_space), std::move(*name), *file_bytes, *count,
      *payload, *object, *layout, *converted};
  auto expected = converted_shard_root(result);
  if (!expected.ok()) return expected.status();
  if (*expected != result.converted_shard_root) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact converted shard root differs");
  }
  return result;
}

}  // namespace

Result<DeepSeekRuntimeArtifactManifest>
DeepSeekRuntimeArtifactManifest::Parse(
    std::string_view json, const Sha256Digest& expected_artifact_root) {
  if (json.empty() || json.size() > kMaximumBytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact manifest byte geometry is invalid");
  }
  JsonLimits limits;
  limits.max_input_bytes = kMaximumBytes;
  limits.max_depth = 8;
  limits.max_nodes = 4096;
  limits.max_string_bytes = 1024;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!exact_fields(*root, kFields)) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact manifest field set is invalid");
  }
  auto canonical = canonical_ascii_json(*root, kMaximumBytes);
  if (!canonical.ok()) return canonical.status();
  if (*canonical != json) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact manifest is not canonical JSON");
  }
  auto schema = text(*root, "schema", 96);
  auto artifact_abi = text(*root, "artifact_abi", 64);
  auto converter_abi = text(*root, "converter_abi", 64);
  auto family = text(*root, "model_family", 64);
  auto support = text(*root, "support_state", 64);
  auto layout = digest(*root, "layout_root");
  auto disposition = digest(*root, "disposition_root");
  auto converter = digest(*root, "converter_identity_root");
  auto conversion = digest(*root, "conversion_root");
  auto artifact = digest(*root, "artifact_root");
  auto body_digest = digest(*root, "manifest_body_sha256");
  auto world = u32(*root, "world_size");
  auto count = u32(*root, "tensor_count");
  auto bytes = u64(*root, "tensor_bytes");
  auto shard_count = u32(*root, "shard_count");
  const auto* dspark = root->at("dspark_enabled");
  const auto* shard_rows = root->at("shards");
  const auto* index_row = root->at("index");
  const auto* records_row = root->at("runtime_records");
  const auto* resources = root->at("resources");
  if (!schema.ok()) return schema.status();
  if (!artifact_abi.ok()) return artifact_abi.status();
  if (!converter_abi.ok()) return converter_abi.status();
  if (!family.ok()) return family.status();
  if (!support.ok()) return support.status();
  if (!layout.ok()) return layout.status();
  if (!disposition.ok()) return disposition.status();
  if (!converter.ok()) return converter.status();
  if (!conversion.ok()) return conversion.status();
  if (!artifact.ok()) return artifact.status();
  if (!body_digest.ok()) return body_digest.status();
  if (!world.ok()) return world.status();
  if (!count.ok()) return count.status();
  if (!bytes.ok()) return bytes.status();
  if (!shard_count.ok()) return shard_count.status();
  if (*schema != kSchema || *artifact_abi != kArtifactAbi ||
      *converter_abi != kConverterAbi || *family != kModelFamily ||
      *support != kSupportState || *world > 4 || dspark == nullptr ||
      !dspark->is_boolean() || shard_rows == nullptr ||
      !shard_rows->is_array() || shard_rows->array().size() != *shard_count ||
      *shard_count == 0 || *shard_count > 47 ||
      index_row == nullptr || !exact_fields(*index_row, kIndexFields) ||
      records_row == nullptr || !exact_fields(*records_row, kRecordsFields) ||
      resources == nullptr || !exact_fields(*resources, kResourceFields) ||
      *artifact != expected_artifact_root) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact identity or profile is invalid");
  }
  for (const auto* ignored_root : {"logical_layout_root",
                                   "source_inventory_root",
                                   "source_payload_closure_root"}) {
    auto ignored = digest(*root, ignored_root);
    if (!ignored.ok()) return ignored.status();
  }
  auto maximum_chunk = u32(*resources, "maximum_copy_chunk_bytes");
  auto source_descriptors = u32(*resources, "source_descriptor_count");
  auto output_descriptors = u32(*resources, "maximum_output_descriptors");
  if (!maximum_chunk.ok()) return maximum_chunk.status();
  if (!source_descriptors.ok()) return source_descriptors.status();
  if (!output_descriptors.ok()) return output_descriptors.status();
  if (*maximum_chunk != 1U << 20 || *source_descriptors != 50 ||
      *output_descriptors != 1) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact resource high-water differs");
  }

  DeepSeekRuntimeArtifactManifest result;
  result.artifact_root_ = *artifact;
  result.conversion_root_ = *conversion;
  result.layout_root_ = *layout;
  result.disposition_root_ = *disposition;
  result.dspark_enabled_ = dspark->boolean();
  result.world_size_ = *world;
  result.tensor_count_ = *count;
  result.tensor_bytes_ = *bytes;
  result.shards_.reserve(shard_rows->array().size());
  std::set<std::string, std::less<>> namespaces;
  std::set<std::string, std::less<>> names;
  std::uint64_t payload_total = 0;
  std::uint64_t count_total = 0;
  for (const auto& row : shard_rows->array()) {
    auto shard = parse_shard(row);
    if (!shard.ok()) return shard.status();
    if (!namespaces.emplace(shard->name_space).second ||
        !names.emplace(shard->shard_name).second) {
      return Status::InvalidArgument(
          "DeepSeek runtime artifact shard is duplicated");
    }
    auto next_payload = checked_add_u64(payload_total, shard->payload_bytes);
    auto next_count = checked_add_u64(count_total, shard->tensor_count);
    if (!next_payload.ok()) return next_payload.status();
    if (!next_count.ok()) return next_count.status();
    payload_total = *next_payload;
    count_total = *next_count;
    result.shards_.push_back(std::move(*shard));
  }
  if (payload_total != result.tensor_bytes_ ||
      count_total != result.tensor_count_) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact shard ledger differs");
  }

  auto index_name = text(*index_row, "name", 64);
  auto index_bytes = u64(*index_row, "file_bytes");
  auto index_sha = digest(*index_row, "object_sha256");
  auto index_root_digest = digest(*index_row, "index_root");
  auto index_object = digest(*index_row, "index_object_root");
  if (!index_name.ok()) return index_name.status();
  if (!index_bytes.ok()) return index_bytes.status();
  if (!index_sha.ok()) return index_sha.status();
  if (!index_root_digest.ok()) return index_root_digest.status();
  if (!index_object.ok()) return index_object.status();
  if (*index_name != "model.safetensors.index.json" ||
      *index_bytes > SafetensorsShardIndex::kMaximumIndexBytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact index geometry is invalid");
  }
  result.index_ = {std::move(*index_name), *index_bytes, *index_sha,
                   *index_root_digest, *index_object};
  auto expected_index = index_object_root(result.index_);
  if (!expected_index.ok()) return expected_index.status();
  if (*expected_index != result.index_.index_object_root) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact index object root differs");
  }

  auto records_name = text(*records_row, "name", 64);
  auto records_abi = text(*records_row, "abi", 64);
  auto records_bytes = u64(*records_row, "file_bytes");
  auto records_sha = digest(*records_row, "object_sha256");
  auto records_body = digest(*records_row, "body_sha256");
  auto records_count = u32(*records_row, "record_count");
  auto record_bytes = u64(*records_row, "record_bytes");
  auto record_set = digest(*records_row, "record_set_root");
  auto records_root = digest(*records_row, "runtime_records_root");
  auto records_object = digest(*records_row, "runtime_records_object_root");
  if (!records_name.ok()) return records_name.status();
  if (!records_abi.ok()) return records_abi.status();
  if (!records_bytes.ok()) return records_bytes.status();
  if (!records_sha.ok()) return records_sha.status();
  if (!records_body.ok()) return records_body.status();
  if (!records_count.ok()) return records_count.status();
  if (!record_bytes.ok()) return record_bytes.status();
  if (!record_set.ok()) return record_set.status();
  if (!records_root.ok()) return records_root.status();
  if (!records_object.ok()) return records_object.status();
  if (*records_name != "pih.runtime-records.json" ||
      *records_abi != kRecordsAbi ||
      *records_bytes > DeepSeekRuntimeRecordsManifest::kMaximumBytes ||
      *records_count != result.tensor_count_ ||
      *record_bytes != result.tensor_bytes_) {
    return Status::InvalidArgument(
        "DeepSeek runtime records authority geometry differs");
  }
  result.runtime_records_ = {
      result.layout_root_, result.disposition_root_, *record_set,
      *records_root, *records_body, *records_sha, *records_bytes,
      *record_bytes, *records_count, result.world_size_,
      result.dspark_enabled_};
  auto expected_records_object =
      runtime_records_object_root(result.runtime_records_);
  if (!expected_records_object.ok()) return expected_records_object.status();
  if (*expected_records_object != *records_object) {
    return Status::InvalidArgument(
        "DeepSeek runtime records object root differs");
  }
  auto expected_conversion = calculate_conversion_root(
      result.layout_root_, *converter, result.shards_,
      result.index_.index_object_root, *records_object,
      result.tensor_count_, result.tensor_bytes_);
  if (!expected_conversion.ok()) return expected_conversion.status();
  if (*expected_conversion != result.conversion_root_) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact conversion root differs");
  }
  const std::array<std::string_view, 2> omitted{
      "artifact_root", "manifest_body_sha256"};
  auto body = canonical_ascii_json(*root, kMaximumBytes, omitted);
  if (!body.ok()) return body.status();
  auto observed_body = sha256(std::as_bytes(std::span(*body)));
  if (!observed_body.ok()) return observed_body.status();
  if (*observed_body != *body_digest) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact manifest body digest differs");
  }
  auto expected_root = calculate_artifact_root(
      result.conversion_root_, result.layout_root_, *body_digest);
  if (!expected_root.ok()) return expected_root.status();
  if (*expected_root != result.artifact_root_) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact root differs");
  }
  return result;
}

Status DeepSeekRuntimeArtifactManifest::validate_flash_0731_geometry() const {
  const auto expected_count = dspark_enabled_ ? 72'317U : 67'612U;
  const auto expected_bytes =
      dspark_enabled_ ? 166'878'536'440ULL : 156'015'698'140ULL;
  const auto expected_shards = dspark_enabled_ ? 47U : 44U;
  if (tensor_count_ != expected_count || tensor_bytes_ != expected_bytes ||
      shards_.size() != expected_shards) {
    return Status::FailedPrecondition(
        "DeepSeek runtime artifact family geometry differs");
  }
  std::vector<std::string> expected_namespaces{"endpoint"};
  for (std::uint32_t layer = 0; layer < 43; ++layer) {
    expected_namespaces.push_back("layers." + std::to_string(layer));
  }
  if (dspark_enabled_) {
    for (std::uint32_t stage = 0; stage < 3; ++stage) {
      expected_namespaces.push_back("mtp." + std::to_string(stage));
    }
  }
  for (std::size_t index = 0; index < shards_.size(); ++index) {
    if (shards_[index].name_space != expected_namespaces[index]) {
      return Status::FailedPrecondition(
          "DeepSeek runtime artifact shard order differs");
    }
  }
  return Status::Ok();
}

Result<DeepSeekRuntimeArtifactManifest::Encoded>
DeepSeekRuntimeArtifactManifest::Encode(const EncodeInput& input) {
  const auto& records = input.records;
  if (input.maximum_copy_chunk_bytes != 1U << 20 ||
      input.source_descriptor_count != 50 || input.maximum_output_descriptors != 1 ||
      input.shards.size() != (records.dspark_enabled ? 47U : 44U) ||
      records.world_size == 0 || records.world_size > 4 ||
      records.object_bytes > DeepSeekRuntimeRecordsManifest::kMaximumBytes ||
      input.index.file_bytes > SafetensorsShardIndex::kMaximumIndexBytes)
    return Status::InvalidArgument("artifact encoding profile or metadata bounds invalid");
  constexpr auto maximum = static_cast<std::uint64_t>(INT64_MAX);
  if (records.record_bytes > maximum)
    return Status::InvalidArgument("artifact record bytes exceed JSON integer range");
  auto string = [](std::string_view value) { return JsonValue(std::string(value)); };
  auto integer = [](std::uint64_t value) { return JsonValue(static_cast<std::int64_t>(value)); };
  auto hex = [&](const Sha256Digest& value) { return string(value.hex()); };
  auto shards = input.shards;
  JsonValue::Array shard_rows;
  for (auto& shard : shards) {
    if (shard.file_bytes > maximum || shard.payload_bytes > maximum ||
        shard.name_space.size() > 32 || shard.shard_name.size() > 255)
      return Status::InvalidArgument("artifact shard exceeds encoding bounds");
    auto converted = converted_shard_root(shard);
    if (!converted.ok()) return converted.status();
    shard.converted_shard_root = *converted;
    shard_rows.emplace_back(JsonValue::Object{
        {"namespace", string(shard.name_space)}, {"name", string(shard.shard_name)},
        {"file_bytes", integer(shard.file_bytes)}, {"tensor_count", integer(shard.tensor_count)},
        {"payload_bytes", integer(shard.payload_bytes)}, {"object_sha256", hex(shard.object_sha256)},
        {"shard_layout_root", hex(shard.shard_layout_root)},
        {"converted_shard_root", hex(shard.converted_shard_root)}});
  }
  auto index = input.index;
  if (index.name != "model.safetensors.index.json")
    return Status::InvalidArgument("artifact index member name invalid");
  auto index_object = index_object_root(index);
  if (!index_object.ok()) return index_object.status();
  index.index_object_root = *index_object;
  auto records_object = runtime_records_object_root(records);
  if (!records_object.ok()) return records_object.status();
  auto conversion = calculate_conversion_root(records.layout_root,
      input.converter_identity_root, shards, *index_object, *records_object,
      records.record_count, records.record_bytes);
  if (!conversion.ok()) return conversion.status();
  JsonValue::Object fields{
      {"schema", string(kSchema)}, {"artifact_abi", string(kArtifactAbi)},
      {"converter_abi", string(kConverterAbi)}, {"model_family", string(kModelFamily)},
      {"support_state", string(kSupportState)}, {"layout_root", hex(records.layout_root)},
      {"logical_layout_root", hex(input.logical_layout_root)},
      {"disposition_root", hex(records.disposition_root)},
      {"source_inventory_root", hex(input.source_inventory_root)},
      {"source_payload_closure_root", hex(input.source_payload_closure_root)},
      {"converter_identity_root", hex(input.converter_identity_root)},
      {"conversion_root", hex(*conversion)}, {"dspark_enabled", JsonValue(records.dspark_enabled)},
      {"world_size", integer(records.world_size)}, {"tensor_count", integer(records.record_count)},
      {"tensor_bytes", integer(records.record_bytes)}, {"shard_count", integer(shards.size())},
      {"shards", JsonValue(std::move(shard_rows))},
      {"index", JsonValue(JsonValue::Object{
          {"name", string(index.name)}, {"file_bytes", integer(index.file_bytes)},
          {"object_sha256", hex(index.object_sha256)}, {"index_root", hex(index.index_root)},
          {"index_object_root", hex(index.index_object_root)}})},
      {"runtime_records", JsonValue(JsonValue::Object{
          {"name", string("pih.runtime-records.json")}, {"abi", string(kRecordsAbi)},
          {"file_bytes", integer(records.object_bytes)}, {"object_sha256", hex(records.object_sha256)},
          {"body_sha256", hex(records.body_sha256)}, {"record_count", integer(records.record_count)},
          {"record_bytes", integer(records.record_bytes)}, {"record_set_root", hex(records.record_set_root)},
          {"runtime_records_root", hex(records.runtime_records_root)},
          {"runtime_records_object_root", hex(*records_object)}})},
      {"resources", JsonValue(JsonValue::Object{
          {"maximum_copy_chunk_bytes", integer(input.maximum_copy_chunk_bytes)},
          {"source_descriptor_count", integer(input.source_descriptor_count)},
          {"maximum_output_descriptors", integer(input.maximum_output_descriptors)}})}};
  auto body = canonical_ascii_json(JsonValue(fields), kMaximumBytes);
  if (!body.ok()) return body.status();
  auto body_hash = sha256(std::as_bytes(std::span(*body)));
  if (!body_hash.ok()) return body_hash.status();
  auto artifact = calculate_artifact_root(*conversion, records.layout_root, *body_hash);
  if (!artifact.ok()) return artifact.status();
  fields.emplace_back("artifact_root", hex(*artifact));
  fields.emplace_back("manifest_body_sha256", hex(*body_hash));
  auto json = canonical_ascii_json(JsonValue(std::move(fields)), kMaximumBytes);
  if (!json.ok()) return json.status();
  auto parsed = Parse(*json, *artifact);
  if (!parsed.ok()) return parsed.status();
  auto geometry = parsed->validate_flash_0731_geometry();
  if (!geometry.ok()) return geometry;
  auto object_hash = sha256(std::as_bytes(std::span(*json)));
  if (!object_hash.ok()) return object_hash.status();
  return Encoded{std::move(*json), *artifact, *object_hash};
}

}  // namespace pih
