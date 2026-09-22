#include "pih/model/deepseek_runtime_records_manifest.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>

#include "pih/core/bounded_json.h"
#include "pih/core/canonical_hash.h"
#include "pih/core/canonical_json.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::string_view kSchema =
    "pih.deepseek_v4_flash_0731_runtime_records.v1";
constexpr std::string_view kAbi = "deepseek_v4_runtime_records_v1";
constexpr std::string_view kModelFamily = "deepseek_v4_flash_0731";
constexpr std::string_view kSupportState = "hardware_evidence_open";
constexpr std::string_view kTransform = "identity_bytes";
constexpr std::array<std::string_view, 14> kRootFields{
    "schema",          "abi",             "model_family",
    "support_state",   "layout_root",     "disposition_root",
    "dspark_enabled",  "world_size",      "record_count",
    "record_bytes",    "record_set_root", "records",
    "body_sha256",     "runtime_records_root"};
constexpr std::array<std::string_view, 17> kRecordFields{
    "name",                    "shard",
    "namespace",               "role",
    "logical_layer",           "owner_rank",
    "dtype",                   "shape",
    "file_begin",              "file_end",
    "tensor_bytes",            "storage_semantics",
    "transform",               "target_logical_root",
    "disposition_record_root", "layout_record_root",
    "runtime_record_root"};

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
                         std::size_t maximum_bytes = 1024) {
  const auto* value = object.at(field);
  if (value == nullptr || !value->is_string() || value->string().empty() ||
      value->string().size() > maximum_bytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime records text field is invalid");
  }
  for (const unsigned char byte : value->string()) {
    if (byte < 0x20 || byte > 0x7e) {
      return Status::InvalidArgument(
          "DeepSeek runtime records text must be printable ASCII");
    }
  }
  return value->string();
}

Result<std::uint64_t> u64(const JsonValue& object, std::string_view field,
                          bool positive) {
  const auto* value = object.at(field);
  if (value == nullptr || !value->is_integer() || value->integer() < 0 ||
      (positive && value->integer() == 0)) {
    return Status::InvalidArgument(
        "DeepSeek runtime records integer field is invalid");
  }
  return static_cast<std::uint64_t>(value->integer());
}

Result<std::uint32_t> u32(const JsonValue& object, std::string_view field,
                          bool positive) {
  auto value = u64(object, field, positive);
  if (!value.ok()) return value.status();
  if (*value > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "DeepSeek runtime records integer exceeds uint32");
  }
  return static_cast<std::uint32_t>(*value);
}

Result<Sha256Digest> digest(const JsonValue& object,
                            std::string_view field) {
  auto value = text(object, field, 64);
  if (!value.ok()) return value.status();
  auto parsed = Sha256Digest::ParseHex(*value);
  if (!parsed.ok()) return parsed.status();
  bool zero = true;
  for (const auto byte : parsed->bytes) zero &= byte == std::byte{0};
  if (zero) {
    return Status::InvalidArgument(
        "DeepSeek runtime records digest is zero");
  }
  return *parsed;
}

Status add_text(CanonicalHashBuilder& builder, std::uint16_t field,
                std::string_view value) {
  return builder.add_bytes(field, std::as_bytes(std::span(value)));
}

Result<Sha256Digest> runtime_record_root(const DeepSeekRuntimeRecord& record) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-runtime-record:v1", 17);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_hash(1, record.layout_record_root);
  if (status.ok()) status = builder->add_hash(2, record.target_logical_root);
  if (status.ok()) status = builder->add_hash(3, record.disposition_record_root);
  if (status.ok()) status = add_text(*builder, 4, record.tensor_name);
  if (status.ok()) status = add_text(*builder, 5, record.shard_name);
  if (status.ok()) status = add_text(*builder, 6, record.name_space);
  const auto role = [&]() -> std::string_view {
    switch (record.role) {
      case DeepSeekTensorRole::kEmbedding: return "embedding";
      case DeepSeekTensorRole::kMainLayer: return "main_layer";
      case DeepSeekTensorRole::kFinalHead: return "final_head";
      case DeepSeekTensorRole::kDspark: return "dspark_stage";
    }
    return {};
  }();
  if (status.ok()) status = add_text(*builder, 7, role);
  if (status.ok()) status = builder->add_u32(8, record.logical_layer);
  if (status.ok()) status = builder->add_u32(9, record.owner_rank);
  const auto dtype = [&]() -> std::string_view {
    switch (record.dtype) {
      case DType::kFloat32: return "F32";
      case DType::kFloat16: return "F16";
      case DType::kBFloat16: return "BF16";
      case DType::kInt8: return "I8";
      case DType::kUInt8: return "U8";
      case DType::kFloat8E4M3: return "F8_E4M3";
      case DType::kFloat8E8M0: return "F8_E8M0";
      case DType::kInt32: return "I32";
      case DType::kInt64: return "I64";
      case DType::kBool: return "BOOL";
      case DType::kFloat64:
      case DType::kUInt32: return {};
    }
    return {};
  }();
  if (status.ok()) status = add_text(*builder, 10, dtype);
  if (status.ok()) {
    status = builder->add_u32(
        11, static_cast<std::uint32_t>(record.shape.size()));
  }
  std::vector<std::byte> packed_shape(record.shape.size() * 8);
  for (std::size_t index = 0; index < record.shape.size(); ++index) {
    for (std::size_t byte = 0; byte < 8; ++byte) {
      packed_shape[index * 8 + byte] = static_cast<std::byte>(
          record.shape[index] >> (byte * 8));
    }
  }
  if (status.ok()) status = builder->add_bytes(12, packed_shape);
  if (status.ok()) status = builder->add_u64(13, record.file_begin);
  if (status.ok()) status = builder->add_u64(14, record.file_end);
  if (status.ok()) status = builder->add_u64(15, record.tensor_bytes);
  const auto semantics = [&]() -> std::string_view {
    switch (record.storage_semantics) {
      case DeepSeekStorageSemantics::kDirectF32LittleEndianBits:
        return "direct_f32_le_bits";
      case DeepSeekStorageSemantics::kDirectF16LittleEndianBits:
        return "direct_f16_le_bits";
      case DeepSeekStorageSemantics::kDirectBf16LittleEndianBits:
        return "direct_bf16_le_bits";
      case DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits:
        return "direct_mxfp4_e2m1_packed_bits";
      case DeepSeekStorageSemantics::kDirectU8Bits: return "direct_u8_bits";
      case DeepSeekStorageSemantics::kDirectFp8E4m3Bits:
        return "direct_fp8_e4m3_bits";
      case DeepSeekStorageSemantics::kDirectUe8m0ScaleBits:
        return "direct_ue8m0_scale_bits";
      case DeepSeekStorageSemantics::kDirectI32LittleEndianBits:
        return "direct_i32_le_bits";
      case DeepSeekStorageSemantics::kDirectI64LittleEndianBits:
        return "direct_i64_le_bits";
      case DeepSeekStorageSemantics::kDirectBoolBits:
        return "direct_bool_bits";
    }
    return {};
  }();
  if (status.ok()) status = add_text(*builder, 16, semantics);
  if (status.ok()) status = add_text(*builder, 17, kTransform);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> root_set(std::string_view domain,
                              std::span<const Sha256Digest> roots) {
  if (roots.empty() || roots.size() > 65'436) {
    return Status::InvalidArgument(
        "DeepSeek runtime record root set geometry is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      domain, static_cast<std::uint32_t>(roots.size() + 1));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t index = 0; status.ok() && index < roots.size(); ++index) {
    status = builder->add_hash(static_cast<std::uint16_t>(100 + index),
                               roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> record_set_root(
    std::span<const DeepSeekRuntimeRecord> records) {
  std::vector<Sha256Digest> chunks;
  for (std::size_t begin = 0; begin < records.size(); begin += 4096) {
    const auto end = std::min(records.size(), begin + 4096);
    std::vector<Sha256Digest> roots;
    roots.reserve(end - begin);
    for (std::size_t index = begin; index < end; ++index) {
      roots.push_back(records[index].runtime_record_root);
    }
    auto chunk = root_set(
        "pih:deepseek-v4-flash-0731-runtime-record-chunk:v1",
        roots);
    if (!chunk.ok()) return chunk.status();
    chunks.push_back(*chunk);
  }
  return root_set(
      "pih:deepseek-v4-flash-0731-runtime-record-set:v1", chunks);
}

Result<Sha256Digest> runtime_records_root(
    const DeepSeekRuntimeRecordsAuthority& authority) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-v4-flash-0731-runtime-records:v1", 9);
  if (!builder.ok()) return builder.status();
  auto status = add_text(*builder, 1, kAbi);
  if (status.ok()) status = builder->add_hash(2, authority.layout_root);
  if (status.ok()) status = builder->add_hash(3, authority.disposition_root);
  if (status.ok()) status = builder->add_hash(4, authority.record_set_root);
  if (status.ok()) status = builder->add_u32(5, authority.record_count);
  if (status.ok()) status = builder->add_u64(6, authority.record_bytes);
  if (status.ok()) status = builder->add_u32(7, authority.dspark_enabled ? 1 : 0);
  if (status.ok()) status = builder->add_u32(8, authority.world_size);
  if (status.ok()) status = builder->add_hash(9, authority.body_sha256);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<std::pair<DType, DeepSeekStorageSemantics>> parse_storage(
    std::string_view dtype, std::string_view semantics) {
  using Pair = std::pair<DType, DeepSeekStorageSemantics>;
  if (dtype == "F32" && semantics == "direct_f32_le_bits")
    return Pair{DType::kFloat32,
                DeepSeekStorageSemantics::kDirectF32LittleEndianBits};
  if (dtype == "F16" && semantics == "direct_f16_le_bits")
    return Pair{DType::kFloat16,
                DeepSeekStorageSemantics::kDirectF16LittleEndianBits};
  if (dtype == "BF16" && semantics == "direct_bf16_le_bits")
    return Pair{DType::kBFloat16,
                DeepSeekStorageSemantics::kDirectBf16LittleEndianBits};
  if (dtype == "I8" && semantics == "direct_mxfp4_e2m1_packed_bits")
    return Pair{DType::kInt8,
                DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits};
  if (dtype == "U8" && semantics == "direct_u8_bits")
    return Pair{DType::kUInt8, DeepSeekStorageSemantics::kDirectU8Bits};
  if (dtype == "F8_E4M3" && semantics == "direct_fp8_e4m3_bits")
    return Pair{DType::kFloat8E4M3,
                DeepSeekStorageSemantics::kDirectFp8E4m3Bits};
  if (dtype == "F8_E8M0" && semantics == "direct_ue8m0_scale_bits")
    return Pair{DType::kFloat8E8M0,
                DeepSeekStorageSemantics::kDirectUe8m0ScaleBits};
  if (dtype == "I32" && semantics == "direct_i32_le_bits")
    return Pair{DType::kInt32,
                DeepSeekStorageSemantics::kDirectI32LittleEndianBits};
  if (dtype == "I64" && semantics == "direct_i64_le_bits")
    return Pair{DType::kInt64,
                DeepSeekStorageSemantics::kDirectI64LittleEndianBits};
  if (dtype == "BOOL" && semantics == "direct_bool_bits")
    return Pair{DType::kBool, DeepSeekStorageSemantics::kDirectBoolBits};
  return Status::InvalidArgument(
      "DeepSeek runtime dtype and storage semantics disagree");
}

Result<std::pair<DeepSeekTensorRole, std::uint32_t>> parse_role(
    const JsonValue& row) {
  auto role = text(row, "role", 32);
  auto layer = u32(row, "logical_layer", false);
  if (!role.ok()) return role.status();
  if (!layer.ok()) return layer.status();
  if (*role == "embedding" && *layer == UINT32_MAX)
    return std::pair{DeepSeekTensorRole::kEmbedding, *layer};
  if (*role == "final_head" && *layer == UINT32_MAX)
    return std::pair{DeepSeekTensorRole::kFinalHead, *layer};
  if (*role == "main_layer" && *layer <= 42)
    return std::pair{DeepSeekTensorRole::kMainLayer, *layer};
  if (*role == "dspark_stage" && *layer <= 2)
    return std::pair{DeepSeekTensorRole::kDspark, *layer};
  return Status::InvalidArgument(
      "DeepSeek runtime role and logical layer disagree");
}

Status validate_owner(const DeepSeekRuntimeRecord& record,
                      const DeepSeekPipelinePlan& pipeline,
                      bool dspark_enabled) {
  if (record.owner_rank >= pipeline.world_size()) {
    return Status::InvalidArgument(
        "DeepSeek runtime record owner is outside world");
  }
  switch (record.role) {
    case DeepSeekTensorRole::kEmbedding:
      if (record.owner_rank != 0 || record.name_space != "endpoint") break;
      return Status::Ok();
    case DeepSeekTensorRole::kFinalHead:
      if (record.owner_rank != pipeline.world_size() - 1 ||
          record.name_space != "endpoint") break;
      return Status::Ok();
    case DeepSeekTensorRole::kMainLayer: {
      const auto expected_namespace =
          "layers." + std::to_string(record.logical_layer);
      const auto range = pipeline.rank(record.owner_rank).layers;
      if (record.name_space != expected_namespace ||
          record.logical_layer < range.first_layer ||
          record.logical_layer > range.last_layer) break;
      return Status::Ok();
    }
    case DeepSeekTensorRole::kDspark:
      if (!dspark_enabled ||
          !pipeline.rank(pipeline.world_size() - 1).owns_dspark ||
          record.owner_rank != pipeline.world_size() - 1 ||
          record.name_space != "mtp." + std::to_string(record.logical_layer))
        break;
      return Status::Ok();
  }
  return Status::InvalidArgument(
      "DeepSeek runtime record role, namespace and owner disagree");
}

Result<DeepSeekRuntimeRecord> parse_record(
    const JsonValue& row, const DeepSeekPipelinePlan& pipeline,
    bool dspark_enabled) {
  if (!exact_fields(row, kRecordFields)) {
    return Status::InvalidArgument(
        "DeepSeek runtime record field set is invalid");
  }
  auto name = text(row, "name",
                   DeepSeekRuntimeRecordsManifest::kMaximumTensorNameBytes);
  auto shard = text(row, "shard", 255);
  auto name_space = text(row, "namespace", 32);
  auto role = parse_role(row);
  auto owner = u32(row, "owner_rank", false);
  auto dtype = text(row, "dtype", 16);
  auto semantics = text(row, "storage_semantics", 48);
  auto transform = text(row, "transform", 32);
  auto begin = u64(row, "file_begin", true);
  auto end = u64(row, "file_end", true);
  auto bytes = u64(row, "tensor_bytes", true);
  auto logical_root = digest(row, "target_logical_root");
  auto disposition_root = digest(row, "disposition_record_root");
  auto layout_root = digest(row, "layout_record_root");
  auto record_root = digest(row, "runtime_record_root");
  if (!name.ok()) return name.status();
  if (!shard.ok()) return shard.status();
  if (!name_space.ok()) return name_space.status();
  if (!role.ok()) return role.status();
  if (!owner.ok()) return owner.status();
  if (!dtype.ok()) return dtype.status();
  if (!semantics.ok()) return semantics.status();
  if (!transform.ok()) return transform.status();
  if (!begin.ok()) return begin.status();
  if (!end.ok()) return end.status();
  if (!bytes.ok()) return bytes.status();
  if (!logical_root.ok()) return logical_root.status();
  if (!disposition_root.ok()) return disposition_root.status();
  if (!layout_root.ok()) return layout_root.status();
  if (!record_root.ok()) return record_root.status();
  if (*transform != kTransform || *end <= *begin || *end - *begin != *bytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime record direct byte geometry is invalid");
  }
  const auto* shape_value = row.at("shape");
  if (shape_value == nullptr || !shape_value->is_array() ||
      shape_value->array().empty() || shape_value->array().size() > 8) {
    return Status::InvalidArgument("DeepSeek runtime record shape is invalid");
  }
  std::vector<std::uint64_t> shape;
  shape.reserve(shape_value->array().size());
  std::uint64_t elements = 1;
  for (const auto& dimension : shape_value->array()) {
    if (!dimension.is_integer() || dimension.integer() <= 0) {
      return Status::InvalidArgument(
          "DeepSeek runtime record dimension is invalid");
    }
    const auto value = static_cast<std::uint64_t>(dimension.integer());
    auto next = checked_mul_u64(elements, value);
    if (!next.ok()) return next.status();
    elements = *next;
    shape.push_back(value);
  }
  auto storage = parse_storage(*dtype, *semantics);
  if (!storage.ok()) return storage.status();
  auto element_bytes = dtype_size(storage->first);
  if (!element_bytes.ok()) return element_bytes.status();
  auto expected_bytes = checked_mul_u64(elements, *element_bytes);
  if (!expected_bytes.ok()) return expected_bytes.status();
  if (*expected_bytes != *bytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime record dtype and shape differ from bytes");
  }
  std::string canonical_shard = "model-";
  for (const char byte : *name_space) {
    canonical_shard.push_back(byte == '.' ? '-' : byte);
  }
  canonical_shard += ".safetensors";
  if (*shard != canonical_shard) {
    return Status::InvalidArgument(
        "DeepSeek runtime record shard and namespace disagree");
  }
  DeepSeekRuntimeRecord record{
      std::move(*name), std::move(*shard), std::move(*name_space),
      role->first, role->second, *owner, storage->first, std::move(shape),
      *begin, *end, *bytes, storage->second, *logical_root,
      *disposition_root, *layout_root, *record_root};
  auto owner_status = validate_owner(record, pipeline, dspark_enabled);
  if (!owner_status.ok()) return owner_status;
  auto expected_root = runtime_record_root(record);
  if (!expected_root.ok()) return expected_root.status();
  if (*expected_root != record.runtime_record_root) {
    return Status::InvalidArgument(
        "DeepSeek runtime record root differs from fields");
  }
  return record;
}

}  // namespace

Result<DeepSeekRuntimeRecordsManifest>
DeepSeekRuntimeRecordsManifest::Parse(
    std::string_view json, const DeepSeekRuntimeRecordsAuthority& authority,
    const DeepSeekPipelinePlan& pipeline) {
  if (json.empty() || json.size() > kMaximumBytes ||
      authority.object_bytes != json.size() || authority.record_count == 0 ||
      authority.record_count > kMaximumRecordCount ||
      authority.record_bytes == 0 || authority.world_size == 0 ||
      authority.world_size > 4 || pipeline.world_size() != authority.world_size) {
    return Status::InvalidArgument(
        "DeepSeek runtime records authority geometry is invalid");
  }
  const bool pipeline_dspark =
      pipeline.rank(pipeline.world_size() - 1).owns_dspark;
  if (pipeline_dspark != authority.dspark_enabled) {
    return Status::InvalidArgument(
        "DeepSeek runtime records profile differs from pipeline");
  }
  auto object_digest = sha256(std::as_bytes(std::span(json)));
  if (!object_digest.ok()) return object_digest.status();
  if (*object_digest != authority.object_sha256) {
    return Status::InvalidArgument(
        "DeepSeek runtime records object digest differs from authority");
  }
  JsonLimits limits;
  limits.max_input_bytes = kMaximumBytes;
  limits.max_depth = 8;
  limits.max_nodes = kMaximumRecordCount * 32 + 64;
  limits.max_string_bytes = kMaximumTensorNameBytes;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!exact_fields(*root, kRootFields)) {
    return Status::InvalidArgument(
        "DeepSeek runtime records root field set is invalid");
  }
  auto canonical = canonical_ascii_json(*root, kMaximumBytes);
  if (!canonical.ok()) return canonical.status();
  if (*canonical != json) {
    return Status::InvalidArgument(
        "DeepSeek runtime records are not canonical JSON");
  }
  auto schema = text(*root, "schema", 96);
  auto abi = text(*root, "abi", 64);
  auto family = text(*root, "model_family", 64);
  auto support = text(*root, "support_state", 64);
  auto layout = digest(*root, "layout_root");
  auto disposition = digest(*root, "disposition_root");
  auto set_root = digest(*root, "record_set_root");
  auto body_digest = digest(*root, "body_sha256");
  auto records_root = digest(*root, "runtime_records_root");
  auto world = u32(*root, "world_size", true);
  auto count = u32(*root, "record_count", true);
  auto bytes = u64(*root, "record_bytes", true);
  const auto* dspark = root->at("dspark_enabled");
  const auto* rows = root->at("records");
  if (!schema.ok()) return schema.status();
  if (!abi.ok()) return abi.status();
  if (!family.ok()) return family.status();
  if (!support.ok()) return support.status();
  if (!layout.ok()) return layout.status();
  if (!disposition.ok()) return disposition.status();
  if (!set_root.ok()) return set_root.status();
  if (!body_digest.ok()) return body_digest.status();
  if (!records_root.ok()) return records_root.status();
  if (!world.ok()) return world.status();
  if (!count.ok()) return count.status();
  if (!bytes.ok()) return bytes.status();
  if (*schema != kSchema || *abi != kAbi || *family != kModelFamily ||
      *support != kSupportState || dspark == nullptr ||
      !dspark->is_boolean() || rows == nullptr || !rows->is_array() ||
      rows->array().size() != authority.record_count ||
      *layout != authority.layout_root ||
      *disposition != authority.disposition_root ||
      *set_root != authority.record_set_root ||
      *body_digest != authority.body_sha256 ||
      *records_root != authority.runtime_records_root ||
      *world != authority.world_size || *count != authority.record_count ||
      *bytes != authority.record_bytes ||
      dspark->boolean() != authority.dspark_enabled) {
    return Status::InvalidArgument(
        "DeepSeek runtime records differ from artifact authority");
  }
  const std::array<std::string_view, 2> omitted{
      "body_sha256", "runtime_records_root"};
  auto body = canonical_ascii_json(*root, kMaximumBytes, omitted);
  if (!body.ok()) return body.status();
  auto observed_body = sha256(std::as_bytes(std::span(*body)));
  if (!observed_body.ok()) return observed_body.status();
  if (*observed_body != authority.body_sha256) {
    return Status::InvalidArgument(
        "DeepSeek runtime records body digest differs");
  }

  DeepSeekRuntimeRecordsManifest result;
  result.authority_ = authority;
  result.records_.reserve(rows->array().size());
  std::uint64_t total_bytes = 0;
  std::map<std::string, std::vector<std::pair<std::uint64_t, std::uint64_t>>,
           std::less<>>
      ranges;
  for (const auto& row : rows->array()) {
    auto record = parse_record(row, pipeline, authority.dspark_enabled);
    if (!record.ok()) return record.status();
    if (!result.records_.empty() &&
        result.records_.back().tensor_name >= record->tensor_name) {
      return Status::InvalidArgument(
          "DeepSeek runtime records are duplicated or unordered");
    }
    auto next = checked_add_u64(total_bytes, record->tensor_bytes);
    if (!next.ok()) return next.status();
    total_bytes = *next;
    ranges[record->shard_name].push_back(
        {record->file_begin, record->file_end});
    result.records_.push_back(std::move(*record));
  }
  if (total_bytes != authority.record_bytes) {
    return Status::InvalidArgument(
        "DeepSeek runtime record byte ledger differs");
  }
  for (auto& [name, intervals] : ranges) {
    (void)name;
    std::ranges::sort(intervals);
    for (std::size_t index = 1; index < intervals.size(); ++index) {
      if (intervals[index - 1].second > intervals[index].first) {
        return Status::InvalidArgument(
            "DeepSeek runtime target ranges overlap");
      }
    }
  }
  auto observed_set = record_set_root(result.records_);
  if (!observed_set.ok()) return observed_set.status();
  if (*observed_set != authority.record_set_root) {
    return Status::InvalidArgument(
        "DeepSeek runtime record set root differs");
  }
  auto observed_root = runtime_records_root(authority);
  if (!observed_root.ok()) return observed_root.status();
  if (*observed_root != authority.runtime_records_root) {
    return Status::InvalidArgument(
        "DeepSeek runtime records root differs");
  }
  return result;
}

Result<DeepSeekRuntimeRecordsManifest::Encoded>
DeepSeekRuntimeRecordsManifest::Encode(std::span<const DeepSeekRuntimeRecord> records,
    Sha256Digest layout_root, Sha256Digest disposition_root,
    const DeepSeekPipelinePlan& pipeline) {
  if (records.empty() || records.size() > kMaximumRecordCount ||
      pipeline.world_size() == 0 || pipeline.world_size() > 4 ||
      layout_root == Sha256Digest{} || disposition_root == Sha256Digest{}) {
    return Status::InvalidArgument("Runtime record encoding authority is invalid");
  }
  DeepSeekRuntimeRecordsAuthority authority;
  authority.layout_root = layout_root;
  authority.disposition_root = disposition_root;
  authority.world_size = pipeline.world_size();
  authority.dspark_enabled = pipeline.rank(pipeline.world_size() - 1).owns_dspark;
  authority.record_count = static_cast<std::uint32_t>(records.size());
  const auto string = [](std::string_view value) { return JsonValue(std::string(value)); };
  const auto integer = [](std::uint64_t value) { return JsonValue(static_cast<std::int64_t>(value)); };
  constexpr auto maximum = static_cast<std::uint64_t>(INT64_MAX);
  constexpr std::array<std::pair<std::string_view, std::string_view>, 10> storage_names{{
      {"F32", "direct_f32_le_bits"}, {"F16", "direct_f16_le_bits"},
      {"BF16", "direct_bf16_le_bits"}, {"I8", "direct_mxfp4_e2m1_packed_bits"},
      {"U8", "direct_u8_bits"}, {"F8_E4M3", "direct_fp8_e4m3_bits"},
      {"F8_E8M0", "direct_ue8m0_scale_bits"}, {"I32", "direct_i32_le_bits"},
      {"I64", "direct_i64_le_bits"}, {"BOOL", "direct_bool_bits"}}};
  std::vector<DeepSeekRuntimeRecord> derived;
  derived.reserve(records.size());
  JsonValue::Array rows;
  rows.reserve(records.size());
  std::string_view previous;
  for (const auto& input : records) {
    if (input.tensor_name.empty() || input.tensor_name.size() > kMaximumTensorNameBytes ||
        input.tensor_name <= previous || input.shard_name.size() > 255 ||
        input.name_space.size() > 32 || input.shape.empty() || input.shape.size() > 8 ||
        input.file_begin > maximum || input.file_end > maximum ||
        input.tensor_bytes > maximum - authority.record_bytes) {
      return Status::InvalidArgument("Runtime record encoding exceeds bounds or order");
    }
    previous = input.tensor_name;
    JsonValue::Array shape;
    for (const auto dimension : input.shape) {
      if (dimension == 0 || dimension > maximum)
        return Status::InvalidArgument("Runtime record encoding dimension is invalid");
      shape.push_back(integer(dimension));
    }
    std::string_view role;
    switch (input.role) {
      case DeepSeekTensorRole::kEmbedding: role = "embedding"; break;
      case DeepSeekTensorRole::kMainLayer: role = "main_layer"; break;
      case DeepSeekTensorRole::kFinalHead: role = "final_head"; break;
      case DeepSeekTensorRole::kDspark: role = "dspark_stage"; break;
    }
    std::string_view dtype_name, semantics_name;
    for (const auto& names : storage_names) {
      auto storage = parse_storage(names.first, names.second);
      if (storage.ok() && storage->first == input.dtype && storage->second == input.storage_semantics) {
        dtype_name = names.first; semantics_name = names.second; break;
      }
    }
    if (role.empty() || dtype_name.empty())
      return Status::InvalidArgument("Runtime record encoding role or storage is invalid");
    auto record = input;
    auto root = runtime_record_root(record);
    if (!root.ok()) return root.status();
    record.runtime_record_root = *root;
    JsonValue row(JsonValue::Object{
        {"name", string(record.tensor_name)}, {"shard", string(record.shard_name)},
        {"namespace", string(record.name_space)}, {"role", string(role)},
        {"logical_layer", integer(record.logical_layer)}, {"owner_rank", integer(record.owner_rank)},
        {"dtype", string(dtype_name)}, {"shape", JsonValue(std::move(shape))},
        {"file_begin", integer(record.file_begin)}, {"file_end", integer(record.file_end)},
        {"tensor_bytes", integer(record.tensor_bytes)}, {"storage_semantics", string(semantics_name)},
        {"transform", string(kTransform)}, {"target_logical_root", string(record.target_logical_root.hex())},
        {"disposition_record_root", string(record.disposition_record_root.hex())},
        {"layout_record_root", string(record.layout_record_root.hex())},
        {"runtime_record_root", string(root->hex())}});
    auto checked = parse_record(row, pipeline, authority.dspark_enabled);
    if (!checked.ok()) return checked.status();
    authority.record_bytes += record.tensor_bytes;
    derived.push_back(std::move(record));
    rows.push_back(std::move(row));
  }
  auto set_root = record_set_root(derived);
  if (!set_root.ok()) return set_root.status();
  authority.record_set_root = *set_root;
  JsonValue::Object fields{
      {"schema", string(kSchema)}, {"abi", string(kAbi)}, {"model_family", string(kModelFamily)},
      {"support_state", string(kSupportState)}, {"layout_root", string(layout_root.hex())},
      {"disposition_root", string(disposition_root.hex())},
      {"dspark_enabled", JsonValue(authority.dspark_enabled)}, {"world_size", integer(authority.world_size)},
      {"record_count", integer(authority.record_count)}, {"record_bytes", integer(authority.record_bytes)},
      {"record_set_root", string(set_root->hex())}, {"records", JsonValue(std::move(rows))}};
  auto body = canonical_ascii_json(JsonValue(fields), kMaximumBytes);
  if (!body.ok()) return body.status();
  auto body_hash = sha256(std::as_bytes(std::span(*body)));
  if (!body_hash.ok()) return body_hash.status();
  authority.body_sha256 = *body_hash;
  auto root = runtime_records_root(authority);
  if (!root.ok()) return root.status();
  authority.runtime_records_root = *root;
  fields.emplace_back("body_sha256", string(body_hash->hex()));
  fields.emplace_back("runtime_records_root", string(root->hex()));
  auto json = canonical_ascii_json(JsonValue(std::move(fields)), kMaximumBytes);
  if (!json.ok()) return json.status();
  auto object_hash = sha256(std::as_bytes(std::span(*json)));
  if (!object_hash.ok()) return object_hash.status();
  authority.object_sha256 = *object_hash;
  authority.object_bytes = json->size();
  // Full reparse checks canonical bytes, record ordering/ranges, all typed roots
  // and the authority ledger using exactly the runtime admission implementation.
  auto checked = Parse(*json, authority, pipeline);
  if (!checked.ok()) return checked.status();
  return Encoded{std::move(*json), authority};
}

const DeepSeekRuntimeRecord* DeepSeekRuntimeRecordsManifest::find(
    std::string_view tensor_name) const noexcept {
  const auto found = std::lower_bound(
      records_.begin(), records_.end(), tensor_name,
      [](const auto& record, std::string_view name) {
        return record.tensor_name < name;
      });
  return found != records_.end() && found->tensor_name == tensor_name
             ? &*found
             : nullptr;
}

}  // namespace pih
