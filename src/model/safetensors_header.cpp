#include "pih/model/safetensors_header.h"

#include <algorithm>
#include <cstring>

#include "pih/core/bounded_json.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<DType> parse_dtype(std::string_view name) {
  if (name == "F32") return DType::kFloat32;
  if (name == "F16") return DType::kFloat16;
  if (name == "BF16") return DType::kBFloat16;
  if (name == "I8") return DType::kInt8;
  if (name == "U8") return DType::kUInt8;
  if (name == "F8_E4M3") return DType::kFloat8E4M3;
  if (name == "F8_E8M0") return DType::kFloat8E8M0;
  if (name == "I32") return DType::kInt32;
  if (name == "I64") return DType::kInt64;
  if (name == "BOOL") return DType::kBool;
  return Status::InvalidArgument("unsupported Safetensors dtype");
}

Result<std::uint64_t> nonnegative_integer(const JsonValue* value,
                                          const char* field) {
  if (value == nullptr || !value->is_integer() || value->integer() < 0) {
    return Status::InvalidArgument(std::string(field) + " must be nonnegative integer");
  }
  return static_cast<std::uint64_t>(value->integer());
}

Result<SafetensorRecord> parse_record(const std::string& name,
                                      const JsonValue& value,
                                      std::uint64_t data_start) {
  if (name.empty() || name.size() > SafetensorsHeader::kMaxTensorNameBytes) {
    return Status::InvalidArgument("Safetensors tensor name is invalid");
  }
  if (!value.is_object() || value.object().size() != 3) {
    return Status::InvalidArgument("Safetensors tensor record must have three fields");
  }
  const auto* dtype_json = value.at("dtype");
  const auto* shape_json = value.at("shape");
  const auto* offsets_json = value.at("data_offsets");
  if (dtype_json == nullptr || !dtype_json->is_string() || shape_json == nullptr ||
      !shape_json->is_array() || offsets_json == nullptr ||
      !offsets_json->is_array() || offsets_json->array().size() != 2) {
    return Status::InvalidArgument("Safetensors tensor record fields are invalid");
  }
  auto dtype = parse_dtype(dtype_json->string());
  if (!dtype.ok()) return dtype.status();
  if (shape_json->array().size() > SafetensorsHeader::kMaxRank) {
    return Status::InvalidArgument("Safetensors tensor rank exceeds limit");
  }
  std::vector<std::uint64_t> shape;
  shape.reserve(shape_json->array().size());
  std::uint64_t elements = 1;
  for (const auto& dimension : shape_json->array()) {
    auto parsed = nonnegative_integer(&dimension, "shape dimension");
    if (!parsed.ok()) return parsed.status();
    shape.push_back(parsed.value());
    auto product = checked_mul_u64(elements, parsed.value());
    if (!product.ok()) return product.status();
    elements = product.value();
  }
  auto begin = nonnegative_integer(&offsets_json->array()[0], "data offset");
  auto end = nonnegative_integer(&offsets_json->array()[1], "data offset");
  if (!begin.ok()) return begin.status();
  if (!end.ok()) return end.status();
  if (end.value() < begin.value()) {
    return Status::InvalidArgument("Safetensors offsets are reversed");
  }
  auto element_bytes = dtype_size(dtype.value());
  auto expected = checked_mul_u64(elements, element_bytes.value());
  if (!expected.ok()) return expected.status();
  if (end.value() - begin.value() != expected.value()) {
    return Status::InvalidArgument("Safetensors tensor byte count mismatches shape");
  }
  auto absolute_begin = checked_add_u64(data_start, begin.value());
  auto absolute_end = checked_add_u64(data_start, end.value());
  if (!absolute_begin.ok()) return absolute_begin.status();
  if (!absolute_end.ok()) return absolute_end.status();
  return SafetensorRecord{name, dtype.value(), std::move(shape),
                          absolute_begin.value(), absolute_end.value()};
}

}  // namespace

Result<SafetensorsHeader> SafetensorsHeader::ParsePrefix(
    std::span<const std::byte> prefix, std::uint64_t file_bytes) {
  if (prefix.size() < 8 || file_bytes < 8) {
    return Status::InvalidArgument("Safetensors file is shorter than length prefix");
  }
  std::uint64_t header_bytes = 0;
  for (int index = 0; index < 8; ++index) {
    header_bytes |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(prefix[index]))
                    << (index * 8);
  }
  if (header_bytes == 0 || header_bytes > kMaxHeaderBytes) {
    return Status::ResourceExhausted("Safetensors header length is outside budget");
  }
  auto data_start_result = checked_add_u64(8, header_bytes);
  if (!data_start_result.ok()) return data_start_result.status();
  const auto data_start = data_start_result.value();
  if (data_start > file_bytes || prefix.size() < data_start) {
    return Status::InvalidArgument("Safetensors header is truncated");
  }
  const auto json = std::string_view(
      reinterpret_cast<const char*>(prefix.data() + 8),
      static_cast<std::size_t>(header_bytes));
  JsonLimits limits;
  limits.max_input_bytes = static_cast<std::size_t>(kMaxHeaderBytes);
  limits.max_depth = 16;
  limits.max_nodes = 65'536;
  limits.max_string_bytes = kMaxTensorNameBytes;
  auto parsed = JsonValue::Parse(json, limits);
  if (!parsed.ok()) return parsed.status();
  if (!parsed->is_object()) {
    return Status::InvalidArgument("Safetensors header root must be an object");
  }
  std::vector<SafetensorRecord> records;
  records.reserve(parsed->object().size());
  for (const auto& entry : parsed->object()) {
    if (entry.first == "__metadata__") {
      if (!entry.second.is_object()) {
        return Status::InvalidArgument("Safetensors metadata must be an object");
      }
      continue;
    }
    if (records.size() == kMaxTensorCount) {
      return Status::ResourceExhausted("Safetensors tensor count exceeds budget");
    }
    auto record = parse_record(entry.first, entry.second, data_start);
    if (!record.ok()) return record.status();
    records.push_back(std::move(record).value());
  }
  if (records.empty()) {
    return Status::InvalidArgument("Safetensors header has no tensors");
  }
  std::vector<const SafetensorRecord*> ordered;
  ordered.reserve(records.size());
  for (const auto& record : records) ordered.push_back(&record);
  std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
    return left->file_begin < right->file_begin;
  });
  std::uint64_t cursor = data_start;
  for (const auto* record : ordered) {
    if (record->file_begin != cursor) {
      return Status::InvalidArgument("Safetensors data ranges overlap or contain a gap");
    }
    cursor = record->file_end;
  }
  if (cursor != file_bytes) {
    return Status::InvalidArgument("Safetensors data ranges do not cover the file");
  }
  return SafetensorsHeader(header_bytes, file_bytes - data_start,
                           std::move(records));
}

const SafetensorRecord* SafetensorsHeader::tensor(std::string_view name) const noexcept {
  for (const auto& tensor : tensors_) if (tensor.name == name) return &tensor;
  return nullptr;
}

}  // namespace pih
