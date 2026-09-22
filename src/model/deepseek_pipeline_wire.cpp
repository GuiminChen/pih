#include "pih/model/deepseek_pipeline_wire.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "pih/core/sha256.h"

namespace pih {
namespace {

void put16(std::byte* p, std::uint16_t v) {
  p[0] = static_cast<std::byte>(v);
  p[1] = static_cast<std::byte>(v >> 8U);
}
void put32(std::byte* p, std::uint32_t v) {
  for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<std::byte>(v >> (8U * i));
}
void put64(std::byte* p, std::uint64_t v) {
  for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<std::byte>(v >> (8U * i));
}
std::uint16_t get16(const std::byte* p) {
  return static_cast<std::uint16_t>(p[0]) |
         (static_cast<std::uint16_t>(p[1]) << 8U);
}
std::uint32_t get32(const std::byte* p) {
  std::uint32_t v = 0;
  for (unsigned i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8U * i);
  return v;
}
std::uint64_t get64(const std::byte* p) {
  std::uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8U * i);
  return v;
}

std::uint32_t crc32_ieee(std::span<const std::byte> bytes) {
  std::uint32_t crc = 0xffffffffU;
  for (const auto byte : bytes) {
    crc ^= static_cast<std::uint8_t>(byte);
    for (unsigned bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
  }
  return ~crc;
}

Status validate_records(const DeepSeekPipelineWireMessage& message) {
  if (message.engine_epoch == 0 || message.plan_sequence == 0 ||
      message.records.empty() ||
      message.records.size() > std::numeric_limits<std::uint32_t>::max() ||
      message.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument("DeepSeek wire message bounds are invalid");
  }
  std::uint64_t next_offset = 0;
  for (const auto& record : message.records) {
    if (record.sequence_generation == 0 || record.state_generation == 0 ||
        record.committed_length > record.reserved_length || record.flags != 0) {
      return Status::InvalidArgument("DeepSeek sequence wire record is invalid");
    }
    if (message.phase == DeepSeekPlanPhase::kDrain) {
      if (record.token_count != 0 || record.token_offset != 0) {
        return Status::InvalidArgument("DeepSeek drain record has a token span");
      }
      continue;
    }
    if (record.token_count == 0 || record.token_offset != next_offset) {
      return Status::InvalidArgument("DeepSeek sequence token span is invalid");
    }
    next_offset += record.token_count;
    if (next_offset > message.token_ids.size()) {
      return Status::InvalidArgument("DeepSeek sequence span exceeds tokens");
    }
  }
  if (message.phase == DeepSeekPlanPhase::kDrain && !message.token_ids.empty()) {
    return Status::InvalidArgument("DeepSeek drain message has token IDs");
  }
  if (next_offset != message.token_ids.size()) {
    return Status::InvalidArgument("DeepSeek sequence spans do not cover tokens");
  }
  return Status::Ok();
}

}  // namespace

Result<std::uint64_t> DeepSeekPipelineWireCodec::PaddedBytes(
    std::uint32_t token_count, std::uint32_t sequence_count) {
  const std::uint64_t logical = kHeaderBytes +
      std::uint64_t{kRecordBytes} * sequence_count +
      std::uint64_t{sizeof(std::uint32_t)} * token_count + kTrailerBytes;
  if (logical > std::numeric_limits<std::uint64_t>::max() - 255U) {
    return Status::ResourceExhausted("DeepSeek control payload overflows");
  }
  return (logical + 255U) & ~std::uint64_t{255U};
}

Result<std::vector<std::byte>> DeepSeekPipelineWireCodec::Encode(
    const DeepSeekPipelineWireMessage& message) {
  const auto valid = validate_records(message);
  if (!valid.ok()) return valid;
  if (static_cast<std::uint8_t>(message.phase) >
      static_cast<std::uint8_t>(DeepSeekPlanPhase::kDrain)) {
    return Status::InvalidArgument("DeepSeek wire phase is unknown");
  }
  const auto token_count = static_cast<std::uint32_t>(message.token_ids.size());
  const auto sequence_count = static_cast<std::uint32_t>(message.records.size());
  auto padded = PaddedBytes(token_count, sequence_count);
  if (!padded.ok()) return padded.status();
  if (*padded > std::numeric_limits<std::size_t>::max() ||
      *padded > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("DeepSeek control payload is too large");
  }
  std::vector<std::byte> output(static_cast<std::size_t>(*padded));
  auto* h = output.data();
  put16(h + 0, kSchemaVersion);
  h[2] = static_cast<std::byte>(message.phase);
  h[3] = message.phase == DeepSeekPlanPhase::kDrain ? std::byte{1} : std::byte{0};
  put16(h + 4, static_cast<std::uint16_t>(kHeaderBytes));
  h[6] = std::byte{1};  // BF16
  put64(h + 8, message.engine_epoch);
  put64(h + 16, message.plan_sequence);
  put64(h + 24, message.microbatch_id);
  put32(h + 32, token_count);
  put32(h + 36, sequence_count);
  put64(h + 40, std::uint64_t{token_count} * 4U * 4096U);
  put32(h + 48, static_cast<std::uint32_t>(*padded));

  std::size_t cursor = kHeaderBytes;
  for (const auto& r : message.records) {
    auto* p = output.data() + cursor;
    put64(p + 0, r.sequence_id.words[0]);
    put64(p + 8, r.sequence_id.words[1]);
    put64(p + 16, r.sequence_generation);
    put64(p + 24, r.state_generation);
    put32(p + 32, r.token_offset);
    put32(p + 36, r.token_count);
    put32(p + 40, r.committed_length);
    put32(p + 44, r.reserved_length);
    put32(p + 48, r.absolute_position_base);
    put32(p + 52, r.flags);
    put32(p + 56, r.sampling_config_id);
    cursor += kRecordBytes;
  }
  for (const auto token : message.token_ids) {
    put32(output.data() + cursor, token);
    cursor += 4;
  }
  const auto payload = std::span<const std::byte>(
      output.data() + kHeaderBytes, cursor - kHeaderBytes);
  auto digest = sha256(payload);
  if (!digest.ok()) return digest.status();
  std::copy_n(digest->bytes.begin(), 16, output.begin() + 56);
  put64(output.data() + cursor, payload.size());
  put32(output.data() + cursor + 8, crc32_ieee(payload));
  put16(output.data() + cursor + 12, kSchemaVersion);
  put16(output.data() + cursor + 14, kTrailerBytes);
  put32(output.data() + 72,
        crc32_ieee(std::span<const std::byte>(output.data(), 72)));
  return output;
}

Result<DeepSeekPipelineWireMessage> DeepSeekPipelineWireCodec::Decode(
    std::span<const std::byte> bytes, std::uint32_t max_tokens,
    std::uint32_t max_sequences) {
  if (bytes.size() < kHeaderBytes + kTrailerBytes ||
      get16(bytes.data()) != kSchemaVersion || get16(bytes.data() + 4) != kHeaderBytes ||
      bytes[6] != std::byte{1} || bytes[7] != std::byte{0} ||
      get32(bytes.data() + 52) != 0 ||
      get32(bytes.data() + 76) != 0 ||
      get32(bytes.data() + 72) != crc32_ieee(bytes.first(72))) {
    return Status::InvalidArgument("DeepSeek wire header integrity failed");
  }
  const auto phase_raw = static_cast<std::uint8_t>(bytes[2]);
  if (phase_raw > static_cast<std::uint8_t>(DeepSeekPlanPhase::kDrain)) {
    return Status::InvalidArgument("DeepSeek wire phase is unknown");
  }
  const bool is_drain = phase_raw == static_cast<std::uint8_t>(DeepSeekPlanPhase::kDrain);
  if (bytes[3] != (is_drain ? std::byte{1} : std::byte{0})) {
    return Status::InvalidArgument("DeepSeek wire flags do not match phase");
  }
  const auto token_count = get32(bytes.data() + 32);
  const auto sequence_count = get32(bytes.data() + 36);
  auto expected_size = PaddedBytes(token_count, sequence_count);
  if (!expected_size.ok() || *expected_size != bytes.size() ||
      get32(bytes.data() + 48) != bytes.size() ||
      sequence_count == 0 || token_count > max_tokens ||
      sequence_count > max_sequences ||
      (!is_drain && (token_count == 0 || sequence_count > token_count)) ||
      (is_drain && token_count != 0) ||
      get64(bytes.data() + 40) != std::uint64_t{token_count} * 4U * 4096U) {
    return Status::InvalidArgument("DeepSeek wire capacity fields are invalid");
  }
  const std::size_t payload_size =
      std::size_t{kRecordBytes} * sequence_count + std::size_t{4} * token_count;
  const std::size_t trailer = kHeaderBytes + payload_size;
  const auto payload = bytes.subspan(kHeaderBytes, payload_size);
  auto digest = sha256(payload);
  if (!digest.ok()) return digest.status();
  if (!std::equal(digest->bytes.begin(), digest->bytes.begin() + 16,
                  bytes.begin() + 56) || get64(bytes.data() + trailer) != payload_size ||
      get32(bytes.data() + trailer + 8) != crc32_ieee(payload) ||
      get16(bytes.data() + trailer + 12) != kSchemaVersion ||
      get16(bytes.data() + trailer + 14) != kTrailerBytes ||
      !std::all_of(bytes.begin() + trailer + kTrailerBytes, bytes.end(),
                   [](std::byte b) { return b == std::byte{0}; })) {
    return Status::InvalidArgument("DeepSeek wire payload integrity failed");
  }
  DeepSeekPipelineWireMessage message;
  message.engine_epoch = get64(bytes.data() + 8);
  message.plan_sequence = get64(bytes.data() + 16);
  message.microbatch_id = get64(bytes.data() + 24);
  message.phase = static_cast<DeepSeekPlanPhase>(phase_raw);
  message.records.resize(sequence_count);
  std::size_t cursor = kHeaderBytes;
  for (auto& r : message.records) {
    const auto* p = bytes.data() + cursor;
    r.sequence_id.words = {get64(p), get64(p + 8)};
    r.sequence_generation = get64(p + 16);
    r.state_generation = get64(p + 24);
    r.token_offset = get32(p + 32);
    r.token_count = get32(p + 36);
    r.committed_length = get32(p + 40);
    r.reserved_length = get32(p + 44);
    r.absolute_position_base = get32(p + 48);
    r.flags = get32(p + 52);
    r.sampling_config_id = get32(p + 56);
    if (get32(p + 60) != 0) {
      return Status::InvalidArgument("DeepSeek wire record reserved field is nonzero");
    }
    cursor += kRecordBytes;
  }
  message.token_ids.resize(token_count);
  for (auto& token : message.token_ids) {
    token = get32(bytes.data() + cursor);
    cursor += 4;
  }
  const auto valid = validate_records(message);
  if (!valid.ok()) return valid;
  return message;
}

}  // namespace pih
