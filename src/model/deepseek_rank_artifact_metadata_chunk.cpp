#include "pih/model/deepseek_rank_artifact_metadata_chunk.h"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kChunkFrameType = 9;
constexpr std::uint16_t kAckFrameType = 10;
constexpr std::uint16_t kFrameVersion = 1;
constexpr std::size_t kChunkHeaderBytes = 292;
constexpr std::size_t kTrailingRootBytes = 32;

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<std::uint32_t> expected_chunk_count(std::uint64_t total_blob_bytes) {
  if (total_blob_bytes == 0 ||
      total_blob_bytes > kDeepSeekRankArtifactMetadataBlobMaximumBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata total bytes are invalid");
  }
  const auto count =
      (total_blob_bytes +
       kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes - 1U) /
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes;
  if (count == 0 || count > UINT32_MAX) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk count is invalid");
  }
  return static_cast<std::uint32_t>(count);
}

Result<std::uint32_t> expected_payload_bytes(
    std::uint64_t total_blob_bytes, std::uint32_t chunk_index,
    std::uint32_t chunk_count) {
  auto expected_count = expected_chunk_count(total_blob_bytes);
  if (!expected_count.ok()) return expected_count.status();
  if (chunk_count != *expected_count || chunk_index >= chunk_count) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk geometry is invalid");
  }
  const auto offset =
      static_cast<std::uint64_t>(chunk_index) *
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes;
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes,
      total_blob_bytes - offset));
}

Status validate_common(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, std::uint32_t rank,
    std::uint32_t chunk_index, std::uint32_t chunk_count,
    std::uint64_t total_blob_bytes,
    std::uint64_t process_manifest_identity,
    std::uint64_t process_identity, std::uint64_t pidfd_identity,
    std::uint64_t control_identity, std::uint64_t challenge_identity,
    const Sha256Digest& transfer_manifest_root,
    const Sha256Digest& descriptor_transfer_transaction_root,
    const Sha256Digest& metadata_root,
    const Sha256Digest& metadata_transaction_root,
    const Sha256Digest& blob_sha256,
    const Sha256Digest& chunk_sha256) {
  auto payload = expected_payload_bytes(
      total_blob_bytes, chunk_index, chunk_count);
  if (!payload.ok()) return payload.status();
  if (engine_epoch == 0 || worker_generation == 0 || world_size < 1 ||
      world_size > 4 || rank >= world_size ||
      process_manifest_identity == 0 || process_identity == 0 ||
      pidfd_identity == 0 || control_identity == 0 ||
      challenge_identity == 0 || !nonzero(transfer_manifest_root) ||
      !nonzero(descriptor_transfer_transaction_root) ||
      !nonzero(metadata_root) || !nonzero(metadata_transaction_root) ||
      !nonzero(blob_sha256) || !nonzero(chunk_sha256)) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk identity is invalid");
  }
  return Status::Ok();
}

Result<Sha256Digest> compile_chunk_root(
    const DeepSeekRankArtifactMetadataChunkFields& fields,
    std::span<const std::byte> payload) {
  auto status = validate_common(
      fields.engine_epoch, fields.worker_generation, fields.world_size,
      fields.rank, fields.chunk_index, fields.chunk_count,
      fields.total_blob_bytes, fields.process_manifest_identity,
      fields.process_identity, fields.pidfd_identity, fields.control_identity,
      fields.challenge_identity, fields.transfer_manifest_root,
      fields.descriptor_transfer_transaction_root, fields.metadata_root,
      fields.metadata_transaction_root, fields.blob_sha256,
      fields.chunk_sha256);
  if (!status.ok()) return status;
  auto expected_bytes = expected_payload_bytes(
      fields.total_blob_bytes, fields.chunk_index, fields.chunk_count);
  if (!expected_bytes.ok()) return expected_bytes.status();
  const auto expected_offset =
      static_cast<std::uint64_t>(fields.chunk_index) *
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes;
  auto payload_root = sha256(payload);
  if (!payload_root.ok()) return payload_root.status();
  if (fields.payload_offset != expected_offset ||
      fields.payload_bytes != *expected_bytes ||
      payload.size() != fields.payload_bytes ||
      *payload_root != fields.chunk_sha256) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata chunk payload differs");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-metadata-chunk:v1", 20);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) status = builder->add_u32(5, fields.chunk_index);
  if (status.ok()) status = builder->add_u32(6, fields.chunk_count);
  if (status.ok()) status = builder->add_u64(7, fields.payload_offset);
  if (status.ok()) status = builder->add_u32(8, fields.payload_bytes);
  if (status.ok()) status = builder->add_u64(9, fields.total_blob_bytes);
  if (status.ok()) {
    status = builder->add_u64(10, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(11, fields.process_identity);
  if (status.ok()) status = builder->add_u64(12, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(13, fields.control_identity);
  if (status.ok()) status = builder->add_u64(14, fields.challenge_identity);
  if (status.ok()) status = builder->add_hash(15, fields.transfer_manifest_root);
  if (status.ok()) {
    status = builder->add_hash(
        16, fields.descriptor_transfer_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(17, fields.metadata_root);
  if (status.ok()) {
    status = builder->add_hash(18, fields.metadata_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(19, fields.blob_sha256);
  if (status.ok()) status = builder->add_hash(20, fields.chunk_sha256);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_ack_root(
    const DeepSeekRankArtifactMetadataChunkAckFields& fields) {
  auto status = validate_common(
      fields.engine_epoch, fields.worker_generation, fields.world_size,
      fields.rank, fields.chunk_index, fields.chunk_count,
      fields.total_blob_bytes, fields.process_manifest_identity,
      fields.process_identity, fields.pidfd_identity, fields.control_identity,
      fields.challenge_identity, fields.transfer_manifest_root,
      fields.descriptor_transfer_transaction_root, fields.metadata_root,
      fields.metadata_transaction_root, fields.blob_sha256,
      fields.chunk_sha256);
  if (!status.ok()) return status;
  auto payload_bytes = expected_payload_bytes(
      fields.total_blob_bytes, fields.chunk_index, fields.chunk_count);
  if (!payload_bytes.ok()) return payload_bytes.status();
  const auto expected_cumulative =
      static_cast<std::uint64_t>(fields.chunk_index) *
          kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes +
      *payload_bytes;
  if (fields.cumulative_payload_bytes != expected_cumulative) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata ACK cumulative bytes differ");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-metadata-chunk-ack:v1", 19);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) status = builder->add_u32(5, fields.chunk_index);
  if (status.ok()) status = builder->add_u32(6, fields.chunk_count);
  if (status.ok()) {
    status = builder->add_u64(7, fields.cumulative_payload_bytes);
  }
  if (status.ok()) status = builder->add_u64(8, fields.total_blob_bytes);
  if (status.ok()) {
    status = builder->add_u64(9, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(10, fields.process_identity);
  if (status.ok()) status = builder->add_u64(11, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(12, fields.control_identity);
  if (status.ok()) status = builder->add_u64(13, fields.challenge_identity);
  if (status.ok()) status = builder->add_hash(14, fields.transfer_manifest_root);
  if (status.ok()) {
    status = builder->add_hash(
        15, fields.descriptor_transfer_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(16, fields.metadata_root);
  if (status.ok()) {
    status = builder->add_hash(17, fields.metadata_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(18, fields.blob_sha256);
  if (status.ok()) status = builder->add_hash(19, fields.chunk_sha256);
  if (!status.ok()) return status;
  return builder->finalize();
}

template <class Output, class T>
void put(Output& output, std::size_t& cursor, T value) {
  using U = std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[cursor++] =
        static_cast<std::byte>((bits >> (index * 8U)) & 0xffU);
  }
}

template <class Output>
void put_digest(Output& output, std::size_t& cursor,
                const Sha256Digest& value) {
  for (const auto byte : value.bytes) output[cursor++] = byte;
}

template <class T>
bool get(std::span<const std::byte> input, std::size_t& cursor, T* value) {
  if (value == nullptr || cursor > input.size() ||
      input.size() - cursor < sizeof(T)) {
    return false;
  }
  using U = std::make_unsigned_t<T>;
  U bits = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    bits |= static_cast<U>(std::to_integer<unsigned>(input[cursor++]))
            << (index * 8U);
  }
  *value = static_cast<T>(bits);
  return true;
}

bool get_digest(std::span<const std::byte> input, std::size_t& cursor,
                Sha256Digest* value) {
  if (value == nullptr || cursor > input.size() ||
      input.size() - cursor < value->bytes.size()) {
    return false;
  }
  for (auto& byte : value->bytes) byte = input[cursor++];
  return true;
}

template <class Output, class Fields>
void put_common(Output& output, std::size_t& cursor, std::uint16_t type,
                const Fields& fields) {
  put(output, cursor, kMagic);
  put(output, cursor, type);
  put(output, cursor, kFrameVersion);
  put(output, cursor, fields.engine_epoch);
  put(output, cursor, fields.worker_generation);
  put(output, cursor, fields.world_size);
  put(output, cursor, fields.rank);
  put(output, cursor, fields.chunk_index);
  put(output, cursor, fields.chunk_count);
}

template <class Fields>
bool get_common(std::span<const std::byte> frame, std::size_t& cursor,
                std::uint16_t expected_type, Fields* fields) {
  std::uint32_t magic = 0;
  std::uint16_t type = 0;
  std::uint16_t version = 0;
  return fields != nullptr && get(frame, cursor, &magic) &&
         get(frame, cursor, &type) && get(frame, cursor, &version) &&
         magic == kMagic && type == expected_type &&
         version == kFrameVersion &&
         get(frame, cursor, &fields->engine_epoch) &&
         get(frame, cursor, &fields->worker_generation) &&
         get(frame, cursor, &fields->world_size) &&
         get(frame, cursor, &fields->rank) &&
         get(frame, cursor, &fields->chunk_index) &&
         get(frame, cursor, &fields->chunk_count);
}

template <class Output, class Fields>
void put_identities_and_roots(Output& output, std::size_t& cursor,
                              const Fields& fields) {
  put(output, cursor, fields.process_manifest_identity);
  put(output, cursor, fields.process_identity);
  put(output, cursor, fields.pidfd_identity);
  put(output, cursor, fields.control_identity);
  put(output, cursor, fields.challenge_identity);
  put_digest(output, cursor, fields.transfer_manifest_root);
  put_digest(output, cursor, fields.descriptor_transfer_transaction_root);
  put_digest(output, cursor, fields.metadata_root);
  put_digest(output, cursor, fields.metadata_transaction_root);
  put_digest(output, cursor, fields.blob_sha256);
  put_digest(output, cursor, fields.chunk_sha256);
}

template <class Fields>
bool get_identities_and_roots(std::span<const std::byte> frame,
                              std::size_t& cursor, Fields* fields) {
  return fields != nullptr &&
         get(frame, cursor, &fields->process_manifest_identity) &&
         get(frame, cursor, &fields->process_identity) &&
         get(frame, cursor, &fields->pidfd_identity) &&
         get(frame, cursor, &fields->control_identity) &&
         get(frame, cursor, &fields->challenge_identity) &&
         get_digest(frame, cursor, &fields->transfer_manifest_root) &&
         get_digest(frame, cursor,
                    &fields->descriptor_transfer_transaction_root) &&
         get_digest(frame, cursor, &fields->metadata_root) &&
         get_digest(frame, cursor, &fields->metadata_transaction_root) &&
         get_digest(frame, cursor, &fields->blob_sha256) &&
         get_digest(frame, cursor, &fields->chunk_sha256);
}

}  // namespace

DeepSeekRankArtifactMetadataChunk::DeepSeekRankArtifactMetadataChunk(
    DeepSeekRankArtifactMetadataChunkFields fields,
    std::vector<std::byte> payload, Sha256Digest frame_root) noexcept
    : fields_(std::move(fields)), payload_(std::move(payload)),
      frame_root_(frame_root) {}

Result<DeepSeekRankArtifactMetadataChunk>
DeepSeekRankArtifactMetadataChunk::Create(
    DeepSeekRankArtifactMetadataChunkFields fields,
    std::vector<std::byte> payload) {
  auto root = compile_chunk_root(fields, payload);
  if (!root.ok()) return root.status();
  return DeepSeekRankArtifactMetadataChunk(
      std::move(fields), std::move(payload), *root);
}

DeepSeekRankArtifactMetadataChunkAck::
    DeepSeekRankArtifactMetadataChunkAck(
        DeepSeekRankArtifactMetadataChunkAckFields fields,
        Sha256Digest ack_root) noexcept
    : fields_(std::move(fields)), ack_root_(ack_root) {}

Result<DeepSeekRankArtifactMetadataChunkAck>
DeepSeekRankArtifactMetadataChunkAck::Create(
    DeepSeekRankArtifactMetadataChunkAckFields fields) {
  auto root = compile_ack_root(fields);
  if (!root.ok()) return root.status();
  return DeepSeekRankArtifactMetadataChunkAck(
      std::move(fields), *root);
}

Result<std::vector<std::byte>> encode_deepseek_rank_artifact_metadata_chunk(
    const DeepSeekRankArtifactMetadataChunk& chunk) {
  auto verified = DeepSeekRankArtifactMetadataChunk::Create(
      chunk.fields(),
      std::vector<std::byte>(chunk.payload().begin(), chunk.payload().end()));
  if (!verified.ok()) return verified.status();
  if (verified->frame_root() != chunk.frame_root()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata chunk object root differs");
  }
  std::vector<std::byte> output(
      kChunkHeaderBytes + chunk.payload().size() + kTrailingRootBytes);
  std::size_t cursor = 0;
  const auto& fields = chunk.fields();
  put_common(output, cursor, kChunkFrameType, fields);
  put(output, cursor, fields.payload_offset);
  put(output, cursor, fields.payload_bytes);
  put(output, cursor, fields.total_blob_bytes);
  put_identities_and_roots(output, cursor, fields);
  if (cursor != kChunkHeaderBytes) {
    return Status::Internal(
        "DeepSeek artifact metadata chunk header size drifted");
  }
  for (const auto byte : chunk.payload()) output[cursor++] = byte;
  put_digest(output, cursor, chunk.frame_root());
  if (cursor != output.size() ||
      output.size() > kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes) {
    return Status::Internal(
        "DeepSeek artifact metadata chunk frame size drifted");
  }
  return output;
}

Result<DeepSeekRankArtifactMetadataChunk>
decode_deepseek_rank_artifact_metadata_chunk(
    std::span<const std::byte> frame) {
  if (frame.size() < kChunkHeaderBytes + 1U + kTrailingRootBytes ||
      frame.size() > kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk frame size is invalid");
  }
  std::size_t cursor = 0;
  DeepSeekRankArtifactMetadataChunkFields fields;
  if (!get_common(frame, cursor, kChunkFrameType, &fields) ||
      !get(frame, cursor, &fields.payload_offset) ||
      !get(frame, cursor, &fields.payload_bytes) ||
      !get(frame, cursor, &fields.total_blob_bytes) ||
      !get_identities_and_roots(frame, cursor, &fields) ||
      cursor != kChunkHeaderBytes ||
      fields.payload_bytes == 0 ||
      fields.payload_bytes >
          kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes ||
      frame.size() != kChunkHeaderBytes + fields.payload_bytes +
                          kTrailingRootBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk header is invalid");
  }
  std::vector<std::byte> payload(
      frame.begin() + static_cast<std::ptrdiff_t>(cursor),
      frame.begin() + static_cast<std::ptrdiff_t>(
                          cursor + fields.payload_bytes));
  cursor += fields.payload_bytes;
  Sha256Digest encoded_root;
  if (!get_digest(frame, cursor, &encoded_root) || cursor != frame.size()) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk root is truncated");
  }
  auto chunk = DeepSeekRankArtifactMetadataChunk::Create(
      std::move(fields), std::move(payload));
  if (!chunk.ok()) return chunk.status();
  if (chunk->frame_root() != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata chunk frame root differs");
  }
  return chunk;
}

std::array<std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>
encode_deepseek_rank_artifact_metadata_chunk_ack(
    const DeepSeekRankArtifactMetadataChunkAck& ack) {
  std::array<std::byte,
             kDeepSeekRankArtifactMetadataChunkAckFrameBytes>
      output{};
  std::size_t cursor = 0;
  const auto& fields = ack.fields();
  put_common(output, cursor, kAckFrameType, fields);
  put(output, cursor, fields.cumulative_payload_bytes);
  put(output, cursor, fields.total_blob_bytes);
  put_identities_and_roots(output, cursor, fields);
  put_digest(output, cursor, ack.ack_root());
  return output;
}

Result<DeepSeekRankArtifactMetadataChunkAck>
decode_deepseek_rank_artifact_metadata_chunk_ack(
    std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankArtifactMetadataChunkAckFrameBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata ACK frame size is invalid");
  }
  std::size_t cursor = 0;
  DeepSeekRankArtifactMetadataChunkAckFields fields;
  Sha256Digest encoded_root;
  if (!get_common(frame, cursor, kAckFrameType, &fields) ||
      !get(frame, cursor, &fields.cumulative_payload_bytes) ||
      !get(frame, cursor, &fields.total_blob_bytes) ||
      !get_identities_and_roots(frame, cursor, &fields) ||
      !get_digest(frame, cursor, &encoded_root) || cursor != frame.size()) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata ACK frame is invalid");
  }
  auto ack = DeepSeekRankArtifactMetadataChunkAck::Create(
      std::move(fields));
  if (!ack.ok()) return ack.status();
  if (ack->ack_root() != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata ACK root differs");
  }
  return ack;
}

}  // namespace pih
