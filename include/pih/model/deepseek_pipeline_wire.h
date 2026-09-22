#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/deepseek_pipeline_transaction.h"

namespace pih {

struct DeepSeekSequenceId final {
  std::array<std::uint64_t, 2> words{};
  bool operator==(const DeepSeekSequenceId&) const = default;
};

struct DeepSeekPipelineSequenceRecord final {
  DeepSeekSequenceId sequence_id;
  std::uint64_t sequence_generation = 0;
  std::uint64_t state_generation = 0;
  std::uint32_t token_offset = 0;
  std::uint32_t token_count = 0;
  std::uint32_t committed_length = 0;
  std::uint32_t reserved_length = 0;
  std::uint32_t absolute_position_base = 0;
  std::uint32_t flags = 0;
  std::uint32_t sampling_config_id = 0;
  bool operator==(const DeepSeekPipelineSequenceRecord&) const = default;
};

struct DeepSeekPipelineWireMessage final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t plan_sequence = 0;
  std::uint64_t microbatch_id = 0;
  DeepSeekPlanPhase phase = DeepSeekPlanPhase::kPrefill;
  std::vector<DeepSeekPipelineSequenceRecord> records;
  std::vector<std::uint32_t> token_ids;
};

class DeepSeekPipelineWireCodec final {
 public:
  static constexpr std::uint16_t kSchemaVersion = 1;
  static constexpr std::size_t kHeaderBytes = 80;
  static constexpr std::size_t kRecordBytes = 64;
  static constexpr std::size_t kTrailerBytes = 16;

  static Result<std::uint64_t> PaddedBytes(std::uint32_t token_count,
                                           std::uint32_t sequence_count);
  static Result<std::vector<std::byte>> Encode(
      const DeepSeekPipelineWireMessage& message);
  static Result<DeepSeekPipelineWireMessage> Decode(
      std::span<const std::byte> bytes, std::uint32_t max_tokens,
      std::uint32_t max_sequences);
};

}  // namespace pih
