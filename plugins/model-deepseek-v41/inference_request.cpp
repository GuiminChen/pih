#include "inference_request.h"
#include <bit>

namespace pih::deepseek_v41 {
Status InferenceRequest::Validate() const {
  const auto& id = sampling.identity;
  if (!id.epoch || !id.plan_seq || !id.sequence_generation || !id.sampling_config_id ||
      !sampling.processed_length || (finish ? count != 0 : !count) || count > tokens.size() || sampling.processed_length < count ||
      sampling.processed_length > FlashConfig::kMaximumPositions ||
      (!finish && sampling.processed_length != count && count != 1))
    return Status::InvalidArgument("Inference request identity or prefill/decode shape invalid");
  const auto parameters = ValidateSamplingParameters(sampling.parameters); if (!parameters.ok()) return parameters;
  for (unsigned i = 0; i < tokens.size(); ++i)
    if ((i < count && tokens[i] >= FlashConfig::kVocabularySize) || (i >= count && tokens[i]))
      return Status::InvalidArgument("Inference request token or unused slot invalid");
  for (unsigned i = sampling.parameters.suppressed_count; i < 17; ++i)
    if (sampling.parameters.suppressed[i]) return Status::InvalidArgument("Inference request unused suppression is nonzero");
  return Status::Ok();
}
Result<std::array<std::uint8_t, InferenceRequest::kWireBytes>> InferenceRequest::Encode() const {
  const auto status = Validate(); if (!status.ok()) return status;
  std::array<std::uint8_t, kWireBytes> out{};
  const auto put = [&](unsigned at, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) out[at + i] = static_cast<std::uint8_t>(value >> (i * 8));
  };
  put(0, 0x3151455246494850ULL, 8); // "PHIFREQ1"
  put(8, 2, 4); put(12, kWireBytes, 4); put(164, finish ? 1 : 0, 4);
  const auto& id = sampling.identity; const auto& p = sampling.parameters;
  put(16, id.epoch, 8); put(24, id.plan_seq, 8); put(32, id.sequence_generation, 8); put(40, id.sampling_config_id, 8);
  put(48, sampling.processed_length, 4); put(52, count, 4);
  put(56, std::bit_cast<std::uint32_t>(p.temperature), 4); put(60, std::bit_cast<std::uint32_t>(p.top_p), 4);
  put(64, p.top_k, 4); put(68, p.logprobs ? 1 : 0, 4); put(72, p.seed, 8); put(80, p.ordinal, 8);
  put(88, p.top_count, 4); put(92, p.suppressed_count, 4);
  for (unsigned i = 0; i < 17; ++i) put(96 + i * 4, p.suppressed[i], 4);
  for (unsigned i = 0; i < count; ++i) put(256 + i * 4, tokens[i], 4);
  return out;
}
Result<InferenceRequest> InferenceRequest::Decode(std::span<const std::uint8_t> frame) {
  if (frame.size() != kWireBytes) return Status::InvalidArgument("Inference request frame length invalid");
  const auto get = [&](unsigned at, unsigned size) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < size; ++i) value |= std::uint64_t(frame[at + i]) << (8 * i);
    return value;
  };
  if (get(0, 8) != 0x3151455246494850ULL || get(8, 4) != 2 || get(12, 4) != kWireBytes || get(68, 4) > 1 || get(164, 4) > 1)
    return Status::InvalidArgument("Inference request magic, version or flags invalid");
  for (unsigned i = 168; i < 256; ++i)
    if (frame[i]) return Status::InvalidArgument("Inference request reserved byte is nonzero");
  InferenceRequest request;
  request.finish = get(164, 4) != 0;
  request.sampling.identity = {get(16, 8), get(24, 8), get(32, 8), get(40, 8)};
  request.sampling.processed_length = static_cast<std::uint32_t>(get(48, 4)); request.count = static_cast<std::uint32_t>(get(52, 4));
  auto& p = request.sampling.parameters;
  p.temperature = std::bit_cast<float>(static_cast<std::uint32_t>(get(56, 4)));
  p.top_p = std::bit_cast<float>(static_cast<std::uint32_t>(get(60, 4)));
  p.top_k = static_cast<std::uint32_t>(get(64, 4)); p.logprobs = get(68, 4) != 0; p.seed = get(72, 8); p.ordinal = get(80, 8);
  p.top_count = static_cast<std::uint32_t>(get(88, 4)); p.suppressed_count = static_cast<std::uint32_t>(get(92, 4));
  for (unsigned i = 0; i < 17; ++i) p.suppressed[i] = static_cast<std::uint32_t>(get(96 + i * 4, 4));
  for (unsigned i = 0; i < 4096; ++i) request.tokens[i] = static_cast<std::uint32_t>(get(256 + i * 4, 4));
  const auto status = request.Validate(); if (!status.ok()) return status;
  return request;
}
}  // namespace pih::deepseek_v41
