#pragma once
#include "inference_request.h"
#include <algorithm>
#include <limits>

namespace pih::deepseek_v41 {
// A rank-independent dispatch descriptor. Token rows are packed once, and
// sequence offsets delimit attention/Engram history: expert token batches must
// never be used as a substitute for these request boundaries.
struct PackedRequestMember final {
  InferenceRequest request;
  std::uint32_t cache_slot = 0;
};
struct PackedRequestSpan final {
  SamplingIdentity identity;
  std::uint32_t cache_slot{}, token_offset{}, token_count{}, position{};
  bool finish{};
};
class PackedRequest final {
 public:
  static Result<PackedRequest> Create(std::span<const PackedRequestMember> members,
      std::uint32_t maximum_sequences, std::uint32_t maximum_tokens) {
    if (!maximum_sequences || !maximum_tokens || members.empty() || members.size() > maximum_sequences)
      return Status::InvalidArgument("V4.1 packed sequence bound invalid");
    std::vector<const PackedRequestMember*> ordered;
    ordered.reserve(members.size());
    for (const auto& member : members) {
      const auto valid = member.request.Validate();
      if (!valid.ok()) return valid;
      if (member.cache_slot >= maximum_sequences)
        return Status::InvalidArgument("V4.1 packed cache slot outside reservation");
      ordered.push_back(&member);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) {
      return a->request.sampling.identity.sequence_generation < b->request.sampling.identity.sequence_generation;
    });
    PackedRequest result;
    const auto first = ordered.front()->request.sampling.identity;
    for (const auto* member : ordered) {
      const auto& request = member->request;
      const auto id = request.sampling.identity;
      if (id.epoch != first.epoch || id.plan_seq != first.plan_seq)
        return Status::InvalidArgument("V4.1 packed rank epoch or plan differs");
      for (const auto& span : result.spans_) {
        if (span.identity.sequence_generation == id.sequence_generation || span.cache_slot == member->cache_slot)
          return Status::InvalidArgument("V4.1 packed sequence identity or cache slot aliases");
      }
      if (request.count > maximum_tokens - result.tokens_.size())
        return Status::ResourceExhausted("V4.1 packed token budget exhausted");
      result.spans_.push_back({id, member->cache_slot, static_cast<std::uint32_t>(result.tokens_.size()),
          request.count, request.sampling.processed_length - request.count, request.finish});
      result.parameters_.push_back(request.sampling.parameters);
      result.tokens_.insert(result.tokens_.end(), request.tokens.begin(), request.tokens.begin() + request.count);
    }
    return result;
  }
  std::span<const PackedRequestSpan> spans() const noexcept { return spans_; }
  std::span<const std::uint32_t> tokens() const noexcept { return tokens_; }
  std::span<const SamplingParameters> parameters() const noexcept { return parameters_; }
  Result<std::vector<std::uint32_t>> LastTokenRows() const {
    std::vector<std::uint32_t> rows; rows.reserve(spans_.size());
    for (const auto& span : spans_) {
      if (!span.token_count || span.finish)
        return Status::InvalidArgument("Terminal packed members have no projection row");
      rows.push_back(span.token_offset + span.token_count - 1);
    }
    return rows;
  }
  // Explicit packed protocol v1; never reinterpret an ordinary PHIFREQ1 frame.
  Result<std::vector<std::uint8_t>> Encode() const {
    constexpr std::size_t stride = 8 + InferenceRequest::kWireBytes;
    if (spans_.size() > 32) return Status::ResourceExhausted("Packed request protocol sequence limit exceeded");
    std::vector<std::uint8_t> bytes(32 + spans_.size() * stride);
    const auto put = [&](std::size_t at, std::uint64_t value, unsigned count) {
      for (unsigned i = 0; i < count; ++i) bytes[at + i] = static_cast<std::uint8_t>(value >> (i * 8));
    };
    put(0, 0x31504b4341504850ULL, 8); put(8, 1, 4); put(12, bytes.size(), 4);
    put(16, spans_.size(), 4); put(20, tokens_.size(), 4);
    for (unsigned i = 0; i < spans_.size(); ++i) {
      const auto& span = spans_[i]; InferenceRequest request;
      request.sampling.identity = span.identity; request.sampling.parameters = parameters_[i];
      request.sampling.processed_length = span.position + span.token_count;
      request.count = span.token_count; request.finish = span.finish;
      std::copy_n(tokens_.begin() + span.token_offset, span.token_count, request.tokens.begin());
      auto encoded = request.Encode(); if (!encoded.ok()) return encoded.status();
      put(32 + i * stride, span.cache_slot, 4);
      std::copy(encoded->begin(), encoded->end(), bytes.begin() + 32 + i * stride + 8);
    }
    return bytes;
  }
  static Result<PackedRequest> Decode(std::span<const std::uint8_t> bytes,
      std::uint32_t maximum_sequences, std::uint32_t maximum_tokens) {
    if (bytes.size() < 32) return Status::InvalidArgument("Packed request frame truncated");
    const auto get = [&](std::size_t at, unsigned count) {
      std::uint64_t value = 0;
      for (unsigned i = 0; i < count; ++i) value |= std::uint64_t(bytes[at + i]) << (i * 8);
      return value;
    };
    const auto count = get(16, 4); constexpr std::size_t stride = 8 + InferenceRequest::kWireBytes;
    if (get(0, 8) != 0x31504b4341504850ULL || get(8, 4) != 1 || get(12, 4) != bytes.size() ||
        !count || count > 32 || count > maximum_sequences || bytes.size() != 32 + count * stride ||
        get(20, 4) > maximum_tokens || get(24, 8))
      return Status::InvalidArgument("Packed request frame version, bounds or reserved bytes invalid");
    std::vector<PackedRequestMember> members; members.reserve(count);
    std::uint64_t previous = 0;
    for (unsigned i = 0; i < count; ++i) {
      const auto at = 32 + i * stride;
      if (get(at + 4, 4)) return Status::InvalidArgument("Packed member reserved bytes nonzero");
      auto request = InferenceRequest::Decode(bytes.subspan(at + 8, InferenceRequest::kWireBytes));
      if (!request.ok()) return request.status();
      if (request->sampling.identity.sequence_generation <= previous)
        return Status::InvalidArgument("Packed member order differs from canonical rank ordering");
      previous = request->sampling.identity.sequence_generation;
      members.push_back({std::move(*request), static_cast<std::uint32_t>(get(at, 4))});
    }
    auto result = Create(members, maximum_sequences, maximum_tokens);
    if (!result.ok()) return result.status();
    if (result->tokens().size() != get(20, 4)) return Status::InvalidArgument("Packed token count differs from header");
    return result;
  }
 private:
  std::vector<PackedRequestSpan> spans_;
  std::vector<std::uint32_t> tokens_;
  std::vector<SamplingParameters> parameters_;
};
}
