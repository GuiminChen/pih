#include "pih/model/deepseek_dspark_three_stage_reference.h"

#include <cmath>

#include "pih/model/deepseek_endpoint_oracle.h"

namespace pih {
namespace {

template <typename Function>
std::vector<BFloat16> bf16_matrix(std::uint32_t rows,
                                 std::uint32_t columns,
                                 Function value) {
  std::vector<BFloat16> result(
      static_cast<std::size_t>(rows) * columns);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < columns; ++column) {
      result[static_cast<std::size_t>(row) * columns + column] =
          BFloat16::FromFloat(value(row, column));
    }
  }
  return result;
}

Result<std::uint32_t> small_argmax(std::span<const float> logits) {
  if (logits.empty()) {
    return Status::InvalidArgument(
        "DeepSeek reference argmax has no vocabulary");
  }
  std::uint32_t best = 0;
  for (std::uint32_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      return Status::InvalidArgument(
          "DeepSeek reference argmax received non-finite logits");
    }
    if (logits[index] > logits[best]) best = index;
  }
  return best;
}

}  // namespace

Result<DeepSeekDsparkThreeStageReferenceReceipt>
build_deepseek_dspark_three_stage_reference() {
  using Receipt = DeepSeekDsparkThreeStageReferenceReceipt;
  constexpr auto kRows = Receipt::kBlockSize;
  constexpr auto kHidden = Receipt::kHiddenSize;
  constexpr auto kHc = Receipt::kHcStreams;
  constexpr auto kVocab = Receipt::kVocabularySize;
  constexpr auto kRank = Receipt::kMarkovRank;

  const auto embedding = bf16_matrix(
      kVocab, kHidden, [](std::uint32_t row, std::uint32_t column) {
        return 0.1F * static_cast<float>(row + 1) +
               0.01F * static_cast<float>(column + 1);
      });
  std::array<std::uint32_t, kRows> initialized_ids{};
  std::vector<BFloat16> hidden(
      static_cast<std::size_t>(kRows) * kHc * kHidden);
  const std::array<std::uint32_t, 1> input_token{2};
  auto status = deepseek_dspark_draft_init_oracle(
      input_token, embedding, 6, kRows, kVocab, kHidden,
      initialized_ids, hidden);
  if (!status.ok()) return status;

  DeepSeekDsparkThreeStageReferenceReceipt receipt;
  for (std::uint32_t stage_index = 0;
       stage_index < kDeepSeekDsparkStageCount; ++stage_index) {
    std::vector<BFloat16> output(hidden.size());
    for (std::size_t index = 0; index < hidden.size(); ++index) {
      const auto column = static_cast<std::uint32_t>(index % kHidden);
      const auto transformed = hidden[index].to_float() +
          0.25F * static_cast<float>(stage_index + 1) +
          0.03125F * static_cast<float>(column + 1);
      output[index] = BFloat16::FromFloat(transformed);
    }
    receipt.stage_hidden_hc[stage_index] = output;
    hidden = std::move(output);
  }

  std::vector<float> hc_fn(static_cast<std::size_t>(kHc) * kHc * kHidden);
  for (std::uint32_t stream = 0; stream < kHc; ++stream) {
    for (std::uint32_t column = 0; column < kHc * kHidden; ++column) {
      const auto sign = (stream + column) % 2 == 0 ? 1.0F : -1.0F;
      hc_fn[static_cast<std::size_t>(stream) * kHc * kHidden + column] =
          sign * 0.005F * static_cast<float>(stream + 1);
    }
  }
  const std::array<float, 1> hc_scale{0.5F};
  const std::array<float, kHc> hc_base{-0.1F, 0.0F, 0.1F, 0.2F};
  receipt.head_hidden.resize(static_cast<std::size_t>(kRows) * kHidden);
  status = deepseek_hc_head_oracle(
      hidden, hc_fn, hc_scale, hc_base, kRows, kHidden,
      1.0e-6F, 1.0e-6F, receipt.head_hidden);
  if (!status.ok()) return status;

  const auto lm_weight = bf16_matrix(
      kVocab, kHidden, [](std::uint32_t row, std::uint32_t column) {
        const auto sign = column % 2 == 0 ? 1.0F : -1.0F;
        return 0.02F * static_cast<float>(row + 1) +
               sign * 0.015F * static_cast<float>(column + 1);
      });
  receipt.raw_logits.resize(static_cast<std::size_t>(kRows) * kVocab);
  status = deepseek_lm_head_oracle(
      receipt.head_hidden, lm_weight, kRows, kVocab, kHidden,
      receipt.raw_logits);
  if (!status.ok()) return status;

  const auto markov_embedding = bf16_matrix(
      kVocab, kRank, [](std::uint32_t row, std::uint32_t column) {
        return 0.04F * static_cast<float>(row + 1) -
               0.01F * static_cast<float>(column);
      });
  const auto markov_head = bf16_matrix(
      kVocab, kRank, [](std::uint32_t row, std::uint32_t column) {
        return 0.025F * static_cast<float>(row + 1) +
               0.02F * static_cast<float>(column + 1);
      });
  std::vector<BFloat16> markov_embeddings(
      static_cast<std::size_t>(kRows) * kRank);
  receipt.markov_bias.resize(static_cast<std::size_t>(kRows) * kVocab);
  receipt.biased_logits.resize(static_cast<std::size_t>(kRows) * kVocab);
  receipt.causal_token_chain[0] = initialized_ids[0];
  for (std::uint32_t row = 0; row < kRows; ++row) {
    const std::array<std::uint32_t, 1> current{
        receipt.causal_token_chain[row]};
    auto raw = std::span<const float>(receipt.raw_logits)
                   .subspan(static_cast<std::size_t>(row) * kVocab, kVocab);
    auto markov = std::span<BFloat16>(markov_embeddings)
                      .subspan(static_cast<std::size_t>(row) * kRank, kRank);
    auto biased = std::span<float>(receipt.biased_logits)
                      .subspan(static_cast<std::size_t>(row) * kVocab, kVocab);
    status = deepseek_dspark_markov_oracle(
        current, markov_embedding, markov_head, raw, kVocab, kRank,
        markov, biased);
    if (!status.ok()) return status;
    for (std::uint32_t word = 0; word < kVocab; ++word) {
      receipt.markov_bias[static_cast<std::size_t>(row) * kVocab + word] =
          biased[word] - raw[word];
    }
    auto best = small_argmax(biased);
    if (!best.ok()) return best.status();
    receipt.proposal_token_ids[row] = *best;
    receipt.causal_token_chain[row + 1] = *best;
  }

  std::vector<float> confidence_weight(kHidden + kRank);
  for (std::uint32_t index = 0; index < confidence_weight.size(); ++index) {
    confidence_weight[index] =
        (index % 2 == 0 ? 1.0F : -1.0F) *
        0.03F * static_cast<float>(index + 1);
  }
  receipt.confidence.resize(kRows);
  status = deepseek_dspark_confidence_oracle(
      receipt.head_hidden, markov_embeddings, confidence_weight,
      kRows, kHidden, kRank, receipt.confidence);
  if (!status.ok()) return status;
  return receipt;
}

}  // namespace pih
