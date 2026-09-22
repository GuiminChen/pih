#include "pih/core/sha256.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

namespace pih {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t load_be32(const std::byte* input) noexcept {
  return (static_cast<std::uint32_t>(input[0]) << 24U) |
         (static_cast<std::uint32_t>(input[1]) << 16U) |
         (static_cast<std::uint32_t>(input[2]) << 8U) |
         static_cast<std::uint32_t>(input[3]);
}

void store_be32(std::uint32_t value, std::byte* output) noexcept {
  output[0] = static_cast<std::byte>(value >> 24U);
  output[1] = static_cast<std::byte>(value >> 16U);
  output[2] = static_cast<std::byte>(value >> 8U);
  output[3] = static_cast<std::byte>(value);
}

}  // namespace

Result<Sha256Digest> Sha256Digest::ParseHex(std::string_view value) {
  if (value.size() != 64) {
    return Status::InvalidArgument("SHA-256 text must contain exactly 64 digits");
  }
  auto nibble = [](char character) -> int {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    return -1;
  };
  Sha256Digest digest;
  for (std::size_t index = 0; index < digest.bytes.size(); ++index) {
    const int high = nibble(value[index * 2]);
    const int low = nibble(value[index * 2 + 1]);
    if (high < 0 || low < 0) {
      return Status::InvalidArgument("SHA-256 text must be lowercase hexadecimal");
    }
    digest.bytes[index] = static_cast<std::byte>((high << 4) | low);
  }
  return digest;
}

std::string Sha256Digest::hex() const {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(64, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const auto value = static_cast<std::uint8_t>(bytes[index]);
    output[index * 2] = kHex[value >> 4U];
    output[index * 2 + 1] = kHex[value & 0x0fU];
  }
  return output;
}

Sha256::Sha256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::compress(const std::byte* block) noexcept {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t index = 0; index < 16; ++index) {
    schedule[index] = load_be32(block + index * 4);
  }
  for (std::size_t index = 16; index < schedule.size(); ++index) {
    const std::uint32_t s0 = std::rotr(schedule[index - 15], 7) ^
                             std::rotr(schedule[index - 15], 18) ^
                             (schedule[index - 15] >> 3U);
    const std::uint32_t s1 = std::rotr(schedule[index - 2], 17) ^
                             std::rotr(schedule[index - 2], 19) ^
                             (schedule[index - 2] >> 10U);
    schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];
  for (std::size_t index = 0; index < schedule.size(); ++index) {
    const std::uint32_t sigma1 =
        std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    const std::uint32_t choose = (e & f) ^ (~e & g);
    const std::uint32_t temporary1 =
        h + sigma1 + choose + kRoundConstants[index] + schedule[index];
    const std::uint32_t sigma0 =
        std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sigma0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

Status Sha256::update(std::span<const std::byte> input) {
  if (finalized_) {
    return Status::FailedPrecondition("SHA-256 state is already finalized");
  }
  if (input.size() >
      (std::numeric_limits<std::uint64_t>::max() / 8U) - total_bytes_) {
    return Status::ResourceExhausted("SHA-256 bit length overflows u64");
  }
  total_bytes_ += input.size();
  std::size_t offset = 0;
  if (buffered_bytes_ != 0) {
    const std::size_t copied =
        std::min(buffer_.size() - buffered_bytes_, input.size());
    std::memcpy(buffer_.data() + buffered_bytes_, input.data(), copied);
    buffered_bytes_ += copied;
    offset += copied;
    if (buffered_bytes_ == buffer_.size()) {
      compress(buffer_.data());
      buffered_bytes_ = 0;
    }
  }
  while (input.size() - offset >= buffer_.size()) {
    compress(input.data() + offset);
    offset += buffer_.size();
  }
  if (offset != input.size()) {
    buffered_bytes_ = input.size() - offset;
    std::memcpy(buffer_.data(), input.data() + offset, buffered_bytes_);
  }
  return Status::Ok();
}

Result<Sha256Digest> Sha256::finalize() {
  if (finalized_) {
    return Status::FailedPrecondition("SHA-256 state is already finalized");
  }
  const std::uint64_t bit_length = total_bytes_ * 8U;
  buffer_[buffered_bytes_++] = std::byte{0x80};
  if (buffered_bytes_ > 56) {
    std::fill(buffer_.begin() + buffered_bytes_, buffer_.end(), std::byte{0});
    compress(buffer_.data());
    buffered_bytes_ = 0;
  }
  std::fill(buffer_.begin() + buffered_bytes_, buffer_.begin() + 56,
            std::byte{0});
  for (std::size_t index = 0; index < 8; ++index) {
    buffer_[56 + index] =
        static_cast<std::byte>(bit_length >> ((7 - index) * 8U));
  }
  compress(buffer_.data());
  Sha256Digest digest;
  for (std::size_t index = 0; index < state_.size(); ++index) {
    store_be32(state_[index], digest.bytes.data() + index * 4);
  }
  finalized_ = true;
  std::fill(buffer_.begin(), buffer_.end(), std::byte{0});
  return digest;
}

Result<Sha256Digest> sha256(std::span<const std::byte> input) {
  Sha256 state;
  const Status updated = state.update(input);
  if (!updated.ok()) return updated;
  return state.finalize();
}

}  // namespace pih
