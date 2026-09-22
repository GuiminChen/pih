#include "pih/backend/cuda/deepseek_component_sha256.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "pih/backend/cuda/cuda_status.h"

namespace pih { namespace {

__device__ __forceinline__ std::uint32_t rotate_right(std::uint32_t value,
                                                       std::uint32_t shift) {
  return __funnelshift_r(value, value, shift);
}

__device__ void compress_sha256(const std::uint8_t* block,
                                std::uint32_t* state) {
  constexpr std::uint32_t constants[64] = {
      0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
      0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
      0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
      0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
      0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
      0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
      0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
      0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
  std::uint32_t words[64];
  for (std::uint32_t i = 0; i < 16; ++i) {
    const auto offset = i * 4;
    words[i] = static_cast<std::uint32_t>(block[offset]) << 24U |
               static_cast<std::uint32_t>(block[offset + 1]) << 16U |
               static_cast<std::uint32_t>(block[offset + 2]) << 8U |
               static_cast<std::uint32_t>(block[offset + 3]);
  }
  for (std::uint32_t i = 16; i < 64; ++i) {
    const auto s0 = rotate_right(words[i - 15], 7) ^
                    rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3U);
    const auto s1 = rotate_right(words[i - 2], 17) ^
                    rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10U);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }
  auto a=state[0],b=state[1],c=state[2],d=state[3];
  auto e=state[4],f=state[5],g=state[6],h=state[7];
  for (std::uint32_t i = 0; i < 64; ++i) {
    const auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    const auto choice = (e & f) ^ (~e & g);
    const auto temp1 = h + s1 + choice + constants[i] + words[i];
    const auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    const auto majority = (a & b) ^ (a & c) ^ (b & c);
    const auto temp2 = s0 + majority;
    h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
  }
  state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
  state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__global__ void component_sha256_kernel(const std::uint8_t* input,
                                         std::uint64_t bytes,
                                         std::uint8_t* output,
                                         std::uint32_t* error_flag) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint32_t state[8] = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                            0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  const auto full_blocks = bytes / 64U;
  for (std::uint64_t block = 0; block < full_blocks; ++block)
    compress_sha256(input + block * 64U, state);
  std::uint8_t tail[128]{};
  const auto remainder = static_cast<std::uint32_t>(bytes % 64U);
  for (std::uint32_t i = 0; i < remainder; ++i)
    tail[i] = input[full_blocks * 64U + i];
  tail[remainder] = 0x80U;
  const auto padded_bytes = remainder < 56U ? 64U : 128U;
  const auto bit_length = bytes * 8U;
  for (std::uint32_t i = 0; i < 8; ++i)
    tail[padded_bytes - 1U - i] =
        static_cast<std::uint8_t>(bit_length >> (i * 8U));
  compress_sha256(tail, state);
  if (padded_bytes == 128U) compress_sha256(tail + 64U, state);
  for (std::uint32_t word = 0; word < 8; ++word)
    for (std::uint32_t byte = 0; byte < 4; ++byte)
      output[word * 4U + byte] =
          static_cast<std::uint8_t>(state[word] >> (24U - byte * 8U));
  if (bytes == 0) atomicOr(error_flag, 1U);
}

}  // namespace

Status launch_deepseek_component_sha256(
    DeepSeekComponentSha256Launch launch) {
  if (launch.input == 0 || launch.bytes == 0 ||
      launch.bytes > std::numeric_limits<std::uint64_t>::max() / 8U ||
      launch.output_digest == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0) {
    return Status::InvalidArgument("DeepSeek component SHA-256 launch is invalid");
  }
  auto status = cuda_status(cudaPeekAtLastError(),
                            "cudaPeekAtLastError before component SHA-256");
  if (!status.ok()) return status;
  component_sha256_kernel<<<1, 1, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const std::uint8_t*>(launch.input), launch.bytes,
      reinterpret_cast<std::uint8_t*>(launch.output_digest),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32));
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek component SHA-256 launch");
}

}  // namespace pih
