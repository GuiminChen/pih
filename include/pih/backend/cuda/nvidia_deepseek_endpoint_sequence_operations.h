#pragma once

#include "pih/model/deepseek_endpoint_sequence_executor.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaDeepSeekEndpointSequenceOperations final
    : public DeepSeekEndpointSequenceOperations {
 public:
  static Result<NvidiaDeepSeekEndpointSequenceOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels);
  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status embedding(DeepSeekEmbeddingLaunch launch) override;
  Status hc_head(DeepSeekHcHeadLaunch launch) override;
  Status rms_norm(DeepSeekRmsNormLaunch launch) override;
  Status lm_head(DeepSeekLmHeadLaunch launch) override;
  Status sample(DeepSeekArgmaxLaunch launch) override;
  Status stochastic_sample(DeepSeekStochasticSampleLaunch launch) override;
  Status copy_token_d2h_async(std::uint32_t* host, std::uintptr_t device,
                              std::uintptr_t stream) override;
  Status copy_error_d2h_async(std::uint32_t* host,
                              std::uintptr_t device,
                              std::uintptr_t stream) override;
  Status copy_logprob_d2h_async(float* host, std::uintptr_t device,
                                std::uintptr_t stream) override;
  Status copy_rng_d2h_async(std::uint32_t* host, std::uintptr_t device,
                            std::uintptr_t stream) override;
  Status copy_top_ids_d2h_async(std::uint32_t* host, std::uintptr_t device,
                                std::uint32_t count,
                                std::uintptr_t stream) override;
  Status copy_top_logprobs_d2h_async(float* host, std::uintptr_t device,
                                     std::uint32_t count,
                                     std::uintptr_t stream) override;

 private:
  explicit NvidiaDeepSeekEndpointSequenceOperations(
      std::uint64_t context_identity,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels) noexcept
      : context_identity_(context_identity), async_api_(async_api),
        kernels_(kernels) {}
  Status require_context() const;
  Status copy_d2h_async(void* host, std::uintptr_t device,
                        std::size_t bytes, std::uintptr_t stream,
                        const char* operation);
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_ = nullptr;
};

}  // namespace pih
