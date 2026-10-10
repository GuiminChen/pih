#include "pih/model/deepseek_endpoint_device_resources.h"
#include <cstdlib>
#if defined(_WIN32)
#include <malloc.h>
#endif
static void Require(bool condition) { if (!condition) std::abort(); }
#include <set>
class DeviceAllocator final : public pih::Allocator {
 public:
  uint64_t allocated=0, generation=0;
  pih::Result<pih::Allocation> allocate(uint64_t bytes,uint64_t alignment) override {
    void* data=nullptr;
#if defined(_WIN32)
    data=_aligned_malloc(static_cast<size_t>(bytes),static_cast<size_t>(alignment));
#else
    const auto host_alignment=static_cast<size_t>(alignment < alignof(void*) ? alignof(void*) : alignment);
    if(posix_memalign(&data,host_alignment,static_cast<size_t>(bytes))) data=nullptr;
#endif
    if(!data) return pih::Status::ResourceExhausted("simulation allocation failed");
    allocated+=bytes;
    return pih::Allocation{data,bytes,alignment,++generation,pih::Device::Create(pih::DeviceType::kCuda,0).value()};
  }
  void deallocate(pih::Allocation value) noexcept override {
    allocated-=value.bytes;
#if defined(_WIN32)
    _aligned_free(value.data);
#else
    std::free(value.data);
#endif
  }
};
int main() {
  for(uint32_t count : {1U,2U,4U}) {
    DeviceAllocator allocator;
    {
      pih::DeepSeekStagePlan stage{};
      stage.layers={0,42}; stage.owns_embedding=true; stage.owns_lm_head=true;
      auto resources=pih::DeepSeekEndpointDeviceResources::Allocate(stage,count,allocator,1,0,count).value();
      Require(resources.backing_bytes()==allocator.allocated);
      std::set<uintptr_t> errors,tokens,logits;
      const auto first=resources.view();
      for(uint32_t index=0;index<count;++index) {
        auto lane=resources.view(index);
        Require(lane.embedding_output_hc_bf16==first.embedding_output_hc_bf16+index*4ULL*4096*2);
        Require(errors.insert(lane.error_flag_u32).second);
        Require(tokens.insert(lane.sampled_token_u32).second);
        Require(logits.insert(lane.logits_f32).second);
      }
    }
    Require(allocator.allocated==0);
  }
}

