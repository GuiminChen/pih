#include "pih/model/qwen3_int4_gemm_plan.h"

#include <array>
#include <span>
#include <string_view>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::string_view kDescriptor =
    "pih.typed_kernel_params.v1|qwen.linear.w4a16.compatibility.v1|"
    "input:device_pointer_u64@0|packed_weight:device_pointer_u64@8|"
    "scales:device_pointer_u64@16|output:device_pointer_u64@24|error_flag:"
    "device_pointer_u64@32|m:u64@40|n:u64@48|k:u64@56|total:64|"
    "activation:bf16|weight:xing-w4a16-sym-g128-v1|scales:fp16|"
    "accumulator:fp32|output:bf16|layout:canonical-nk-low-nibble-k-v1|"
    "reduction:strict-increasing-k-v1";

Result<std::pair<std::uint64_t, std::uint64_t>> dimensions(
    QwenInt4LinearShapeFamily family) {
  // uint64_t is unsigned long on LP64 Linux, not necessarily unsigned long
  // long. Construct the exact result value type instead of CTAD from ULL.
  using Dimensions = std::pair<std::uint64_t, std::uint64_t>;
  switch (family) {
    case QwenInt4LinearShapeFamily::kQProj: return Dimensions{2048, 1024};
    case QwenInt4LinearShapeFamily::kKvProj: return Dimensions{1024, 1024};
    case QwenInt4LinearShapeFamily::kOProj: return Dimensions{1024, 2048};
    case QwenInt4LinearShapeFamily::kGateUpProj: return Dimensions{3072, 1024};
    case QwenInt4LinearShapeFamily::kDownProj: return Dimensions{1024, 3072};
  }
  return Status::InvalidArgument("unknown Qwen INT4 Linear family");
}

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t i=0;i<8;++i) wire[i]=static_cast<std::byte>(value>>(i*8U));
  return digest.update(wire);
}

}  // namespace

Result<QwenInt4GemmPlan> QwenInt4GemmPlan::Create(
    QwenInt4LinearShapeFamily family, std::uint64_t tokens,
    QwenInt4KernelVariant variant) {
  if (tokens == 0 || tokens > kMaximumTokens ||
      variant != QwenInt4KernelVariant::kCompatibilityCanonical) {
    return Status::InvalidArgument("Qwen INT4 GEMM plan predicate mismatch");
  }
  auto dims = dimensions(family); if (!dims.ok()) return dims.status();
  const auto [n,k] = *dims;
  auto input_elements=checked_mul_u64(tokens,k); if(!input_elements.ok()) return input_elements.status();
  auto output_elements=checked_mul_u64(tokens,n); if(!output_elements.ok()) return output_elements.status();
  auto weight_elements=checked_mul_u64(n,k); if(!weight_elements.ok()) return weight_elements.status();
  auto scale_elements=checked_mul_u64(n,k/128); if(!scale_elements.ok()) return scale_elements.status();
  auto input_bytes=checked_mul_u64(*input_elements,2); if(!input_bytes.ok()) return input_bytes.status();
  auto output_bytes=checked_mul_u64(*output_elements,2); if(!output_bytes.ok()) return output_bytes.status();
  auto scale_bytes=checked_mul_u64(*scale_elements,2); if(!scale_bytes.ok()) return scale_bytes.status();
  const std::uint64_t blocks64=(*output_elements+kThreadsPerBlock-1)/kThreadsPerBlock;
  if (blocks64 == 0 || blocks64 > UINT32_MAX) return Status::ResourceExhausted("Qwen INT4 GEMM grid overflow");
  return QwenInt4GemmPlan(family,tokens,n,k,*input_bytes,*weight_elements/2,
                          *scale_bytes,*output_bytes,static_cast<std::uint32_t>(blocks64));
}

Result<Sha256Digest> QwenInt4GemmPlan::semantic_digest() const {
  Sha256 digest;
  Status status=digest.update(std::as_bytes(std::span(kDescriptor)));
  for(const auto value:{static_cast<std::uint64_t>(family_),rows_,output_features_,input_features_,input_bytes_,packed_bytes_,scale_bytes_,output_bytes_,static_cast<std::uint64_t>(blocks_)}) if(status.ok()) status=update_u64(digest,value);
  if(!status.ok()) return status; return digest.finalize();
}

Result<KernelSignatureManifest> QwenInt4GemmPlan::signature(
    std::string selected_cubin_sha256) const {
  auto abi=sha256(std::as_bytes(std::span(kDescriptor))); if(!abi.ok()) return abi.status();
  const std::array<KernelParameterSpec,8> parameters{{
      {"input",KernelWireType::kDevicePointerU64,0,"qwen_w4a16_input_bf16_read_v1"},
      {"packed_weight",KernelWireType::kDevicePointerU64,8,"qwen_w4a16_packed_read_v1"},
      {"scales",KernelWireType::kDevicePointerU64,16,"qwen_w4a16_scales_fp16_read_v1"},
      {"output",KernelWireType::kDevicePointerU64,24,"qwen_w4a16_output_bf16_write_v1"},
      {"error_flag",KernelWireType::kDevicePointerU64,32,"qwen_w4a16_error_atomic_v1"},
      {"m",KernelWireType::kU64,40,"qwen_w4a16_m_exact_v1"},
      {"n",KernelWireType::kU64,48,"qwen_w4a16_n_official_v1"},
      {"k",KernelWireType::kU64,56,"qwen_w4a16_k_g128_v1"}}};
  return KernelSignatureManifest::Create(std::string(logical_id()),
      std::move(selected_cubin_sha256),abi->hex(),parameters,64);
}

Result<std::vector<KernelPointerContract>> QwenInt4GemmPlan::pointer_contracts(
    std::int32_t rank, std::int32_t device_index) const {
  struct Item{std::string_view id;KernelPointerAccess access;KernelPointerOwnerClass owner;std::uint64_t bytes;std::uint64_t alignment;};
  const std::array items{
      Item{"qwen_w4a16_input_bf16_read_v1",KernelPointerAccess::kRead,KernelPointerOwnerClass::kActivation,input_bytes_,256},
      Item{"qwen_w4a16_packed_read_v1",KernelPointerAccess::kRead,KernelPointerOwnerClass::kWeight,packed_bytes_,256},
      Item{"qwen_w4a16_scales_fp16_read_v1",KernelPointerAccess::kRead,KernelPointerOwnerClass::kWeight,scale_bytes_,256},
      Item{"qwen_w4a16_output_bf16_write_v1",KernelPointerAccess::kWrite,KernelPointerOwnerClass::kActivation,output_bytes_,256},
      Item{"qwen_w4a16_error_atomic_v1",KernelPointerAccess::kAtomic,KernelPointerOwnerClass::kWorkspace,4,4}};
  std::vector<KernelPointerContract> result; result.reserve(items.size());
  for(const auto& item:items){auto contract=KernelPointerContract::Create(std::string(item.id),item.access,item.owner,rank,device_index,item.bytes,item.alignment,0,false);if(!contract.ok())return contract.status();result.push_back(std::move(*contract));}
  return result;
}

}  // namespace pih
