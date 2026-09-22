#include "pih/model/qwen3_int4_pinned_artifact.h"
#include <cstring>
#include "pih/model/qwen3_int4_artifact_layout.h"
namespace pih {
Result<QwenInt4PinnedArtifact> QwenInt4PinnedArtifact::LoadVerifiedBytes(
    std::span<const std::byte> bytes,RegisteredPinnedAllocator& allocator,
    PinnedPlacementVerifier& verifier,std::int32_t numa_node,
    std::uint64_t owner_id,std::uint32_t rank,
    Sha256Digest expected_artifact_digest){
 if(bytes.empty()||numa_node<0||owner_id==0||rank==UINT32_MAX||
    expected_artifact_digest==Sha256Digest{})
  return Status::InvalidArgument("Qwen INT4 artifact snapshot identity invalid");
 if(bytes.size()!=QwenInt4ArtifactLayout::kOfficialFileBytes)
  return Status::FailedPrecondition("Qwen INT4 artifact length is noncanonical");
 auto actual=sha256(bytes);if(!actual.ok())return actual.status();
 if(*actual!=expected_artifact_digest)
  return Status::FailedPrecondition("Qwen INT4 artifact digest mismatch");
 auto receipt=verify_qwen_int4_artifact(bytes);if(!receipt.ok())return receipt.status();
 auto buffer=Buffer::Allocate(allocator,bytes.size(),256);if(!buffer.ok())return buffer.status();
 if(buffer->device().type()!=DeviceType::kCpu||buffer->device().index()!=0||
    buffer->generation()==0||buffer->data()==nullptr)
  return Status::FailedPrecondition("Qwen INT4 pinned allocation is invalid");
 std::memcpy(buffer->data(),bytes.data(),bytes.size());
 const Status placed=verifier.verify(buffer->data(),buffer->size_bytes(),numa_node);
 if(!placed.ok())return placed;
 CudaCopyEndpoint endpoint{reinterpret_cast<std::uintptr_t>(buffer->data()),buffer->size_bytes(),0,
  owner_id,buffer->generation(),CudaCopyMemoryType::kRegisteredPinnedHost,rank,numa_node};
 return QwenInt4PinnedArtifact(std::move(*buffer),endpoint,*receipt);
}
}
