#include "pih/model/deepseek_nccl_bootstrap_lease.h"

#include <array>
#include <cstring>
#include <string_view>
#include <utility>

namespace pih {
namespace {

void put64(std::byte* output, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) {
    output[i] = static_cast<std::byte>(value >> (8U * i));
  }
}

void put32(std::byte* output, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) {
    output[i] = static_cast<std::byte>(value >> (8U * i));
  }
}

void secure_zero(Allocation allocation) noexcept {
  volatile std::byte* bytes = static_cast<volatile std::byte*>(allocation.data);
  for (std::size_t i = 0; i < allocation.bytes; ++i) bytes[i] = std::byte{0};
}

}  // namespace

Result<DeepSeekNcclBootstrapLease> DeepSeekNcclBootstrapLease::Create(
    RegisteredPinnedAllocator& allocator, std::span<const std::byte> raw_id,
    std::uint64_t engine_epoch, std::uint32_t edge_id,
    std::uint64_t lease_id) {
  if (raw_id.size() != kUniqueIdBytes || engine_epoch == 0 || lease_id == 0) {
    return Status::InvalidArgument("DeepSeek NCCL bootstrap identity is invalid");
  }
  auto allocation = allocator.allocate(kUniqueIdBytes, 64);
  if (!allocation.ok()) return allocation.status();
  if (allocation->data == nullptr || allocation->bytes != kUniqueIdBytes ||
      allocation->alignment < 64 || allocation->device != Device::Cpu()) {
    allocator.deallocate(*allocation);
    return Status::Internal("DeepSeek NCCL pinned lease allocation is invalid");
  }
  std::memcpy(allocation->data, raw_id.data(), raw_id.size());
  static constexpr char kDomain[] = "pih-nccl-unique-id-v1";
  std::array<std::byte, 20> identity{};
  put64(identity.data(), engine_epoch);
  put32(identity.data() + 8, edge_id);
  put64(identity.data() + 12, lease_id);
  Sha256 hasher;
  Status status = hasher.update(std::as_bytes(std::span(kDomain)));
  if (status.ok()) status = hasher.update(identity);
  if (status.ok()) status = hasher.update(raw_id);
  if (!status.ok()) {
    secure_zero(*allocation);
    allocator.deallocate(*allocation);
    return status;
  }
  auto commitment = hasher.finalize();
  if (!commitment.ok()) {
    secure_zero(*allocation);
    allocator.deallocate(*allocation);
    return commitment.status();
  }
  return DeepSeekNcclBootstrapLease(allocator, *allocation, engine_epoch,
                                    edge_id, lease_id, *commitment);
}

DeepSeekNcclBootstrapLease::DeepSeekNcclBootstrapLease(
    RegisteredPinnedAllocator& allocator, Allocation allocation,
    std::uint64_t engine_epoch, std::uint32_t edge_id, std::uint64_t lease_id,
    Sha256Digest commitment) noexcept
    : allocator_(&allocator), allocation_(allocation), engine_epoch_(engine_epoch),
      edge_id_(edge_id), lease_id_(lease_id), commitment_(commitment) {}

DeepSeekNcclBootstrapLease::DeepSeekNcclBootstrapLease(
    DeepSeekNcclBootstrapLease&& other) noexcept {
  *this = std::move(other);
}

DeepSeekNcclBootstrapLease& DeepSeekNcclBootstrapLease::operator=(
    DeepSeekNcclBootstrapLease&& other) noexcept {
  if (this != &other) {
    release();
    allocator_ = std::exchange(other.allocator_, nullptr);
    allocation_ = std::exchange(other.allocation_, {});
    engine_epoch_ = other.engine_epoch_;
    edge_id_ = other.edge_id_;
    lease_id_ = other.lease_id_;
    commitment_ = other.commitment_;
    zeroized_ = other.zeroized_;
  }
  return *this;
}

DeepSeekNcclBootstrapLease::~DeepSeekNcclBootstrapLease() { release(); }

Result<std::span<const std::byte>> DeepSeekNcclBootstrapLease::borrow(
    std::uint64_t engine_epoch, std::uint32_t edge_id,
    std::uint64_t lease_id) const {
  if (allocator_ == nullptr || zeroized_ || engine_epoch != engine_epoch_ ||
      edge_id != edge_id_ || lease_id != lease_id_) {
    return Status::FailedPrecondition("DeepSeek NCCL bootstrap lease is unavailable");
  }
  return std::span<const std::byte>(
      static_cast<const std::byte*>(allocation_.data), allocation_.bytes);
}

Status DeepSeekNcclBootstrapLease::zeroize() noexcept {
  if (zeroized_) return Status::Ok();
  if (allocator_ == nullptr || allocation_.data == nullptr) {
    return Status::FailedPrecondition("DeepSeek NCCL bootstrap lease has no storage");
  }
  secure_zero(allocation_);
  zeroized_ = true;
  return Status::Ok();
}

void DeepSeekNcclBootstrapLease::release() noexcept {
  if (allocator_ == nullptr) return;
  (void)zeroize();
  allocator_->deallocate(allocation_);
  allocator_ = nullptr;
  allocation_ = {};
}

}  // namespace pih
