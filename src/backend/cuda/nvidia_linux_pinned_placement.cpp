#include "pih/backend/cuda/nvidia_linux_pinned_placement.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>

#ifdef __linux__
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace pih {
namespace {

Status platform_unavailable() {
  return Status::Unavailable(
      "verified pinned NUMA placement requires Linux");
}

#ifdef __linux__
Status errno_status(const char* operation) {
  std::string message(operation);
  message.append(" failed: ");
  message.append(std::strerror(errno));
  return Status::Unavailable(std::move(message));
}
#endif

}  // namespace

Result<NvidiaLinuxNumaPolicyGuard> NvidiaLinuxNumaPolicyGuard::Bind(
    std::int32_t numa_node) {
#ifndef __linux__
  (void)numa_node;
  return platform_unavailable();
#else
  if (numa_node < 0 || numa_node >= 1024) {
    return Status::InvalidArgument("NUMA node is outside the bounded mask");
  }
  int current_policy = -1;
  if (syscall(SYS_get_mempolicy, &current_policy, nullptr, 0, nullptr, 0) !=
      0) {
    return errno_status("get_mempolicy");
  }
  if (current_policy != MPOL_DEFAULT) {
    return Status::FailedPrecondition(
        "bootstrap thread already has a non-default NUMA policy");
  }
  unsigned long mask[1024 / (8 * sizeof(unsigned long))]{};
  const auto bits = static_cast<unsigned>(8 * sizeof(unsigned long));
  mask[static_cast<unsigned>(numa_node) / bits] |=
      1UL << (static_cast<unsigned>(numa_node) % bits);
  if (syscall(SYS_set_mempolicy, MPOL_BIND, mask, 1024) != 0) {
    return errno_status("set_mempolicy(MPOL_BIND)");
  }
  return NvidiaLinuxNumaPolicyGuard(true);
#endif
}

NvidiaLinuxNumaPolicyGuard::~NvidiaLinuxNumaPolicyGuard() {
  if (active_) (void)restore();
}

NvidiaLinuxNumaPolicyGuard::NvidiaLinuxNumaPolicyGuard(
    NvidiaLinuxNumaPolicyGuard&& other) noexcept
    : active_(std::exchange(other.active_, false)) {}

Status NvidiaLinuxNumaPolicyGuard::restore() {
  if (!active_) return Status::Ok();
#ifndef __linux__
  return platform_unavailable();
#else
  if (syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0) != 0) {
    return errno_status("set_mempolicy(MPOL_DEFAULT)");
  }
  active_ = false;
  return Status::Ok();
#endif
}

Result<std::int32_t> NvidiaLinuxPinnedPlacementVerifier::DeviceNumaNode(
    std::int32_t device_ordinal) {
#ifndef __linux__
  (void)device_ordinal;
  return platform_unavailable();
#else
  if (device_ordinal < 0) {
    return Status::InvalidArgument("CUDA device ordinal is negative");
  }
  CUdevice device{};
  CUresult result = cuDeviceGet(&device, device_ordinal);
  if (result != CUDA_SUCCESS) {
    return Status::Unavailable("cannot resolve CUDA device for NUMA lookup");
  }
  char pci_bus_id[32]{};
  result = cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device);
  if (result != CUDA_SUCCESS) {
    return Status::Unavailable("cannot query CUDA PCI bus ID");
  }
  const std::filesystem::path path =
      std::filesystem::path("/sys/bus/pci/devices") / pci_bus_id /
      "numa_node";
  std::ifstream stream(path);
  int node = -1;
  if (!(stream >> node) || node < 0) {
    return Status::Unavailable("GPU sysfs NUMA node is unavailable");
  }
  return static_cast<std::int32_t>(node);
#endif
}

Status NvidiaLinuxPinnedPlacementVerifier::verify(
    const void* data, std::uint64_t bytes, std::int32_t numa_node) {
#ifndef __linux__
  (void)data;
  (void)bytes;
  (void)numa_node;
  return platform_unavailable();
#else
  if (data == nullptr || bytes == 0 || numa_node < 0) {
    return Status::InvalidArgument("pinned placement query is invalid");
  }
  const long page_size_long = sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0) return errno_status("sysconf(_SC_PAGESIZE)");
  const auto page_size = static_cast<std::uintptr_t>(page_size_long);
  const auto begin = reinterpret_cast<std::uintptr_t>(data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return Status::InvalidArgument("pinned placement range overflows");
  }
  const auto first = begin - begin % page_size;
  const auto end = begin + static_cast<std::uintptr_t>(bytes);
  std::vector<void*> pages;
  for (auto page = first; page < end; page += page_size) {
    pages.push_back(reinterpret_cast<void*>(page));
  }
  std::vector<int> status(pages.size(), -1);
  if (syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr,
              status.data(), 0) != 0) {
    return errno_status("move_pages(query)");
  }
  for (const int observed : status) {
    if (observed != numa_node) {
      return Status::FailedPrecondition(
          "registered pinned page is outside the GPU NUMA node");
    }
  }
  return Status::Ok();
#endif
}

}  // namespace pih
