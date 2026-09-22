#include "pih/io/mapped_file.h"

#include <limits>
#include <string>
#include <utility>

#include "pih/core/checked_math.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pih {

struct MappedFile::Impl final {
#ifdef _WIN32
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
#else
  int file = -1;
#endif
  void* view = nullptr;
  std::size_t view_bytes = 0;

  ~Impl() {
#ifdef _WIN32
    if (view != nullptr) UnmapViewOfFile(view);
    if (mapping != nullptr) CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
    if (view != nullptr) munmap(view, view_bytes);
    if (file >= 0) close(file);
#endif
  }
};

Result<MappedFile> MappedFile::OpenReadOnly(const std::filesystem::path& path,
                                            std::uint64_t maximum_bytes) {
  auto impl = std::make_shared<Impl>();
  std::uint64_t file_bytes = 0;
#ifdef _WIN32
  impl->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl->file == INVALID_HANDLE_VALUE) {
    return Status::Unavailable("failed to open mapped file");
  }
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(impl->file, &size) || size.QuadPart < 0) {
    return Status::Internal("failed to read mapped file size");
  }
  file_bytes = static_cast<std::uint64_t>(size.QuadPart);
#else
  impl->file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (impl->file < 0) return Status::Unavailable("failed to open mapped file");
  struct stat info {};
  if (fstat(impl->file, &info) != 0 || info.st_size < 0) {
    return Status::Internal("failed to read mapped file size");
  }
  if (!S_ISREG(info.st_mode)) return Status::InvalidArgument("mapped path is not regular file");
  file_bytes = static_cast<std::uint64_t>(info.st_size);
#endif
  if (file_bytes > maximum_bytes) {
    return Status::ResourceExhausted("mapped file exceeds byte budget");
  }
  if (file_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("mapped file exceeds address space");
  }
  if (file_bytes == 0) return MappedFile(std::move(impl), nullptr, 0);

#ifdef _WIN32
  impl->mapping = CreateFileMappingW(impl->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (impl->mapping == nullptr) return Status::ResourceExhausted("failed to create file mapping");
  impl->view = MapViewOfFile(impl->mapping, FILE_MAP_READ, 0, 0, 0);
  if (impl->view == nullptr) return Status::ResourceExhausted("failed to map file view");
#else
  impl->view = mmap(nullptr, static_cast<std::size_t>(file_bytes), PROT_READ,
                    MAP_PRIVATE, impl->file, 0);
  if (impl->view == MAP_FAILED) {
    impl->view = nullptr;
    return Status::ResourceExhausted("failed to map file view");
  }
#endif
  impl->view_bytes = static_cast<std::size_t>(file_bytes);
  const auto* data = static_cast<const std::byte*>(impl->view);
  return MappedFile(std::move(impl), data, file_bytes);
}

MappedFile::~MappedFile() = default;

MappedFile::MappedFile(MappedFile&& other) noexcept
    : impl_(std::move(other.impl_)),
      data_(std::exchange(other.data_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    data_ = std::exchange(other.data_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }
  return *this;
}

Result<std::span<const std::byte>> MappedFile::slice(std::uint64_t offset,
                                                    std::uint64_t bytes) const {
  auto end = checked_add_u64(offset, bytes);
  if (!end.ok() || end.value() > size_bytes_) {
    return Status::InvalidArgument("mapped file slice is out of bounds");
  }
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("mapped slice exceeds address space");
  }
  const std::byte* begin = data_;
  if (offset != 0) begin += static_cast<std::size_t>(offset);
  return std::span<const std::byte>(begin, static_cast<std::size_t>(bytes));
}

}  // namespace pih
