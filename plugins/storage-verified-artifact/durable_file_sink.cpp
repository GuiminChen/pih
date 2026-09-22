#include "pih/io/durable_file_sink.h"

#include <algorithm>
#include <cerrno>
#include <limits>

#include "pih/core/checked_math.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pih {

Result<DurableFileSink> DurableFileSink::CreateExclusive(
    const std::filesystem::path& path, std::uint64_t expected_file_bytes) {
  if (expected_file_bytes == 0) {
    return Status::InvalidArgument("durable sink expected length is zero");
  }
#ifdef _WIN32
  const int descriptor = _wopen(path.c_str(),
                                _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                                _S_IREAD | _S_IWRITE);
#else
  const int descriptor = ::open(path.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
#endif
  if (descriptor < 0) {
    return Status::FailedPrecondition(
        "durable sink cannot exclusively create target");
  }
  return DurableFileSink(descriptor, expected_file_bytes);
}

DurableFileSink::DurableFileSink(DurableFileSink&& other) noexcept
    : descriptor_(other.descriptor_),
      expected_file_bytes_(other.expected_file_bytes_),
      bytes_written_(other.bytes_written_), synced_(other.synced_) {
  other.descriptor_ = -1;
}

DurableFileSink& DurableFileSink::operator=(DurableFileSink&& other) noexcept {
  if (this == &other) return *this;
  close();
  descriptor_ = other.descriptor_;
  expected_file_bytes_ = other.expected_file_bytes_;
  bytes_written_ = other.bytes_written_;
  synced_ = other.synced_;
  other.descriptor_ = -1;
  return *this;
}

DurableFileSink::~DurableFileSink() { close(); }

void DurableFileSink::close() noexcept {
  if (descriptor_ < 0) return;
#ifdef _WIN32
  _close(descriptor_);
#else
  ::close(descriptor_);
#endif
  descriptor_ = -1;
}

Result<std::size_t> DurableFileSink::write(
    std::span<const std::byte> input) {
  if (descriptor_ < 0 || synced_ || input.empty()) {
    return Status::FailedPrecondition("durable sink is not writable");
  }
  auto next = checked_add_u64(bytes_written_, input.size());
  if (!next.ok() || *next > expected_file_bytes_) {
    return Status::ResourceExhausted("durable sink exceeds expected length");
  }
#ifdef _WIN32
  const auto request = static_cast<unsigned int>(std::min<std::size_t>(
      input.size(), std::numeric_limits<unsigned int>::max()));
  const int written = _write(descriptor_, input.data(), request);
#else
  const ssize_t written = ::write(descriptor_, input.data(), input.size());
#endif
  if (written <= 0) {
    return Status::Internal("durable sink write failed");
  }
  bytes_written_ += static_cast<std::uint64_t>(written);
  return static_cast<std::size_t>(written);
}

Status DurableFileSink::sync() {
  if (descriptor_ < 0 || synced_ || bytes_written_ != expected_file_bytes_) {
    return Status::FailedPrecondition(
        "durable sink cannot sync incomplete file");
  }
#ifdef _WIN32
  if (_commit(descriptor_) != 0) {
#else
  if (::fsync(descriptor_) != 0) {
#endif
    return Status::Internal("durable sink fsync failed");
  }
  synced_ = true;
  return Status::Ok();
}

}  // namespace pih
