#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include "pih/core/status.h"

namespace pih {

template <typename T>
class [[nodiscard]] Result final {
 public:
  Result(T value) : storage_(std::in_place_type<T>, std::move(value)) {}

  Result(Status status) : storage_(std::in_place_type<Status>, std::move(status)) {
    if (std::get<Status>(storage_).ok()) {
      throw std::invalid_argument("Result error status must not be OK");
    }
  }

  [[nodiscard]] bool ok() const noexcept { return std::holds_alternative<T>(storage_); }

  [[nodiscard]] const Status& status() const noexcept {
    if (const auto* status = std::get_if<Status>(&storage_)) {
      return *status;
    }
    static const Status kOk = Status::Ok();
    return kOk;
  }

  T& value() & {
    if (!ok()) {
      throw std::logic_error("Result has no value");
    }
    return std::get<T>(storage_);
  }

  const T& value() const& {
    if (!ok()) {
      throw std::logic_error("Result has no value");
    }
    return std::get<T>(storage_);
  }

  T&& value() && {
    if (!ok()) {
      throw std::logic_error("Result has no value");
    }
    return std::get<T>(std::move(storage_));
  }

  T& operator*() & { return value(); }
  const T& operator*() const& { return value(); }
  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }

 private:
  std::variant<T, Status> storage_;
};

}  // namespace pih
