#pragma once

#include <cstdint>
#include <string_view>

// We compile the codec TUs with -fno-exceptions, so the public API uses
// `Result<T> = std::expected<T, Error>` at factory boundaries and plain
// `Error` return codes on hot paths. std::expected is C++23; we use a
// minimal local fallback if the toolchain lacks <expected>.

#if __has_include(<expected>)
#  include <expected>
#endif

namespace tq {

enum class Error : std::uint16_t {
    Ok = 0,
    InvalidDim,
    InvalidBits,
    InvalidSeed,
    InvalidArgument,
    CodebookMissing,
    CodebookCorrupt,
    BufferTooSmall,
    ShapeMismatch,
    RotationFailed,
    LapackFailed,
    NotImplemented,
    IoError,
};

constexpr std::string_view error_name(Error e) noexcept
{
    switch (e) {
        case Error::Ok:              return "Ok";
        case Error::InvalidDim:      return "InvalidDim";
        case Error::InvalidBits:     return "InvalidBits";
        case Error::InvalidSeed:     return "InvalidSeed";
        case Error::InvalidArgument: return "InvalidArgument";
        case Error::CodebookMissing: return "CodebookMissing";
        case Error::CodebookCorrupt: return "CodebookCorrupt";
        case Error::BufferTooSmall:  return "BufferTooSmall";
        case Error::ShapeMismatch:   return "ShapeMismatch";
        case Error::RotationFailed:  return "RotationFailed";
        case Error::LapackFailed:    return "LapackFailed";
        case Error::NotImplemented:  return "NotImplemented";
        case Error::IoError:         return "IoError";
    }
    return "Unknown";
}

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
template <class T> using Result = std::expected<T, Error>;
template <class T>
constexpr auto make_error(Error e) { return std::unexpected<Error>(e); }
#else
// Minimal stand-in until the toolchain ships <expected>. Same shape as
// std::expected for our use cases. Non-exceptional failure only.
template <class T>
class Result {
 public:
    constexpr Result(T value) noexcept : value_(static_cast<T&&>(value)), err_(Error::Ok), has_(true) {}
    constexpr Result(Error e) noexcept : err_(e), has_(false) {}

    constexpr bool has_value() const noexcept { return has_; }
    constexpr explicit operator bool() const noexcept { return has_; }

    constexpr T&       value() &       noexcept { return value_; }
    constexpr const T& value() const & noexcept { return value_; }
    constexpr T&&      value() &&      noexcept { return static_cast<T&&>(value_); }

    constexpr T&       operator*() &       noexcept { return value_; }
    constexpr const T& operator*() const & noexcept { return value_; }

    constexpr T*       operator->()       noexcept { return &value_; }
    constexpr const T* operator->() const noexcept { return &value_; }

    constexpr Error error() const noexcept { return err_; }

 private:
    union {
        T value_;
        char dummy_;
    };
    Error err_ = Error::Ok;
    bool  has_ = false;

 public:
    // Simple destructor — T is always trivially destructible in our use
    // cases (AlignedBuffer + PODs wrapped in aggregate types). If this
    // ever hosts a non-trivial T, replace with a proper variant.
    constexpr ~Result() { if (has_) value_.~T(); else (void)dummy_; }

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;
    Result(Result&& other) noexcept : err_(other.err_), has_(other.has_) {
        if (has_) new (&value_) T(static_cast<T&&>(other.value_));
    }
    Result& operator=(Result&& other) noexcept {
        if (has_) value_.~T();
        err_ = other.err_;
        has_ = other.has_;
        if (has_) new (&value_) T(static_cast<T&&>(other.value_));
        return *this;
    }
};

template <class T>
constexpr Result<T> make_error(Error e) noexcept { return Result<T>(e); }
#endif

} // namespace tq
