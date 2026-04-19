#pragma once

#include <cstdint>
#include <expected>
#include <string_view>

// We compile the codec TUs with -fno-exceptions, so the public API uses
// `Result<T> = std::expected<T, Error>` at factory boundaries and plain
// `Error` return codes on hot paths.

#if !defined(__cpp_lib_expected) || __cpp_lib_expected < 202202L
#error "turboquant-cpp requires a toolchain that ships <expected> (C++23)"
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

constexpr std::string_view error_name(Error e) noexcept {
    switch (e) {
    case Error::Ok: return "Ok";
    case Error::InvalidDim: return "InvalidDim";
    case Error::InvalidBits: return "InvalidBits";
    case Error::InvalidSeed: return "InvalidSeed";
    case Error::InvalidArgument: return "InvalidArgument";
    case Error::CodebookMissing: return "CodebookMissing";
    case Error::CodebookCorrupt: return "CodebookCorrupt";
    case Error::BufferTooSmall: return "BufferTooSmall";
    case Error::ShapeMismatch: return "ShapeMismatch";
    case Error::RotationFailed: return "RotationFailed";
    case Error::LapackFailed: return "LapackFailed";
    case Error::NotImplemented: return "NotImplemented";
    case Error::IoError: return "IoError";
    }
    return "Unknown";
}

template <class T>
using Result = std::expected<T, Error>;

template <class T>
constexpr auto make_error(Error e) {
    return std::unexpected<Error>(e);
}

}  // namespace tq
