#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "turboquant/config.hpp"

namespace tq {

// Owning, 128-byte aligned, POSIX-memalign backed buffer.
// Never throws. Move-only. Size is fixed at construction and cannot grow.
template <class T>
class AlignedBuffer {
    static_assert(std::is_trivially_destructible_v<T>,
                  "AlignedBuffer requires trivially destructible element type");

 public:
    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(std::size_t n) noexcept { (void)resize(n); }

    ~AlignedBuffer() noexcept { free_(); }

    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept
        : data_(o.data_), size_(o.size_) { o.data_ = nullptr; o.size_ = 0; }

    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            free_();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    // Allocates n elements, 128-byte aligned. Returns false on failure;
    // in that case the buffer is left empty.
    [[nodiscard]] bool resize(std::size_t n) noexcept {
        free_();
        if (n == 0) return true;
        void* p = nullptr;
        const std::size_t bytes = n * sizeof(T);
        // posix_memalign rounds up to a multiple of alignment/sizeof(void*).
        const int rc = ::posix_memalign(&p, kCacheLine, bytes);
        if (rc != 0 || p == nullptr) return false;
        data_ = static_cast<T*>(p);
        size_ = n;
        std::memset(data_, 0, bytes);
        return true;
    }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    T&       operator[](std::size_t i)       noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

 private:
    void free_() noexcept {
        if (data_) { std::free(data_); data_ = nullptr; }
        size_ = 0;
    }

    T*          data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace tq
