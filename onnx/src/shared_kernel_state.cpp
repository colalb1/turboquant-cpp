// Process-wide state cache implementation. One std::unordered_map
// guarded by std::shared_mutex. A thread-local 2-slot LRU in front of
// the mutex covers the prefill→decode pair within a single layer's
// kernel compute without contention.
//
// The map stores shared_ptr<void> so a single container holds any
// codec type; the templated get_state<T>() function does the typed
// shared_ptr cast via aliasing-constructor-style shared_ptr copy.

#include "shared_kernel_state.hpp"

#include <shared_mutex>
#include <unordered_map>

namespace tq::onnx {

namespace {

struct Entry {
    std::shared_ptr<void> ptr;  // type-erased; caller knows the actual T
};

using Map = std::unordered_map<StateKey, Entry, StateKeyHash>;

Map& global_map() noexcept {
    static Map m;
    return m;
}
std::shared_mutex& global_mutex() noexcept {
    static std::shared_mutex mu;
    return mu;
}

// Per-thread 2-slot cache — last-used slot rotates.
struct TlsSlot {
    StateKey              key{};
    std::shared_ptr<void> ptr;
    bool                  valid = false;
};

thread_local TlsSlot      tls_slots[2];
thread_local std::uint8_t tls_next = 0;

std::shared_ptr<void> tls_find(const StateKey& key) noexcept {
    for (auto& s : tls_slots) {
        if (s.valid && s.key == key) return s.ptr;
    }
    return {};
}

void tls_insert(const StateKey& key, std::shared_ptr<void> ptr) noexcept {
    tls_slots[tls_next] = TlsSlot{key, std::move(ptr), true};
    tls_next            = (tls_next + 1) & 1;
}

template <class T>
std::shared_ptr<const T> lookup_or_create(const StateKey& key) noexcept {
    // Fast path: thread-local.
    if (auto hit = tls_find(key)) {
        return std::static_pointer_cast<const T>(hit);
    }

    // Shared read path.
    {
        std::shared_lock<std::shared_mutex> rd(global_mutex());
        auto                                it = global_map().find(key);
        if (it != global_map().end()) {
            auto p = it->second.ptr;
            tls_insert(key, p);
            return std::static_pointer_cast<const T>(std::move(p));
        }
    }

    // Cold path: construct.
    auto res = T::make(key.dim, key.seed);
    if (!res) return {};
    auto sp = std::make_shared<T>(std::move(*res));

    {
        std::unique_lock<std::shared_mutex> wr(global_mutex());
        auto [it, inserted] = global_map().try_emplace(key, Entry{sp});
        if (!inserted) {
            // Another thread raced us — reuse its instance.
            sp = std::static_pointer_cast<T>(it->second.ptr);
        }
    }

    tls_insert(key, std::shared_ptr<void>(sp));
    return std::shared_ptr<const T>(sp);
}

}  // namespace

template <class T>
std::shared_ptr<const T> get_state(const StateKey& key) noexcept {
    return lookup_or_create<T>(key);
}

template std::shared_ptr<const TurboQuantMSE<1>>
get_state<TurboQuantMSE<1>>(const StateKey&) noexcept;
template std::shared_ptr<const TurboQuantMSE<2>>
get_state<TurboQuantMSE<2>>(const StateKey&) noexcept;
template std::shared_ptr<const TurboQuantMSE<3>>
get_state<TurboQuantMSE<3>>(const StateKey&) noexcept;
template std::shared_ptr<const TurboQuantMSE<4>>
get_state<TurboQuantMSE<4>>(const StateKey&) noexcept;
template std::shared_ptr<const TurboQuantProd<2>>
get_state<TurboQuantProd<2>>(const StateKey&) noexcept;
template std::shared_ptr<const TurboQuantProd<3>>
get_state<TurboQuantProd<3>>(const StateKey&) noexcept;
template std::shared_ptr<const TurboQuantProd<4>>
get_state<TurboQuantProd<4>>(const StateKey&) noexcept;

}  // namespace tq::onnx
