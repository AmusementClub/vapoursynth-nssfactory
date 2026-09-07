#pragma once

#include "nss/params.hpp"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace nss {
inline std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::bad_array_new_length();
    return a + b;
}
template<class... Terms> inline std::size_t checked_sum(Terms... terms) {
    std::size_t value = 0;
    auto add = [&](auto term) {
        if constexpr (std::is_signed_v<decltype(term)>)
            if (term < 0) throw std::bad_array_new_length();
        value = checked_add(value, static_cast<std::size_t>(term));
    };
    (add(terms), ...);
    return value;
}
inline std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) throw std::bad_array_new_length();
    return a * b;
}
inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {
    std::size_t value = 1;
    for (auto factor : factors) value = checked_mul(value, factor);
    return value;
}
inline int checked_int(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("nss: size exceeds the kernel integer range");
    return static_cast<int>(value);
}
inline void checked_solver_shape(int m, int n) {
    if (m < 1 || m > kSvdMaxM || n < 1 || n > kSvdMaxN)
        throw std::invalid_argument("nss: unsupported solver workspace shape");
}
inline int checked_fat_height(int height, int radius) {
    if (height < 1 || radius < 0) throw std::invalid_argument("nss: invalid temporal dimensions");
    return checked_int(static_cast<std::uint64_t>(height) * (2ull * radius + 1) * 2);
}
} // namespace nss
