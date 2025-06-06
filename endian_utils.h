#pragma once
#include <cstdint>

// use compiler-provided byte order macros for constexpr evaluation
constexpr bool is_little_endian() {
    return __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
}

static_assert(is_little_endian(), "Big-endian architectures are not supported");

