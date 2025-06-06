#pragma once
#include <cstdint>

constexpr bool is_little_endian() {
    const uint16_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

static_assert(is_little_endian(), "Big-endian architectures are not supported");

