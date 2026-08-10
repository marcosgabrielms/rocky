#pragma once

#include <cstdint>

#include "esp_timer.h"

inline uint64_t nowMs() {
    return static_cast<uint64_t>(esp_timer_get_time()) / 1000ULL;
}
