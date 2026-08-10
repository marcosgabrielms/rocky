#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "driver/i2c_master.h"
#include "esp_lcd_panel_ops.h"

class Display {
public:
    bool begin();

    void clear();
    void fillRect(int16_t x, int16_t y, int16_t width, int16_t height);
    void fillRoundRect(int16_t x, int16_t y, int16_t width, int16_t height, int16_t radius);
    bool show();

    constexpr int16_t width() const { return WIDTH; }
    constexpr int16_t height() const { return HEIGHT; }

private:
    static constexpr int16_t WIDTH = 128;
    static constexpr int16_t HEIGHT = 64;
    static constexpr size_t FRAMEBUFFER_BYTES = WIDTH * HEIGHT / 8;

    i2c_master_bus_handle_t i2cBus = nullptr;
    esp_lcd_panel_handle_t panel = nullptr;
    std::array<uint8_t, FRAMEBUFFER_BYTES> framebuffer{};

    void setPixel(int32_t x, int32_t y);
};
