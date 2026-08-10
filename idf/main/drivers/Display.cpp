#include "drivers/Display.h"

#include <algorithm>
#include <cstring>

#include "esp_err.h"
#include "esp_lcd_io_i2c.h"
#include "esp_lcd_panel_ssd1306.h"
#include "esp_log.h"

namespace {

constexpr const char *TAG = "DISPLAY";
constexpr gpio_num_t SDA_PIN = GPIO_NUM_8;
constexpr gpio_num_t SCL_PIN = GPIO_NUM_9;
constexpr uint8_t I2C_ADDRESS = 0x3C;
constexpr uint32_t I2C_FREQUENCY_HZ = 400000;

}  // namespace

bool Display::begin() {
    ESP_LOGI(TAG, "[DISPLAY] Initializing I2C...");

    const i2c_master_bus_config_t busConfig = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = SDA_PIN,
        .scl_io_num = SCL_PIN,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .intr_priority = 0,
        .trans_queue_depth = 0,
        .flags = {.enable_internal_pullup = true, .allow_pd = false},
    };
    if (i2c_new_master_bus(&busConfig, &i2cBus) != ESP_OK) {
        ESP_LOGE(TAG, "[DISPLAY] I2C bus initialization failed");
        return false;
    }

    if (i2c_master_probe(i2cBus, I2C_ADDRESS, 100) != ESP_OK) {
        ESP_LOGE(TAG, "[I2C] device 0x%02X not found", I2C_ADDRESS);
        return false;
    }
    ESP_LOGI(TAG, "[I2C] device found at 0x%02X", I2C_ADDRESS);

    const esp_lcd_panel_io_i2c_config_t ioConfig = {
        .dev_addr = I2C_ADDRESS,
        .on_color_trans_done = nullptr,
        .user_ctx = nullptr,
        .control_phase_bytes = 1,
        .dc_bit_offset = 6,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .flags = {},
        .scl_speed_hz = I2C_FREQUENCY_HZ,
        .transaction_timeout_ms = 1000,
    };
    esp_lcd_panel_io_handle_t io = nullptr;
    if (esp_lcd_new_panel_io_i2c(i2cBus, &ioConfig, &io) != ESP_OK) {
        ESP_LOGE(TAG, "[DISPLAY] SSD1306 I2C IO initialization failed");
        return false;
    }

    esp_lcd_panel_ssd1306_config_t ssd1306Config = {.height = HEIGHT};
    const esp_lcd_panel_dev_config_t panelConfig = {
        .reset_gpio_num = -1,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .data_endian = LCD_RGB_DATA_ENDIAN_LITTLE,
        .bits_per_pixel = 1,
        .flags = {},
        .vendor_config = &ssd1306Config,
    };
    if (esp_lcd_new_panel_ssd1306(io, &panelConfig, &panel) != ESP_OK ||
        esp_lcd_panel_reset(panel) != ESP_OK ||
        esp_lcd_panel_init(panel) != ESP_OK ||
        esp_lcd_panel_disp_on_off(panel, true) != ESP_OK) {
        ESP_LOGE(TAG, "[DISPLAY] SSD1306 initialization failed");
        return false;
    }

    clear();
    if (!show()) {
        return false;
    }
    ESP_LOGI(TAG, "[DISPLAY] Device: 0x%02X", I2C_ADDRESS);
    ESP_LOGI(TAG, "[DISPLAY] SSD1306: initialized");
    ESP_LOGI(TAG, "[DISPLAY] Resolution: %dx%d", WIDTH, HEIGHT);
    ESP_LOGI(TAG, "[DISPLAY] Framebuffer: %u bytes", static_cast<unsigned>(framebuffer.size()));
    return true;
}

void Display::clear() {
    framebuffer.fill(0);
}

void Display::fillRect(int16_t x, int16_t y, int16_t rectWidth, int16_t rectHeight) {
    if (rectWidth <= 0 || rectHeight <= 0) {
        return;
    }
    const int32_t left = std::max<int32_t>(0, x);
    const int32_t top = std::max<int32_t>(0, y);
    const int32_t right = std::min<int32_t>(WIDTH, static_cast<int32_t>(x) + rectWidth);
    const int32_t bottom = std::min<int32_t>(HEIGHT, static_cast<int32_t>(y) + rectHeight);
    for (int32_t pixelY = top; pixelY < bottom; ++pixelY) {
        for (int32_t pixelX = left; pixelX < right; ++pixelX) {
            setPixel(pixelX, pixelY);
        }
    }
}

void Display::fillRoundRect(int16_t x, int16_t y, int16_t rectWidth, int16_t rectHeight, int16_t radius) {
    if (rectWidth <= 0 || rectHeight <= 0) {
        return;
    }
    const int32_t limitedRadius = std::clamp<int32_t>(radius, 0, std::min(rectWidth, rectHeight) / 2);
    if (limitedRadius == 0) {
        fillRect(x, y, rectWidth, rectHeight);
        return;
    }

    const int32_t radiusSquared = limitedRadius * limitedRadius;
    for (int32_t localY = 0; localY < rectHeight; ++localY) {
        for (int32_t localX = 0; localX < rectWidth; ++localX) {
            const int32_t centerX = localX < limitedRadius ? limitedRadius - 1
                                  : localX >= rectWidth - limitedRadius ? rectWidth - limitedRadius
                                                                         : localX;
            const int32_t centerY = localY < limitedRadius ? limitedRadius - 1
                                  : localY >= rectHeight - limitedRadius ? rectHeight - limitedRadius
                                                                          : localY;
            const int32_t deltaX = localX - centerX;
            const int32_t deltaY = localY - centerY;
            if (deltaX * deltaX + deltaY * deltaY <= radiusSquared) {
                setPixel(static_cast<int32_t>(x) + localX, static_cast<int32_t>(y) + localY);
            }
        }
    }
}

bool Display::show() {
    if (panel == nullptr) {
        return false;
    }
    const esp_err_t result = esp_lcd_panel_draw_bitmap(panel, 0, 0, WIDTH, HEIGHT, framebuffer.data());
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "[DISPLAY] framebuffer transfer failed: %s", esp_err_to_name(result));
        return false;
    }
    return true;
}

void Display::setPixel(int32_t x, int32_t y) {
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
        return;
    }
    framebuffer[static_cast<size_t>(y / 8) * WIDTH + x] |= static_cast<uint8_t>(1U << (y % 8));
}
