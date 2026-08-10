#include <inttypes.h>

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_psram.h"

namespace {

constexpr const char *TAG = "ROCKY_IDF";
constexpr size_t BYTES_PER_MEGABYTE = 1024U * 1024U;

}  // namespace

extern "C" void app_main(void) {
    esp_chip_info_t chipInfo = {};
    esp_chip_info(&chipInfo);

    uint32_t flashBytes = 0;
    ESP_ERROR_CHECK(esp_flash_get_size(nullptr, &flashBytes));

    const size_t internalFree = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t psramFree = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    const size_t psramBytes = esp_psram_get_size();

    ESP_LOGI(TAG, "========================");
    ESP_LOGI(TAG, "Rocky ESP-IDF");
    ESP_LOGI(TAG, "========================");
    ESP_LOGI(TAG, "[IDF] %s", esp_get_idf_version());
    ESP_LOGI(TAG, "[CHIP] ESP32-S3 | cores=%d | revision=%d", chipInfo.cores, chipInfo.revision);
    ESP_LOGI(TAG, "[FLASH] %" PRIu32 " MB", flashBytes / BYTES_PER_MEGABYTE);
    ESP_LOGI(TAG, "[PSRAM] %s | %u MB", esp_psram_is_initialized() ? "available" : "unavailable",
             static_cast<unsigned>(psramBytes / BYTES_PER_MEGABYTE));
    ESP_LOGI(TAG, "[MEM] internal=%u", static_cast<unsigned>(internalFree));
    ESP_LOGI(TAG, "[MEM] psram=%u", static_cast<unsigned>(psramFree));
    ESP_LOGI(TAG, "[BOOT] ESP-IDF base OK");
}
