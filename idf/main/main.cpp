#include <inttypes.h>

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_psram.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "drivers/Display.h"
#include "graphics/Animator.h"
#include "graphics/Eyes.h"
#include "graphics/Time.h"

namespace {

constexpr const char *TAG = "ROCKY_IDF";
constexpr size_t BYTES_PER_MEGABYTE = 1024U * 1024U;
constexpr TickType_t VISUAL_TASK_PERIOD = pdMS_TO_TICKS(33);
constexpr uint64_t LISTENING_TEST_START_MS = 5000;
constexpr uint64_t LISTENING_TEST_END_MS = 10000;
constexpr uint64_t LISTENING_TEST_CYCLE_MS = 15000;
constexpr uint64_t VISUAL_REPORT_INTERVAL_MS = 5000;

Display display;
Eyes eyes(display);
Animator animator(eyes);

void visualTask(void *) {
    const uint64_t startedAt = nowMs();
    uint64_t lastReportAt = startedAt;
    uint64_t renderTotalUs = 0;
    uint32_t renderedFrames = 0;
    uint32_t lastReportedFrames = 0;
    Eyes::VisualState testState = Eyes::VisualState::Idle;
    TickType_t lastWakeAt = xTaskGetTickCount();

    animator.begin(startedAt);
    ESP_LOGI(TAG, "[VISUAL] task running at approximately 30 FPS");
    for (;;) {
        const uint64_t now = nowMs();
        const uint64_t testCycleMs = (now - startedAt) % LISTENING_TEST_CYCLE_MS;
        const Eyes::VisualState nextState =
            testCycleMs >= LISTENING_TEST_START_MS && testCycleMs < LISTENING_TEST_END_MS
                ? Eyes::VisualState::Listening
                : Eyes::VisualState::Idle;
        if (nextState != testState) {
            testState = nextState;
            eyes.setVisualState(testState);
            ESP_LOGI(TAG, "[VISUAL] test state: %s",
                     testState == Eyes::VisualState::Listening ? "LISTENING" : "IDLE");
        }

        const int64_t renderStartedAtUs = esp_timer_get_time();
        animator.update(now);
        renderTotalUs += static_cast<uint64_t>(esp_timer_get_time() - renderStartedAtUs);
        ++renderedFrames;

        if (now - lastReportAt >= VISUAL_REPORT_INTERVAL_MS) {
            const uint32_t framesSinceReport = renderedFrames - lastReportedFrames;
            const uint64_t elapsedMs = now - lastReportAt;
            const uint32_t fps = elapsedMs == 0 ? 0 : static_cast<uint32_t>(framesSinceReport * 1000ULL / elapsedMs);
            const uint32_t averageRenderUs = renderedFrames == 0 ? 0 :
                static_cast<uint32_t>(renderTotalUs / renderedFrames);
            ESP_LOGI(TAG, "[VISUAL] fps=%u render_avg=%u us internal=%u",
                     static_cast<unsigned>(fps), static_cast<unsigned>(averageRenderUs), static_cast<unsigned>(
                         heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
            lastReportAt = now;
            lastReportedFrames = renderedFrames;
            renderTotalUs = 0;
            renderedFrames = 0;
            lastReportedFrames = 0;
        }

        vTaskDelayUntil(&lastWakeAt, VISUAL_TASK_PERIOD);
    }
}

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

    if (!display.begin()) {
        ESP_LOGE(TAG, "[DISPLAY] initialization failed; stopping display test");
        return;
    }
    ESP_LOGI(TAG, "[MEM] after_display internal=%u", static_cast<unsigned>(
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
    ESP_LOGI(TAG, "[MEM] after_display psram=%u", static_cast<unsigned>(
                 heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
    ESP_LOGI(TAG, "[BOOT] ESP-IDF base OK");
    xTaskCreatePinnedToCore(visualTask, "visual_task", 4096, nullptr, 4, nullptr, 1);
}
