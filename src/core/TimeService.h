#pragma once

#include <Arduino.h>

class TimeService
{
public:
    void begin();
    void update();
    bool isSynchronized() const;
    String getCurrentTime() const;

private:
    static constexpr int32_t UTC_OFFSET_SECONDS = -3 * 60 * 60;
    static constexpr uint32_t SYNC_TIMEOUT_MS = 10000;

    bool syncing = false;
    bool synchronized = false;
    uint32_t syncStartedAt = 0;
};
