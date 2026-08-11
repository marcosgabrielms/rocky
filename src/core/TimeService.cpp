#include "TimeService.h"

#include <time.h>

void TimeService::begin()
{
    if (syncing || synchronized)
        return;

    Serial.println("[TIME] syncing...");
    configTime(UTC_OFFSET_SECONDS, 0, "pool.ntp.org");
    syncStartedAt = millis();
    syncing = true;
}

void TimeService::update()
{
    if (!syncing)
        return;

    tm timeInfo{};
    if (getLocalTime(&timeInfo, 0))
    {
        syncing = false;
        synchronized = true;
        Serial.println("[TIME] synchronized");
        return;
    }

    if (millis() - syncStartedAt >= SYNC_TIMEOUT_MS)
    {
        syncing = false;
        Serial.println("[TIME] sync failed");
    }
}

bool TimeService::isSynchronized() const
{
    return synchronized;
}

String TimeService::getCurrentTime() const
{
    tm timeInfo{};
    if (!getLocalTime(&timeInfo, 0))
        return String();

    char formattedTime[6]{};
    strftime(formattedTime, sizeof(formattedTime), "%H:%M", &timeInfo);
    return String(formattedTime);
}
