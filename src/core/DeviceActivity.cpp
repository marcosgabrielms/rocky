#include "DeviceActivity.h"

#include <Preferences.h>

bool DeviceActivity::begin()
{
    Preferences preferences;
    if (!preferences.begin(NVS_NAMESPACE, true))
    {
        Serial.println("[DEVICE] state storage unavailable");
        sleeping = false;
        return false;
    }

    const uint8_t storedState = preferences.getUChar(NVS_SLEEPING_KEY, 0);
    preferences.end();
    sleeping = storedState == 1;
    Serial.printf("[DEVICE] state=%s\n", getStateName());
    return true;
}

bool DeviceActivity::setSleeping(bool nextSleeping)
{
    if (sleeping == nextSleeping)
        return true;

    Preferences preferences;
    if (!preferences.begin(NVS_NAMESPACE, false))
    {
        Serial.println("[DEVICE] state persistence failed");
        return false;
    }

    const uint8_t value = nextSleeping ? 1 : 0;
    const bool persisted = preferences.putUChar(NVS_SLEEPING_KEY, value) == sizeof(value);
    preferences.end();
    if (!persisted)
    {
        Serial.println("[DEVICE] state persistence failed");
        return false;
    }

    sleeping = nextSleeping;
    Serial.printf("[DEVICE] state=%s\n", getStateName());
    return true;
}

bool DeviceActivity::isSleeping() const
{
    return sleeping;
}

const char* DeviceActivity::getStateName() const
{
    return sleeping ? "SLEEPING" : "ACTIVE";
}
