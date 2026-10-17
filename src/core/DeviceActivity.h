#pragma once

#include <Arduino.h>

class DeviceActivity
{
public:
    bool begin();
    bool setSleeping(bool sleeping);
    bool isSleeping() const;
    const char* getStateName() const;

private:
    static constexpr const char* NVS_NAMESPACE = "koda_device";
    static constexpr const char* NVS_SLEEPING_KEY = "sleeping";

    bool sleeping = false;
};
