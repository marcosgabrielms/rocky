#pragma once

#include <Arduino.h>

class WifiManager
{
public:
    void begin();
    void update();
    bool isConnected() const;

private:
    static constexpr uint32_t CONNECTION_TIMEOUT_MS = 10000;

    bool connecting = false;
    bool connectionReported = false;
    uint32_t connectionStartedAt = 0;

    void reportConnection();
    void reportFailure();
};
