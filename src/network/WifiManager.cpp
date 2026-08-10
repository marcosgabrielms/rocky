#include "WifiManager.h"

#include <WiFi.h>

#include "config/WifiConfig.h"

void WifiManager::begin()
{
    Serial.println("[WIFI] Connecting...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    connectionStartedAt = millis();
    connecting = true;
    connectionReported = false;
}

void WifiManager::update()
{
    if (!connecting || connectionReported)
        return;

    if (WiFi.status() == WL_CONNECTED)
    {
        reportConnection();
        return;
    }

    if (millis() - connectionStartedAt >= CONNECTION_TIMEOUT_MS)
        reportFailure();
}

bool WifiManager::isConnected() const
{
    return WiFi.status() == WL_CONNECTED;
}

void WifiManager::reportConnection()
{
    connecting = false;
    connectionReported = true;
    Serial.println("[WIFI] Connected");
    Serial.printf("[WIFI] IP: %s\n", WiFi.localIP().toString().c_str());
}

void WifiManager::reportFailure()
{
    connecting = false;
    connectionReported = true;
    WiFi.disconnect();
    Serial.println("[WIFI] Connection failed");
}
