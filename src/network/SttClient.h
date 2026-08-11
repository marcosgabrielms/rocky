#pragma once

#include <Arduino.h>
#include <WiFiClient.h>

class SttClient
{
public:
    struct BackendAction
    {
        enum class Type : uint8_t { None, Expression, ShowText };
        Type type = Type::None;
        String value;
        String line1;
        String line2;
        uint32_t durationMs = 0;
    };

    struct BackendResponse
    {
        String text;
        String interactionState;
        uint32_t commandWindowMs = 0;
        BackendAction action;
        bool valid = false;
    };

    bool checkHealth();
    bool transcribe(const int16_t* pcm16, size_t pcmByteCount, BackendResponse& response);
    bool uploadDataset(const int16_t* pcm16, size_t pcmByteCount, const char* label, uint32_t& index);

private:
    static constexpr uint32_t HTTP_TIMEOUT_MS = 15000;

    bool sendGetRequest(WiFiClient& client) const;
    bool sendMultipartRequest(WiFiClient& client,
                              const int16_t* pcm16,
                              size_t pcmByteCount) const;
    bool sendDatasetRequest(WiFiClient& client,
                            const int16_t* pcm16,
                            size_t pcmByteCount,
                            const char* label) const;
    static bool writeAll(WiFiClient& client, const uint8_t* data, size_t dataSize);
    static int readStatusCode(WiFiClient& client);
    static String readResponseBody(WiFiClient& client);
    static bool isHealthyResponse(const String& json);
    static bool extractText(const String& json, String& text);
    static bool extractBackendResponse(const String& json, BackendResponse& response);
    static bool extractIndex(const String& json, uint32_t& index);
    static bool extractJsonString(const String& json, const char* key, String& value);
    static bool extractJsonUnsigned(const String& json, const char* key, uint32_t& value);
    static void createWavHeader(uint8_t (&header)[44], size_t pcmByteCount);
};
