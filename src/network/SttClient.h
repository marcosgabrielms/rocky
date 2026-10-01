#pragma once

#include <Arduino.h>
#include <WiFiClient.h>

class Speaker;

class SttClient
{
public:
    struct CalibrationMetadata
    {
        const char* mode = "vad";
        const char* voiceLevel = "normal";
        const char* fanState = "off";
        uint32_t distanceCm = 0;
        uint32_t sampleId = 0;
        uint32_t noiseRms = 0;
        uint32_t noisePeakRms = 0;
        uint32_t raw24Rms = 0;
        uint32_t pcm16Rms = 0;
        int32_t peakAbsolute = 0;
        size_t clippingCount = 0;
        uint32_t vadTriggerRms = 0;
        uint32_t vadTriggerDelayMs = 0;
    };

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
        bool audioAvailable = false;
        BackendAction action;
        bool valid = false;
    };

    struct RealtimeResponse
    {
        bool realtime = false;
        uint32_t interactionId = 0;
        BackendResponse backendResponse;
        bool valid = false;
    };

    struct PlaybackCallbacks
    {
        void (*onStarted)() = nullptr;
        void (*onProgress)() = nullptr;
    };

    bool checkHealth();
    bool transcribe(const int16_t* pcm16, size_t pcmByteCount, BackendResponse& response);
    bool transcribeRealtime(const int16_t* pcm16, size_t pcmByteCount, RealtimeResponse& response);
    bool playResponseAudio(Speaker& speaker, const PlaybackCallbacks& callbacks);
    bool playRealtimeInteraction(Speaker& speaker, uint32_t interactionId,
                                 const PlaybackCallbacks& callbacks);
    bool uploadDataset(const int16_t* pcm16, size_t pcmByteCount, const char* label, uint32_t& index);
    bool uploadCalibration(const int16_t* pcm16,
                           size_t pcmByteCount,
                           const CalibrationMetadata& metadata,
                           String& filename);

private:
    struct WavInfo
    {
        const uint8_t* pcmData = nullptr;
        size_t pcmByteCount = 0;
        uint32_t sampleRate = 0;
        uint16_t bitsPerSample = 0;
        uint16_t channels = 0;
    };

    struct HttpResponseHeaders
    {
        size_t contentLength = 0;
        bool isWavResponse = false;
        bool hasInteractionId = false;
        bool hasSequence = false;
        uint32_t interactionId = 0;
        uint32_t sequence = 0;
    };

    static constexpr uint32_t HTTP_TIMEOUT_MS = 60000;

    bool sendGetRequest(WiFiClient& client) const;
    bool sendMultipartRequest(WiFiClient& client,
                              const int16_t* pcm16,
                              size_t pcmByteCount) const;
    bool sendRealtimeMultipartRequest(WiFiClient& client,
                                      const int16_t* pcm16,
                                      size_t pcmByteCount) const;
    bool playAudioPath(Speaker& speaker, const String& path, const PlaybackCallbacks& callbacks);
    bool playWavResponse(Speaker& speaker, WiFiClient& client, size_t wavSize,
                         const PlaybackCallbacks& callbacks, const uint32_t* sequence = nullptr,
                         const uint32_t* previousPlaybackEnd = nullptr) const;
    bool acknowledgeRealtimeSegment(uint32_t interactionId, uint32_t sequence) const;
    bool sendDatasetRequest(WiFiClient& client,
                            const int16_t* pcm16,
                            size_t pcmByteCount,
                            const char* label) const;
    bool sendCalibrationRequest(WiFiClient& client,
                                const int16_t* pcm16,
                                size_t pcmByteCount,
                                const CalibrationMetadata& metadata) const;
    static bool writeAll(WiFiClient& client, const uint8_t* data, size_t dataSize);
    static int readStatusCode(WiFiClient& client);
    static int readResponseHeaders(WiFiClient& client, HttpResponseHeaders& headers,
                                    void (*onProgress)() = nullptr);
    static String readResponseBody(WiFiClient& client, void (*onProgress)() = nullptr);
    static bool parseWav(const uint8_t* wavData, size_t wavSize, WavInfo& wavInfo);
    static bool isHealthyResponse(const String& json);
    static bool extractText(const String& json, String& text);
    static bool extractBackendResponse(const String& json, BackendResponse& response);
    static bool extractIndex(const String& json, uint32_t& index);
    static bool extractJsonString(const String& json, const char* key, String& value);
    static bool extractJsonUnsigned(const String& json, const char* key, uint32_t& value);
    static bool extractJsonBoolean(const String& json, const char* key, bool& value);
    static void createWavHeader(uint8_t (&header)[44], size_t pcmByteCount);
};
