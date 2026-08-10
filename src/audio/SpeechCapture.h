#pragma once

#include <Arduino.h>

class SpeechCapture
{
public:
    struct Pcm16Metrics
    {
        int16_t minimum = 0;
        int16_t maximum = 0;
        int32_t mean = 0;
        uint32_t rms = 0;
        size_t clippingCount = 0;
    };

    static constexpr uint32_t SAMPLE_RATE = 16000;
    static constexpr uint32_t MAX_SPEECH_DURATION_MS = 4000;
    static constexpr uint32_t PRE_SPEECH_DURATION_MS = 160;
    static constexpr size_t MAX_SAMPLES = SAMPLE_RATE * MAX_SPEECH_DURATION_MS / 1000;
    static constexpr size_t PRE_BUFFER_SAMPLES = SAMPLE_RATE * PRE_SPEECH_DURATION_MS / 1000;

    bool begin();
    void pushSamples(const int32_t* samples, size_t sampleCount);
    void start();
    void finish();

    bool isCapturing() const;
    bool isSegmentReady() const;
    bool consumeCompleted();
    bool wasLimited() const;
    size_t getSampleCount() const;
    uint32_t getDurationMs() const;
    size_t getPcmByteCount() const;
    const int16_t* getPcm16() const;
    const Pcm16Metrics& getMetrics() const;
    bool usesPsram() const;

private:
    int16_t* samples = nullptr;
    int32_t preBuffer[PRE_BUFFER_SAMPLES];
    Pcm16Metrics metrics;
    size_t sampleCount = 0;
    size_t preBufferCount = 0;
    size_t preBufferWriteIndex = 0;
    bool capturing = false;
    bool segmentReady = false;
    bool completed = false;
    bool limitReached = false;
    bool psramAllocated = false;

    void pushPreBufferSamples(const int32_t* input, size_t inputCount);
    void appendSamples(const int32_t* input, size_t inputCount);
    void copyPreBuffer();
    void calculateMetrics();
    static int16_t convertToPcm16(int32_t sample);
};
