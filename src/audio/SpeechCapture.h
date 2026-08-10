#pragma once

#include <Arduino.h>

class SpeechCapture
{
public:
    static constexpr uint32_t SAMPLE_RATE = 16000;
    static constexpr uint32_t MAX_SPEECH_DURATION_MS = 2000;
    static constexpr uint32_t PRE_SPEECH_DURATION_MS = 160;
    static constexpr size_t MAX_SAMPLES = SAMPLE_RATE * MAX_SPEECH_DURATION_MS / 1000;
    static constexpr size_t PRE_BUFFER_SAMPLES = SAMPLE_RATE * PRE_SPEECH_DURATION_MS / 1000;

    void begin();
    void pushSamples(const int32_t* samples, size_t sampleCount);
    void start();
    void finish();

    bool isCapturing() const;
    bool isSegmentReady() const;
    bool consumeCompleted();
    bool wasLimited() const;
    size_t getSampleCount() const;
    uint32_t getDurationMs() const;
    const int32_t* getSamples() const;

private:
    int32_t samples[MAX_SAMPLES];
    int32_t preBuffer[PRE_BUFFER_SAMPLES];
    size_t sampleCount = 0;
    size_t preBufferCount = 0;
    size_t preBufferWriteIndex = 0;
    bool capturing = false;
    bool segmentReady = false;
    bool completed = false;
    bool limitReached = false;

    void pushPreBufferSamples(const int32_t* input, size_t inputCount);
    void appendSamples(const int32_t* input, size_t inputCount);
    void copyPreBuffer();
};
