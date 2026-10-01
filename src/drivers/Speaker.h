#pragma once

#include <Arduino.h>
#include <ESP_I2S.h>

class Speaker
{
public:
    bool begin();
    void playTestTone();
    bool beginPcmPlayback(uint32_t sampleRate, uint16_t bitsPerSample, uint16_t channels);
    bool playMonoPcm(const int16_t* samples, size_t sampleCount);
    void stop();

private:
    static constexpr int8_t BCLK_PIN = 16;
    static constexpr int8_t LRC_PIN = 17;
    static constexpr int8_t DATA_OUT_PIN = 18;
    static constexpr uint32_t SAMPLE_RATE = 32000;
    static constexpr uint16_t TONE_FREQUENCY_HZ = 440;
    static constexpr uint32_t TONE_DURATION_MS = 1000;
    static constexpr int16_t TONE_AMPLITUDE = 3932;
    static constexpr int32_t OUTPUT_VOLUME_PERCENT = 30;
    static constexpr size_t FRAMES_PER_BLOCK = 126;
    static constexpr size_t SAMPLES_PER_FRAME = 2;

    bool writeFrames(const int16_t* frames, size_t frameCount);
    void writeSilence();
    static int16_t scaleOutputSample(int16_t sample);

    I2SClass i2s{I2S_NUM_1};
    bool initialized = false;
};
