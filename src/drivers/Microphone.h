#pragma once

#include <Arduino.h>
#include <ESP_I2S.h>

class Microphone
{
public:
    bool begin();
    bool available();
    size_t readSamples(int32_t* buffer, size_t sampleCount);

private:
    static constexpr int8_t SCK_PIN = 4;
    static constexpr int8_t WS_PIN = 5;
    static constexpr int8_t SD_PIN = 6;
    static constexpr uint32_t SAMPLE_RATE = 16000;
    static constexpr size_t BYTES_PER_SAMPLE = sizeof(int32_t);

    I2SClass i2s;
    bool initialized = false;
};
