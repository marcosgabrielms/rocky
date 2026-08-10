#include "Microphone.h"

bool Microphone::begin()
{
    i2s.setPins(SCK_PIN, WS_PIN, -1, SD_PIN);

    initialized = i2s.begin(I2S_MODE_STD,
                            SAMPLE_RATE,
                            I2S_DATA_BIT_WIDTH_32BIT,
                            I2S_SLOT_MODE_MONO,
                            I2S_STD_SLOT_LEFT,
                            I2S_ROLE_MASTER);
    return initialized;
}

bool Microphone::available()
{
    return initialized && i2s.available() >= static_cast<int>(BYTES_PER_SAMPLE);
}

size_t Microphone::readSamples(int32_t* buffer, size_t sampleCount)
{
    if (!initialized || buffer == nullptr || sampleCount == 0)
        return 0;

    const size_t availableSamples = static_cast<size_t>(i2s.available()) / BYTES_PER_SAMPLE;
    const size_t samplesToRead = sampleCount < availableSamples ? sampleCount : availableSamples;
    const size_t bytesRead = i2s.readBytes(reinterpret_cast<char*>(buffer),
                                           samplesToRead * BYTES_PER_SAMPLE);
    const size_t samplesRead = bytesRead / BYTES_PER_SAMPLE;

    // O INMP441 entrega 24 bits assinados alinhados nos bits mais altos do slot de 32 bits.
    for (size_t index = 0; index < samplesRead; ++index)
        buffer[index] >>= 8;

    return samplesRead;
}
