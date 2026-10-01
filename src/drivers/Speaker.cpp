#include "Speaker.h"

#include <math.h>

namespace
{
constexpr float FULL_CYCLE_RADIANS = 6.28318530717958647692F;
constexpr uint32_t SILENCE_BLOCKS = 4;
}

bool Speaker::begin()
{
    i2s.setPins(BCLK_PIN, LRC_PIN, DATA_OUT_PIN);
    initialized = i2s.begin(I2S_MODE_STD,
                            SAMPLE_RATE,
                            I2S_DATA_BIT_WIDTH_16BIT,
                            I2S_SLOT_MODE_STEREO,
                            I2S_STD_SLOT_BOTH,
                            I2S_ROLE_MASTER);

    if (!initialized)
    {
        Serial.println("[SPEAKER] init failed");
        return false;
    }

    Serial.println("[SPEAKER] init ok");
    writeSilence();
    Serial.println("[SPEAKER] silence");
    return true;
}

void Speaker::playTestTone()
{
    if (!initialized)
        return;

    Serial.println("[SPEAKER] test tone start");

    int16_t frames[FRAMES_PER_BLOCK * SAMPLES_PER_FRAME]{};
    const float phaseStep = FULL_CYCLE_RADIANS * static_cast<float>(TONE_FREQUENCY_HZ) /
                            static_cast<float>(SAMPLE_RATE);
    float phase = 0.0F;
    const size_t frameCount = SAMPLE_RATE * TONE_DURATION_MS / 1000UL;

    for (size_t frameOffset = 0; frameOffset < frameCount; frameOffset += FRAMES_PER_BLOCK)
    {
        const size_t framesInBlock = min(FRAMES_PER_BLOCK, frameCount - frameOffset);

        for (size_t frame = 0; frame < framesInBlock; ++frame)
        {
            const int16_t sample = static_cast<int16_t>(sinf(phase) * TONE_AMPLITUDE);
            frames[frame * SAMPLES_PER_FRAME] = sample;
            frames[frame * SAMPLES_PER_FRAME + 1] = sample;
            phase += phaseStep;

            if (phase >= FULL_CYCLE_RADIANS)
                phase -= FULL_CYCLE_RADIANS;
        }

        if (!writeFrames(frames, framesInBlock))
            break;
    }

    Serial.println("[SPEAKER] test tone end");
    writeSilence();
    Serial.println("[SPEAKER] silence");
}

bool Speaker::beginPcmPlayback(uint32_t sampleRate, uint16_t bitsPerSample, uint16_t channels)
{
    if (!initialized || sampleRate != SAMPLE_RATE || bitsPerSample != 16 || channels != 1)
        return false;

    Serial.println("[AUDIO] playback start");
    return true;
}

bool Speaker::playMonoPcm(const int16_t* samples, size_t sampleCount)
{
    if (!initialized || samples == nullptr || sampleCount == 0)
        return false;

    int16_t frames[FRAMES_PER_BLOCK * SAMPLES_PER_FRAME]{};
    size_t sampleOffset = 0;

    while (sampleOffset < sampleCount)
    {
        const size_t framesInBlock = min(FRAMES_PER_BLOCK, sampleCount - sampleOffset);
        for (size_t frame = 0; frame < framesInBlock; ++frame)
        {
            const int16_t sample = samples[sampleOffset + frame];
            frames[frame * SAMPLES_PER_FRAME] = sample;
            frames[frame * SAMPLES_PER_FRAME + 1] = sample;
        }

        if (!writeFrames(frames, framesInBlock))
            return false;

        sampleOffset += framesInBlock;
    }

    return true;
}

void Speaker::stop()
{
    if (initialized)
    {
        writeSilence();
        Serial.println("[AUDIO] silence");
    }
}

bool Speaker::writeFrames(const int16_t* frames, size_t frameCount)
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(frames);
    constexpr size_t BYTES_PER_STEREO_FRAME = SAMPLES_PER_FRAME * sizeof(int16_t);
    const size_t totalBytes = frameCount * BYTES_PER_STEREO_FRAME;
    size_t offset = 0;

    while (offset < totalBytes)
    {
        const size_t bytesWritten = i2s.write(data + offset, totalBytes - offset);
        if (bytesWritten == 0 || bytesWritten % BYTES_PER_STEREO_FRAME != 0)
            return false;

        offset += bytesWritten;
    }

    return true;
}

void Speaker::writeSilence()
{
    const int16_t silence[FRAMES_PER_BLOCK * SAMPLES_PER_FRAME]{};

    for (uint32_t block = 0; block < SILENCE_BLOCKS; ++block)
        writeFrames(silence, FRAMES_PER_BLOCK);

}
