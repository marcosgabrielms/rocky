#include "SpeechCapture.h"

void SpeechCapture::begin()
{
    sampleCount = 0;
    preBufferCount = 0;
    preBufferWriteIndex = 0;
    capturing = false;
    segmentReady = false;
    completed = false;
    limitReached = false;
}

void SpeechCapture::pushSamples(const int32_t* input, size_t inputCount)
{
    if (input == nullptr || inputCount == 0)
        return;

    if (capturing)
        appendSamples(input, inputCount);
    else
        pushPreBufferSamples(input, inputCount);
}

void SpeechCapture::start()
{
    if (capturing)
        return;

    sampleCount = 0;
    segmentReady = false;
    completed = false;
    limitReached = false;
    capturing = true;
    copyPreBuffer();
}

void SpeechCapture::finish()
{
    if (!capturing)
        return;

    capturing = false;
    segmentReady = sampleCount > 0;
    completed = segmentReady;
    preBufferCount = 0;
    preBufferWriteIndex = 0;
}

bool SpeechCapture::isCapturing() const
{
    return capturing;
}

bool SpeechCapture::isSegmentReady() const
{
    return segmentReady;
}

bool SpeechCapture::consumeCompleted()
{
    const bool event = completed;
    completed = false;
    return event;
}

bool SpeechCapture::wasLimited() const
{
    return limitReached;
}

size_t SpeechCapture::getSampleCount() const
{
    return sampleCount;
}

uint32_t SpeechCapture::getDurationMs() const
{
    return static_cast<uint32_t>(sampleCount * 1000UL / SAMPLE_RATE);
}

const int32_t* SpeechCapture::getSamples() const
{
    return samples;
}

void SpeechCapture::pushPreBufferSamples(const int32_t* input, size_t inputCount)
{
    for (size_t index = 0; index < inputCount; ++index)
    {
        preBuffer[preBufferWriteIndex] = input[index];
        preBufferWriteIndex = (preBufferWriteIndex + 1) % PRE_BUFFER_SAMPLES;
        if (preBufferCount < PRE_BUFFER_SAMPLES)
            ++preBufferCount;
    }
}

void SpeechCapture::appendSamples(const int32_t* input, size_t inputCount)
{
    const size_t remaining = MAX_SAMPLES - sampleCount;
    const size_t samplesToCopy = inputCount < remaining ? inputCount : remaining;

    for (size_t index = 0; index < samplesToCopy; ++index)
        samples[sampleCount + index] = input[index];

    sampleCount += samplesToCopy;

    if (sampleCount == MAX_SAMPLES)
    {
        limitReached = true;
        finish();
    }
}

void SpeechCapture::copyPreBuffer()
{
    const size_t firstIndex = (preBufferWriteIndex + PRE_BUFFER_SAMPLES - preBufferCount) %
                              PRE_BUFFER_SAMPLES;

    for (size_t index = 0; index < preBufferCount; ++index)
        samples[index] = preBuffer[(firstIndex + index) % PRE_BUFFER_SAMPLES];

    sampleCount = preBufferCount;
}
