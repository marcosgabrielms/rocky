#include "SpeechCapture.h"

#include <cmath>
#include <limits>

bool SpeechCapture::begin()
{
    if (!psramFound())
        return false;

    if (samples == nullptr)
        samples = static_cast<int16_t*>(ps_malloc(MAX_SAMPLES * sizeof(int16_t)));

    psramAllocated = samples != nullptr;
    sampleCount = 0;
    preBufferCount = 0;
    preBufferWriteIndex = 0;
    capturing = false;
    segmentReady = false;
    completed = false;
    limitReached = false;
    metrics = {};
    raw24Metrics = {};
    fixedDurationCapture = false;
    captureSampleLimit = MAX_SAMPLES;
    resetCaptureMetrics();

    return psramAllocated;
}

void SpeechCapture::pushSamples(const int32_t* input, size_t inputCount)
{
    if (samples == nullptr || input == nullptr || inputCount == 0)
        return;

    if (capturing)
        appendSamples(input, inputCount);
    else
        pushPreBufferSamples(input, inputCount);
}

void SpeechCapture::start()
{
    if (samples == nullptr || capturing)
        return;

    resetCaptureMetrics();
    fixedDurationCapture = false;
    captureSampleLimit = MAX_SAMPLES;
    capturing = true;
    copyPreBuffer();
}

void SpeechCapture::startFixedCapture(uint32_t durationMs)
{
    if (samples == nullptr || capturing)
        return;

    resetCaptureMetrics();
    fixedDurationCapture = true;
    captureSampleLimit = min(MAX_SAMPLES, static_cast<size_t>(SAMPLE_RATE * durationMs / 1000UL));
    capturing = captureSampleLimit > 0;
}

void SpeechCapture::finish()
{
    if (!capturing)
        return;

    capturing = false;
    segmentReady = sampleCount > 0;
    completed = segmentReady;
    calculateMetrics();
    calculateRaw24Metrics();
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

size_t SpeechCapture::getPcmByteCount() const
{
    return sampleCount * sizeof(int16_t);
}

const int16_t* SpeechCapture::getPcm16() const
{
    return samples;
}

const SpeechCapture::Pcm16Metrics& SpeechCapture::getMetrics() const
{
    return metrics;
}

const SpeechCapture::Raw24Metrics& SpeechCapture::getRaw24Metrics() const
{
    return raw24Metrics;
}

bool SpeechCapture::usesPsram() const
{
    return psramAllocated;
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
    const size_t remaining = captureSampleLimit - sampleCount;
    const size_t samplesToCopy = inputCount < remaining ? inputCount : remaining;

    for (size_t index = 0; index < samplesToCopy; ++index)
        appendSample(input[index]);

    if (sampleCount == captureSampleLimit)
    {
        limitReached = !fixedDurationCapture;
        finish();
    }
}

void SpeechCapture::appendSample(int32_t sample)
{
    samples[sampleCount] = convertToPcm16(sample);
    raw24Sum += sample;
    raw24SumOfSquares += static_cast<int64_t>(sample) * sample;
    ++raw24SampleCount;

    if (raw24SampleCount == 1)
    {
        raw24Metrics.minimum = sample;
        raw24Metrics.maximum = sample;
    }
    else
    {
        raw24Metrics.minimum = min(raw24Metrics.minimum, sample);
        raw24Metrics.maximum = max(raw24Metrics.maximum, sample);
    }

    const int64_t absolute = sample < 0 ? -static_cast<int64_t>(sample) : sample;
    raw24Metrics.peakAbsolute = max(raw24Metrics.peakAbsolute, static_cast<int32_t>(absolute));
    ++sampleCount;
}

void SpeechCapture::copyPreBuffer()
{
    const size_t firstIndex = (preBufferWriteIndex + PRE_BUFFER_SAMPLES - preBufferCount) %
                              PRE_BUFFER_SAMPLES;

    for (size_t index = 0; index < preBufferCount; ++index)
        appendSample(preBuffer[(firstIndex + index) % PRE_BUFFER_SAMPLES]);

}

void SpeechCapture::calculateMetrics()
{
    metrics = {};

    if (samples == nullptr || sampleCount == 0)
        return;

    int64_t sum = 0;
    uint64_t sumOfSquares = 0;
    metrics.minimum = std::numeric_limits<int16_t>::max();
    metrics.maximum = std::numeric_limits<int16_t>::min();

    for (size_t index = 0; index < sampleCount; ++index)
    {
        const int16_t sample = samples[index];
        sum += sample;
        sumOfSquares += static_cast<int64_t>(sample) * sample;

        if (sample < metrics.minimum)
            metrics.minimum = sample;
        if (sample > metrics.maximum)
            metrics.maximum = sample;
        if (sample == std::numeric_limits<int16_t>::min() ||
            sample == std::numeric_limits<int16_t>::max())
        {
            ++metrics.clippingCount;
        }

        const int32_t absolute = sample < 0 ? -static_cast<int32_t>(sample) : sample;
        metrics.peakAbsolute = max(metrics.peakAbsolute, absolute);
    }

    metrics.mean = static_cast<int32_t>(sum / static_cast<int64_t>(sampleCount));
    metrics.rms = static_cast<uint32_t>(std::sqrt(
        static_cast<double>(sumOfSquares) / sampleCount));
}

void SpeechCapture::calculateRaw24Metrics()
{
    if (raw24SampleCount == 0)
        return;

    raw24Metrics.mean = static_cast<int32_t>(raw24Sum / static_cast<int64_t>(raw24SampleCount));
    raw24Metrics.rms = static_cast<uint32_t>(std::sqrt(
        static_cast<double>(raw24SumOfSquares) / raw24SampleCount));
}

void SpeechCapture::resetCaptureMetrics()
{
    sampleCount = 0;
    segmentReady = false;
    completed = false;
    limitReached = false;
    metrics = {};
    raw24Metrics = {};
    raw24Sum = 0;
    raw24SumOfSquares = 0;
    raw24SampleCount = 0;
}

int16_t SpeechCapture::convertToPcm16(int32_t sample)
{
    constexpr int32_t PCM24_TO_PCM16_DIVISOR = 256;
    constexpr int32_t ROUNDING_OFFSET = PCM24_TO_PCM16_DIVISOR / 2;

    const int32_t scaled = sample >= 0
                               ? (sample + ROUNDING_OFFSET) / PCM24_TO_PCM16_DIVISOR
                               : (sample - ROUNDING_OFFSET) / PCM24_TO_PCM16_DIVISOR;

    if (scaled > std::numeric_limits<int16_t>::max())
        return std::numeric_limits<int16_t>::max();
    if (scaled < std::numeric_limits<int16_t>::min())
        return std::numeric_limits<int16_t>::min();

    return static_cast<int16_t>(scaled);
}
