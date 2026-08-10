#include <Arduino.h>

#include "drivers/Display.h"
#include "drivers/Microphone.h"
#include "audio/SpeechCapture.h"
#include "audio/VoiceActivityDetector.h"
#include "graphics/Eyes.h"
#include "graphics/Animator.h"

Display display;
Microphone microphone;
VoiceActivityDetector voiceActivityDetector;
SpeechCapture speechCapture;
Eyes eyes(display);
Animator animator(eyes);
bool displayReady = false;

namespace
{
constexpr size_t MIC_BLOCK_SAMPLES = 64;
constexpr uint32_t MIC_REPORT_INTERVAL_MS = 250;
constexpr uint32_t VISUAL_TEST_CYCLE_MS = 20000;
constexpr uint32_t IDLE_TEST_DURATION_MS = 5000;
constexpr uint32_t ATTENTION_TEST_DURATION_MS = 10000;
constexpr uint32_t LISTENING_TEST_DURATION_MS = 15000;

int32_t micSamples[MIC_BLOCK_SAMPLES];
uint64_t micSumOfSquares = 0;
int64_t micSum = 0;
size_t micSampleCount = 0;
uint32_t lastMicReportAt = 0;
uint32_t visualTestStartedAt = 0;

void updateTemporaryVisualTest(uint32_t now)
{
    const uint32_t elapsed = (now - visualTestStartedAt) % VISUAL_TEST_CYCLE_MS;

    if (elapsed < IDLE_TEST_DURATION_MS)
    {
        animator.setVisualState(Eyes::VisualState::Idle);
        return;
    }

    if (elapsed < ATTENTION_TEST_DURATION_MS)
    {
        animator.setVisualState(Eyes::VisualState::Attention);
        return;
    }

    if (elapsed < LISTENING_TEST_DURATION_MS)
    {
        animator.setVisualState(Eyes::VisualState::Listening);
        return;
    }

    animator.setVisualState(Eyes::VisualState::Idle);
}

void accumulateMicrophoneSamples(const int32_t* samples, size_t sampleCount)
{
    for (size_t index = 0; index < sampleCount; ++index)
    {
        const int32_t sample = samples[index];
        micSum += sample;
        micSumOfSquares += static_cast<int64_t>(sample) * sample;
    }

    micSampleCount += sampleCount;
}

float calculateRms(const int32_t* samples, size_t sampleCount)
{
    if (samples == nullptr || sampleCount == 0)
        return 0.0F;

    int64_t sum = 0;
    uint64_t sumOfSquares = 0;
    for (size_t index = 0; index < sampleCount; ++index)
    {
        sum += samples[index];
        sumOfSquares += static_cast<int64_t>(samples[index]) * samples[index];
    }

    const double mean = static_cast<double>(sum) / sampleCount;
    const double meanOfSquares = static_cast<double>(sumOfSquares) / sampleCount;
    const double variance = meanOfSquares - mean * mean;
    return static_cast<float>(sqrt(variance > 0.0 ? variance : 0.0));
}

void reportCompletedSpeech()
{
    if (!speechCapture.consumeCompleted())
        return;

    Serial.printf("[SPEECH] samples=%u\n",
                  static_cast<unsigned>(speechCapture.getSampleCount()));
    Serial.printf("[SPEECH] duration=%lu ms%s\n",
                  static_cast<unsigned long>(speechCapture.getDurationMs()),
                  speechCapture.wasLimited() ? " (maximum reached)" : "");
}

void processMicrophoneSamples(const int32_t* samples, size_t sampleCount)
{
    if (sampleCount == 0)
        return;

    accumulateMicrophoneSamples(samples, sampleCount);
    speechCapture.pushSamples(samples, sampleCount);
    voiceActivityDetector.update(calculateRms(samples, sampleCount));

    if (voiceActivityDetector.didSpeechStart())
    {
        speechCapture.start();
        animator.setVisualState(Eyes::VisualState::Listening);
        Serial.println("[VAD] Speech started");
    }

    if (voiceActivityDetector.didSpeechEnd())
    {
        speechCapture.finish();
        animator.setVisualState(Eyes::VisualState::Idle);
        Serial.println("[VAD] Speech ended");
    }

    reportCompletedSpeech();
}

void reportMicrophoneLevel(uint32_t now)
{
    if (now - lastMicReportAt < MIC_REPORT_INTERVAL_MS || micSampleCount == 0)
        return;

    const double mean = static_cast<double>(micSum) / micSampleCount;
    const double meanOfSquares = static_cast<double>(micSumOfSquares) / micSampleCount;
    const uint32_t rms = static_cast<uint32_t>(sqrt(
        meanOfSquares > mean * mean ? meanOfSquares - mean * mean : 0.0));
    if (voiceActivityDetector.isSpeaking())
        Serial.printf("[VAD] Speaking | RMS=%lu\n", static_cast<unsigned long>(rms));
    else
        Serial.println("[VAD] Silence");

    micSumOfSquares = 0;
    micSum = 0;
    micSampleCount = 0;
    lastMicReportAt = now;
}
} // namespace

void setup()
{
    Serial.begin(115200);

    if (!display.begin())
    {
        Serial.println("Falha ao inicializar o display.");
        return;
    }

    displayReady = true;
    animator.begin();

    Serial.println("Inicializando INMP441...");
    if (!microphone.begin())
    {
        Serial.println("Falha ao inicializar INMP441.");
        return;
    }

    Serial.println("INMP441 inicializado.");
    voiceActivityDetector.begin();
    speechCapture.begin();
    lastMicReportAt = millis();
    visualTestStartedAt = lastMicReportAt;
}

void loop()
{
    if (!displayReady)
        return;

    if (microphone.available())
        processMicrophoneSamples(micSamples, microphone.readSamples(micSamples, MIC_BLOCK_SAMPLES));

    updateTemporaryVisualTest(millis());
    animator.update();
    reportMicrophoneLevel(millis());
}
