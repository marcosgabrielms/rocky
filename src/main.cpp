#include <Arduino.h>
#include <esp_heap_caps.h>

#include "drivers/Display.h"
#include "drivers/Microphone.h"
#include "core/CommandProcessor.h"
#include "core/TimeService.h"
#include "dataset/DatasetCollector.h"
#include "audio/SpeechCapture.h"
#include "audio/VoiceActivityDetector.h"
#include "graphics/Eyes.h"
#include "graphics/Animator.h"
#include "network/SttClient.h"
#include "network/WifiManager.h"

Display display;
Microphone microphone;
VoiceActivityDetector voiceActivityDetector;
SpeechCapture speechCapture;
Eyes eyes(display);
Animator animator(eyes);
CommandProcessor commandProcessor;
SttClient sttClient;
TimeService timeService;
DatasetCollector datasetCollector;
WifiManager wifiManager;
bool displayReady = false;

namespace
{
constexpr size_t MIC_BLOCK_SAMPLES = 64;
constexpr uint32_t MIC_REPORT_INTERVAL_MS = 250;
constexpr uint32_t HEALTH_CHECK_DELAY_MS = 500;
constexpr uint32_t TIME_SCREEN_DURATION_MS = 3000;
constexpr uint32_t COMMAND_WINDOW_DURATION_MS = 6000;
constexpr bool SERIAL_VERBOSE_AUDIO = false;

enum class InteractionState : uint8_t
{
    Idle,
    WaitingCommand,
    ProcessingCommand,
    ShowingResponse
};

int32_t micSamples[MIC_BLOCK_SAMPLES];
uint64_t micSumOfSquares = 0;
int64_t micSum = 0;
size_t micSampleCount = 0;
uint32_t lastMicReportAt = 0;
uint32_t wifiConnectedAt = 0;
uint32_t timeScreenStartedAt = 0;
uint32_t visualCommandWindowStartedAt = 0;
uint32_t visualCommandWindowDurationMs = 0;
uint32_t displayedCommandWindowSeconds = 0;
uint32_t interactionId = 0;
bool healthCheckAttempted = false;
bool sttServerAvailable = false;
bool timeSyncStarted = false;
bool showingTime = false;
bool backendConversationActive = false;
bool visualCommandWindowActive = false;
bool showingAttentionPrompt = false;
InteractionState interactionState = InteractionState::Idle;
Eyes::VisualState currentVisualState = Eyes::VisualState::Idle;

void handleTranscription(const String& text);
void handleBackendResponse(const SttClient::BackendResponse& response);
void handleWakeWord(const String& text);
void handleCommand(const String& text);
void updateTimeScreen(uint32_t now);
void updateCommandWindow(uint32_t now);
void returnToIdle();
void setVisualState(Eyes::VisualState state, bool showConfirmation = true);
void showAttentionPrompt(uint32_t secondsRemaining);

void reportBootMemory()
{
    Serial.println("[MEMORY]");
    Serial.printf("psram_found=%s\n", psramFound() ? "yes" : "no");
    Serial.printf("flash_size=%u\n", static_cast<unsigned>(ESP.getFlashChipSize()));
    Serial.printf("psram_size=%u\n", static_cast<unsigned>(ESP.getPsramSize()));
    Serial.printf("psram_free=%u\n", static_cast<unsigned>(ESP.getFreePsram()));
}

void reportCaptureMemory(const char* stage)
{
    const size_t internalFree = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t psramFree = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    Serial.printf("[MEM] %s internal=%u psram=%u\n",
                  stage,
                  static_cast<unsigned>(internalFree),
                  static_cast<unsigned>(psramFree));
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

bool reportCompletedSpeech()
{
    if (!speechCapture.consumeCompleted())
        return false;

    const SpeechCapture::Pcm16Metrics& metrics = speechCapture.getMetrics();

    Serial.println("[SPEECH]");
    Serial.printf("samples=%u\n",
                  static_cast<unsigned>(speechCapture.getSampleCount()));
    Serial.printf("duration_ms=%lu%s\n",
                  static_cast<unsigned long>(speechCapture.getDurationMs()),
                  speechCapture.wasLimited() ? " (maximum reached)" : "");
    Serial.printf("pcm_bytes=%u\n", static_cast<unsigned>(speechCapture.getPcmByteCount()));
    Serial.println("format=PCM16 mono 16000Hz");
    Serial.printf("min=%d max=%d mean=%ld rms=%lu clipping=%u\n",
                  metrics.minimum,
                  metrics.maximum,
                  static_cast<long>(metrics.mean),
                  static_cast<unsigned long>(metrics.rms),
                  static_cast<unsigned>(metrics.clippingCount));
    Serial.printf("[AUDIO] captured duration_ms=%lu\n",
                  static_cast<unsigned long>(speechCapture.getDurationMs()));
    reportCaptureMemory("after_capture");
    return true;
}

void sendCompletedSpeech()
{
    if (datasetCollector.isActive())
    {
        Serial.printf("[DATASET] samples=%u\n", static_cast<unsigned>(speechCapture.getSampleCount()));
        Serial.printf("[DATASET] duration_ms=%lu\n", static_cast<unsigned long>(speechCapture.getDurationMs()));
        Serial.println("[DATASET] sending...");
        uint32_t index = 0;
        if (sttClient.uploadDataset(speechCapture.getPcm16(), speechCapture.getPcmByteCount(), datasetCollector.getLabel(), index))
        {
            Serial.printf("[DATASET] saved index=%lu\n", static_cast<unsigned long>(index));
            datasetCollector.recordSaved();
        }
        else
            datasetCollector.recordFailed();
        return;
    }

    if (!sttServerAvailable)
        return;

    reportCaptureMemory("before_stt");
    const uint32_t requestStartedAt = millis();
    SttClient::BackendResponse response;
    if (sttClient.transcribe(speechCapture.getPcm16(), speechCapture.getPcmByteCount(), response))
    {
        Serial.printf("[STT] text=\"%s\"\n", response.text.c_str());
        Serial.printf("[STT] elapsed_ms=%lu\n",
                      static_cast<unsigned long>(millis() - requestStartedAt));
        reportCaptureMemory("before_backend_parse");
        handleBackendResponse(response);
        reportCaptureMemory("after_backend_parse");
    }
    else
    {
        Serial.println("[BACKEND] invalid response");
        if (interactionState == InteractionState::ProcessingCommand)
            returnToIdle();
    }

    reportCaptureMemory("after_stt");
}

void handleBackendResponse(const SttClient::BackendResponse& response)
{
    Serial.printf("[BACKEND] state=%s\n", response.interactionState.c_str());

    if (response.action.type == SttClient::BackendAction::Type::Expression)
    {
        Serial.printf("[ACTION] expression=%s\n", response.action.value.c_str());
        if (response.action.value == "attention")
            setVisualState(Eyes::VisualState::Attention, false);
        else if (response.action.value == "idle")
            setVisualState(Eyes::VisualState::Idle);
        else
            Serial.printf("[ACTION] unsupported=%s\n", response.action.value.c_str());
    }
    else if (response.action.type == SttClient::BackendAction::Type::ShowText)
    {
        if (response.action.line1 == "Agora sao")
        {
            visualCommandWindowActive = false;
            showingAttentionPrompt = false;
            display.showTime(response.action.line2);
            showingTime = true;
            interactionState = InteractionState::ShowingResponse;
            timeScreenStartedAt = millis();
            Serial.println("[BACKEND] action=show_text");
            Serial.println("[DISPLAY] show_text start");
        }
        else
            Serial.println("[ACTION] unsupported=show_text");
    }
    else if (!response.action.value.isEmpty())
        Serial.printf("[ACTION] unsupported=%s\n", response.action.value.c_str());

    if (response.interactionState == "attention")
    {
        interactionState = InteractionState::WaitingCommand;
        backendConversationActive = true;
        if (response.commandWindowMs > 0)
        {
            visualCommandWindowActive = true;
            visualCommandWindowStartedAt = millis();
            visualCommandWindowDurationMs = response.commandWindowMs;
            displayedCommandWindowSeconds = 0;
            Serial.printf("[BACKEND] command_window_ms=%lu\n", static_cast<unsigned long>(response.commandWindowMs));
            Serial.printf("[BODY] command window started: %lums\n",
                          static_cast<unsigned long>(response.commandWindowMs));
            showAttentionPrompt((response.commandWindowMs + 999) / 1000);
        }
        setVisualState(Eyes::VisualState::Attention, false);
    }
    else if (response.interactionState == "idle" && !voiceActivityDetector.isSpeaking())
    {
        interactionState = InteractionState::Idle;
        backendConversationActive = false;
        visualCommandWindowActive = false;
        showingAttentionPrompt = false;
        setVisualState(Eyes::VisualState::Idle);
    }
    else if (response.interactionState != "idle")
        Serial.printf("[BACKEND] unsupported state=%s\n", response.interactionState.c_str());
}

void handleTranscription(const String& text)
{
    if (interactionState == InteractionState::Idle)
    {
        handleWakeWord(text);
        return;
    }

    if (interactionState == InteractionState::ProcessingCommand)
        handleCommand(text);
}

void handleWakeWord(const String& text)
{
    String normalized;
    commandProcessor.normalize(text, normalized);
    Serial.printf("[WAKE] text=\"%s\"\n", text.c_str());
    Serial.printf("[WAKE] normalized=\"%s\"\n", normalized.c_str());

    if (normalized != "rocky")
    {
        Serial.printf("[WAKE] ignored=\"%s\"\n", normalized.c_str());
        return;
    }

    Serial.println("[WAKE] matched");
    interactionState = InteractionState::WaitingCommand;
    setVisualState(Eyes::VisualState::Attention, false);
    Serial.println("[WAKE] command window started: 6000ms");
}

void handleCommand(const String& text)
{
    String normalized;
    const CommandProcessor::Command command = commandProcessor.identify(text, normalized);
    Serial.printf("[COMMAND] normalized=\"%s\"\n", normalized.c_str());

    if (command != CommandProcessor::Command::Hours)
    {
        Serial.println("[COMMAND] unknown");
        returnToIdle();
        return;
    }

    Serial.println("[COMMAND] matched=hours");
    if (!timeService.isSynchronized())
    {
        Serial.println("[TIME] unavailable");
        returnToIdle();
        return;
    }

    const String currentTime = timeService.getCurrentTime();
    if (currentTime.isEmpty())
    {
        Serial.println("[TIME] unavailable");
        returnToIdle();
        return;
    }

    Serial.printf("[TIME] current=%s\n", currentTime.c_str());
    display.showTime(currentTime);
    showingTime = true;
    interactionState = InteractionState::ShowingResponse;
    timeScreenStartedAt = millis();
    Serial.println("[DISPLAY] showing time");
}

void updateTimeScreen(uint32_t now)
{
    if (!showingTime || now - timeScreenStartedAt < TIME_SCREEN_DURATION_MS)
        return;

    showingTime = false;
    Serial.println("[DISPLAY] show_text end");
    returnToIdle();
}

void updateCommandWindow(uint32_t now)
{
    if (!visualCommandWindowActive)
        return;

    const uint32_t elapsedMs = now - visualCommandWindowStartedAt;
    if (elapsedMs >= visualCommandWindowDurationMs)
    {
        visualCommandWindowActive = false;
        showingAttentionPrompt = false;
        Serial.println("[BODY] command window timeout");
        setVisualState(Eyes::VisualState::Idle);
        return;
    }

    const uint32_t remainingMs = visualCommandWindowDurationMs - elapsedMs;
    showAttentionPrompt((remainingMs + 999) / 1000);
}

void showAttentionPrompt(uint32_t secondsRemaining)
{
    if (showingAttentionPrompt && displayedCommandWindowSeconds == secondsRemaining)
        return;

    display.showAttentionPrompt(secondsRemaining);
    showingAttentionPrompt = true;
    displayedCommandWindowSeconds = secondsRemaining;
    Serial.printf("[DISPLAY] attention prompt: %lus\n", static_cast<unsigned long>(secondsRemaining));
}

void returnToIdle()
{
    interactionState = InteractionState::Idle;
    showingAttentionPrompt = false;
    visualCommandWindowActive = false;
    setVisualState(Eyes::VisualState::Idle);
}

void setVisualState(Eyes::VisualState state, bool showConfirmation)
{
    if (currentVisualState == state)
        return;

    const char* const previous = currentVisualState == Eyes::VisualState::Idle
                                     ? "IDLE"
                                     : currentVisualState == Eyes::VisualState::Attention ? "ATTENTION" : "LISTENING";
    const char* const next = state == Eyes::VisualState::Idle
                                 ? "IDLE"
                                 : state == Eyes::VisualState::Attention ? "ATTENTION" : "LISTENING";
    currentVisualState = state;
    animator.setVisualState(state, showConfirmation);
    Serial.printf("[STATE] %s -> %s\n", previous, next);
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
        if (interactionState == InteractionState::Idle || interactionState == InteractionState::WaitingCommand)
        {
            ++interactionId;
            Serial.printf("\n[INTERACTION #%lu]\n", static_cast<unsigned long>(interactionId));
        }
        reportCaptureMemory("before_capture");
        speechCapture.start();
        if (visualCommandWindowActive)
        {
            visualCommandWindowActive = false;
            showingAttentionPrompt = false;
            Serial.println("[BODY] command window cancelled by speech");
        }
        Serial.println("[VAD] SILENCE -> SPEECH");
        if (interactionState == InteractionState::WaitingCommand)
            interactionState = InteractionState::ProcessingCommand;
        setVisualState(Eyes::VisualState::Listening);
    }

    if (voiceActivityDetector.didSpeechEnd())
    {
        speechCapture.finish();
        Serial.println("[VAD] SPEECH -> SILENCE");
        if (interactionState == InteractionState::ProcessingCommand)
            setVisualState(Eyes::VisualState::Attention, false);
        else
            setVisualState(Eyes::VisualState::Idle);
    }

    if (reportCompletedSpeech())
        sendCompletedSpeech();
}

void reportMicrophoneLevel(uint32_t now)
{
    if constexpr (!SERIAL_VERBOSE_AUDIO)
        return;

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
    reportBootMemory();

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
    if (!speechCapture.begin())
        Serial.println("Falha ao alocar buffer de captura na PSRAM.");
    else
        Serial.println("Buffer de captura PCM16 alocado na PSRAM.");

    wifiManager.begin();
    lastMicReportAt = millis();
    Serial.println("[SYSTEM] Rocky ready");
}

void loop()
{
    if (!displayReady)
        return;

    datasetCollector.update();
    if (!showingTime && microphone.available())
        processMicrophoneSamples(micSamples, microphone.readSamples(micSamples, MIC_BLOCK_SAMPLES));

    wifiManager.update();
    if (!healthCheckAttempted && wifiManager.isConnected())
    {
        if (!timeSyncStarted)
        {
            timeService.begin();
            timeSyncStarted = true;
        }

        if (wifiConnectedAt == 0)
            wifiConnectedAt = millis();
        else if (millis() - wifiConnectedAt >= HEALTH_CHECK_DELAY_MS)
        {
            healthCheckAttempted = true;
            sttServerAvailable = sttClient.checkHealth();
            Serial.println(sttServerAvailable ? "[STT] server available" : "[STT] server unavailable");
        }
    }

    if (!datasetCollector.isActive())
        timeService.update();
    updateTimeScreen(millis());
    updateCommandWindow(millis());
    if (!showingTime && !showingAttentionPrompt)
        animator.update();
    reportMicrophoneLevel(millis());
}
