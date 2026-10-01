#include <Arduino.h>
#include <esp_heap_caps.h>

#include <cstdlib>
#include <cstring>

#include "drivers/Display.h"
#include "drivers/Microphone.h"
#include "drivers/Speaker.h"
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
Speaker speaker;
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
constexpr bool MIC_DIAGNOSTIC_MODE = false;
constexpr bool VAD_TRIGGER_DIAGNOSTICS = false;
constexpr bool SPEAKER_TEST_TONE_ENABLED = false;
constexpr uint32_t MIC_FRAME_DURATION_MS = MIC_BLOCK_SAMPLES * 1000UL / SpeechCapture::SAMPLE_RATE;
constexpr uint32_t NOISE_WINDOW_MS = 750;
constexpr size_t NOISE_WINDOW_FRAMES = NOISE_WINDOW_MS / MIC_FRAME_DURATION_MS;
constexpr size_t CALIBRATION_COMMAND_MAX_LENGTH = 24;
constexpr uint32_t RAW_CAPTURE_DURATION_MS = 3000;

enum class InteractionState : uint8_t
{
    Idle,
    WaitingCommand,
    ProcessingCommand,
    Speaking,
    ShowingResponse
};

enum class DiagnosticCaptureMode : uint8_t
{
    None,
    Raw,
    Noise,
    Vad
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
uint32_t calibrationId = 0;
uint32_t calibrationDistanceCm = 0;
uint32_t calibrationFirstActiveAt = 0;
uint32_t calibrationTriggerDelayMs = 0;
float calibrationFirstActiveRms = 0.0F;
float calibrationTriggerRms = 0.0F;
float calibrationSpeechPeakRms = 0.0F;
double calibrationSpeechRmsSum = 0.0;
size_t calibrationSpeechFrameCount = 0;
float noiseFrameRms[NOISE_WINDOW_FRAMES]{};
double noiseRmsSum = 0.0;
size_t noiseFrameCount = 0;
size_t noiseFrameWriteIndex = 0;
char calibrationCommand[CALIBRATION_COMMAND_MAX_LENGTH]{};
size_t calibrationCommandLength = 0;
DiagnosticCaptureMode diagnosticCaptureMode = DiagnosticCaptureMode::None;
char calibrationVoiceLevel[7] = "normal";
char calibrationFanState[4] = "off";
bool diagnosticVadArmed = false;
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
void handleBackendResponse(const SttClient::BackendResponse& response, bool deferVisualState = false);
void beginSpeaking();
void updateSpeakingVisual();
void handleWakeWord(const String& text);
void handleCommand(const String& text);
void updateTimeScreen(uint32_t now);
void updateCommandWindow(uint32_t now);
void returnToIdle();
void setVisualState(Eyes::VisualState state, bool showConfirmation = true);
void showAttentionPrompt(uint32_t secondsRemaining);
void processDiagnosticSerial();
void updateNoiseMeasurement(float rms);
void updateCalibrationTrigger(float rms, uint32_t now);
void startCalibrationCapture(float triggerRms, uint32_t now);
void updateCalibrationSpeechMetrics(float rms);
void reportDiagnosticCapture();
void uploadDiagnosticCapture();
uint32_t getNoiseAverageRms();
uint32_t getNoisePeakRms();

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

void processDiagnosticSerial()
{
    while (Serial.available() > 0)
    {
        const char character = static_cast<char>(Serial.read());
        if (character == '\n' || character == '\r')
        {
            if (calibrationCommandLength == 0)
                continue;

            calibrationCommand[calibrationCommandLength] = '\0';
            constexpr char DISTANCE_PREFIX[] = "cal distance ";
            constexpr char LEVEL_PREFIX[] = "cal level ";
            constexpr char FAN_PREFIX[] = "cal fan ";
            const size_t distancePrefixLength = sizeof(DISTANCE_PREFIX) - 1;
            const size_t levelPrefixLength = sizeof(LEVEL_PREFIX) - 1;
            const size_t fanPrefixLength = sizeof(FAN_PREFIX) - 1;
            if (strncmp(calibrationCommand, DISTANCE_PREFIX, distancePrefixLength) == 0)
            {
                char* end = nullptr;
                const unsigned long distance = strtoul(calibrationCommand + distancePrefixLength, &end, 10);
                if (*end != '\0' || (distance != 20 && distance != 40 && distance != 60 && distance != 80))
                    Serial.println("[CAL] distance must be 20, 40, 60 or 80");
                else
                {
                    calibrationDistanceCm = static_cast<uint32_t>(distance);
                    Serial.printf("[CAL] distance_cm=%lu\n",
                                  static_cast<unsigned long>(calibrationDistanceCm));
                }
            }
            else if (strncmp(calibrationCommand, LEVEL_PREFIX, levelPrefixLength) == 0)
            {
                const char* const level = calibrationCommand + levelPrefixLength;
                if (strcmp(level, "low") != 0 && strcmp(level, "normal") != 0 && strcmp(level, "high") != 0)
                    Serial.println("[CAL] level must be low, normal or high");
                else
                {
                    strcpy(calibrationVoiceLevel, level);
                    Serial.printf("[CAL] voice_level=%s\n", calibrationVoiceLevel);
                }
            }
            else if (strncmp(calibrationCommand, FAN_PREFIX, fanPrefixLength) == 0)
            {
                const char* const fanState = calibrationCommand + fanPrefixLength;
                if (strcmp(fanState, "on") != 0 && strcmp(fanState, "off") != 0)
                    Serial.println("[CAL] fan must be on or off");
                else
                {
                    strcpy(calibrationFanState, fanState);
                    Serial.printf("[CAL] fan=%s\n", calibrationFanState);
                }
            }
            else if (strcmp(calibrationCommand, "cal raw") == 0)
            {
                if (calibrationDistanceCm == 0)
                    Serial.println("[CAL] set distance before arming capture");
                else if (speechCapture.isCapturing())
                    Serial.println("[CAL] capture already active");
                else
                {
                    diagnosticCaptureMode = DiagnosticCaptureMode::Raw;
                    diagnosticVadArmed = false;
                    ++calibrationId;
                    calibrationTriggerRms = 0.0F;
                    calibrationTriggerDelayMs = 0;
                    calibrationFirstActiveRms = 0.0F;
                    speechCapture.startFixedCapture(RAW_CAPTURE_DURATION_MS);
                    Serial.println("[CAL] RAW armed");
                    Serial.println("[CAL] speak now");
                }
            }
            else if (strcmp(calibrationCommand, "cal vad") == 0)
            {
                if (calibrationDistanceCm == 0)
                    Serial.println("[CAL] set distance before arming capture");
                else
                {
                    diagnosticCaptureMode = DiagnosticCaptureMode::Vad;
                    diagnosticVadArmed = true;
                    Serial.println("[CAL] VAD armed");
                    Serial.println("[CAL] speak now");
                }
            }
            else if (strcmp(calibrationCommand, "cal noise") == 0)
            {
                if (calibrationDistanceCm == 0)
                    Serial.println("[CAL] set distance before arming capture");
                else if (speechCapture.isCapturing())
                    Serial.println("[CAL] capture already active");
                else
                {
                    diagnosticCaptureMode = DiagnosticCaptureMode::Noise;
                    diagnosticVadArmed = false;
                    ++calibrationId;
                    calibrationTriggerRms = 0.0F;
                    calibrationTriggerDelayMs = 0;
                    calibrationFirstActiveRms = 0.0F;
                    speechCapture.startFixedCapture(RAW_CAPTURE_DURATION_MS);
                    Serial.println("[CAL] noise capture armed");
                    Serial.println("[CAL] remain silent");
                }
            }
            else if (strcmp(calibrationCommand, "cal status") == 0)
            {
                Serial.println("[CAL STATUS]");
                Serial.printf("distance_cm=%lu\n", static_cast<unsigned long>(calibrationDistanceCm));
                Serial.printf("voice_level=%s\n", calibrationVoiceLevel);
                Serial.printf("fan=%s\n", calibrationFanState);
                Serial.printf("sample_rate=%lu\n", static_cast<unsigned long>(SpeechCapture::SAMPLE_RATE));
                Serial.printf("start_threshold=%.0f\n", VoiceActivityDetector::THRESHOLD_ON);
                Serial.printf("end_threshold=%.0f\n", VoiceActivityDetector::THRESHOLD_OFF);
                Serial.printf("attack_ms=%lu\n", static_cast<unsigned long>(VoiceActivityDetector::ATTACK_TIME_MS));
                Serial.printf("release_ms=%lu\n", static_cast<unsigned long>(VoiceActivityDetector::RELEASE_TIME_MS));
                Serial.printf("pre_roll_ms=%lu\n", static_cast<unsigned long>(SpeechCapture::PRE_SPEECH_DURATION_MS));
                Serial.println("diagnostic_mode=on");
            }
            else
                Serial.println("[CAL] command invalid");
            calibrationCommandLength = 0;
            continue;
        }

        if (calibrationCommandLength < CALIBRATION_COMMAND_MAX_LENGTH - 1)
            calibrationCommand[calibrationCommandLength++] = character;
    }
}

void updateNoiseMeasurement(float rms)
{
    if (voiceActivityDetector.isSpeaking() || rms >= VoiceActivityDetector::THRESHOLD_ON)
        return;

    if (noiseFrameCount < NOISE_WINDOW_FRAMES)
        ++noiseFrameCount;
    else
        noiseRmsSum -= noiseFrameRms[noiseFrameWriteIndex];

    noiseFrameRms[noiseFrameWriteIndex] = rms;
    noiseRmsSum += rms;
    noiseFrameWriteIndex = (noiseFrameWriteIndex + 1) % NOISE_WINDOW_FRAMES;
}

uint32_t getNoiseAverageRms()
{
    return noiseFrameCount == 0 ? 0 : static_cast<uint32_t>(noiseRmsSum / noiseFrameCount);
}

uint32_t getNoisePeakRms()
{
    float peak = 0.0F;
    for (size_t index = 0; index < noiseFrameCount; ++index)
        peak = max(peak, noiseFrameRms[index]);
    return static_cast<uint32_t>(peak);
}

void updateCalibrationTrigger(float rms, uint32_t now)
{
    if (voiceActivityDetector.isSpeaking())
        return;

    if (rms < VoiceActivityDetector::THRESHOLD_ON)
    {
        calibrationFirstActiveAt = 0;
        calibrationFirstActiveRms = 0.0F;
        return;
    }

    if (calibrationFirstActiveAt == 0)
    {
        calibrationFirstActiveAt = now;
        calibrationFirstActiveRms = rms;
    }
}

void startCalibrationCapture(float triggerRms, uint32_t now)
{
    ++calibrationId;
    calibrationTriggerRms = triggerRms;
    calibrationTriggerDelayMs = calibrationFirstActiveAt == 0 ? 0 : now - calibrationFirstActiveAt;
    calibrationSpeechPeakRms = triggerRms;
    calibrationSpeechRmsSum = 0.0;
    calibrationSpeechFrameCount = 0;
    Serial.printf("[CAL] noise_avg_rms=%lu\n", static_cast<unsigned long>(getNoiseAverageRms()));
    Serial.printf("[CAL] noise_peak_rms=%lu\n", static_cast<unsigned long>(getNoisePeakRms()));
}

void updateCalibrationSpeechMetrics(float rms)
{
    if (!voiceActivityDetector.isSpeaking())
        return;

    calibrationSpeechPeakRms = max(calibrationSpeechPeakRms, rms);
    calibrationSpeechRmsSum += rms;
    ++calibrationSpeechFrameCount;
}

void reportDiagnosticCapture()
{
    const SpeechCapture::Pcm16Metrics& metrics = speechCapture.getMetrics();
    const SpeechCapture::Raw24Metrics& raw24Metrics = speechCapture.getRaw24Metrics();
    const uint32_t noiseRms = getNoiseAverageRms();
    const float speechAverageRms = calibrationSpeechFrameCount == 0
                                       ? 0.0F
                                       : static_cast<float>(calibrationSpeechRmsSum / calibrationSpeechFrameCount);

    Serial.printf("\n[CAL #%lu]\n", static_cast<unsigned long>(calibrationId));
    const char* const mode = diagnosticCaptureMode == DiagnosticCaptureMode::Raw ? "raw"
                             : diagnosticCaptureMode == DiagnosticCaptureMode::Noise ? "noise" : "vad";
    Serial.printf("mode=%s\n", mode);
    Serial.printf("distance_cm=%lu\n", static_cast<unsigned long>(calibrationDistanceCm));
    Serial.printf("voice_level=%s\n", calibrationVoiceLevel);
    Serial.printf("fan=%s\n", calibrationFanState);
    Serial.printf("duration_ms=%lu\n", static_cast<unsigned long>(speechCapture.getDurationMs()));
    Serial.printf("samples=%u\n", static_cast<unsigned>(speechCapture.getSampleCount()));
    Serial.printf("noise_rms=%lu\n", static_cast<unsigned long>(noiseRms));
    Serial.printf("noise_peak_rms=%lu\n", static_cast<unsigned long>(getNoisePeakRms()));
    Serial.printf("raw24_min=%ld\n", static_cast<long>(raw24Metrics.minimum));
    Serial.printf("raw24_max=%ld\n", static_cast<long>(raw24Metrics.maximum));
    Serial.printf("raw24_rms=%lu\n", static_cast<unsigned long>(raw24Metrics.rms));
    Serial.printf("pcm16_min=%d\n", metrics.minimum);
    Serial.printf("pcm16_max=%d\n", metrics.maximum);
    Serial.printf("pcm16_rms=%lu\n", static_cast<unsigned long>(metrics.rms));
    Serial.printf("peak_abs=%ld\n", static_cast<long>(metrics.peakAbsolute));
    Serial.printf("clipping=%u\n", static_cast<unsigned>(metrics.clippingCount));
    Serial.printf("snr_estimate_db=%.1f\n",
                  noiseRms == 0 ? 0.0F : 20.0F * log10f(static_cast<float>(raw24Metrics.rms) / noiseRms));
    if (diagnosticCaptureMode == DiagnosticCaptureMode::Vad)
    {
        Serial.printf("frame_before_trigger_rms=%.0f\n", calibrationFirstActiveRms);
        Serial.printf("vad_trigger_rms=%.0f\n", calibrationTriggerRms);
        Serial.printf("speech_peak_rms=%.0f\n", calibrationSpeechPeakRms);
        Serial.printf("speech_avg_rms=%.0f\n", speechAverageRms);
    }
    Serial.printf("vad_trigger_delay_ms=%lu\n", static_cast<unsigned long>(calibrationTriggerDelayMs));
    Serial.printf("pre_roll_ms=%lu\n", static_cast<unsigned long>(SpeechCapture::PRE_SPEECH_DURATION_MS));
    uploadDiagnosticCapture();
    diagnosticCaptureMode = DiagnosticCaptureMode::None;
    diagnosticVadArmed = false;
}

void uploadDiagnosticCapture()
{
    if (!sttServerAvailable)
    {
        Serial.println("[CAL] upload skipped=server_unavailable");
        return;
    }

    const SpeechCapture::Raw24Metrics& raw24Metrics = speechCapture.getRaw24Metrics();
    const SpeechCapture::Pcm16Metrics& pcm16Metrics = speechCapture.getMetrics();
    SttClient::CalibrationMetadata metadata;
    metadata.mode = diagnosticCaptureMode == DiagnosticCaptureMode::Raw ? "raw"
                    : diagnosticCaptureMode == DiagnosticCaptureMode::Noise ? "noise" : "vad";
    metadata.voiceLevel = calibrationVoiceLevel;
    metadata.fanState = calibrationFanState;
    metadata.distanceCm = calibrationDistanceCm;
    metadata.sampleId = calibrationId;
    metadata.noiseRms = getNoiseAverageRms();
    metadata.noisePeakRms = getNoisePeakRms();
    metadata.raw24Rms = raw24Metrics.rms;
    metadata.pcm16Rms = pcm16Metrics.rms;
    metadata.peakAbsolute = pcm16Metrics.peakAbsolute;
    metadata.clippingCount = pcm16Metrics.clippingCount;
    metadata.vadTriggerRms = static_cast<uint32_t>(calibrationTriggerRms);
    metadata.vadTriggerDelayMs = calibrationTriggerDelayMs;
    String filename;
    if (sttClient.uploadCalibration(speechCapture.getPcm16(), speechCapture.getPcmByteCount(), metadata, filename))
        Serial.printf("[CAL] saved=%s\n", filename.c_str());
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
    {
        returnToIdle();
        return;
    }

    reportCaptureMemory("before_stt");
    const uint32_t requestStartedAt = millis();
    const SttClient::PlaybackCallbacks playbackCallbacks{beginSpeaking, updateSpeakingVisual};
    SttClient::RealtimeResponse realtimeResponse;
    if (sttClient.transcribeRealtime(speechCapture.getPcm16(), speechCapture.getPcmByteCount(), realtimeResponse))
    {
        Serial.printf("[STT] elapsed_ms=%lu\n",
                      static_cast<unsigned long>(millis() - requestStartedAt));
        reportCaptureMemory("before_backend_parse");
        if (!realtimeResponse.realtime)
        {
            Serial.println("[RT] realtime=false");
            Serial.printf("[STT] text=\"%s\"\n", realtimeResponse.backendResponse.text.c_str());
            handleBackendResponse(realtimeResponse.backendResponse,
                                  realtimeResponse.backendResponse.audioAvailable);
            if (realtimeResponse.backendResponse.audioAvailable)
            {
                sttClient.playResponseAudio(speaker, playbackCallbacks);
                returnToIdle();
                if (showingTime)
                {
                    display.showTime(realtimeResponse.backendResponse.action.line2);
                    timeScreenStartedAt = millis();
                }
            }
        }
        else
        {
            Serial.printf("[RT] interaction=%lu\n", static_cast<unsigned long>(realtimeResponse.interactionId));
            if (!sttClient.playRealtimeInteraction(speaker, realtimeResponse.interactionId, playbackCallbacks))
                Serial.println("[RT] interaction_failed");
            returnToIdle();
        }
        reportCaptureMemory("after_backend_parse");
    }
    else
    {
        Serial.println("[BACKEND] invalid response");
        returnToIdle();
    }

    reportCaptureMemory("after_stt");
}

void beginSpeaking()
{
    if (interactionState == InteractionState::Speaking)
        return;

    interactionState = InteractionState::Speaking;
    visualCommandWindowActive = false;
    showingAttentionPrompt = false;
    setVisualState(Eyes::VisualState::Speaking, false);
}

void updateSpeakingVisual()
{
    if (interactionState == InteractionState::Speaking)
        animator.update();
}

void handleBackendResponse(const SttClient::BackendResponse& response, bool deferVisualState)
{
    Serial.printf("[BACKEND] state=%s\n", response.interactionState.c_str());

    if (response.action.type == SttClient::BackendAction::Type::Expression)
    {
        Serial.printf("[ACTION] expression=%s\n", response.action.value.c_str());
        if (response.action.value == "attention")
            setVisualState(Eyes::VisualState::Attention, false);
        else if (response.action.value == "idle")
        {
            if (!deferVisualState)
                setVisualState(Eyes::VisualState::Idle);
        }
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
    else if (response.interactionState == "idle")
    {
        interactionState = InteractionState::Idle;
        backendConversationActive = false;
        visualCommandWindowActive = false;
        showingAttentionPrompt = false;
        if (!deferVisualState && !showingTime)
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

    const auto stateName = [](Eyes::VisualState visualState) -> const char*
    {
        switch (visualState)
        {
        case Eyes::VisualState::Idle:
            return "IDLE";
        case Eyes::VisualState::Attention:
            return "ATTENTION";
        case Eyes::VisualState::Listening:
            return "LISTENING";
        case Eyes::VisualState::Thinking:
            return "THINKING";
        case Eyes::VisualState::Speaking:
            return "SPEAKING";
        }

        return "UNKNOWN";
    };
    const char* const previous = stateName(currentVisualState);
    const char* const next = stateName(state);
    currentVisualState = state;
    animator.setVisualState(state, showConfirmation);
    Serial.printf("[STATE] %s -> %s\n", previous, next);
}

void processMicrophoneSamples(const int32_t* samples, size_t sampleCount)
{
    if (sampleCount == 0)
        return;

    const float rms = calculateRms(samples, sampleCount);
    if constexpr (MIC_DIAGNOSTIC_MODE)
    {
        const uint32_t now = millis();
        updateNoiseMeasurement(rms);
        if (diagnosticCaptureMode == DiagnosticCaptureMode::Raw ||
            diagnosticCaptureMode == DiagnosticCaptureMode::Noise)
        {
            speechCapture.pushSamples(samples, sampleCount);
            if (reportCompletedSpeech())
                reportDiagnosticCapture();
            return;
        }
        updateCalibrationTrigger(rms, now);
    }

    accumulateMicrophoneSamples(samples, sampleCount);
    speechCapture.pushSamples(samples, sampleCount);
    voiceActivityDetector.update(rms);

    if (voiceActivityDetector.didSpeechStart())
    {
        if constexpr (VAD_TRIGGER_DIAGNOSTICS)
            Serial.printf("[VAD DIAG] trigger_raw24_rms=%.0f\n", rms);

        if constexpr (MIC_DIAGNOSTIC_MODE)
        {
            if (diagnosticCaptureMode != DiagnosticCaptureMode::Vad || !diagnosticVadArmed)
                return;

            ++calibrationId;
            startCalibrationCapture(rms, millis());
        }
        else if (interactionState == InteractionState::Idle || interactionState == InteractionState::WaitingCommand)
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
        if (!MIC_DIAGNOSTIC_MODE && interactionState == InteractionState::WaitingCommand)
            interactionState = InteractionState::ProcessingCommand;
        if constexpr (!MIC_DIAGNOSTIC_MODE)
            setVisualState(Eyes::VisualState::Listening);
    }

    if (voiceActivityDetector.didSpeechEnd())
    {
        speechCapture.finish();
        Serial.println("[VAD] SPEECH -> SILENCE");
        if constexpr (MIC_DIAGNOSTIC_MODE)
        {
            if (diagnosticCaptureMode != DiagnosticCaptureMode::Vad || !diagnosticVadArmed)
                return;
        }
    }

    if constexpr (MIC_DIAGNOSTIC_MODE)
        updateCalibrationSpeechMetrics(rms);

    if (reportCompletedSpeech())
    {
        if constexpr (MIC_DIAGNOSTIC_MODE)
            reportDiagnosticCapture();
        else
        {
            setVisualState(Eyes::VisualState::Thinking, false);
            sendCompletedSpeech();
        }
    }
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
    if (speaker.begin() && SPEAKER_TEST_TONE_ENABLED)
        speaker.playTestTone();

    voiceActivityDetector.begin();
    if (!speechCapture.begin())
        Serial.println("Falha ao alocar buffer de captura na PSRAM.");
    else
        Serial.println("Buffer de captura PCM16 alocado na PSRAM.");

    wifiManager.begin();
    lastMicReportAt = millis();
    Serial.println("[SYSTEM] Rocky ready");
    if constexpr (MIC_DIAGNOSTIC_MODE)
    {
        Serial.println("[CAL] diagnostic mode enabled");
        Serial.printf("[CAL] threshold_on=%.0f threshold_off=%.0f\n",
                      VoiceActivityDetector::THRESHOLD_ON,
                      VoiceActivityDetector::THRESHOLD_OFF);
        Serial.printf("[CAL] attack_ms=%lu release_ms=%lu frame_ms=%lu pre_roll_ms=%lu\n",
                      static_cast<unsigned long>(VoiceActivityDetector::ATTACK_TIME_MS),
                      static_cast<unsigned long>(VoiceActivityDetector::RELEASE_TIME_MS),
                      static_cast<unsigned long>(MIC_FRAME_DURATION_MS),
                      static_cast<unsigned long>(SpeechCapture::PRE_SPEECH_DURATION_MS));
        Serial.println("[CAL] commands: cal distance, cal level, cal fan, cal raw, cal vad, cal noise, cal status");
    }
}

void loop()
{
    if (!displayReady)
        return;

    if constexpr (MIC_DIAGNOSTIC_MODE)
        processDiagnosticSerial();
    else
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

    if (!MIC_DIAGNOSTIC_MODE && !datasetCollector.isActive())
        timeService.update();
    updateTimeScreen(millis());
    updateCommandWindow(millis());
    if (!showingTime && !showingAttentionPrompt)
        animator.update();
    reportMicrophoneLevel(millis());
}
