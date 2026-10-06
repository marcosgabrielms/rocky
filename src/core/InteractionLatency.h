#pragma once

#include <Arduino.h>

class InteractionLatency
{
public:
    void begin(uint32_t interaction, uint32_t now)
    {
        interactionId = interaction;
        active = true;
        captureFinished = false;
        hasSilenceCandidate = false;
        hasSpeechEnd = false;
        playbackStarted = false;
        attentionShown = false;
        event("speech_start_ms", now);
    }

    void observeLevel(float rms, float endThreshold, uint32_t now)
    {
        if (!active || captureFinished)
            return;
        if (rms > endThreshold)
            hasSilenceCandidate = false;
        else if (!hasSilenceCandidate)
        {
            silenceCandidateAt = now;
            hasSilenceCandidate = true;
        }
    }

    void speechEnded(uint32_t now)
    {
        if (!active || captureFinished)
            return;
        hasSpeechEnd = true;
        speechEndAt = hasSilenceCandidate ? silenceCandidateAt : now;
        // Estimativa pelo threshold; não é uma medição acústica do fim da palavra.
        event("speech_end_ms", speechEndAt);
        event("speech_end_confirmed_ms", now);
        duration("vad_release_observed_ms", now - speechEndAt);
    }

    void captureDone(bool limited, uint32_t now)
    {
        if (!active || captureFinished)
            return;
        captureFinished = true;
        captureDoneAt = now;
        event("capture_done_ms", now);
        Serial.printf("[VAD] end_reason=%s\n", limited ? "max_duration" : "silence");
        if (limited)
            ++maxDurationCount;
        else
            ++silenceCount;
        Serial.printf("[KODA LATENCY] interaction=%lu captures_silence=%lu captures_max_duration=%lu\n",
                      static_cast<unsigned long>(interactionId),
                      static_cast<unsigned long>(silenceCount), static_cast<unsigned long>(maxDurationCount));
        if (!hasSpeechEnd)
            Serial.println("[KODA LATENCY] speech_end_ms=unavailable end_reference=capture_done");
    }

    void requestStarted(uint32_t now)
    {
        requestStartedAt = now;
        event("stt_request_start_ms", now);
        duration("capture_done_to_request_ms", now - captureDoneAt);
    }

    void responseReceived(uint32_t now)
    {
        event("http_response_ms", now);
        duration("http_roundtrip_ms", now - requestStartedAt);
        reportEndTo("http_response", now);
    }

    void attention(uint32_t now)
    {
        if (!active || attentionShown)
            return;
        attentionShown = true;
        event("attention_ms", now);
        reportEndTo("attention", now);
    }

    void playback(uint32_t now)
    {
        if (!active || playbackStarted)
            return;
        playbackStarted = true;
        event("playback_start_ms", now);
        reportEndTo("playback_start", now);
    }

    void finish() { active = false; }

private:
    uint32_t interactionId = 0;
    uint32_t silenceCandidateAt = 0;
    uint32_t speechEndAt = 0;
    uint32_t captureDoneAt = 0;
    uint32_t requestStartedAt = 0;
    uint32_t silenceCount = 0;
    uint32_t maxDurationCount = 0;
    bool active = false;
    bool captureFinished = false;
    bool hasSilenceCandidate = false;
    bool hasSpeechEnd = false;
    bool playbackStarted = false;
    bool attentionShown = false;

    void event(const char* name, uint32_t value) const
    {
        Serial.printf("[KODA LATENCY] clock=firmware_boot interaction=%lu kind=event %s=%lu\n",
                      static_cast<unsigned long>(interactionId), name, static_cast<unsigned long>(value));
    }

    void duration(const char* name, uint32_t value) const
    {
        Serial.printf("[KODA LATENCY] interaction=%lu kind=duration %s=%lu\n",
                      static_cast<unsigned long>(interactionId), name, static_cast<unsigned long>(value));
    }

    void reportEndTo(const char* target, uint32_t now) const
    {
        Serial.printf("[KODA LATENCY] interaction=%lu kind=duration %s_to_%s_ms=%lu\n",
                      static_cast<unsigned long>(interactionId), hasSpeechEnd ? "speech_end" : "capture_done",
                      target, static_cast<unsigned long>(now - (hasSpeechEnd ? speechEndAt : captureDoneAt)));
    }
};
