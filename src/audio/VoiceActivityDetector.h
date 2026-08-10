#pragma once

#include <Arduino.h>

class VoiceActivityDetector
{
public:
    // Ajuste estes quatro valores durante a calibração no hardware.
    static constexpr float THRESHOLD_ON = 12000.0F;
    static constexpr float THRESHOLD_OFF = 8000.0F;
    static constexpr uint32_t ATTACK_TIME_MS = 150;
    static constexpr uint32_t RELEASE_TIME_MS = 450;

    void begin();
    bool update(float rms);

    bool isSpeaking() const;
    bool didSpeechStart();
    bool didSpeechEnd();
    float getLevel() const;

private:
    enum class State : uint8_t
    {
        Silence,
        SpeechStarting,
        Speaking,
        SpeechEnding
    };

    float level = 0.0F;
    uint32_t stateStartedAt = 0;
    State state = State::Silence;
    bool speechStarted = false;
    bool speechEnded = false;
};
