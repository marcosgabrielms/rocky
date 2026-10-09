#pragma once

#include <Arduino.h>

#include "Eyes.h"

class Animator
{
public:
    explicit Animator(Eyes& eyes);

    void begin();
    void update();
    void setVisualState(Eyes::VisualState state, bool showConfirmation = true);

private:
    enum class BlinkState : uint8_t
    {
        Waiting,
        Closing,
        Opening
    };

    static constexpr uint32_t BLINK_INTERVAL_MS = 3200;
    static constexpr uint32_t BLINK_CLOSE_MS = 120;
    static constexpr uint32_t BLINK_OPEN_MS = 130;
    static constexpr uint32_t IDLE_GAZE_INTERVAL_MS = 2200;
    static constexpr uint32_t THINKING_GAZE_INTERVAL_MS = 900;
    static constexpr uint32_t ATTENTION_PULSE_STEP_MS = 175;
    static constexpr uint32_t LISTENING_PULSE_STEP_MS = 100;
    static constexpr uint32_t SPEAKING_POSE_INTERVAL_MS = 220;

    Eyes& eyes;
    Eyes::VisualState visualState = Eyes::VisualState::Idle;
    BlinkState blinkState = BlinkState::Waiting;
    uint32_t stateStartedAt = 0;
    uint32_t lastGazeChangeAt = 0;
    uint8_t gazeIndex = 0;
    uint8_t attentionPulseIndex = 0;
    uint8_t listeningPulseIndex = 0;
    uint8_t speakingPoseIndex = 0;
    uint32_t lastPulseAt = 0;
    uint32_t lastSpeakingPoseAt = 0;

    void updateBlink(uint32_t now);
    void updateIdleGaze(uint32_t now);
    void updateThinking(uint32_t now);
    void updateAttention(uint32_t now);
    void updateListening(uint32_t now);
    void updateSpeaking(uint32_t now);
    void applyAttentionPulse();
    void applyListeningPulse();
    void applyNextIdleGaze();
    void applyNextThinkingGaze();
};
