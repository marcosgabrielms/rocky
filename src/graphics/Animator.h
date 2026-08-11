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
    static constexpr uint32_t GAZE_INTERVAL_MS = 2200;
    static constexpr uint32_t ATTENTION_GAZE_INTERVAL_MS = 1100;

    Eyes& eyes;
    Eyes::VisualState visualState = Eyes::VisualState::Idle;
    BlinkState blinkState = BlinkState::Waiting;
    uint32_t stateStartedAt = 0;
    uint32_t lastGazeChangeAt = 0;
    uint8_t gazeIndex = 0;
    uint8_t attentionGazeIndex = 0;

    void updateBlink(uint32_t now);
    void updateGaze(uint32_t now);
    void applyNextGaze();
    void applyNextAttentionGaze();
};
