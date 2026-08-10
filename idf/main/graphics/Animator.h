#pragma once

#include <cstdint>

#include "graphics/Eyes.h"

class Animator {
public:
    explicit Animator(Eyes &eyes);

    void begin(uint64_t now);
    void update(uint64_t now);

private:
    enum class BlinkState : uint8_t {
        Waiting,
        Closing,
        Opening,
    };

    static constexpr uint64_t BLINK_INTERVAL_MS = 3200;
    static constexpr uint64_t BLINK_CLOSE_MS = 120;
    static constexpr uint64_t BLINK_OPEN_MS = 130;
    static constexpr uint64_t GAZE_INTERVAL_MS = 2200;

    Eyes &eyes;
    BlinkState blinkState = BlinkState::Waiting;
    uint64_t stateStartedAt = 0;
    uint64_t lastGazeChangeAt = 0;
    uint8_t gazeIndex = 0;

    void updateBlink(uint64_t now);
    void updateGaze(uint64_t now);
    void applyNextGaze();
};
