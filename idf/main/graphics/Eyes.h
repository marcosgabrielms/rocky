#pragma once

#include <cstdint>

#include "drivers/Display.h"

class Eyes {
public:
    enum class VisualState : uint8_t {
        Idle,
        Listening,
    };

    enum class Mood : uint8_t {
        Neutral,
        Happy,
        Angry,
    };

    explicit Eyes(Display &display);

    void begin(uint64_t now);
    void update(uint64_t now);

    void lookLeft();
    void lookRight();
    void lookUp();
    void lookDown();
    void lookCenter();
    void setBlinking(bool blinking);
    void setVisualState(VisualState state);
    void setMood(Mood mood);

private:
    static constexpr int16_t LEFT_EYE_CENTER_X = 38;
    static constexpr int16_t RIGHT_EYE_CENTER_X = 90;
    static constexpr int16_t EYE_CENTER_Y = 32;
    static constexpr int16_t EYE_WIDTH = 36;
    static constexpr int16_t EYE_HEIGHT = 42;
    static constexpr int16_t LISTENING_EYE_HEIGHT = 28;
    static constexpr int16_t MIN_EYE_HEIGHT = 3;
    static constexpr int16_t MAX_GAZE_X = 7;
    static constexpr int16_t MAX_GAZE_Y = 5;
    static constexpr float GAZE_SPEED = 38.0F;
    static constexpr float BLINK_SPEED = 8.5F;
    static constexpr uint64_t FRAME_INTERVAL_MS = 16;

    Display &display;
    VisualState visualState = VisualState::Idle;
    Mood mood = Mood::Neutral;
    float gazeX = 0.0F;
    float gazeY = 0.0F;
    float targetGazeX = 0.0F;
    float targetGazeY = 0.0F;
    float blinkAmount = 0.0F;
    float targetBlinkAmount = 0.0F;
    uint64_t lastUpdate = 0;
    uint64_t lastDraw = 0;
    bool redrawRequested = true;

    void setGaze(int16_t x, int16_t y);
    void draw() const;
    static float approach(float current, float target, float maximumStep);
    int16_t cornerRadius(int16_t height) const;
};
