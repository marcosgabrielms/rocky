#pragma once

#include <Arduino.h>

#include "../drivers/Display.h"

class Eyes
{
public:
    enum class VisualState : uint8_t
    {
        Idle,
        Attention,
        Listening,
        Thinking,
        Speaking
    };

    enum class Mood : uint8_t
    {
        Neutral,
        Happy,
        Angry
    };

    explicit Eyes(Display& display);

    void begin(uint32_t now);
    void update(uint32_t now);
    void render();

    void lookLeft();
    void lookRight();
    void lookUp();
    void lookDown();
    void lookCenter();
    void setGazeOffset(int16_t x, int16_t y);
    void setBlinking(bool blinking);
    void setVisualState(VisualState state, bool showConfirmation = true);
    void setMood(Mood mood);
    void setHeightOffset(int16_t offset);

    struct EyeConfig
    {
        int16_t offsetX;
        int16_t offsetY;
        int16_t width;
        int16_t height;
        int16_t radius;
    };

    struct FacePreset
    {
        EyeConfig left;
        EyeConfig right;
        bool acceptsGaze;
        bool acceptsBlink;
        bool acceptsHeightOffset;
    };

private:
    static constexpr int16_t LEFT_EYE_CENTER_X = 38;
    static constexpr int16_t RIGHT_EYE_CENTER_X = 90;
    static constexpr int16_t EYE_CENTER_Y = 32;
    static constexpr float HEIGHT_OFFSET_SPEED = 24.0F;
    static constexpr int16_t MIN_EYE_HEIGHT = 3;
    static constexpr int16_t MAX_GAZE_X = 7;
    static constexpr int16_t MAX_GAZE_Y = 5;
    static constexpr float GAZE_SPEED = 38.0F;
    static constexpr float BLINK_SPEED = 8.5F;
    static constexpr uint32_t FRAME_INTERVAL_MS = 16;
    static constexpr int16_t AUTO_RADIUS = -1;

    Display& display;
    VisualState visualState = VisualState::Idle;
    Mood mood = Mood::Neutral;
    float gazeX = 0.0F;
    float gazeY = 0.0F;
    float targetGazeX = 0.0F;
    float targetGazeY = 0.0F;
    float blinkAmount = 0.0F;
    float targetBlinkAmount = 0.0F;
    float heightOffset = 0.0F;
    float targetHeightOffset = 0.0F;
    uint32_t lastUpdate = 0;
    uint32_t lastDraw = 0;
    bool redrawRequested = true;

    void setGaze(int16_t x, int16_t y);
    void draw() const;
    static const FacePreset& presetFor(VisualState state);
    void drawEye(const EyeConfig& config, int16_t centerX, float gazeOffsetX,
                 float gazeOffsetY, float blink) const;
    int16_t eyeHeight(const FacePreset& preset, const EyeConfig& config, float blink) const;
    int16_t resolveRadius(const EyeConfig& config, int16_t height) const;
    static float approach(float current, float target, float maximumStep);
    int16_t cornerRadius(int16_t height, int16_t width) const;
};
