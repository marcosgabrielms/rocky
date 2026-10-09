#include "Eyes.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr Eyes::EyeConfig IDLE_EYE{0, 0, 38, 40, 10};
constexpr Eyes::EyeConfig ATTENTION_EYE{0, 0, 34, 50, 10};
constexpr Eyes::EyeConfig LISTENING_EYE{0, 0, 42, 24, 8};
constexpr Eyes::EyeConfig THINKING_LEFT_EYE{0, -2, 30, 16, 8};
constexpr Eyes::EyeConfig THINKING_RIGHT_EYE{0, 2, 30, 16, 8};
constexpr Eyes::EyeConfig SPEAKING_EYE{0, 0, 38, 34, 10};

constexpr Eyes::FacePreset IDLE_PRESET{IDLE_EYE, IDLE_EYE, true, true, false};
constexpr Eyes::FacePreset ATTENTION_PRESET{ATTENTION_EYE, ATTENTION_EYE, false, false, false};
constexpr Eyes::FacePreset LISTENING_PRESET{LISTENING_EYE, LISTENING_EYE, false, false, false};
constexpr Eyes::FacePreset THINKING_PRESET{THINKING_LEFT_EYE, THINKING_RIGHT_EYE, true, false, false};
constexpr Eyes::FacePreset SPEAKING_PRESET{SPEAKING_EYE, SPEAKING_EYE, false, false, true};
}

Eyes::Eyes(Display& display)
    : display(display)
{
}

void Eyes::begin(uint32_t now)
{
    lastUpdate = now;
    lastDraw = now;
    draw();
    redrawRequested = false;
}

void Eyes::update(uint32_t now)
{
    const float elapsedSeconds = static_cast<float>(now - lastUpdate) / 1000.0F;
    lastUpdate = now;

    gazeX = approach(gazeX, targetGazeX, GAZE_SPEED * elapsedSeconds);
    gazeY = approach(gazeY, targetGazeY, GAZE_SPEED * elapsedSeconds);
    blinkAmount = approach(blinkAmount, targetBlinkAmount, BLINK_SPEED * elapsedSeconds);
    heightOffset = approach(heightOffset, targetHeightOffset, HEIGHT_OFFSET_SPEED * elapsedSeconds);

    const bool isAnimating = gazeX != targetGazeX || gazeY != targetGazeY ||
                             blinkAmount != targetBlinkAmount ||
                             heightOffset != targetHeightOffset;

    if ((!redrawRequested && !isAnimating) || now - lastDraw < FRAME_INTERVAL_MS)
        return;

    draw();
    lastDraw = now;
    redrawRequested = false;
}

void Eyes::render()
{
    draw();
    lastDraw = millis();
    redrawRequested = false;
}

void Eyes::lookLeft()
{
    setGaze(-MAX_GAZE_X, 0);
}

void Eyes::lookRight()
{
    setGaze(MAX_GAZE_X, 0);
}

void Eyes::lookUp()
{
    setGaze(0, -MAX_GAZE_Y);
}

void Eyes::lookDown()
{
    setGaze(0, MAX_GAZE_Y);
}

void Eyes::lookCenter()
{
    setGaze(0, 0);
}

void Eyes::setGazeOffset(int16_t x, int16_t y)
{
    const int16_t limitedX = std::max<int16_t>(
        -MAX_GAZE_X,
        std::min<int16_t>(x, MAX_GAZE_X));
    const int16_t limitedY = std::max<int16_t>(
        -MAX_GAZE_Y,
        std::min<int16_t>(y, MAX_GAZE_Y));
    setGaze(limitedX, limitedY);
}

void Eyes::setBlinking(bool blinking)
{
    targetBlinkAmount = blinking ? 1.0F : 0.0F;
    redrawRequested = true;
}

void Eyes::setVisualState(VisualState newState, bool showConfirmation)
{
    static_cast<void>(showConfirmation);

    if (visualState == newState)
        return;

    visualState = newState;
    heightOffset = 0.0F;
    targetHeightOffset = 0.0F;
    const FacePreset& preset = presetFor(newState);
    if (!preset.acceptsBlink)
    {
        blinkAmount = 0.0F;
        targetBlinkAmount = 0.0F;
    }
    if (!preset.acceptsGaze)
    {
        lookCenter();
        gazeX = 0.0F;
        gazeY = 0.0F;
    }

    redrawRequested = true;
}

void Eyes::setMood(Mood newMood)
{
    mood = newMood;
    redrawRequested = true;
}

void Eyes::setHeightOffset(int16_t offset)
{
    targetHeightOffset = static_cast<float>(offset);
    redrawRequested = true;
}

void Eyes::setGaze(int16_t x, int16_t y)
{
    targetGazeX = static_cast<float>(x);
    targetGazeY = static_cast<float>(y);
    redrawRequested = true;
}

void Eyes::draw() const
{
    const FacePreset& preset = presetFor(visualState);
    const float drawGazeX = preset.acceptsGaze ? gazeX : 0.0F;
    const float drawGazeY = preset.acceptsGaze ? gazeY : 0.0F;
    const float drawBlink = preset.acceptsBlink ? blinkAmount : 0.0F;

    display.clear();
    drawEye(preset.left, LEFT_EYE_CENTER_X, drawGazeX, drawGazeY, drawBlink);
    drawEye(preset.right, RIGHT_EYE_CENTER_X, drawGazeX, drawGazeY, drawBlink);
    display.show();
}

const Eyes::FacePreset& Eyes::presetFor(VisualState state)
{
    switch (state)
    {
    case VisualState::Idle:
        return IDLE_PRESET;
    case VisualState::Attention:
        return ATTENTION_PRESET;
    case VisualState::Listening:
        return LISTENING_PRESET;
    case VisualState::Thinking:
        return THINKING_PRESET;
    case VisualState::Speaking:
        return SPEAKING_PRESET;
    }

    return IDLE_PRESET;
}

void Eyes::drawEye(const EyeConfig& config, int16_t centerX, float gazeOffsetX,
                   float gazeOffsetY, float blink) const
{
    const FacePreset& preset = presetFor(visualState);
    const int16_t height = eyeHeight(preset, config, blink);
    const int16_t x = centerX + config.offsetX + static_cast<int16_t>(std::lround(gazeOffsetX)) - config.width / 2;
    const int16_t y = EYE_CENTER_Y + config.offsetY + static_cast<int16_t>(std::lround(gazeOffsetY)) - height / 2;
    display.fillRoundRect(x, y, config.width, height, resolveRadius(config, height));
}

int16_t Eyes::eyeHeight(const FacePreset& preset, const EyeConfig& config, float blink) const
{
    int16_t height = config.height;
    if (preset.acceptsHeightOffset)
        height += static_cast<int16_t>(std::lround(heightOffset));
    if (preset.acceptsBlink)
        height = static_cast<int16_t>(std::lround(height * (1.0F - blink)));

    return std::max(MIN_EYE_HEIGHT, height);
}

int16_t Eyes::resolveRadius(const EyeConfig& config, int16_t height) const
{
    const int16_t radius = config.radius == AUTO_RADIUS
                               ? cornerRadius(height, config.width)
                               : config.radius;
    const int16_t maximumRadius = std::min<int16_t>(config.width / 2, height / 2);
    return std::min(radius, maximumRadius);
}

float Eyes::approach(float current, float target, float maximumStep)
{
    if (current < target)
        return std::min(current + maximumStep, target);

    return std::max(current - maximumStep, target);
}

int16_t Eyes::cornerRadius(int16_t height, int16_t width) const
{
    const int16_t neutralRadius = std::min(width / 2, height / 2);

    if (mood == Mood::Happy)
        return std::max<int16_t>(2, neutralRadius / 2);

    return neutralRadius;
}
