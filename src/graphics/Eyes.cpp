#include "Eyes.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr int16_t LISTENING_EYE_SIZE = 36;
constexpr int16_t LISTENING_EYE_RADIUS = 4;
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

    if (attentionConfirming && now - attentionStartedAt >= ATTENTION_CONFIRMATION_DURATION_MS)
    {
        attentionConfirming = false;
        redrawRequested = true;
    }

    const bool isAnimating = gazeX != targetGazeX || gazeY != targetGazeY ||
                             blinkAmount != targetBlinkAmount;

    if ((!redrawRequested && !isAnimating) || now - lastDraw < FRAME_INTERVAL_MS)
        return;

    draw();
    lastDraw = now;
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
    if (visualState == newState)
        return;

    visualState = newState;
    attentionStartedAt = millis();
    attentionConfirming = newState == VisualState::Attention && showConfirmation;

    if (newState == VisualState::Attention)
        lookCenter();

    redrawRequested = true;
}

void Eyes::setMood(Mood newMood)
{
    mood = newMood;
    redrawRequested = true;
}

bool Eyes::isAttentionConfirming() const
{
    return attentionConfirming;
}

void Eyes::setGaze(int16_t x, int16_t y)
{
    targetGazeX = static_cast<float>(x);
    targetGazeY = static_cast<float>(y);
    redrawRequested = true;
}

void Eyes::draw() const
{
    if (attentionConfirming)
    {
        display.clear();
        drawAttentionConfirmation();
        display.show();
        return;
    }

    const bool isListening = visualState == VisualState::Listening;
    const int16_t width = isListening ? LISTENING_EYE_SIZE : EYE_WIDTH;
    const int16_t baseHeight = baseEyeHeight();
    const int16_t height = std::max(
        MIN_EYE_HEIGHT,
        static_cast<int16_t>(std::lround(baseHeight * (1.0F - blinkAmount))));
    const int16_t y = EYE_CENTER_Y + static_cast<int16_t>(std::lround(gazeY)) - height / 2;
    const int16_t leftX = LEFT_EYE_CENTER_X + static_cast<int16_t>(std::lround(gazeX)) - width / 2;
    const int16_t rightX = RIGHT_EYE_CENTER_X + static_cast<int16_t>(std::lround(gazeX)) - width / 2;
    const int16_t radius = isListening ? LISTENING_EYE_RADIUS : cornerRadius(height);

    display.clear();
    display.fillRoundRect(leftX, y, width, height, radius);
    display.fillRoundRect(rightX, y, width, height, radius);
    display.show();
}

void Eyes::drawAttentionConfirmation() const
{
    constexpr int16_t CARET_HALF_WIDTH = 8;
    constexpr int16_t CARET_HEIGHT = 8;

    const int16_t apexY = EYE_CENTER_Y - CARET_HEIGHT / 2;
    const int16_t baseY = apexY + CARET_HEIGHT;

    display.drawLine(LEFT_EYE_CENTER_X - CARET_HALF_WIDTH, baseY,
                     LEFT_EYE_CENTER_X, apexY);
    display.drawLine(LEFT_EYE_CENTER_X, apexY,
                     LEFT_EYE_CENTER_X + CARET_HALF_WIDTH, baseY);
    display.drawLine(RIGHT_EYE_CENTER_X - CARET_HALF_WIDTH, baseY,
                     RIGHT_EYE_CENTER_X, apexY);
    display.drawLine(RIGHT_EYE_CENTER_X, apexY,
                     RIGHT_EYE_CENTER_X + CARET_HALF_WIDTH, baseY);
}

int16_t Eyes::baseEyeHeight() const
{
    if (visualState == VisualState::Listening)
        return LISTENING_EYE_SIZE;

    if (visualState == VisualState::Attention)
        return ATTENTION_EYE_HEIGHT;

    return EYE_HEIGHT;
}

float Eyes::approach(float current, float target, float maximumStep)
{
    if (current < target)
        return std::min(current + maximumStep, target);

    return std::max(current - maximumStep, target);
}

int16_t Eyes::cornerRadius(int16_t height) const
{
    const int16_t neutralRadius = std::min(EYE_WIDTH / 2, height / 2);

    if (mood == Mood::Happy)
        return std::max<int16_t>(2, neutralRadius / 2);

    return neutralRadius;
}
