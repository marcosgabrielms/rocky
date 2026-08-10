#include "graphics/Eyes.h"

#include <algorithm>
#include <cmath>

Eyes::Eyes(Display &display)
    : display(display) {
}

void Eyes::begin(uint64_t now) {
    lastUpdate = now;
    lastDraw = now;
    draw();
    redrawRequested = false;
}

void Eyes::update(uint64_t now) {
    const float elapsedSeconds = static_cast<float>(now - lastUpdate) / 1000.0F;
    lastUpdate = now;

    gazeX = approach(gazeX, targetGazeX, GAZE_SPEED * elapsedSeconds);
    gazeY = approach(gazeY, targetGazeY, GAZE_SPEED * elapsedSeconds);
    blinkAmount = approach(blinkAmount, targetBlinkAmount, BLINK_SPEED * elapsedSeconds);

    const bool isAnimating = gazeX != targetGazeX || gazeY != targetGazeY ||
                             blinkAmount != targetBlinkAmount;
    if ((!redrawRequested && !isAnimating) || now - lastDraw < FRAME_INTERVAL_MS) {
        return;
    }

    draw();
    lastDraw = now;
    redrawRequested = false;
}

void Eyes::lookLeft() {
    setGaze(-MAX_GAZE_X, 0);
}

void Eyes::lookRight() {
    setGaze(MAX_GAZE_X, 0);
}

void Eyes::lookUp() {
    setGaze(0, -MAX_GAZE_Y);
}

void Eyes::lookDown() {
    setGaze(0, MAX_GAZE_Y);
}

void Eyes::lookCenter() {
    setGaze(0, 0);
}

void Eyes::setBlinking(bool blinking) {
    targetBlinkAmount = blinking ? 1.0F : 0.0F;
    redrawRequested = true;
}

void Eyes::setVisualState(VisualState newState) {
    if (visualState == newState) {
        return;
    }
    visualState = newState;
    redrawRequested = true;
}

void Eyes::setMood(Mood newMood) {
    mood = newMood;
    redrawRequested = true;
}

void Eyes::setGaze(int16_t x, int16_t y) {
    targetGazeX = static_cast<float>(x);
    targetGazeY = static_cast<float>(y);
    redrawRequested = true;
}

void Eyes::draw() const {
    const int16_t baseEyeHeight = visualState == VisualState::Listening
                                      ? LISTENING_EYE_HEIGHT
                                      : EYE_HEIGHT;
    const int16_t height = std::max(
        MIN_EYE_HEIGHT,
        static_cast<int16_t>(std::lround(baseEyeHeight * (1.0F - blinkAmount))));
    const int16_t y = EYE_CENTER_Y + static_cast<int16_t>(std::lround(gazeY)) - height / 2;
    const int16_t leftX = LEFT_EYE_CENTER_X + static_cast<int16_t>(std::lround(gazeX)) - EYE_WIDTH / 2;
    const int16_t rightX = RIGHT_EYE_CENTER_X + static_cast<int16_t>(std::lround(gazeX)) - EYE_WIDTH / 2;
    const int16_t radius = cornerRadius(height);

    display.clear();
    display.fillRoundRect(leftX, y, EYE_WIDTH, height, radius);
    display.fillRoundRect(rightX, y, EYE_WIDTH, height, radius);
    display.show();
}

float Eyes::approach(float current, float target, float maximumStep) {
    if (current < target) {
        return std::min(current + maximumStep, target);
    }
    return std::max(current - maximumStep, target);
}

int16_t Eyes::cornerRadius(int16_t height) const {
    const int16_t neutralRadius = std::min(EYE_WIDTH / 2, height / 2);
    if (mood == Mood::Happy) {
        return std::max<int16_t>(2, neutralRadius / 2);
    }
    return neutralRadius;
}
