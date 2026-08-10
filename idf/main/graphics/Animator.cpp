#include "graphics/Animator.h"

Animator::Animator(Eyes &eyes)
    : eyes(eyes) {
}

void Animator::begin(uint64_t now) {
    stateStartedAt = now;
    lastGazeChangeAt = now;
    eyes.begin(now);
}

void Animator::update(uint64_t now) {
    updateBlink(now);
    updateGaze(now);
    eyes.update(now);
}

void Animator::updateBlink(uint64_t now) {
    const uint64_t elapsed = now - stateStartedAt;
    if (blinkState == BlinkState::Waiting && elapsed >= BLINK_INTERVAL_MS) {
        eyes.setBlinking(true);
        blinkState = BlinkState::Closing;
        stateStartedAt = now;
        return;
    }
    if (blinkState == BlinkState::Closing && elapsed >= BLINK_CLOSE_MS) {
        eyes.setBlinking(false);
        blinkState = BlinkState::Opening;
        stateStartedAt = now;
        return;
    }
    if (blinkState == BlinkState::Opening && elapsed >= BLINK_OPEN_MS) {
        blinkState = BlinkState::Waiting;
        stateStartedAt = now;
    }
}

void Animator::updateGaze(uint64_t now) {
    if (now - lastGazeChangeAt < GAZE_INTERVAL_MS) {
        return;
    }
    applyNextGaze();
    lastGazeChangeAt = now;
}

void Animator::applyNextGaze() {
    switch (gazeIndex++ % 5) {
    case 0:
        eyes.lookLeft();
        break;
    case 1:
        eyes.lookUp();
        break;
    case 2:
        eyes.lookRight();
        break;
    case 3:
        eyes.lookDown();
        break;
    default:
        eyes.lookCenter();
        break;
    }
}
