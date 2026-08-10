#include "Animator.h"

Animator::Animator(Eyes& eyes)
    : eyes(eyes)
{
}

void Animator::begin()
{
    const uint32_t now = millis();
    stateStartedAt = now;
    lastGazeChangeAt = now;
    eyes.begin(now);
}

void Animator::update()
{
    const uint32_t now = millis();
    updateBlink(now);
    updateGaze(now);
    eyes.update(now);
}

void Animator::setVisualState(Eyes::VisualState newState)
{
    if (visualState == newState)
        return;

    visualState = newState;
    eyes.setVisualState(newState);

    if (visualState == Eyes::VisualState::Attention)
    {
        attentionGazeIndex = 0;
        eyes.lookCenter();
    }

    lastGazeChangeAt = millis();
}

void Animator::updateBlink(uint32_t now)
{
    const uint32_t elapsed = now - stateStartedAt;

    if (blinkState == BlinkState::Waiting && elapsed >= BLINK_INTERVAL_MS)
    {
        eyes.setBlinking(true);
        blinkState = BlinkState::Closing;
        stateStartedAt = now;
        return;
    }

    if (blinkState == BlinkState::Closing && elapsed >= BLINK_CLOSE_MS)
    {
        eyes.setBlinking(false);
        blinkState = BlinkState::Opening;
        stateStartedAt = now;
        return;
    }

    if (blinkState == BlinkState::Opening && elapsed >= BLINK_OPEN_MS)
    {
        blinkState = BlinkState::Waiting;
        stateStartedAt = now;
    }
}

void Animator::updateGaze(uint32_t now)
{
    if (visualState == Eyes::VisualState::Attention && eyes.isAttentionConfirming())
        return;

    const uint32_t interval = visualState == Eyes::VisualState::Attention
                                  ? ATTENTION_GAZE_INTERVAL_MS
                                  : GAZE_INTERVAL_MS;

    if (now - lastGazeChangeAt < interval)
        return;

    if (visualState == Eyes::VisualState::Attention)
        applyNextAttentionGaze();
    else
        applyNextGaze();

    lastGazeChangeAt = now;
}

void Animator::applyNextGaze()
{
    switch (gazeIndex++ % 5)
    {
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

void Animator::applyNextAttentionGaze()
{
    switch (attentionGazeIndex++ % 5)
    {
    case 0:
        eyes.setGazeOffset(-2, 0);
        break;
    case 1:
        eyes.setGazeOffset(1, -1);
        break;
    case 2:
        eyes.setGazeOffset(2, 0);
        break;
    case 3:
        eyes.setGazeOffset(-1, 1);
        break;
    default:
        eyes.lookCenter();
        break;
    }
}
