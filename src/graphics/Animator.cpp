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
    if (visualState == Eyes::VisualState::Speaking)
        updateSpeaking(now);
    else
    {
        updateBlink(now);
        updateGaze(now);
    }
    eyes.update(now);
}

void Animator::setVisualState(Eyes::VisualState newState, bool showConfirmation)
{
    if (visualState == newState)
        return;

    visualState = newState;
    eyes.setVisualState(newState, showConfirmation);

    if (visualState == Eyes::VisualState::Attention ||
        visualState == Eyes::VisualState::Listening ||
        visualState == Eyes::VisualState::Thinking ||
        visualState == Eyes::VisualState::Speaking)
    {
        eyes.lookCenter();

        if (visualState == Eyes::VisualState::Attention)
            attentionGazeIndex = 0;
    }

    lastGazeChangeAt = millis();
    if (visualState == Eyes::VisualState::Speaking)
    {
        speakingPoseIndex = 0;
        lastSpeakingPoseAt = lastGazeChangeAt;
        blinkState = BlinkState::Waiting;
        stateStartedAt = lastGazeChangeAt;
        eyes.setBlinking(false);
    }
    eyes.render();
}

void Animator::updateSpeaking(uint32_t now)
{
    if (now - lastSpeakingPoseAt < SPEAKING_POSE_INTERVAL_MS)
        return;

    constexpr int16_t HEIGHT_OFFSETS[] = {0, -3, 0, 3};
    speakingPoseIndex = (speakingPoseIndex + 1) % 4;
    eyes.setSpeakingHeightOffset(HEIGHT_OFFSETS[speakingPoseIndex]);
    lastSpeakingPoseAt = now;
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

    if (visualState == Eyes::VisualState::Listening)
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
