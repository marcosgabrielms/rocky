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
    if (visualState == Eyes::VisualState::Idle)
    {
        updateBlink(now);
        updateIdleGaze(now);
    }
    else if (visualState == Eyes::VisualState::Attention)
        updateAttention(now);
    else if (visualState == Eyes::VisualState::Listening)
        updateListening(now);
    else if (visualState == Eyes::VisualState::Thinking)
        updateThinking(now);
    else if (visualState == Eyes::VisualState::Speaking)
        updateSpeaking(now);
    eyes.update(now);
}

void Animator::setVisualState(Eyes::VisualState newState, bool showConfirmation)
{
    if (visualState == newState)
    {
        eyes.render();
        return;
    }

    visualState = newState;
    eyes.setVisualState(newState, showConfirmation);

    const uint32_t now = millis();
    blinkState = BlinkState::Waiting;
    stateStartedAt = now;
    lastGazeChangeAt = now;
    lastPulseAt = now;
    eyes.setBlinking(false);
    eyes.setHeightOffset(0);
    eyes.lookCenter();

    attentionPulseIndex = 1;
    listeningPulseIndex = 1;

    if (visualState == Eyes::VisualState::Speaking)
    {
        speakingPoseIndex = 0;
        lastSpeakingPoseAt = lastGazeChangeAt;
    }
    eyes.render();
}

void Animator::updateSpeaking(uint32_t now)
{
    if (now - lastSpeakingPoseAt < SPEAKING_POSE_INTERVAL_MS)
        return;

    constexpr int16_t HEIGHT_OFFSETS[] = {0, -3, 0, 3};
    speakingPoseIndex = (speakingPoseIndex + 1) % 4;
    eyes.setHeightOffset(HEIGHT_OFFSETS[speakingPoseIndex]);
    lastSpeakingPoseAt = now;
}

void Animator::updateAttention(uint32_t now)
{
    if (now - lastPulseAt < ATTENTION_PULSE_STEP_MS)
        return;

    applyAttentionPulse();
    lastPulseAt = now;
}

void Animator::updateListening(uint32_t now)
{
    if (now - lastPulseAt < LISTENING_PULSE_STEP_MS)
        return;

    applyListeningPulse();
    lastPulseAt = now;
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

void Animator::updateIdleGaze(uint32_t now)
{
    if (now - lastGazeChangeAt < IDLE_GAZE_INTERVAL_MS)
        return;

    applyNextIdleGaze();
    lastGazeChangeAt = now;
}

void Animator::updateThinking(uint32_t now)
{
    if (now - lastGazeChangeAt < THINKING_GAZE_INTERVAL_MS)
        return;

    applyNextThinkingGaze();
    lastGazeChangeAt = now;
}

void Animator::applyNextIdleGaze()
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

void Animator::applyNextThinkingGaze()
{
    switch (gazeIndex++ % 4)
    {
    case 0:
        eyes.setGazeOffset(-4, -1);
        break;
    case 1:
        eyes.setGazeOffset(2, -1);
        break;
    case 2:
        eyes.setGazeOffset(4, 1);
        break;
    default:
        eyes.lookCenter();
        break;
    }
}

void Animator::applyAttentionPulse()
{
    constexpr int16_t HEIGHT_OFFSETS[] = {0, 2, 4, 2};
    eyes.setHeightOffset(HEIGHT_OFFSETS[attentionPulseIndex++ % 4]);
}

void Animator::applyListeningPulse()
{
    constexpr int16_t HEIGHT_OFFSETS[] = {0, 3};
    eyes.setHeightOffset(HEIGHT_OFFSETS[listeningPulseIndex++ % 2]);
}
