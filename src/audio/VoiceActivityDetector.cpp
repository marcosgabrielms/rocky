#include "VoiceActivityDetector.h"

void VoiceActivityDetector::begin()
{
    level = 0.0F;
    stateStartedAt = millis();
    state = State::Silence;
    speechStarted = false;
    speechEnded = false;
}

bool VoiceActivityDetector::update(float rms)
{
    level = rms;
    const uint32_t now = millis();
    speechStarted = false;
    speechEnded = false;

    switch (state)
    {
    case State::Silence:
        if (level < THRESHOLD_ON)
            break;

        state = State::SpeechStarting;
        stateStartedAt = now;
        break;

    case State::SpeechStarting:
        if (level < THRESHOLD_ON)
        {
            state = State::Silence;
            break;
        }

        if (now - stateStartedAt >= ATTACK_TIME_MS)
        {
            state = State::Speaking;
            speechStarted = true;
        }
        break;

    case State::Speaking:
        if (level > THRESHOLD_OFF)
            break;

        state = State::SpeechEnding;
        stateStartedAt = now;
        break;

    case State::SpeechEnding:
        if (level > THRESHOLD_OFF)
        {
            state = State::Speaking;
            break;
        }

        if (now - stateStartedAt >= RELEASE_TIME_MS)
        {
            state = State::Silence;
            speechEnded = true;
        }
        break;
    }

    return isSpeaking();
}

bool VoiceActivityDetector::isSpeaking() const
{
    return state == State::Speaking || state == State::SpeechEnding;
}

bool VoiceActivityDetector::didSpeechStart()
{
    const bool event = speechStarted;
    speechStarted = false;
    return event;
}

bool VoiceActivityDetector::didSpeechEnd()
{
    const bool event = speechEnded;
    speechEnded = false;
    return event;
}

float VoiceActivityDetector::getLevel() const
{
    return level;
}
