#include "Animator.h"

Animator::Animator(Eyes& eyes)
    : eyes(eyes)
{
}

void Animator::begin()
{
    eyes.drawOpen();

    lastBlink = millis();
}

void Animator::update()
{
    unsigned long now = millis();

    if (!closed && now - lastBlink >= 3000)
    {
        eyes.drawClosed();

        closed = true;

        lastBlink = now;
    }

    if (closed && now - lastBlink >= 150)
    {
        eyes.drawOpen();

        closed = false;

        lastBlink = now;
    }
}