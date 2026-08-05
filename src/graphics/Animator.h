#pragma once

#include "Eyes.h"

class Animator
{
public:

    explicit Animator(Eyes& eyes);

    void begin();

    void update();

private:

    Eyes& eyes;

    bool closed = false;

    unsigned long lastBlink = 0;
};