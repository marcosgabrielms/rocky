#pragma once

#include "../drivers/Display.h"

class Eyes
{
public:
    explicit Eyes(Display& display);

    void drawOpen();

    void drawClosed();

private:
    Display& display;
};