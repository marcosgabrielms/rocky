#include "Eyes.h"

Eyes::Eyes(Display& display)
    : display(display)
{
}

void Eyes::drawOpen()
{
    display.clear();

    display.printText(30, 10, "ROCKY");

    display.drawCircle(40, 45, 8);
    display.drawCircle(88, 45, 8);

    display.show();
}

void Eyes::drawClosed()
{
    display.clear();

    display.printText(30, 10, "ROCKY");

    display.drawLine(30, 45, 50, 45);
    display.drawLine(78, 45, 98, 45);

    display.show();
}