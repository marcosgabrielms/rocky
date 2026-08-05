#include "Display.h"

Display::Display()
    : oled(WIDTH, HEIGHT, &Wire, -1)
{
}

bool Display::begin()
{
    Wire.begin(8, 9);

    if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3C))
        return false;

    oled.clearDisplay();
    oled.display();

    return true;
}

void Display::clear()
{
    oled.clearDisplay();
}

void Display::show()
{
    oled.display();
}

void Display::drawCircle(int16_t x,
                         int16_t y,
                         int16_t r)
{
    oled.fillCircle(x, y, r, SSD1306_WHITE);
}

void Display::drawLine(int16_t x1,
                       int16_t y1,
                       int16_t x2,
                       int16_t y2)
{
    oled.drawLine(x1, y1, x2, y2, SSD1306_WHITE);
}

void Display::printText(int16_t x,
                        int16_t y,
                        const String& text)
{
    oled.setTextSize(1);
    oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(x, y);
    oled.print(text);
}