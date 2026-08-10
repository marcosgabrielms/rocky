#include "Display.h"

Display::Display()
    : oled(WIDTH, HEIGHT, &Wire, -1)
{
}

bool Display::begin()
{
    Wire.begin(SDA_PIN, SCL_PIN);

    if (!oled.begin(SSD1306_SWITCHCAPVCC, I2C_ADDRESS))
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

void Display::fillRoundRect(int16_t x,
                            int16_t y,
                            int16_t width,
                            int16_t height,
                            int16_t radius)
{
    oled.fillRoundRect(x, y, width, height, radius, SSD1306_WHITE);
}

void Display::drawLine(int16_t x1, int16_t y1, int16_t x2, int16_t y2)
{
    oled.drawLine(x1, y1, x2, y2, SSD1306_WHITE);
}
