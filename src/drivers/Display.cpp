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

void Display::showAttentionPrompt(uint32_t secondsRemaining)
{
    constexpr char GREETING[] = "Oi?";
    const String waiting = String("Aguardando... ") + secondsRemaining + "s";

    clear();
    oled.setTextColor(SSD1306_WHITE);
    oled.setTextSize(2);

    int16_t greetingX = 0;
    int16_t greetingY = 0;
    uint16_t greetingWidth = 0;
    uint16_t greetingHeight = 0;
    oled.getTextBounds(GREETING, 0, 0, &greetingX, &greetingY, &greetingWidth, &greetingHeight);
    oled.setCursor((WIDTH - greetingWidth) / 2, 8);
    oled.print(GREETING);

    oled.setTextSize(1);
    int16_t waitingX = 0;
    int16_t waitingY = 0;
    uint16_t waitingWidth = 0;
    uint16_t waitingHeight = 0;
    oled.getTextBounds(waiting, 0, 0, &waitingX, &waitingY, &waitingWidth, &waitingHeight);
    oled.setCursor((WIDTH - waitingWidth) / 2, 40);
    oled.print(waiting);
    show();
}

void Display::showTime(const String& time)
{
    constexpr char LABEL[] = "Agora sao";

    clear();
    oled.setTextColor(SSD1306_WHITE);
    oled.setTextSize(1);
    int16_t labelX = 0;
    int16_t labelY = 0;
    uint16_t labelWidth = 0;
    uint16_t labelHeight = 0;
    oled.getTextBounds(LABEL, 0, 0, &labelX, &labelY, &labelWidth, &labelHeight);
    oled.setCursor((WIDTH - labelWidth) / 2, 8);
    oled.print(LABEL);

    oled.setTextSize(3);
    int16_t timeX = 0;
    int16_t timeY = 0;
    uint16_t timeWidth = 0;
    uint16_t timeHeight = 0;
    oled.getTextBounds(time, 0, 0, &timeX, &timeY, &timeWidth, &timeHeight);
    oled.setCursor((WIDTH - timeWidth) / 2, 29);
    oled.print(time);
    show();
}
