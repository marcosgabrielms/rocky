#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

class Display
{
public:
    Display();

    bool begin();

    void clear();

    void show();

    void drawCircle(int16_t x, int16_t y, int16_t r);

    void drawLine(int16_t x1,
                  int16_t y1,
                  int16_t x2,
                  int16_t y2);

    void printText(int16_t x,
                   int16_t y,
                   const String& text);

private:
    static constexpr uint8_t WIDTH = 128;
    static constexpr uint8_t HEIGHT = 64;

    Adafruit_SSD1306 oled;
};