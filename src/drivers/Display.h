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

    void fillRoundRect(int16_t x,
                       int16_t y,
                       int16_t width,
                       int16_t height,
                       int16_t radius);

    void drawLine(int16_t x1, int16_t y1, int16_t x2, int16_t y2);

private:

    static constexpr uint8_t WIDTH = 128;
    static constexpr uint8_t HEIGHT = 64;
    static constexpr uint8_t SDA_PIN = 8;
    static constexpr uint8_t SCL_PIN = 9;
    static constexpr uint8_t I2C_ADDRESS = 0x3C;

    Adafruit_SSD1306 oled;
};
