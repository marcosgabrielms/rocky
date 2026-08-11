#pragma once

#include <Arduino.h>

class CommandProcessor
{
public:
    enum class Command : uint8_t
    {
        Unknown,
        Hours
    };

    void normalize(const String& text, String& normalized) const;
    Command identify(const String& text, String& normalized) const;

private:
    static void removeTrailingPunctuation(String& text);
};
