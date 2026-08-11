#include "CommandProcessor.h"

CommandProcessor::Command CommandProcessor::identify(const String& text, String& normalized) const
{
    normalize(text, normalized);
    return normalized == "horas" ? Command::Hours : Command::Unknown;
}

void CommandProcessor::normalize(const String& text, String& normalized) const
{
    normalized = text;
    normalized.trim();
    normalized.toLowerCase();
    removeTrailingPunctuation(normalized);
    normalized.trim();
}

void CommandProcessor::removeTrailingPunctuation(String& text)
{
    while (!text.isEmpty())
    {
        const char character = text.charAt(text.length() - 1);
        if (character != '.' && character != '?' && character != '!' &&
            character != ',' && character != ';' && character != ':')
        {
            return;
        }

        text.remove(text.length() - 1);
    }
}
