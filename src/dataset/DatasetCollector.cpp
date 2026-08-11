#include "DatasetCollector.h"

#include <cstring>

void DatasetCollector::update()
{
    while (Serial.available() > 0)
    {
        const char character = static_cast<char>(Serial.read());
        if (character == '\n' || character == '\r')
        {
            if (commandLength > 0)
                processCommand();
            commandLength = 0;
            continue;
        }

        if (commandLength < COMMAND_MAX_LENGTH - 1)
            command[commandLength++] = character;
    }
}

bool DatasetCollector::isActive() const
{
    return active;
}

const char* DatasetCollector::getLabel() const
{
    return "rocky";
}

void DatasetCollector::recordSaved()
{
    ++collectedCount;
    Serial.printf("[DATASET] rocky count=%lu\n", static_cast<unsigned long>(collectedCount));
}

void DatasetCollector::recordFailed()
{
    Serial.println("[DATASET] upload failed");
}

void DatasetCollector::processCommand()
{
    command[commandLength] = '\0';
    if (strcmp(command, "dataset rocky start") == 0)
        startRocky();
    else if (strcmp(command, "dataset stop") == 0)
        stop();
}

void DatasetCollector::startRocky()
{
    active = true;
    collectedCount = 0;
    Serial.println("[DATASET] mode=rocky");
    Serial.println("[DATASET] ready");
}

void DatasetCollector::stop()
{
    if (!active)
        return;

    active = false;
    Serial.println("[DATASET] stopped");
    Serial.printf("[DATASET] collected=%lu\n", static_cast<unsigned long>(collectedCount));
}
