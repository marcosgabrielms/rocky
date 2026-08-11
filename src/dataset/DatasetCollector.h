#pragma once

#include <Arduino.h>

class DatasetCollector
{
public:
    void update();
    bool isActive() const;
    const char* getLabel() const;
    void recordSaved();
    void recordFailed();

private:
    static constexpr size_t COMMAND_MAX_LENGTH = 32;

    char command[COMMAND_MAX_LENGTH]{};
    size_t commandLength = 0;
    bool active = false;
    uint32_t collectedCount = 0;

    void processCommand();
    void startRocky();
    void stop();
};
