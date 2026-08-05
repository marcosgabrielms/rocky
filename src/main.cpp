#include <Arduino.h>

#include "drivers/Display.h"
#include "graphics/Eyes.h"
#include "graphics/Animator.h"

Display display;
Eyes eyes(display);
Animator animator(eyes);

void setup()
{
    Serial.begin(115200);

    if (!display.begin())
    {
        Serial.println("Falha ao inicializar o display.");

        while (true)
        {
            delay(100);
        }
    }

    animator.begin();
}

void loop()
{
    animator.update();
}