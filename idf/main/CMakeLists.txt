idf_component_register(
    SRCS "main.cpp" "drivers/Display.cpp" "graphics/Eyes.cpp" "graphics/Animator.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_i2c esp_lcd esp_psram esp_timer spi_flash
)
