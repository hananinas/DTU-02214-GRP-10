#include <cstdio>
#include <cstdint>
#include <cstring>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "driver/usb_serial_jtag.h"

// Project includes
#include "camera.h"
#include "inference.h"
#include "model.h"

// Static constants and variables
static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C];

void setup()
{
    // Initialize NVS (required by some drivers)
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // Initialize camera
    if (!camera_init()) {
        abort();
    }

    if (!inference_init()) {
        abort();
    }

    // Initialize USB serial for logs/control.
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 32768,
        .rx_buffer_size = 512,
    };
    err = usb_serial_jtag_driver_install(&cfg);
    ESP_ERROR_CHECK(err);

    // Wait for incoming S on serial port so tests can start from a known point.
    printf("On-device face classifier ready. Send 'S' to start.\n");
    char c;
    do {
        int r = usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY);
        if (r < 0) {
            abort();
        }
    } while (c != 'S');
}

// Set to true to stream raw RGB565 frames over serial so the pygame app
// can display the camera feed alongside on-device inference results.
static constexpr bool STREAM_FRAMES = true;

static void stream_frame(void)
{
    // Preamble the pygame app uses to find frame boundaries.
    const char *preamble = "===FRAME===\n";
    usb_serial_jtag_write_bytes(preamble, strlen(preamble), portMAX_DELAY);

    // Send raw RGB565 frame in chunks (USB FS packet size = 64 bytes).
    constexpr size_t FRAME_BYTES = FRAME_W * FRAME_H * FRAME_C;
    constexpr size_t CHUNK = 512;
    for (size_t offset = 0; offset < FRAME_BYTES; offset += CHUNK) {
        size_t n = (offset + CHUNK <= FRAME_BYTES) ? CHUNK : (FRAME_BYTES - offset);
        usb_serial_jtag_write_bytes(image_buffer + offset, n, portMAX_DELAY);
    }
}

void loop(void)
{
    if (camera_capture_frame(image_buffer)) {
        float member_probability = 0.0f;
        if (inference_run_rgb565(image_buffer, &member_probability)) {
            const bool is_member = member_probability >= MEMBER_THRESHOLD;

            if (STREAM_FRAMES) {
                stream_frame();
            }

            printf("member_probability=%.3f threshold=%.3f decision=%s\n",
                   member_probability,
                   static_cast<double>(MEMBER_THRESHOLD),
                   is_member ? "MEMBER" : "NON_MEMBER");
        }
    }

    // When streaming frames the USB transfer provides natural pacing; only
    // add a small yield so the idle task can run the watchdog.
    vTaskDelay(pdMS_TO_TICKS(STREAM_FRAMES ? 10 : 100));
}

// ---------- ESP-IDF entry point ----------

extern "C" void app_main()
{
    setup();
    while (true) {
        loop();
    }
}
