#include <cstdio>
#include <cstdint>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "nvs_flash.h"

// Project includes
#include "camera.h"
#include "inference.h"
#include "model.h"

// Static constants and variables
// Buffers aligned to 16 bytes so that any future SIMD loads/stores are optimal
static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C] __attribute__((aligned(16)));
static constexpr uint32_t TASK_WDT_TIMEOUT_MS = 15000;

struct MemorySnapshot {
    size_t total;
    size_t free;
    size_t minimum_free;
};

static uint32_t successful_inferences = 0;
static uint64_t total_latency_us = 0;
static int64_t first_metrics_us = 0;

static MemorySnapshot memory_snapshot(uint32_t caps)
{
    return {
        .total = heap_caps_get_total_size(caps),
        .free = heap_caps_get_free_size(caps),
        .minimum_free = heap_caps_get_minimum_free_size(caps),
    };
}

static size_t used_bytes(const MemorySnapshot &snapshot)
{
    return snapshot.total > snapshot.free ? snapshot.total - snapshot.free : 0;
}

static void configure_task_watchdog(void)
{
    esp_task_wdt_config_t wdt_config = {
        .timeout_ms = TASK_WDT_TIMEOUT_MS,
        .idle_core_mask = (1 << CONFIG_FREERTOS_NUMBER_OF_CORES) - 1,
        .trigger_panic = false,
    };

    esp_err_t err = esp_task_wdt_reconfigure(&wdt_config);
    if (err == ESP_ERR_INVALID_STATE) {
        err = esp_task_wdt_init(&wdt_config);
    }
    ESP_ERROR_CHECK(err);
    printf("task_wdt_timeout_ms=%u\n", static_cast<unsigned>(TASK_WDT_TIMEOUT_MS));
}

static void print_static_model_metrics(void)
{
    printf("model_size_bytes=%u model_size_kib=%.2f threshold=%.3f ",
           model_binary_len,
           static_cast<double>(model_binary_len) / 1024.0,
           static_cast<double>(MEMBER_THRESHOLD));

    if (MODEL_TEST_ACCURACY >= 0.0f) {
        printf("accuracy=%.4f\n", static_cast<double>(MODEL_TEST_ACCURACY));
    } else {
        printf("accuracy=N/A\n");
    }

    printf("tensor_arena_size_bytes=%u tensor_arena_used_bytes=%u\n",
           static_cast<unsigned>(inference_tensor_arena_size_bytes()),
           static_cast<unsigned>(inference_tensor_arena_used_bytes()));
}

static void print_runtime_metrics(float member_probability, bool is_member, const InferenceTiming &timing)
{
    const int64_t now_us = esp_timer_get_time();
    if (first_metrics_us == 0) {
        first_metrics_us = now_us;
    }

    ++successful_inferences;
    total_latency_us += static_cast<uint64_t>(timing.total_us);

    const double latency_ms = static_cast<double>(timing.total_us) / 1000.0;
    const double avg_latency_ms = static_cast<double>(total_latency_us) /
                                  static_cast<double>(successful_inferences) / 1000.0;
    const double elapsed_s = static_cast<double>(now_us - first_metrics_us) / 1000000.0;
    const double fps = elapsed_s > 0.0 ? static_cast<double>(successful_inferences) / elapsed_s : 0.0;

    const MemorySnapshot all_ram = memory_snapshot(MALLOC_CAP_8BIT);
    const MemorySnapshot internal_ram = memory_snapshot(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const MemorySnapshot psram = memory_snapshot(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    printf("member_probability=%.3f threshold=%.3f decision=%s ",
           static_cast<double>(member_probability),
           static_cast<double>(MEMBER_THRESHOLD),
           is_member ? "MEMBER" : "NON_MEMBER");
    printf("latency_ms=%.2f avg_latency_ms=%.2f preprocess_ms=%.2f invoke_ms=%.2f fps=%.2f ",
           latency_ms,
           avg_latency_ms,
           static_cast<double>(timing.preprocess_us) / 1000.0,
           static_cast<double>(timing.invoke_us) / 1000.0,
           fps);
    if (MODEL_TEST_ACCURACY >= 0.0f) {
        printf("accuracy=%.4f ", static_cast<double>(MODEL_TEST_ACCURACY));
    } else {
        printf("accuracy=N/A ");
    }
    printf("model_size_bytes=%u ram_used_bytes=%u ram_free_bytes=%u ram_min_free_bytes=%u ",
           model_binary_len,
           static_cast<unsigned>(used_bytes(all_ram)),
           static_cast<unsigned>(all_ram.free),
           static_cast<unsigned>(all_ram.minimum_free));
    printf("internal_ram_free_bytes=%u psram_used_bytes=%u psram_free_bytes=%u tensor_arena_used_bytes=%u\n",
           static_cast<unsigned>(internal_ram.free),
           static_cast<unsigned>(used_bytes(psram)),
           static_cast<unsigned>(psram.free),
           static_cast<unsigned>(inference_tensor_arena_used_bytes()));
}

void setup()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    configure_task_watchdog();

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

    printf("On-device face classifier ready. Running without camera_app streaming.\n");
    print_static_model_metrics();
}

void loop(void)
{
    if (camera_capture_frame(image_buffer)) {
        float member_probability = 0.0f;
        InferenceTiming timing = {};
        if (inference_run_rgb565(image_buffer, &member_probability, &timing)) {
            const bool is_member = member_probability >= MEMBER_THRESHOLD;
            print_runtime_metrics(member_probability, is_member, timing);
        }
    }

    // Block for at least one tick so IDLE0 can reset the task watchdog.
    vTaskDelay(1);
}

// ---------- ESP-IDF entry point ----------

extern "C" void app_main()
{
    setup();
    while (true) {
        loop();
    }
}
