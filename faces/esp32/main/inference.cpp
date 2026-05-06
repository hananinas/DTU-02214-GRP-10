#include "inference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "Inference";
static constexpr size_t TENSOR_ARENA_SIZE = 6 * 1024 * 1024;

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static uint8_t *tensor_arena = nullptr;

static int clamp_to_int(float value, int min_value, int max_value)
{
    int rounded = static_cast<int>(std::lround(value));
    return std::min(std::max(rounded, min_value), max_value);
}

static void rgb565_to_rgb888(uint8_t byte1, uint8_t byte2, uint8_t *r, uint8_t *g, uint8_t *b)
{
    *r = byte1 & 0xF8;
    *g = static_cast<uint8_t>(((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3));
    *b = static_cast<uint8_t>((byte2 & 0x1F) << 3);
}

static bool preprocess_rgb565_int8(const uint8_t *frame)
{
    const int crop_size = std::min(FRAME_W, FRAME_H);
    const int crop_x = (FRAME_W - crop_size) / 2;
    const int crop_y = (FRAME_H - crop_size) / 2;

    constexpr float scale  = 1.0f / 127.5f;
    constexpr float offset = -1.0f;

    for (int y = 0; y < MODEL_INPUT_HEIGHT; ++y) {
        const int src_y = crop_y + (y * crop_size) / MODEL_INPUT_HEIGHT;
        for (int x = 0; x < MODEL_INPUT_WIDTH; ++x) {
            const int src_x = crop_x + (x * crop_size) / MODEL_INPUT_WIDTH;
            const size_t src_index = static_cast<size_t>(src_y * FRAME_W + src_x) * FRAME_C;

            uint8_t r;
            uint8_t g;
            uint8_t b;
            rgb565_to_rgb888(frame[src_index], frame[src_index + 1], &r, &g, &b);

            const size_t dst_index = static_cast<size_t>(y * MODEL_INPUT_WIDTH + x) * MODEL_INPUT_CHANNELS;
            input->data.int8[dst_index]     = static_cast<int8_t>(
                clamp_to_int((static_cast<float>(r) * scale + offset) / input->params.scale + input->params.zero_point, -128, 127));
            input->data.int8[dst_index + 1] = static_cast<int8_t>(
                clamp_to_int((static_cast<float>(g) * scale + offset) / input->params.scale + input->params.zero_point, -128, 127));
            input->data.int8[dst_index + 2] = static_cast<int8_t>(
                clamp_to_int((static_cast<float>(b) * scale + offset) / input->params.scale + input->params.zero_point, -128, 127));
        }
    }
    return true;
}

static bool preprocess_rgb565_uint8(const uint8_t *frame)
{
    const int crop_size = std::min(FRAME_W, FRAME_H);
    const int crop_x = (FRAME_W - crop_size) / 2;
    const int crop_y = (FRAME_H - crop_size) / 2;

    constexpr float scale  = 1.0f / 127.5f;
    constexpr float offset = -1.0f;

    for (int y = 0; y < MODEL_INPUT_HEIGHT; ++y) {
        const int src_y = crop_y + (y * crop_size) / MODEL_INPUT_HEIGHT;
        for (int x = 0; x < MODEL_INPUT_WIDTH; ++x) {
            const int src_x = crop_x + (x * crop_size) / MODEL_INPUT_WIDTH;
            const size_t src_index = static_cast<size_t>(src_y * FRAME_W + src_x) * FRAME_C;

            uint8_t r;
            uint8_t g;
            uint8_t b;
            rgb565_to_rgb888(frame[src_index], frame[src_index + 1], &r, &g, &b);

            const size_t dst_index = static_cast<size_t>(y * MODEL_INPUT_WIDTH + x) * MODEL_INPUT_CHANNELS;
            input->data.uint8[dst_index]     = static_cast<uint8_t>(
                clamp_to_int((static_cast<float>(r) * scale + offset) / input->params.scale + input->params.zero_point, 0, 255));
            input->data.uint8[dst_index + 1] = static_cast<uint8_t>(
                clamp_to_int((static_cast<float>(g) * scale + offset) / input->params.scale + input->params.zero_point, 0, 255));
            input->data.uint8[dst_index + 2] = static_cast<uint8_t>(
                clamp_to_int((static_cast<float>(b) * scale + offset) / input->params.scale + input->params.zero_point, 0, 255));
        }
    }
    return true;
}

static bool preprocess_rgb565_float32(const uint8_t *frame)
{
    const int crop_size = std::min(FRAME_W, FRAME_H);
    const int crop_x = (FRAME_W - crop_size) / 2;
    const int crop_y = (FRAME_H - crop_size) / 2;

    constexpr float scale  = 1.0f / 127.5f;
    constexpr float offset = -1.0f;

    for (int y = 0; y < MODEL_INPUT_HEIGHT; ++y) {
        const int src_y = crop_y + (y * crop_size) / MODEL_INPUT_HEIGHT;
        for (int x = 0; x < MODEL_INPUT_WIDTH; ++x) {
            const int src_x = crop_x + (x * crop_size) / MODEL_INPUT_WIDTH;
            const size_t src_index = static_cast<size_t>(src_y * FRAME_W + src_x) * FRAME_C;

            uint8_t r;
            uint8_t g;
            uint8_t b;
            rgb565_to_rgb888(frame[src_index], frame[src_index + 1], &r, &g, &b);

            const size_t dst_index = static_cast<size_t>(y * MODEL_INPUT_WIDTH + x) * MODEL_INPUT_CHANNELS;
            input->data.f[dst_index]     = static_cast<float>(r) * scale + offset;
            input->data.f[dst_index + 1] = static_cast<float>(g) * scale + offset;
            input->data.f[dst_index + 2] = static_cast<float>(b) * scale + offset;
        }
    }
    return true;
}

static bool preprocess_rgb565(const uint8_t *frame)
{
    switch (input->type) {
    case kTfLiteFloat32:
        return preprocess_rgb565_float32(frame);
    case kTfLiteInt8:
        return preprocess_rgb565_int8(frame);
    case kTfLiteUInt8:
        return preprocess_rgb565_uint8(frame);
    default:
        ESP_LOGE(TAG, "Unsupported input tensor type: %s", TfLiteTypeGetName(input->type));
        return false;
    }
}

static float read_probability()
{
    switch (output->type) {
    case kTfLiteFloat32:
        return output->data.f[0];
    case kTfLiteInt8:
        return (static_cast<float>(output->data.int8[0]) - output->params.zero_point) * output->params.scale;
    case kTfLiteUInt8:
        return (static_cast<float>(output->data.uint8[0]) - output->params.zero_point) * output->params.scale;
    default:
        ESP_LOGE(TAG, "Unsupported output tensor type: %s", TfLiteTypeGetName(output->type));
        return NAN;
    }
}

bool inference_init(void)
{
    if (model_binary_len <= 1) {
        ESP_LOGE(TAG, "Missing generated model. Run `uv run python export_tflite_micro.py` from faces/.");
        return false;
    }

    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: model=%ld runtime=%d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Align tensor arena to 16 bytes so the NN backend can use SIMD loads/stores
    tensor_arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate %u byte tensor arena in PSRAM", static_cast<unsigned>(TENSOR_ARENA_SIZE));
        return false;
    }

    static tflite::MicroMutableOpResolver<16> resolver;
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddMean();
    resolver.AddMul();
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddStridedSlice();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Model bytes: %u", model_binary_len);
    ESP_LOGI(TAG, "ESP-NN SIMD optimisations active (target=%s)", CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, "Input: %s [%d, %d, %d, %d]", TfLiteTypeGetName(input->type),
             input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Output: %s", TfLiteTypeGetName(output->type));
    ESP_LOGI(TAG, "Tensor arena used: %u bytes", static_cast<unsigned>(interpreter->arena_used_bytes()));

    if (input->dims->size != 4 || input->dims->data[1] != MODEL_INPUT_HEIGHT ||
        input->dims->data[2] != MODEL_INPUT_WIDTH || input->dims->data[3] != MODEL_INPUT_CHANNELS) {
        ESP_LOGE(TAG, "Unexpected input shape. Expected [1,%d,%d,%d]", MODEL_INPUT_HEIGHT,
                 MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS);
        return false;
    }

    return true;
}

size_t inference_tensor_arena_size_bytes(void)
{
    return TENSOR_ARENA_SIZE;
}

size_t inference_tensor_arena_used_bytes(void)
{
    if (interpreter == nullptr) {
        return 0;
    }
    return interpreter->arena_used_bytes();
}

bool inference_run_rgb565(const uint8_t *frame, float *member_probability, InferenceTiming *timing)
{
    if (interpreter == nullptr || input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Inference is not initialized");
        return false;
    }

    int64_t start_us = esp_timer_get_time();
    if (!preprocess_rgb565(frame)) {
        return false;
    }
    int64_t preprocess_us = esp_timer_get_time();

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Model invocation failed");
        return false;
    }
    int64_t inference_us = esp_timer_get_time();

    *member_probability = read_probability();
    if (timing != nullptr) {
        timing->preprocess_us = preprocess_us - start_us;
        timing->invoke_us = inference_us - preprocess_us;
        timing->total_us = inference_us - start_us;
    }

    return !std::isnan(*member_probability);
}
