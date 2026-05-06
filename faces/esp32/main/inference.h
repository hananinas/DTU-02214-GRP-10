#pragma once

#include <cstddef>
#include <cstdint>

struct InferenceTiming {
    int64_t preprocess_us;
    int64_t invoke_us;
    int64_t total_us;
};

bool inference_init(void);
bool inference_run_rgb565(const uint8_t *frame, float *member_probability, InferenceTiming *timing);
size_t inference_tensor_arena_size_bytes(void);
size_t inference_tensor_arena_used_bytes(void);
