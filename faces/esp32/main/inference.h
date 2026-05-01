#pragma once

#include <cstdint>

bool inference_init(void);
bool inference_run_rgb565(const uint8_t *frame, float *member_probability);
