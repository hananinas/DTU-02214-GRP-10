idf_component_register(SRCS "main.cpp" "camera.cpp" "inference.cpp" "model.c"
                    INCLUDE_DIRS ".")

# Expose ESP-NN macro so we can log at runtime whether SIMD kernels are active
target_compile_definitions(${COMPONENT_LIB} PRIVATE -DESP_NN)

# Permit aggressive vectorization of the preprocessing normalisation loops
target_compile_options(${COMPONENT_LIB} PRIVATE -ffast-math)
