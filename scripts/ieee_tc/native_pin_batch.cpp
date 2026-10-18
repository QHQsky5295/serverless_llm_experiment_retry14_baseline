// Diagnostic only: D685, no serving integration or cross-call state.
// Preserve CUDAHooks.cpp's per-pointer query/error semantics, but invoke the
// native loop once per observation. The caller verifies its single initialized
// owner context, retains all tensors, and holds the GIL throughout this call.
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

extern "C" int ieee_pin_batch(const uintptr_t* pointers, size_t count,
                              unsigned char* output, size_t* completed) {
    if (!completed) return static_cast<int>(cudaErrorInvalidValue);
    *completed = 0;
    if (count && (!pointers || !output))
        return static_cast<int>(cudaErrorInvalidValue);
    for (size_t i = 0; i < count; ++i) {
        if (!pointers[i]) return static_cast<int>(cudaErrorInvalidValue);
        cudaPointerAttributes attributes{};
        cudaError_t status = cudaPointerGetAttributes(
            &attributes, reinterpret_cast<const void*>(pointers[i]));
        if (status == cudaErrorInvalidValue) {
            (void)cudaGetLastError();
            output[i] = 0;
        } else if (status != cudaSuccess) {
            return static_cast<int>(status);
        } else {
            output[i] = attributes.type == cudaMemoryTypeHost;
        }
        *completed = i + 1;
    }
    return static_cast<int>(cudaSuccess);
}
