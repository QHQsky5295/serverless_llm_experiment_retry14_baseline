// Config-derived Llama 3 frequency scaling; no request-dependent constants.
#pragma once
namespace flashinfer {
__device__ __forceinline__ float tc_llama3_frequency(float freq) {
  const float wavelength = 6.283185307179586f / freq;
  if (wavelength > 8192.0f / 1.0f) return freq / 32.0f;
  if (wavelength < 8192.0f / 4.0f) return freq;
  const float smooth = (8192.0f / wavelength - 1.0f) / (4.0f - 1.0f);
  return (1.f - smooth) * freq / 32.0f + smooth * freq;
}
}
