# Save originals
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _OLD_CUDA_HOME="${CUDA_HOME-}"
export _OLD_CUDA_PATH="${CUDA_PATH-}"
export _OLD_PATH="${PATH-}"
export _OLD_CC="${CC-}"
export _OLD_CXX="${CXX-}"

# Critical: avoid system CUDA libs overriding torch cu130 bundled libs
unset LD_LIBRARY_PATH

# Ensure system tools are always discoverable (sudo/gcc/nvidia-smi)
export PATH="/usr/bin:/bin:$PATH"

# Keep CUDA toolkit available for nvcc (without overriding runtime libs)
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=
export PATH="$CUDA_HOME/bin:$PATH"

# Triton JIT needs a C/C++ compiler
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# vLLM stability
export VLLM_ENABLE_V1_MULTIPROCESSING=0
