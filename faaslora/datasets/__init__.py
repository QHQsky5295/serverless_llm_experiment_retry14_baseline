"""
FaaSLoRA Dataset & Workload Generation Module

Provides realistic request traces for experiments comparable to S-LoRA, Punica,
ServerlessLLM and related work, using:
  - ShareGPT-style prompt distributions (embedded + optional HF download)
  - Poisson arrival process
  - Zipf-distributed LoRA selection with evolving hotness
  - Burst / scale-up event simulation
"""
from .workload_generator import WorkloadGenerator, RequestTrace, WorkloadConfig
