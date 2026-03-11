"""
sitecustomize for FaaSLoRA experiments.

This module is automatically imported by Python (via the standard 'site' module)
whenever it is found on sys.path. We use it to patch torch.serialization.load
so that PyTorch 2.6+ (which defaults to weights_only=True) remains compatible
with legacy .tar-style checkpoints used for some LoRA adapters.

Context:
  - Newer PyTorch versions changed torch.load(..., weights_only=True) behavior.
  - vLLM 0.16 uses torch.load(..., weights_only=True) to load LoRA adapters.
  - Our generated LoRA artifacts use the legacy checkpoint format, which causes:
        RuntimeError: Cannot use ``weights_only=True`` with files saved in the
        legacy .tar format.

To keep experiments working without downgrading PyTorch or vLLM, we globally
override torch.serialization.load so that any explicit weights_only=True is
treated as weights_only=False. This is safe in this controlled research setup,
where all checkpoints are produced locally or come from trusted model sources.
"""

from __future__ import annotations

try:
    import torch  # type: ignore[import]
    import torch.serialization as _ts  # type: ignore[attr-defined]

    _orig_torch_load = getattr(_ts, "load", None)

    if callable(_orig_torch_load):

        def _faaslora_torch_load(f, *args, **kwargs):
            # If caller explicitly requested weights_only=True (or relied on
            # PyTorch's default), relax it to False for compatibility.
            if kwargs.get("weights_only", True) is True:
                kwargs["weights_only"] = False
            return _orig_torch_load(f, *args, **kwargs)

        _ts.load = _faaslora_torch_load  # type: ignore[assignment]
        torch.load = _faaslora_torch_load  # PEFT uses torch.load directly
except Exception:
    # Any failure here should not block running the main program.
    pass


try:
    # vLLM v1 worker shutdown does not always tear down torch.distributed
    # before process exit on single-GPU child workers, which leaves noisy
    # destroy_process_group warnings in our experiment logs. Patch the GPU
    # worker shutdown path so worker processes explicitly clean their dist env.
    from vllm.v1.worker.gpu_worker import Worker as _FaaSLoRAVLLMGPUWorker  # type: ignore[import]
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory as _faaslora_cleanup_dist  # type: ignore[import]

    _orig_vllm_worker_shutdown = getattr(_FaaSLoRAVLLMGPUWorker, "shutdown", None)

    if callable(_orig_vllm_worker_shutdown):

        def _faaslora_vllm_worker_shutdown(self, *args, **kwargs):
            try:
                return _orig_vllm_worker_shutdown(self, *args, **kwargs)
            finally:
                try:
                    _faaslora_cleanup_dist()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        _FaaSLoRAVLLMGPUWorker.shutdown = _faaslora_vllm_worker_shutdown  # type: ignore[assignment]
except Exception:
    # vLLM may be unavailable in non-vLLM environments. Do not block startup.
    pass
