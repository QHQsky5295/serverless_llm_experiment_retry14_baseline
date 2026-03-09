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

