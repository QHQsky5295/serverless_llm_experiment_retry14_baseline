"""
Global compatibility patch for FaaSLoRA experiments on PyTorch 2.6+.

Problem:
  - Newer PyTorch versions changed torch.load default to weights_only=True.
  - vLLM (0.16.x) calls torch.load(..., weights_only=True) when loading LoRA
    adapters.
  - Our LoRA checkpoints use the legacy .tar format, which is incompatible
    with weights_only=True and triggers:

      RuntimeError: Cannot use ``weights_only=True`` with files saved in the legacy .tar format.

Solution:
  - This module is automatically imported by Python (via the 'site' module)
    whenever it is present on sys.path (standard sitecustomize hook).
  - We monkey-patch torch.serialization.load so that any explicit
    weights_only=True is treated as weights_only=False.
  - This is safe in this controlled research setting where all checkpoints
    are produced locally or by trusted model providers.
"""

from __future__ import annotations

try:
    import torch  # type: ignore[import]
    import torch.serialization as _ts  # type: ignore[attr-defined]

    _orig_torch_load = getattr(_ts, "load", None)

    if callable(_orig_torch_load):

        def _faaslora_torch_load(f, *args, **kwargs):
            # If caller requests weights_only=True (or relies on the default),
            # relax it to False for compatibility with legacy checkpoints.
            if kwargs.get("weights_only", True) is True:
                kwargs["weights_only"] = False
            return _orig_torch_load(f, *args, **kwargs)

        _ts.load = _faaslora_torch_load  # type: ignore[assignment]
except Exception:
    # Any failure here should not block the main program.
    pass

