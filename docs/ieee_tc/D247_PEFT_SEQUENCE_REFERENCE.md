# D247 — independent 7B PEFT sequence reference protocol

This correctness-only run compares the 22 captured native PrimeLoRA sequences
from D246 with an offline original-checkpoint HuggingFace/PEFT execution.  It
reuses the existing 7B checkpoint, frozen W0 trace prefix, canonical renderer,
and 500-adapter pool.  It creates no workload, weights, remote archive, or
serving result and does not enter a performance denominator.

The reference uses FP16, eager attention, `use_cache=false`, one visible GPU,
the same canonical prompt and target map, `temperature=0`, greedy decoding,
EOS disabled, and the exact existing adapter tensors.  Before decoding it
checks every adapter's safetensors keys, dtype, and values against the loaded
PEFT state.  It then requires exact equality between the captured native token
IDs and the independent sequence.  A mismatch is retained with request ID,
adapter, first divergent position, and short prefixes; no tolerance, text
re-tokenization, or successful-subset filtering is allowed.

The existing D236 first-position/full-vocabulary reference remains supportive
evidence only; it cannot replace this sequence-level gate because it uses a
different five-arm native receipt.

Launcher: `scripts/ieee_tc/run_d247_7b_peft_sequence_reference.sh`.
