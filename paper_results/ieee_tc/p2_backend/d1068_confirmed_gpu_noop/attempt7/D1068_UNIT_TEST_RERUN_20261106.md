# D1068 CPU-side regression receipt

## Command and environment

The existing planning/snapshot suite was rerun after the D1068 implementation
using the frozen experiment interpreter:

```text
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python -m pytest -q \
tests/test_ieee_tc_planning_cpu.py tests/test_ieee_tc_planning_snapshot.py
```

## Outcome

* 45 tests collected and 45 passed.
* Runtime: 116.06 s.
* No test failure, skipped test, or GPU allocation occurred.
* The run emitted 19 existing warnings (pytest configuration, TorchScript
  deprecation, and Pydantic deprecation); none is a D1068 assertion failure.

This receipt validates the CPU planning worker, no-preparation proof boundary,
native/file snapshot checks, and the planning transport tests at the current
source revision. It is a code-path regression result, not a serving-performance
or semantic-correctness claim.

An initial invocation with the system Python 3.13 produced three dependency
errors (`vllm`/`msgspec` absent). That environment mistake was not merged into
the result; the fixed venv rerun above is the authoritative test outcome.
