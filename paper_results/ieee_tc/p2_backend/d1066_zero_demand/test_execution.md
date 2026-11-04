# CPU test execution receipts

2026-11-05, no GPU replay. All scopes used MemoryHigh=3G, MemoryMax=4G,
MemorySwapMax=0, CPUs 2,3,26,27, hidden CUDA, one BLAS/OMP thread.

- pytest attempt: scope primelora-tc-test-cd5b1ca40ffc47a0b0eb5d6f80fe46ce.scope,
  exit 1, LLM_vllm0102/bin/python: No module named pytest. No tests executed or
  XML produced; no dependency installation or code change to mask this failure.
- unittest ZeroDemandResidency: scope
  primelora-tc-test-b078d05ed2254a02b854b44cac369b41.scope, 5 tests, 0.362 s,
  exit 0 / OK, terminal receipt retained in Codex tool session 8992.
- unittest ControlRoundFileObservation +
  OwnedPreparationPlanning.test_zero_demand_never_invents_a_missing_preparation_measurement
  + tests.test_basic_smoke: scope
  primelora-tc-test-a101b02adc634fe38eba4bba39e83d43.scope, 306 tests,
  42.334 s, exit 0 / OK, terminal receipt in Codex tool session 49890.

The original ControlRoundFileObservation calls were missing await after the
earlier production async conversion. Those calls now await the actual API;
assertions and ownership semantics are unchanged. No performance repetitions.
These terminal receipts are not claimed as machine-generated JUnit records.
