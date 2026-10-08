# D357: return to Full after measured control-policy calibration

Status: preparation and native-environment tests passed; **not yet launched**.
Mainline remains 7B → 3B → paused baselines. No figures or formal qualification.

## Why this follows D356

D355's first100 stopped before model creation because strict timing-profile
identity rejected the new control policy. This was not a performance result.
D356 now supplies compatible measurements: original1569 calls plus a100-call
complement, 1635 measured and34 warmup calls, covering200/200service classes and
24/24preparation classes. Old-policy latencies were not relabelled or defaulted.
The two calibration clocks/resources remain separate; this is not two independent
Full repetitions or a warm SLO calibration.

D357 reuses D353's Full controller, C14 runtime and soft8 scale targets. The
single optimization remains `atomic_control_yield_v1`; affected service/preparation
initialization and completed-length identity now come from D356. Only output,
cache paths and the explicit first100 diagnostic identity differ otherwise.
The original runner assembles the exact D356 child model configuration and
loads both profiles strictly:200service/24preparation classes. No bypass added.

## Retain a trigger witness without extra sampling

Source calibration already records native scheduler counters, but normal Full
did not retain them. D356's source counters cannot demonstrate Full triggering.
The existing `_core_pending_admission` register acknowledgement now includes
the already-maintained cumulative interleave counters and scheduler PID only
when the policy is enabled. Existing owner, clock and observation time remain.
The controller already retains the full pending receipt, including the combined
GPU transaction and ordinary HOST/NVMe/remote registration paths.

- No new RPC, polling, timers, full KV scan, cache or policy parameter.
- Canonical pending entry is unchanged; the returned dictionary is copied.
- Register/bind/withdraw semantics and the GPU transaction remain intact.
- Default-off behaviour and bind/withdraw receipts are unchanged.
- Counters precede completion of the current UTILITY. They are not GPU steps,
  kernel counts, GPU idle time or per-request causal latency savings.
- The small record/serialization overhead stays inside measured Full timings;
  it is not declared free or subtracted after measurement.

Historical basis: D354 input-turn evidence, D355 guarded interleave,
D356 calibration and the current Full receipt-preservation path. Rechecked
[vLLM0.30.0 EngineCore source](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py):
`run_busy_loop`, `_process_input_queue`, `_handle_client_request` and
`_invoke_utility_method` retain the native dispatch/return mechanism. The vLLM
skill informed this mechanism-specific audit; its generic tuning thresholds
and benchmark examples were not adopted as this project's targets.

## Validation and handoff

128 tests passed in each of the CPU and installed native environments, covering
default-off, copied counters, same native owner, no additional calls, frontend/
controller preservation, cancellation and the existing interleave contracts.
Tests use synthetic fixtures; they are not GPU/numerical qualification.
Full prelaunch assembly also passed with the real strict profile loaders.

Next: finish the offline Full counter audit (group by scheduler owner; verify
PID/clock/time and nondecreasing counters; retain single-snapshot/zero-trigger
owners and prior attempts without cross-owner subtraction), checkpoint, then
run first100. Require100native contracts/bindings and observed trigger evidence,
actual local/remote cleanup, then ordinary4000W0 with the same frozen candidate.
Compare full-run physicalGPU-s, rawTTFT/TPOT and provisional D170 timing-only
joint attainment to D353/D349/D336, without treating100requests as4000 or calling
unfrozen references/numerical identity a formal G1/G2 pass.

Formal warm reference, Resident budgets, numerical N_correct, 3B, baselines,
main comparisons, ablations and sensitivity matrix all remain open.
