# D250 — 7B Full/Delayed readiness-publication diagnostic

Date: 2026-10-06

## Purpose and boundary

D250 is the next bounded Prime-only diagnostic authorized by D248. It tests the
pre-registered readiness-publication hypothesis with the existing runner, frozen
W0 trace, frozen 7B adapter subset, vLLM backend, service/preparation profiles,
remote delivery contract, and resource envelope. It is not a G1/G2 result and
does not compare an external baseline.

The paired variants differ only in the existing `ieee_handoff_policy` gate:

| variant | policy | interpretation |
|---|---|---|
| Full | `full` | existing activation preparation path |
| Delayed | `delayed` | same planned preparation is started only after engine initialization |

The diagnostic uses eight consecutive 500-request phases, with a fixed 2-second
idle interval between phases. It does not add artificial request pressure,
change adapter content, change arrival times, or relabel a non-GPU source as
GPU-ready. Each phase uses the runner's existing scale-down/scale-up path; all
activation and first-dispatch events are retained with their causal category.

## Configurations and immutable inputs

- Full config: `configs/ieee_tc/20261006_d250_7b_readiness_8phase_full.yaml`
- Delayed config: `configs/ieee_tc/20261006_d250_7b_readiness_8phase_delayed.yaml`
- Full launcher: `scripts/ieee_tc/run_d250_7b_readiness_8phase_full.sh`
- Delayed launcher: `scripts/ieee_tc/run_d250_7b_readiness_8phase_delayed.sh`
- Model: frozen Llama-2-7B checkpoint and publicmix V2 adapter pool
- Workload: existing 4,000-request W0 trace, seed 42, fixed-length greedy v1
- Generation: same canonical prompts, target map and no-EOS contract as D246/D247
- Remote: existing prepublished gzip delivery object; no request-time packing

The two configurations have fresh result, NVMe and HOST roots. No historical
result directory is reused or overwritten.

## Required evidence and acceptance

For every successful request, the result must contain the dispatch-time tier
snapshot and consistent readiness booleans. For every activation epoch, retain
trigger category, trigger/ready times, preparation state, plan identity, owner,
first dispatch and first token. The first-service record is bound to the first
actual dispatch for that activation; another request or replica cannot fill a
missing sample.

The diagnostic gate requires:

1. 4,000 planned, 4,000 terminal, zero failed requests;
2. eight complete 500-request phases and a 2-second inter-phase interval;
3. 100% dispatch-tier coverage and zero tier invariant conflicts;
4. at least eight natural scale-up events per variant;
5. at least 20 first-service samples across the paired diagnostic, or one
   identically configured cold repeat if the first run is below that threshold;
6. complete physical-resource sidecars and released GPU owners; and
7. no unresolved watchdog, cgroup, remote-service, or cleanup failures.

If the first-service threshold is not met, the run remains a descriptive
readiness observation and may be expanded once with the same execution key. It
must not be made to pass by relabeling initial or controlled activations,
substituting another request, or pooling incompatible configurations.

## Interpretation

The primary causal quantities are engine initialization time, preparation
terminal time, activation-to-first-token, and first-dispatch TTFT. A reduction
under Delayed supports the hypothesis only when common workload, tier coverage,
resource, and correctness gates pass. No reduction is evidence under the null
hypothesis that publication timing was not the dominant bottleneck. Either
outcome narrows the next optimization; neither authorizes a claim of global
optimality or baseline superiority.
