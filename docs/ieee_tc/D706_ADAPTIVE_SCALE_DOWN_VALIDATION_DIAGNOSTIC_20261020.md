# D706: Adaptive scale-down validation diagnostic

Date: 2026-10-20
Status: read-only diagnostic; no production configuration or runtime result was changed.

## Purpose

D703 showed three natural scale-ups and no natural scale-down while the service
was already using four physical GPUs. D705 established that the controller was
sampling complete `overall_ttft_ms`, whereas its lower TTFT bound came from an
earlier source-only acquisition calibration. D706 records a measurable,
testable control hypothesis before any change to `IEEEReplicaControl`.

The D703 adapter files are all-zero according to D704. Consequently this
diagnostic does not qualify D703 for G1/G2 and does not claim a PrimeLoRA
performance improvement.

## Candidate derived reserve

Let the common TTFT upper bound be `T_SLO`, and let
the P95 overall TTFT of affected requests be `Q_0.95(T_scaleup)`; it is
measured for requests affected by a scale-up or handoff on a completed
validation prefix. The candidate lower bound is

\[
  \tau_{\mathrm{auto}}
  = T_{\mathrm{SLO}} - Q_{0.95}(T_{\mathrm{scaleup}}).
\]

For D703, `T_SLO=5000` ms and the recorded affected-request P95
was 1979.495699 ms, giving

\[
  \tau_{\mathrm{auto}}=3020.504301\ \mathrm{ms}.
\]

This leaves the observed 95th-percentile activation and handoff cost inside the
common TTFT budget. It is a diagnostic candidate because the D703 value was
computed after the run. It must not be installed for D703 or used as a formal
test threshold.

The control decision remains conjunctive:

\[
q < q_{\mathrm{lower}},\qquad
\frac{\mathrm{active}}{\mathrm{ready\ capacity}} < a_{\mathrm{lower}},\qquad
P95_{W}(\mathrm{overall\ TTFT}) < \tau_{\mathrm{auto}},
\]

with no pending activation, a cooldown interval, and more than one ready
replica. If the reserve cannot be derived from a completed validation prefix,
the controller must keep the conservative existing gate and report that the
adaptive threshold is unavailable. A missing reserve is never replaced with a
hand-selected constant.

## D703 replay evidence

The replay used the 1,936 recorded control snapshots from
`mechanism_events._ieee_control_events`. It is a counterfactual state replay;
it does not rerun requests, release a GPU, or measure a changed SLO.

| lower-bound policy | all-low snapshots | qualifying runs | longest run | runs reaching 48 s |
|---|---:|---:|---:|---:|
| D703 legacy 219.518056 ms | 0 | 0 | 0 s | 0 |
| derived diagnostic 3020.504301 ms | 1,611 | 165 | 213.4299 s | 11 |

The replay therefore shows that the legacy gate explains the absence of a
scale-down opportunity, while the derived candidate creates observable
cooldown opportunities. It cannot establish the counterfactual physical
GPU-s, request latency, or SLO because the replica set and request routing did
not change during the replay.

## Formal validation contract

Before a formal run, the reserve must be produced from a frozen validation
prefix or a separately registered historical profile completed before the test
trace. The prefix may contain only requests that precede the test in the
execution contract; future requests, future generated lengths, and the test
outcome are forbidden. The resulting reserve and its source hash are frozen in
the new execution key.

The validation run must record, for every request, numerical adapter
correctness, completion, native TTFT/TPOT, dispatch and activation spans,
physical GPU allocation and release intervals, and the full control trace. The
candidate is admissible only if the run satisfies the existing correctness
contract and joint TTFT/TPOT SLO contract. G1/G2 ranking remains unavailable
when adapter correctness is unknown, even if the latency SLO is high.

The validation must compare the derived policy with the frozen current policy
under the same trace, model, adapter set, resource envelope, and remote delivery
contract. A new execution key is required. Formal acceptance requires evidence
of the resulting GPU-s and SLO; the replay table alone is insufficient.

## Source integrity

| source | SHA-256 |
|---|---|
| D703 request summary | `eeab36caab728bdb9217ebfd54f509ecc49398994f9a9646a050f5bda24d2c84` |
| D703 control event log | `e4c2c52c8ad4c730aca1e8a9dedf3690427c5cd85e4a72fd6a31a217eb696211` |
| D705 diagnosis | `e4ba9fe7342697785feb7d14868a925852e992870354a020cf412760a0bb4ab4` |

The D703 source remains immutable. This file records a falsifiable validation
entry and does not rewrite the D703 result or its ranking status.
