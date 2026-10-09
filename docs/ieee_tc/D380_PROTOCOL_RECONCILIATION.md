# D380 — reconcile development identity and D379 comparison scope

Date: 2026-10-10. Documentation-only audit while D379 is running; no runtime,
configuration, input, threshold, or raw result is changed.

## 1. Authority and correction to D372

The current approved plan, section 4.7, and frozen metric V1, section 8, explicitly
retain `source_trace_seed=42`. The labels 41 (development), 42 (smoke), and
43–47 (formal blocks) identify experiment roles/repetitions; they are not new
independently sampled traces. The plan forbids generating a replacement workload
and explicitly states that repeated fixed-trace runs measure execution variation.

D372 correctly reported that its search found no materialized seed-41/1,000-
request trace. Its further claim that this missing file is a mandatory IEEE-TC
prerequisite is incorrect: it carried forward the earlier EuroSys V2 seed rule
instead of the current approved plan. That claimed blocker is superseded by this
audit. No user authorization for a new trace is required or requested.

Future development continues with the existing immutable source trace/subset,
explicit request-index views where a bounded diagnostic is needed, and a new
execution identity for each run. Record source identity separately from run role
and block. Previously observed inputs do not become held-out workloads by being
assigned a new block number. D371/D378/D379 retain their original exploratory,
non-formal identities; none is retroactively promoted.

Authority hashes (read and checked during this audit):

| File | SHA-256 |
|---|---|
| `PrimeLoRA-PLAN.md` | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| `METRIC_PROTOCOL_FROZEN_V1.md` | `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22` |
| D371 configuration | `eb80bb5702751fb20ec160d429c1c4fd8d9dc411f24a3bf15fe6dcf395a85ff2` |
| D379 configuration | `2685c03555e7e5073fac7edca663e470f521c671a51cf92fb6e4442d8180a1fc` |

## 2. Actual remote path versus legacy limiter label

The shared D379 launcher explicitly enables `FAASLORA_REMOTE_ARTIFACT_ENABLED=1`
and `prepublished_gzip_v1` at the remote 7B endpoint. The current service log
records the enabled endpoint and successful remote artifact fetches. In
`scripts/run_all_experiments.py`, `_materialize_remote_adapter` selects the HTTP
client when present and propagates native transfer errors, without falling back
to the local frozen directory.

In contrast, the top-level `bandwidth_limit_mode` is currently derived only
from the configured bandwidth number: zero emits `local_sim_no_delay` even when
the HTTP client is enabled. That label establishes absence of injected limiter
delay, not the transport actually used. Earlier descriptions calling D379 a
local-copy run on this label alone are superseded. The running configuration is
not changed; final closure must verify request/transfer evidence, delivery mode,
bytes and packing status before asserting complete remote-path coverage.

Both NICs were re-read at this audit and report `1000/full`. No network settings
were changed. This is negotiated link capacity, not a measured 1-Gbit/s payload
throughput. Real request-triggered read/transfer/materialization time stays in
the user timeline; one-time publication remains outside the business replay.

The metadata inconsistency is recorded for correction after D379 closes, with
tests covering HTTP versus local transport independently of limiter state. Do
not silently rewrite historical raw metadata or use the misleading label to
discard otherwise recoverable remote evidence.

## 3. D379 is not a one-factor capacity ablation

The exact D371/D379 YAML diff changes more than capacity:

| Setting | D371 | D379 |
|---|---|---|
| LoRA slots / sequences / runtime cap | 14 | 8 |
| Scaling active upper / lower | 0.5 / 0.25 | 0.875 / 0.4375 |
| Service / preparation profiles | D362 / D362 | D321 / D298 |
| Atomic control interleave opt-in | enabled | omitted |
| Protected file-demand transaction opt-in | enabled | omitted |

Both threshold pairs correspond to seven/fewer-than-3.5 active requests when
multiplied by their respective runtime caps; this algebra does not prove equal
runtime behavior. The profiles and policy switches are substantive confounders.
The current result may describe the combined C=8 configuration, but cannot
isolate the causal effect of capacity or prove that the omitted policies are
unnecessary. Preserve all differences in any old/new comparison.

## 4. Remaining acceptance work and next action

The frozen objectives are unchanged: G1 requires all requests correct and
joint TTFT/TPOT attainment at least 95%, then compares physical lifecycle
GPU-s/correct request. G2 additionally requires the common 75%-of-Resident
budget and compares W1/W2 P95 user TTFT, with separate SLO non-inferiority.
CE and a development 5,000-ms TTFT counter are not acceptance substitutes.

D170 already contains the 7B batch-8 warm measurements and their raw/projection
evidence; reuse and audit them before considering any repeat. D213 is not a
valid Resident budget reference; D285 is only a 100-request protocol pilot.
Neither missing fields nor a renamed block can establish numerical correctness,
the common-reference freeze, or G1/G2. Recover existing evidence first; run only
the outstanding qualifications/references under the actual approved protocol.

Immediate order: let D379 finish under its original scope; collect full request
and physical-release receipts; stop only its owned remote service after receipt
capture; reconcile actual remote evidence and comparison differences; then close
the 7B correctness/common-reference gaps before another speculative optimization.
No 3B optimization, external baseline, ablation, or figure is started by this
audit. The source plan and all frozen metric files remain unchanged.
