# PrimeLoRA IEEE-TC metric protocol: remote and HydraServe addendum (2026-10-06)

This addendum freezes the interpretation used by all future PrimeLoRA and
baseline comparisons.  It is deliberately versioned: it clarifies the
existing protocol and does not rewrite the previously frozen files or change
the identity of any completed run.

## 1. Parent documents and precedence

The parent documents are:

| document | SHA-256 |
|---|---|
| `METRIC_PROTOCOL_FROZEN_V1.md` | `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22` |
| `METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` | `485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29` |
| `METRIC_PROTOCOL_FINAL_V2.md` | `5878ed9ce4494edb02916962ecb588a2b32a5cf59e66211521b90b9af95a65c9` |
| `METRIC_PROTOCOL_HYDRASERVE_REMOTE_AUDIT_20261006.md` | `ea5b82c0ddfdfcb4ac736f8fe76862a976cccb9ad081a4ec26eae4efe7d022ac` |
| `PrimeLoRA-PLAN.md` | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |

The parent protocol remains authoritative when this file is silent.  A future
run must record this addendum's SHA alongside the parent SHAs.

## 2. What is primary, and what is secondary

The paper's system objective is a feasible serving point, not an arbitrary
latency--cost product.  We therefore use a two-stage comparison.

**G1 (resource efficiency under correct service):** first require the common
generation contract, `N_correct == N_plan`, and the joint TTFT/TPOT SLO.  Among
systems that satisfy those gates, rank physical GPU-seconds per correct
request:

\[
 U = \sum_d \int_{t_0}^{t_1} a_d(t)\,dt,\qquad
 R_{\mathrm{GPU}} = U/N_{\mathrm{correct}}.
\]

Here `a_d(t)` is the number of physically allocated cards owned by the
deployment at time `t`; it is not a legacy instance-billing estimate and is
not replaced by GPU memory-time.

**G2 (tail latency under the same resource budget):** use the common reference
resource envelope `U_ref` and the pre-registered budget
`B = 0.75 U_ref`.  After the same correctness and joint-SLO gates, compare the
absolute P95 user-observed TTFT under `U <= B`.  If a run does not satisfy the
budget or SLO, it is not a winning G2 point; it is reported as an infeasible
point with the reason.

CE, monetary cost/request, throughput, mean TTFT and TPOT remain important
diagnostics.  They are not allowed to replace G1/G2.  CE is reported only with
its full billing factor and price assumptions, plus sensitivity results.

The joint per-request indicator is

\[
 I_r = \mathbf 1[TTFT_r\leq S_{TTFT}\;\land\;TPOT_r\leq S_{TPOT}],
 \qquad A_{joint}=\frac{1}{N_{correct}}\sum_{r\in correct} I_r .
\]

The denominator is never silently changed to offered, submitted, or only
successful requests.  We publish `N_plan`, `N_arrived`, `N_submitted`,
`N_terminal`, `N_correct`, `N_good`, and the exclusion reason for every run.

## 3. HydraServe alignment, without importing its assumptions

HydraServe is useful as a metric precedent because it evaluates cold-start
TTFT, TTFT/TPOT SLO attainment, TPOT, and resource/cost trade-offs rather than
claiming a universal CE score.  Its paper describes a modified vLLM serving
stack, remote model storage, pipeline/overlap mechanisms, and reports warm
TTFT/TPOT together with end-to-end cold-start thresholds.  The official
implementation is based on modified vLLM 0.4.2.  See the
[NSDI'26 paper](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf),
the [paper page](https://www.usenix.org/conference/nsdi26/presentation/lou),
and the [official repository](https://github.com/LLMServe/hydraserve).

We adopt only the *structure* of that reporting:

* publish warm reference TTFT/TPOT and cold/scale-out TTFT separately;
* report joint TTFT/TPOT SLO attainment and the resource trade-off;
* decompose user TTFT and E2E into queue/admission, dispatch, artifact
  movement, service TTFT and decode components.

The warm-derived thresholds sometimes used by HydraServe (TTFT up to five
times warm TTFT and TPOT up to two times warm TPOT) are not an industry
standard.  If used diagnostically, they must be recomputed from this machine,
model, token-length bin and generation contract, and labelled
`hydraserve_style_reference`, never silently substituted for the frozen joint
SLO.

## 4. Request timestamp and decomposition contract

For each request, retain the monotonic-clock spans

`scheduled_arrival -> arrival_observed -> admission_start -> admitted ->
dispatch/reserve -> artifact read/receive/write/extract/materialize ->
service_start -> first_token -> final_token`.

The primary user TTFT is scheduled-trace arrival to the system-observed first
generated token/chunk.  Primary E2E is scheduled arrival to observed response
completion.  Service TTFT starts at the admitted/backend-execution boundary;
it must not be used to hide queue or dispatch delay.  TPOT is recomputed from
native token IDs under the frozen generation contract.  No dispatch or remote
span is counted twice in E2E.

## 5. What counts as request work on the remote path

The following are part of the request's user-visible E2E whenever causally
triggered by that request: opening and reading the remote artifact, server send
and back-pressure, wire transfer, client receive, local write, extraction,
verification, materialization, cache-miss fetch, migration and eviction needed
to admit that request.  The request record must retain logical payload bytes,
wire/archive bytes, transfer ID, and additive monotonic spans.

The following are outside request E2E and are never subtracted after the fact:

* one-time creation of the read-only prepublished gzip delivery cache;
* administrator actions, service start/stop/restart and health checks;
* full-pool hash scans, bookkeeping, log collection and cleanup;
* compression or packing launched outside a request.

The current approved delivery mode is `prepublished_gzip_v1`; no request-time
repacking is permitted.  The one-time cache is a delivery preparation step,
not free request work and not a reason to remove receive/extract/materialize
spans from E2E.

## 6. Fairness and remote-run isolation

PrimeLoRA and each baseline use the same immutable content manifest, adapter
subset, trace, request/adapter/arrival/target map, generation contract,
deployment notice, physical resource envelope and observed link state.  We do
not force identical cache misses or fetch counts: a lower miss/fetch count is a
mechanism outcome and must remain visible.

Remote administration and service health work is frozen during a formal run.
If it overlaps the run, the run is marked externally interfered and is either
excluded and repeated or reported in a separate execution stratum; raw data
are retained.  No manual adjustment repairs an interfered metric.

## 7. Link-state record and the 100-Mbps observation

Every formal run records a link-state key containing local and remote interface,
carrier, negotiated speed/duplex, auto-negotiation, boot ID, driver, error
counters, qdisc state, logical/wire bytes, transfer concurrency and achieved
throughput.  Application-layer limiting is recorded in MiB/s; it is not
labelled as Mbps.

The historical remote diagnosis D77 observed `eno1` at 100 Mb/s because the
link partner advertised only 10/100, while the adapter/driver was a Broadcom
BCM5720/tg3 capable of gigabit.  After reboot D78 and the current D270
read-only evidence observed 1000 Mb/s full duplex with auto-negotiation and no
agent-forced qdisc or speed.  The evidence does not identify whether the
historical transient was the cable, switch port, partner state or another
physical-layer condition.  Copper auto-negotiation chooses the highest common
speed when both ends advertise it, and 1000BASE-T requires auto-negotiation;
see the [kernel Ethernet documentation](https://www.kernel.org/doc/html/next/networking/device_drivers/ethernet/intel/e1000.html)
and [ethtool documentation](https://man7.org/linux/man-pages/man8/ethtool.8.html).

D77 and D78/D270 are separate execution strata and must not be pooled.  The
current PrimeLoRA qualification work uses the observed 1-Gbps/full-duplex
workpoint.  A 100-Mbps run may be retained as a historical sensitivity point or
repeated as its own sensitivity stratum; it cannot be used to claim a permanent
hardware limit or to erase remote work from E2E.  We make no H100, NVLink or
100/400-Gbps performance claim from this testbed.

## 8. Publication columns and gate

Each curated request/run table must include, at minimum:

`run_id, model, seed, trace_sha, subset_sha, generation_contract, link_state,
transfer_id, logical_bytes, wire_bytes, remote_read_ms, remote_send_ms,
client_receive_ms, local_write_ms, extract_ms, materialize_ms, tier_before,
tier_after, cache_hit, request/dispatch/admission/service timestamps,
TTFT_ms, TPOT_ms, E2E_ms, correct, failure_reason`.

The analyzer rejects incomplete dispatch-tier snapshots, missing generation or
link-state keys, mismatched trace/subset hashes, non-additive spans, or failed
correctness contracts.  A plotting script may not change denominators or
silently copy an old result into a new publication directory.

## 9. Effect on existing and future results

This file is a clarification/addendum only.  It does not modify
`final_v2/`, old paper figures, or any prior run identity.  D219 and D278 keep
their existing diagnostic boundaries; D278's readiness gate is not silently
converted into a G1/G2 result.  All future comparison manifests must record
this addendum SHA and the exact link-state stratum.

