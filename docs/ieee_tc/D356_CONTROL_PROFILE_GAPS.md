# D356: supplement six measured class gaps, not a second optimization

Mainline: 7B optimization → 3B → paused baseline matrix. The source plan and
frozen metric V1 remain unchanged. No new weights, prompts, traces, figures,
runtime policy, concurrency capacity, SLO threshold or timing estimator.

## Observation and interpretation

D356 source measurement completed 1,569 native calls and 125 waves. The measured
service domain is 194/200; preparation coverage is complete (24 classes).
All six missing classes are short declared output (≤24), admission bin 9–14,
rank 8 or 16, from native HOST, verified file HOST, or verified NVMe.
For these short-output/source/rank groups, the observed admitted counts range
from 1 to 6. Intended wave size is not actual admitted count. Earlier requests
can finish while subsequent requests are being admitted, particularly under
the new native-loop interleave; this interpretation does not establish causality.

The original source measurement, SHA
`1be8c0b963176599e4f1a3096fe250046383697675efff8be83d29ebd80b0b12`,
and its complete coverage/counter audit remain immutable. A profile must not
claim coverage from intended lanes, reuse old-policy latency, or fill defaults.

## Minimal complementary measurement

Reuse the existing D344 long-background/short-target selector and D157 native
source collector. Extend only the selector's controlled source identity:

- Eight existing long-output background requests (the missing admission bin's
  lower edge) precede the two existing short targets, one of each rank.
- Three measured sources: native HOST, verified file HOST, verified NVMe.
- Three interleaved rounds; alternate target order and rotate source order.
- One retained GPU warmup wave; 10 waves × 10 calls = 100 calls, 90 measured.
- Every prompt, adapter identity and output target is indexed from the original
  first-1,000-request population. No synthesized requests or changed lengths.
- No artificial delay, admission barrier, held-open completion or extra GPU step.
  The collector records actual post-admission class and joins all tasks.

The shared published remote objects and prior configuration are reused. Source
setup is explicit controlled calibration, not a representative workload or
warm SLO reference. Original calibration and supplement have separate clocks,
resource lifetimes and run identities; only compatible measured classes merge.
All samples, including warmup and any missed target class, are retained.

## Checks and next action

150 CPU tests passed before launch, including original GPU-gap compatibility,
HOST/NVMe source/representation rejection, capacity/missing-input rejection,
input immutability, native interleave and source-profile validation. Preparation
confirms the same child model/configuration and completed-length initialization.

Use the existing bounded launcher, actual GPU release and bilateral transfer
closure. Audit native counts, source state, actual class coverage, physical
resources and control counters before merging. Export only if the merged
domain is complete. No source calibration establishes Full or formal G1/G2.
Then retain control counters in existing Full admission receipts, validate
Full 100 requests, and return to the complete W0 replay. Baselines stay paused.

Evidence: `paper_results/ieee_tc/p2_backend/d356_control_profiles/`; supplement
uses its new `gaps1/` directory. The 835e186 checkpoint preserves the original
measurement. Old submission results and user manifest are untouched.
