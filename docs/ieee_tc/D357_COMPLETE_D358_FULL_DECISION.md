# D357 complete first100 → D358 unchanged Full4000

D357 executed at d861130722b7ad1151ee8549d29fdb3a3462560a. Source SHA
e7a7d50c727c49b26f3d569ec1d9778d377eebd8ddb73288f83293f0d4eb6674,
7,314,012 bytes.100/100 native contracts and content/control bindings passed.
All measured requests completed, but numerical N_correct and formal G1/G2
qualification remain unknown. This is one development prefix, no confidence
interval or extrapolation to complete W0.

| Diagnostic | D357 |
|---|---:|
| Physical lifecycle GPU-s | 654.7159873729106 |
| Mean / P95 TTFT (ms) | 799.6601438790094 / 1790.3715358115733 |
| Mean / P95 TPOT (ms) | 26.61758728932433 / 33.61822651688951 |
| D170 provisional timing-only joint | 98/100 |
| TTFT-only / TPOT-only / joint misses | 2 / 0 / 0 |
| Remote transfers / wire bytes | 29 / 50,650,865 |
| Request-time packing | 0 |
| Service memory peak bytes | 18,468,663,296 |
| Minimum host available bytes | 93,995,520,000 |
| high / max / OOM / swap | 0 / 0 / 0 / 0 |

Four native owners retained100final registration receipts, each bound to its
retirement owner, TP1 worker PID and monotonic clock. Recorded per-owner yield
deltas are753/336/213/7; queued-input yield deltas206/100/40/0. The zero queued
owner is retained. Counts describe intervals between retained snapshots, not
total model steps, GPU idle time, RPC savings or causal improvement.

The actual Full trigger gate passed. The CPU audit suite passed41checks after
the run (169checks before launch, including runtime fixtures). Local service
and GPU contexts released; exact-owned remote units stopped once, bilateral
transfer records match. No model/data/pool regeneration or old result overwrite.

Decision: proceed to D358 ordinary4000W0 with identical runtime/controller,
profiles, input and generation contracts. Only fresh output/cache paths and
the diagnostic-prefix removal differ. The100-prefix is not a performance
selection. D352 prefix remains descriptive context, not a single-factor proof.

Full-run questions: versus D353, does native control interleaving reduce the
short-output TPOT misses without worsening lifecycle GPU occupation? Preserve
the D349/D336 historical context; do not close the7B goal merely by beating the
last candidate. D353 previously had3752/4000 timing passes and13326.57089GPU-s;
D336 had3837/4000 and13814.33450GPU-s under different capacity/profile settings.
Both are diagnostics, not formal warm/Resident/numerical qualifications.

The original source hypothesis and official vLLM0.30.0 references remain in
D357_FULL_CONTROL_VALIDATION.md. No new scheduling formula or policy was added
in this handoff. Remaining mainline:7B →3B →paused baselines →M1/M2 →ablations
and sensitivities. No performance figures before the optimization target.
