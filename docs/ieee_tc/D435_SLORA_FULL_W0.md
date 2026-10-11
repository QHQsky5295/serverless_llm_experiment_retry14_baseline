# D435 — S-LoRA 7B complete W0, preregistered single observation

2026-10-11 launch22:58:29Asia/Shanghai; completed and audited2026-10-12.
Execution MAIN84d6b4d0607944c5d6ef47bd9ada2d30817e4887 and
BASEcae13c57c012ea55a8199740f2696b9db732362f, both pushed/remoteSHA verified.
Tmux tc-d435-7b; service InvocationIDfd0490c67700443fbe7238bb4d06578a.
Remote link1000Mb/s/full, published cache reused; input/cache freeze passed.
The same handle completed without restart or baseline tuning; see outcome below.

## Purpose and immutable comparison contract

Advance the approved sequence after D432 original100, D433 all500-ID coverage,
and D434 original1000 compatibility. Run the original4000 W0 once, using the
same native S-LoRA configuration. D434 joint SLO74.9% does not trigger baseline
tuning, repetition, warm-only reporting or threshold changes.

Inputs: original seed42 source SHA
efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c;
full W0 view SHA a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d;
4000 requests,132 requested logical adapters,458224 target output tokens.
Use existing frozen500 pool/index SHA
e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
No new trace, prompt or weights. The pool's498zero/2nonzero adapters and limited
independent weight diversity remain disclosed. Native request/source/adapter
binding and bounded numerical evidence are distinct; unknown raw flags stay unknown.

Native config configs/ieee_tc/slora_7b_resident_native_v1.json SHA
2dd3a7415c5b00f0bbcf6ac7b31906da4399285e6df824ba270988015c72ae5d,
D430 source-view manifest SHA
51c6f1da21cc5070a7d71e567c10564d3066605546602705c0b1c5d8b9277004.
Same DP4/TP1/RR, sequential startup, D428 packed HOST views, native paging/swap,
input984/context1024/tokenbudget14000/batch1024/running1000; no performance search.
Full500 static HOST preparation only after notice. Business starts notice+60s,
without waiting for all workers ready; actual native GPU allocations/releases
define lifecycle usage. Native SSE integer token IDs and original event times,
fixed_length_greedy_v1, no retokenized or expected-count substitution.

Same metric V1 plus single-run V2 and frozen D394 common SLO SHA
2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3.
All4000 work contracts plus joint SLO>=95% are required for feasible G1 comparison.
Any nonfeasibility remains visible; not an optimized baseline or W1/W2/G2 result.
Qualification/development identity41 and formal_run=false remain explicit. After
complete audit, a derived single observed W0 comparison may be made with sealed
PrimeLoRA/ServerlessLLM/vLLM, without changing raw identity or their prior results.
S-LoRA does not replace the frozen Resident-vLLM reference or set new SLO values.

## Execution, safety and evidence

Exclusive raw root:
/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011

Private root /tmp/tc-d435-7bs; new metadata only. Reuse D434 run/start/stop/health/
freeze/preserve/seal recipes; only campaign paths and population differ.
No runtime/analyzer/scheduler changes. Existing Triton cache identity is recorded,
not an engine preload. Qualification protection1800s remains unchanged.

One heavy run. Service72/80GiB,swap2GiB,CPUs4-23,28-47; auxiliary3/4GiB,swap0,
CPUs2,3,26,27. Actual workers enclosed before initialization, independent watchdog,
150GiB start and100GiB runtime disk floor. Predicted growth20GiB includes full500
materialization and logs. Fresh GPU/protected-result/input/resource gates first.
D434 verified recreatable objects alone were reclaimed; its raw evidence and
remote D78/D80 once-only published cache remain. No republishing or extracted pool.
Real reads/transfers/contention/local materialization count; no request-time packing.
No remote administrative work, scans/cleanup/tests or concurrent heavy work during run.

Before launch back up task-only preregistration in both corresponding repositories.
After completion: actual GPU release -> exact remote identity stop -> preserve
dual-end journals/receipts -> strict slora_resident_rr_v1/full4000 analyzer ->
historical protection and evidence manifest -> backup -> comparison/next main-line task.
Do not pass the prefix100-only numeric reference argument to full4000.
Report all4000 and descriptive source-order8x500 phases; phases are not independent
repetitions. Full500 deployment transfers count once, not per referring request.
No figures or paper edits here; remaining7B baselines, necessary PrimeLoRA optimization,
3B and all approved ablations/mechanisms/sensitivities remain open.

Prelaunch CPU37 tests pass1.696s. Fresh preflight passed;147protected entries
unchanged. D434578 source-file hashes verified after object cleanup; baseline
runtime/config and protected user-file identities unchanged. No test install or
code workaround. Execution commits will be sealed by freeze_inputs.py after backup.

## Completed single-run outcome — 2026-10-12

The original W0 completed once. All4000 offered/submitted/terminal responses passed
the existing strict native token, source, adapter/content and timing checks;
458224 actual native output tokens,132 requested IDs,0failed requests. All four
native500-ID workers served. E2E/TPOT/native recomputation errors are0ms.
Raw `n_correct=null` and incomplete independent numerical-proof flags remain
unchanged; this is a full native-binding development observation, not an unseen
evaluation or universal arithmetic proof.

| Metric | Observed value |
|---|---:|
| Common TTFT SLO | 3783/4000 (94.575%) |
| Common TPOT SLO | 3994/4000 (99.85%) |
| Common joint SLO | 3777/4000 (94.425%) |
| Physical lifecycle GPU-s | 15157.07787045301 |
| GPU-s / binding-valid request | 3.7892694676132526 |
| Mean / Type-1 P95 TTFT | 7396.416512 / 17699.067366 ms |
| Mean / Type-1 P95 TPOT | 31.251644 / 45.953041 ms |
| Mean E2E (external HTTP completion) | 10962.479328 ms |

**Observed nonfeasible under the frozen95% joint requirement.** The23-request
shortfall is not rounded away. No baseline tuning, repeated performance key,
threshold change or startup-free replacement follows from this outcome.
All217 TTFT violations occur among source indices0–499; this first500 has
283joint-good requests, while the subsequent3500 has3494joint-good. These are
descriptive slices of one run, not independent repetitions or a warm-only main
result. They support startup concentration of the observed delay, not a claim
that every S-LoRA deployment must exhibit the same result.

Physical windows: pre-arrival0, arrival15129.519531617174, drain1.9117999318987131,
cleanup25.646538903936744GPU-s. CPU-only preparation is not invented GPU time;
actual native allocations through actual exit determine the total. Full500
deployment preparation made500HTTP transfers/420740410wire bytes and verified
12985984450materialized bytes, with0request-time packs. The4000 preparation
references are not4000additional transfers. Published D78/D80 objects were reused.

Service peak62637105152B, host minimum available61200150528B, disk minimum
150091988992B; high/max/OOM/swap0. Service/replay/watchdog returncodes0, actual
GPU contexts released and service path removed. Owned remote3B/7B/monitor units
stopped only after matching InvocationIDs, with MainPID0 and ports absent;
dual-end journals and resolver receipts preserved, no object-pool copy.
All147 protected entries and the three dirty user-file hashes remain unchanged;
the native ten-file historical patch also remains unchanged.

Curated main-repository directory:
`paper_results/ieee_tc/baseline_audit/d435_7b_slora_full_w0/`.
Audit SHA `1c9361f2d995c27e0672611728697f4de48bd0ff3c4e9f4fd778523565b6e4c0`.
Evidence manifest SHA
`d0f6b8b4aca2ac8b34f13230e9ebd559982ed7bce684441c7c5a998e8b0886c2`.
All578 source-file hashes and sizes independently rechecked;227275259retained
source bytes. No independent-review-agent PASS is asserted by these checks.

Closure backed up to MAIN212cec45ad5a31d4363356a205a13350210637e4 and
BASEf1e7310a16ac40bfbdd9f25d047742bc46c25f9a, both remoteSHA verified.
Next: perform the separate full input/generation/resource
join with sealed PrimeLoRA/ServerlessLLM/vLLM, then continue remaining7B baseline
qualification. Do not overwrite D424 or refreeze its Resident/SLO reference.
After backup and reference checks, only owned recreatable downloads were reclaimed:
5000files/12985984450payloadB, observed freed13004468224B, free162938236928B.
All578 sealed source hashes and147protected entries still match after cleanup.
Receipt `paper_results/ieee_tc/storage/20261012_d435_download_cache.json` records
retained originals, reference-check scope and three unreadable same-UID processes.
Cleanup-only memory.high3416/maxOOMswap0 is separate from the performance record.
All D435 auxiliary scopes closed; fresh complete safety gates still precede any
next heavy run. Do not redownload to repeat already passed historical audits.
W1/W2, G2, Loquetier/HydraServe,3B and the full approved matrix remain open.
