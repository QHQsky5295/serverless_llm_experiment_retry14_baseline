# D435 — S-LoRA 7B complete W0, preregistered single observation

2026-10-11 launch22:58:29Asia/Shanghai; completed and audited2026-10-12.
Execution MAIN84d6b4d0607944c5d6ef47bd9ada2d30817e4887 and
BASEcae13c57c012ea55a8199740f2696b9db732362f. No native configuration change.

## Closed outcome

Original4000 W0 completed once,0failures; all4000 native/source/adapter/timing
contracts,458224native output tokens,132requested IDs. Four actual500-ID native
workers served, physical leases released. Common D394 TTFT3783/4000,
TPOT3994/4000,joint3777/4000=94.425%: observed nonfeasible under95%, not a reason
to tune this baseline or repeat the same performance key. U15157.07787045301
GPU-s,3.7892694676132526per binding-valid request. Mean/P95TTFT7396.416512/
17699.067366ms,mean/P95TPOT31.251644/45.953041ms,meanE2E10962.479328ms.
All217 TTFT violations lie in the first500; later3500 have3494joint-good.
Do not substitute this descriptive warm slice for the full population.

Full500 preparation500HTTP/420740410wireB/12985984450materializedB,0request-time
packs; deployment references counted once. Native timing recomputation errors0ms.
Servicepeak62637105152B,high/max/OOM/swap0. Actual GPU release/service removal
and exact-owned remote stop verified. No run repeated; raw unknown numerical
correctness flags remain unchanged. Not a full independent numerical proof or
cross-scenario optimum; a separate matched-system join is still required.

Main curated directory `paper_results/ieee_tc/baseline_audit/d435_7b_slora_full_w0/`:
auditSHA1c9361f2d995c27e0672611728697f4de48bd0ff3c4e9f4fd778523565b6e4c0,
manifestSHAd0f6b8b4aca2ac8b34f13230e9ebd559982ed7bce684441c7c5a998e8b0886c2.
All578 source hashes/sizes rechecked,227275259retained source bytes;147protected
entries and dirty user sources unchanged. Main D435 report has full interpretation.
Closure backed up; owned recreatable downloads subsequently reclaimed with source/
metadata/reference checks,5000files/12985984450payloadB,free162938236928B.
All578 sealed source hashes and147protected entries still match. Main receipt:
paper_results/ieee_tc/storage/20261012_d435_download_cache.json. No unique data removed.
Next: separate matched W0 join and remaining7B qualifications, fresh safety gates;
no further S-LoRA tuning/repeat and no3B jump.

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
