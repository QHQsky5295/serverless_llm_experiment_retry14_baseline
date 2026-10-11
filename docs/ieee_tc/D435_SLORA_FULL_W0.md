# D435 — S-LoRA 7B complete W0, preregistered single observation

2026-10-11. Launched22:58:29Asia/Shanghai; no terminal performance result yet.
Execution MAIN84d6b4d0607944c5d6ef47bd9ada2d30817e4887 and
BASEcae13c57c012ea55a8199740f2696b9db732362f, both pushed/remoteSHA verified.
Tmux tc-d435-7b; service InvocationIDfd0490c67700443fbe7238bb4d06578a.
Remote link1000Mb/s/full, published cache reused; input/cache freeze passed.
Monitor same live handle without restart or baseline tuning.

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
