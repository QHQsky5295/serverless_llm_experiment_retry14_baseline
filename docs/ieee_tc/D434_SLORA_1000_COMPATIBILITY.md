# D434 — S-LoRA 7B original1000 compatibility

2026-10-11 preregistration; no performance outcome yet.

## Question and fixed scope

After D432 native100 and D433 all500-ID functionality, execute the missing original
1000-request compatibility view once. Reuse original seed42 trace, first1000 entries,
60 requested logical adapters,122510 target output tokens. No remapping, regenerated
trace/weights or baseline performance tuning. This is qualification/development,
not full4000 W0, a baseline optimum, G1/G2 ranking or independent holdout evidence.

Native config remains slora_7b_resident_native_v1.json SHA
2dd3a7415c5b00f0bbcf6ac7b31906da4399285e6df824ba270988015c72ae5d.
DP4/TP1, original ordinal round-robin, sequential worker startup, native S-LoRA
paging/swap/scheduling, D428 packed HOST views and D430 measurement/source view.
Full500 static HOST preparation starts after notice; business starts60s after notice
without waiting for ready. Actual native allocations and release define GPU-s.
No changes to baseline runtime, scheduler, parameter values or qualification thresholds.

## Input, resource and remote contract

View SHA 0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981,
source SHA efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c.
Same fixed_length_greedy_v1/canonical prompts/native SSE token IDs.
Frozen D394 common SLO, frozen metric V1 + single-run V2; new baseline execution rules.
Independent numeric limitations remain; raw unknown fields are not promoted.
The original1000 view does not reuse D433's adapter-remapped performance numbers.

Exclusive raw root:
 /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d434_20261011
Private root: /tmp/tc-d434-7bs. Reuse D433 orchestration, changing paths and population
only. No previously successful100/500 gate repeated. Existing Triton disk cache is
sealed separately; no prestarted GPU engine or free adapter preload.

One heavy job; service72/80GiB,swap2GiB,CPUs4-23,28-47; auxiliary3/4GiB,zero swap,
CPUs2,3,26,27; independent watchdog and preflight disk150/100GiB rules.
Reuse D78/D80 readonly published gzip cache. User reapproved once-only cache;
do not republish or duplicate extracted pool. Actual remote reads/transmission/
contention and local materialization remain measured; request-time packing absent.
No remote changes, hashing/cleanup/tests or concurrent heavy work during service.

## Acceptance and continuation

All1000 offered/submitted/terminal/native/source/adapter/time contracts, no fallback,
actual_tokens==target for every request; complete original-view identity.
Physical actual-worker leases closed; mandatory timing recomputation within1ms.
Full500 deployment transfers count once, not once per referring request.
Report common joint SLO even if below95%; that does not trigger baseline tuning.
Use existing audit_d402_serverlessllm_path.py with scheduler slora_resident_rr_v1,
population prefix1000 and common D394 manifest. Do not apply prefix100-only numerical
reference argument to this full1000 population or invent a new GPU oracle.

After actual GPU release: close only owned remote units, preserve receipts/journals,
audit/protect/seal and back up. Then safely reclaim only proven recreatable private
downloads if needed; fresh gates before one full4000 W0 using the same configuration.
PrimeLoRA/ServerlessLLM/vLLM sealed W0 points stay unchanged;3B and later fullmatrix
remain open. No figures or paper edits in this qualification step.

Prelaunch CPU checks:37 passed1.858s with existing Python3.12 environment.
Initial systemPython3.10 attempt had a missing numpy import; failure retained,
correct interpreter reused, no dependency installation or runtime modification.
Fresh resource preflight passed;147 protected historical entries unchanged.
