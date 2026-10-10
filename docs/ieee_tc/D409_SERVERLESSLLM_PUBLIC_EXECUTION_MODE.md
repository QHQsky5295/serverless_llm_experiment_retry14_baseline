# D409: explicit public ServerlessLLM execution-mode candidate

2026-10-11. CPU-qualified interface; the initial status below is retained.
The affected-path GPU gate subsequently completed; see closure at the end.

D408 used the original 1,000-request W0 prefix, target4/min1/max4/keep10,
native ServerlessLLM loader and published remote LoRA objects. All1,000 native
binding/count/timing contracts passed, but common joint SLO was674/1,000.
Mean TTFT4,683.001ms and TPOT63.562ms. Post-startup descriptive slices retained
TPOT failures. Preserve the completed candidate, not an automatically tuned
optimum. Main detailed report: `docs/ieee_tc/D408_SERVERLESSLLM_1000_RESULT.md`.

The qualification helper unconditionally selected eager execution. The pinned
official backend allows `enforce_eager` and defaults it to false; vLLM0.10.2
documents graph/eager hybrid execution when false. This supports a falsifiable
candidate: allowing native graph execution may reduce persistent decode cost,
but capture/compile may increase startup and memory. It is not yet a measured
cause. Sources checked2026-10-11:
[ServerlessLLM pinned backend](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/vllm_backend.py),
[vLLM versioned arguments](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html#enforce-eager).

The existing helper now exposes `qualify-model --no-enforce-eager`; the default
remains true to preserve every existing recipe. The config builder accepts only
real booleans and records the effective mode in configuration.json. Tests prove
this option changes exactly one public backend field; CLI help exposes it.
No policy/fast-loader change, eager fallback, automatic retries, new environment,
weights, trace or copied pool. Actual native graph execution still needs evidence.

89 existing-plus-new CPU tests passed in7.234s, four-GiB/zero-swap scope:
`tests.test_ieee_tc_serverless_launch`, `tests.test_ieee_tc_serverless_measurement`,
`tests.test_ieee_tc_serverless_router`. The tool transcript contains the result;
no separate raw unittest log is claimed.

Next: recheck full plan/status and safety, run one affected-path native100 gate
under a new key with only this config difference, then original1000 validation
if valid. Reuse D406's unchanged content inventory/500-ID coverage evidence;
native graph adapter-slot execution must be verified, not presumed from eager.
Do not expand graph shapes, concurrency, target or memory simultaneously.
Retain common60s notice,1800s validation protection,actual GPU lifecycle and all
failures. The candidate is not allowed to skip capture cost or silently relabel
eager output as graph output. If invalid, diagnose once from specific evidence,
not a blind retry. Then return to full4000W0 and the remaining baseline mainline.

## GPU affected-path closure, 2026-10-11

One original100-request run completed at02:27 Asia/Shanghai. Main execution
5a3dac21ffe52e0651c988584feecda003f1a64d,baseline execution
d6dcf14449e595f461bb933007b62c87423377d2. Raw:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d409_20261011`.

100native/source/adapter/timing contracts,zero failures,17369native tokens,
4served engines. Actual four PIECEWISE captures completed;logs show2/9/9/10s
capture and.09GiB each. Not a per-request claim that every operator used graphs.
TTFTmean63240.115/P95111377.364ms;TPOTmean31.700/P9534.571ms. CommonD394
TTFT0/100,TPOT100/100,joint0/100: NONFEASIBLE short prefix,not fullW0.
Ready receipts atnotice+200.191..213.958s;commonarrivalnotice+60s unchanged.
GPU-s1033.029004,complete actual releases.29HTTP/50650865wireB/724665100verified
payloadB,71cache-hit requests,zero request packing/republication.
HF/PEFT reference reused offline:97exact sequences,3retained differences with
existing first-divergence/top2 evidence. Unknown raw numerical flags unchanged.

Full audit remains in main repository:
`paper_results/ieee_tc/baseline_audit/d409_7b_graph/audit_v2.json`, SHA
9c35cbb2661567d718f9671e605cf69b38099fdbcefbebbf9c688f2200db1218.
Main report `docs/ieee_tc/D409_SERVERLESSLLM_GRAPH_RESULT.md` includes sources,
limits,resource decomposition and cleanup. AllGPU/service/aux/ownedremote units
released;loader overlay restored.15analysisCPUtests pass0.052s.

Next original1000 validation,then full4000;do not repeat100or expand unrelated
parameters. Native compilation diskcache fromqualification is retained and must
be explicitly recorded before the next run,distinct from cold per-run LoRA cache.
No offline GPU prewarming,request-time packing,new pool or hidden retry.
