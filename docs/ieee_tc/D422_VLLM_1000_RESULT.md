# D422 native vLLM 7B original-prefix compatibility

2026-10-11, completed and actual GPU contexts released. Unchanged baseline
execution commit `d9971a14e18eca3391665d9da1244594e69b6041`; native config SHA
`361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5`.
Main execution `9f334caab8a2c305f7e29909532a65f7cc8db1d1`.
No runtime code/config optimization or performance retry in this gate.

Exclusive raw root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d422_20261011`.
Existing original trace_prefix_v1,1000requests/60IDs/122510native tokens;
viewSHA `0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981`.
1000/1000 offered/submitted/terminal/native/source/adapter/content/timing
contracts,zero failures,E2E/TPOT/native reconstruction errors0ms.

Four actual workers/closed physical leases,U4601.208847534144GPU-s.
Common D394 joint991/1000=99.1%,TTFT991,TPOT1000;mean/P95TTFT455.189460/
2118.832637ms,mean/P95TPOT24.974164/27.243896ms,E2Emean3490.662821ms.
Short compatibility observation only,not fullW0 comparison/U_ref/config optimum.
60published HTTP fetches,74953014wireB,1474429899verifiedpayloadB,
940cachehits,0request-time packs. Existing628-file compiler diskcache recorded
separately from cold LoRA cache;all startup and GPU holding counted.

Bounded original first100 D385 HF/PEFT witness reused:97exact sequences,
3differences with existing first-divergence/top2 witnesses;93same-as-D419,
7changed,all rows retained. No newGPU oracle/exact-match qualification rule,
no independent numeric proof inferred for other900,raw unknown fields unchanged.
Mostly-zero/repeated weights remain a limitation. D421500-ID qualification reused.

Peak service12567740416B,hostavailablemin100156542976B,high/max/OOM/swap0.
Actual GPU release/service path removal/exit0;owned remote units stopped by
InvocationID and MainPID0/ports absent;empty owned auxiliary stopped.
147protected entries unchanged. Downloaded object cache not yet reclaimed.

Main audit: `paper_results/ieee_tc/baseline_audit/d422_7b_compatibility/audit.json`,
SHA `43f07985abd12189288597587fc0128939c779e1dad65f6875afbdfe82b81517`.
133-file evidence manifest SHA
`42f06cbed073ec760b056db59fef4807e57187202f086c78c0db51f5d108abc8`.
Next: backup closure,then one full4000 W0 with unchanged native configuration.
Do not repeat100/500/1000 or retune sealed PrimeLoRA/ServerlessLLM observations.
