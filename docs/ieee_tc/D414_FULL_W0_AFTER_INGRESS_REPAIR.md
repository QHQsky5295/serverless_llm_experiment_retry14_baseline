# D414: ServerlessLLM full W0 after measurement-ingress repair

Started 2026-10-11 04:46:28 Asia/Shanghai, tmux `tc-d414-7b`.
Execution baseline `35dda117faecee5375c146a2c23b16fc2562d82a`, main
`2860b53740bfea5fc8afae7e2139a13e2855a50f`; both remote branch hashes verified.
This document is a running checkpoint, not a performance result. No runtime
code or policy changed since the D413 CPU-qualified checkpoint.

Raw root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d414_20261011`.
Main contract: `docs/ieee_tc/D414_SERVERLESSLLM_FULL_W0.md` in the main repo;
current status: main `docs/ieee_tc/EXECUTION_STATUS.md`. Read full authoritative
PrimeLoRA-PLAN.md and status after compaction, not this short note alone.

Same original4000/source42/development41/nonformal W0 and native configuration
as D411. Only new service-owned extraingress connection lifetime from D413:
single_request_loopback_no_retry_v1. No retries, timeout retuning, new workload,
policy transplant, parameter sweep or repeated short GPU qualifications.
D411 failures remain in full. D406 whole-pool and bounded D409 evidence reused.

Service scope `primelora-tc-svc-6fc1e91952774c4dbf89415c84e35a60.scope`,
InvocationID `2a7b6ec60c284d0981714c35296b037e`. Aux
`primelora-tc-aux-d4140000000000000000000000000001.scope`,
InvocationID `018dc11883334acf8eddaa38a6f921db`.
Deployment notice1172460.705107463,business t0=1172520.705107463.
Loader-only overlay INSTALLED using exact D414 receipt/backups, to restore
only after real GPU release. Remote published services ACTIVE,1Gbps/full;
no cache republication/request packing. Their exact identities and health
clock are preserved in raw remote_start.log/remote_health.json.

Monitor the existing run without tests, broad scans, cleanup, remote management
or another heavyjob in business. After terminal release: restore own overlay,
stop exact owned remote units, save evidence and strict full4000 analysis.
Do not restart D411/D414 or replace failure with another same-key winner.
Then return to 7B vLLM; PrimeLoRA D371 remains sealed. All remaining comparisons,
Resident reference,W1/W2/G2,3B,ablations/sensitivities remain open.
