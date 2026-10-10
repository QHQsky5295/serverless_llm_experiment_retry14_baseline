# D414: ServerlessLLM full W0 after measurement-ingress repair

Started 2026-10-11 04:46:28 Asia/Shanghai, tmux `tc-d414-7b`.
Execution baseline `35dda117faecee5375c146a2c23b16fc2562d82a`, main
`2860b53740bfea5fc8afae7e2139a13e2855a50f`; both remote branch hashes verified.
TERMINAL by05:54:GPU released,full offline audit complete. The launch contract
below is retained history. No runtime code/policy changed after D413.

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
At launch the loader-only overlay was INSTALLED using exact D414 receipt/backups;
it is now RESTORED after real GPU release. Remote services were ACTIVE,1Gbps/full;
no cache republication/request packing. Their exact identities and health
clock are preserved in raw remote_start.log/remote_health.json.

Monitor the existing run without tests, broad scans, cleanup, remote management
or another heavyjob in business. After terminal release: restore own overlay,
stop exact owned remote units, save evidence and strict full4000 analysis.
Do not restart D411/D414 or replace failure with another same-key winner.
Then return to 7B vLLM; PrimeLoRA D371 remains sealed. All remaining comparisons,
Resident reference,W1/W2/G2,3B,ablations/sensitivities remain open.

## Completed observation and evidence boundaries

4,000offered/submitted/terminal/native-contract-valid responses,0failed,
458,224actualtokens;source,prompt,adapter/content andnative timing revalidated.
E2E/TPOT/native maximum recomputation error0ms. Common D394 joint3,925/4,000
(98.125%);TTFT3,929/4,000;TPOT3,996/4,000. Physical four-GPUstore pool closes
at16,089.762743524276GPU-s,4.022440685881069perbinding-validrequest;no openlease.
Five servedepochs do not mean five physicalcards. Mean/P95TTFT973.686780/
1,052.246361ms,mean/P95TPOT26.736734/32.929275ms,meanE2E4,000.733946ms.
132publishedHTTP deliveries,131,480,060wirebytes,3,300,789,780verifiedpayloadbytes,
3,868cache-hitrequests,0request-timepacks. All actualpath overhead retained.

Service returned0;GPUrelease/service-domainremovalconfirmed,freshGPUcensusempty.
Exact D414overlay restored;3ownedremoteunits stopped byInvocationID/portsabsent;
emptyownedauxdomain stopped. Worker/resolver/remoteevidence preserved. Service
peak57,570,185,216bytes,high/max/OOM/swap0. Protected147entries zerochange.
Existing23CPUanalysis tests passed0.095s under4GiB/zero-swap bound.

Main report D414_SERVERLESSLLM_FULL_W0.md andcurated
paper_results/ieee_tc/baseline_audit/d414_7b_full_w0/audit.json
SHA714a2fdb01b2d5a0e2e6f3a9b0faf7cb23b29632efbf5b343822b7dfe4b7dd74.
237-fileevidence_manifest SHA8912ab5259d7b10e990ab3b6233b789ce4432c0af2c1544498b7587b700904fd.
Raw n_correct=null/independentnumericalflags unchanged;bounded D406/D409evidence
reused separately. Not unseen/formal or a cross-scenario optimum;cross-system
ranking still requires full contractjoin. Preserve D411failure andall attempts.
No same-key rerun,localparametersearch or3Bjump. Next7BvLLM aftersave/backup;
its D412measurement gaps must be addressed before a new qualified fullrun.
