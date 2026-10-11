# IEEE TC execution status — D434 S-LoRA original1000 RUNNING

Updated 2026-10-11. This is a current checkpoint, NOT a replacement experiment plan.
Full earlier status is preserved byte-for-byte in EXECUTION_HISTORY_THROUGH_D433.md;
archive SHA60aeae9c98335a1b9c8d131441cb052b0351e09ac48f6b07d57ecf7ab6560d98.
earlier history through D402 remains EXECUTION_HISTORY_THROUGH_D402.md.
Search those histories and D-series reports before changing code. Their old NEXT/
RUNNING entries are historical. No approved experiment or failed evidence was removed.

## Mandatory authority and evaluation

- Read full /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md at task start,
  after compaction and before experiments; source and approved FULL_NAMES snapshot
  SHA f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5.
- Read BASELINE_EXECUTION_RULES_20261011.md. No non-PrimeLoRA performance
  optimization/configuration search; only necessary compatibility, correctness,
  measurement and safety adaptations. Reuse qualified prior points and report
  their actual outcomes, including nonfeasibility; do not call unsearched points optimal.
- Read METRIC_PROTOCOL_FROZEN_V1.md and METRIC_PROTOCOL_SINGLE_RUN_V2.md
  before configuration/comparison. Their SHAs remain respectively
  5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22 and
  193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
- G1: all4000 native/source/adapter contracts and common joint TTFT/TPOT >=95%,
  then minimum physical lifecycle GPU-s/request. G2: same + U<=.75U_ref,
  then W1/W2 P95TTFT and observed joint-SLO difference. CE supplementary.
  Relative TPOT regression alone is not failure. Raw unknown numeric flags stay unknown;
  derived V2 binding metrics require complete evidence, not HTTP200 alone.
- One performance run per frozen execution key, no repeated means/CI/significance
  or best-repeat selection. Preserve development/reuse identity and every failure.
- Order: sealed7B PrimeLoRA -> remaining7B baselines -> comparison/necessary
  PrimeLoRA optimization ->3B -> ablations/mechanisms/sensitivities.
- Full names PrimeLoRA/ServerlessLLM. No figures before near target; no13B,
  new SGLang, paper edits, new weights/trace or pool regeneration.

## Current action — monitor SAME live D434 run

Full PLAN/STATUS/V1/V2/new baseline rules read this turn. Fresh compute census empty,
no tmux, RAM available112566226944B/disk163265679360B at prelaunch. These are observations,
not later launch guarantees. User reconfirms one-time readonly delivery-cache approval:
D78/D80 already done, reuse it, no republication or new extracted pool.

D434 launched2026-10-11 22:25:08Asia/Shanghai in tmux tc-d434-7b.
Execution MAIN47be5422cbf65c8c0905f78668a83c188b7b9ccf and
BASE5059cb0b86c5a5378a772adaa25aa80bb7a1b240 pushed and remoteSHA verified.
Service primelora-tc-svc-b0ce62d358e94489b1d5958c0e83ede0.scope,
InvocationID18c0f4b4575a45be824a81b181b46476; auxiliary
primelora-tc-aux-d4340000000000000000000000000001.scope.
Notice1235966.54994207; business starts60s later. Exactly1000 request contracts
registered. Watchdogsample23:available112207171584B/service3223691264B,
disk163260563456B,high/max/OOM/swap0,no abort. Not terminal/no performance verdict.
HTTPconfigSHAfa864b420f1c33047b2ef9bcecdbb09e98c2e2b00b43a9bef65e4f36dca5fb1d;
original1000view/60IDs/122510targets verified. Nativecache2055files/62246523B.
Ownedaux0002/0003/0004 empty-tree/identity checked and stopped,pressure/swap0.
Remote unchanged publishedcache/link1000Mb/s/full;7Bclock
remote-process-monotonic:8e030525ed52454e89698c6a7e7b6287.
3B/7B/monitorInvocationIDs158b40360fdf46bb818a036493b0246c /
e0525a84e0af4598a1e715bfac1e7255 /1028fcd55b374827a52f7d705422add0.
NEXT: monitor SAME actual handle and local journals. No live remote changes,
hashscan/cleanup/tests/retries/second heavyjob or configuration search.
Do not repeat100/500 or interpret observation timeout as stopped experiment.
After actual GPU release: stop only owned remote units, preserve logs/receipts,
strict existing slora_resident_rr_v1/prefix1000 audit with D394 SLO, seal/backup,
safely reclaim only recreatable private downloaded objects, then one full4000W0.
D434 is running. Full goal ACTIVE; all remaining approved scope OPEN.

Verified wait22:27:30: same active InvocationID,27 created/sent,1000 contracts,
no native responses yet during startup; no failure/terminal receipt. Watchdog130:
available111669166080B,service14172696576B,disk150252163072B,high/max/OOM/swap0,
noabort. Below150GiB forbids NEW heavylaunch;currentrun remainsabove100GiB.
No live cleanup, remote operation, tests, hashscan, configchange or restart.
This turn made actual launch progress; current unchanged state is expected wait.

Prepared D434 recipes and preregistration in both repositories; runtime unchanged.
37 CPU contract tests passed1.858s in existing outer Python3.12 environment.
Earlier systemPython3.10 attempt failed import(numpy absent); retained log, no install
or code workaround. Fresh preflight passed and147 protected entries unchanged.
PresealSHA856204eb4f7e42fc86d11896c5cbae3a3dc2f7d1b7222863d3dcfa63b2e92907.
These prelaunch steps are complete; monitor the already launched run.

Raw new root: /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d434_20261011
Private new root: /tmp/tc-d434-7bs. Reuse D433 run/start/stop/health/freeze/preserve/seal
recipes, changing campaign paths and population only. Existing original1000 view has
60IDs/122510 targets; use D422 input identity, not guessed SHA.
Native config BASE/configs/ieee_tc/slora_7b_resident_native_v1.json SHA
2dd3a7415c5b00f0bbcf6ac7b31906da4399285e6df824ba270988015c72ae5d.
D430 source-view manifest SHA
51c6f1da21cc5070a7d71e567c10564d3066605546602705c0b1c5d8b9277004.
DP4/TP1/RR/sequential startup/packed HOST views/native swap non-BMM:
input984/context1024/tokenbudget14000/batch1024/running1000.
All500 HOST preparations after common notice; business origin notice+60, no ready shift.
CPU-only prep is not GPU-s; actual allocations/release define physical ownership.

## D433 closed — do NOT repeat

500-ID functional coverage:500 offered/submitted/terminal/native/source/adapter,
68925 native tokens,0failures,exact500IDs,four actual workers served/500-ID registries.
E2E/TPOT/native recompute error0ms. D394 TTFT268/TPOT494/joint266 of500=53.2%;
NONFEASIBLE functional view, not fullW0, ranking, reference or optimum.
U1743.8732772790827 GPU-s,3.487746554558165/request.
Fullpool500HTTP/420740410wireB/12985984450materializedB,0packs.
Deployment bytes counted once; per-request references are not extra transfers.
Peakservice62410342400B,high/max/OOM/swap0; all physical leases closed.
D433 report: D433_SLORA_500_ADAPTER_COVERAGE.md.
Curated baseline_audit/d433_7b_slora_coverage/audit.json SHA
7cc9f8e220e0b5cf761cd5c416e6d3126d22f42cb77f46494af10341feedf3f7.
576-file evidence_manifest SHA
cb03ba5ddf53241e427712c285197bdcb3f89def15c2480f45e6872a4b20003c.
All576 source hashes verified.37CPU tests passed1.636s before launch.
Execution MAIN627aebff3fdaf67822946e52eacb7c9291f2eb30,
BASE15077ab4aeeac9532b1cb8d9f18567d76883163c.
Closure and cleanup backups subsequently pushed/verified; latest MAIN
29e98482d0375181fb020f53747d29ecdb136336,
BASE219ad363a83e44487d5df4831861a0edad7e3d2f.
All GPU/remote/aux scopes closed by actual identity and empty process trees.
Only /tmp/tc-d433-7bs/remote_artifacts/objects reclaimed after references, retained
source bytes/inodes/hash and process checks:5000files,12985984450payloadB,
freed13004427264B. Originals/raw/remote cache retained. Three unreadable sameUID
process entries disclosed. Do not repeat cleanup or redownload to repeat a passed audit.
Receipt paper_results/ieee_tc/storage/20261011_d433_download_cache.json.
D432 original100 passed; bounded D385 numerical witness95exact/5differences,
3known margins/2unknown retained; no full numeric proof or exact-token gate.

## Sealed W0 comparison and reference

D397 seals PrimeLoRA D371 reused-development W0, not independent holdout:
4000/132IDs/458224 native tokens,97.70% joint,3.6206309810630626 GPU-s/request.
D414 ServerlessLLM98.125%,4.022440685881069;D423 vLLM99.475%,4.028844290537294.
PrimeLoRA GPU-s about10% lower, but P95TTFT2837.896ms vs1052.246/1200.507ms
and jointSLO not leading. All meet95%; not all-metric/cross-scenario/G2 optimality.
Allthree132HTTP/131480060wireB: no W0 remote-byte-reduction evidence.
See D424_MATCHED_7B_W0.md and paper_results/ieee_tc/main_comparison/d424_7b_w0/.
W0 U_ref16115.377162149176,Q_ref1200.5068596918136ms,B12086.532871611882GPU-s.
W1/W2 references and G2 OPEN. D395/D396 cannot set reference due measurement gaps.
D394 SLO manifest paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json,
SHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3.
<=616input:TTFT2995.348713684507ms/TPOT59.74409694902162ms;
>616:TTFT4633.345350589479ms/TPOT85.24698166015581ms. Do not regenerate.
Mostly-zero/repeated weights:500logical IDs,498zero/2nonzero,4weightSHA/6tree classes.
Native bindings, independent numerical evidence and content diversity are distinct.

## Safety, ownership, storage and backups

- MAIN=/home/qhq/serverless_llm_experiment_retry14_baseline,branch retry14_continuous_queue_v2,
  remote faaslora_origin. BASE=/home/qhq/serverless_llm_baselines,main,origin.
- One heavy job. Disk start>=max(150GiB,100GiB+1.5growth), runtimefloor100GiB.
  Service memory.high72/max80GiB,swap2GiB,CPUs4-23,28-47.
  Auxhigh3/max4GiB,swap0,CPUs2,3,26,27. Actual workers contained before init.
  Separate replay/watchdog; stop only exact owned units after identity checks.
- Strict SSH alias primelora-artifact-174=lab14@10.199.227.174:8122.
  Data192.168.4.174ports18080/18081,inference192.168.4.178;credentials outside Git.
  Token file /home/qhq/.config/primelora-tc-d75/artifact.token, never print.
  Current pre-D433 link1000Mb/s/full; historical100Mbps separate state, cause unknown.
- D78/D80 published gzip reused, no request packing. Real reads/transfers/competition/
  local materialization counted. No live remote administration/hashscan/cleanup,
  tests/second heavy job. Never subtract cumulative spans to invent corrected latency.
- Original7B pool /home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix;
  original seed42 trace and existing environments reused.
- Protected147-entry seal D433raw/protected_before.json. Historical results unchanged.
  NEVER stage MAIN configs/generated/lora_manifest_1000.json SHA
  3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df;
  BASE scripts/replay_openai_trace.py SHA
  04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
  scripts/run_serverlessllm_relayserve_continuation.sh SHA
  337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
  Preserve nestedS-LoRA ten historical patches/default diff SHA
  7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60.
  Preserve all other unrelated tracked/untracked files.
- Explicit task-only staging, diff/tests/source/secrets checks; no force/rawlarge/secrets.
  Update current status before yielding. Do not claim full goal complete.

## Remaining full approved scope — not canceled

Remaining7B baseline qualifications/comparisons, W1/W2/G2 and references, PrimeLoRA
optimization where justified,3B optimization/comparisons,dLoRA3B,ElasticLocality,
Loquetier/HydraServe native qualification, all A1-A5 and S1-S13, CE/cost sensitivity,
two-review action/design documents and later academic figures. Reuse only same contracts.
B1/B2/C1/C2/C3 and formula/novelty evidence remain required. Hardware limits disclosed.
Full original plan + preserved history define every setting; this tracker is not a shortened plan.
