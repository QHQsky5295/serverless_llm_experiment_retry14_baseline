# IEEE TC execution status

## Authority and recovery

- Authoritative plan: /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md
- Approved snapshot: PLAN_APPROVED_20260925.md
- Plan SHA256: a9ed2d8136073f5a5d5c8088489e1d8642b916a610fc7e9d3b3ff8eaf35301b0
- User authorized execution, milestone Git backups and recurring paper-oriented
  progress reports. Read full plan and this current ledger before work, after
  compaction and before experiments. Preserve equations, data and fair comparison.
- Main branch retry14_continuous_queue_v2; baseline main. No agents authorized.
- Main starting commit7ebb1ca9688362736710d94db78ec416a88a2384; baseline
  startingd85263e00e976cc61c8938e662d77a2b249cdf58. Goal remains incomplete.
- D67 start main319487d6bfc6e7a88379caee801585d2406802aa, baseline9b12427ca7bcf8b7f20771fe0ecd671930029cf6.
- Resume by inspecting actual handles/Git/resources; never restart a live run
  from stale notes. No model/TMUX remains after D67; both loader overlays RESTORED.

## Mainline ledger

| Block | Status | Evidence / next |
|---|---|---|
| Safety/physical measurement | Actual Ray/worker and dedicated7B owner qualification; Full reducer connected | RESOURCE_QUALIFICATION, EXTERNAL_REPLAY_QUALIFICATION, PHYSICAL_GPU_MEASUREMENT D27/D55; actual Full multi-activation still open |
| Old results | All147 protected entries unchanged at D67 | paper_results/ieee_tc/safety/20260925_execution_start_protected.json |
| Remote174 | Strict key login works; service qualification pending | Disk138.52GiB below approved150GiB; user decision pending |
| P0 Table1/Full | 7B source-pair audit complete | Same inputs, different execution;202 scalars retained; no rerun needed for provenance finding |
| P1 IEEE alignment | Frozen h/d planning/replacement, staging/admission, observed HOST return connected | P1 throughD54, P2 throughD61; representative profiles and actual Full remain open |
| P2 backend/artifacts | vLLM0.30 installed; mechanical/numerical diagnostics; complete pool content audits | Full semantic qualification OPEN;3B500/500 and7B498/500 zero weights |
| Serverless | D67 real3B native backbone loading/4 requests/worker identity/cleanup pass | No polling-benefit, LoRA,500-pool,7B or original/repaired pair claim |
| Baseline qualification | Pending | Serverless first, then vLLM/S-LoRA/dLoRA3B/Loquetier/HydraServe |
| M1/M2, A1–A5, S1–S13 | NOT STARTED | No formal performance or optimality claim |
| Documents/figures | Design and qualification tables in progress | Formal performance figures pending |

## Immediate next action

D67 native loading is complete for its narrow3B case. Do NOT repeat the passed
Ray-only/checkpoint/allocator/four-request loader witnesses or old request prefixes.
Move toward the approved two-model original/repaired1,000-request development
pairs. First establish the necessary shared generation/LoRA instrumentation and
the7B native checkpoint identity/path, reusing prior assets. Only the3B native
checkpoint exists under the inspected baseline models/vllm namespace; audit
other referenced historical locations before any necessary conversion.
The real-remote and independent LoRA-correctness gates remain separate.
Do not silently replace native loading by ordinary HF/direct loading, relabel
backbone requests as LoRA correctness, or start M1/M2 on this evidence.

While progressing independent prerequisites, keep both unanswered user choices:
1. Remote174 has148732485632B available (~138.52GiB), below150GiB. Services18080/
   18081 are not started/qualified. User must free space or authorize a separately
   measured artifact-node peak rule. Do not delete unique/unrelated data, lower
   the floor unilaterally or repeat the same question every few minutes.
2. Existing3B weights are all zero. A few trained nonzero correctness fixtures
   (not pool replacement) need authorization, versus an explicitly limited
   claim. No new weights acquired/generated; zero controls cannot prove use.

Full still rejects legacy warmup and needs representative measured file/remote/
concurrency profiles tied to actual content classes, backend and allocator.
D26 serial samples or test constants cannot fill gaps. D60 uncached_background_v1
and fixed HOST partition are explicit candidates; actual observed capacity wakes
D61 deferred work without early eviction, assumed future bytes or budget growth.
Do not remove the Full guard. Return to actual Full/profile/lifecycle once its
prerequisites are satisfied. No more isolated allocator microloops.

## D67 actual native loading result, 2026-09-27

- Skills run-experiment/github-sync/academic-plotting read. Full plan/status and
  AGENTS read including recovery. Original user dirt preserved; no new artifact,
  trace, remote service, manuscript edit or baseline-policy change.
- Read actual TC9f50241 backend/store/native launcher and prior successful
  other-project loader evidence. Browsed original official backend. Reused
  D66 audited3B checkpoint, complete compiled store and reversible loader-only
  port; no other-project routing overlay. Baseline source2f9f6d3 pushed/full
  remote SHA verified before attempt1.
- Extended existing contained helper with qualify-model, not another framework.
  Native pool32GB,2raylets4+4GiB, common72/80GiB/swap2GiB/40CPU, explicit native
  confirmation, four store GPU UUIDs, one persistent TP1FP16 engine. Private
  TMPDIR contains native100MiB calibration. No live-migration qualification.
- Four reused seed42 request messages; explicit backbone-only targets152/59/123/
  217,759-token input cap,greedy/ignore-EOS. No adapter application claim.
  Native observed counts and input IDs/hashes preserved, not text retokenization.
- First CPU test command omitted systemd WorkingDirectory: discovery error,
  no tests run. Corrected26 tests pass. First actual model attempt succeeds in
  native GPU load confirmation and4/4 output targets, then caller-side Ray
  actor readback fails TypeError/too-many-positional-arguments. Preserved as
  protocol_or_launcher_error, NOT inference/OOM/performance failure.
- Diagnostic driver had not selected the same source/library composition as
  workers before Python startup. Inspected Ray2.54 actor reconstruction/fake
  class behavior; original unpickle exception was not recorded, so its exact
  cause is not asserted. Require correct driver imports before service launch.
  Second26 tests pass, corrected7f2de0de20d51f6bdc79cecff96d67daeb736629 pushed/
  fresh origin full SHA checked. Same model/configuration/requests; no policy fix.
- Attempt2 PASSES: actual backend PID249255, TC backend source SHA
  994c80ae9c6106a6469f9d8d5889132e5c2032608c4e497141c00127f91cc777, actual compiled
  store, native load_format/serverless_llm and checkpoint identity, expected
  cgroup/40CPU. GPU confirmation0a9cc932-fedd-410a-97ae-2969e0c0f87a.
 4/4 requests,551 native tokens. No LoRA/numerical/performance inference.
- Attempt1/2 watchdog samples141/143; service peaks41180987392/41225433088B;
  minimum host68445958144/67196321792B; high/max/OOM/swap0. Service/watchdog1/0
  then0/0. Actual model/store GPU contexts clear and service paths removed.
  Both auxiliary scopes actual empty/events0 then stopped. GPUs15MiB/0%.
- Both loader-only overlays restored byte-for-byte AFTER owned workers exit.
  Original receipt d67_20260927/loader_install.json + loader_backup, restore
  receipt at root; second overlay2/install.json + overlay2/backup, separate
  overlay2/restore_receipt.json. Both restore receipts retained; no shared source
  left changed. Original five native launch scripts untouched.
- Store itself holds CUDA contexts on all four visible GPUs. Include this in
  physical lifecycle accounting; one engine does not imply one GPU held.
- Immediate exact result tables in baseline CONTAINED_LAUNCH, main curated
  paper_results/ieee_tc/serverless_audit/20260927_native_model_qualification.json
  and.csv (eight rows include failed attempt1). No artificial performance plot.
  Both private native log trees copied into raw results; every member SHA equals
  original. Raw/model/launch/watchdog/store/serve/restore hashes in curated JSON.
- Raw root: results/ieee_tc/serverless_qualification/d67_20260927.
  Successful model SHA65fb224aadbe1a5a51ca83d660711605e75a5a685cfca6d53c0105191bed072d;
  launch SHAe3f9f744f2018ef5e8af1df28ad5001a8937fcf6071cda0139193b972b41a445.
- Protected147 and plan reverified unchanged. Main288 smoke checks pass twice:
  first command omitted offline flags and spent time on a dummy-model HTTP
  timeout; it completed successfully before the attempted stop (unit already
  gone). Second explicit-offline check also passes. Both logs retained; these
  are overlapping CPU checks, not two model-performance repetitions. All test
  units are inactive/removed; no high-event claim is made for these collected
  CPU-only units. Model watchdog event counts above are independently measured.
- All8 curated request rows, raw/launch/watchdog/restore hashes and every copied
  native-log member independently reconcile. Both original vLLM source sets
  are restored; source SHA verified. Full pre-D67 ledger archived verbatim
  before compacting current status, cmp passed; nothing historical was dropped.
- Baseline source7f2de0de20d51f6bdc79cecff96d67daeb736629 and evidence checkpoint
  08153b457f43d3d587dae171f9dc49024fde4752 are pushed; fresh full origin/main SHA
  matches08153b4. Original dirty replay/relayserve files excluded. Main evidence
  backup contains only status/archive and two curated files. No full-goal,
  formal performance, optimality, full-pool or LoRA correctness claim.

## Current assets and non-negotiable gates

- vLLM0.30/torch2.13/CUDA13 actual environment:
  /home/qhq/.venvs/primelora_vllm0300_tc_20260925. Do not reinstall/upgrade driver.
  Stable CPU tests: /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python.
- Serverless actual native environment:
  /home/qhq/anaconda3/envs/sllm_vllm0102_newserverless_20260518 (vLLM0.10.2,
  torch2.8+cu128,Ray2.54). Compiled store0.8.0 under baseline installs.
  Select TC source + store PYTHONPATH and bundled LD_LIBRARY_PATH in driver AND
  all workers. Use original loader-only reversible installer, fresh receipts.
  Details: baseline ServerlessLLM_new_project/ieee_tc/CONTAINED_LAUNCH.md.
- Reuse ONLY the two D62 candidate content indices in inputs/README:
  3B materialized SHA bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  7B e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
 3B/7B have2/4 independent weight SHAs, not500 trained models.
- D63 independent7B reference matches all five native controls at the declared
  tolerance, BUT all20 wrong/correct controls also pass. Not discriminative;
  no tolerance tuning/same-prompt rerun. Current backbone SHA is not historical
  proof. Read NATIVE_ADAPTER_NUMERIC_CONTROL and ARTIFACT_CONTENT_AUDIT.
- Physical GPU possession is not utilization, ready status, request completion,
  empty_cache or shutdown return. D55 deployment accounting includes all owners;
  actual Full multi-activation/all-baseline qualification still open.
- CPU proof is actual affinity, not unavailable delegated cpuset. Service72/80
  GiB, swap2GiB, auxiliary4GiB. Ray2.54 native memory threshold is host-based;
  retain native semantics plus qualified OS/watchdog, no Ray rebuild needed.
- One heavy run at a time; actual watchdog and ownership cleanup. No global
  Ray-stop/pkill/TMUX cleanup, unrelated process termination or blind OOM retry.
- G1/G2 require correct complete work + common SLO; CE supplementary. Failed,
  incomplete, changed-contract and development runs cannot become formal winners.

## Protection, reporting and backup

- Seal SHA fa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Old final_v2, figs/paper and protected user work remain zero-change.
- Never stage main configs/generated/lora_manifest_1000.json, AAAI archive/
  directory, old fig7/fig2/fig3 artifacts or scripts/regenerate_motivation_figs.py.
- Baseline dirt: scripts/replay_openai_trace.py,
  scripts/run_serverlessllm_relayserve_continuation.sh, cache/, installs/, repos/,
  configs/relayserve_motivation_serverlessllm.yaml. Named task files only.
- Main remote faaslora_origin/retry14_continuous_queue_v2; baseline origin/main.
  Before push: diff/secrets/protected names/SHA/test/smoke; verify remote full SHA.
- Updates in Chinese at least every minute while active. Every experiment/group:
  cleanup -> validation -> figure/table -> interpretation -> next.
  Report completed evidence/current question and why/remaining experiments, not
  engineering activity disguised as completed performance experiments.

## Evidence archives

- EXECUTION_HISTORY_THROUGH_D26.md: SHA
  4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb.
- EXECUTION_HISTORY_D27_D52.md: SHA
  bd56dec4a43d7e03f35e8a9b1b26ac598378df4ca75429e0d1344d237e0d1dfe.
- EXECUTION_HISTORY_D53_D66.md preserves the exact1062-line pre-D67 ledger:
  SHA47b9a76adad6b7dacb1977133389c56cebbb921531d0c1921a5f2ea5618ee35c.
  cmp against main319487 HEAD version passed. No failed attempts or historical
  receipts removed. This concise current ledger supersedes obsolete next-action
  paragraphs, not the plan or evidence.
- Before optimization read relevant original logs/source/history, not every
  superseded chronological next-action paragraph.
- Key design documents: P1_FORMULA_IMPLEMENTATION, P2_BACKEND_QUALIFICATION,
  PHYSICAL_GPU_MEASUREMENT, RESOURCE_QUALIFICATION, EXTERNAL_REPLAY_QUALIFICATION,
  ARTIFACT_CONTENT_AUDIT, NATIVE_ADAPTER_NUMERIC_CONTROL, P0_FULL_PROVENANCE,
  REMOTE_ACCESS. All are under docs/ieee_tc.
