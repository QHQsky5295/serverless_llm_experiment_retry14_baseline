# IEEE TC execution status

## Authority and recovery

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before work, after compaction, and before experiments. AGENTS.md applies.
  No subagents authorized. Default mode; execution authorized.
- Approved snapshot PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md. Plan SHA:
  fe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.
  Historical snapshots and unsuccessful attempts remain preserved.
- Active goal incomplete. G1=correct complete workload/common SLO then minimum
  physical lifecycle GPU-s; G2=common budget/SLO then tail TTFT. CE supplementary.
  M1/M2/A/S formal matrices have NOT started; no optimality claim.
- D74 Serverless3B repaired finished/analyzed/plotted/backed up. Baselines PAUSED.
  Do NOT run the prepared3B original. Complete Prime IEEE Full first.
- User APPROVED once-only immutable compressed delivery cache, not new weights,
  trace or a second extracted pool. D78 publication and D80 delivery functionality
  are COMPLETE; DO NOT repeat either full pool.
- Formal artifacts are already published before deployment notice. No request
  packing/compression/formatting/full-pool hashing. Necessary reads, transfer,
  request-caused competition and local loading stay observed. Preserve legitimate
  NVMe/HOST/GPU reuse/fewer fetches. Never subtract old cumulative packing from E2E.
- Only artifact-only node exits150GiB disk floor; use actual incremental space,
  log and safety reserve with1.5margin. Inference150/100GiB and ALL OOM rules remain.
- No unrelated remote restart/config/hash/cleanup during actual inference.
  Shared service conditions freeze after qualification. No silent local fallback.

## D78–D80 completed milestone, 2026-09-27

| Check | 3B | 7B |
|---|---:|---:|
| D78 published objects / exact files |500/4000|500/5000|
| Immutable archive bytes |1160474851|420740410|
| Original logical bytes |19482573296|12985984450|
| Offline publication seconds |1210.560770|516.222248|
| D80 verified real HTTP transfers / UUID pairs |500/500|500/500|
| D80 request packing / temporary archive |0/0|0/0|
| Serial functionality-loop seconds, NOT production profile |154.636926|101.359877|
| Post-header cancellation / false publication / leaked files |pass/none/none|pass/none/none|

- Total compressed cache1581215261B (~1.47GiB), files0444/dirs0555. Both publishers
  exited0 and stopped. Builder e00e3f7c898cb90022361b0afc7381d9ba947d40.
  Offline1GiB memory.high reclaim was recorded, not production latency.
- D79 tested server a125b54a6cd2436ea3784d8ead859eca55be9ad4 deployed separately:
  /home/lab14/primelora_remote/tc/a125b54a6cd2436ea3784d8ead859eca55be9ad4/server.py
  SHA a31fa9212143914f5a7fb21dea126a23eaafedffe199ff9513de9ce03eb62a2c.
  Original remote server/source pools unchanged. Published objects reused.
- Server startup verifies immutable archives; HTTP reads/streams only. Client
  requires prepublished_gzip_v1 and validates archive AND exact original content.
  Read/socket-write/client-receive/local-verify spans are nested, not additive.
  Blocking socket write is not pure wire time.
- D80 first activation failed before transfers: io.stat unavailable under remote
  cpu/memory/pids delegation; monitor exit2 stopped owned services. Failure kept.
  Corrected monitor reports io_accounting=not_delegated; host diskstats separately.
  No fabricated I/O zero or weakened memory protection.
- Remote lab14 Linger enabled no→yes under existing service-management authority,
  without sudo/password. Services not boot-enabled. SSH logout persistence works.
- D80 service functional-only envelope high2GiB/max4GiB/swap0, CPU2–19,22–39,
  Tasks128. Not yet a validated whole-service concurrent production contract.
- 3B/7B coverage local high/max/OOM/swap all0; peak96903168/75149312B.
  Both clients/watchdogs exited0, service scopes removed, GPU compute empty.
  Empty auxiliary scopes verified with actual cgroup.procs content AND populated0,
  then stopped by matching invocation. Do not use file size to test pseudo-files.
- Cancellation UUID3B92adf056201f4609b381953f4569b0ab, server BrokenPipeError;
  UUID7B09e08c7d905a421d8977361aec1ecc17, ConnectionResetError.
  Both application body bytes0/not_published, owned workspaces empty BEFORE outer
  cleanup. Remote reads already occurred. Zero completed writes != zero network.
- Local cancel service/watchdog0/0, contexts clear, sampled high/max/OOM/swap0.
- After all clients exited, remote monitor invocation7c3039d545374a19b07d5cf3bab16b4f
  stopped; its trap stopped matching artifact services. All3 inactive/MainPID0/
  Result=success. NO LIVE TMUX/model/remote qualification task remains.
- Remote complete monitor838samples: minhost109657817088B, maxfull PSI0,
  mindisk147973099520B, no abort. SHA
  c9200043f87982c28aee9d00e5562351c5b1d596cb75aa64bffe37c60d4e0b3f.
- Raw results/ieee_tc/remote_qualification/d78_20260927 and d80_20260927.
  Curated 20260927_d78_{3b,7b}_publication, d80_{3b,7b}_coverage,d80_cancellation.
  REMOTE_ACCESS has immediate functional tables. Old500-row snapshots unchanged;
  new after-cancel journals501rows. No performance speedup inferred from them.
- D79 source tests already pass:182HTTP/lifecycle/preparation checks6.310s;
  288offline basic smoke21.870s under bounded unit. Terminal capture of latter
  truncated, accurately recorded as footer not full log. Full D78 smoke preserved.
  No source code changed during D80 qualification.
- Sourcea125b54 and earlier evidencea2488dd PUSHED and remote SHA verified.
  New D80 evidence-only backup follows validation; record receipt in raw directory.

## Mainline: next action (do not reopen completed qualification)

### D81 qualification-client extension tested; real concurrent check NOT STARTED

- D80 evidence checkpoint88e6ad9f4174d43da98446763678d8d88de1374b PUSHED;
  fresh remote SHA matched. All42new raw SHA references/147protected entries pass.
- Existing scripts/remote_artifact_client.py adds opt-in verify-concurrent only:
  explicit static IDs/lanes, simultaneous waves, exact published content, per-lane
  UUIDs, join failures/cleanup, exclusive JSONL. No server/core algorithm change.
  No artificial sleep or claim that client overlap proves wire/server overlap.
-36HTTP/client tests PASS4.162s;288offline basic smoke PASS22.341s, bounded unit
  runtime30.096s. Complete HTTP log retained; smoke terminal output truncated,
  footer proves success but is not a full log. Rawd81_20260927/client_tests.json
  and basic_smoke_terminal.json. No model/GPU experiment launched.
- IMPORTANT correction: PreloadingManager class default5 is NOT necessarily
  effective Full concurrency. _build_experiment_config uses coord.max_concurrent_loads,
  then preload.max_concurrent_operations, then3. Existing YAML has2/3 overrides.
  Derive final effective model/workload configuration before choosing the
  concurrent qualification lanes. Do not launch current default Qwen profile;
  IEEE models remain Llama3.2-3B and Llama2-7B.
- Candidate static largest compressed objects (read from completed D78 manifest):
  3Bcode_lora_0315=2339602B wire/56327468B logical;
  7Bmedical_lora=14736516B wire/20938679B logical.
  7B largest logical object differs:code_lora_0015=37716322B logical.
  Use measured identity/footprint; no new data. Do not confuse compressed size
  with original payload, HOST footprint or training diversity.
- Remote services remain INACTIVE. Next check reuses D80 qualified server/units/
  immutable caches/monitor with new unit identity and exclusive log files. Do
  NOT rerun activation script that refuses existing D80 logs or overwrite them.
  Common all-baseline concurrency qualification is still open; a Prime-specific
  bound must not be called an all-system production guarantee.

1. Check actual whole-service transfer entrypoints/concurrency and derive common
   remote resource envelope. No request archive allocation now; retain published
   cache, streaming buffers, thread/socket/metadata/log overhead. Qualify shared
   concurrency on existing selected objects, not another500-pool pass.
2. Extend existing source/profile collector for representative Remote, file-HOST/
   NVMe, native-HOST and GPU sources plus actual concurrency. Reuse frozen inputs,
   completed loading events, and existing admission helpers with
   collect_profile_only=True. Do not fabricate router estimates to bootstrap.
3. Choose/validate native HOST B/C under unchanged common80GiB inference budget.
   D60 uncached_background_v1 + D61 observed return are accepted candidates, not
   a selected production budget/profile. No allocator/bootstrap/install repeats.
4. Existing development1000 prefixes cover3B21/24 and7B6/6 exact file-content
   classes. Missing3B static representatives:support_lora_0148,research_lora_0104,
   finance_lora_0073. D76 P2 document records source trace hashes.
   D26 serial32 was28GPU/4HOST with16primed; NOT representative Full costs.
5. Then actual integrated IEEE Full multi-activation/lifecycle qualification.
   _require_ieee_full_qualification still intentionally rejects; do not remove
   guard or invent missing D/T/O/preparation profiles. Frozen profiles need exact
   backend/config/environment/resource/input identity and complete observed classes.
6. Baselines resume only after Prime Full:Serverless→vLLM→S-LoRA→dLoRA3B→
   Loquetier→HydraServe. M1/M2/A1–A5/S1–S13 all pending (442conditional core
   slots plus qualifications/extras, not442unique completed jobs).
7. All-zero limitation:3B500/500,7B498/500 weights zero,2/4distinct weightSHAs.
   D63 wrong/correct controls not discriminative. Nonzero fixture permission
   unanswered; do not re-ask repeatedly/download/generate or claim numerical
   correctness. Other Full implementation/profile work can continue independently.

## Assets and current service state

- Main /home/qhq/serverless_llm_experiment_retry14_baseline,
  retry14_continuous_queue_v2. Baseline /home/qhq/serverless_llm_baselines/main,
  HEAD9e2cf28903ed11bc8ee891dd4cc9636b94307573, PAUSED.
- Native /home/qhq/.venvs/primelora_vllm0300_tc_20260925:
  vLLM0.30/torch2.13/CUDA13; no reinstall/driver upgrade.
  CPU tests /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;
  OS guards /usr/bin/python3 (required pidfd API).
- Real remote alias primelora-artifact-174, lab14@192.168.4.174:8122,
  strict BatchMode/host check. Fingerprint
  SHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
  Private key ~/.ssh/primelora_artifact_174_ed25519_20260925,0600,outsideGit.
  Local/remote token ~/.config/primelora-tc-d75/artifact.token,0600.
  NEVER print values/put credentials in journals or Git.
- Remote primelora-artifact-tc-{3b,7b}.service and
  primelora-artifact-monitor-d80-v2.service INACTIVE now. Do not reuse dead
  PID165665/165667/165670 or prior invocations for new management.
  Unit fragments remain deployed; caches not deleted.
- Cache root /home/lab14/primelora_remote/tc/d78_20260927/published/{3b,7b}.
  Source pools /home/lab14/primelora_remote_artifacts/
  llama32_3b_a500_v1_modelscope and llama2_7b_a500_v2_publicmix.
  HTTP18080/18081, no13B.
- Correct indices ONLY:
  paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json,
  SHA bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  20260927_7b_materialized_content_index.json,
  SHA e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
  Canonical HTTP SHA3Bac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e;
  7B684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050.
  Misnamed7b_remote_content_index4998files is local diagnostic, NOT valid full7B.
- External reboot~15:06, not agent initiated: localboot
  aef67ed2-79d1-46bc-82e4-37401a5b45bf; remote
  b53c1f16-1dd2-4741-9271-6b75e86582ec. Current direct link1000Mb/s/full BOTH
  endpoints and remote peer advertises1000. D77 earlier100M historical only;
  physical reason not confirmed. No agent network change, no inspected qdisc cap.
- Main results symlinks to /home/qhq/serverless_llm_experiment/results.
  D80 snapshots/failed activation retained. No old results overwritten.

## Protection, reporting and backup

- Common inference72/80GiB/swap2GiB, auxiliary4GiB; one heavy run at a time.
  Actual workers contained BEFORE start, external watchdog, native release.
  No global kill/ray-stop/reset/remote reboot or unrelated cleanup.
- Protected147entries seal
  paper_results/ieee_tc/safety/20260925_execution_start_protected.json,
  SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Verify with scripts.ieee_tc_preflight.verify_seal.
- Preserve/exclude user configs/generated/lora_manifest_1000.json, AAAI archive/
  dir, oldfig7/fig2/fig3,scripts/regenerate_motivation_figs.py; rejected D72/D73
  previews untracked. Baseline unrelated dirty files untouched.
- Main push faaslora_origin/retry14_continuous_queue_v2; baseline own origin/main.
  Before push diff/secrets/protected-name/checksum/test/smoke; fresh remote SHA.
  Raw large logs/credentials never Git. Explicit paths only, no force push.
- Each run cleanup→validation→table/figure→interpretation→next. Chinese reports
  at least each minute during work from paper-evidence view. Functional checks
  use status tables, not manufactured performance plots. Use academic-plotting
  for actual figures per plan IEEE single-column/TNR/no overlap rules.
- Relevant pre-optimization evidence: P1_FORMULA_IMPLEMENTATION,
  P2_BACKEND_QUALIFICATION, PHYSICAL_GPU_MEASUREMENT, RESOURCE_QUALIFICATION,
  EXTERNAL_REPLAY_QUALIFICATION, ARTIFACT_CONTENT_AUDIT,
  NATIVE_ADAPTER_NUMERIC_CONTROL,P0_FULL_PROVENANCE,REMOTE_ACCESS.
- Verbatim archives contain SUPERSEDED live/next-action text, not fresh directives:
  EXECUTION_HISTORY_D78_D80.md SHA090798cbdf7ab651797e8fa7df2245f7c0dc3c3fe911397b37a67183feb3f99e;
  D67_D77 SHA608d83d2440b27cd071d5e85a3d41d92d8b74c100952bf530ec58c5e7c8dab21;
  THROUGH_D26 SHA4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb;
  D27_D52 SHAbd56dec4a43d7e03f35e8a9b1b26ac598378df4ca75429e0d1344d237e0d1dfe;
  D53_D66 SHA47b9a76adad6b7dacb1977133389c56cebbb921531d0c1921a5f2ea5618ee35c.
