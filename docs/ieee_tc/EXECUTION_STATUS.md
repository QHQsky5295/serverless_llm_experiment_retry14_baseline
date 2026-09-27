# IEEE TC execution status

## Authority and recovery

- Read the FULL authoritative /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md
  and this current ledger before work, after compaction and before experiments.
  AGENTS.md applies. No subagents authorized. Default mode; execution authorized.
- Approved snapshot PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md; original
 20260925/20260927 snapshots retained. User explicitly excludes artificial
 per-request packaging from formal research path (latest D77 clarification).
  Plan SHA fe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.
  Do not mutate plan to hide a failed or unavailable condition.
- Active goal is incomplete. M1/M2/A/S formal matrices have NOT started.
  G1=correct complete workload/common SLO then minimum lifecycle GPU-s;
  G2=common budget/SLO then tail TTFT. CE supplementary. Never claim optimality
  from qualification tests or choose formal winners/seeds post hoc.
- Latest user order: D74 Serverless3B repaired finished, analyzed/plotted/backed
  up. Baselines PAUSED. Do NOT execute prepared3B original. Complete Prime IEEE
  Full first. Retain all incomplete baseline qualifications and failed attempts.
- Only artifact-only node uses concurrent/retained temporary-archive peak plus
  logs/reserve and1.5growth margin instead of150GiB floor. Inference150/100GiB
  disk floors and all OOM rules unchanged. No unique artifacts deleted.
- Preserve legitimate Prime NVMe/HOST/GPU reuse and fewer fetches/bytes/contention.
  Unrelated remote work is interference; packing is delivery implementation cost;
  demand-caused reads/transfers/misses are in scope. Shared configuration freezes
  before measurement. No remote restart/tuning/hash/cleanup during inference.
  Keep full E2E; overlapping pack spans cannot simply be subtracted.
- Latest clarification: formal remote artifacts are ALREADY PUBLISHED before
  deployment. Remove our dynamic pack/compress/format/hash-scan work from actual
  request path, not by subtracting old logs. Preserve real transfers/contention
  and local tiers. Necessary storage reads/HTTP response are distinguished from
  artificial preparation, not assumed zero. D75–D77 pack results remain legacy
  functionality/diagnosis, not formal performance. D78 source delivery change
  tested; offline3B publication LIVE below, not yet serving new HTTP mode.
- User now asks why actual remote link is100M and whether latency should be
  excluded. D77 read-only diagnosis below answers; no network-change permission
  inferred. Do not silently replace measured real-remote main protocol by sim.

## D78 approved prepublished cache: 3B publication LIVE (do not duplicate)

- User explicitly APPROVED once-only immutable compressed transport cache,
  expected1.47GiB, no second extracted pool/new weights/trace. No longer a
  pending choice. Source plan and same current approved snapshot include this
  specific exception; previous7d91a347/d79d7eb6 versions remain in Git/raw records.
- Existing standalone server now has OFFLINE prepare-delivery-cache: full static
  ID/file set, regular nonsymlink files, streamed original SHA verification,
  exclusive destination, gzip9/PAX, read-only objects and complete manifest only
  after all succeed. Failed cache retained but cannot be served/resumed silently.
- Optional delivery-cache serving validates immutable archive hashes at startup,
  then only opens/streams frozen objects. No request-time packing/hash/temporary
  archive or miss fallback. Mutation rejected, all attempts correlated.
  artifact_timing_v2 explicitly states prepublished mode/no pack; zero pack is
  structural absence, not unknown legacy timing silently filled with zero.
- Existing client verifies published archive AND original exact file content;
  explicit required_delivery_mode prevents reverting to dynamic packing. Both
  env factory and configured RemoteStorageClient forward the contract. No
  routing/admission/equation/backend/weight change.
- Initial30 client checks3.951s pass; expanded40 client/content checks3.680s pass,
  including corrupted source, incomplete cache, mutation/no fallback, content/
  archive/mode mismatch, cancellation and no HTTP-path pack/hash/temp tests.
  Basic smoke288 checks PASS68.316s (bounded CPU unit, no GPU). Protected147
  entries unchanged, diff-check clean. Sourcee00e3f7c898cb90022361b0afc7381d9ba947d40
  PUSHED; fresh remote branch SHA identical, no user changes staged.
- Raw root results/ieee_tc/remote_qualification/d78_20260927, PROTOCOL.md.
  Next: observe SAME3B offline publication, validate/table, then7B. No parallel
  inference, baseline or new pool. Cache destination is exclusive; NEVER restart
  or overwrite this live attempt because a context boundary occurred.
- Remote newversion /home/lab14/primelora_remote/tc/e00e3f7c898cb90022361b0afc7381d9ba947d40/server.py,
  sourceSHAcdcdbe1f4a90db9a5546e77d386075b62baf88b5ee1fefdd678d6b420fb346bc.
  Both frozen indices copied/SHAs matched; original serverSHAa3650722 unchanged.
- Live3B publication started16:29:46+08, unitprimelora-publish-tc-3b-d78.service,
  PID128063/invocationefce4fd2c63f4b83822de2ccdb88ff45, high1G/max2G/swap0,
  CPU2,22/TasksMax16. Separate monitor scopeprimelora-publish-monitor-3b-d78.scope,
  invocationa4d98a1482c44fc29e2e11fe3b9c8563, high128M/max256M/swap0, CPU0,20.
  Monitor checks host memory/pressure each loop, disk/inodes every30 samples,
  and only stops the matching invocation on abort/exit. No GPU/runtime on174.
- Local TMUXprimelora-d78-publish3b owns SSH/monitor; rawpublication_3b_monitor.log
  grows. Remote events /home/lab14/primelora_remote/tc/d78_20260927/publication_3b.jsonl;
  destination sameparent/published/3b. Incomplete directory is0700; complete
  manifest appears only after all500 success, then directory0555/objects0444.
- Last observation63/500 published,152monitor samples, remotepeak1074266112B,
  high4144/max0/OOM0/swap0, hostavailable~102GiB, free149404114944B. Offline
  page-cache reclaim under1G high is not production latency or a failed run.
- Raw launcher publish_one_remote.sh SHA8bc12cb37ee7bde595ddc6287b45282dca465037259e035bdda6e04373e07497;
  run_publish_3b.sh SHA5d501c1b1bf9d0ef6b855823ff9d18e74db40d36ef4950725f2ea46f58cf47ca.
  They select one explicit model, never automatically chain7B before3B table.
- NewHTTP delivery/all-pool qualification and common performance envelope still
  PENDING. Fresh1Gbps state must key subsequent preparation/service profiles.

### D78 external reboot and link change, 2026-09-27 16:25+08

- Local boot aef67ed2-79d1-46bc-82e4-37401a5b45bf (15:05:52), remote
  b53c1f16-1dd2-4741-9271-6b75e86582ec (15:06:36). We did not reboot either.
- Remote eno1 now1000Mb/s/full AND peer advertises1000baseT/Full. Local
  eno1np0 also1000/full; direct route unchanged, inspected qdisc has no cap.
  No network setting changed by this agent; physical cause remains unconfirmed.
  D77's100M evidence remains historical, not current or new profile evidence.
- Old artifact units both inactive/MainPID0/empty InvocationID after reboot;
  ports18080/18081 unoccupied. Old PIDs/invocations must not be reused.
- Remote free149560934400B, > conservative97407250432B preparation bound;
  MemAvailable107319648KiB (~102GiB), swap0, pressure0. Local free337391349760B,
  MemAvailable120724268KiB, GPU compute empty. Check again before publication.
- Raw post_reboot_link.json SHA68acaf331f2f7a04490198805154455e5420026c3cc5aa496bca0e9679ecbde4.
  Remote rg absent (recorded); immediate grep fallback supplied memory values.
  Plan/snapshot observation updated, no experimental/safety threshold relaxed.

## D77 completed checkpoint, 2026-09-27

- Started main6b3246ff184893d536ceb6587a18dbd207f6ce2e; baseline remains
  9e2cf28903ed11bc8ee891dd4cc9636b94307573, backed up and PAUSED.
- SAME D75 7B coverage ENDED normally:500/500 IDs,5000 exact files verified,
  logical12985984450B/wire420756591B,645.243s serial functional loop,0failures.
  All500 client/server UUIDs reconcile identity/bytes/pack-rounding/spans/cleanup.
  Remote copied journal SHA e26ab45e286792bcea3f02ac60691c26dea75dfbd73ad09c95c83f367fb685fc.
  All per-download temporary copies removed. Do NOT repeat coverage.
- Local638 watchdog samples, peak74416128B,min host110448508928B; all high/max/
  OOM/OOM-kill/swap0. Service/watchdog0/0, service path removed, auxiliary inactive,
  all GPUs15MiB/0%, no model/TMUX. Remotehigh22352,max/OOM0. NOT Full profiles.
- Prior3B SAME D75 run completed D76:500/500 IDs,4000files,19482573296B logical/
 1160493734B wire,0failures,500UUID matches/cleanup. Local1392 samples,peak95641600B,
  high/max/OOM/swap0. Remotehigh34982,max/OOM0. Do NOT repeat it.
- Existing CLI extended ONLY with opt-in verify-cancel witness; production
  downloader/remote policy unchanged. Trigger after real headers, before body.
  Check downloader workspace BEFORE outer temp cleanup. Tests cover actual HTTP,
  unrelated network error and leaked workspace; outer cleanup cannot hide failure.
-26 client checks PASS1.905s;288 basic smoke PASS23.267s, bounded CPU units.
 147 protected entries/plan unchanged. No new weights/pool/trace/backend install.
  Source +7B evidence c4767844300804e4eb47566c868efcc61b3cbbd2 pushed; fresh
  faaslora_origin/retry14_continuous_queue_v2 full SHA matched.
- ONE actual cancellation/model completed in existing gated launcher/TMUX:
 3BUUID92125202dde74e9d814889e37b117ef4;
 7BUUID9d87fa4e53c64d9195247d30ae752efd.
  Both exact expected cancellation, not published, zero application body-read,
  no downloader leftovers before outer removal; remote ConnectionResetError,
  temp removed. Already-completed remote packing is not cancelled.
  bytes_written0 counts successful socket writes, not proof of zero network bytes.
- Cancel3B/7B watchdog3/2 samples,peak39182336/38817792B,minhost110771978240/
 110932742144B,high/max/OOM/swap0; service/watchdog0/0,actual scopes gone,
  auxiliaryd770...001/002 inactive. Do NOT repeat these checks.
- All checks have immediate REMOTE_ACCESS tables, curated7B coverage and two
  cancellation cases, raw SHAs. No performance plot or inference qualification.
- Owned remote vmstat invocation181b9be672ca41b08512532c8046cfd0 verified and
  stopped after checks. MainPID0/inactive; local exec65199 ENDED/exit0. Do not
  resume/start duplicate. Final log SHA33b8b9601125dfca7c99560a31908af4b9b7ba114241617aa4abf29acc1c40de.
- Old current ledger archived VERBATIM as EXECUTION_HISTORY_D67_D77.md; cmp
  passed, SHA608d83d2440b27cd071d5e85a3d41d92d8b74c100952bf530ec58c5e7c8dab21.
  Archive has superseded next-action paragraphs, not new instructions.
- All D77 measured qualifications used OLD plan SHAd79d7eb6; new user-approved
  published-artifact boundary snapshot matches source, SHA7d91a347. Preflight
  snapshot pointer updated, no safety gate weakened. Never relabel old runs.
- Published-artifact snapshot/source cmp passes;55 OS/preflight tests pass,
 147 protected entries unchanged under new plan. All9+9+16 raw qualification
 SHA references independently match. Existing26client/288smoke passes remain
 source-check evidence; no model execution added for the plan-only pointer.

## D77 actual100M link: evidence and limits

- Remoteeno1 Broadcom BCM5720 Gigabit/tg3, supports AND advertises1000baseT/Full.
  Partner advertises ONLY10/100; negotiated100Mb/s/full/autoneg on.
  Localeno1np0 sysfs1000Mb/s/full. Direct192.168.4.174↔192.168.4.178 route.
  Remote mq/fq_codel; no inspected ingress/egress filter or rate shaper.
- This locates the immediate constraint at peer/physical negotiation, NOT GPU
  or the NIC's maximum capability. Switch capability/configuration, intervening
  equipment and cable/downshift are not distinguishable without peer inspection.
  Cannot claim entire machine room is100M, nor exclude unobserved path QoS.
- No speed/MTU/qdisc/link/reset/driver change, no iperf/stress or installation.
  Optional ethtool netlink permission warning retained; main link fields returned.
  Local ethtool absent, sysfs used. Unsupported remote diagnostic option rejected.
- Raw d77_remote_link_diagnosis.json SHA30b1acd7f03257c5da9b47dfb237a627cf11da36e12b1fe901a3a432766dcb15.
  REMOTE_ACCESS has primary-source Linux/Intel/ServerlessLLM/HydraServe references.
- Existing serial500 functional samples: mean total/remote-pack/receive-write
  ms=3B2810.138/2331.979/199.532;7B1255.401/959.539/74.296.
  NOT production profiles: qualification memory.high reclaim and mostly-zero
  compressed artifacts. Packing nested in header wait; receive includes local
  reserve/write. Do not infer network saturation or inference causality.
- Keep actual request transfer in TTFT/E2E. Service-only metrics can separately
  diagnose compute.100M may magnify caching gains; needs existing S1/S2/S3,
  LastKnown and feasible bandwidth evidence.100M cap cannot create0.25/0.5/1G
  achieved points. Faster local-sim stays labelled controlled supplementary,
  never real faster Ethernet. No automatic network or plan change.
- Common delivery/performance envelope needs qualification before Full profiles;
  eliminate avoidable request-path delivery costs only via an explicit shared,
  validated frozen contract, preserving legitimate native caching. Never subtract
  cumulative packaging time to fabricate a no-interference E2E.

## Mainline ledger and exact next action

| Block | Status | Evidence / next |
|---|---|---|
| Safety/physical measurement | Bounded actual workers/ownership and reducer connected | Full multi-activation qualification still open |
| Protected history |147entries unchanged at D77 | safety/20260925_execution_start_protected.json |
| Remote174 | Start/stop/restart, both full pools and post-header cancellation pass | Shared performance envelope still open |
| P0 Table1/Full | Source-pair audit complete,202scalars retained | Different executions, do not relabel as one run |
| P1 IEEE | Frozen planning/replacement/admission/HOST return connected | Representative measured profiles and integrated Full open |
| P2 | vLLM0.30 installed and native mechanical tests | Full semantic qualification open; zero-weight limitation |
| Serverless |7B polling pair and3B repaired completed | PAUSED, original3B unexecuted |
| Other baselines | Pending | Later vLLM/S-LoRA/dLoRA3B/Loquetier/HydraServe |
| M1/M2/A1–A5/S1–S13 | NOT STARTED | No formal rank or optimality claim |

1. Return DIRECTLY to Prime Full, not baseline/allocator/bootstrap repeats.
2. Implement/qualify common already-published artifact delivery, no request-time
   packing. Two actual alternatives audited: once-only immutable compressed
   transport cache (same content/compression semantic, estimated1.5GiB total)
   versus direct existing files (no duplicate cache but substantially more wire
   bytes). User now APPROVED once-only compressed cache; implement that choice.
   3B cache publication is LIVE; observe it rather than restarting. Do not inflate
   network work by switching to uncompressed files merely to avoid disk use.
   Both alternatives must retain exact input
   SHA, identical common protocol, cold local state and legitimate caching.
   Hugging Face official file-download docs/pinned vLLM0.30 resolver inspected.
   Do not freeze obsolete packaging timings or ask the answered choice again.
3. Derive common remote performance resource contract from actual whole-service
   concurrent transfer/retained-archive bounds and metadata. Current1/2GiB
   qualification limits must not silently become production profiles.
4. Extend existing native source/profile collector for representative Remote,
   file-HOST/NVMe, native-HOST and GPU sources plus actual concurrency; choose
   native HOST B/C partition under the unchanged common80GiB inference budget.
   Reuse existing inputs/content classes, not new weights/full traces.
5. Existing1000-request development prefixes cover3B21/24content classes,
  7B6/6. Missing3B representatives are existing support_lora_0148,
   research_lora_0104,finance_lora_0073. Static content audit is not future demand.
   P2 document D76 gives exact coverage. Do not repeat D26 serial32 prefix.
6. Actual integrated IEEE Full multi-activation/lifecycle qualification follows.
   _require_ieee_full_qualification in run_all_experiments still intentionally
   rejects; do not remove guard or invent missing D/T/O or preparation profiles.
7. All-zero issue remains:3B500/500,7B498/500weights zero,2/4distinct weightSHAs.
   D63 wrong/correct controls are not discriminative. Independent nonzero
   correctness fixtures require user authorization, still pending; do not
   repeatedly ask, acquire/generate new weights, or claim numerical proof.
   Other Full implementation/profile work may continue independently.

## Assets / remote services (currently inactive after reboot)

- Main /home/qhq/serverless_llm_experiment_retry14_baseline,
  retry14_continuous_queue_v2. Baseline /home/qhq/serverless_llm_baselines/main.
- Native environment /home/qhq/.venvs/primelora_vllm0300_tc_20260925:
  vLLM0.30/torch2.13/CUDA13. No reinstall/driver upgrade.
  CPU tests /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;
  OS guard tests /usr/bin/python3 (native interpreter lacks required pidfd API).
- Correct indices ONLY (do not select by wildcard misleading filename):
 3B paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json,
 SHA bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
 7B20260927_7b_materialized_content_index.json,
 SHA e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
 Canonical HTTP3B ac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e;
 7B684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050.
 Historical index fields unchanged; join separate coverage by SHA.
- Remote alias primelora-artifact-174: strict BatchMode SSH works via
 192.168.4.174:8122 lab14. Known host fingerprint
 SHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
 Private key ~/.ssh/primelora_artifact_174_ed25519_20260925,0600,outsideGit.
 Local/remote tokens ~/.config/primelora-tc-d75/artifact.token,0600,outsideGit;
 never print/read into journal/commit secret values.
- Remote units primelora-artifact-tc-{3b,7b}.service are INACTIVE/MainPID0
 after reboot. Historical3BPID3240469/invocation5d1621ec44f44d5495e0fd2c86a182ba
 and7BPID3240475/invocation856cbbb9dfd94974ae19fffc2c8549c3 are no longer live.
 UID1000 cgroup (local UID1001). Each high1GiB/max2GiB/swap0/TasksMax128,
 CPU2–19,22–39. Qualification only. Revalidate identities before any operation.
 Ports18080/18081, original500 pools. No13B.
- Versioned deployed server under
 /home/lab14/primelora_remote/tc/d8a0b49443e413947130a14a62c7aa5562bbee17/server.py
 SHA942a8e52956a58c22e822d68011fbd47ffa3b64bd4306904168a46bad7059101.
 Original remote_artifact_node/server.py SHAa365072244512f4880432d7f4198cf3e45897ff19063a2b25eb141c8ca4e2a02 unchanged.
- D75 bound (functional ONLY): max archive3B63725568B/7B42684416B,502 attempts/
 model including sample+500coverage+cancel, all retained worst-case53417811968B,
 logs64MiB/reserve16GiB/1.5margin =>97407250432B required. Remote last available
 148647604224B. Not the Full all-worker/concurrency bound; no blanket150GiB floor.
- Raw /home/qhq/serverless_llm_experiment/results/ieee_tc/remote_qualification/d75_20260927.
  Main results symlinks to original root. Do not overwrite previous500-row
  remote_*_complete snapshots; current remote journals now501rows after cancel.
- D60 uncached_background_v1 is explicit candidate; D61 observed native return
  wakes deferred work, no early eviction/future bytes/budget growth. Actual
  representative backend/content/resource context must key new profiles.

## Protection, reporting, backup and history

- Common inference service72/80GiB/swap2GiB; aux4GiB; one heavy run at a time.
  Actual workers limited BEFORE start; external watchdog, native release;
  no global pkill/ray-stop/TMUX cleanup, no unrelated process termination.
-147entry seal SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Never stage user configs/generated/lora_manifest_1000.json, AAAI archive/dir,
  old fig7/fig2/fig3 or scripts/regenerate_motivation_figs.py; rejected D72/D73
  plot previews stay untracked. Baseline legacy replay/relayserve/cache/install/
  repos/config dirt preserved.
- Main push faaslora_origin/retry14_continuous_queue_v2; baseline origin/main.
  Before push: diff/secrets/protected-name/checksum/test/smoke, then fresh full
  remote SHA. Raw large logs/credentials never Git. Current source checkpoint
  c476784 is backed up; later evidence-only checkpoint is recorded by Git/raw receipt.
- Every run: cleanup→validation→table/figure→interpretation→next. Reports in
  Chinese from paper-evidence perspective at least each minute while active.
  academic-plotting is used for actual data figures; functional status uses tables.
- Archives (verbatim, do not reinterpret obsolete next actions):
  THROUGH_D26 SHA4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb;
  D27_D52 SHAbd56dec4a43d7e03f35e8a9b1b26ac598378df4ca75429e0d1344d237e0d1dfe;
  D53_D66 SHA47b9a76adad6b7dacb1977133389c56cebbb921531d0c1921a5f2ea5618ee35c;
  D67_D77 SHA608d83d2440b27cd071d5e85a3d41d92d8b74c100952bf530ec58c5e7c8dab21.
- Relevant design/history before optimization: P1_FORMULA_IMPLEMENTATION,
  P2_BACKEND_QUALIFICATION, PHYSICAL_GPU_MEASUREMENT, RESOURCE_QUALIFICATION,
  EXTERNAL_REPLAY_QUALIFICATION, ARTIFACT_CONTENT_AUDIT,
  NATIVE_ADAPTER_NUMERIC_CONTROL, P0_FULL_PROVENANCE, REMOTE_ACCESS.
