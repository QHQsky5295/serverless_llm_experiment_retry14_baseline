# D69: shared request observations before the polling comparison

This is preparation/qualification, not a completed 1,000-request comparison.
The figure/table display name remains **Serverless**.

## Hypothesis and source evidence

If prompt construction, token counts or clocks differ between router variants,
their latency difference cannot isolate polling. The legacy client explicitly
restricts fixed_length_greedy_v1 to S-LoRA; that dirty user file is untouched.
The previous four-request loader witness used a different role renderer and is
not promoted into a matched-input performance result.

The official [router at the pinned commit](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)
still supplies the polling hypothesis. The [vLLM 0.10.2 output processor](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/engine/output_processor.py)
updates RequestStateStats before constructing RequestOutput, but does not attach
the stats to that output. Its [stats definitions](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/metrics/stats.py)
distinguish frontend wall time from engine-core monotonic timestamps. Do not
subtract them or substitute the response-completion time for the last token.

## Shared contract

- Extend existing FrozenReplayPlan, guarded launcher and contained native helper.
  Do not invoke the dirty legacy replay or rebuild the experimental framework.
- Prime and HTTP comparison now share the existing role_lines_v1 rendering and
  canonical fixed-output token guard: content cap759, target=min(source,256),
  actual special-token IDs separately hashed, greedy/ignore-EOS/no stops.
- Preserve existing source trace, adapter ID, input/target hashes and timestamps.
  The static preparation reads inputs only in the external client; the backend
  receives each request at its scheduled arrival, not future request contents.
- HTTP publisher is an auxiliary child inside the same4GiB cap as the watchdog;
  no client semaphore or connection concurrency cap shifts offered arrivals.
  Preparation precedes the shared deployment notice; t0=notice+60, not readiness.
  Connection submission, raw responses, rejected responses and timeout records
  are retained. Timeout remains1800s from planned arrival for qualification.
- Source views symlink unchanged upstream files and instrument only API receipt,
  enqueue/assignment and backend observations. Neither dirty upstream nor the
  installed package is overwritten. Original and repaired views differ only in
  the already-audited load-balancer method; AST comparison verifies other code.
- The optional0.10.2 frontend hook copies the existing scalar stats at output
  construction and carries the latest snapshot through native collector merging.
  Without the latter, queued token IDs can grow while metrics stay stale.
  It is version/SHA bound and identical across variants. No
  scheduling, sampling, model loading, admission or autoscaling change.
- Native output IDs, prompt IDs and engine-request LoRA name are checked, not
  inferred from expected counts or text. Binding evidence is NOT independent
  numerical proof that a nonzero adapter changed inference; that gate stays open.
- Common boot/time-namespace identity is mandatory for joining stages. Exact
  monotonic boundaries expose submission, dispatch, service TTFT, decode and
  completion notification. TPOT/E2E recomputation tolerance is1ms.

## Checks completed before model qualification

| Check | Result | Limit |
|---|---|---|
| Shared replay/native-timing tests |40 passed | Includes6 new fixed-input/open-loop checks; no model run |
| Baseline native launcher/router/measurement tests |38 passed | Includes native collector-merge regression and a real local HTTP endpoint delaying all responses until five requests arrive; endpoint is a unit fixture |
| OS guard tests |51 passed with qualified system Python | Incorrect mixed-suite invocation used a native Python lacking pidfd;2 errors/2 failures preserved, no weakened guard |
| Main smoke + shared replay/native timing |328 passed in separate rerun | The initial mixed379-test invocation was NOT successful; OS-only51 tests separately passed in system Python |
| Historical protection |147 unchanged | Not a performance result |

The source plan still requires100-request smoke, complete pool coverage and
two-model original/repaired1,000-request development pairs. Local mechanical
qualification does not fulfill the real-remote main protocol. No M1/M2 winner,
polling benefit, independent LoRA correctness or full baseline reproduction is
claimed by these tests.

## D69 first launch: measurement failure before model construction

| Attempt | Model/request execution | Result | Interpretation |
|---|---|---|---|
| `d69_20260927/model_launch1.json` | No service started, no request arrived | Publisher readiness JSON rejected | Launcher/import isolation error; not a Serverless startup, throughput or OOM result |
| Tokenizer-only diagnosis | Existing tokenizer only, no model | Ready record emitted without inherited serving source paths | Reproduced serving `PYTHONPATH` imports `sitecustomize`, torch/vLLM and a stdout platform log before the handshake |
| `model_launch2.json` | Publisher ready, zero arrivals, no model construction | Driver rejects changed library path | Shared repository startup hook imports cv2 through vLLM and cv2 prepends its libraries; not a baseline performance failure |

The [Python startup documentation](https://docs.python.org/3.12/library/site.html)
explains why this precedes helper-level imports. The publisher now has its own
startup environment, with no serving PYTHONPATH/PYTHONHOME; shared request code
is explicitly imported after startup. The ready record must report no imported
torch/vLLM/Ray/Serverless modules. The parser remains strict; it does not skip
unexpected output or silently retry. Service imports and policies are unchanged.
The first loader overlay was restored byte-for-byte after verifying no service
or GPU context existed. The auxiliary group was empty before stopping it.
The direct diagnostic intentionally had no deployment-origin input; its EOF
exception is retained and is not an inference attempt.

Attempt2 confirms an empty serving-module set in the external publisher. Its
separate driver check exposed the same repository startup-hook coupling in the
service. Rather than permitting arbitrary library drift, the exclusive source
view now exposes only a symlink to the shared `faaslora` package, not the main
repository root. This excludes legacy torch.load/shutdown patches from Serverless
and keeps original native imports/library paths. Attempt2 had12 watchdog samples,
confirmed GPU release/service removal and no GPU model constructed. Overlay2
restored; empty auxiliary stopped. All attempts remain retained.

## D69 native loading reaches GPU; LoRA layout qualification fails

| Observation | Attempt3 result | Interpretation |
|---|---:|---|
| Planned/arrived requests |100/99 | Truncated after fatal engine exit; one unarrived request is not a timeout |
| Protocol-valid responses |0 | No latency, TPOT or polling-effect estimate |
| Initial startup failures |1 connection refusal +3 HTTP500 missing-router | Shared notice+60 preserved; no readiness-based shift |
| Errors during manual owned cleanup |89 disconnected +6 connection refused | Do not mislabel these95 as independently observed inference failures |
| Native store GPU confirmation | PASS |7B native weights loaded before LoRA initialization crashed |
| Service peak/minimum host available |40,590,913,536 /68,924,452,864 bytes |238 watchdog samples; high/max/OOM/OOM-kill/swap all0 |
| Cleanup | PASS |No hard kill, GPU contexts clear, service removed, empty auxiliary stopped; overlay3 restored |

EngineCore exits in `VocabParallelEmbeddingWithLoRA.set_lora`:0 rows available
versus1024 requested. The native checkpoint contains32000×4096 input embeddings;
the default4×256 extra vocabulary requires33024 rows. The loader assigns the
stored tensor directly. This layout mismatch is distinct from parameter-byte
identity, which remains valid. This is not native throughput saturation.

Primary sources: [vLLM0.10.2 LoRA configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/config/lora.py),
[embedding layer](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/lora/layers/vocal_parallel_embedding.py),
[Serverless native loader patch](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm_store/vllm_patch/sllm_load.patch).
In particular0 extra-vocab is not an accepted stock0.10.2 setting; do not silently
relax that validation or pad/truncate adapter weights.

Historical7B deployment already selected `disable_lora_embeddings=true` through
`generate_serverlessllm_deploy_config.py`; its old `load_format=auto` does NOT
prove the native path. The existing environment has the corresponding Llama
compatibility mode (source SHA231282afc57850184704a0ab064a83828701ab8662c9547042fbe4f987a4ea93).
Next candidate reuses this content-derived selector, after checking every
adapter config and actual safetensors header. Embedding/saved-module deltas,
missing files or an unavailable inspector must reject the candidate. It does
not infer absence from adapter names or zero output differences. No new vLLM
patch, weight/pool/trace generation or checkpoint re-export is needed on this
evidence. Actual candidate model execution remains to be checked.

Candidate39 unit checks pass (2.468s), including rejection of extra-token files,
embedding tensor keys and missing weight files. The actual500-adapter header/
config check passes without CUDA initialization (6.729s service wall, not model
startup). Config-set SHA0597fa687254333e626255672113abcb2a2efd1594044e7d6f59f237969a09fc;
existing selector SHA45ab8b151a9fb3b53e2b1a8fb86d0c180604eb3f14ef999792e2a413ad9d53f4.
These are layout prerequisites, not a numerical LoRA or inference pass.

Raw root `results/ieee_tc/serverless_qualification/d69_20260927`; copied private
Ray logs match every original regular-file SHA (`--no-ignore` required inside
the ignored results directory). Main curated `20260927_http_qualification_d69`
JSON/CSV preserves all three attempts. No paper performance claim is supported.

## D70: complete observed replay, but four startup failures

| Observation | Result | Interpretation |
|---|---:|---|
| Planned / arrived / terminal |100 /100 /100 | No truncation or missing terminal request |
| Protocol-valid / failed |96 /4 | NOT a100/100 qualification pass |
| Failure stages |1 connection refusal,3 HTTP500 before router creation | No readiness-shifted arrival or retry hiding these failures |
| Native7B initialization and inference | PASS for96 observed responses | Layout candidate works for linear-only existing adapters; not numerical LoRA proof |
| Native prompt / adapter name / output target |96/96 match | Original raw tokens preserved; no text retokenization |
| Watchdog samples |1237 | No high/max/OOM/OOM-kill/swap events |
| Peak service / minimum host available |42,924,400,640 /67,016,757,248 bytes | Not a serving-memory extrapolation |
| Actual cleanup / loader restore | PASS |GPU contexts clear, service removed, empty auxiliary stopped; exact overlay restore |

The same source_repaired2 and100-request trace view are reused. The single
instance has native target1 (one in-flight admission); backend max_num_seqs4
does not make this a four-concurrent-request throughput experiment. All store
GPU contexts remain part of physical possession, not just the one model GPU.

The client writes http_replay_complete with96 responses/four failures, then
exits1. The supervisor treats any publisher nonzero exit as a broken replay
and stops the service before its final model_qualification.json is saved.
The raw classification protocol_or_launcher_error is retained, but is not a
root-cause classification of the four startup request failures. All100 requests
were terminal before cleanup; none of these failures was induced by cleanup.
Next separate an intact failed-workload journal from a failed measurement
process, retaining qualification failure and bounded finalization. Do not
weaken monitoring or reinterpret failed requests as successful inference.

Actual primary code was rechecked: the
[controller](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/controller.py)
registers artifacts before constructing the model router; the
[router](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)
queues only after that router exists. Local startup stage ordering and the
API errors must be audited separately from native engine load or repaired
allocation latency. This is NOT the two-model1,000-request polling pair.

Raw root results/ieee_tc/serverless_qualification/d70_20260927;93 regular native
log files copied and SHA-equal. Main curated20260927_http_qualification_d70
JSON and100-row CSV retain complete status, conditional times and raw hashes.
No whole-run correctness, real-remote qualification, baseline winner or main
matrix completion is claimed. No new weights, pool, trace or backend patch.

Post-run measurement correction: the main supervisor now distinguishes exit1
with a complete, identity/count-checked failed-workload journal from a crashed
publisher. It allows at most60s normal service finalization while all watchdog
checks remain active, and still returns qualification failure. Truncated,
duplicate, inconsistent or missing terminal records remain fatal.53 OS tests
and328 main smoke/shared-protocol checks pass; this exact D70 journal is
recognized as measurement_complete=true/workload_passed=false.39 baseline
checks pass2.565s. No native rerun was made just to test report finalization;
its actual behavior remains to be checked during the next useful qualification.

## D71 candidate: remove a false bootstrap dependency, not a serving policy

D70 store.log records pinned-pool construction from08:12:15.828617 to
08:12:36.064198 (20.236s). The launcher started that independent process only
after the Ray head and worker. API startup was08:12:58; artifact registration
began08:12:59 and the router was constructed afterward. A missing router and an
existing router waiting for an engine are different request paths. Do not
subtract these wall stamps from request monotonic stamps without an anchor.

The [official storage service](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm_store/sllm_store/server.py)
constructs CheckpointStore from storage/pool parameters without Ray membership.
Conversely, the [store manager](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/store_manager.py)
requires worker discovery and the store endpoint. Therefore the candidate
starts store in parallel with head→worker, then retains both readiness barriers
before the controller. Native commands,32GiB pool, admission/CPU/memory envelope,
one-instance target1, RR/loader/scaler,500-adapter map and notice+60 stay fixed.
Earlier GPU possession by the store remains charged; shorter startup is not
automatically less GPU time. No client retry, free prewarm or arrival shift.

Hypothesis: removing this unnecessary ordering reduces absent-front-door/router
exposure. It does NOT assert that all100 requests will pass or that native
engine initialization fits60s. Use one100-request development comparison with
the same D70 inputs and record every failure. This is a changed bootstrap
experiment, not an identical repeat to recover a final report. If startup
failures remain, retain the boundary and return to the mainline; do not tune
the preparation window or indefinitely optimize infrastructure. The eventual
original/repaired router pair must use the same selected bootstrap on both sides.

Candidate40 launch/router/measurement unit checks pass (2.425s); actual startup
and bounded failed-workload finalization are untested at this checkpoint.
New paired wall/monotonic startup events will preserve registration boundaries.

| D71 attempt | Current result | Interpretation |
|---|---|---|
|1 | Refused before native service execution: missing NVML binding environment | GPU-monitor CLI options do not configure gated-launch's environment-based interface; no model or arrivals, not system failure |

Attempt1 service path removed and auxiliary verified empty/events0 then stopped;
all GPUs15MiB. Loader overlay restored exactly. Original command/receipt remain.
Attempt2 changes only the command's explicit monitoring environment and fresh
output/ownership paths; no guard relaxation or baseline configuration change.

## D71 completed result: remaining startup boundary is preserved

| Observation | D70 serial bootstrap | D71 overlapping bootstrap |
|---|---:|---:|
| Planned / arrived / terminal |100/100/100 |100/100/100 |
| Protocol-valid / failed |96/4 |97/3 |
| Early failure type |1 refused connection +3 HTTP500 |3 HTTP500 missing model router |
| Whole-run qualification |FAIL |FAIL |
| Native prompt / target / adapter-name match |96/96 |97/97 |
| Actual native output tokens |16,818 |17,035 |
| Service memory peak (bytes) |42,924,400,640 |41,928,572,928 |
| Minimum host available (bytes) |67,016,757,248 |66,548,797,440 |
| High / max / OOM / swap events |0 |0 |

Each column is one development run, NOT a formal paired CI or evidence of
statistically improved latency/resource consumption. The configuration SHA is
identical:2471d58ec951eb8fca1fd0dd24b7a8d0bdb841a5727cd7ef1fea407c9c4b9a50.
No new request failures occurred after router creation in either observed run.
The extra successful D71 request is req_00003, whose217 tokens explain the
output-count difference. Both runs use29 actual unique adapter IDs.

D71 monotonic boundaries relative to the unchanged common deployment notice:

| Boundary | Seconds after notice | Meaning |
|---|---:|---|
| Native launcher starts |7.346 | Pre-launch validation/imports are included in the window |
| Launcher sees API ready and returns |60.667 | Observation is not the exact socket-bind time |
| Model registration starts |60.855 | Artifacts registered before router exists |
| Registration returns |78.254 | Not native-engine-ready |
| Router start observed |79.257 | Polling observation, not exact construction timestamp |

All97 successful requests pass the exact prompt/adapter/count binding; E2E
identity and TPOT recomputation errors are0ms. This is not independent numerical
LoRA correctness. All100 terminal records are retained; no readiness-based
arrival shift, retry, early truncation or denominator change was applied.

The bounded finalization correction now has an actual native failed-workload
witness: model_qualification.json is saved, classification is
qualification_request_failure, service/replay/watchdog exits1/1/0. Measurement
completion is true and workload pass is false. Actual GPU contexts clear,
service path removed, empty auxiliary stopped, overlay2 exactly restored.
1226 watchdog samples; swap0 and no high/max/OOM/OOM-kill.93 regular native log
files copied with every SHA equal; four obsolete socket files are not copied.

Raw root results/ieee_tc/serverless_qualification/d71_20260927. Main curated
20260927_http_qualification_d71.json/.csv preserves both attempts and all100 rows.
Do not repeat this startup experiment to obtain a cosmetic100/100. Return to
the planned original/repaired1,000-request development pairs with equal
bootstrap, native loading, scaling and inputs; retain any startup failures
there too. Those pairs isolate dispatch polling, not the absent-front-door
boundary. Formal, full-pool, remote and numerical correctness gates stay open.

## D72: 7B original side of the 1,000-request polling pair

The first planned development side is complete. Native loading, D71 bootstrap,
FP16 TP1 and existing inputs are retained. Exposed the native historical
min1/max4/keep_alive10,target2 settings in the existing contained helper.
The diagnostic engine retains max_num_seqs4/eager/prefix-cache; this is not
claimed identical to the historical main-table backend or an optimized M1 point.
Source7f135add13082af1cc2e428c7a8e497e9360d949 passed41 checks and was backed up
before launch. Original/repaired source views differ only in the measured
load-balancer method. 3B development target8 comes from the later historical
seq8 deployment, not the earlier target4 deployment.

| Observation | 7B original, one development run |
|---|---:|
| Planned / arrived / terminal |1000 /1000 /1000 |
| Protocol-valid / failed |997 /3 |
| Actual native output tokens |122176 |
| Offered unique adapter IDs / observed serving instances |60 /4 |
| Mean router queue, valid responses |294.746s |
| Mean service TTFT, valid responses |380.685ms |
| Mean / P95 TTFT, valid responses |295.140s /534.178s |
| Mean E2E, valid responses |302.789s |
| P50 / P95 observed assignment gap |1.004s /5.009s |
| E2E identity / TPOT recomputation max error |0ms /0ms |
| Service memory peak / minimum host available |51780816896B /57868718080B |
| high / max / OOM / OOM-kill / swap |0 /0 /0 /0 /0 |

The same initial three HTTP500 missing-router failures are retained. No later
failure, shifted arrival, hidden retry, new weight or generated trace. Native
responses validate the declared token/prompt/adapter binding, not independent
LoRA numerical correctness. The one-second cadence supports a control-path
limitation; gaps above one second also retain capacity waits. This original
side alone does NOT measure how much a polling repair removes, nor permit a
comparison to the old four-minute mean under a different execution contract.

1704 watchdog observations, actual GPU contexts released, service path removed,
empty auxiliary stopped, loader overlay restored byte-for-byte. Final outcome
qualification_request_failure (service/replay/watchdog1/1/0), measurement
complete=true/workload_passed=false.108 regular native files copied/SHA equal;
four obsolete socket files excluded. Raw root d72_20260927 is70MiB.

Existing summarizer now consumes a finished native JSONL journal, rejects
truncated/duplicate/inconsistent records, preserves failed/offered rows and
recomputes E2E/TPOT.46 baseline checks PASS2.982s. Main existing plotter renders
the diagnostic without rewriting historical figures. First invocation used an
environment without matplotlib; reused conda base instead of installing. First
render had overlapping note/legend; retained as rejected preview. Final v2 has
collision checking,3.45×2.85-inch PDF/PNG, embedded Times New Roman, below-bold
subtitles and visual QA. No model rerun for either plotting correction.

Main artifacts:
- paper_results/ieee_tc/serverless_audit/20260927_7b_original_polling_d72.json
- same stem _evidence.json:24 raw hashes, resource/cleanup/test/QA evidence
- figs/ieee_tc/serverless_audit/d72_7b_original_v2/: two plots and full summary

Next: the already prepared run7b_repaired.sh, SAME HTTP/deployment settings,
fresh private/cold runtime and fresh overlay receipt; do not repeat the original
side or change bootstrap/loading/target after seeing its result. Then the 3B
pair (repaired first). Complete cleanup/validation/table/figure between runs.
