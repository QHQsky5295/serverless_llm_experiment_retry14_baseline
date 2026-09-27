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
