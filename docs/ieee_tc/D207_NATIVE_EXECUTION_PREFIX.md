# D207 — native execution prefix qualification

Status: SEALED. Actual 22-request execution-metadata diagnostic passed on attempt
4; all three earlier failures are retained. This is P2 correctness evidence, not a serving optimization,
Full replay, new workload, numerical qualification or model acceptance.

## Question and scope

D166/D167 checked checkpoint-to-slot contents before/after generation. D205
checked the serialized mapping validator; D206 qualified an isolated observer
using CPU fixtures. None alone shows that the intended request uses the intended
physical adapter slot at every real forward/logits boundary. D207 connects these
existing components, without repeating content audits or changing scheduling.

Reuse the existing 7B first 22 requests, local frozen pool and
`configs/ieee_tc/20261002_d166_7b_slot_content.yaml`. These requests contain the
four existing weight-content classes, not 500 independently trained adapters.
Local artifacts here isolate the execution diagnostic; this is not a real-remote
performance point. Production remote cache D78/D80 is unchanged and not rebuilt.

The opt-in `native_execution_metadata` preflight mode starts D206 after native
initialization and a drained scheduler, retains/drains its events after each
request, and stops it before final cleanup. Independent expected lease, owner,
adapter ID and target come from the input/request-side acquisition, not the
observed batch. Native begin/end acknowledgements bind actual backend request
IDs, joined to internal request IDs by the existing exact frontend/core retirement
receipt, not by removing randomized suffixes. Every nonempty iteration must have ordered forward/logits before/return
events with matching real rows, padding, physical mapping and layer/buffer
identity. Native scheduled/completed counter deltas independently check total
nonempty iteration coverage. Empty iterations are retained separately.

Raw observations are retained before validation; errors remain failed/partial.
No retry, timeout change, eager-only bypass, numerical tolerance adjustment,
SLO/configuration selection or new production hook. D206's device readback and
stream fences perturb timing, so none of these latencies enters performance
statistics. Metadata at call boundaries does not establish kernel arithmetic,
CUDA graph arithmetic, concurrent Full correctness or full-pool qualification.

Implementation reference: [vLLM v0.30.0 V2 native model runner](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/model_runner.py).
The actual V2 path is NOT the older `v1/worker/gpu_model_runner.py` inspected
for D206. Installed V2 source SHA matches the fetched official version:
`174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce`.
Native execution and scheduler counters are distinct boundaries;
successful logical request binding alone is insufficient.

## CPU results

| Check | Result | Scope |
|---|---:|---|
| Artifact/execution journal checks | 33 passed, 1.677 s | Includes 9 new tests; CPU fixtures |
| Corrected native drained-schema checks | 34 passed, 1.681 s | 10 new tests total; original attempt retained |
| Existing worker/warm-reference/basic smoke regression | 376 passed, 19.617 s | Overlapping test groups, not independent repeats |
| Corrected V2 split forward/sample observer + integration/regression | 414 passed, 20.539 s | Adds FULL/PIECEWISE/NONE, split sampling, empty and unsupported-runner tests |
| Exact external/internal request identity + regression | 416 passed, 20.225 s | Adds independent-retirement negative control and raw internal-ID collection |
| Host OOM/high/max/swap events | 0 in test scopes | 3/4 GiB, swap 0, CPUs 2,3,26,27 |
| Actual GPU execution coverage | 22 requests; 3,917 nonempty iterations | Actual FULL and NONE; PIECEWISE remains CPU-only coverage |
| Numeric correctness / `n_correct` | Unknown | Not upgraded by mapping checks |

All test scopes terminated and their actual cgroup paths were absent. One premature
read of regression terminal resource files occurred while that scope was still
active; the later terminal records were captured and checked. No test failed.
Protected historical result seals passed before/after all test groups.

## GPU result and next action

Attempt 1 stopped before installing the observer or submitting a request.
The native scheduler correctly returned `admitted=[]`, scheduled/completed=0
and no unretired/deferred work. The new preflight had incorrectly compared the
list with numeric zero. This was an integration/schema error, not a model,
scheduler or resource failure. The exact schema check now requires an empty
list and integer zero counters; malformed numeric/boolean inputs are rejected.
All failure evidence remains, and the original production logic is unchanged.
One physical GPU lease was released and both service/auxiliary scopes disappeared.

Attempt 2 also stopped before submitting requests. The D206 observer assumed
the older runner's `_model_forward`, but the active V2 runner has no such method.
Installation rolled back; `execution_journal=[]`, no generated-request records.
The engine itself initialized, including FULL/PIECEWISE graph capture. This is
an observer compatibility failure, not a numerical or performance result.
The GPU lease was released, exact service/auxiliary units and paths were absent.

The correction targets the actual V2 runner explicitly; unsupported runner
classes fail before hook installation. It captures the returned native
`prepare_inputs` batch; observes FULL via `run_fullgraph`, PIECEWISE via
`run_pw_graph`, and NONE via `model.forward`; and binds the same batch across
the separate `execute_model` and `sample_tokens` phases. PIECEWISE's internal
model call is not double-counted. `iteration_return` means the completed pair,
while `forward_phase_return` and `sampling_begin` preserve the split. Native
request-state indices and LoRA state must agree with independent request leases.
Graph mode is not changed and no installed vLLM source is patched. Default Full
still installs no observer. The new execution contract is
`vllm_v2_split_forward_sample_v1`; old CPU-only journal fixtures are not treated
as this contract's actual GPU qualification.

Attempt 3 reached the first request, then the observer rejected the difference
between external `req_1` and native `req_1-ab14385f`. The native adapter ID was
963725 (finance), not evidence of a different adapter. The exception terminated
the core; zero requests completed, and only the initial observer snapshot was
retrievable. Error logs, failed cleanup-observation response and GPU release are
retained. This must not be presented as an adapter arithmetic error.

The final correction reuses the existing frontend submission registry and
idempotent `ieee_retire_generation`: its cached acknowledgement binds the exact
external/internal request IDs and confirms native retirement. The collector
retains internal execution counters separately from external lease bindings.
Only the offline validator joins them; no suffix stripping, execution-order or
adapter-based guessing is allowed. Wrong native ID, external ID, clock, lease,
owner or retirement state fails validation. No production ID policy is changed.
See [official request-ID assignment](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/input_processor.py).

## Actual GPU result

| Item | Observed result | Interpretation |
|---|---:|---|
| Original requests completed | 22/22 | Same D166 inputs and configuration |
| Native output tokens | 3,917 | Every request exactly matched its target |
| Native nonempty iterations | 3,917 | Independent scheduler scheduled/completed deltas both 3,917 |
| Empty iterations | 44 | Retained separately, no invented forward calls |
| Forward / logits boundaries | 3,917 / 3,917 | All real rows checked against exact intended adapter IDs |
| Events / drained snapshots | 31,468 / 24 | Continuous sequence and final counter agreement |
| Actual graph modes | FULL, NONE | No actual PIECEWISE claim from this prefix |
| Earlier attempts completed | 0, 0, 0 | Schema, runner-interface, then external/internal-ID observation failures |
| Performance / arithmetic / full-pool qualification | No / No / No | `n_correct` remains unknown |

Attempt 4 passed with worker PID 3749672, TP=1, GPU0, the common service resource
domain and its recorded CPU set. One physical lease released; service and
auxiliary cgroup paths disappeared; GPU idle was verified at 23:06:14+08.
The 35,555,149-byte raw result SHA is
`993c2956ee2c007dd678327932dc5368e4f454f43595587e8d3197088a46467d`.
Launch receipt SHA is
`d8a47fdcad9ce9f9336f94f4bce50c708c7f626f658e2c0ba4aa153fc1541489`.
No remote service was required or changed.

This closes the specific missing execution-metadata link for the existing
22-request, four-content-class diagnostic. It does not establish numerical
equivalence of all kernels, concurrent Full execution, 500 independent trained
adapters, or explain D203's 126 output-hash changes. Detailed readback/synchrony
means its measured time is excluded from performance summaries. The status
table, rather than a speedup plot, follows the academic-plotting execution rule.

Disk prerequisite: the PyPI-only cache audit found 187609088 B but was not applied.
A separate audit matched one older CUDA PyTorch wheel against the official index
SHA. Its first urllib HEAD received HTTP403, with no deletion. The same official
URL via curl returned 200 and matching size; reference/open-file/ownership guards
were rerun. Only that verified HTTP download plus header (2267332608 allocated B)
was removed. Installed environments, models, artifacts, measurements and four
tracked project states were unchanged. All audits/failures are retained.

After diagnostic cleanup: independent retained-data validation and scoped
backup. Then address remaining numerical/output-difference and common-reference
evidence toward actual old/new Prime G1/G2 acceptance. SevenB acceptance precedes
3B; baselines, M1/M2, A1–A5 and S1–S13 remain pending. Do not rerun Full without
a new evidence-backed hypothesis.
