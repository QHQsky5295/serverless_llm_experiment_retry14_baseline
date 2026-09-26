# Serverless: independent native launcher view

Status: D65 infrastructure, D66 native-input identity and D67 actual3B native
backbone loading qualification pass. Four fixed-output requests and actual
worker/source/resource release are verified. LoRA, full-pool,7B loading and
original/repaired1,000-request pairs remain open. No comparative performance or
polling-benefit claim follows from this narrow prerequisite.

## D67 final result: native3B loading qualified, not performance

| Item | Attempt1 | Attempt2 |
|---|---|---|
| Native store GPU-load confirmation | Yes | Yes |
| Native output counts | 152,59,123,217;4/4 | 152,59,123,217;4/4 |
| Actual backend readback | Caller signature error | TC9f50241 backend; native store; `serverless_llm`; existing checkpoint |
| Worker group / CPU | Readback incomplete | Actual PID249255 in72/80GiB service group; all40 allowed CPUs |
| Peak service RAM | 41,180,987,392B | 41,225,433,088B |
| Minimum host available | 68,445,958,144B | 67,196,321,792B |
| Watchdog samples | 141 | 143 |
| high/max/OOM/swap | 0 | 0 |
| Native context release / service group removed | Yes | Yes |
| Overlay restored after exit | Yes | Yes |
| Overall narrow qualification | Incomplete | Pass |

Attempt2 uses the same model/configuration/four inputs, with only the diagnostic
driver's pre-start source/library composition corrected. Actual readback SHA is
994c80ae9c6106a6469f9d8d5889132e5c2032608c4e497141c00127f91cc777;
the worker sees one GPU, while native store sees all four correct device UUIDs.
The complete supervisor census also checks actual GPU-owning descendants. Native
confirmation UUID0a9cc932-fedd-410a-97ae-2969e0c0f87a is preserved. Service/watchdog
exit0/0; private TMUX and owned domains cleaned, both overlays restored byte-for-
byte after exit. No new artifact/trace/remote service or baseline policy change.

Qualification driver7f2de0de20d51f6bdc79cecff96d67daeb736629 is backed up;26 CPU
checks pass. Main curated JSON/CSV:
`paper_results/ieee_tc/serverless_audit/20260927_native_model_qualification.*`.
Eight request rows preserve both attempts; native logs copied from the private
temporary area and every copied member SHA checked. Raw successful receipt
SHA65fb224aadbe1a5a51ca83d660711605e75a5a685cfca6d53c0105191bed072d,
launch SHAe3f9f744f2018ef5e8af1df28ad5001a8937fcf6071cda0139193b972b41a445.

Do not repeat this passed four-request loader experiment. Next: approved paired
original/repaired development replay, with necessary native7B identity/format
and shared generation/LoRA instrumentation. Only the3B native checkpoint exists
under the inspected `models/vllm` namespace; do not assume7B is already converted.
First audit other referenced historical locations before any necessary conversion.
The real-remote and independent LoRA correctness gates remain separate; these
four explicitly backbone-only requests cannot satisfy either. Storage daemon
GPU ownership must remain in lifecycle measurements. No M1/M2/ablation starts
solely because this diagnostic passed.

## D67 preregistered actual native model qualification

Use the existing 3B TP1 FP16 checkpoint, the exact D66 compiled store, audited
TC9f50241 source and loader-only reversible overlay. No new checkpoint, adapter,
trace or alternative loading path. The official backend's absent `load_format`
selects `serverless_llm`; explicitly setting an ordinary format would instead
bypass native storage and is prohibited here. Primary source checked again:
https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/vllm_backend.py .

Extend the existing contained launch helper with `qualify-model`. One native
instance (min=max=1) remains alive for actual worker/source inspection; all four
GPUs are visible to the native store, as required for correct device UUID mapping.
This is a loading diagnostic, NOT an elasticity or selected M1 working point.
Native pool32GB, four store threads,32MB chunks retain current launcher defaults;
two raylets retain4+4GiB object stores within the common72/80GiB envelope.
Keep confirmation enabled. Set the already bundled library directory explicitly,
as established in D66, without rebuilding binaries. Set a private TMPDIR so the
native100MiB temporary disk calibration cannot overwrite another job's file.
Storage-aware scheduling is enabled; live migration is not asserted.

Reuse the first four request messages from the existing 3B seed42 mainv1 trace.
For this backbone-only loader test, encode `role: content` with its existing
tokenizer, cap input759 and use original expected output capped256, greedy and
ignore-EOS. Record full input token IDs/hash and native observed output counts.
Do not attach adapters and do not interpret success as adapter correctness, a
complete fixed-output comparative contract, or a workload performance result.
No invented prompt, regenerated trace or base-model fallback is involved: the
explicit tested target is the backbone loader, not a failed LoRA request.

Acceptance: four successful fixed-length responses; actual backend actor has
native load format/checkpoint/source/store identity, expected cgroup and CPU
affinity; native GPU load confirmation exists; external supervisor verifies
whole-tree/context release. Capture registration, responses, worker/process
identities, full native logs and resource monitoring even on failure. Registration
alone is not readiness. Request protection1800s; infrastructure/router timeout600s.
The final loader-only restore occurs ONLY after actual owned processes exit.
Record original/installed/restored hashes and do not touch another project's
routing overlay. Publish a qualification table immediately, not a performance
plot. Then return to approved original/repaired model pairs; semantic/remote and
full-pool gates remain independently open.

### D67 attempt1 immediate result (qualification incomplete)

| Evidence | Observed result | Interpretation |
|---|---|---|
| Native registered/load path | Existing6,425,499,648B; GPU replica confirmation success | Actual native loading, not ordinary HF loading |
| Four fixed output targets | 152/59/123/217, all exact native counts | Backbone generation only, not LoRA correctness |
| Actual actor readback | `TypeError: too many positional arguments` in Ray caller signature check | Diagnostic failure; overall qualification NOT passed |
| Resource samples | 141; peak41,180,987,392B; minimum host68,445,958,144B | No high/max/OOM/swap events |
| Cleanup | Delete succeeds; supervisor confirms native contexts clear/group removed | Owned resources released |
| Reversible overlay | Restored after complete exit; restore receipt retained | Shared environment original sources recovered |

Classification: `protocol_or_launcher_error`, not baseline inference failure.
All raw responses and the failure remain. The diagnostic driver, unlike its
native workers, did not select TC source/store PYTHONPATH and bundled libraries
before Python startup. Ray named-actor reconstruction imports classes in that
driver. The inspected Ray2.54 import-failure path creates placeholder methods;
the observed signature failure is consistent with that path, but its original
unpickle traceback was not recorded, so the exact import exception is unknown.
Correct the driver composition, require actual imports before starting service,
and repeat the affected qualification with a new attempt key. Do not change any
baseline policy, byte, model config or resource limit. This is not a blind OOM
retry. Store daemon itself holds GPU contexts on all four visible cards; physical
lifecycle accounting must include that ownership even when only one engine runs.

Raw root: main `results/ieee_tc/serverless_qualification/d67_20260927`.
Model receipt SHA1f9a495d2acb28c785c758b60895204f49af47d41db7bc3d943d1406e0d0ac8a;
launch SHAfc94520e250b0a693ff45b6906f5b96fd797e6b3d46aefac32fe0f57eee1e06e;
store log SHAa6851e6823f0ef8ad9ad8c5e74372d0b1239c7dfe3a358983151d8a13d627682.

## D66 native-input qualification (before model loading)

Reuse the existing native checkpoint
`models/vllm/v43-sllm-native-smoke-llama32-3b` and original local
`LLM-Research--Llama-3.2-3B-Instruct`; do not reconvert or copy either model.
The native directory contains one 6,425,499,648-byte tensor file, 170 indexed
FP16 tensors. The original HF index contains 254 tensors. These counts differ
because native vLLM packs Q/K/V and gate/up projections, not necessarily because
weights are missing. The audit mapping follows the
[vLLM 0.10.2 Llama implementation](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/model_executor/models/llama.py).

`audit-checkpoint` in the existing launcher adapter compares **all** tensor
elements after the source-to-FP16 cast, in the original projection order.
Bounded CPU row slices, exact source/index/member closure, shape/stride/extent
checks, whole native/source SHA values and unchanged input identities are
required. Config/tokenizer files must match. No CUDA context, conversion,
model/pool/trace regeneration or old-result change is permitted. Wrong weights,
swapped projections, missing keys and changed tokenizer must fail in CPU tests.
The receipt records current identity only; it cannot backfill an old run's SHA
or establish runtime loading/LoRA correctness. A unique output preserves failure.

Reuse the existing official-loader-only version port:
`/home/qhq/relayserve_serverless_llm/scripts/relayserve_v4_3_apply_serverlessllm_vllm_store_overlay.py`.
Its `preflight` checks six exact vLLM preimages and the official patch without
installing. Do not import that project's M4/routing policy overlay or global
cleanup. The native environment has no `sllm_store.torch` or `bin/sllm-store`;
the previous successful native smoke used the existing compiled package at
`installs/serverless-llm-store-0.8.0-vllm0102-py312-v1/site-packages`.
Future launch must explicitly select that package and its executable, preserving
the complete package/build identity. No reinstall is needed. The
[official native loader](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/sllm_store/sllm_store/torch.py)
calls `confirm_model_loaded` unconditionally through `load_dict`; the old
`SLLM_SKIP_CONFIRM_MODEL_LOADED` switch applies to the separate transformers
loader. Still set it to 0 explicitly for the native qualification.

| D66 check | Current result | Interpretation |
|---|---|---|
| Launcher/router and packing-layout tests | 24 pass, no failure/error/skip | No scheduling policy changed |
| Actual pinned-environment CPU tensor fixtures | 3 pass, no failure/error/skip; CUDA uninitialized | Complete-byte comparator detects corrupted/reordered weights |
| Existing 3B native checkpoint vs original weights | All 170 native tensors match all 254 HF source tensors after FP16 cast; all 6,425,499,648 native bytes checked | Existing native checkpoint is reusable; no new conversion/copy |
| Loader overlay preflight / package import | Six preimages and official patch pass; all96 package files match; corrected library selection imports actual native modules | No install, engine creation or model load yet |

First audit launch stopped before service start: its outer receipt output was
relative, while the established guard requires an absolute new output path.
`checkpoint_console.log` preserves exit1; no checkpoint bytes were read and no
model ran. Classify `protocol_or_launcher_error`; retry with an absolute new
receipt/attempt path, without changing or relaxing the guard.

Attempt2 completes in 93.656 s (audit wall time, **not startup latency**).
Native file SHA256:
`3f937cdc2c3b637cf61a19809670a0e6146b943a049ba407fa5dc5e2b46f9979`.
All five shared config/tokenizer files match.95 external resource samples,
peak6,769,774,592B, minimum host available110,954,188,800B; swap/high/max/OOM0.
Service/watchdog exit0/0, service released, auxiliary group empty then stopped;
CUDA remained uninitialized. Raw receipt SHA256
`6f074b1d07d993664d390eb6a00297b60b61dcfd7844dfc4b8b8e4341f3a0cb3`.
Main repo delivers the full170-row CSV and summary JSON under
`paper_results/ieee_tc/serverless_audit/20260927_native_checkpoint_identity.*`.
Do not repeat this successful complete byte audit or relabel it as inference.

The loader-only `preflight` now passes all6 exact vLLM preimages plus the official
patch; nothing installed. All96 compiled-store bundle members and their hashes
match the preserved manifest (no extra/missing members). First CPU import stops
at `_checkpoint_store`: dynamic linker cannot find `libglog.so.1`. The exact
bundle already contains this library; do not rebuild or install another copy.
Record the dependency and explicitly select the bundle's library directory for
the qualification process. This is a deployment environment correction, not a
baseline policy change. No Ray/store/API/CUDA model was started; the failed
import is retained in `native_import.log`, owned CPU group empty/events0.

Second import succeeds using only the bundle directory in `LD_LIBRARY_PATH`,
consistent with the [Linux dynamic-loader search order](https://man7.org/linux/man-pages/man8/ld.so.8.html).
Read ELF `RUNPATH` confirms an obsolete temporary build directory, explaining
the first failure. Exact libraries/binaries remain unchanged. Actual TC9f50241
backend import constructs `AsyncEngineArgs` with `load_format=serverless_llm`,
the audited native checkpoint, TP1 and FP16; `engine` remains None and CUDA
uninitialized. Readback: Python3.12.12, torch2.8.0+cu128, vLLM0.10.2, Ray2.54.0,
grpc1.76.0. The compiled store's CUDA-runtime dependencies still need actual
GPU/store qualification; import success is not proof of GPU compatibility.
Second CPU group also empty/events0 then stopped. Full module paths and hashes,
preflight members, exact bundle and both attempts are recorded in main
`paper_results/ieee_tc/serverless_audit/20260927_native_inputs_preflight.json`.

For the **next actual guarded native launch**, inherit these explicit selections
in head, worker, API and store (no global environment/package installation):

```text
SLLM_REPO_ROOT=/home/qhq/serverless_llm_baselines/vendor_new_baselines/ServerlessLLM_new_main_20260518
SLLM_EXTRA_PYTHONPATH=/home/qhq/serverless_llm_baselines/installs/serverless-llm-store-0.8.0-vllm0102-py312-v1/site-packages
SLLM_STORE_BIN=/home/qhq/serverless_llm_baselines/installs/serverless-llm-store-0.8.0-vllm0102-py312-v1/site-packages/bin/sllm-store
LD_LIBRARY_PATH=/home/qhq/serverless_llm_baselines/installs/serverless-llm-store-0.8.0-vllm0102-py312-v1/site-packages/sllm_store
SLLM_SKIP_CONFIRM_MODEL_LOADED=0
```

Keep both old failed paths and all prior results. Next use the existing
reversible loader installer with a new backup/receipt; restore exact bytes only
after owned workers actually exit. Prove actual store registration, complete GPU
UUID coverage, native load/confirm path, model-worker containment and cleanup.
Do not merely repeat this import or the successful byte/Ray-only audits. No
remote service, LoRA qualification, repaired-polling benefit or M1/M2 result is
established by D66.

After this table is completed, advance directly to actual owned store/model
loading. Do not repeat the D65 infrastructure-only witness. Native loading,
LoRA qualification and the original/repaired model pairs remain separate gates.

## D64 source findings and decision

The historical new-Serverless wrapper selects the existing
`sllm_vllm0102_newserverless_20260518` environment for head and worker. Its Ray
`_version.py` records 2.54.0, commit
`48bd1f8fa43d0e8222b0f57357b99b48c7437ed3`. The old default head/worker
environments contain Ray 2.48.0; they are not the selected native-loader path.
An environment directory or version string alone does not verify actual imports.

The unchanged `start_serverlessllm_stack.sh` invokes source installation and
global cleanup before launch. Its stop script uses broad session/process
matching and `ray stop --force`. Both head and worker accept the same optional
`SLLM_RAY_OBJECT_STORE_MEMORY_BYTES`, so setting that variable to 8 GiB would
configure 16 GiB for the one-head/one-worker deployment. This is a conditional
source finding, **not a claim that historical runs allocated 16 GiB or OOMed**.

The new `scripts/prepare_ieee_tc_serverless_stack.py` reuses the exact five
native shell scripts and writes an exclusive, small per-run view. It does not
rewrite the originals, installed packages, another project's pinned launch
sources, models, adapters or traces. Original source SHA values and generated
script SHA values are saved in the view manifest.

The adapted view:

- Verifies the existing TC launch receipt, actual service domain and external
  watchdog before native startup. It is not a separate resource supervisor.
- Creates no service at preparation time; the generated stack rejects an
  unguarded invocation before starting tmux/Ray. Runtime paths are explicit,
  private, exclusive and short enough for the tmux socket.
- Uses a fresh explicitly named tmux socket with no user tmux configuration;
  removes automatic installed-source synchronization, global cleanup and log
  replacement. External TC gated-launch cleanup owns the whole service group.
- Uses the established one-head/one-worker-raylet deployment. Each gets 4 GiB
  of object store, totaling the plan's initial 8 GiB. This equal initial
  partition is not a performance-selected configuration and is not a claim
  that all service RAM fits. Native store, model copies, HOST cache, controllers
  and spill remain inside the 80 GiB service budget.
- Gives the head a private Ray temp directory; workers learn that directory
  from the head. Both nodes get explicit separate spill directories. This
  matches the installed CLI and the [Ray 2.54 start interface](https://raw.githubusercontent.com/ray-project/ray/ray-2.54.0/python/ray/scripts/scripts.py)
  and [spilling implementation](https://raw.githubusercontent.com/ray-project/ray/ray-2.54.0/python/ray/_private/node.py).
- Exposes the full worker GPU set to the store, as required by the prior
  successful native-loader witness, and retains the storage-aware native CLI.
  Direct-path fallback is rejected. This does not install the absent vLLM
  native-loader overlay or substitute for validating it.
- Checks private control/API/store ports before starting anything. This is
  not a reservation or ownership proof for a later listener; actual worker,
  store UUID, API/source and raylet-capacity readbacks remain mandatory.

No router, autoscaler, migration, inference, generation or adapter policy is
changed by this adapter. The previously authorized ready-before-wait patch
remains a separate source identity and comparison variable.

## Immediate qualification table

| Question | Current evidence | Remaining requirement |
|---|---|---|
| Were old launch scripts overwritten? | All five pinned originals retain exact SHA | Recheck on each view creation |
| Can a new run stop unrelated experiments? | Global cleanup calls removed from this view; private socket; guarded launch required | Actual whole-tree ownership and release |
| Is the object-store total 8 GiB? | Actual generated head/worker command arguments sum to 8 GiB in CPU tests | Read back both live raylet capacities and cgroups |
| Does the store address every worker GPU? | Generated command exposes the same complete set | Actual native store GPU UUID inventory |
| Is the paper's loader in use? | Native storage-aware command retained; direct mode rejected | Reversible overlay, checkpoint identity and actual load evidence |
| Has polling's contribution been measured? | Historical logs and six real-method tests retained | Two model-specific original/repaired replay pairs |

The table is used instead of a performance plot: no inference timings were
measured here. It follows the approved per-experiment delivery/plotting rules;
there is no new ranking or apparent zero-latency point.

Final D64 validation: 11 new launcher checks and the 6 existing real-method
router checks pass (17 total; zero failures/errors/skips). Tests execute the
generated leaf command arguments using a non-Ray recorder, verify shell syntax
and original SHA, reject source/aggregate drift and unguarded stack invocation.
They do **not** launch Ray or infer live worker containment. The initial 15-test
pass is an overlapping earlier selection, not another independent experiment.
Both named CPU-only resource domains have empty actual process lists and zero
high/max/oom/oom_kill events, then were stopped. No model, remote service or
performance job remains running from this change.

## Entry and integration boundary

Preparation only (new paths are required; parent directories must exist):

```bash
/usr/bin/python3 scripts/prepare_ieee_tc_serverless_stack.py prepare \
  --output /absolute/new/qualification/launcher \
  --private-root /tmp/ptc-sllm-unique-run \
  --main-repo /home/qhq/serverless_llm_experiment_retry14_baseline \
  --gpu-ids 0,1,2,3
```

The view must run from a controller already admitted through the main repo's
existing `ieee_tc_preflight.py gated-launch` entry. Supply explicit audited
head/worker/store environment prefixes, native source path, checkpoint namespace
and private ports. Do not use the old stop script. The controller must stay alive
through qualification/replay; return from native `start` does **not** mean GPU
release. The external owner verifies/cleans remaining processes. A future full
performance runner still needs external replay, physical GPU accounting, model
qualification and the shared generation/remote contracts.

Run the CPU tests without starting Ray, a model, or a remote service:

```bash
/usr/bin/python3 -m unittest discover -s tests -p 'test_ieee_tc_serverless*.py' -v
```

## D65 preregistered actual two-raylet witness

Next measure the same guarded head/worker prefix, stopping explicitly before
native store/API/model startup. `qualify-ray` runs in the existing admitted
72/80 GiB service domain with independent 4 GiB auxiliary watchdog, under TMUX.
Use the existing Ray2.54 native environment and four logical GPU resources;
no CUDA/model inference is performed. This is not a substitute full baseline.

Require exactly two live owned raylets, head GPU0 and worker GPU4, each reporting
4 GiB object store (also read from both actual process command lines). Schedule
one CPU witness on the head and four distinct one-GPU-resource witnesses on
the worker. Their actual processes and subprocesses must share the admitted
resource domain and service CPU set; all four logical GPU assignments must be
covered. Record actual imports, native node IDs, PIDs/birth identities, limits,
raw private logs and watchdog evidence. No native inference means no numerical
adapter-correctness or physical-GPU-use claim follows.

After observations, kill only those actor handles and the explicitly private
tmux server; the existing external supervisor still verifies whole-tree release.
Keep failures and new attempt paths. No global Ray stop, source installation,
checkpoint conversion, remote fetch, model/pool scan or performance ranking.
The result table and cleanup evidence must precede any next experiment.

| D65 attempt | Observation | Interpretation / next action |
|---|---|---|
| 1 | Outer argument parser rejected forwarded `--host` as ambiguous with its own `--host-copy-*`; exit 2 before any service start | Launcher error, not a Serverless result. Preserve console log; disable abbreviation at the argv-forwarding boundary and test both outer and nested parsers before retry |
| 2 | Real TC service admission succeeded, but script-view verification compared a logical result path with its resolved physical path and stopped before Ray startup | The repository's existing results-parent symlink is legitimate. Canonicalize the destination once at preparation and preserve the requested alias; retain strict identity checks. Add a real symlink-parent regression case |
| 3 | Head session exited before a Ray cluster became reachable; timeout retained. A shell-only reproduction shows that explicit resource JSON becomes invalid with an extra closing brace | Generated leaf scripts now separate default JSON assignment from parameter expansion; both default and explicit arguments are parsed in CPU tests. No Ray/model result follows |
| 4 | Actual two-raylet and five-worker witness passes, including spawned children; full owned cleanup passes | Advance to native checkpoint/loader qualification, not formal performance. Do not repeat this infrastructure-only witness |

Attempt1's auxiliary group is empty with all memory pressure/OOM events zero
and was stopped. No Ray/model/remote process started; no inference time or
resource score can be derived from this failed invocation.

Attempt2's service and auxiliary groups are released/empty; watchdog reports
one resource sample, no abort, native contexts clear and the service path
removed. It imported the actual Ray2.54 environment but did not start a cluster.
Do not misclassify either launcher failure as a baseline resource failure.

Attempt3 is cleaned: service/watchdog exit1/0, service removed, no owned GPU
context, auxiliary actual process list empty and all memory events0, then stopped.
Its dead tmux pane had already disappeared, so the precise native stderr was not
retained; the invalid-JSON cause is independently reproduced, not falsely quoted
from that run. Future views use one hash-bound private tmux configuration with
`remain-on-exit on`, retaining an exited pane until owned cleanup. This is the
documented [tmux pane-retention behavior](https://man.openbsd.org/tmux.1), not a
process restart or changed baseline policy. The original five scripts remain
unchanged. No readiness marker alone can satisfy the actual-node witness.

## D65 measured result and next boundary

| Measurement | Actual observation | Scope |
|---|---|---|
| Ray import | 2.54.0, exact inspected commit | Native existing environment |
| Live raylets | 2, both owned; command-line and node-table agreement | One head, one worker on this host |
| Object-store capacities | 4,294,967,296 B each; 8,589,934,592 B total | Configured live capacities, not 8 GiB resident usage |
| Worker/child ownership | 5/5 workers and 5/5 spawned children in the admitted group | 1 head CPU actor plus 4 logical-GPU actors; no CUDA |
| CPU affinity | All ten use `4–23,28–47` | Actual affinity, not delegated cpuset |
| Service limits | high72/max80 GiB, swap2 GiB | Shared actual OS memory limit |
| Sampled memory peak | 1,324,113,920 B over 11 watchdog samples | Infrastructure only |
| Witness memory.peak | 1,347,006,464 B | Separate instantaneous readback; not a model-serving footprint |
| Pressure / swap | high/max/oom/oom_kill all0; sampled swap0 | No OOM or budget relaxation |
| Cleanup | service/watchdog0/0, private tmux0, group removed, owned native contexts clear | Auxiliary empty/events0 and stopped too |

Raw root (existing results symlink resolves here):
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d65_20260927/`.
Curated exact values, all four attempts and source SHA values are in the main
repo's `paper_results/ieee_tc/serverless_audit/20260927_contained_ray_qualification.json`.
No inference, native store or model was loaded. All22 CPU launcher/router checks
pass, zero failures/errors/skips; earlier20/21 selections overlap, not repeats.

Important readback: Ray advertises per-node logical `memory` resources of
107,262,640,128 B (head) and106,816,192,512 B (worker), larger than the shared
80 GiB OS limit. These are scheduling estimates, not enforced separate RAM
allowances or actual allocation. Do not add them as physical usage or claim
Ray's native memory monitor understands the shared cgroup. Inspection of the
same two actual raylet logs confirms total memory134626840576B and a0.99
threshold133280571392B, not the service cap. This agrees with the inspected
[Ray2.54 memory-monitor source](https://raw.githubusercontent.com/ray-project/ray/ray-2.54.0/src/ray/common/memory_monitor.cc),
which checks root cgroup files rather than resolving this nested user scope.
The native monitor is left enabled and unchanged; its behavior is recorded,
not called service-aware. The independently verified shared OS cap and external
watchdog remain authoritative. This known behavior does not justify another
infrastructure-only rerun, relaxed budgets, or an unrelated Ray rebuild.
Next reuse and verify existing native checkpoint/overlay/store artifacts, then
measure actual model loading and finally the original/repaired development pair.
The previous three failed attempts stay visible and are not baseline failures.

The existing3B native checkpoint and reversible official-loader port are still
available from the earlier project. Only read them in D65: no installation,
checkpoint conversion or model service. Reuse the exact source-gated loader
installer `relayserve_v4_3_apply_serverlessllm_vllm_store_overlay.py`, not that
project's unrelated M4 routing/affinity overlay. The TC audited source remains
official9f50241; the other project's witness used0fd00ca. Their runtime behavior
and evidence must not be interchanged just because the Python environment is
shared. Record the full effective source/loader identity during model qualification.
# D68 missing 7B representation: export completed, audit pending

This is a prerequisite for the approved original/repaired request comparison,
not a performance run. Source checkpoint3052ab9559a117de6d4a5e56525ba0b165df9769
was pushed and independently verified before execution. Existing native exporter
unchanged; TP1/FP16, one visible GPU, offline original local 7B weights.

| Check | Result | Scope |
|---|---|---|
| Native serialization | PASS;13,476,831,232 weight bytes, two partitions | Existing weights, new required native representation only |
| Export duration | 357.651s including metadata copy and hash | NOT serving startup latency |
| Requests served | 0 | No inference/performance/correct-adapter claim |
| External supervision | 359 samples, service/watchdog exit0/0 | Full72/80GiB service envelope used |
| Process/GPU cleanup | Owned contexts cleared; service path gone; auxiliary empty then stopped | All GPUs back to15MiB |
| Loader-only environment modification | Exact-byte restore completed after worker exit | No active overlay remains |
| Elementwise source identity | PENDING | Must compare both parts and untied output head |

Exclusive native directory:
`models/vllm/tc-native-llama2-7b-fp16-20260927`.
Raw evidence: main `results/ieee_tc/serverless_qualification/d68_20260927`.
The official exporter also copies its source `.cache` (about5GiB), which is not
needed by native loading. Keep the original; remove only the new duplicate after
exact content/reference/open-handle checks and save a cleanup receipt. No broad
cache cleanup or unique data deletion is authorized by this observation.

The native format splits large checkpoints into numbered parts, with global
offsets. The existing comparator is extended to stream them in numeric order;
it rejects missing/extra parts and checks every byte, including independent
`lm_head.weight`.29 CPU router/launcher/partition tests pass. The completed3B
byte audit is not repeated.
