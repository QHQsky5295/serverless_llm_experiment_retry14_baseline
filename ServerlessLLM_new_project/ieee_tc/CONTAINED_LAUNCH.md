# Serverless: independent native launcher view

Status: source adaptation and CPU checks only; no new Serverless model replay.
This is a prerequisite for the approved original/repaired 1,000-request pairs,
not evidence that the repair improves TTFT or that native loading is qualified.

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
