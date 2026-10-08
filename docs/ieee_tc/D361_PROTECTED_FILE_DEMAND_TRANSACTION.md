# D361: protected file input and live worker-local demand admission

2026-10-09. Candidate implementation/CPU qualification, not a measured speedup.
Mainline remains 7B. No new GPU performance run, remote change or figure.

## Why this candidate

D360 reuses D353/D358's complete W0 evidence: global-epoch rechecks rise from
593 to 1,609 while mean final native loading remains about42ms. An immutable
file read does not depend on every other adapter's reference count staying
unchanged. Its safety depends on that file remaining protected and verified,
and on checking the native worker's actual constraints at the time of loading.

Official[vLLM0.30 LRU worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
performs lookup/loading/capacity handling on its serialized core loop. The
candidate reuses that execution unit and existing Prime ownership rules. It
does not add a parallel CUDA loader, another cache, delay, magic threshold or
unchecked stale-command retry. D333/D352's existing atomic ownership design is
retained; D329's unsafe cross-request HOST observation reuse is not repeated.

## Exact opt-in and safety argument

`model.ieee_file_demand_transaction=protected_file_owner_turn_v1` (absent by
default). Only the file-demand branch attaches an explicit immutable-input
witness. This covers materialized remote or managed NVMe/HOST files, not native
HOST tensors or selected executable GPU copies.

1. The local file owner verifies its actual live lease under its lock, checks
   the confirmed publication/signatures and content SHA, and attests the exact
   logical adapter, canonical path, device/inode and lease/owner identities.
2. The serialized native owner rejects malformed/content-mismatched witnesses,
   wrong native owners, future epochs, changed directory incarnations and
   attempts to mix file permission with selected GPU/HOST copy semantics.
3. A file witness permits older global epochs because it reuses no native
   capacity verdict. Current native content binding, cache membership, pins,
   unpinned victims, live HOST budget, loader and completion fence still run in
   that one native owner turn. Live capacity failure remains a no-load conflict.
4. Receipt replay binds the same file witness to the same native lease. A lost
   reply/cancellation keeps file/native/request ownership unresolved until the
   existing reconciliation path settles it; no early source deletion or permit
   return is introduced. Successful replay need not reopen a file already
   released after the original acknowledgement.
5. The controller checks the new transaction receipt before releasing its file
   reference. `observed_epoch`, `validated_epoch`, and `owner_epoch_advance`
   measure use of this path; the latter is not labelled exclusively unrelated
   work, since same-adapter progress can occur too.

This is a cooperative in-process/service ownership contract, not an untrusted
client authentication boundary. File content verification remains at publication;
the worker does not rehash all weights on every request. It checks the directory
incarnation while the real cooperative read lease protects managed mutation.

No change to paper equations, routing order, confirmed dispatch tier, proactive
Eq.(8), demand priority, HOST tensor budgets, native cached-copy guards, actual
KV ownership, capacity wait events, autoscaler or generation. Optional core
capacity-witness forwarding accepts the same content-bound field and still
claims observation only, never synthetic physical KV reservation.

## Tests and limits

- First targeted run:71tests,3failures, preserved. All three came from a newly
  written fixture assuming adapter1 was present although the fixture starts
  empty. Corrected the test to perform a real unrelated load/release; production
  safety rules were not weakened to make it pass.
- Next broader run:417passed. Final suite adds cancellation, native GPU/HOST
  unchanged-path checks and optional scheduler forwarding:443passed in the
  installed native-vLLM environment and443in the existing CPU test environment.
- Checks cover stale unrelated progress, malformed/cross-content/cross-owner
  witnesses, future state, changed inode, actual file-owner lease validation,
  pinned capacity, live HOST rejection, replay without rereading released input,
  uncertain/cancelled acknowledgements and preserved selected tiers.
- These use real control methods with tiny cache/file fixtures, not model/CUDA
  performance. Numerical correctness and formal G1/G2 qualification remain open.
- Memory protection was active for every test; receipts retain each scope's
  peak/events. No performance inference is made from test runtime.

## Next mainline task: qualify the changed execution identity before Full

The service/preparation profile loaders bind the complete runtime configuration.
This demand-path change affects ready-state preparation/control waiting; it is
not a startup-only exception. Do not remove the field from profile identity or
relabell D356 timing samples as measurements of this candidate. This avoids
repeating D355's already-understood profile mismatch at a GPU launch.

Reuse the existing D356/D344 source-measurement design, existing artifacts,
traces, collectors and exporter to refresh the affected measured initialization.
Reuse measured output lengths only under the existing explicit length-transfer
contract; do not transfer timing silently. Verify actual class coverage and
transaction receipts, preserve failures, and only fill measured missing classes.
No new weight/trace pool or capacity/autoscaler parameter scan is authorized.

Then same first100 qualification → ordinary Full4000 W0. Compare retained
D358/D353/D336 using frozen V1 and provisional D170 timing thresholds, with
full TTFT/TPOT/GPU-s and all output changes retained. Report both actual file
witness transactions and rechecks; no full-run conclusion from the short smoke.
The comparison includes a measured initialization refresh and must disclose it.
No performance victory or7Bclosure is asserted here.

Raw tests: `results/ieee_tc/p2_backend_qualification/d361_20261009/`.
Checkpoint: `paper_results/ieee_tc/p2_backend/d361_file_demand_transaction/`.
3B, paused baselines, M1/M2, ablations and sensitivities remain pending in order.
