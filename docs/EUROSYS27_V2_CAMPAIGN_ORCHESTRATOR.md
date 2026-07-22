# EuroSys'27 V2 campaign orchestrator

`scripts/run_eurosys27_v2_campaign.py` is the fail-closed outer scheduler for
the large V2 matrix. It keeps the experiment semantics in the existing
`run_full_fair_round.sh`; it only decides which frozen round runs next and
preserves its provenance.

The example configuration is
`configs/eurosys27_v2_campaign.example.json`. Its machine-readable contract is
`configs/eurosys27_v2_campaign.schema.json`. Copy the example to a new file and
give every independent campaign a new absolute `campaign_root`. Do not run the
example in place after changing an initialized campaign: the orchestrator
rejects config drift by design.

## Safety and state model

- `plan` and `status` are read-only. They do not create the campaign root, run
  hooks, prepare traces, or invoke a serving runner. In particular, `plan`
  never approximates execution with `FAIR_ROUND_DRY_RUN=1`.
- Every executable run has a canonical `run_key`, derived from the runner,
  working directory, explicit resolved-config registry, declared environment,
  and manifest contract. A logical `reuse_of` entry has the same key as its
  source and never starts a second GPU run.
- The first mutating command freezes the canonical config, selected safe base
  environment, and Git identities under `<campaign_root>/protocol/`. All later
  mutations verify those exact files and current source identities.
- Every launch or retry receives a unique `attempt_id`, `SLLM_RUN_TAG`, and
  `FAIR_ROUND_DIR`. `resume` alone re-enters the same attempt directory so the
  existing runner can honor its stage markers.
- `campaign_ledger.jsonl` is append-only, sequence-numbered, locked with
  `flock`, and `fsync`ed after each event. A separate non-blocking execution
  lock prevents two launchers from selecting the same pending run.
- A nonzero runner result, missing/incomplete manifest, failed hook, or failed
  post-run GPU-idle check records `attempt_failed`. `run-next` then stops until
  that run is explicitly resumed or retried. Failed attempt directories and
  logs are never deleted.
- The orchestrator forces `FAIR_ROUND_FORCE=0`,
  `FAIR_ROUND_DRY_RUN=0`, and `PAPER_QUEUE_DRY_RUN=0`. It injects the explicit
  registry path and refuses config attempts to override orchestrator-owned
  paths, tags, or attempt identities.

The orchestrator records only an allowlisted set of inherited environment
variables. A listed variable that is unset is simply absent when the campaign
is initialized. Credential-like names are rejected so frozen protocol files do
not become a secret store. Put experimental controls directly in each run's
`env`; do not rely on the launching shell.

## Commands

First inspect the exact plan. This is safe even when the campaign root does not
exist:

```bash
python3 scripts/run_eurosys27_v2_campaign.py \
  --config configs/my_eurosys27_v2_campaign.json plan
```

Use `--json` on `plan` or `status` for automation. To execute one round at a
time (normally inside the existing experiment tmux workflow):

```bash
python3 scripts/run_eurosys27_v2_campaign.py \
  --config configs/my_eurosys27_v2_campaign.json run-next

python3 scripts/run_eurosys27_v2_campaign.py \
  --config configs/my_eurosys27_v2_campaign.json status
```

`run-next` follows the dependency DAG and skips `reuse_of` aliases. An optional
`--run-id ID` requests one already-runnable pending entry without bypassing its
dependencies. It executes exactly one run and returns; a later invocation moves
to the next one.

If a run failed after completing some runner stages, resume the same immutable
attempt:

```bash
python3 scripts/run_eurosys27_v2_campaign.py \
  --config configs/my_eurosys27_v2_campaign.json resume \
  --run-id c5_7b_heldout_seed43
```

If the attempt itself must remain untouched, create a new attempt instead:

```bash
python3 scripts/run_eurosys27_v2_campaign.py \
  --config configs/my_eurosys27_v2_campaign.json retry \
  --run-id c5_7b_heldout_seed43
```

`resume` appends a new invocation log inside the same attempt and reuses its
byte-validated frozen environment. `retry` gets a new directory and tag. Neither
operation converts an incomplete result into a completed result without a
fresh `MANIFEST.json` validation and post-cleanup check.

## Dependencies and reuse

`depends_on` is an execution dependency. All listed runs must be complete
before the dependent run becomes runnable. It is appropriate for held-out
rounds that require the seed-41 validation registry evidence.

`reuse_of` creates a logical alias for one already declared run. It cannot have
a runner, registry, or environment of its own and therefore cannot silently
rerun or retune the experiment. It is useful when one physical result feeds
several plots or supplementary analyses. Reuse chains and dependency cycles
are rejected during read-only planning.

Each executable run names a registry key, and every key maps to an absolute
path in top-level `registries`. There is no implicit production registry. Use a
different named path for non-formal seed-41 tuning candidates; only the selected
formal validation and its held-out dependents should share the formal registry.

## Cleanup hooks

`cleanup.pre_commands` and `cleanup.post_commands` are argv arrays, not shell
strings. They run with the same frozen environment as the experiment. When
`strict_gpu_idle=true`, the orchestrator queries every configured GPU before
launch and after all post hooks. The existing fair runner still performs its
own per-system cleanup; these outer checks verify the boundary between rounds.

The post hooks and idle verifier run even after runner failure. If a machine
needs an extra local cleanup command, add the smallest explicit argv command to
the campaign config. Do not put a destructive, broad `pkill` shell expression
in the config.

## Analyzer input exports

An `analyzer_exports` entry names logical run IDs and relative glob selectors.
After all selected runs are complete, generate curated input lists with:

```bash
python3 scripts/run_eurosys27_v2_campaign.py \
  --config configs/my_eurosys27_v2_campaign.json export-inputs \
  --export c5_7b_heldout
```

The command checks each ledger-backed completed manifest, resolves reuse aliases
to their single physical attempt, rejects missing or ambiguous selectors, and
writes `inputs.json` plus one `<selector>.txt` list. Each export invocation uses
a new timestamped subdirectory; older analyzer input lists are not overwritten.
Its paths and SHA are appended to the campaign ledger. Omit `--export` to
generate every configured export.

## Formal handoff checklist

1. Commit all source and protocol changes before initializing a formal
   campaign. The baseline source must be tracked-clean; the example permits only
   FaaSLoRA's existing generated manifest exception.
2. Run `plan --json` and archive the output with the campaign notes.
3. Confirm seed 41 uses 1,000 validation requests and seed 43/44/45 use 4,000
   held-out requests. Keep smoke seed 42 outside result selection.
4. Confirm each run declares the correct two-system order, generation contract,
   C4 workload axes, fixed configuration family, and explicit registry.
5. Start one `run-next` in tmux. Never start a second campaign executor while it
   is active.
6. On failure, inspect the preserved attempt log and manifest before choosing
   `resume` (same attempt) or `retry` (new attempt). Do not edit state markers.
7. Export analyzer inputs only after `status` reports the required physical and
   reused runs as complete.
