# EuroSys'27 V2 formal configuration gate

`scripts/run_full_fair_round.sh` writes a machine-readable configuration
sidecar before it starts any serving system:

```text
<round>/protocol/system_resolved_config.json
```

The sidecar has two different hashes. They must not be conflated.

- `system_resolved_config_sha256` identifies the frozen PrimeLoRA, S-LoRA, and
  ServerlessLLM-new system tuning/runtime configuration. It covers the resolved
  model, runtime topology and serving caps, Prime mechanism/scenario settings,
  autoscaling/admission settings, relevant environment overrides, wrapper
  source identities, and upstream system revisions. Because the checked-out
  S-LoRA and ServerlessLLM trees carry compatibility patches, their identity
  also includes the canonical tracked binary-diff SHA, dirty-path list, and
  final SHA/size of every modified tracked file. A commit id alone is never
  treated as sufficient provenance for those runtimes.
- `full_run_identity_sha256` identifies one concrete run. It additionally
  covers trace and adapter-subset hashes, sampling seed, request count, run tag,
  execution order, trace time-scale, bandwidth, and C4 workload axes.

The system-configuration hash deliberately excludes sampling/generation seed,
trace and subset identity, run tag, execution order, request count, aggregate
bandwidth, trace time-scale, and C4 Zipf/active-cap/rotation/overlap controls.
These are workload
or experimental axes, not permission to retune a system. Consequently, the
system hash must remain constant across seeds 43--45 and across A6/C4
sensitivity points for the same model/scenario/generation contract, while the
full-run identity changes at each point.

## Trace-role gate

For every `v2_*` scenario, roles and request counts are fail-closed:

| Seed | `trace_role` | Request gate |
|---:|---|---:|
| 41 | `validation` | exactly 1,000 |
| 42 | `smoke` | 1--100 |
| 43, 44, 45 | `heldout` | exactly 4,000 |

Any other V2 seed, a mismatched explicit `FAIR_TRACE_ROLE`, or a wrong request
count is rejected before GPU launch. Legacy `faaslora_full` rounds retain the
`legacy` role for backward compatibility.

The default registry is:

```text
<FAIR_ROUND_ROOT>/_protocol/system_resolved_config_registry.json
```

Seed 41 may register multiple validation candidates. A held-out configuration
is accepted only if its hash was already observed in a seed-41 validation
record for the same family. The first accepted held-out run atomically freezes
that selected validation hash. Later held-out seeds and sensitivity points in
that family must match. Therefore neither a new formal-only tuning choice nor a
switch between two validation candidates after seeing seed 43 can pass.
The family includes the model and Prime scenario, but not bandwidth or C4
workload axes. File locking and atomic replacement prevent concurrent launchers
from establishing different frozen hashes.

## Formal source gate

Set `FAIR_FORMAL_RUN=1` for paper-eligible GPU runs. The runner checks tracked
files before shared-artifact preparation and again before final manifest
publication:

- the baseline repository must have no tracked changes;
- the FaaSLoRA repository may have only
  `configs/generated/lora_manifest_1000.json` as a tracked change;
- untracked files, including baseline `cache/`, are intentionally ignored.

Formal execution fails closed if these conditions are not met. The round
manifest records `formal_run`, `trace_role`, `source_clean_for_formal`,
`system_resolved_config_sha256`, and `full_run_identity_sha256`.

The fair runner propagates the resolved-config audit to PrimeLoRA with:

- `FAASLORA_SYSTEM_RESOLVED_CONFIG_SHA256`
- `FAASLORA_TRACE_ROLE`
- `FAASLORA_FORMAL_RUN`

PrimeLoRA result metadata must record these values verbatim. The runner also
passes the generic `FAIR_SYSTEM_RESOLVED_CONFIG_SHA256`, `FAIR_TRACE_ROLE`, and
`FAIR_FORMAL_RUN` variables to S-LoRA and ServerlessLLM-new wrappers.

## S-LoRA fixed-length stream integrity

The C5 fixed-output path includes a local compatibility correction in S-LoRA's
HTTP manager: detokenizer updates are retained in a per-request FIFO until the
SSE generator consumes them. The upstream single-slot mailbox can overwrite
unread token events when detokenization runs ahead of the HTTP consumer. The
correction does not change generation length, scheduling, batching, paging, or
adapter placement; it only prevents loss of already-produced native token IDs.
The exact patched file is covered by `upstream_worktree_identity`, and
`tests/test_fixed_length_generation_contract.py` contains a burst regression
that fails under the single-slot behavior.
