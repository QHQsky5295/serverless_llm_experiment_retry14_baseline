# D362 — refresh affected initialization measurements for D361

2026-10-09. Mainline remains **7B optimization → 3B → paused baselines →
M1/M2 → ablations/sensitivities**. No figures, paper edits or new assets.

## Why this measurement is necessary

D360 recovered D353/D358 loading receipts. The final actual loading duration
was almost unchanged, whereas waiting before acquisition and stale owner-epoch
rechecks increased. D361 introduced one opt-in candidate:
`ieee_file_demand_transaction=protected_file_owner_turn_v1`. It binds a held,
content-verified file source to an atomic native validation/load/acquire turn.
It does not reuse old capacity, weaken actual source ownership or alter the
nine IEEE equations. Native GPU/HOST sources retain their original guards.

The candidate passed 443 CPU contract tests in each of the native and historical
environments, but has **no GPU performance evidence yet**. Its new field changes
the exact runtime identity. Reusing old timing profiles under the new identity
would not be a valid qualification shortcut.

The implementation rationale remains aligned with the pinned official
[vLLM 0.30.0 worker implementation](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py):
LoRA validation, cache-capacity handling and activation occur in the serialized
native owner. This is a source-level comparison, not evidence of a measured
speedup. D362 does not introduce a second serving candidate.

## Exact reuse and changed evidence

- Existing D356 source specification SHA:
  `c4fdbed1ef75f60a424cdd5d74d5f7a140896a23bd523e955f48eb1c66e880e4`.
- Existing D356 supplementary specification SHA:
  `ae7e630b4e21bfa61b875e90c0ef6252912b04b34bdbca2d67858c10ae0d403e`.
- Reuse their exact cases and ordering: **135 waves, 1,669 calls**, including
  34 retained warmup calls and 1,635 representative calls. These are calibration
  observations, not 1,669 independent workload repetitions.
- Reuse original first-1,000 trace indices, static content identities and the
  already published common gzip delivery cache. No new weights or traces.
- C14, iteration-token budget, control interleave, HOST limits, allocator and
  D358 autoscaling rules stay unchanged. The only candidate model-field delta
  is the explicit file-demand policy; new initialization IDs follow that delta.
- A narrowly named length-transfer contract permits the declared policy and
  existing capacity/control differences from the old completed-length source.
  It transfers only observed fixed-output lengths. All other input, generation,
  backend and configuration checks remain. Natural-stop lengths, old latency,
  numerical correctness and SLO qualification are not transferred.
- Service/preparation runtime loaders keep their exact identity checks. Fresh
  affected timing must be measured; no policy field is removed from identity.

## Existing execution and validation path

Extend `prepare_d298_capacity.py`, `run_d298_capacity_source.sh` and
`export_d298_profiles.py`; use the existing native source collector and ordinary
ScenarioRunner source owner. Preparation passed **536 CPU tests** in the native
environment. This is functional evidence only.

Before measurement: verify pushed source checkpoint, protected files, GPU idle,
current memory/disk gates, remote frozen services and actual NIC negotiation.
Use existing service/auxiliary containment, watchdog and tmux. No simultaneous
build, compression, hashing or second GPU campaign. Save raw observations under
`results/ieee_tc/p2_backend_qualification/d362_20261009/`; no overwrite.

After measurement: actual GPU release → exact-owned remote/service cleanup →
source/byte/journal checks → all-case native token and timing checks → fresh
class coverage. Audit every file-demand witness against the held file reference,
immutable content, request intent and native acknowledgement; retain all owner
epoch deltas, including zero. GPU/native-HOST cases must not acquire this policy.
Audit control-interleave observations as before. No numerical-output or speedup
claim follows from these receipts alone.

If actual class coverage is missing, retain the run and inspect only the missing
classes before a bounded complementary measurement; never fill with old timing,
zeros or relabelled samples. Export only complete, matched initialization data.

## Next performance question

Once profiles qualify, first100 smoke, then complete W0 4,000-request replay
against retained D358/D353/D336. Compare all-request timing, phase-2 tails,
admission/acquisition waits, actual load spans, rechecks, TPOT and physical GPU-s.
Do not accept a TTFT improvement at the expense of hidden failures or resource
growth. D170 thresholds remain provisional; numerical `N_correct`, final common
warm reference and full Resident budget qualification remain open. Do not claim
7B closure, formal G1/G2 success or a baseline victory from calibration.

Plan SHA: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.
Frozen V1 SHA: `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`.
