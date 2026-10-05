# D236 — independent PEFT-reference result audit

**Date:** 2026-10-05  
**Plan SHA-256:** `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`  
**Branch:** `retry14_continuous_queue_v2`  
**Scope:** one guarded 7B numerical reference; no replay, remote service, baseline, or ranking run.

## Result identity

The existing `backend-peft-reference` action was run once through the guarded
TC launcher, using the already audited native five-arm receipt and the existing
frozen W0 trace. No weights, adapters, prompts, or workload trace were
generated or modified.

| Artifact | Path | SHA-256 |
|---|---|---|
| Reference result | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/model_20261005/peft_reference_d232_attempt2.json` | `0a9d2fa237961b79a0c282821e7161f9edd4a34bf98c7e78772aeb147566a8fe` |
| Launch receipt | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/model_20261005/peft_reference_d232_attempt2_launch.json` | `cd49bebad07d76a4de6270013d707cb5d2316d4a343b621defe2ab020a3d8d43` |

The result reports `kind=existing_peft_numeric_reference_v1`, `pass=true`,
`measurement_complete=true`, and `stage=complete`. It explicitly reports
`semantic_full_pool_qualification=false` and
`production_launch_authorized=false`.

## Checks that passed

- All five declared roles were evaluated: `nonzero_a`, `zero`, `nonzero_b`,
  `base`, and `nonzero_a_repeat`; the source native receipt was complete for
  all five controls.
- The reused W0 trace has SHA
  `efb903254fdcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c` and
  the reference used source request `req_00003`.
- Prompt and native prompt-ID hashes match the native receipt exactly.
- The checkpoint was loaded read-only from the existing local safetensors
  files. The reference recorded checkpoint content SHA
  `01ec976c8ae885754569dec8152de7166be71dfd02ce0223e4f1b3b6a3841fe7`.
- The three adapter files passed exact values/dtype checks for all 256 tensors
  per adapter.
- PEFT/native first-position comparisons passed for every role under the
  declared `atol=0.01`, `rtol=0.01` contract; the largest per-role comparison
  difference was below 0.030 and all violation lists were empty.
- Full-vocabulary effects distinguish the controls: the zero adapter has zero
  difference from base, while nonzero A and B have nonzero differences; the
  repeated A control is identical to A.
- The launch receipt has `pass=true`; watchdog evidence records zero service
  OOM/high/max events and a final empty service GPU context. No TC scope is
  active after cleanup.

## Interpretation boundary

This is a bounded semantic reference, not a correctness denominator for the
4,000-request workload. It does not establish `N_correct` for a full replay,
joint TTFT/TPOT SLO, physical GPU lifecycle budget, or superiority over any
baseline. The next authorized Prime step remains the pre-registered bounded
physical-admission diagnostic, followed by a full replay only if its numerical
and ownership evidence passes.

## Memory-gate provenance note

The current Plan SHA above defines the startup check as

\[
A_0 \ge 80+4+16+2=102\text{ GiB}
\]

when no service is running. Older status entries still describe a 109.5-GiB
gate. This D236 bounded reference was admitted using the current Plan's
explicit 102-GiB equation and the launcher-enforced 72/80-GiB service scope;
the older 109.5-GiB prose is not silently treated as equivalent. Before the
next performance or admission run, the two historical documents must be
reconciled and one threshold recorded in the run manifest.

