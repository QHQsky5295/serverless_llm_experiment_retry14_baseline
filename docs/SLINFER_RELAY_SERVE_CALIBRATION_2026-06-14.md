# SLINFER RelayServe-Workload Calibration Closure

Date: 2026-06-14

SLINFER was calibrated on the frozen, disjoint 512-request traces using the
official `sota` GPU-only scheduler. The hardware adaptation is limited to the
available 4x RTX 3090 testbed: two workers per GPU for 3B, one worker per GPU
for 7B, a 20 GiB host store pool, and a 32 GiB minimum available-memory guard.

## Frozen outcome

| Model | Candidates | Zero-failure candidates | Selection | Formal status |
|---|---|---:|---|---|
| Llama-3.2 3B | keep-alive 1/10/30/60 s | 1 | keep-alive 1 s | eligible |
| Llama-2 7B | keep-alive 1/10/30/60 s | 0 | none | blocked |

The 3B keep-alive 1 s candidate completed 512/512 requests. The other 3B
candidates completed 493/512, 467/512, and 460/512.

The 7B candidates completed 509/512, 507/512, 508/512, and 508/512. All
failures were native SLINFER `deadline_violation` results concentrated in a
late burst region. No candidate therefore satisfies the preregistered
zero-failure calibration gate.

## Safety

No new host OOM, GPU Xid, or NVRM allocation event occurred during the eight
calibration runs. The minimum-memory guard was not triggered. The earlier
unsafe upstream topology is not used by these results.

## Evidence

- 3B table:
  `paper_artifacts/relayserve_v4/slinfer_3b_calibration512.csv`
- 7B table:
  `paper_artifacts/relayserve_v4/slinfer_7b_calibration512.csv`
- Frozen policy:
  `configs/relayserve_slinfer_calibration_policy_v1.yaml`
- Raw run root:
  `results/relayserve_continuation/slinfer/`

Each table records absolute raw-record, source-summary, and manifest paths
with SHA-256 hashes. Failed requests remain in the raw evidence and in the
reported completion counts.
