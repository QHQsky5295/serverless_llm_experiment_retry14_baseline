# D993 — common-SLO and physical-resource outcome audit

Subsequent D994 provenance correction: commit-time CRLF-to-LF formatting changed
the saved stage CSV bytes without updating its sealed reference. The old CSV
therefore no longer passes that byte seal; it is retained unchanged. D994 rebuilt
the stage analysis from raw requests, verified all 4,000 parsed rows identical,
and reproduced every count below. See D994_3B_CURRENT_REMOTE_CRITICAL_PATH_20261031.md
for both hashes and the new valid source-bound analysis. This is a formatting
provenance defect, not evidence of changed measurements; do not bypass the seal.

2026-10-31. Reuses retained measurements; no new performance replay or figure.
The same D992 run completed 4,000/4,000 requests, zero failures, and actual
GPU/service release. The previously running process/session is closed.

## Frozen-rule results

| Model / scenario / run | Joint TTFT/TPOT | Type-1 P95 TTFT (ms) | Physical GPU-s | GPU-s/offered request |
|---|---:|---:|---:|---:|
| 7B W0 D983 | 3,794/4,000 = 94.85% | 3,429.479 | 9,720.172815 | 2.430043 |
| 7B W0 D988 | 3,996/4,000 = 99.90% | 1,571.805 | 15,211.234817 | 3.802809 |
| 7B W0 D989 | 3,961/4,000 = 99.025% | 1,766.285 | 9,745.848822 | 2.436462 |
| 3B W2 D972 | 3,650/4,000 = 91.25% | 2,316.050 | 11,310.910082 | 2.827728 |
| 3B W2 D992 | 3,678/4,000 = 91.95% | 2,197.860 | 8,268.639918 | 2.067160 |

Thresholds use the existing D394 7B and D648 3B manifests and their actual
native-prompt groups (616/506 boundaries). Each group of compared runs has
identical reconstructed request/adapter/arrival/prompt/target identity hashes.
All 4,000 requests in each run pass the retained native binding checks;
physical intervals are released, disjoint per device and consistent with the
mutually exclusive accounting windows. Stage identity error is zero for D992.
Original `n_correct=null`, development identity and absence of independent
holdout remain unchanged; the table explicitly uses the offered denominator.
No CI or statistical superiority claim is made.

## Observation and decision

1. **7B:** D989 clears 95% joint timing and uses 35.93% less physical GPU time
   than D988, with higher P95 TTFT. Relative to D983 it uses 0.264% more GPU
   time but raises joint timing from below to above the fixed requirement.
   Retain this W0 point; do not rerun its key. W1/W2 remain separate required
   coverage for the frozen controller.
2. **3B:** D992 reduces physical GPU time by 26.90% and P95 TTFT by 5.10%
   relative to D972, but **neither run clears 95% joint timing**. D992 is not a
   feasible G1/G2 result. It is 122 passing requests short of 3,800; do not
   lower thresholds or use the legacy 5-second percentage.
3. D992 has 168 TTFT-only misses, 132 TPOT-only misses and 22 both-metric misses.
   Its marginal counts are 3,810 TTFT passes and 3,846 TPOT passes. Both
   marginals exceed 95%, but their intersection does not. This explains why
   judging either marginal alone would incorrectly close the optimization.
4. D992 records four scale-ups (three ordinary, one forecast) and three
   forecast capacity releases. D972 records two ordinary scale-ups and no
   release. Mechanism execution is demonstrated; the aggregate comparison
   alone does not isolate all causal effects or compiler/cache variation.

## Avoid a repeated qualification detour

D393/D637 already reject full cross-backend exact token equality as a new
V1 condition. D991's 94/100 exact PEFT witness remains visible, not relabeled
as 100%, but does not by itself hold all future performance work. Separate
the native execution contract, independent arithmetic proof, formal identity
and timing feasibility. Here the actionable 3B failure is directly measured
joint timing, independent of that numerical-proof limitation.

## Completed retained-stage diagnosis

The existing D703 common-SLO audit independently reproduced all counts and
the W2 input view. The existing D716 stage analysis then processed all 4,000
requests, with zero timing-identity error; no new script or GPU replay.

- 128/168 TTFT-only misses selected the remote tier. Their mean TTFT is
  2,715.689 ms and admission-to-acquisition span is 1,894.040 ms. Across the
  whole TTFT-only group, 118/168 have acquisition as their largest TTFT stage.
- 128/132 TPOT-only misses have 1–64 output tokens; four have 65–128.
  Seventy already selected a GPU-resident adapter, so their own cache miss
  cannot explain the entire decode problem. Mean TPOT is 44.226 ms.
- 99/132 TPOT-only misses overlap another request's prefill; 77 overlap a
  demand load. These are same-owner observed span associations, not kernel
  timing or proof that either activity caused every miss. Passing requests
  also overlap work; keep the full population and conditional denominators.

Saved evidence: `d993_3b_control_audit/d992_joint_requests.json` and
`d993_3b_control_audit/d992_stage_diagnosis/{diagnosis.json,requests.csv}`.

## Next main-line action

Use the completed retained-stage diagnosis to inspect remote acquisition and
short-decode interference against prior accepted/rejected changes and primary
implementation sources before proposing one new 3B candidate; no new performance
run until a concrete mechanism hypothesis is established. Do not stack a new
controller patch merely because it uses fewer GPUs. Preserve D989 and finish
the same-config scenario comparison after the current model's timing gap is
resolved. The existing D593/D810 baseline matrices are retained, including
nonfeasible results; shared k3s/basic HydraServe adaptation is not reopened.

Core A1–A5, S1–S13, complete G1/G2 comparison, reviewer documentation and final
publication figures remain in scope. No figure before the target is approached.

## Evidence and implementation

- `paper_results/ieee_tc/p2_backend/d993_3b_control_audit/summary.json`
- `paper_results/ieee_tc/p2_backend/d993_7b_control_audit/summary.json`
- Existing `scripts/ieee_tc/analyze_d455_stages.py` extended with explicit
  model/reference and successful-launch selection; D989 uses `launch_retry5`.
- Seven focused CPU tests passed in the existing vLLM0300 environment;
  no serving code, environment or frozen protocol changed during D992.
- Both analysis commands exited successfully in auxiliary 3/4-GiB scopes,
  CUDA hidden, after D992's service exited. Source/protocol/analyzer SHA values
  are in the summaries. No historical result is overwritten.
- The separate existing common-SLO and stage commands also completed with
  matching counts. The protected-artifact check confirms 147/147 old items
  unchanged; the user-generated manifest and source PLAN SHA are unchanged.
- D992 cleanup restored free disk to 173,914,898,432 bytes (about 162 GiB)
  at the observed post-run check. No extra manual deletion was needed then.

Plan SHA: `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`.
