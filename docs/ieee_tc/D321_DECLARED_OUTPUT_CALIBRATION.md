# D321 — declared-output service-class calibration (development)

## Question and fixed candidate rule

D320 found opposite signed errors for short and long outputs under the single
2..256 output class. Test one refinement of the paper's configured observation
classes; do not change D+T+O, EWMA, routing ordering, admission, lifecycle control,
generation, the nine equations, or frozen evaluation V1.

Use Type-1 quartiles of **declared** output limits from the complete historical
D315 development run. Merge duplicate edges and append the fixed cap256. This
rule uses no latency, future realized natural output, or held-out winner selection.
The same model-specific class definition will be frozen before any candidate run.

## Evidence reuse and missing-data rule

The existing D298 exporter now reconstructs historical classes from native
admission/source/reference/token events, validates old admission classes, and
retains each request's measured D/T/O intervals. Original D298 source samples
are retained at their actual declared limits. Neither source may be relabelled
as an unobserved short-output measurement. Complete native identity/timing does
not establish numerical adapter qualification or formal N_correct.

Required coverage is the original qualified non-output class domain crossed
with every reachable new output class. Export missing classes explicitly; no
pooled, nearest, zero or guessed value. History export is **not launchable**.
Only actually missing measurements may be added using the existing collector.

## Initial checks and pending result

| Item | Observation |
|---|---|
| New historical export/rebin unit tests | 4 pass |
| Combined source/export/routing tests | 69 pass,0.505s |
| CPU test memory peak | 495972352B; memory events/swap0 |
| Fresh GPU or remote run | None |
| Full historical export and coverage | 4000history+240source;141/160classes |

Test scope8b2ff8a22a654806960e2125726a2409 was actually empty/populated0
before its owned-scope stop. Preserve all old results and user-dirty manifest.

Interpretation: class refinement is a falsifiable calibration hypothesis, not
yet a demonstrated latency/resource improvement. First export exact historical
samples/coverage; then measure only gaps, validate one candidate and complete
W0. No 3B, baseline, repeated cohort analysis or figures before that mainline.

## Relevant primary sources

[Llumnix load observation](https://docs.llumnix.ai/design/scheduler/instant_accurate_load)
distinguishes timely load observation from accurate work estimation. Its newer
model-specific examples are not transplanted into this dense7B model.
[Llumnix OSDI24](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)
discusses request heterogeneity and unknown natural output lengths. Here the
declared limit is available at admission; no claim that natural output is known.

The IEEE configured class already includes declared output, prompt, rank,
footprint, representation and admitted count. This is initialization/bin quality,
not a new prediction formula or a change to the service-quality ruler.

## Historical export completed; narrow gap measurement prepared

Actual historical edges are[24,91,217,256].141of160classes are measured;
17missing classes are file-HOST and2remote. No GPU/NVMe/native-HOST gaps.
`history1` retains all4000historical samples and exact coverage, not a usable
partial runtime profile. Canonical admission prompt length, not post-special-token
native length, defines the old/new prompt class; both lengths are retained.
The production native-event observer verifies first/last timestamps. First
callback may carry multiple already-timed tokens; count1 is not required by that
contract. Earlier strict-export failures are retained in history1/2/3 raw logs,
with no successful output overwritten or request dropped. History4 completed.

`gaps1/source_spec.json` indexes only original first1000requests:13existing
adapters,7necessary source/length/rank groups,3rounds,81measurements plus5retained
warmups (22waves). Each group's size is the lowest admitted count reaching its
largest missing admitted bin. No barrier forces an artificial count; actual
observed coverage must pass afterward. Rotation is predetermined, not selected
by latency. Reuse D298 exact configuration, initialized admission model and
collector; no Full policy or model change. This is calibration, not an ablation.

213CPUtests passed50.423s (8newD321tests); peak2480730112B,events/swap0.
History4peak1245851648B; preparation625909760B; actualempty/populated0 checked
before stopping each owned scope. Usermanifest,sourceplan,V1,serving unchanged.
Before GPU work: verify all SHA chains/protected data, normal tested checkpoint,
fresh resource gate and frozen published remote service. Collection pending.
