# D298 — reuse the qualified 7B capacity contract in Full

2026-10-07. P2/P3 development; candidate preparation, not a completed Full run.
Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.
Read frozen V1 before selection. D297 remains the current completed Full.

## Evidence and falsifiable question

D297's mean user TTFT is1.527145s: arrival→admission0.817574s,
admission→engine0.421192s, engine-entry→queue0.080526s, native queue0.009212s,
prefill0.198641s. Its provisional joint timing upper bound is94.65%; this is
not numerical correctness or formal qualification. Four cards remain held
almost throughout;15860.686503GPU-s. Merely changing TTFTlower is unsupported:
the longest observed lowqueue/lowactive/no-pending interval is16.882s<38s.

D169/D170 already establish native batch8 on this SAME hardware/backend:
8sequences,8LoRA slots, actual pool maxrank16, GPU memory fraction0.92,
1022observed KV blocks. Four extreme batches passed, then1552native calls.
Those measurements are reusable hardware/length evidence, NOT Full performance.
D156/D157 Full instead uses4sequences/4slots/rank64/fraction0.70 and278blocks.

Hypothesis: the conservative runtime/cache capacity limits useful request
overlap and local-ready reuse; adopting the already qualified contract can
reduce the observed pre-engine wait without changing the paper's decisions.
Counter-risk: extra overlap increases TPOT or startup; resource release may
remain controlled by the existing policy. No claim that KV preemption currently
dominates, or that more capacity guarantees GPU-s savings.

## Official basis, checked2026-10-07

- [vLLM0.30 optimization](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/):
  sequence/iteration budgets and KV availability have latency trade-offs.
  Keep the1024iteration budget; do not import datacenter throughput defaults.
- [cache configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/cache.py):
 0.92 is the version's actual default, a per-instance allocation limit, not an
 independent guarantee of free memory or sharing. Physical whole-card billing
 remains unchanged.
- [LoRA configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py):
 rank and simultaneous LoRA capacity size backend buffers. Use the existing
 static pool's actual maxrank; weights and output work remain unchanged.

## Bounded candidate and reuse

Only this capacity package is under consideration: max_num_seqs/runtime cap8,
max_loras8, max_lora_rank16, memory fraction0.92. It is a configuration package,
NOT a single-factor algorithm ablation. Preserve TP1/FP16/prefix-off/context,
generation, nine equations, D88 controller thresholds/cooldown, preparation
window, remote delivery, movement limit, EWMA and external safety envelope.
Do not claim capacity alone solves resource reclamation.

Reuse D169/D170 instead of repeating their capacity/warm measurements. Native
completed lengths may initialize E(t) only after strict model/backend/trace/
prompt/native-count binding; extend the existing offline initializer to accept
the explicit warm-evidence format, never relabel it as a source-matrix run.
Reuse D170's unchanged compact projection, verified index and SHA chain; no
uncompressed raw duplication. Unknown numerical identity remains unknown.

Because service/preparation timing classes depend on the changed backend,
D157 timings CANNOT be relabelled. Reuse the existing native_source_matrix
collector and six static content-class representatives, adding the first two
distinct adapter/request pairs in the same existing development prefix to fill
eight lanes (no new input generation), with admitted bins[1,2,4,8]. Three rounds,
two eight-lane rotations per source/round, five sources:240measurements plus8
separately retained warmup calls. Refresh only affected profiles using real
published remote artifacts; verify coverage before any Full replay. No new
load/weights/framework. This qualification is not a system performance result.

After profile validation: one existing100-request Full smoke for contract and
safety, then the unchanged4000W0 replay. At most one follow-up minimal check if
new evidence requires it; do not keep repeating warm/reference/numerical gates.
Keep failures and regressions. Compare provisional SLO margins, stages, TPOT,
remote/local reuse and physical lifecycle with D297/D219. Same inputs are
development observations, not independent formal blocks or new workloads.

No figures until the user's performance milestone; no3B or external baselines
yet. Final shared warm/Resident and G1/G2 remain open. A changed backend is not
accepted as the final per-model configuration merely because it starts.

## Prepared evidence

Offline validation reused all1536D170 measurements and16warmups;512distinct
completed original requests initialize native-prompt buckets[759], with means
100.86411149825784/125.56888888888889 output tokens. Warmup/repeated rounds do
not multiply demand weight. No warm TTFT/TPOT values are imported into Full's
service profiles or changed for scoring. `N_correct` remainsunknown.

The original D169 model configuration is the base. The source-run YAML adds
only `safetensors_load_strategy=auto`, preserving D297's current startup policy;
the prepared child configuration already includes that field. New source-run
config identity and exact child-assembly check must be saved before launching.
First preparation failed on symlink-vs-resolved path lookup before outputs/GPU;
the same SHA lookup after resolving the path passed, failure receipt retained.
Twenty warm/index/length-initialization CPU tests passed; no serving algorithm
or nine-equation change. Freeze/push this checkpoint before source calibration.

Actual source YAML now verified against D169 plus only the auto loader; exact
child assembly matches the new contract and all500static adapter metadata have
rank≤16. Config SHA6b96df57e5f05a9b4a0b597ef988b464d37af6b61e4dac33b3e4444778d1487f.
Combined CPU suite:209/213 passed in the model-test Python; four OS-gate cases
failed because that interpreter lacks pidfd_send_signal. All50OS-gate tests
then passed in the actually used /usr/bin/python3. No safety fallback or change
to guard semantics. This test-environment mistake remains in cpu_tests.json.
