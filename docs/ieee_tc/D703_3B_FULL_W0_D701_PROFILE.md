# D703：3B full W0 with the D701 native profile

## Registration

D703 is a new execution key for one complete 3B W0 replay. It restores the
D693 production parent (`preloading_manager.py` SHA
`825ead7f23686b8f501dd8d833cfa27111cc55fbbfda1191bed6a25b1f48654f`) and
changes no request trace, adapter subset, model, generation contract, scaling
threshold, or backend version. Its only measurement-facing profile change from
D682/D697 is replacing the incomplete D645 profile with the strictly exported
D701 same-source-spec profile:

* service profile SHA `c2586bc32df1d9346d7439c2b70f3dac5a8de4e32ed0c7f27ccaaaf8481fb2b2`;
* preparation profile SHA `e56fec47f21d030dcd20a647f604ee1193ae27cb454f5387c767f0cd6f3a6dff`.

The raw result is isolated at
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d703_20261019/3b_full_w0`.
The launcher refuses an existing path and uses the frozen 3B W0 trace,
500-adapter subset, `fixed_length_greedy_v1`, prepublished gzip delivery, and
the existing native vLLM 0.30.0 toolchain.

## Scientific question

D703 tests whether the prior 3B full-run qualification gap was caused by the
incomplete service/preparation profile rather than by a new runtime mechanism.
It is therefore a profile-correction run, not a parameter search and not a
repeat of D682 or D697. The acceptance record must distinguish:

1. completion and native generation-contract evidence;
2. numerical adapter correctness (not inferred from token-contract success);
3. joint SLO attainment and lifecycle physical GPU-s/request;
4. whether the D701 profile is actually used without fallback or interpolation.

Only a complete, contract-valid run can be considered for later G1/G2 analysis.
A W0 result alone does not close W1/W2 or establish a baseline comparison.
If the run is below the frozen 95% joint-SLO reference, it remains a closed
diagnostic candidate and no same-key rerun or figure is allowed.

## Required closure

Before launch: disk/memory/swap/GPU-idle gates, remote artifact health, and
prepublished-delivery verification must pass. After launch: stop the remote
service, release all local GPU/host/NVMe allocations, validate every request,
verify no request-time packing, and retain failed evidence. The strict audit
must record the source/config/runtime SHA chain, profile fallback count, native
token source, adapter/content/lease bindings, dispatch fields, physical
allocation journal, SLO counts, and watchdog state.

No paper figure is generated for D703 before the strict audit and candidate
decision. Existing D645, D682, D695, D697, and D702 evidence remains
immutable.
