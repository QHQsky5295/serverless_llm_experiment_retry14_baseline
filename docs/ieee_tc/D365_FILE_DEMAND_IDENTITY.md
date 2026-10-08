# D365: file-demand identity is not an allocation snapshot

2026-10-09. Correctness repair on the existing 7B candidate; no new GPU
performance result, profile retuning, figure or baseline run.

## Failure retained and hypothesis tested

D364 completed all 4,000 terminal requests, with 3,999 native-contract returns
and one failed request (`req_03511`, `education_lora_0127`). Its original source
SHA is `b90a4f6874d0f5f8a9c0a1ad8ddf749430e11c46b32c35c9eaa71c3edb6762aa`.
The error came from `LocalSourceReferences.demand_witness` before native demand
loading: `file demand source content is not confirmed`. D364 remains failed and
outside feasible G1/G2 rankings. Its compared operands were not recorded; the
specific historical filesystem event is not established by this repair.

D310 (`4ea7daa`) already separated bounded physical allocation updates from
immutable content during confirmed read-pin acquisition. D361 (`5f3053a`) added
the protected file-demand path, but compared the entire copied publication record
again. That record includes mutable `allocated_file_bytes` and `allocated_bytes`.
`_validated_source` explicitly refreshes those observations and advances the
planning epoch while preserving the verified content and charged inode envelope.
Thus a valid live lease could fail its next demand check without content mutation.

Before changing serving code, the existing D310 tiny-publication fixture was
reused. Only its reported `st_blocks` varied within the real precharged envelope;
path, inode, timestamps and bytes stayed unchanged. Four cases (increasing or
decreasing blocks, observed inside the demand check or by an intervening reader)
all reproduced the same exception. The separate over-envelope case was rejected
as required. This is a deterministic protocol defect, not evidence identifying
the exact kernel transition in D364. No production artifact or trace was created.

## Repair and unchanged invariants

Demand validation now compares the complete publication schema and every field
except the two physical allocation observations. The retained observations must
still be nonnegative integers. The live `_validated_source` call still verifies
all path membership, content signatures and allocation envelopes before this
comparison. Logical byte count, file count, representation, content SHA, adapter,
path and tier remain equal. The actual lease still binds owner, adapter, path,
tier, device and inode; released or forged leases fail.

Physical accounting is not frozen or discarded: live inventory is refreshed,
the planning epoch advances, and the original reserved envelope remains charged.
No file/native ownership is released on error. True publication failures now
record both publication observations and the identity-comparison result, so a
future failure need not rely on one generic error without operands.

This introduces no catch/retry, fallback, sleep/fsync, capacity increase,
autoscaler change, new parameter or change to the nine IEEE equations. Native
HOST/GPU selected-copy guards, worker-local live capacity checks, cancellation
and acknowledgement reconciliation are unchanged.

## External implementation check and limits

Linux [stat structure documentation](https://man7.org/linux/man-pages/man3/stat.3type.html)
distinguishes logical file size and allocated 512-byte blocks. The kernel's
[ext4 allocation documentation](https://docs.kernel.org/filesystems/ext4/allocators.html)
describes allocation/writeback behavior; neither source proves which event
occurred in D364. The relevant principle is separating content identity from
the explicitly bounded physical allocation observation already supported here.

The frozen [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
validates adapter loading before LRU replacement and executes that path in its
core owner loop. It supports keeping content validation and current capacity
handling distinct; it does not provide Prime's file-owner lease protocol or
establish a performance benefit from this correction.

## Validation and next mainline action

Raw before/after tests, source snapshots and resource receipts are in
`results/ieee_tc/p2_backend_qualification/d365_20261009/`. The first two-test run
retains all four failing subcases. The next 77 targeted checks passed. Broader
regression passed 662 checks in each existing environment (historical analysis
Python and native vLLM 0.30 Python). Peak auxiliary memory was 647,557,120 bytes;
high/max/OOM/swap events were zero. Both source snapshots and exact-owned cleanup
receipts are sealed in the accompanying checkpoint. These are control/file
correctness checks, not GPU performance or numerical-model validation.

After checks and source backup, reuse the unchanged D363/D364 model configuration
and D362 initialization measurements for a native 100-request check, then a fresh
complete 4,000-request W0 replay. This restores the existing immutable-source
contract rather than changing the profile configuration or adding a control law;
do not regenerate the measured profiles or workload without new evidence. Verify
the existing strict model/profile binding before launch. All new source commits
and paths remain explicit; old runs are not relabelled.

Retain the D363 prefix regression and D364 failure. Full comparison still uses
D358/D353/D336 under frozen V1, with D170 timing thresholds labelled provisional.
Numerical correctness, final common warm/Resident references and formal G1/G2
remain open. Continue 7B, then 3B, then paused baselines; no figures yet.
