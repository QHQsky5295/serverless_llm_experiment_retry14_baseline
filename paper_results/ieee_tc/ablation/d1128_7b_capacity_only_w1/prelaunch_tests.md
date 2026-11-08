# D1128 targeted CPU check receipt

2026-11-08. Existing vLLM0.30 environment; CUDA hidden, CPU2,3,26,27,
MemoryHigh3G/Max4G/SwapMax0, PYTHONDONTWRITEBYTECODE=1, PYTHONNOUSERSITE=1.
Unit primelora-tc-d1128-prelaunch-tests-20261108.service; InvocationID
f11f2d9302184036b4da6786f9f95b29.

Observed six unittest methods all `ok`; Ran 6 tests in 0.608s, OK:

- tests.test_ieee_tc_launch.IEEEControlContract.test_current_7b_capacity_only_changes_only_soft_admission_and_paths
- tests.test_ieee_tc_transfer_pressure.ActivationPreparation.test_initial_and_natural_activation_forward_only_soft_admission_gate
- tests.test_ieee_tc_admission.AdmissionEquationContract.test_native_slot_and_workspace_checks_remain_in_capacity_only
- tests.test_ieee_tc_gpu_references.NativeProactiveTransactions.test_full_pressure_defers_without_eviction_or_lru_touch
- tests.test_ieee_tc_gpu_references.NativeProactiveTransactions.test_capacity_only_uses_same_native_victim_and_retains_reference
- tests.test_ieee_tc_gpu_references.NativeProactiveTransactions.test_all_pinned_capacity_cannot_be_bypassed_by_capacity_only

Main exit0; systemd wrapper exit1/result timeout, runtime1min44.109s,
CPU23.784s. Journal at14:01:26 records control/memory inotify ENOSPC;
at14:03:10 stop-sigterm timed out. Subsequent MainPID0/ExecMainStatus0 and
empty ControlGroup; actual test work finished. This management timeout is
distinct from the six passing assertions. No repeated tests/global-limit fix.
This is a summary of observed tool output, not a claimed complete raw stdout.

Separate source verification output is saved in prelaunch.json; it verifies
23 source hashes, FullD1115 raw SHA,144 protected publication files, current
user manifest and byte-identical PLAN snapshot, and unused result/cache paths.
Existing 3B CapacityOnly implementation evidence is reused only for the gate
and analysis method, not for current7B performance or profiles.

Source-check unit primelora-tc-d1128-source-check-20261108.service,
InvocationIDb29784f2cf2d4c1bb1da18fef6697e36: main0, wrapper1/timeout;
runtime1min44.297s, CPU24.507s. Passing source assertions preceded this same
management timeout. No repeated verification or global-system modification.
