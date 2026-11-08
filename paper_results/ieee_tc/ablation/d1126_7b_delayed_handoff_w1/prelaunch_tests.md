# D1126 targeted CPU tests — observed receipt

2026-11-08; this is a summary of observed tool output, not a reconstructed raw log.
No GPU replay or production code change.

Command used existing vLLM0.30 Python with CUDA_VISIBLE_DEVICES empty,
PYTHONDONTWRITEBYTECODE=1, PYTHONNOUSERSITE=1, repository PYTHONPATH and cwd.
systemd unit run-u393.service, InvocationID 6d48b091ed7046a1b33a283306a44dc1;
CPU 2,3,26,27; MemoryHigh3GiB/MemoryMax4GiB/MemorySwapMax0.

Four unittest methods, all `ok`:

- tests.test_ieee_tc_launch.IEEEControlContract.test_current_7b_delayed_changes_only_handoff_start_and_paths
- tests.test_ieee_tc_launch.IEEEControlContract.test_delayed_runner_keeps_full_routing_hierarchy_and_admission
- tests.test_ieee_tc_transfer_pressure.ActivationPreparation.test_delayed_keeps_frozen_epoch_but_does_no_copy_before_initialized
- tests.test_ieee_tc_transfer_pressure.ActivationPreparation.test_full_file_copy_precedes_runtime_ready_and_adoption_has_no_budget_gap

Observed unittest summary: `Ran 4 tests in 0.283s`, `OK`.
Observed wrapper: `Finished with result: timeout`, main exited/status0,
runtime1min43.953s, CPU23.096s; wrapper exit1.
Journal: 11:14:47 failed control/memory inotify watch with ENOSPC;
11:16:31 stop-sigterm timed out; unit result timeout.
Subsequent readback: MainPID0, ExecMainStatus0, failed/failed, ControlGroup empty.
Actual /sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/run-u393.service
does not exist. No leaked test group, no repeated test or global-limit change.
This management failure is retained separately from the four passing assertions.
