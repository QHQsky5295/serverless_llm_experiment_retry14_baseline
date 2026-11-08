# D1127 targeted CPU checks: observed receipt

2026-11-08. Existing vLLM0.30 Python, repository cwd/PYTHONPATH, CUDA hidden,
PYTHONDONTWRITEBYTECODE=1, PYTHONNOUSERSITE=1. No GPU replay or serving edit.
Unit primelora-tc-d1127-prelaunch-tests-20261108.service; InvocationID
964823e6d06a4d798dabece104ab204c; CPU2,3,26,27; MemoryHigh3G/Max4G/SwapMax0.

Five unittest methods passed, all observed `ok`:

- tests.test_ieee_tc_launch.IEEEControlContract.test_current_7b_nohost_changes_only_independent_reuse_and_paths
- tests.test_ieee_tc_launch.IEEEControlContract.test_current_7b_nohost_profiles_bind_actual_child
- tests.test_ieee_tc_nohost_controller.NoHostInstallation.test_stack_configuration_preserves_budgets_and_native_reads_back_mode
- tests.test_ieee_tc_nohost.NoHostPlanning.test_no_host_targets_or_service_copies_and_no_budget_transfer
- tests.test_ieee_tc_nohost.NoHostNativeLifecycle.test_gpu_backing_survives_terminal_request_and_plan_close

Observed summary: Ran 5 tests in 20.559s, OK. Main exit0. systemd wrapper
exit1/result timeout, runtime2min8.347s, CPU2min19.233s. Journal at12:55:05
records control/memory inotify ENOSPC; at12:57:14 stop-sigterm timed out.
Subsequent MainPID0/ExecMainStatus0/empty ControlGroup and actual cgroup path
absent. No remaining test processes. This known management failure is distinct
from the passing assertions; no repeated tests or global system-limit changes.
This Markdown summarizes observed tool output, not an invented raw test log.

Separate source check primelora-tc-d1127-source-check-20261108.service completed
main/wrapper0, runtime1.765s/CPU1.764s under the same bounded CPU/memory setup.
All20 source SHA, FullD1115 raw SHA,144 old publication files,user dirty manifest,
main PLAN and byte-identical approved snapshot verified; see prelaunch.json.
Existing NoHOST 3B runtime evidence is reused; no redundant full qualification.
