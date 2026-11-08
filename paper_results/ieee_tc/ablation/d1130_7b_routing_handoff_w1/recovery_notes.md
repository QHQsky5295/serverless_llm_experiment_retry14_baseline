# D1130 recovery and lifecycle compatibility

Nov8: the replay completed4000 requests; cleanup failed on missing Python
os.pidfd_open. launch remains pass=false; all original bytes are retained.
Streaming jq inspection completed in15.015s with512MiB/1GiB containment,
no large reconstructed JSON. Saved scenario has4000 completed/0failed,
coordination owner has no pending plans/replacements/movement interests.
This inspection alone is not final frozen-SLO or physical-resource validation.

Cache reclaim attempt1 (unit primelora-tc-d1130-cache-reclaim-20261108) exited1
before any removal or before-receipt: original finance_lora/config.json and
generation_config.json are existing symlinks in the retained frozen source.
Current private cache files are regular. The retry resolves original sources,
verifies they are outside both removal trees, and hashes the actual retained
contents. It does not delete original links/targets or relax current-cache link
checks. This is a correction to the cleanup assumption, not a performance run.

Lifecycle change reuses the D27 x86-64 pidfd_open binding in a lightweight module,
adds pidfd_send_signal against the same kernel ABI, and retains ownership checks.
CPU-helper waits use poll as already established for physical workers by D541;
high descriptor numbers do not become an artificial cleanup failure. No routing,
generation, SLO, control, budget or nine-equation change.

Focused checks: deployment Python `-S` ran24 tests;23 passed and the high-FD
fixture failed because its inherited soft descriptor limit was1024. The fixture
now temporarily raises only its own soft limit within the existing hard limit;
the four real-process cases passed0.265s. Full CPU lifecycle/integration/physical
regression ran92 tests;91 passed and an old high-FD fixture hit the same harness
limit before cleanup. Its owned domain is gone; the single affected test passed
0.105s with explicit per-test-unit LimitNOFILE8192. No serving resource policy
was altered. Recovery-reader identity/population test passed0.305s. These are
CPU lifecycle/analysis checks, not performance or numerical model qualification.

Recovery attempt1 passed4000 native request/timing/input checks and closed-GPU
integration, then stopped before output on a mistyped63-character journal SHA.
Remote and local current SHA both equal
`dca30d217864e9f5fb8081f660b8547d9a3d0be2e3b815cdb0451ba3b27876ec`.
Only the transcription was corrected; neither journal was changed. No inference
rerun. All attempt failures remain documented; the original launch stays failed.

Primary sources checked: Linux6.8 syscall_64.tbl and installed
asm/unistd_64.h define424/434; [pidfd_send_signal](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)
addresses the pinned process and propagates kernel errors. Historical CPU path
introduced by4e8cee27 and integrated68aef590; D541 documents the prior select
descriptor-limit defect. Qualified deployment Python lacks both public wrappers.
