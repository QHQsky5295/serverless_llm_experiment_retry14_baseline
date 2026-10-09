# D376 7B capacity-revalidation diagnostic: preparation-profile mismatch

Date: 2026-10-10

D376 used the C=8 D321 service profile and entered the experiment runner, but
the preparation profile rejected the configuration before the first request.
The D376 config accidentally bound the D298 preparation profile with the D321
service input-contract context (`ccfb02...`) instead of the D298 preparation
context (`e844f2...`). The loader correctly reported
`preparation profile model/configuration/context mismatch`.

The replay publisher recorded `N_arrived=0` and `N_submitted=0`; the launcher
classified the run as `qualification_command_failure`. GPU-context release,
service cleanup and remote cleanup completed. D376 has no performance data and
is excluded from all comparison, ranking and figures.

This is a configuration-construction error in the diagnostic registration, not
a system or capacity result. D376 is retained as a failed provenance record;
the next retry uses a new identity with the original D298 preparation context
unchanged.
