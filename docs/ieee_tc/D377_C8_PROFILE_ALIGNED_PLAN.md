# D377 7B capacity-revalidation diagnostic

D377 is the corrected fresh-identity retry of D376. It uses the already
qualified C=8 D321 service profile and D298 preparation profile with each
profile's original input-contract context preserved. The current source-
identity code and the unchanged controller envelope remain in use; the two
later opt-in policy fields are absent because they are not part of these C=8
measurements.

The trace, adapter subset, fixed-length generation contract, remote cache and
resource gates are unchanged. D377 is a 100-request seed-42 development
diagnostic only, not a formal G1/G2 result or figure source. The run must be
excluded if any profile check, request-completeness check, or cleanup gate
fails.
