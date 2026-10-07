# D313 initial offline audit — provenance manifest correction required

This is retained analysis history, not a passing all-correct workload or final
publication manifest. All 16 input source hashes were checked; the context and
failed-request outputs retain their actual recorded hashes. The producer opened
`manifest.json` before enumerating output files, thereby including the manifest's
temporary empty-file hash as its own output hash. That self-entry is invalid.

The original manifest is preserved unchanged. A corrected `failure_audit2`
publication will exclude manifest self-hashing and revalidate the same original
execution, without any GPU rerun, request replacement, or changed measurement.
