# Closure-ledger verification — 2026-11-08

No GPU execution, serving/configuration change, remote operation or raw rewrite.

- Collector initial CPU invocation used a relative path; systemd started in
  `/home/qhq`, so Python returned 2 before reading the script or creating output.
- The corrected absolute-path invocation was the only analysis execution:
  `primelora-tc-closure-ledger-20261108-r2.service`; main/wrapper exit 0,
  runtime 330 ms, CPU 99 ms. Explicit taskset 2,3,26,27, CUDA hidden,
  MemoryHigh=3G/MemoryMax=4G/MemorySwapMax=0. Subsequent state:
  inactive/dead, MainPID=0, Result=success, ExecMainStatus=0, empty ControlGroup.
- 64 distinct model/run points, 54 comparisons, 576 fixed-point sensitivity rows.
  Assertions include 30 external full points, two limited dLoRA points and six
  current PrimeLoRA resource points. Historical A1 and D1112 remain separate.
- Fresh verification of all 76 immediate input SHA and six generated data SHA
  passed. This is reuse of sealed raw audits, not a fresh raw-log integrity review.
- Collector parsed successfully; no serving-code smoke rerun was necessary for
  this offline projection. The actual collector and row-count assertions are
  the relevant execution check, not a GPU qualification.
- All 144 old publication files matched the existing protected manifest.
- User dirty manifest SHA remained
  `3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df`.
- Main PLAN SHA remained
  `b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`,
  byte-identical to the approved snapshot. No plan-binding migration repeated.
- Independent experiment-audit reviewer route/shared references unavailable;
  no independent PASS is asserted. This advisory absence does not block mainline.
- Explicit staging contains 13 files; protected/user paths are excluded and
  private-key/token/password assignment pattern checks found no match.
- Default Git whitespace check flagged standard Python CSV CRLF terminators.
  The unchanged CSV bytes and their sealed hashes were retained. A command-local
  check with `blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol` passed; no
  global Git setting, genuine whitespace check, or measurement was changed.

Git checks and actual remote HEAD are recorded after the explicit checkpoint;
this file does not pre-assert push success. User manifest, pre-existing dirty
EXECUTION_STATUS files, unrelated untracked files and historical figures are
excluded from staging. No secret values belong in this ledger.
