# D1064 checkpoint attempt 1

Original checkpoint SHA-256:
`c0a15633720aa0a0ea2de0bc687dbeba1d2ff128bcf781f1f5666931af952723`.

The checkpoint stopped before creating snapshots, staging, committing or pushing.
The test command had exited 0 with 44 collected/passed methods. Its JUnit XML
has 44 concrete testcase elements and reports tests=61 in the aggregate header;
errors/failures/skipped are all 0. The checkpoint incorrectly asserted
`int(suite.attrib['tests']) == 44` and raised AssertionError at line 47.

The parser now checks the 44 concrete testcase elements and preserves the
aggregate header separately. Tests and analysis are not rerun or overwritten.
Original test XML and all analysis outputs remain unchanged. This is a backup
parser correction, not a performance failure or a relaxed correctness threshold.
