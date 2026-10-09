#!/usr/bin/env python3
"""Losslessly archive an explicit set of closed historical outcome duplicates."""
import compress_closed_main_outcomes as archive


TARGETS = [
    'd313_20261007/7b_full_w0',
    'd279_20261006/7b_readiness_8phase_full_retry10',
    'd257_20261006/7b_readiness_8phase_full_retry4',
    'd265_20261006/7b_readiness_8phase_full_retry7',
    'd256_20261006/7b_readiness_8phase_full_retry3',
    'd276_20261006/7b_readiness_8phase_full_retry8',
    'd210_20261004/7b_full_w0_full1',
    'd278_20261006/7b_readiness_8phase_full_retry9',
    'd195_20261003/7b_full_w0_full1',
    'd190_20261003/7b_full_w0_full1',
    'd185_20261003/7b_full_w0_full1',
    'd182_20261003/7b_full_w0_full1',
    'd175_20261002/7b_full_w0_full1',
    'd177_20261003/7b_full_w0_full1',
    'd179_20261003/7b_full_w0_full1',
    'd173_20261002/7b_full_w0_full1',
    'd328_20261008/7b_full_w0_prefix1',
    'd160_20261002/7b_full_w0_full1',
    'd154_20261002/7b_full_w0_full1',
    'd158_20261002/7b_full_w0_full1',
    'd152_20261002/7b_full_w0_full1',
    'd150_20261001/7b_full_w0_full1',
    'd148_20261001/7b_full_w0_full1',
    'd135_20261001/7b_full_w0_full1',
    'd137_20261001/7b_full_w0_full1',
    'd145_20261001/7b_full_w0_full1',
    'd131_20261001/7b_full_w0_full1',
    'd129_20260930/7b_full_w0_full1',
    'd125_20260930/7b_full_w0_full1',
    'd162_20261002/7b_full_w0_prefix1',
    'd119_20260930/7b_full_w0_full1',
    'd187_20261003/7b_full_w0_prefix1',
    'd269_20261006/7b_identity_diag_phase1_prefix_retry2',
]


if __name__ == '__main__':
    archive.OUT = archive.ROOT / 'paper_results/ieee_tc/storage/20261010_legacy_closed_outcomes_lossless_compression.json'
    archive.TARGETS = [dict(run=run, audit_sha=None) for run in TARGETS]
    if archive.OUT.exists():
        raise RuntimeError('refusing a second legacy archive run')
    raise SystemExit(archive.main())
