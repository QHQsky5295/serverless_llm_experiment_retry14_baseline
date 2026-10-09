#!/usr/bin/env python3
"""One fixed D364 archive using the existing guarded compressor."""
import compress_closed_main_outcomes as archive


if __name__ == '__main__':
    archive.OUT = archive.ROOT / 'paper_results/ieee_tc/storage/20261010_d364_closed_outcome_lossless_compression.json'
    archive.TARGETS = [dict(run='d364_20261009/7b_full_w0', audit_sha=None)]
    raw = archive.RAW_ROOT / archive.TARGETS[0]['run']
    if (raw / 'compression_receipt.json').exists():
        raise RuntimeError('refusing to replace an existing D364 archive receipt')
    raise SystemExit(archive.main())
