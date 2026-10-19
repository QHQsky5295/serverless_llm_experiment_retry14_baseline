"""Curate D697's bounded diagnostic analysis without copying request lists."""
import hashlib
import json
from pathlib import Path

RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d697_20261019')
OUT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/ieee_tc/p2_backend/d697_3b_full_w0')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

text = (RAW / 'analysis2.log').read_text()
analysis = json.loads(text[text.index('{'):])
for key in ('request_ids',):
    analysis.get('host_transaction', {}).pop(key, None)
    analysis.get('host_lease_transfer', {}).pop(key, None)
analysis.get('gpu_owner_dispatch', {}).pop('overlap_examples', None)
analysis['diagnostic_status'] = {
    'strict_analyzer_passed': False,
    'strict_failure': 'same-owner GPU action interval overlap',
    'strict_failure_log': str(RAW / 'analysis1.log'),
    'strict_failure_log_sha256': sha(RAW / 'analysis1.log'),
    'relaxed_analysis_only': True,
    'ranking_eligible': False,
    'n_correct_remains_unknown': True,
}
analysis['closure'] = json.loads((RAW / 'completion_receipt.json').read_text())
analysis['remote_stop'] = json.loads((RAW / 'remote_stop.json').read_text())
analysis['local_cleanup'] = json.loads((RAW / 'local_cleanup.json').read_text())
analysis['source_analysis_log_sha256'] = sha(RAW / 'analysis2.log')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'qualification.json').write_text(json.dumps(analysis, indent=2, sort_keys=True) + '\n')
(RAW / 'analysis_identity.json').write_text(json.dumps(analysis.get('scope', {}), indent=2) + '\n')
(RAW / 'analysis_receipt.json').write_text(json.dumps({
    'passed': True,
    'strict_analyzer_passed': False,
    'relaxed_diagnostic_returncode': 0,
    'analysis_log': str(RAW / 'analysis2.log'),
    'analysis_log_sha256': analysis['source_analysis_log_sha256'],
    'strict_failure_log': str(RAW / 'analysis1.log'),
    'strict_failure_log_sha256': analysis['diagnostic_status']['strict_failure_log_sha256'],
    'scope': analysis.get('scope'),
    'resources': 'scope was closed after analysis; no post-close process remained',
}, indent=2) + '\n')
print(json.dumps({
    'output': str(OUT / 'qualification.json'),
    'source_analysis_log_sha256': analysis['source_analysis_log_sha256'],
    'strict_failure_log_sha256': analysis['diagnostic_status']['strict_failure_log_sha256'],
    'same_owner_gpu_action_overlaps': analysis['gpu_owner_dispatch']['same_owner_gpu_action_overlaps'],
    'joint': analysis['common_threshold_joint_count'],
    'gpu_seconds': analysis['stats']['physical_gpu_seconds'],
}, sort_keys=True))
