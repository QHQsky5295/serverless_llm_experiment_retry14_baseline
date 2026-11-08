"""D1130 post-failure duplicate-cache reclamation, not a successful run export.

Reuse D370 digest/reference/receipt primitives and D532 retained-original checks.
Only exact runner-claimed roots are removed after actual service/GPU release.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import ijson

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1130_20261108/7b_routing_handoff_w1')
OUTCOME = RAW/'launch.launch/physical_deployment/main_outcome.json'
ORIGINAL = Path('/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix')
INDEX = ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
CONFIG = ROOT/'configs/ieee_tc/20261108_d1130_7b_routing_handoff_w1.yaml'
BASE = Path(__file__).resolve().parent
TARGETS = [RAW.parent/'7b_nvme_routing_handoff_w1/tc_ieee_routing_handoff',
           Path('/dev/shm/tc-d1130-7b-routing-handoff-w1/tc_ieee_routing_handoff')]
sys.path.insert(0, str(ROOT/'scripts/ieee_tc'))
from cleanup_d370_aborted_cache import digest, active_reference, write_json


def item(prefix):
    with OUTCOME.open('rb') as stream:
        return next(ijson.items(stream, prefix, use_float=True))


def main():
    assert not (BASE/'cleanup_before.json').exists()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    launch = json.loads((RAW/'launch.json').read_text())
    assert launch['pass'] is False
    assert launch['classification'] == 'qualification_command_failure'
    assert launch['native_gpu_context_release_confirmed'] and launch['service_path_removed']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
    cleanup = item('cleanup')
    assert cleanup['complete'] is False and cleanup['reason'] == 'shutdown_unresolved'
    assert [Path(r['path']) for r in cleanup['roots']] == TARGETS
    pending = item('coordination_after_shutdown.ieee_file_pending_targets')
    assert not pending['plans'] and not pending['replacements']
    assert not pending['movement_interests'] and not pending['physical_resources_reserved']
    assert digest(CONFIG) == '517c2367ddd55b4d8221d37d36cb7ed08efb0475b836dd098510be7aa86e2572'
    assert digest(INDEX) == 'e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c'
    index = {a['id']: a for a in json.loads(INDEX.read_text())['artifacts']}
    inventories, checked_sources = [], {}
    for target, claim in zip(TARGETS, cleanup['roots']):
        assert target.is_dir() and not target.is_symlink() and target.resolve() == target
        st = target.stat()
        assert (st.st_dev, st.st_ino) == (claim['device'], claim['inode'])
        assert not any(p.is_symlink() for p in target.rglob('*'))
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            mount = line.split()[4]
            assert mount != str(target) and not mount.startswith(str(target)+'/')
        files, total = [], 0
        for adapter in target.iterdir():
            assert adapter.is_dir() and adapter.name in index
            expected = set()
            for row in index[adapter.name]['files']:
                file, source = adapter/row['path'], ORIGINAL/adapter.name/row['path']
                assert file.is_file() and source.is_file()
                retained = source.resolve(strict=True)
                assert not any(retained.is_relative_to(p.parent) for p in TARGETS)
                assert file.stat().st_size == source.stat().st_size == row['size_bytes']
                if source not in checked_sources:
                    checked_sources[source] = digest(source)
                assert digest(file) == checked_sources[source] == row['sha256']
                expected.add(file)
                files.append(dict(row, path=str(file.relative_to(target)),
                                  retained_source=str(retained)))
                total += row['size_bytes']
            assert {p for p in adapter.rglob('*') if p.is_file()} == expected
        assert not active_reference(target.parent)
        inventories.append(dict(claim=claim, files=files, payload_bytes=total))
    # Process/cwd/FD references: no global kills; permission gaps stay explicit.
    hits, unreadable = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for link in [proc/'cwd', proc/'root', *list((proc/'fd').iterdir())]:
                try:
                    value = os.readlink(link).removesuffix(' (deleted)')
                    if any(value == str(p.parent) or value.startswith(str(p.parent)+'/') for p in TARGETS):
                        hits.append(dict(pid=proc.name, link=str(link)))
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            unreadable.append(proc.name)
    assert not hits, hits
    references = []
    for repo in (ROOT, Path('/home/qhq/serverless_llm_baselines')):
        for target in TARGETS:
            result = subprocess.run(['git', 'grep', '-l', '-F', str(target.parent),
                                     '--', 'configs', 'scripts'], cwd=repo,
                                    capture_output=True, text=True)
            assert result.returncode in (0, 1), result.stderr
            references += [str(repo/name) for name in result.stdout.splitlines()]
    assert set(references) <= {str(CONFIG)}, references
    before = shutil.disk_usage('/home/qhq').free
    write_json(BASE/'cleanup_before.json', dict(kind='d1130_failed_run_cache_cleanup_v1',
        launch_sha256=digest(RAW/'launch.json'), outcome_sha256=digest(OUTCOME),
        config_sha256=digest(CONFIG), index_sha256=digest(INDEX), recipe_sha256=digest(Path(__file__)),
        inventories=inventories, retained_original=str(ORIGINAL), current_source_hashes_verified=True,
        references=references, readable_same_uid_process_hits=hits, unreadable_same_uid=unreadable,
        disk_before_bytes=before, recorded_unix=time.time(), failure_preserved=True))
    for target in TARGETS:
        shutil.rmtree(target)
        target.parent.rmdir()  # Empty private wrapper only; refuses other contents.
    after = shutil.disk_usage('/home/qhq').free
    write_json(BASE/'cleanup_after.json', dict(removed=[str(p) for p in TARGETS],
        disk_before_bytes=before, disk_after_bytes=after, observed_reclaimed_bytes=after-before,
        before_sha256=digest(BASE/'cleanup_before.json'), raw_results_removed=False,
        unique_artifacts_removed=False, recorded_unix=time.time()))
    print(json.dumps(dict(removed=True, disk_after_bytes=after, reclaimed_bytes=after-before)))


if __name__ == '__main__':
    main()
