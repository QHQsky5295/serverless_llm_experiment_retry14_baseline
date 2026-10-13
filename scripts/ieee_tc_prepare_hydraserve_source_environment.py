#!/usr/bin/env python3
"""Necessary native-source prerequisites; never installs into an old environment.

Reuse D482's bounded pip procedure and the common preflight. Reuse only verified
Python-independent NVIDIA wheels from the old environment; never expose its
Python 3.11 packages to Python 3.10. No model, kernel or performance qualification.
"""
import argparse
import base64
import csv
import email
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path('/home/qhq/serverless_llm_baselines')
NATIVE = BASE / 'vendor_new_baselines/HydraServe_main_20260520'
OLD = Path('/home/qhq/anaconda3/envs/slinfer_official_20260612')
PYTHON = '/home/qhq/anaconda3/envs/hydraserve_py_20260520/bin/python'
spec = importlib.util.spec_from_file_location('primelora_tc_preflight', MAIN / 'scripts/ieee_tc_preflight.py')
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


def cuda_distribution_manifest(site, expected):
    """Check actual bytes against installed RECORD; reject Python ABI libraries."""
    records = {}
    for name, version in expected.items():
        normalized = name.replace('-', '_')
        info = site / (normalized + '-' + version + '.dist-info')
        wheel = (info / 'WHEEL').read_text()
        tags = [x.removeprefix('Tag: ') for x in wheel.splitlines() if x.startswith('Tag: ')]
        if not tags or not all(x.startswith('py3-none-') for x in tags):
            raise ValueError('not a Python-independent NVIDIA distribution: ' + name)
        metadata = email.message_from_string((info / 'METADATA').read_text())
        if metadata['Version'] != version or metadata['Name'].replace('_', '-') != name:
            raise ValueError('distribution identity mismatch: ' + name)
        members = {}
        with (info / 'RECORD').open(newline='') as handle:
            for relative, encoded, size in csv.reader(handle):
                path = site / relative
                if not path.resolve().is_relative_to(site.resolve()):
                    raise ValueError('RECORD leaves source site: ' + relative)
                if 'cpython-' in path.name:
                    if path.suffix == '.pyc':
                        continue
                    raise ValueError('Python-specific native library: ' + relative)
                if not encoded:
                    if path.suffix != '.pyc' and path.name != 'RECORD':
                        raise ValueError('unhashed source member: ' + relative)
                    continue
                kind, encoded_hash = encoded.split('=', 1)
                if kind != 'sha256' or path.stat().st_size != int(size):
                    raise ValueError('invalid member identity: ' + relative)
                actual = p.digest(path)
                decoded = base64.urlsafe_b64decode(encoded_hash + '=' * (-len(encoded_hash) % 4)).hex()
                if actual != decoded:
                    raise ValueError('source RECORD hash mismatch: ' + relative)
                members[relative] = {'sha256': actual, 'bytes': int(size)}
        records[name] = {'version': version, 'metadata': str(info), 'tags': tags,
                         'members': members, 'record_sha256': p.digest(info / 'RECORD')}
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    args = parser.parse_args()
    root, target = args.output.resolve(), args.environment.absolute()
    root.mkdir(parents=True, exist_ok=True)

    def save(name, data):
        with (root / name).open('x') as handle:
            json.dump(data, handle, indent=2)

    identity = p.scope_identity(p.cg_path().name)
    save('environment_identity.json', identity)
    limits = p.cgroup_snapshot(Path(identity['path']))
    assert all(limits[k] == v for k, v in {
        'memory.high': 3*p.GIB, 'memory.max': 4*p.GIB, 'memory.swap.max': 0}.items())
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert target.parent == BASE / 'installs' and not target.exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=NATIVE)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=NATIVE, text=True).strip() == '8ae605de354ccfa2e9095514cdb4a9e9c56aa56b'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    gate = p.preflight([root, target.parent], predicted_growth=12*p.GIB)
    save('environment_preflight.json', gate)
    assert gate['preflight_pass'], gate['issues']
    prior = root.parent / 'd482_20261013'
    audit = json.loads((prior / 'source_build_audit.json').read_text())
    for relative, record in audit['source_manifest'].items():
        assert p.digest(NATIVE / relative) == record['sha256']
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               MAX_JOBS='2', NVCC_THREADS='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    started, steps = time.monotonic(), []
    result = {'passed': False, 'performance_result': False, 'model_started': False,
              'kernel_compiled': False, 'environment': str(target), 'steps': steps}

    def run(label, command):
        proc = None
        try:
            with (root / (label + '.log')).open('x') as log:
                proc = subprocess.Popen(command, env=env, cwd=root, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                owner = p.gpu_process_identity(proc.pid)
                assert owner and Path(owner['cgroup']).is_relative_to(Path(identity['path']))
                save(label + '_worker.json', owner)
                while proc.poll() is None:
                    assert p.host_sample()['available_bytes'] >= 16*p.GIB
                    assert shutil.disk_usage(root).free >= 100*p.GIB
                    if time.monotonic() - started > 1800:
                        raise TimeoutError('environment_preparation_wall_guard')
                    time.sleep(1)
                steps.append({'label': label, 'command': command, 'returncode': proc.returncode})
                assert proc.returncode == 0, label
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)

    try:
        site = OLD / 'lib/python3.11/site-packages'
        expected = {name: record['version'] for name, record in audit['installed_packages'].items()
                    if name.startswith('nvidia-') and name != 'nvidia-ml-py'}
        cuda = cuda_distribution_manifest(site, expected)
        save('cuda_reuse_manifest.json', cuda)
        run('venv', [PYTHON, '-B', '-m', 'venv', str(target)])
        target_site = target / 'lib/python3.10/site-packages'
        # Only the Python-independent NVIDIA namespace is linked. Metadata is
        # privately copied so normal metadata discovery is complete. Never pip
        # uninstall/upgrade these links; container deployment mounts OLD readonly.
        (target_site / 'nvidia').symlink_to(site / 'nvidia', target_is_directory=True)
        for record in cuda.values():
            info = Path(record['metadata'])
            shutil.copytree(info, target_site / info.name)
        wheels = []
        for name in ('torch', 'xformers'):
            record, = audit['public_metadata'][name]['wheels']
            path = prior / 'build_wheels' / record['filename']
            assert path.stat().st_size == record['bytes'] and p.digest(path) == record['sha256']
            wheels.append(str(path))
        python = str(target / 'bin/python')
        pip = [python, '-B', '-m', 'pip', '--isolated', 'install', '--no-cache-dir',
               '--index-url=https://pypi.org/simple', '--timeout=30', '--retries=2']
        run('exact_wheels', [*pip, '--no-deps', '--report', str(root / 'exact_wheels_report.json'), *wheels])
        # Common official requirements; CMake3 preserves compatibility with the
        # old Torch CMake policies. Ray2.9.3 is within the official >=2.9 range.
        run('dependencies', [*pip, '--only-binary=torch,xformers,triton,ray,numpy,scipy,pyarrow', '--report', str(root / 'dependencies_report.json'),
            '-r', str(NATIVE / 'requirements-common.txt'),
            '-r', str(NATIVE / 'requirements-build.txt'),
            'cmake<4', 'ray==2.9.3', 'triton==2.3.0', 'nvidia-ml-py',
            'modelscope==1.15.0', 'datasets==2.18.0', 'flask[async]', 'accelerate', 'hf_transfer'])
        run('pip_check', [python, '-B', '-m', 'pip', '--isolated', 'check'])
        run('import_check', [python, '-B', '-c',
            "import torch,xformers,triton,transformers; import json; "
            "assert torch.__version__=='2.3.0+cu121'; "
            "assert xformers.__version__=='0.0.26.post1'; "
            "assert triton.__version__=='2.3.0'; "
            "assert transformers.__version__=='4.43.1'; "
            "assert not torch.cuda.is_initialized(); "
            "print(json.dumps(dict(torch=torch.__version__,xformers=xformers.__version__,"
            "triton=triton.__version__,transformers=transformers.__version__,"
            "cuda=torch.version.cuda,abi=torch._C._GLIBCXX_USE_CXX11_ABI,cuda_initialized=False)))"])
        assert cuda_distribution_manifest(site, expected) == cuda
        run('freeze', [python, '-B', '-m', 'pip', '--isolated', 'freeze'])
        result['passed'] = True
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        result.update(elapsed_s=time.monotonic()-started, free_bytes=shutil.disk_usage(root).free,
                      scope_after=p.cgroup_snapshot(Path(identity['path'])))
        save('environment_result.json', result)
        print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
