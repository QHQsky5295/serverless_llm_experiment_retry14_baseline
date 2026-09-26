#!/usr/bin/env python3
"""Materialize an owned TC view of the existing native launch scripts.

This is a source adapter, not a second service supervisor. The existing TC
gated launcher/watchdog owns process-tree teardown. `prepare` starts nothing;
the generated stack refuses launch outside that admitted service domain.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
SOURCE_SHA = {
    "start_serverlessllm_stack.sh": "9ded784fd6c1990dc9c0994d3ec7744b799cfbff830554ff22b85b8f9ddbb95d",
    "run_serverlessllm_head.sh": "8bb2cbb3e257085183f3cd586e86f0073770d98289f515850c7a84bb488c93ac",
    "run_serverlessllm_worker.sh": "d55ddb27716683e9866a5bd72c06e9f0302b35f5f776872b37fc63da0c720f09",
    "run_serverlessllm_serve.sh": "4b3a232f587f1c00986469f59ab7641b1e69835c45115b70b557e450545a27ec",
    "run_serverlessllm_store.sh": "8f7d23d2023d0e856dd2398265891417c3a8c475af904ac7d096c51fa8d2e07d",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("native launcher anchor differs; inspect before adapting")
    return source.replace(old, new, 1)


def render(sources: dict[str, str], *, script_dir: Path, private_root: Path,
           main_repo: Path, gpu_ids: tuple[int, ...], ray_only: bool = False) -> tuple[dict[str, str], dict]:
    if (not gpu_ids or len(set(gpu_ids)) != len(gpu_ids)
            or any(type(i) is not int or i not in range(4) for i in gpu_ids)):
        raise ValueError("explicit unique local GPU IDs from 0..3 required")
    if set(sources) != set(SOURCE_SHA):
        raise ValueError("incomplete native source set")
    for name, source in sources.items():
        if sha(source.encode()) != SOURCE_SHA[name]:
            raise ValueError(f"native source drift: {name}")
    # Only the established single-host, one worker-raylet deployment is adapted.
    # The head has its own object store too: the qualification total is 8 GiB,
    # NOT 8 GiB per Ray node. This equal partition is not a selected M1 optimum.
    allocation = {"head": 4 * GIB, "worker_0": 4 * GIB}
    q = shlex.quote
    for path in (script_dir, private_root, main_repo):
        if not path.is_absolute() or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
            raise ValueError("native shell adapter requires explicit safe absolute paths")
    if len(str(private_root / "tmux.sock").encode()) >= 100:
        raise ValueError("private socket path is too long")
    helper = Path(__file__).resolve()
    guard = f'''
# TC ownership is verified before any native launch side effect.
/usr/bin/python3 {q(str(helper))} verify --manifest {q(str(script_dir / 'launch_manifest.json'))}
unset TMUX TMUX_PANE
export RAY_TMPDIR={q(str(private_root / 'ray_tmp'))}
export SLLM_TC_RAY_TEMP={q(str(private_root / 'ray_tmp' / 'ray'))}
export SLLM_TC_SPILL={q(str(script_dir / 'spill'))}
export SLLM_TC_TMUX_SOCKET={q(str(private_root / 'tmux.sock'))}
export SLLM_SINGLE_HOST_MULTI_GPU=1
export SLLM_WORKER_GPUS={q(','.join(map(str, gpu_ids)))}
export SLLM_SERVE_LOG_PATH={q(str(script_dir / 'serve.log'))}
export SLLM_STORE_LOG={q(str(script_dir / 'store.log'))}
# The fixed native scripts use env prefixes; require explicit environment/source
# identity instead of accidentally selecting the old default 2.48 environment.
: "${{SLLM_HEAD_ENV_PREFIX:?explicit native head environment required}}"
: "${{SLLM_WORKER_ENV_PREFIX:?explicit native worker environment required}}"
: "${{SLLM_STORE_ENV_PREFIX:?explicit native store environment required}}"
: "${{SLLM_REPO_ROOT:?explicit audited native source required}}"
: "${{SLLM_RAY_HEAD_HOST:?explicit cluster host required}}"
: "${{SLLM_RAY_PORT:?explicit private cluster port required}}"
: "${{SLLM_PORT:?explicit service port required}}"
: "${{SLLM_STORE_PATH:?explicit existing checkpoint namespace required}}"
[[ -d "${{SLLM_STORE_PATH}}" ]] || exit 1
[[ "${{SLLM_DIRECT_PATH_MODE:-0}}" == 0 ]] || {{ echo "TC native loader may not fall back to direct path" >&2; exit 1; }}
[[ ! -e "${{SLLM_SERVE_LOG_PATH}}" && ! -e "${{SLLM_STORE_LOG}}" ]] || exit 1
tmux() {{ command tmux -f /dev/null -S "${{SLLM_TC_TMUX_SOCKET}}" "$@"; }}
[[ ! -e "${{SLLM_TC_TMUX_SOCKET}}" ]] || {{ echo "private socket already exists" >&2; exit 1; }}
'''
    result = dict(sources)
    stack = result["start_serverlessllm_stack.sh"]
    stack = replace_once(stack, "set -euo pipefail\n", "set -euo pipefail\n" + guard)
    stack = replace_once(stack, 'SCRIPTS_DIR="${ROOT_DIR}/scripts"',
                         f'SCRIPTS_DIR={q(str(script_dir))}')
    begin = stack.index('bash "${SCRIPTS_DIR}/sync_serverlessllm_runtime_sources.sh"')
    end = stack.index('\ntmux new-session -d -s "${HEAD_SESSION}"', begin)
    stack = stack[:begin] + "# No installed-source mutation or pre-existing process cleanup.\n" + stack[end:]
    stack = replace_once(stack, 'CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh',
                         'CUDA_VISIBLE_DEVICES=${WORKER_GPUS} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh')
    stack = replace_once(stack, 'store  : ${STORE_SESSION} (gpu=${GPU_LIST[0]})',
                         'store  : ${STORE_SESSION} (gpus=${WORKER_GPUS})')
    stack = replace_once(stack, 'rm -f "${SERVE_LOG_PATH}"', '# New log only; checked before launch.')
    stack = replace_once(stack, '"bash -lc \'env${VLLM_ENV_PREFIX}', '"bash -c \'env${VLLM_ENV_PREFIX}')
    if ray_only:
        # Explicit infrastructure qualification, not a direct-path fallback.
        # Execute the identical guarded head/worker prefix; do not start store,
        # API, load a model or submit inference requests in this mode.
        marker = 'wait_for_workers "${EXPECTED_WORKERS}"\n'
        if stack.count(marker) != 1:
            raise ValueError("native worker-start boundary differs")
        stack = stack[:stack.index(marker) + len(marker)] + '\necho "TC ray-only infrastructure ready"\n'
    result["start_serverlessllm_stack.sh"] = stack
    for role in ("head", "worker"):
        name = f"run_serverlessllm_{role}.sh"
        source = result[name]
        source = replace_once(source, 'RAY_OBJECT_STORE_MEMORY_BYTES="${SLLM_RAY_OBJECT_STORE_MEMORY_BYTES:-}"',
                              f'RAY_OBJECT_STORE_MEMORY_BYTES={allocation["head" if role == "head" else "worker_0"]}')
        extra = ['  --object-spilling-directory="${SLLM_TC_SPILL}/' + role + '"']
        if role == "head":
            extra.append('  --temp-dir="${SLLM_TC_RAY_TEMP}"')
        source = replace_once(source, "  --block\n)", "  --block\n" + "\n".join(extra) + "\n)")
        result[name] = source
    name = "run_serverlessllm_serve.sh"
    result[name] = replace_once(result[name], '-m sllm.cli.clic start --host "${HOST}" --port "${PORT}"',
                               '-m sllm.cli.clic start --host "${HOST}" --port "${PORT}" --enable-storage-aware')
    return result, allocation


def prepare(output: Path, private_root: Path, main_repo: Path, gpu_ids: tuple[int, ...],
            *, ray_only: bool = False) -> dict:
    for path in (output, private_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refuse to reuse or overwrite {path}")
    sources = {name: (ROOT / "scripts" / name).read_text() for name in SOURCE_SHA}
    rendered, allocation = render(sources, script_dir=output, private_root=private_root,
                                 main_repo=main_repo, gpu_ids=gpu_ids, ray_only=ray_only)
    manifest = {
        "schema": "ieee_tc_serverless_native_launcher_view_v1", "display_name": "Serverless",
        "main_repo": str(main_repo), "private_root": str(private_root),
        "script_dir": str(output), "source_sha256": SOURCE_SHA,
        "rendered_sha256": {name: sha(data.encode()) for name, data in rendered.items()},
        "helper_sha256": sha(Path(__file__).read_bytes()),
        "object_store_bytes_by_raylet": allocation,
        "object_store_bytes_total": sum(allocation.values()), "worker_gpu_ids": list(gpu_ids),
        "ray_only_infrastructure": ray_only,
        "qualification_only": True, "actual_workers_verified": False,
        "native_loader_qualified": False, "performance_run_authorized": False,
        "cleanup_owner": "existing external TC gated launcher and watchdog",
    }
    output.mkdir(mode=0o700)
    private_root.mkdir(mode=0o700)
    for sub in (private_root / "ray_tmp", output / "spill", output / "spill" / "head", output / "spill" / "worker"):
        sub.mkdir(mode=0o700)
    for name, data in rendered.items():
        with (output / name).open("x") as handle:
            handle.write(data)
    with (output / "launch_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def load_guard(main_repo: Path):
    checker = main_repo / "scripts" / "ieee_tc_preflight.py"
    spec = importlib.util.spec_from_file_location("tc_native_launch_guard", checker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent.resolve(strict=True)
    if manifest.get("schema") != "ieee_tc_serverless_native_launcher_view_v1" or str(root) != manifest["script_dir"]:
        raise ValueError("launcher identity differs")
    if sha(Path(__file__).read_bytes()) != manifest["helper_sha256"]:
        raise ValueError("launcher adapter changed after preparation")
    if (manifest.get("source_sha256") != SOURCE_SHA
            or set(manifest["rendered_sha256"]) != set(SOURCE_SHA)
            or manifest["object_store_bytes_by_raylet"] != {"head": 4 * GIB, "worker_0": 4 * GIB}
            or manifest["object_store_bytes_total"] != 8 * GIB):
        raise ValueError("native source or aggregate object-store contract differs")
    for name, expected in manifest["rendered_sha256"].items():
        if Path(name).name != name or sha((root / name).read_bytes()) != expected:
            raise ValueError("generated launch script changed")
    private = Path(manifest["private_root"])
    if private.is_symlink() or private.stat().st_uid != os.getuid() or private.stat().st_mode & 0o077:
        raise ValueError("private runtime directory ownership differs")
    module = load_guard(Path(manifest["main_repo"]))
    admission = module.verify_current_service()
    if os.environ.get("SLLM_RAY_OBJECT_STORE_MEMORY_BYTES"):
        raise ValueError("legacy per-node object-store override conflicts with aggregate TC contract")
    # Binding is only a pre-launch collision check, not proof that a later HTTP
    # listener belongs to us. Actual Ray workers/API/store still need ownership
    # and native-source readback in the qualification experiment.
    endpoints = [(os.environ["SLLM_RAY_HEAD_HOST"], int(os.environ["SLLM_RAY_PORT"])),
                 (os.environ.get("SLLM_HOST", "127.0.0.1"), int(os.environ["SLLM_PORT"])),
                 ("0.0.0.0", 8073)]
    if len({port for _, port in endpoints}) != len(endpoints):
        raise ValueError("service/control/store ports collide")
    for host, port in endpoints:
        if not 1024 <= port <= 65535:
            raise ValueError("invalid unprivileged service port")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    return admission


def validate_ray_nodes(nodes: list[dict], gpu_count: int) -> dict:
    alive = [node for node in nodes if node["Alive"]]
    if len(alive) != 2 or len({n['NodeID'] for n in alive}) != 2:
        raise ValueError("expected exactly two live native raylets")
    result = {}
    for node in alive:
        resources = node["Resources"]
        is_head, is_worker = resources.get("control_node", 0) == 1, resources.get("worker_node", 0) == 1
        if is_head == is_worker:
            raise ValueError("native raylet role is ambiguous")
        role = "head" if is_head else "worker_0"
        if role in result or resources.get("GPU", 0) != (0 if is_head else gpu_count):
            raise ValueError("native role or GPU inventory differs")
        if resources.get("object_store_memory") != 4 * GIB:
            raise ValueError("live raylet object-store capacity differs")
        result[role] = node
    if set(result) != {"head", "worker_0"}:
        raise ValueError("native head/worker roles incomplete")
    return result


def qualify_ray(args) -> dict:
    """Observe the actual two-raylet launch inside the existing external gate."""
    guard = load_guard(args.main_repo)
    admission = guard.verify_current_service()
    if Path(sys.executable).resolve() != (args.environment / 'bin/python').resolve():
        raise ValueError("use the explicitly selected native Ray interpreter")
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    if ray.__version__ != '2.54.0' or ray.__commit__ != '48bd1f8fa43d0e8222b0f57357b99b48c7437ed3':
        raise ValueError("native Ray identity differs from the inspected environment")
    result = dict(schema="ieee_tc_serverless_two_raylet_witness_v1", passed=False,
                  inference_requests=0, model_loaded=False, actual_model_workers_qualified=False,
                  performance_run_authorized=False, service=admission,
                  ray_version=ray.__version__, ray_commit=ray.__commit__, ray_file=ray.__file__,
                  ray_init_sha256=sha(Path(ray.__file__).read_bytes()))
    manifest = prepare(args.output, args.private_root, args.main_repo, args.gpu_ids, ray_only=True)
    result['launcher_manifest_sha256'] = sha((args.output / 'launch_manifest.json').read_bytes())
    env = dict(os.environ)
    env.pop('TMUX', None)
    env.pop('TMUX_PANE', None)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = ''
    env.update(SLLM_HEAD_ENV_PREFIX=str(args.environment), SLLM_WORKER_ENV_PREFIX=str(args.environment),
               SLLM_STORE_ENV_PREFIX=str(args.environment), SLLM_REPO_ROOT=str(args.native_source),
               SLLM_RAY_HEAD_HOST=args.host, SLLM_RAY_PORT=str(args.ray_port), SLLM_PORT=str(args.api_port),
               SLLM_STORE_PATH=str(args.checkpoint_root), SLLM_DIRECT_PATH_MODE='0',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', RAY_USAGE_STATS_ENABLED='0',
               RAY_TMPDIR=str(args.private_root / 'ray_tmp'),
               NO_PROXY=f"{args.host},127.0.0.1,localhost", no_proxy=f"{args.host},127.0.0.1,localhost")
    env.update(SLLM_HEAD_SESSION='sllm_head_formal', SLLM_WORKER_SESSION_PREFIX='sllm_worker_formal',
               SLLM_RAY_HEAD_ADDRESS=f'{args.host}:{args.ray_port}', RAY_ADDRESS=f'{args.host}:{args.ray_port}',
               SLLM_HEAD_RESOURCES='{"control_node": 1}',
               SLLM_WORKER_RESOURCES='{"worker_node": 1, "worker_id_0": 1}')
    # Native shell prefixes must not inherit alternate binary selections.
    for key in ('SLLM_HEAD_RAY_BIN', 'SLLM_WORKER_RAY_BIN', 'SLLM_HEAD_PYTHON_BIN'):
        env.pop(key, None)
    os.environ.update({key: env[key] for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                       'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy', 'RAY_TMPDIR')})
    tmux = ['tmux', '-f', '/dev/null', '-S', str(args.private_root / 'tmux.sock')]
    actors, failure = [], None
    try:
        with (args.output / 'startup.log').open('x') as log:
            startup = subprocess.run(['bash', str(args.output / 'start_serverlessllm_stack.sh')],
                                     env=env, stdout=log, stderr=subprocess.STDOUT, timeout=210)
        result['startup_returncode'] = startup.returncode
        if startup.returncode:
            raise RuntimeError('native Ray startup failed; preserve startup and private Ray logs')
        ray.init(address=f'{args.host}:{args.ray_port}', namespace='tc-launch-qualification',
                 log_to_driver=False)
        nodes = ray.nodes()
        result['nodes'] = nodes
        by_role = validate_ray_nodes(nodes, len(args.gpu_ids))

        @ray.remote(max_restarts=0)
        class Witness:
            def report(self):
                import os
                import pathlib
                import subprocess
                import sys
                import json
                import ray
                child = subprocess.check_output([sys.executable, '-c',
                    'import os,json,pathlib; print(json.dumps(dict(pid=os.getpid(),'
                    'cgroup=pathlib.Path("/proc/self/cgroup").read_text(),'
                    'affinity=sorted(os.sched_getaffinity(0)))))'], text=True, timeout=10)
                return dict(pid=os.getpid(), cgroup=pathlib.Path('/proc/self/cgroup').read_text(),
                            affinity=sorted(os.sched_getaffinity(0)), child=json.loads(child),
                            node_id=ray.get_runtime_context().get_node_id(),
                            gpu_ids=ray.get_gpu_ids(), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))

        for role, node in by_role.items():
            for _ in range(1 if role == 'head' else len(args.gpu_ids)):
                actors.append(Witness.options(num_cpus=1, num_gpus=0 if role == 'head' else 1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'], soft=False)).remote())
        reports = ray.get([actor.report.remote() for actor in actors], timeout=60)
        result['worker_reports'] = reports
        expected_group = str(Path(admission['service_identity']['path']).relative_to('/sys/fs/cgroup'))
        if len({row['pid'] for row in reports}) != len(actors):
            raise RuntimeError('expected distinct native worker processes')
        for row in reports:
            for observed in (row, row['child']):
                if observed['cgroup'].strip() != '0::/' + expected_group or observed['affinity'] != sorted(guard.SERVICE_CPUS):
                    raise RuntimeError('actual Ray worker or its child escaped the common service envelope')
        worker_reports = [r for r in reports if r['node_id'] == by_role['worker_0']['NodeID']]
        if (len(worker_reports) != len(args.gpu_ids)
                or sorted(int(g) for r in worker_reports for g in r['gpu_ids']) != list(range(len(args.gpu_ids)))):
            raise RuntimeError('native witness GPU scheduling coverage differs')
        group = Path(admission['service_identity']['path'])
        result['owned_processes'] = guard.owned_pids(group)
        result['resource_snapshot'] = guard.cgroup_snapshot(group)
        raylets = []
        for proc in result['owned_processes']:
            try:
                command = Path(f"/proc/{proc['pid']}/cmdline").read_bytes().split(b'\0')
            except FileNotFoundError:
                continue
            if command and Path(command[0].decode()).name == 'raylet':
                args_readback = [part.decode() for part in command if part]
                if f'--object_store_memory={4 * GIB}' not in args_readback:
                    raise RuntimeError('actual raylet process capacity differs')
                raylets.append(dict(identity=proc, command=args_readback))
        result['raylet_processes'] = raylets
        if len(raylets) != 2:
            raise RuntimeError('two live owned raylet processes were not observed')
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        failure = exc
    finally:
        cleanup_errors = []
        if ray.is_initialized():
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            try:
                ray.shutdown()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        # Only this fresh private tmux server is addressed. Its descendants
        # remain covered by the original gated launcher's final whole-tree check.
        for session in ('sllm_head_formal', 'sllm_worker_formal_0'):
            capture = subprocess.run(tmux + ['capture-pane', '-pJt', session, '-S', '-'],
                                     text=True, capture_output=True, timeout=5)
            with (args.output / f'{session}.log').open('x') as handle:
                handle.write(capture.stdout + capture.stderr)
        killed = subprocess.run(tmux + ['kill-server'], capture_output=True, text=True, timeout=5)
        result['private_tmux_stop_returncode'] = killed.returncode
        result['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            result['passed'] = False
            failure = failure or RuntimeError('native actor cleanup failed')
        result['external_cleanup_required'] = True
        with (args.output / 'ray_witness.json').open('x') as handle:
            json.dump(result, handle, indent=2)
    if failure:
        raise RuntimeError('Serverless Ray qualification failed; see preserved witness') from failure
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--private-root", type=Path, required=True)
    prep.add_argument("--main-repo", type=Path, required=True)
    prep.add_argument("--gpu-ids", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    witness = sub.add_parser('qualify-ray')
    for name in ('output', 'private-root', 'main-repo', 'environment', 'native-source', 'checkpoint-root'):
        witness.add_argument('--' + name, type=Path, required=True)
    witness.add_argument('--gpu-ids', required=True)
    witness.add_argument('--host', required=True)
    witness.add_argument('--ray-port', required=True, type=int)
    witness.add_argument('--api-port', required=True, type=int)
    args = parser.parse_args()
    if args.action == 'qualify-ray':
        args.gpu_ids = tuple(int(i) for i in args.gpu_ids.split(','))
        print(json.dumps(qualify_ray(args), indent=2))
        return
    result = (prepare(args.output, args.private_root, args.main_repo,
                      tuple(int(i) for i in args.gpu_ids.split(','))) if args.action == "prepare"
              else verify(args.manifest))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
