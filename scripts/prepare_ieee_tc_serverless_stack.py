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
           main_repo: Path, gpu_ids: tuple[int, ...]) -> tuple[dict[str, str], dict]:
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


def prepare(output: Path, private_root: Path, main_repo: Path, gpu_ids: tuple[int, ...]) -> dict:
    for path in (output, private_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refuse to reuse or overwrite {path}")
    sources = {name: (ROOT / "scripts" / name).read_text() for name in SOURCE_SHA}
    rendered, allocation = render(sources, script_dir=output, private_root=private_root,
                                 main_repo=main_repo, gpu_ids=gpu_ids)
    manifest = {
        "schema": "ieee_tc_serverless_native_launcher_view_v1", "display_name": "Serverless",
        "main_repo": str(main_repo), "private_root": str(private_root),
        "script_dir": str(output), "source_sha256": SOURCE_SHA,
        "rendered_sha256": {name: sha(data.encode()) for name, data in rendered.items()},
        "helper_sha256": sha(Path(__file__).read_bytes()),
        "object_store_bytes_by_raylet": allocation,
        "object_store_bytes_total": sum(allocation.values()), "worker_gpu_ids": list(gpu_ids),
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
    checker = Path(manifest["main_repo"]) / "scripts" / "ieee_tc_preflight.py"
    spec = importlib.util.spec_from_file_location("tc_native_launch_guard", checker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    args = parser.parse_args()
    result = (prepare(args.output, args.private_root, args.main_repo,
                      tuple(int(i) for i in args.gpu_ids.split(','))) if args.action == "prepare"
              else verify(args.manifest))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
