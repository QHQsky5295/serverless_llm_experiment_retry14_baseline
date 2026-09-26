"""Native script adaptation checks; these do not qualify actual Ray workers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/prepare_ieee_tc_serverless_stack.py"
spec = importlib.util.spec_from_file_location("tc_launch", HELPER)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
MAIN = Path("/home/qhq/serverless_llm_experiment_retry14_baseline")


class NativeLaunchTests(unittest.TestCase):
    def sources(self):
        return {name: (ROOT / "scripts" / name).read_text() for name in launch.SOURCE_SHA}

    def render(self, **changes):
        options = dict(script_dir=Path("/tmp/tc-launch-test/scripts"),
                       private_root=Path("/tmp/tc-launch-test/runtime"),
                       main_repo=MAIN, gpu_ids=(0, 1, 2, 3))
        options.update(changes)
        return launch.render(self.sources(), **options)

    def test_exact_original_sources_preserved_and_all_rendered_scripts_parse(self):
        rendered, _ = self.render()
        for name, source in rendered.items():
            with self.subTest(name=name):
                subprocess.run(["bash", "-n"], input=source, text=True, check=True)
                self.assertEqual(launch.sha((ROOT / "scripts" / name).read_bytes()), launch.SOURCE_SHA[name])

    def test_global_cleanup_source_mutation_and_log_overwrite_removed(self):
        rendered, _ = self.render()
        stack = rendered["start_serverlessllm_stack.sh"]
        for forbidden in ("stop_serverlessllm_stack.sh", "sync_serverlessllm_runtime_sources.sh",
                          "kill-session", 'rm -f "${SERVE_LOG_PATH}"', '"bash -lc'):
            self.assertNotIn(forbidden, stack)
        self.assertIn('command tmux -f /dev/null -S "${SLLM_TC_TMUX_SOCKET}"', stack)
        self.assertLess(stack.index(" verify --manifest "), stack.index("tmux new-session"))
        self.assertIn('unset TMUX TMUX_PANE', stack)

    def test_one_worker_and_aggregate_eight_gib_not_per_node(self):
        rendered, allocation = self.render()
        self.assertEqual(sum(allocation.values()), 8 * 1024**3)
        self.assertEqual(len(allocation), 2)
        self.assertIn("export SLLM_SINGLE_HOST_MULTI_GPU=1", rendered["start_serverlessllm_stack.sh"])
        for role in ("head", "worker"):
            source = rendered[f"run_serverlessllm_{role}.sh"]
            self.assertIn(f"RAY_OBJECT_STORE_MEMORY_BYTES={4 * 1024**3}", source)
            self.assertNotIn('${SLLM_RAY_OBJECT_STORE_MEMORY_BYTES:-}', source)

    def test_native_loader_store_gpu_set_and_private_spill(self):
        rendered, _ = self.render()
        self.assertIn("--enable-storage-aware", rendered["run_serverlessllm_serve.sh"])
        self.assertIn("CUDA_VISIBLE_DEVICES=${WORKER_GPUS} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh",
                      rendered["start_serverlessllm_stack.sh"])
        for role in ("head", "worker"):
            self.assertIn('--object-spilling-directory="${SLLM_TC_SPILL}/' + role + '"',
                          rendered[f"run_serverlessllm_{role}.sh"])
        self.assertIn('--temp-dir="${SLLM_TC_RAY_TEMP}"', rendered["run_serverlessllm_head.sh"])
        self.assertNotIn('--temp-dir=', rendered["run_serverlessllm_worker.sh"])

    def test_source_drift_refused(self):
        sources = self.sources()
        sources["run_serverlessllm_worker.sh"] += "\n# changed\n"
        with self.assertRaisesRegex(ValueError, "source drift"):
            launch.render(sources, script_dir=Path('/tmp/a'), private_root=Path('/tmp/b'),
                          main_repo=MAIN, gpu_ids=(0,))

    def test_invalid_gpu_and_shell_paths_refused(self):
        for ids in ((), (0, 0), (4,), (True,)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.render(gpu_ids=ids)
        for path in (Path("relative"), Path("/tmp/a b"), Path("/tmp/a;touch-oops")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.render(script_dir=path)

    def test_prepare_exclusive_and_does_not_start_services(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            manifest = launch.prepare(output, private, MAIN, (0, 1))
            self.assertFalse(manifest["performance_run_authorized"])
            self.assertFalse(manifest["actual_workers_verified"])
            self.assertFalse((private / "tmux.sock").exists())
            self.assertFalse((output / "serve.log").exists())
            with self.assertRaises(FileExistsError):
                launch.prepare(output, private, MAIN, (0, 1))

    def test_generated_stack_refuses_before_native_launch_outside_guard(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            launch.prepare(output, private, MAIN, (0,))
            env = dict(os.environ)
            env.pop("FAASLORA_TC_LAUNCH_RECEIPT", None)
            result = subprocess.run(["bash", str(output / "start_serverlessllm_stack.sh")],
                                    env=env, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing guarded TC launch receipt", result.stderr)
            self.assertFalse((private / "tmux.sock").exists())
            self.assertFalse((output / "serve.log").exists())

    def test_changed_leaf_is_rejected_before_gate_or_native_launch(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            launch.prepare(output, private, MAIN, (0,))
            leaf = output / "run_serverlessllm_head.sh"
            leaf.write_text(leaf.read_text() + "\n# changed after preparation\n")
            with self.assertRaisesRegex(ValueError, "script changed"):
                launch.verify(output / "launch_manifest.json")

    def test_changed_aggregate_contract_is_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            manifest = launch.prepare(output, private, MAIN, (0,))
            manifest["object_store_bytes_total"] = 16 * 1024**3
            (output / "launch_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "aggregate object-store contract differs"):
                launch.verify(output / "launch_manifest.json")

    def test_actual_leaf_arguments_have_distinct_allocations_and_no_global_stop(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            launch.prepare(output, private, MAIN, (0, 1))
            fake = root / "capture"
            fake.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o700)
            env = dict(os.environ, SLLM_HEAD_RAY_BIN=str(fake), SLLM_WORKER_RAY_BIN=str(fake),
                       SLLM_TC_RAY_TEMP=str(private / "ray_tmp/ray"), SLLM_TC_SPILL=str(output / "spill"),
                       SLLM_STORE_PATH=str(root / "store"), SLLM_WORKER_NUM_GPUS="2")
            total = 0
            for role in ("head", "worker"):
                result = subprocess.run(["bash", str(output / f"run_serverlessllm_{role}.sh")],
                                        env=env, text=True, capture_output=True, timeout=10, check=True)
                args = json.loads(result.stdout)
                size = next(arg for arg in args if arg.startswith("--object-store-memory="))
                total += int(size.split('=')[1])
                self.assertIn("--block", args)
                self.assertIn(f"--object-spilling-directory={output}/spill/{role}", args)
            self.assertEqual(total, 8 * 1024**3)

    def test_ray_only_executes_same_prefix_without_store_api_or_model(self):
        full, _ = self.render()
        probe, _ = self.render(ray_only=True)
        marker = 'wait_for_workers "${EXPECTED_WORKERS}"\n'
        self.assertEqual(full['start_serverlessllm_stack.sh'].split(marker)[0],
                         probe['start_serverlessllm_stack.sh'].split(marker)[0])
        suffix = probe['start_serverlessllm_stack.sh'].split(marker)[1]
        self.assertNotIn('new-session', suffix)
        self.assertIn('ray-only infrastructure', suffix)

    def nodes(self):
        return [dict(Alive=True, NodeID='head', Resources=dict(control_node=1, object_store_memory=4 * launch.GIB)),
                dict(Alive=True, NodeID='worker', Resources=dict(worker_node=1, GPU=4, object_store_memory=4 * launch.GIB))]

    def test_live_node_capacity_and_roles_match(self):
        self.assertEqual(set(launch.validate_ray_nodes(self.nodes(), 4)), {'head', 'worker_0'})

    def test_live_node_resource_or_membership_drift_is_not_qualified(self):
        for change in ('capacity', 'gpu', 'role', 'third', 'dead'):
            with self.subTest(change=change):
                nodes = self.nodes()
                if change == 'capacity':
                    nodes[1]['Resources']['object_store_memory'] *= 2
                elif change == 'gpu':
                    nodes[1]['Resources']['GPU'] = 1
                elif change == 'role':
                    nodes[1]['Resources']['control_node'] = 1
                elif change == 'third':
                    nodes.append(dict(Alive=True, NodeID='foreign', Resources={}))
                else:
                    nodes[1]['Alive'] = False
                with self.assertRaises(ValueError):
                    launch.validate_ray_nodes(nodes, 4)


if __name__ == "__main__":
    unittest.main()
